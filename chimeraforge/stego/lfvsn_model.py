"""LF-VSN 预训练模型封装：构建 VSN 网络、加载权重、隐藏/恢复推理。

兼容性处理：
- `basicsr` 缺失时注入占位模块（源码 import 了但推理路径不使用的符号）；
- DWT/IWT 使用本地实现（官方 common.py 的 IWT 硬编码 `.cuda()`，CPU 上会崩溃）；
- 直接构建 `VSN`（跳过官方 Model_VSN 的 DataParallel/训练逻辑），逐 GOP 推理以控制显存。

用法（由 lfvsn.py 后端调用）：
    net = load_lfvsn(weight_path, device)          # 构建 + 加载权重
    stego_frame = hide_gop(net, host_g, secret_g) # 一帧正向隐藏
    secret_g = reveal_gop(net, stego_frame)        # 一帧反向恢复
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

# 官方 1 视频模式网络配置（对齐 options/train/train_LF-VSN_1video.yml 的 network_G）
_NET_OPT = {
    "model": "MIMO-VRN-h",
    "gop": 3,
    "num_video": 1,
    "datasets": {},  # define_G_v2 会读取该键（虽然未使用）
    "network_G": {
        "which_model_G": {"subnet_type": "DBNet"},
        "in_nc": 12,
        "out_nc": 12,
        "block_num": [8, 8],
        "scale": 2,
        "init": "xavier_group",
        "block_num_rbm": 8,
    },
}

_REPO_DIR = Path(__file__).resolve().parents[2] / "third_party" / "LF-VSN" / "code"


def _install_basicsr_shim() -> None:
    """LF-VSN 源码从 basicsr 导入 flow_warp/ResidualBlockNoBN，但推理路径不使用。"""
    try:
        __import__("basicsr")  # 已安装则无需 shim
        return
    except ImportError:
        pass
    for name in ("basicsr", "basicsr.archs", "basicsr.archs.arch_util"):
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        mod.__path__ = []  # 声明为包
        sys.modules[name] = mod
    sys.modules["basicsr.archs.arch_util"].flow_warp = None  # type: ignore[attr-defined]
    sys.modules["basicsr.archs.arch_util"].ResidualBlockNoBN = None  # type: ignore[attr-defined]


def _import_networks():
    _install_basicsr_shim()
    code_dir = str(_REPO_DIR)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    import models.networks as networks  # type: ignore

    return networks


def load_lfvsn(weight_path: str | Path, device: str = "auto") -> torch.nn.Module:
    """构建 VSN 网络并加载预训练权重。返回 .eval() 的 fp32 模型。"""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    networks = _import_networks()
    net = networks.define_G_v2(_NET_OPT)
    sd = torch.load(weight_path, map_location="cpu", weights_only=True)
    if not any(k.startswith("irn.") or k.startswith("pm.") for k in sd):
        # 个别仓库把权重包在 G/netG 键下
        for wrap in ("G", "netG", "state_dict"):
            if wrap in sd and isinstance(sd[wrap], dict):
                sd = sd[wrap]
                break
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=True)
    if missing or unexpected:
        raise ValueError(f"权重与网络结构不匹配: missing={missing}, unexpected={unexpected}")
    return net.to(device).eval()


def dwt(x: torch.Tensor) -> torch.Tensor:
    """2D Haar DWT（与官方 common.py 逐位一致，device 无关）。"""
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    ll = x1 + x2 + x3 + x4
    hl = -x1 - x2 + x3 + x4
    lh = -x1 + x2 - x3 + x4
    hh = x1 - x2 - x3 + x4
    return torch.cat((ll, hl, lh, hh), 1)


def iwt(x: torch.Tensor) -> torch.Tensor:
    """2D Haar IWT（官方 IWT 的 device 无关版，去除硬编码 .cuda()）。"""
    b, c, h, w = x.shape
    oc = c // 4
    x1 = x[:, 0:oc, :, :] / 2
    x2 = x[:, oc : 2 * oc, :, :] / 2
    x3 = x[:, 2 * oc : 3 * oc, :, :] / 2
    x4 = x[:, 3 * oc : 4 * oc, :, :] / 2
    out = torch.zeros(b, oc, h * 2, w * 2, device=x.device, dtype=x.dtype)
    out[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    out[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    out[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    out[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return out


@torch.no_grad()
def hide_gop(
    net: torch.nn.Module,
    host: torch.Tensor,
    secret: torch.Tensor,
) -> torch.Tensor:
    """正向隐藏一个 GOP（3 帧封面 + 3 帧秘密）→ 中心隐写帧。

    host/secret: (3, 3, h, w)，值域 [0,1]，同一分辨率（偶数边长）。
    返回中心隐写帧 (3, h, w)，值域 [0,1]。
    """
    b, gop, c, h, w = 1, host.shape[0], 3, host.shape[2], host.shape[3]
    hx = dwt(host.unsqueeze(0).reshape(b, c * gop, h, w))
    sx = dwt(secret.unsqueeze(0).reshape(b, c * gop, h, w))
    out_y, _ = net(x=hx, x_h=[sx])
    out = iwt(out_y).reshape(b, gop, c, h, w)
    return torch.clamp(out[0, gop // 2], 0.0, 1.0)


@torch.no_grad()
def reveal_gop(net: torch.nn.Module, stego_frame: torch.Tensor) -> torch.Tensor:
    """反向恢复一个 GOP：中心隐写帧 → 中心秘密帧。

    stego_frame: (3, h, w)，值域 [0,1]。返回 (3, h, w) 秘密帧，值域 [0,1]。
    """
    gop, c, h, w = 3, 3, stego_frame.shape[1], stego_frame.shape[2]
    center = torch.clamp(stego_frame, 0.0, 1.0)
    y = (center * 255.0).round() / 255.0  # 与官方 Quantization 一致
    y = y.repeat(gop, 1, 1).unsqueeze(0)  # (1, 9, h, w)
    yx = dwt(y)
    out_x, out_x_h, _ = net(x=yx, rev=True)
    rec = iwt(out_x_h[0]).reshape(1, 1, gop, c, h, w)
    return torch.clamp(rec[0, 0, gop // 2], 0.0, 1.0)
