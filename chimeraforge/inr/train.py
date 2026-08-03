"""INR 编译（compile 阶段）：将内容字节过拟合进一个小型网络。

内容按 block_size 分块，训练网络学习 索引 → 字节块 的映射。
主训练后可选 int8 QAT 微调（`qat_epochs > 0`）：余弦退火把输出收敛到
字节舍入阈值内，结束再把权重 snap 到 int8 格点，使 compress 阶段
pack_model(bits=8) 与训练前向位级一致，重建可字节级精确。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .model import OneDimINR
from .qat import convert_to_qat

try:  # pragma: no cover - 依赖 omegaconf 运行时配置
    from omegaconf import DictConfig
except ImportError:  # pragma: no cover
    DictConfig = None  # type: ignore


@dataclass
class TrainStats:
    epochs: int
    final_loss: float
    byte_accuracy: float  # 重建字节与原文一致的占比
    seconds: float
    qat_epochs: int = 0


def _targets(content: bytes, block_size: int) -> tuple[torch.Tensor, int]:
    if not content:
        raise ValueError("内容为空，无法编译")
    n = len(content)
    nblocks = (n + block_size - 1) // block_size
    padded = content + b"\x00" * (nblocks * block_size - n)
    arr = np.frombuffer(padded, dtype=np.uint8).astype(np.float32) / 255.0
    return torch.from_numpy(arr.reshape(nblocks, block_size)), n


def _train_loop(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    tgt: torch.Tensor,
    epochs: int,
    progress: bool,
    progress_every: int,
    sched=None,
) -> float:
    loss_fn = nn.MSELoss()
    idx = torch.arange(tgt.shape[0], device=tgt.device)
    final_loss = float("inf")
    for epoch in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        pred = model(idx)
        loss = loss_fn(pred, tgt)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        final_loss = float(loss.item())
        if progress and epoch % progress_every == 0:
            print(f"  epoch {epoch:5d}  loss {final_loss:.3e}")
        if final_loss < 1e-8:
            break
    return final_loss


def _snap_to_grid(model: nn.Module, bits: int = 8) -> None:
    """QAT 结束后把权重拉回量化格点（bias 除外）。

    STE 训练只把权重"推向"格点附近而非精确落点；显式 snap 后
    权重恰好等于 pack→unpack 复原值，重建可字节级精确。
    """
    from .qat import quantize_weight_ste

    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(".bias"):
                continue
            p.data.copy_(quantize_weight_ste(p.detach(), bits))


def train_inr(
    content: bytes,
    cfg,
    film_seed: int,
    device: str = "auto",
    progress: bool = False,
    qat_epochs: int = 0,
) -> tuple[OneDimINR, TrainStats]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(getattr(cfg, "seed", 0)))  # 固定 RNG，保证可复现

    tgt, orig_size = _targets(content, cfg.block_size)
    tgt = tgt.to(device)
    num_blocks = tgt.shape[0]

    model = OneDimINR(
        num_blocks=num_blocks,
        block_size=cfg.block_size,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        pos_freqs=cfg.pos_freqs,
        film_seed=film_seed,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    idx = torch.arange(num_blocks, device=device)

    t0 = time.time()
    epochs = int(cfg.epochs)
    final_loss = _train_loop(model, opt, tgt, epochs, progress, progress_every=100)

    qat_epochs = int(qat_epochs)
    if qat_epochs > 0:
        convert_to_qat(model, bits=8)  # 原位替换 nn.Linear → LinearQAT（state_dict 键不变）
        model = model.to(device)  # 替换后的层是 CPU 新建的，需移回训练设备
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)  # QAT 后需重建优化器
        # 余弦退火稳定收敛（恒定 lr 会在量化噪声下振荡，无法收敛到舍入阈值内）
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=qat_epochs)
        final_loss = _train_loop(model, opt, tgt, qat_epochs, progress, progress_every=100, sched=sched)
        _snap_to_grid(model, bits=8)  # 权重拉回格点 → pack/unpack 位级一致

    model.eval()
    with torch.no_grad():
        pred = model(idx).clamp(0.0, 1.0)
        rebuilt = (pred * 255.0).round().to(torch.uint8).view(-1).cpu().numpy().tobytes()[:orig_size]
    correct = sum(a == b for a, b in zip(rebuilt, content))
    stats = TrainStats(
        epochs=epochs + qat_epochs,
        final_loss=final_loss,
        byte_accuracy=correct / max(len(content), 1),
        seconds=time.time() - t0,
        qat_epochs=qat_epochs,
    )
    return model, stats


@torch.no_grad()
def infer_file(model: OneDimINR, orig_size: int, device: str = "auto") -> bytes:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    idx = torch.arange(model.num_blocks, device=device)
    pred = model(idx).clamp(0.0, 1.0)
    out = (pred * 255.0).round().to(torch.uint8).view(-1).cpu().numpy().tobytes()
    return out[:orig_size]
