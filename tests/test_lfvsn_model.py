"""LF-VSN 模型封装单元测试：DWT/IWT、隐藏/恢复形状、权重加载错误路径。

不依赖预训练权重（随机初始化网络只验证形状与可逆性基础组件）。
"""

import pytest
import torch

from chimeraforge.stego import lfvsn_model as lm


@pytest.fixture(scope="module")
def net():
    return _random_net()


def _random_net():
    networks = lm._import_networks()
    return networks.define_G_v2(lm._NET_OPT).eval()


@pytest.mark.parametrize("h,w", [(64, 64), (36, 48)])
def test_dwt_iwt_roundtrip(h, w):
    x = torch.rand(1, 9, h, w)
    y = lm.iwt(lm.dwt(x))
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-4)


def test_dwt_channel_split():
    x = torch.rand(1, 9, 32, 32)
    y = lm.dwt(x)
    assert y.shape == (1, 36, 16, 16)


def test_dwt_matches_official():
    """本地 dwt 与 LF-VSN 官方 dwt_init 逐位一致（官方 IWT 硬编码 .cuda()，仅比对 DWT）。"""
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1] / "third_party" / "LF-VSN" / "code"
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from models.modules.common import dwt_init  # noqa: E402

    x = torch.rand(1, 9, 32, 32)
    assert torch.equal(lm.dwt(x), dwt_init(x))


def test_hide_reveal_shapes(net):
    h = w = 64
    host = torch.rand(3, 3, h, w)
    secret = torch.rand(3, 3, h, w)
    stego = lm.hide_gop(net, host, secret)
    assert stego.shape == (3, h, w)
    assert stego.min() >= 0.0 and stego.max() <= 1.0

    rec = lm.reveal_gop(net, stego)
    assert rec.shape == (3, h, w)
    assert rec.min() >= 0.0 and rec.max() <= 1.0


def test_load_lfvsn_invalid_weight(tmp_path):
    bad = tmp_path / "bad.pth"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(Exception):
        lm.load_lfvsn(bad, device="cpu")
