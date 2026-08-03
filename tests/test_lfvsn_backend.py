"""LF-VSN 后端媒体层测试：GOP 边界、合成封面确定性、帧 I/O（PNG 无损 / MP4 有损）。

不依赖预训练权重；cv2 缺失时跳过 I/O 用例（CI 用 [dev] extra 时无 opencv）。
"""

import numpy as np
import pytest
import torch

from chimeraforge.stego.lfvsn import _gop3, _read_frames, _synthetic_cover, _write_frames

try:
    import cv2  # noqa: F401

    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

needs_cv2 = pytest.mark.skipif(not HAVE_CV2, reason="需要 opencv-python")


def _t5():
    return torch.arange(5).reshape(5, 1, 1, 1).float()


# ---------- GOP 边界帧 ----------
def test_gop3_middle():
    assert torch.equal(_gop3(_t5(), 2), _t5()[1:4])


def test_gop3_start_boundary():
    """首帧：复制第 0 帧补足前导。"""
    got = _gop3(_t5(), 0)
    expect = torch.cat([_t5()[:1].repeat(2, 1, 1, 1), _t5()[:2]], 0)
    assert torch.equal(got, expect)


def test_gop3_end_boundary():
    """尾帧：复制最后一帧补足后缀。"""
    got = _gop3(_t5(), 4)
    expect = torch.cat([_t5()[-2:], _t5()[-1:].repeat(2, 1, 1, 1)], 0)
    assert torch.equal(got, expect)


def test_gop3_single_frame():
    """仅 1 帧时（退化场景）：全为该帧。"""
    t = torch.rand(1, 3, 4, 4)
    assert torch.equal(_gop3(t, 0), t.repeat(3, 1, 1, 1))


# ---------- 合成封面 ----------
def test_synthetic_cover_deterministic():
    a = _synthetic_cover(4, 64, 48, seed=7)
    b = _synthetic_cover(4, 64, 48, seed=7)
    assert len(a) == len(b) == 4
    for fa, fb in zip(a, b):
        assert np.array_equal(fa, fb)  # 同种子 → 逐位一致（测试可复现）
    assert a[0].shape == (64, 48, 3)
    assert a[0].dtype == np.uint8


def test_synthetic_cover_range():
    frames = _synthetic_cover(2, 32, 32)
    for f in frames:
        assert f.min() >= 0 and f.max() <= 255


# ---------- 帧 I/O ----------
def _media_frames():
    return _synthetic_cover(3, 32, 32, seed=3)


@needs_cv2
def test_frames_write_read_png_roundtrip(tmp_path):
    """PNG 目录无损：写入后读回逐位一致。"""
    dst = tmp_path / "frames"
    fmt, _ = _write_frames(_media_frames(), dst)
    assert fmt == "png"
    got = _read_frames(dst)
    assert len(got) == 3
    for a, b in zip(_media_frames(), got):
        assert np.array_equal(a, b)


@needs_cv2
def test_frames_write_read_mp4_roundtrip(tmp_path):
    """MP4 有损：帧数与尺寸一致，像素内容接近（编码失真允许）。"""
    dst = tmp_path / "stego.mp4"
    fmt, size = _write_frames(_media_frames(), dst)
    assert fmt == "mp4"
    assert size > 0
    got = _read_frames(dst)
    assert len(got) == 3
    assert got[0].shape == (32, 32, 3)
    errs = [np.abs(a.astype(int) - b.astype(int)).max() for a, b in zip(_media_frames(), got)]
    assert max(errs) <= 40  # mp4v 量化失真在可接受范围


@needs_cv2
def test_read_frames_unsupported_format(tmp_path):
    bad = tmp_path / "cover.bin"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError, match="不支持的载体格式"):
        _read_frames(bad)
