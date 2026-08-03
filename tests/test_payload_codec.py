"""媒体层载荷编解码单元测试：bit 平面映射 + 冗余 + 密钥。"""

import pytest

from chimeraforge.stego import payload_codec


@pytest.mark.parametrize("size", [1, 7, 64, 257, 2048])
def test_roundtrip(tmp_path, size):
    payload = bytes((i * 37 + 11) % 256 for i in range(size))
    frames = payload_codec.encode(payload, b"key", height=32, width=32, redundancy=4)
    assert frames.shape[1:] == (3, 32, 32)
    assert payload_codec.decode(frames, b"key", redundancy=4) == payload


def test_wrong_key_gives_garbage(tmp_path):
    """媒体层头是明文；密钥错误 → 载荷体为垃圾（机密性由密钥流保证，
    完整性由载荷层 MAC 在 reconstruct 阶段校验）。"""
    payload = b"secret payload bytes"
    frames = payload_codec.encode(payload, b"good-key", height=16, width=16, redundancy=2)
    out = payload_codec.decode(frames, b"bad-key", redundancy=2)
    assert out != payload


def test_redundancy_corrects_mild_errors(tmp_path):
    """多数投票能纠正少量比特翻转（模拟 LF-VSN 有损重建）。"""
    payload = bytes(range(48))
    r = 5
    frames = payload_codec.encode(payload, b"k", height=16, width=16, redundancy=r)
    # 注入少量翻转：每 50 个冗余组翻转一个像素
    vals = frames.numpy().reshape(-1)
    for i in range(0, vals.size, 50):
        vals[i] = 1.0 - vals[i]
    corrupted = __import__("torch").from_numpy(vals.reshape(frames.shape))
    assert payload_codec.decode(corrupted, b"k", redundancy=r) == payload


def test_capacity_frames_count():
    """帧数 = ceil(总比特 / 每帧容量)。"""
    payload = b"x" * 100  # 800 比特 + 96 头 = 896 比特
    h, w, r = 16, 16, 4
    cap = h * w * 3 // r  # 192 比特/帧
    frames = payload_codec.encode(payload, b"k", height=h, width=w, redundancy=r)
    assert frames.shape[0] == (896 + cap - 1) // cap


def test_keystream_deterministic():
    assert payload_codec.keystream(b"k", 100) == payload_codec.keystream(b"k", 100)
    assert payload_codec.keystream(b"k", 100) != payload_codec.keystream(b"k2", 100)
