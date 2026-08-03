"""codecs.py 单元测试：Reed-Solomon 纠错（含依赖缺失路径）。"""

import pytest

from chimeraforge import codecs


def test_available():
    assert codecs.available() is True


def test_roundtrip():
    data = b"hello reedsolo " * 8
    enc = codecs.encode(data, nsym=16)
    assert len(enc) > len(data)
    assert codecs.decode(enc, nsym=16) == data


def test_corrects_errors():
    data = b"error correction test " * 6
    enc = bytearray(codecs.encode(data, nsym=16))
    for i in range(4):  # nsym=16 可纠正 ≤8 字节
        enc[10 + i] ^= 0xFF
    assert codecs.decode(bytes(enc), nsym=16) == data


def test_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(codecs, "_HAS_REEDSOLO", False)
    assert codecs.available() is False
    with pytest.raises(RuntimeError, match="reedsolo"):
        codecs.encode(b"x")
    with pytest.raises(RuntimeError, match="reedsolo"):
        codecs.decode(b"x")
