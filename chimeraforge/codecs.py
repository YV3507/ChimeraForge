"""Reed-Solomon 前向纠错（可选，依赖 reedsolo）。

用于抵抗载体视频经平台二次转码/重压缩导致的部分比特损坏。
"""

from __future__ import annotations

try:  # pragma: no cover - 可选依赖
    import reedsolo  # type: ignore

    _HAS_REEDSOLO = True
except ImportError:  # pragma: no cover
    _HAS_REEDSOLO = False


def available() -> bool:
    return _HAS_REEDSOLO


def encode(data: bytes, nsym: int = 16) -> bytes:
    if not _HAS_REEDSOLO:
        raise RuntimeError("需要安装 reedsolo：pip install reedsolo")
    return bytes(reedsolo.RSCodec(nsym).encode(data))


def decode(data: bytes, nsym: int = 16) -> bytes:
    if not _HAS_REEDSOLO:
        raise RuntimeError("需要安装 reedsolo：pip install reedsolo")
    return bytes(reedsolo.RSCodec(nsym).decode(data)[0])
