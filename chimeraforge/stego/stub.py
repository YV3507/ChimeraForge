"""占位后端（stub）：密钥流 XOR + 伪载体头。

仅用于打通工程链路（M0 冒烟测试），不是真实隐写。
结构：载体头(≤32B) | 密文长度(8B) | XOR(载荷, 密钥流)
"""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path

from .base import StegoBackend


def _keystream(key: bytes, n: int) -> bytes:
    out = bytearray()
    ctr = 0
    while len(out) < n:
        out += hmac.new(key, ctr.to_bytes(8, "big"), hashlib.sha256).digest()
        ctr += 1
    return bytes(out[:n])


class StubBackend(StegoBackend):
    name = "stub"

    def embed(self, data: bytes, cover_path: Path | None, out_path: Path, key: bytes) -> dict:
        ks = _keystream(key, len(data))
        body = bytes(a ^ b for a, b in zip(data, ks))
        head = b"\x00" * 32
        if cover_path is not None and Path(cover_path).exists():
            head = Path(cover_path).read_bytes()[:32]
        blob = head + len(body).to_bytes(8, "big") + body
        Path(out_path).write_bytes(blob)
        return {"backend": self.name, "size": len(blob), "cover_header": len(head)}

    def extract(self, stego_path: Path, out_path: Path, key: bytes) -> int:
        blob = Path(stego_path).read_bytes()
        body = blob[40:]
        ks = _keystream(key, len(body))
        data = bytes(a ^ b for a, b in zip(body, ks))
        Path(out_path).write_bytes(data)
        return len(data)
