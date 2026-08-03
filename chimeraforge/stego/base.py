"""隐写后端接口。

embed 将载荷字节写入载体输出文件；extract 反向恢复。
后端可插拔：stub（占位）/ lfvsn（M1 接入，官方预训练权重）。
"""

from __future__ import annotations

from pathlib import Path


class StegoBackend:
    name = "base"

    def embed(self, data: bytes, cover_path: Path | None, out_path: Path, key: bytes) -> dict:
        raise NotImplementedError

    def extract(self, stego_path: Path, out_path: Path, key: bytes) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover
        return f"<StegoBackend {self.name}>"
