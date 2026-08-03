"""embed 阶段：载荷 → 载体视频（隐写后端嵌入）。"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from chimeraforge import keys
from chimeraforge.stego import get_backend


def run(payload_path: str, cover_path: str | None, out_stego: str, cfg: DictConfig, master_key: bytes) -> dict:
    data = Path(payload_path).read_bytes()
    backend = get_backend(cfg)
    key = keys.stego_key(master_key, backend.name)
    info = backend.embed(data, Path(cover_path) if cover_path else None, Path(out_stego), key)
    return {"stage": "embed", "backend": backend.name, **info}
