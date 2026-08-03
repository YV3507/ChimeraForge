"""extract 阶段：隐写视频 → 载荷（隐写后端提取）。"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from chimeraforge import keys
from chimeraforge.stego import get_backend


def run(stego_path: str, out_payload: str, cfg: DictConfig, master_key: bytes) -> dict:
    backend = get_backend(cfg)
    key = keys.stego_key(master_key, backend.name)
    n = backend.extract(Path(stego_path), Path(out_payload), key)
    return {"stage": "extract", "backend": backend.name, "size": n}
