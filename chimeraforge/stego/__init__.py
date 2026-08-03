"""隐写后端工厂。"""

from __future__ import annotations

from omegaconf import DictConfig

from .base import StegoBackend
from .lfvsn import LFVSNBackend
from .stub import StubBackend


def get_backend(cfg: DictConfig) -> StegoBackend:
    name = cfg.stego.backend
    if name == "stub":
        return StubBackend()
    if name == "lfvsn":
        lf = cfg.stego.lfvsn
        ecc_nsym = int(cfg.ecc.nsym) if cfg.ecc.enabled else 0
        return LFVSNBackend(
            weight_dir=lf.weight_dir,
            mode=int(lf.mode),
            frame_size=int(getattr(lf, "frame_size", 256)),
            redundancy=int(getattr(lf, "redundancy", 4)),
            device=str(getattr(lf, "device", "auto")),
            ecc_nsym=ecc_nsym,
        )
    raise ValueError(f"未知的隐写后端: {name}（可用: stub | lfvsn）")
