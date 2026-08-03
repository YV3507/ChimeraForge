"""compile 阶段：内容字节 → INR 权重文件（.pt）。"""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig

from chimeraforge import keys
from chimeraforge.inr import load_model, save_model, train_inr


def run(content_path: str, out_model: str, cfg: DictConfig, master_key: bytes, progress: bool = False) -> dict:
    content = Path(content_path).read_bytes()
    film_seed = keys.film_seed(master_key)
    qat_epochs = int(getattr(cfg.inr, "qat_epochs", 0))
    model, stats = train_inr(
        content, cfg.inr, film_seed, device=cfg.inr.device, progress=progress, qat_epochs=qat_epochs
    )
    save_model(model, len(content), out_model)

    _, model_cfg, orig_size = load_model(out_model)
    return {
        "stage": "compile",
        "model": out_model,
        "orig_size": orig_size,
        "inr_cfg_json": json.dumps(model_cfg, sort_keys=True),
        "stats": stats.__dict__,
    }
