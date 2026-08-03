"""compress 阶段：INR 权重 → 认证载荷文件（量化 + 熵编码 + 载荷头）。"""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig

from chimeraforge import codecs, keys, payload as payload_mod
from chimeraforge.inr import load_model, pack_model


def run(model_path: str, out_payload: str, cfg: DictConfig, master_key: bytes) -> dict:
    model, model_cfg, orig_size = load_model(model_path)
    inr_cfg_json = json.dumps(model_cfg, sort_keys=True).encode("utf-8")

    packed = pack_model(model, model_cfg, bits=int(cfg.compress.bits))

    ecc = bool(cfg.ecc.enabled)
    if ecc:
        packed = codecs.encode(packed, nsym=int(cfg.ecc.nsym))

    blob = payload_mod.wrap(
        packed,
        mac_key=keys.mac_key(master_key),
        inr_hash=payload_mod.hash_inr_cfg(inr_cfg_json),
        num_blocks=model_cfg["num_blocks"],
        block_size=model_cfg["block_size"],
        orig_size=orig_size,
        ecc=ecc,
    )
    Path(out_payload).write_bytes(blob)

    return {
        "stage": "compress",
        "payload": out_payload,
        "size": len(blob),
        "ecc": ecc,
    }
