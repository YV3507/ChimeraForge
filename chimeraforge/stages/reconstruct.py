"""reconstruct 阶段：载荷 → 原始内容字节（MAC 校验 + INR 推理）。"""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig

from chimeraforge import codecs, keys, payload as payload_mod
from chimeraforge.inr import infer_file, unpack_model


def run(payload_path: str, out_content: str, cfg: DictConfig, master_key: bytes) -> dict:
    blob = Path(payload_path).read_bytes()
    header, packed = payload_mod.unwrap(blob, mac_key=keys.mac_key(master_key))

    if header.ecc_enabled:
        packed = codecs.decode(packed, nsym=int(cfg.ecc.nsym))

    model, model_cfg = unpack_model(packed)

    got_hash = payload_mod.hash_inr_cfg(json.dumps(model_cfg, sort_keys=True).encode("utf-8"))
    if not header.inr_hash == got_hash:
        raise ValueError("INR 结构哈希不匹配：载荷与模型不一致")

    content = infer_file(model, header.orig_size, device=cfg.inr.device)
    Path(out_content).write_bytes(content)
    return {
        "stage": "reconstruct",
        "out": out_content,
        "orig_size": header.orig_size,
        "got_size": len(content),
    }
