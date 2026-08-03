"""模型压缩（compress 阶段）：int8 量化 + 熵编码。

M1 设计：权重做对称量化（int8，配合 QAT 后落在量化格点上 → 无损），
bias 与 FiLM buffer 保留 fp32 原样打包（QAT 不量化 bias，量化会引入误差）。
打包后经 zlib 熵编码。QAT 在 compile 阶段通过 `train_inr(qat_epochs=...)` 完成。
"""

from __future__ import annotations

import pickle
import zlib

import numpy as np
import torch

from .model import OneDimINR

FORMAT = "chimeraforge-inr-v2"
_DTYPES = {8: np.int8, 16: np.int16}


def _quantize_sym(w: torch.Tensor, bits: int) -> tuple[np.ndarray, float]:
    """对称量化；scale 用 fp32 张量计算（与 qat.quantize_weight_ste 完全一致），
    保证 QAT 模型的 pack→unpack 能位级复现前向的有效权重。"""
    limit = 2 ** (bits - 1) - 1
    scale = float((w.abs().max() / limit).item())
    scale = max(scale, 1e-12)
    q = torch.round(w / scale).clamp(-limit - 1, limit)
    return q.numpy().astype(_DTYPES[bits]), scale


def pack_model(model: OneDimINR, cfg: dict, bits: int = 8) -> bytes:
    """量化并打包模型 → 熵编码字节流（权重量化，bias/FiLM 保持 fp32）。"""
    weights, biases, bias_shapes = [], {}, {}
    for name, p in model.named_parameters():
        w = p.detach().cpu().float()
        if name.endswith(".bias"):
            biases[name] = w.numpy().tobytes()
            bias_shapes[name] = list(w.shape)
            continue
        q, scale = _quantize_sym(w, bits)
        weights.append({"name": name, "q": q.tobytes(), "shape": list(q.shape), "scale": scale})

    film = {k: v.detach().cpu().float().numpy().tobytes() for k, v in model.named_buffers()}
    film_shapes = {k: list(v.shape) for k, v in model.named_buffers()}

    blob = pickle.dumps(
        {
            "format": FORMAT,
            "bits": bits,
            "cfg": cfg,
            "layers": weights,
            "biases": biases,
            "bias_shapes": bias_shapes,
            "film": film,
            "film_shapes": film_shapes,
        }
    )
    return zlib.compress(blob, 9)


def unpack_model(blob: bytes) -> tuple[OneDimINR, dict]:
    """解压并还原模型（权重反量化，供推理/重建使用）。"""
    data = pickle.loads(zlib.decompress(blob))
    if data["format"] != FORMAT:
        raise ValueError(f"不支持的模型格式 {data['format']}")
    bits = data["bits"]
    dtype = _DTYPES[bits]
    cfg = data["cfg"]

    model = OneDimINR(film_seed=0, **cfg)
    sd = model.state_dict()
    for layer in data["layers"]:
        name = layer["name"]
        q = np.frombuffer(layer["q"], dtype=dtype).reshape(layer["shape"])
        sd[name].copy_(torch.from_numpy(q.astype(np.float32)) * layer["scale"])
    for k, raw in data["biases"].items():
        arr = np.frombuffer(raw, dtype=np.float32).reshape(data["bias_shapes"][k])
        sd[k].copy_(torch.from_numpy(arr.copy()))
    for k, raw in data["film"].items():
        arr = np.frombuffer(raw, dtype=np.float32).reshape(data["film_shapes"][k])
        sd[k].copy_(torch.from_numpy(arr.copy()))
    model.load_state_dict(sd)
    return model, cfg
