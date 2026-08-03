"""载荷 ↔ 秘密视频张量映射（媒体层编码）。

LF-VSN 的秘密视频重建是近似的（有损），因此把载荷字节编码为 0/1 比特、
映射到像素值 0/255（归一化 0.0/1.0），接收端用 0.5 阈值 + 冗余多数投票恢复。
载荷先与密钥流 XOR（机密性），可选 Reed-Solomon 整体纠错（抗残余比特翻转）。

帧布局（每帧容量 = H*W*3 // R 比特，R = redundancy）：
  明文头 | payload_len(8B)  → 明文（96 比特，供解码端定位长度）
  payload（已 XOR 密钥流）  → 密文比特流
所有比特（含头）均按 R 冗余逐比特复制到像素值 {0, 1}。
"""

from __future__ import annotations

import hashlib
import hmac

import numpy as np
import torch

MAGIC = b"CFM1"
_HDR_BITS = 96  # (4 + 8) * 8


def keystream(key: bytes, n: int) -> bytes:
    """HMAC-SHA256 计数器密钥流（与 stub 后端共用）。"""
    out = bytearray()
    ctr = 0
    while len(out) < n:
        out += hmac.new(key, ctr.to_bytes(8, "big"), hashlib.sha256).digest()
        ctr += 1
    return bytes(out[:n])


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    """比特数组（长度 8 的倍数）→ 字节（MSB 在前）。"""
    bits = bits.reshape(-1, 8)
    vals = (bits << np.arange(8)).sum(axis=1)
    return vals.astype(np.uint8).tobytes()


def _bytes_to_bits(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    return ((arr[:, None] >> np.arange(8)) & 1).reshape(-1)


def encode(
    payload: bytes,
    key: bytes,
    height: int,
    width: int,
    redundancy: int = 4,
) -> torch.Tensor:
    """载荷 → 秘密视频帧张量 (N, 3, H, W)，值域 {0, 1}。"""
    body = bytes(a ^ b for a, b in zip(payload, keystream(key, len(payload))))
    header = MAGIC + len(payload).to_bytes(8, "big")
    bits = np.concatenate([_bytes_to_bits(header), _bytes_to_bits(body)])

    cap = height * width * 3 // redundancy  # 每帧可承载的原始比特数
    total = bits.size
    n_frames = (total + cap - 1) // cap
    padded = np.pad(bits, (0, n_frames * cap - total))

    frame = np.zeros((n_frames, 3, height, width), dtype=np.float32)
    expanded = np.repeat(padded, redundancy)  # 每比特复制 R 次
    frame.reshape(-1)[: expanded.size] = expanded
    return torch.from_numpy(frame)


def decode(
    frames: torch.Tensor,
    key: bytes,
    redundancy: int = 4,
) -> bytes:
    """秘密视频帧张量 → 载荷字节（0.5 阈值 + 多数投票）。"""
    vals = frames.numpy() if isinstance(frames, torch.Tensor) else np.asarray(frames)
    flat = vals.reshape(-1)

    n_bits = (flat.size // redundancy) * redundancy
    if n_bits < _HDR_BITS * redundancy:
        raise ValueError("帧数不足，无法读取媒体层头")
    groups = flat[:n_bits].reshape(-1, redundancy)
    bits = (groups > 0.5).sum(axis=1) > redundancy // 2

    header = _bits_to_bytes(bits[:_HDR_BITS])
    if header[:4] != MAGIC:
        raise ValueError("媒体层头校验失败：密钥错误或载体损坏")
    payload_len = int.from_bytes(header[4:12], "big")
    need = (_HDR_BITS + payload_len * 8) * redundancy
    if n_bits < need:
        raise ValueError("载荷长度超出可用帧容量")
    body_bits = bits[_HDR_BITS : _HDR_BITS + payload_len * 8]
    body = _bits_to_bytes(body_bits)
    return bytes(a ^ b for a, b in zip(body, keystream(key, len(body))))


def disagreement_rate(frames: torch.Tensor, redundancy: int = 4) -> float:
    """诊断辅助：还原帧中冗余不一致分组的比例（越大越危险）。"""
    vals = frames.numpy()
    groups = vals.reshape(-1, redundancy)
    agrees = ((groups > 0.5).sum(axis=1) == 0) | ((groups > 0.5).sum(axis=1) == redundancy)
    return float(1.0 - agrees.mean())
