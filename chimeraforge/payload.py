"""载荷格式：magic + 版本 + 元数据 + 数据 + MAC。

载荷 = Header | 压缩后的 INR 模型字节流 | HMAC(Header+数据)
元数据携带重建所需的 INR 结构哈希、块参数与原始长度。
"""

from __future__ import annotations

import hashlib
import hmac
import struct
from dataclasses import dataclass

MAGIC = b"CFP1"
VERSION = 1
_MAC_LEN = 16
_FLAG_ECC = 1

# magic(4s) version(B) flags(B) reserved(H) orig_size(Q) num_blocks(I) block_size(I) inr_hash(32s)
_HEADER = struct.Struct(">4s B B H Q I I 32s")


@dataclass
class PayloadHeader:
    version: int
    flags: int
    orig_size: int
    num_blocks: int
    block_size: int
    inr_hash: bytes
    data_size: int

    @property
    def ecc_enabled(self) -> bool:
        return bool(self.flags & _FLAG_ECC)


def hash_inr_cfg(cfg_json: bytes) -> bytes:
    return hashlib.sha256(cfg_json).digest()


def wrap(
    data: bytes,
    *,
    mac_key: bytes,
    inr_hash: bytes,
    num_blocks: int,
    block_size: int,
    orig_size: int,
    ecc: bool = False,
) -> bytes:
    flags = _FLAG_ECC if ecc else 0
    header = _HEADER.pack(MAGIC, VERSION, flags, 0, orig_size, num_blocks, block_size, inr_hash)
    mac = hmac.new(mac_key, header + data, hashlib.sha256).digest()[:_MAC_LEN]
    return header + data + mac


def unwrap(blob: bytes, *, mac_key: bytes) -> tuple[PayloadHeader, bytes]:
    if len(blob) < _HEADER.size + _MAC_LEN:
        raise ValueError("载荷过短")
    magic, version, flags, _reserved, orig_size, num_blocks, block_size, inr_hash = _HEADER.unpack_from(blob)
    if magic != MAGIC:
        raise ValueError("非法载荷：magic 不匹配")
    if version != VERSION:
        raise ValueError(f"不支持的载荷版本 {version}")
    data = blob[_HEADER.size : -_MAC_LEN]
    expect = hmac.new(mac_key, blob[: _HEADER.size] + data, hashlib.sha256).digest()[:_MAC_LEN]
    if not hmac.compare_digest(expect, blob[-_MAC_LEN:]):
        raise ValueError("MAC 校验失败：主密钥错误或数据被篡改")
    header = PayloadHeader(version, flags, orig_size, num_blocks, block_size, inr_hash, len(data))
    return header, data
