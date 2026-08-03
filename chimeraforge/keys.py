"""密钥派生：主密钥（先验密钥）→ 各环节子密钥。

采用 HKDF 简化结构（HMAC-SHA256，域分离），保证：
  - 同一主密钥在不同环节派生互不相关的子密钥
  - 接收方只需持有主密钥即可重建所有环节密钥
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Union

_MAGIC = b"chimeraforge/v1"

KeyLike = Union[str, bytes]


def kdf(master_key: bytes, domain: str, *, salt: bytes = b"") -> bytes:
    """返回 32 字节域分离子密钥。"""
    prk = hmac.new(master_key, _MAGIC + salt, hashlib.sha256).digest()
    return hmac.new(prk, domain.encode("utf-8"), hashlib.sha256).digest()


def normalize_master_key(key: KeyLike) -> bytes:
    return key.encode("utf-8") if isinstance(key, str) else bytes(key)


def film_seed(master_key: bytes) -> int:
    """INR FiLM 调制的确定性种子（先验密钥控制）。"""
    return int.from_bytes(kdf(master_key, "inr/film"), "big")


def stego_key(master_key: bytes, backend: str) -> bytes:
    """隐写后端的环节密钥（后验密钥）。"""
    return kdf(master_key, f"stego/{backend}")


def mac_key(master_key: bytes) -> bytes:
    """载荷完整性/认证 MAC 密钥。"""
    return kdf(master_key, "payload/mac")


def gen_master_key() -> str:
    """生成 32 字节随机主密钥（hex）。"""
    return hashlib.sha256(__import__("os").urandom(32)).hexdigest()
