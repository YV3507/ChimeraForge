"""payload.py 单元测试：载荷头、版本、MAC、篡改检测。"""

import pytest

from chimeraforge import payload as P

MAC = b"mac-key"
HASH = bytes(range(32))


def _wrap(**kw):
    kw.setdefault("mac_key", MAC)
    kw.setdefault("inr_hash", HASH)
    kw.setdefault("num_blocks", 2)
    kw.setdefault("block_size", 256)
    kw.setdefault("orig_size", 300)
    return P.wrap(b"data-bytes", **kw)


def test_roundtrip():
    header, data = P.unwrap(_wrap(), mac_key=MAC)
    assert data == b"data-bytes"
    assert header.orig_size == 300
    assert header.num_blocks == 2
    assert header.block_size == 256
    assert header.inr_hash == HASH
    assert header.data_size == len(data)
    assert not header.ecc_enabled


def test_ecc_flag():
    header, _ = P.unwrap(_wrap(ecc=True), mac_key=MAC)
    assert header.ecc_enabled


def test_short_blob():
    with pytest.raises(ValueError, match="过短"):
        P.unwrap(b"short", mac_key=MAC)


def test_bad_magic():
    blob = _wrap()
    with pytest.raises(ValueError, match="magic"):
        P.unwrap(b"XXXX" + blob[4:], mac_key=MAC)


def test_bad_version():
    blob = bytearray(_wrap())
    blob[4] = P.VERSION + 1
    with pytest.raises(ValueError, match="版本"):
        P.unwrap(bytes(blob), mac_key=MAC)


def test_bad_mac():
    with pytest.raises(ValueError, match="MAC"):
        P.unwrap(_wrap(), mac_key=b"wrong-mac-key")


def test_tamper_data():
    blob = bytearray(_wrap())
    blob[-20] ^= 0xFF  # 数据区尾字节翻转
    with pytest.raises(ValueError, match="MAC"):
        P.unwrap(bytes(blob), mac_key=MAC)


def test_hash_inr_cfg():
    assert P.hash_inr_cfg(b"{}") == P.hash_inr_cfg(b"{}")
    assert P.hash_inr_cfg(b"{}") != P.hash_inr_cfg(b"{x}")
    assert len(P.hash_inr_cfg(b"{}")) == 32
