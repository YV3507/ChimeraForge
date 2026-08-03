"""keys.py 单元测试：KDF 域分离、归一化、种子派生。"""

from chimeraforge import keys


def test_kdf_domain_separation():
    mk = b"master"
    assert keys.kdf(mk, "a") != keys.kdf(mk, "b")
    assert len(keys.kdf(mk, "a")) == 32


def test_kdf_deterministic_and_keyed():
    mk1, mk2 = b"k1", b"k2"
    assert keys.kdf(mk1, "a") == keys.kdf(mk1, "a")
    assert keys.kdf(mk1, "a") != keys.kdf(mk2, "a")
    assert keys.kdf(mk1, "a", salt=b"s") != keys.kdf(mk1, "a")


def test_normalize_master_key():
    assert keys.normalize_master_key("abc") == b"abc"
    assert keys.normalize_master_key(b"abc") == b"abc"


def test_film_seed_deterministic():
    mk = b"mk"
    s1 = keys.film_seed(mk)
    assert isinstance(s1, int)
    assert keys.film_seed(mk) == s1
    assert keys.film_seed(b"other") != s1


def test_stego_key_domain():
    mk = b"mk"
    assert keys.stego_key(mk, "stub") != keys.stego_key(mk, "lfvsn")
    assert keys.mac_key(mk) != keys.stego_key(mk, "stub")
    assert len(keys.mac_key(mk)) == 32


def test_gen_master_key():
    k1, k2 = keys.gen_master_key(), keys.gen_master_key()
    assert len(k1) == 64 and all(c in "0123456789abcdef" for c in k1)
    assert k1 != k2
