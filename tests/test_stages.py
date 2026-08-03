"""stages 补充测试：ECC 启用链路、INR 结构哈希校验失败。"""

import pytest

from chimeraforge import keys, payload as payload_mod
from chimeraforge.inr import load_model, pack_model
from chimeraforge.stages import compile_ as stage_compile
from chimeraforge.stages import compress as stage_compress
from chimeraforge.stages import reconstruct as stage_reconstruct

KEY = b"test-master-key"


def test_ecc_roundtrip(tmp_path, cfg):
    secret = tmp_path / "s.txt"
    secret.write_bytes(b"ecc roundtrip " * 10)
    cfg.ecc.enabled = True
    model, payload, out = tmp_path / "m.pt", tmp_path / "p.bin", tmp_path / "o.bin"

    stage_compile.run(str(secret), str(model), cfg, KEY)
    info = stage_compress.run(str(model), str(payload), cfg, KEY)
    assert info["ecc"] is True
    assert payload_mod.unwrap(payload.read_bytes(), mac_key=keys.mac_key(KEY))[0].ecc_enabled

    stage_reconstruct.run(str(payload), str(out), cfg, KEY)
    assert out.read_bytes() == secret.read_bytes()


def test_inr_hash_mismatch(tmp_path, cfg):
    secret = tmp_path / "s.txt"
    secret.write_bytes(b"hash mismatch test " * 5)
    model = tmp_path / "m.pt"
    stage_compile.run(str(secret), str(model), cfg, KEY)

    m, model_cfg, orig_size = load_model(str(model))
    packed = pack_model(m, model_cfg, bits=int(cfg.compress.bits))
    blob = payload_mod.wrap(
        packed,
        mac_key=keys.mac_key(KEY),
        inr_hash=b"\x00" * 32,  # 与实际模型结构哈希不一致
        num_blocks=model_cfg["num_blocks"],
        block_size=model_cfg["block_size"],
        orig_size=orig_size,
    )
    bad = tmp_path / "bad.bin"
    bad.write_bytes(blob)
    with pytest.raises(ValueError, match="INR 结构哈希"):
        stage_reconstruct.run(str(bad), str(tmp_path / "o.bin"), cfg, KEY)
