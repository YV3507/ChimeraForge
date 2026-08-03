"""M0 roundtrip 冒烟测试：五阶段端到端 + 密钥安全。

运行：python -m pytest tests/ -v
"""

from __future__ import annotations

import hashlib

import pytest
from omegaconf import OmegaConf

from chimeraforge import keys, payload as payload_mod
from chimeraforge.stages import compile_ as stage_compile
from chimeraforge.stages import compress as stage_compress
from chimeraforge.stages import embed as stage_embed
from chimeraforge.stages import extract as stage_extract
from chimeraforge.stages import reconstruct as stage_reconstruct

CONFIG = "configs/default.yaml"
MASTER_KEY = b"test-master-key"


@pytest.fixture
def cfg():
    c = OmegaConf.load(CONFIG)
    c.inr.hidden_dim = 128
    c.inr.num_layers = 3
    c.inr.epochs = 1000
    c.inr.qat_epochs = 0
    c.compress.bits = 16
    c.stego.backend = "stub"
    return c


def _run_roundtrip(cfg, content_path, outdir):
    model = outdir / "model.pt"
    payload = outdir / "payload.bin"
    stego = outdir / "stego.cfb"
    extracted = outdir / "payload.extracted.bin"
    content_out = outdir / "content.out.bin"

    stage_compile.run(str(content_path), str(model), cfg, MASTER_KEY)
    stage_compress.run(str(model), str(payload), cfg, MASTER_KEY)
    stage_embed.run(str(payload), None, str(stego), cfg, MASTER_KEY)
    stage_extract.run(str(stego), str(extracted), cfg, MASTER_KEY)
    stage_reconstruct.run(str(extracted), str(content_out), cfg, MASTER_KEY)
    return content_out


def test_roundtrip_smoke(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"ChimeraForge M0 roundtrip " + bytes(range(256)) * 2)
    out = _run_roundtrip(OmegaConf.load(CONFIG), secret, tmp_path)
    assert out.read_bytes() == secret.read_bytes()


def test_roundtrip_with_config(tmp_path):
    cfg = OmegaConf.load(CONFIG)
    cfg.inr.hidden_dim = 128
    cfg.inr.num_layers = 3
    cfg.inr.epochs = 1000
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"hello chimeraforge! " * 20)
    out = _run_roundtrip(cfg, secret, tmp_path)
    assert out.read_bytes() == secret.read_bytes()


def test_wrong_key_rejected(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"top-secret-content")
    out = _run_roundtrip(OmegaConf.load(CONFIG), secret, tmp_path)
    blob = (tmp_path / "payload.bin").read_bytes()
    with pytest.raises(ValueError, match="MAC"):
        payload_mod.unwrap(blob, mac_key=keys.mac_key(b"wrong-key"))


def test_demo_hash(tmp_path):
    """验证 demo 报告出的哈希与逐阶段一致。"""
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"demo hash check " * 10)
    out = _run_roundtrip(OmegaConf.load(CONFIG), secret, tmp_path)
    assert hashlib.sha256(out.read_bytes()).hexdigest() == hashlib.sha256(secret.read_bytes()).hexdigest()
