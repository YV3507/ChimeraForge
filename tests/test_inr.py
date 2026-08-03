"""inr 包单元测试：模型、训练、量化（PTQ）。"""

import io
import pickle
import zlib

import pytest
import torch

from chimeraforge.inr import model as inr_model
from chimeraforge.inr import quantize, train
from chimeraforge.inr.model import OneDimINR, load_model, save_model


def _trained(cfg):
    content = b"test content bytes " * 4  # 76 B → 1 block
    model, stats = train.train_inr(content, cfg.inr, film_seed=123, device="cpu")
    return model, stats, content


# ---------- model ----------
def test_forward_shape():
    m = OneDimINR(num_blocks=3, block_size=32, hidden_dim=16, num_layers=2, pos_freqs=8, film_seed=1)
    out = m(torch.tensor([0, 1, 2]))
    assert out.shape == (3, 32)


def test_num_blocks_zero_raises():
    with pytest.raises(ValueError):
        OneDimINR(num_blocks=0, block_size=32)


def test_cfg_dict_roundtrip(cfg):
    model, _, _ = _trained(cfg)
    m2 = OneDimINR(film_seed=0, **model.cfg_dict())
    assert m2.cfg_dict() == model.cfg_dict()


def test_save_load(tmp_path, cfg):
    model, _, content = _trained(cfg)
    path = str(tmp_path / "m.pt")
    save_model(model, len(content), path)
    m2, cfg2, size = load_model(path)
    assert size == len(content)
    assert cfg2 == model.cfg_dict()
    for k, v in model.state_dict().items():
        assert torch.equal(v, m2.state_dict()[k])


# ---------- train ----------
def test_empty_content_raises(cfg):
    with pytest.raises(ValueError, match="为空"):
        train.train_inr(b"", cfg.inr, film_seed=0, device="cpu")


def test_targets_padding():
    tgt, orig = train._targets(b"abc", 4)
    assert orig == 3
    assert tgt.shape == (1, 4)
    assert tgt[0, 3].item() == 0.0  # 末尾 padding 为 0


def test_train_converges(cfg):
    model, stats, content = _trained(cfg)
    assert stats.byte_accuracy == 1.0
    assert stats.final_loss < 1e-6
    assert stats.epochs >= 1
    assert train.infer_file(model, len(content), device="cpu") == content


def test_train_reproducible(cfg):
    _, s1, _ = _trained(cfg)
    _, s2, _ = _trained(cfg)
    assert s1.final_loss == s2.final_loss
    assert s1.byte_accuracy == s2.byte_accuracy


def test_train_progress(cfg, capsys):
    content = b"progress test"
    model, stats = train.train_inr(content, cfg.inr, film_seed=7, device="cpu", progress=True)
    assert stats.epochs >= 1
    capsys.readouterr()  # 吞掉进度输出


# ---------- quantize ----------
@pytest.mark.parametrize("bits", [8, 16])
def test_quantize_roundtrip(cfg, bits):
    model, _, content = _trained(cfg)
    packed = quantize.pack_model(model, model.cfg_dict(), bits=bits)
    m2, meta = quantize.unpack_model(packed)
    assert meta == model.cfg_dict()
    assert train.infer_file(m2, len(content), device="cpu") == content


def test_quantize_bad_format():
    blob = zlib.compress(pickle.dumps({"format": "bogus"}))
    with pytest.raises(ValueError, match="格式"):
        quantize.unpack_model(blob)


def test_quantize_reduces_size(cfg):
    model, _, _ = _trained(cfg)
    buf = io.BytesIO()
    torch.save({"sd": model.state_dict()}, buf)
    fp32 = len(buf.getvalue())
    p8 = len(quantize.pack_model(model, model.cfg_dict(), bits=8))
    p16 = len(quantize.pack_model(model, model.cfg_dict(), bits=16))
    assert p8 < p16 < fp32


# ---------- QAT（int8 量化感知训练） ----------
def test_qat_roundtrip_lossless(cfg):
    """QAT 后 int8 打包 → 解包 → 推理，必须字节级无损。"""
    content = b"QAT lossless content " * 8
    model, stats = train.train_inr(
        content, cfg.inr, film_seed=5, device="cpu", qat_epochs=400
    )
    assert stats.qat_epochs == 400
    assert stats.byte_accuracy == 1.0

    packed = quantize.pack_model(model, model.cfg_dict(), bits=8)
    m2, meta = quantize.unpack_model(packed)
    assert meta == model.cfg_dict()
    assert train.infer_file(m2, len(content), device="cpu") == content


def test_qat_pack_matches_forward(cfg):
    """pack→unpack 必须复现 QAT 前向的有效权重（scale 公式一致），保证无损。"""
    content = b"grid check " * 6
    model, _ = train.train_inr(content, cfg.inr, film_seed=9, device="cpu", qat_epochs=400)
    packed = quantize.pack_model(model, model.cfg_dict(), bits=8)
    m2, _ = quantize.unpack_model(packed)
    idx = torch.arange(model.num_blocks)
    with torch.no_grad():
        pred1 = model(idx)
        pred2 = m2(idx)
    assert torch.equal(pred1, pred2)
    assert train.infer_file(m2, len(content), device="cpu") == content
