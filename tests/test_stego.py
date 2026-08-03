"""stego 包单元测试：后端抽象、stub、lfvsn 错误路径、工厂。"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from chimeraforge.stego import get_backend, lfvsn
from chimeraforge.stego.base import StegoBackend
from chimeraforge.stego.stub import StubBackend


# ---------- base ----------
def test_base_not_implemented(tmp_path):
    b = StegoBackend()
    with pytest.raises(NotImplementedError):
        b.embed(b"x", None, tmp_path / "o", b"k")
    with pytest.raises(NotImplementedError):
        b.extract(tmp_path / "s", tmp_path / "o", b"k")


# ---------- stub ----------
def test_stub_roundtrip_with_cover(tmp_path):
    data = bytes(range(256)) * 2
    key = b"stego-key"
    cover = tmp_path / "cover.mp4"
    cover.write_bytes(b"FAKE-COVER-HEADER" + b"\x00" * 40)
    stego, out = tmp_path / "stego.cfb", tmp_path / "out.bin"

    info = StubBackend().embed(data, cover, stego, key)
    assert info["cover_header"] == 32
    assert stego.read_bytes()[:32] == cover.read_bytes()[:32]

    assert StubBackend().extract(stego, out, key) == len(data)
    assert out.read_bytes() == data


def test_stub_roundtrip_no_cover(tmp_path):
    data, key = b"payload-data", b"k"
    stego, out = tmp_path / "stego.cfb", tmp_path / "out.bin"
    StubBackend().embed(data, None, stego, key)
    assert StubBackend().extract(stego, out, key) == len(data)
    assert out.read_bytes() == data


def test_stub_wrong_key(tmp_path):
    stego, out = tmp_path / "stego.cfb", tmp_path / "out.bin"
    StubBackend().embed(b"secret", None, stego, b"good")
    StubBackend().extract(stego, out, b"bad")
    assert out.read_bytes() != b"secret"


# ---------- factory ----------
def test_factory_unknown_backend():
    with pytest.raises(ValueError, match="未知"):
        get_backend(OmegaConf.create({"stego": {"backend": "nope"}}))


def _lfvsn_cfg(tmp_path):
    # 基于完整默认配置，仅覆盖 stego 段（保证 ecc 等顶层键存在）
    cfg = OmegaConf.load("configs/default.yaml")
    cfg.stego.backend = "lfvsn"
    cfg.stego.lfvsn.weight_dir = str(tmp_path)
    cfg.stego.lfvsn.mode = 1
    return cfg


def test_factory_lfvsn_missing_weight(tmp_path):
    with pytest.raises(RuntimeError, match="预训练权重"):
        get_backend(_lfvsn_cfg(tmp_path))


def test_factory_lfvsn_ok_with_fake_weight(tmp_path):
    (tmp_path / "lfvsn_mode1.pth").write_bytes(b"fake")
    backend = get_backend(_lfvsn_cfg(tmp_path))
    assert backend.name == "lfvsn"
    # 权重文件非合法 checkpoint → 加载时报清晰错误（M1 起 embed 为真实实现）
    with pytest.raises(RuntimeError, match="权重加载失败"):
        backend.embed(b"x", None, tmp_path / "o", b"k")


def test_lfvsn_missing_repo(tmp_path, monkeypatch):
    import pathlib

    real_exists = pathlib.Path.exists
    repo = pathlib.Path(__file__).resolve().parents[1] / "third_party" / "LF-VSN"

    def fake_exists(self):
        if self.resolve() == repo.resolve():
            return False
        return real_exists(self)

    monkeypatch.setattr(pathlib.Path, "exists", fake_exists)
    with pytest.raises(RuntimeError, match="LF-VSN 源码"):
        lfvsn.LFVSNBackend(weight_dir=str(tmp_path), mode=1)


def test_lfvsn_extract_missing_stego(tmp_path, monkeypatch):
    """权重不可用时应优先报权重错误，而不是静默失败。"""
    with pytest.raises(RuntimeError, match="预训练权重"):
        lfvsn.LFVSNBackend(weight_dir=str(tmp_path), mode=1)
