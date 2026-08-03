"""cli.py 单元测试：各子命令、密钥解析、错误处理、__main__。"""

import subprocess
import sys
from pathlib import Path

import pytest

from chimeraforge.cli import load_config, main, resolve_master_key


def test_genkey(capsys):
    assert main(["genkey"]) == 0
    out = capsys.readouterr().out.strip()
    assert len(out) == 64


def test_full_pipeline_cli(tmp_path, secret, fast_cfg_file):
    art = tmp_path / "art"
    art.mkdir()
    model, payload = art / "model.pt", art / "payload.bin"
    stego, extracted = art / "stego.cfb", art / "extracted.bin"
    out = art / "out.bin"
    args = ["--config", fast_cfg_file, "--key", "cli-test-key"]

    assert main(["compile", "--content", str(secret), "--out", str(model), *args]) == 0
    assert main(["compress", "--model", str(model), "--out", str(payload), *args]) == 0
    assert main(["embed", "--payload", str(payload), "--out", str(stego), *args]) == 0
    assert main(["extract", "--stego", str(stego), "--out", str(extracted), *args]) == 0
    assert main(["reconstruct", "--payload", str(extracted), "--out", str(out), *args]) == 0
    assert out.read_bytes() == secret.read_bytes()


def test_demo_cli(tmp_path, secret, fast_cfg_file):
    outdir = tmp_path / "demo"
    assert main(["demo", "--content", str(secret), "--outdir", str(outdir), "--config", fast_cfg_file]) == 0
    assert (outdir / "content.out.bin").read_bytes() == secret.read_bytes()


def test_error_missing_content(tmp_path, fast_cfg_file, capsys):
    rc = main(
        ["compile", "--content", str(tmp_path / "nope.txt"), "--out", str(tmp_path / "m.pt"),
         "--config", fast_cfg_file]
    )
    assert rc == 1
    assert "错误:" in capsys.readouterr().err


def test_error_unknown_backend(tmp_path, secret, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("stego:\n  backend: nope\n")
    rc = main(["embed", "--payload", str(secret), "--out", str(tmp_path / "x.cfb"), "--config", str(bad)])
    assert rc == 1
    assert "未知的隐写后端" in capsys.readouterr().err


def test_error_lfvsn_missing_weights(tmp_path, secret, capsys):
    cfg = tmp_path / "lfvsn.yaml"
    weight_dir = str(tmp_path / "w").replace("\\", "/")
    cfg.write_text(f"stego:\n  backend: lfvsn\n  lfvsn:\n    weight_dir: {weight_dir}\n    mode: 1\n")
    rc = main(["embed", "--payload", str(secret), "--out", str(tmp_path / "x.cfb"), "--config", str(cfg)])
    assert rc == 1
    assert "预训练权重" in capsys.readouterr().err


def test_key_resolution_precedence(monkeypatch, tmp_path):
    cfg = load_config(None)
    assert resolve_master_key(cfg, None) == cfg.master_key.encode()  # 兜底：config
    monkeypatch.setenv("CHIMERAFORGE_MASTER_KEY", "env-key")
    assert resolve_master_key(cfg, None) == b"env-key"  # 环境变量优先
    assert resolve_master_key(cfg, "--key-value") == b"--key-value"  # --key 最高


def test_main_module_runs():
    r = subprocess.run([sys.executable, "-m", "chimeraforge", "genkey"], capture_output=True, text=True)
    assert r.returncode == 0
    assert len(r.stdout.strip()) == 64


def test_main_module_via_runpy(monkeypatch):
    import runpy

    monkeypatch.setattr(sys, "argv", ["chimeraforge", "genkey"])
    with pytest.raises(SystemExit) as e:
        runpy.run_module("chimeraforge.__main__", run_name="__main__")
    assert e.value.code == 0


def test_cli_script_entry():
    r = subprocess.run([sys.executable, "-m", "chimeraforge.cli", "genkey"], capture_output=True, text=True)
    assert r.returncode == 0
    assert len(r.stdout.strip()) == 64


def test_cli_script_entry_runpy(monkeypatch):
    """进程内执行 cli.py 的 __main__ 入口（供 coverage 统计）。"""
    import runpy

    monkeypatch.setattr(sys, "argv", ["chimeraforge", "genkey"])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "chimeraforge" / "cli.py"), run_name="__main__")
    assert e.value.code == 0
