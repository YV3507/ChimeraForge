"""共享 fixtures：快速训练配置 + 临时秘密文件。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

# 禁止导入时生成 __pycache__（尤其 third_party/LF-VSN submodule 内的 .pyc
# 会污染其 git 状态；配合 .gitmodules 的 ignore=dirty 双保险）
sys.dont_write_bytecode = True

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def cfg():
    """默认配置（适度缩小模型与轮数，保证测试速度与确定性）。"""
    c = OmegaConf.load(str(CONFIG))
    c.inr.hidden_dim = 192
    c.inr.num_layers = 3
    c.inr.epochs = 800
    c.inr.qat_epochs = 0  # QAT 由专门测试覆盖，链路测试保持快速
    c.inr.seed = 0
    c.compress.bits = 16  # int16 PTQ 无损，链路测试不依赖 QAT
    c.stego.backend = "stub"
    return c


@pytest.fixture
def secret(tmp_path):
    p = tmp_path / "secret.txt"
    p.write_bytes(b"ChimeraForge test secret " + bytes(range(64)))
    return p


@pytest.fixture
def fast_cfg_file(tmp_path):
    """快速训练配置的 YAML 覆盖文件（供 CLI 测试使用）。"""
    path = tmp_path / "fast.yaml"
    path.write_text(
        "inr:\n"
        "  hidden_dim: 192\n"
        "  num_layers: 3\n"
        "  epochs: 800\n"
        "  qat_epochs: 0\n"
        "  seed: 0\n"
        "compress:\n"
        "  bits: 16\n"
        "stego:\n"
        "  backend: stub\n"
    )
    return str(path)
