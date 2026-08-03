"""1-D 隐式神经表示（INR）：块索引 → 字节块。

NeRV-MLP 风格：位置编码 + GELU MLP + Sigmoid 输出（[0,1] 字节值）。
先验密钥经 FiLM 调制（逐层 scale/shift）注入网络，实现密钥控制：
不同主密钥 → 不同调制 → 训练出的权重无法在错误密钥下重建内容。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, freqs: int):
        super().__init__()
        self.freqs = freqs

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (..., 1) 归一化索引，∈ [0,1)
        k = torch.arange(self.freqs, device=t.device, dtype=t.dtype).view(1, -1)
        ang = t * (math.pi * (2.0 ** k))  # (..., freqs)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class OneDimINR(nn.Module):
    """输入块索引，输出 block_size 个 [0,1] 归一化字节值。"""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        pos_freqs: int = 64,
        film_seed: int = 0,
    ):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks 必须 >= 1")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pos_freqs = pos_freqs

        self.pos = PositionalEncoding(pos_freqs)
        layers = [nn.Linear(pos_freqs * 2, hidden_dim)]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, block_size))
        self.net = nn.ModuleList(layers)
        self.act = nn.GELU()

        scale, shift = self._make_film(film_seed, hidden_dim, num_layers)
        self.register_buffer("_film_scale", scale)
        self.register_buffer("_film_shift", shift)

    @staticmethod
    def _make_film(seed: int, hidden_dim: int, num_layers: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(seed & 0xFFFFFFFF)
        scale = 1.0 + 0.1 * torch.randn(num_layers, hidden_dim, generator=g)
        shift = 0.05 * torch.randn(num_layers, hidden_dim, generator=g)
        return scale, shift

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        t = ((idx.float() + 0.5) / self.num_blocks).view(-1, 1)
        h = self.pos(t)  # (N, 2*freqs)
        for i, lin in enumerate(self.net[:-1]):
            h = lin(h)
            if i < self.num_layers:
                h = h * self._film_scale[i] + self._film_shift[i]
            h = self.act(h)
        # 线性输出头（推理时裁剪到 [0,1]），避免 Sigmoid 在 0/255 极值处饱和
        return self.net[-1](h)

    def cfg_dict(self) -> dict:
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "pos_freqs": self.pos_freqs,
        }


def save_model(model: OneDimINR, content_len: int, path: str) -> None:
    torch.save(
        {"state_dict": model.state_dict(), "cfg": model.cfg_dict(), "orig_size": content_len},
        path,
    )


def load_model(path: str) -> tuple[OneDimINR, dict, int]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    model = OneDimINR(film_seed=0, **data["cfg"])
    model.load_state_dict(data["state_dict"])
    return model, data["cfg"], data["orig_size"]
