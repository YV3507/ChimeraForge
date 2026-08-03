"""int8 量化感知训练（QAT）：STE 权重量化 + 微调。

将 nn.Linear 替换为 LinearQAT，前向使用量化到 int8 格点上的权重，
反向梯度按恒等流（STE）。微调用余弦退火稳定收敛，结束后由
train._snap_to_grid 把权重显式拉回格点，pack_model(bits=8) 与之位级一致。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def quantize_weight_ste(w: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """STE 量化：前向用量化值 q*scale，反向梯度恒等穿透。"""
    limit = 2 ** (bits - 1) - 1
    scale = w.detach().abs().max() / limit
    scale = torch.clamp(scale, min=1e-12)
    q = torch.clamp(torch.round(w / scale), -limit - 1, limit)
    wq = q * scale
    return w + (wq - w).detach()


class LinearQAT(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bits: int = 8, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        self.bits = bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, quantize_weight_ste(self.weight, self.bits), self.bias)


def convert_to_qat(model: nn.Module, bits: int = 8) -> nn.Module:
    """原位替换模型中的 nn.Linear 为 LinearQAT（state_dict 键不变）。"""
    for name, module in list(model._modules.items()):
        if isinstance(module, nn.Linear):
            q = LinearQAT(module.in_features, module.out_features, bits=bits, bias=module.bias is not None)
            q.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                q.bias.data.copy_(module.bias.data)
            model._modules[name] = q
        elif len(list(module.children())) > 0:
            convert_to_qat(module, bits)
    return model
