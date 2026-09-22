from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Linear):
    """Drop-in ``nn.Linear`` with a zero-initialized low-rank residual.

    The original ``weight`` and ``bias`` keep their exact state-dict names. This
    lets staged VTON checkpoints load before and after LoRA is enabled without
    renaming pretrained PFT tensors to ``base.weight``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.lora_rank = int(rank)
        self.lora_alpha = float(alpha)
        self.lora_scaling = self.lora_alpha / self.lora_rank
        self.lora_dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Linear(
            in_features, self.lora_rank, bias=False, device=device, dtype=dtype
        )
        self.lora_up = nn.Linear(
            self.lora_rank, out_features, bias=False, device=device, dtype=dtype
        )
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    @classmethod
    def from_linear(
        cls,
        source: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> "LoRALinear":
        result = cls(
            source.in_features,
            source.out_features,
            source.bias is not None,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        # Reuse the exact Parameter objects: values, dtype, device and frozen
        # state are preserved, and their checkpoint keys remain unchanged.
        result.weight = source.weight
        if source.bias is not None:
            result.bias = source.bias
        result.train(source.training)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        residual = self.lora_up(self.lora_down(self.lora_dropout(x)))
        return base + residual * self.lora_scaling


@dataclass(frozen=True)
class LoRAReport:
    modules: int
    parameters: int


def add_transformer_lora(
    blocks: nn.ModuleList,
    *,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    include_adaln: bool = True,
) -> LoRAReport:
    """Add LoRA to attention, MLP and optionally adaLN in every DiT block."""

    targets: list[tuple[nn.Module, str]] = []
    for block in blocks:
        targets.extend([
            (block.attn, "qkv"),
            (block.attn, "proj"),
            (block.mlp, "fc1"),
            (block.mlp, "fc2"),
        ])
        if include_adaln:
            targets.append((block.adaLN_modulation, "1"))

    module_count = 0
    parameter_count = 0
    for parent, name in targets:
        source = parent[int(name)] if isinstance(parent, nn.Sequential) else getattr(parent, name)
        if isinstance(source, LoRALinear):
            continue
        if not isinstance(source, nn.Linear):
            raise TypeError(f"Expected nn.Linear for LoRA target, got {type(source)!r}")
        replacement = LoRALinear.from_linear(
            source, rank=rank, alpha=alpha, dropout=dropout
        )
        if isinstance(parent, nn.Sequential):
            parent[int(name)] = replacement
        else:
            setattr(parent, name, replacement)
        module_count += 1
        parameter_count += sum(
            p.numel()
            for child_name, p in replacement.named_parameters()
            if child_name.startswith("lora_")
        )
    return LoRAReport(module_count, parameter_count)
