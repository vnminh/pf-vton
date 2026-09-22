from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from timm.models.vision_transformer import PatchEmbed

from patch_flow.models.pf_transformer import PatchForcingDiT, pf_modulate

from .geometry import mask_aligned_base_grid
from .lora import LoRAReport, add_transformer_lora
from .utils import extract_state_dict, pool_valid_mask, rectangular_pos_from_square


class GarmentCrossAttention(nn.Module):
    """Person-query -> garment-key/value cross attention.

    K gets normalized garment content + spatial position.
    V gets content only, preserving appearance payload instead of mixing position into values.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qk_norm: bool = True,
        logit_scale: float = 10.0,
        init_gate: float = 0.01,
    ):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qk_norm = qk_norm

        self.query_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(logit_scale), dtype=torch.float32))
        self.gate = nn.Parameter(torch.tensor(float(init_gate), dtype=torch.float32))

    def init_from_self_attention(self, attn: nn.Module) -> None:
        with torch.no_grad():
            qkv_w = attn.qkv.weight
            qkv_b = attn.qkv.bias
            h = self.hidden_size
            self.q_proj.weight.copy_(qkv_w[:h])
            self.k_proj.weight.copy_(qkv_w[h : 2 * h])
            self.v_proj.weight.copy_(qkv_w[2 * h :])
            if qkv_b is not None:
                self.q_proj.bias.copy_(qkv_b[:h])
                self.k_proj.bias.copy_(qkv_b[h : 2 * h])
                self.v_proj.bias.copy_(qkv_b[2 * h :])
            self.out_proj.weight.copy_(attn.proj.weight)
            if attn.proj.bias is not None:
                self.out_proj.bias.copy_(attn.proj.bias)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        person_tokens: torch.Tensor,
        garment_keys: torch.Tensor,
        garment_values: torch.Tensor,
        garment_valid: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        q = self._heads(self.q_proj(self.query_norm(person_tokens)))
        k = self._heads(self.k_proj(garment_keys))
        v = self._heads(self.v_proj(garment_values))

        if self.qk_norm:
            # Multiplying Q by the trainable temperature keeps the fused SDPA path
            # differentiable w.r.t. logit_scale (passing a detached Python float would not).
            temperature = self.logit_scale.exp().clamp(max=100.0).to(q.dtype)
            q = F.normalize(q.float(), dim=-1).to(q.dtype) * temperature
            k = F.normalize(k.float(), dim=-1).to(k.dtype)
            scale = 1.0
        else:
            scale = 1.0 / math.sqrt(self.head_dim)

        additive_mask = None
        if garment_valid is not None:
            additive_mask = torch.zeros(
                garment_valid.shape[0], 1, 1, garment_valid.shape[1],
                device=q.device, dtype=q.dtype,
            )
            additive_mask.masked_fill_(~garment_valid[:, None, None, :], torch.finfo(q.dtype).min)

        weights = None
        if need_weights:
            logits = torch.matmul(q, k.transpose(-2, -1)) * scale
            if additive_mask is not None:
                logits = logits + additive_mask
            weights = logits.float().softmax(dim=-1).to(v.dtype)
            out = torch.matmul(weights, v)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=additive_mask,
                dropout_p=0.0,
                scale=scale,
            )

        out = out.transpose(1, 2).reshape(person_tokens.shape[0], person_tokens.shape[1], self.hidden_size)
        out = self.out_proj(out)
        out = self.gate.to(out.dtype) * out
        return out, weights


class GarmentRoutingHead(nn.Module):
    """Single-purpose correspondence head with no generative value path."""

    def __init__(self, hidden_size: int, routing_dim: int = 64, logit_scale: float = 10.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.key_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.q_proj = nn.Linear(hidden_size, routing_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, routing_dim, bias=True)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(logit_scale), dtype=torch.float32))

    def forward(
        self,
        person_tokens: torch.Tensor,
        garment_tokens: torch.Tensor,
        garment_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        q = F.normalize(
            self.q_proj(self.query_norm(person_tokens)).float(), dim=-1
        )
        k = F.normalize(
            self.k_proj(self.key_norm(garment_tokens)).float(), dim=-1
        )
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale
        if garment_valid is not None:
            logits = logits.masked_fill(
                ~garment_valid[:, None, :], torch.finfo(logits.dtype).min
            )
        return logits.softmax(dim=-1)[:, None]


class VTONPatchForcingDiT(PatchForcingDiT):
    """512x384 VTON extension that leaves the pretrained PFT backbone structurally intact.

    New information enters by:
      1. a zero-init additive person-condition patch embedder (agnostic + DensePose + mask), and
      2. garment-only cross-attention adapters at selected PFT blocks.
    """

    def __init__(
        self,
        image_hw: Tuple[int, int] = (512, 384),
        latent_hw: Tuple[int, int] = (64, 48),
        patch_size: int = 2,
        garment_patch_size: int = 1,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        cross_blocks: Sequence[int] = (4, 8, 12, 16, 20, 24, 27),
        coral_blocks: Sequence[int] = (16, 24),
        qk_norm: bool = True,
        cross_logit_scale: float = 10.0,
        cross_init_gate: float = 0.01,
        use_dense_transport: bool = True,
        transport_mask_aligned_base: bool = False,
        transport_geometry_only: bool = False,
        transport_learned_residual: bool = True,
        transport_max_offset: float = 1.0,
        transport_fixed_gate: float = 0.02,
        transport_blocks: Sequence[int] = (20, 24, 27),
        transport_block_gate: float = 0.15,
        transport_output_scale: float = 1.0,
        cross_attention_scale: float = 0.25,
        source_condition_scale: float = 1.0,
        joint_garment_tokens: bool = False,
        joint_attention_blocks: Sequence[int] = (24,),
        joint_highres_cross_attention: bool = False,
        attention_transport: bool = False,
        attention_transport_block: int = 27,
        attention_transport_init_gate: float = 0.0,
        dedicated_router: bool = True,
        router_dim: int = 64,
        final_detail_attention: bool = False,
        final_detail_dim: int = 256,
        final_detail_heads: int = 8,
        predict_uncertainty: bool = True,
        num_classes: int = 1000,
        class_dropout_prob: float = 0.1,
    ):
        # Build a checkpoint-compatible PFT-XL first. input_size=32 is upstream ImageNet latent size.
        super().__init__(
            input_size=32,
            patch_size=patch_size,
            in_channels=4,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            predict_uncertainty=predict_uncertainty,
            num_classes=num_classes,
            class_dropout_prob=class_dropout_prob,
            compile=False,
        )
        self.image_hw = tuple(image_hw)
        self.latent_hw = tuple(latent_hw)
        self.token_hw = (latent_hw[0] // patch_size, latent_hw[1] // patch_size)
        self.garment_patch_size = int(garment_patch_size)
        self.garment_token_hw = (
            latent_hw[0] // garment_patch_size,
            latent_hw[1] // garment_patch_size,
        )
        self.cross_blocks = tuple(int(i) for i in cross_blocks)
        self.coral_blocks = tuple(int(i) for i in coral_blocks)
        self.num_classes_vton = num_classes
        self.use_dense_transport = bool(use_dense_transport)
        self.transport_mask_aligned_base = bool(transport_mask_aligned_base)
        self.transport_geometry_only = bool(transport_geometry_only)
        self.transport_learned_residual = bool(transport_learned_residual)
        self.transport_max_offset = float(transport_max_offset)
        self.transport_fixed_gate = float(transport_fixed_gate)
        if self.transport_fixed_gate < 0:
            raise ValueError("transport_fixed_gate must be non-negative")
        self.transport_blocks = tuple(int(i) for i in transport_blocks)
        invalid_transport_blocks = [i for i in self.transport_blocks if not 0 <= i < depth]
        if invalid_transport_blocks:
            raise ValueError(f"Invalid transport block indices: {invalid_transport_blocks}")
        self.transport_block_gate = float(transport_block_gate)
        self.transport_output_scale = float(transport_output_scale)
        self.cross_attention_scale = float(cross_attention_scale)
        self.source_condition_scale = float(source_condition_scale)
        self.joint_garment_tokens = bool(joint_garment_tokens)
        self.joint_attention_blocks = tuple(int(i) for i in joint_attention_blocks)
        self.joint_highres_cross_attention = bool(joint_highres_cross_attention)
        self.attention_transport = bool(attention_transport)
        self.attention_transport_block = int(attention_transport_block)
        self.attention_transport_gate = nn.Parameter(
            torch.tensor(float(attention_transport_init_gate), dtype=torch.float32)
        )
        self.dedicated_router = bool(dedicated_router)
        self.final_detail_attention = bool(final_detail_attention)
        if self.joint_garment_tokens and self.use_dense_transport:
            raise ValueError(
                "joint_garment_tokens and use_dense_transport are mutually exclusive"
            )
        invalid_joint_blocks = [i for i in self.joint_attention_blocks if not 0 <= i < depth]
        if invalid_joint_blocks:
            raise ValueError(f"Invalid joint-attention block indices: {invalid_joint_blocks}")
        if self.attention_transport:
            if not self.joint_highres_cross_attention:
                raise ValueError(
                    "attention_transport requires joint_highres_cross_attention=true"
                )
            if self.attention_transport_block not in self.cross_blocks:
                raise ValueError(
                    "attention_transport_block must be one of cross_blocks"
                )
            subpixels = self.patch_size * self.patch_size
            if self.garment_patch_size != 1:
                raise ValueError("attention_transport requires garment_patch_size=1")
            if num_heads % subpixels:
                raise ValueError(
                    "attention heads must be divisible by patch_size**2"
                )
        if self.final_detail_attention:
            if not self.joint_highres_cross_attention:
                raise ValueError(
                    "final_detail_attention requires joint_highres_cross_attention=true"
                )
            if self.garment_patch_size != 1:
                raise ValueError("final_detail_attention requires garment_patch_size=1")
            if int(final_detail_dim) % int(final_detail_heads):
                raise ValueError("final_detail_dim must be divisible by final_detail_heads")

        # Replace only the resolution-bound patch/position tensors. Conv weight shape stays checkpoint-compatible.
        self.x_embedder = PatchEmbed(
            img_size=self.latent_hw,
            patch_size=patch_size,
            in_chans=4,
            embed_dim=hidden_size,
            bias=True,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.token_hw[0] * self.token_hw[1], hidden_size),
            requires_grad=False,
        )

        # Person-side VTON condition: agnostic latent (4) + DensePose latent (4) + mask (1).
        self.person_condition_embedder = nn.Conv2d(
            9, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True
        )
        nn.init.zeros_(self.person_condition_embedder.weight)
        nn.init.zeros_(self.person_condition_embedder.bias)

        # CatVTON-style garment injection: clean, unwarped garment patches are
        # concatenated with person patches and participate in the *same*
        # pretrained self-attention at every block. This makes appearance a
        # first-class input instead of an optional late adapter. Reusing the
        # pretrained x_embedder also avoids a randomly initialized garment
        # encoder. The segment embedding disambiguates coincident coordinates.
        self.joint_garment_segment = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Persistent source endpoint: current x_t becomes increasingly target-
        # like as t grows, so it cannot by itself retain exact source identity.
        # Re-inject the deterministic warped latent and its confidence at every
        # time through the input token stream. Zero init keeps staged loads safe.
        self.source_condition_embedder = nn.Conv2d(
            5, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True
        )
        nn.init.zeros_(self.source_condition_embedder.weight)
        nn.init.zeros_(self.source_condition_embedder.bias)

        self.garment_embedder = PatchEmbed(
            img_size=self.latent_hw,
            patch_size=garment_patch_size,
            in_chans=4,
            embed_dim=hidden_size,
            bias=True,
        )
        self.garment_pos_embed = nn.Parameter(
            torch.zeros(1, self.garment_token_hw[0] * self.garment_token_hw[1], hidden_size),
            requires_grad=False,
        )
        self.garment_content_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Local garment transport complements global cross-attention. Cross-attention
        # can average unrelated sleeve/body colors; this branch samples one local
        # garment position per person location and therefore preserves blocks, logos,
        # and other high-frequency appearance signals.
        self.garment_transport_flow = nn.Sequential(
            nn.Conv2d(13, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.garment_transport_flow[-1].weight)
        nn.init.zeros_(self.garment_transport_flow[-1].bias)
        self.garment_transport_embedder = nn.Conv2d(
            5, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True
        )
        nn.init.zeros_(self.garment_transport_embedder.weight)
        nn.init.zeros_(self.garment_transport_embedder.bias)

        # The RGB branch is not a dataset-provided warped cloth. The learned grid
        # above warps the original shop garment online at 1/4 resolution. A CNN
        # packs sub-token lettering and color edges into transformer channels
        # before spatial downsampling would otherwise erase them.
        self.garment_rgb_transport_embedder = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(256, hidden_size, kernel_size=2, stride=2),
        )
        nn.init.zeros_(self.garment_rgb_transport_embedder[-1].weight)
        nn.init.zeros_(self.garment_rgb_transport_embedder[-1].bias)
        self.garment_transport_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )

        # A direct residual velocity head makes transported appearance reachable
        # from the model output instead of relying on 28 transformer blocks to
        # preserve one small early residual. It is zero-init for a safe V3 load.
        self.garment_transport_output = nn.Sequential(
            nn.Conv2d(5, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 4, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.garment_transport_output[-1].weight)
        nn.init.zeros_(self.garment_transport_output[-1].bias)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, latent_hw[0]),
            torch.linspace(-1.0, 1.0, latent_hw[1]),
            indexing="ij",
        )
        self.register_buffer(
            "transport_base_grid", torch.stack([xx, yy], dim=-1)[None], persistent=False
        )

        self.garment_cross_attn = nn.ModuleDict({
            str(i): GarmentCrossAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                qk_norm=qk_norm,
                logit_scale=cross_logit_scale,
                init_gate=cross_init_gate,
            )
            for i in self.cross_blocks
        })
        self.garment_router = None
        if self.attention_transport and self.dedicated_router:
            self.garment_router = GarmentRoutingHead(
                hidden_size=hidden_size,
                routing_dim=int(router_dim),
                logit_scale=cross_logit_scale,
            )

        self.attention_transport_rgb_encoder = None
        self.attention_transport_state_encoder = None
        self.attention_transport_output = None
        if self.attention_transport:
            # Directly transplanting a shop-image VAE latent is invalid because
            # its value encodes the surrounding shop-image context. Retrieve
            # local RGB instead, then learn a small target-context correction.
            self.attention_transport_rgb_encoder = nn.Sequential(
                nn.Conv2d(4, 64, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
            )
            self.attention_transport_state_encoder = nn.Sequential(
                nn.Conv2d(13, 128, kernel_size=3, padding=1),
                nn.SiLU(),
            )
            self.attention_transport_output = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(128, 4, kernel_size=3, padding=1),
            )
            nn.init.zeros_(self.attention_transport_output[-1].weight)
            nn.init.zeros_(self.attention_transport_output[-1].bias)

        # The backbone has one 1152-D query for each 2x2 latent patch. A single
        # correspondence distribution is therefore shared by four output
        # latents, which averages narrow letters and color boundaries. This
        # lightweight final branch creates one query per latent subpixel and
        # reads the native 64x48 garment tokens independently. Its output is
        # zero initialized, so enabling it preserves an existing checkpoint at
        # step zero and it can learn to add detail without becoming the source
        # state of the diffusion process.
        self.final_detail_person_proj = None
        self.final_detail_garment_proj = None
        self.final_detail_cross_attn = None
        self.final_detail_subpixel = None
        self.final_detail_output = None
        if self.final_detail_attention:
            detail_dim = int(final_detail_dim)
            self.final_detail_person_proj = nn.Linear(hidden_size, detail_dim)
            self.final_detail_garment_proj = nn.Linear(hidden_size, detail_dim)
            self.final_detail_cross_attn = GarmentCrossAttention(
                hidden_size=detail_dim,
                num_heads=int(final_detail_heads),
                qk_norm=qk_norm,
                logit_scale=cross_logit_scale,
                init_gate=1.0,
            )
            self.final_detail_subpixel = nn.Parameter(
                torch.zeros(1, 1, patch_size * patch_size, detail_dim)
            )
            nn.init.normal_(self.final_detail_subpixel, std=0.02)
            self.final_detail_output = nn.Linear(detail_dim, 4)
            nn.init.zeros_(self.final_detail_output.weight)
            nn.init.zeros_(self.final_detail_output.bias)

    @property
    def num_person_tokens(self) -> int:
        return self.token_hw[0] * self.token_hw[1]

    @property
    def num_garment_tokens(self) -> int:
        return self.garment_token_hw[0] * self.garment_token_hw[1]

    def _unpatchify_rect(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.patch_size
        gh, gw = self.token_hw
        if x.shape[1] != gh * gw:
            raise ValueError(f"Expected {gh*gw} person tokens, got {x.shape[1]}")
        x = x.reshape(x.shape[0], gh, gw, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, gh * p, gw * p)

    def load_pretrained_pft(self, checkpoint_path: str) -> dict:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = extract_state_dict(ckpt)
        source_pos = state.pop("pos_embed", None)

        # Shape-safe load; only resolution-specific position tensor is intentionally omitted.
        own = self.state_dict()
        compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        result = self.load_state_dict(compatible, strict=False)

        if source_pos is None:
            raise KeyError("PFT checkpoint has no pos_embed; cannot initialize rectangular positions")
        with torch.no_grad():
            self.pos_embed.copy_(rectangular_pos_from_square(source_pos, self.token_hw).to(self.pos_embed.dtype))
            self.garment_pos_embed.copy_(
                rectangular_pos_from_square(source_pos, self.garment_token_hw).to(self.garment_pos_embed.dtype)
            )

            # Garment latent patch embed starts from the pretrained image-latent patch embed.
            src_w = self.x_embedder.proj.weight
            src_b = self.x_embedder.proj.bias
            if self.garment_patch_size == self.patch_size:
                self.garment_embedder.proj.weight.copy_(src_w)
            elif self.garment_patch_size == 1 and self.patch_size == 2:
                # Average spatial kernel direction, rescaled to roughly preserve output variance.
                self.garment_embedder.proj.weight.copy_(src_w.mean(dim=(-2, -1), keepdim=True) * 2.0)
            else:
                nn.init.xavier_uniform_(self.garment_embedder.proj.weight.flatten(1))
            if src_b is not None:
                self.garment_embedder.proj.bias.copy_(src_b)

            # Start dense transport as a weak identity warp using the pretrained
            # latent patch projection. The learned flow then bends it to the pose.
            self.garment_transport_embedder.weight.zero_()
            self.garment_transport_embedder.weight[:, :4].copy_(src_w)
            if src_b is not None:
                self.garment_transport_embedder.bias.copy_(src_b)

            # Cross-attention projections inherit each block's pretrained self-attention projections.
            for idx in self.cross_blocks:
                self.garment_cross_attn[str(idx)].init_from_self_attention(self.blocks[idx].attn)

        return {
            "loaded_keys": len(compatible),
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
        }

    def configure_trainable(
        self,
        train_self_attention: bool = False,
        train_final_layer: bool = True,
        train_transport_flow: bool = True,
        train_transport_output: bool = True,
        train_source_condition: bool = True,
        train_garment_embedder: bool = True,
        train_cross_attention: bool = True,
        train_transport_features: bool = True,
        self_attention_blocks: Sequence[int] | None = None,
    ) -> None:
        self.requires_grad_(False)
        self.person_condition_embedder.requires_grad_(True)
        if self.joint_garment_tokens:
            self.joint_garment_segment.requires_grad_(True)
        if train_source_condition:
            self.source_condition_embedder.requires_grad_(True)
        if train_garment_embedder:
            self.garment_embedder.requires_grad_(True)
        if train_cross_attention:
            self.garment_cross_attn.requires_grad_(True)
            if self.attention_transport:
                self.attention_transport_gate.requires_grad_(True)
                if self.garment_router is not None:
                    self.garment_router.requires_grad_(True)
                self.attention_transport_rgb_encoder.requires_grad_(True)
                self.attention_transport_state_encoder.requires_grad_(True)
                self.attention_transport_output.requires_grad_(True)
            if self.final_detail_attention:
                self.final_detail_person_proj.requires_grad_(True)
                self.final_detail_garment_proj.requires_grad_(True)
                self.final_detail_cross_attn.requires_grad_(True)
                self.final_detail_subpixel.requires_grad_(True)
                self.final_detail_output.requires_grad_(True)
        if train_transport_flow:
            self.garment_transport_flow.requires_grad_(True)
        if train_transport_features:
            self.garment_transport_embedder.requires_grad_(True)
            self.garment_rgb_transport_embedder.requires_grad_(True)
        if train_transport_output:
            self.garment_transport_output.requires_grad_(True)
        if train_final_layer:
            self.final_layer.requires_grad_(True)
        if train_self_attention:
            selected = range(len(self.blocks)) if self_attention_blocks is None else self_attention_blocks
            for idx in selected:
                if not 0 <= int(idx) < len(self.blocks):
                    raise ValueError(f"Invalid self-attention block index: {idx}")
                self.blocks[int(idx)].attn.requires_grad_(True)

    def add_backbone_lora(
        self,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        include_adaln: bool = True,
    ) -> LoRAReport:
        return add_transformer_lora(
            self.blocks,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            include_adaln=include_adaln,
        )

    def _attention_transport_grid(
        self,
        attention: torch.Tensor,
        garment_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert semantic attention into a local RGB grid and confidence.

        The transformer query grid is 32x24 and each query decodes a 2x2 VAE
        patch. Head groups independently predict the source coordinate for the
        four subpixels. Attention entropy becomes an explicit
        confidence, so uncertain correspondence cannot overwrite the generator.
        The grid is consumed only by an output residual and never becomes x0.
        """
        b, heads, queries, source_tokens = attention.shape
        if source_tokens != garment_latent.shape[-2] * garment_latent.shape[-1]:
            raise ValueError("Attention/source token mismatch in attention transport")
        if queries != self.num_person_tokens:
            raise ValueError("Unexpected attention shape in attention transport")
        distribution = attention.float().mean(dim=1)
        gh, gw = garment_latent.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, gh, device=attention.device),
            torch.linspace(-1.0, 1.0, gw, device=attention.device),
            indexing="ij",
        )
        source_coords = torch.stack([xx, yy], dim=-1).reshape(source_tokens, 2)
        expected = torch.einsum("bqk,kd->bqd", distribution, source_coords)

        # Normalized entropy is zero for a certain local match and one for a
        # uniform map. Detach is intentionally avoided: routing supervision can
        # learn both the coordinate and whether it is safe to copy.
        entropy = -(
            distribution.clamp_min(1e-8) * distribution.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(max(source_tokens, 2))
        confidence = (1.0 - entropy).clamp(0.0, 1.0)

        h, w = self.token_hw
        sample_grid = F.interpolate(
            expected.reshape(b, h, w, 2).permute(0, 3, 1, 2),
            size=self.latent_hw,
            mode="bilinear",
            align_corners=True,
        ).permute(0, 2, 3, 1)
        confidence = F.interpolate(
            confidence.reshape(b, 1, h, w),
            size=self.latent_hw,
            mode="bilinear",
            align_corners=True,
        )
        return sample_grid, confidence

    @staticmethod
    def _masked_joint_attention(
        attn: nn.Module,
        x: torch.Tensor,
        key_valid: torch.Tensor,
        *,
        person_tokens: int,
        need_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run a timm Attention module with invalid garment keys masked.

        The fast SDPA path is used at all ordinary blocks. At the one diagnostic
        block requested by training, the explicit probabilities from person
        queries to garment keys are retained for routing regularization.
        """
        b, n, c = x.shape
        qkv = attn.qkv(x).reshape(
            b, n, 3, attn.num_heads, c // attn.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = attn.q_norm(q), attn.k_norm(k)
        additive_mask = torch.zeros(
            b, 1, 1, n, device=x.device, dtype=q.dtype
        )
        additive_mask.masked_fill_(
            ~key_valid[:, None, None, :], torch.finfo(q.dtype).min
        )

        garment_attention = None
        dropout_p = float(attn.attn_drop.p) if attn.training else 0.0
        if need_weights:
            logits = torch.matmul(q, k.transpose(-2, -1)) * attn.scale
            probabilities = (logits + additive_mask).float().softmax(dim=-1)
            out = torch.matmul(probabilities.to(v.dtype), v)
            # Head averaging keeps this tensor small while preserving the total
            # person->garment probability mass needed by the loss.
            garment_attention = probabilities[
                :, :, :person_tokens, person_tokens:
            ].mean(dim=1)
        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=additive_mask,
                dropout_p=dropout_p,
                scale=attn.scale,
            )
        out = out.transpose(1, 2).reshape(b, n, c)
        out = attn.proj(out)
        out = attn.proj_drop(out)
        return out, garment_attention

    def predict_transport_grid(
        self,
        agnostic_latent: torch.Tensor,
        densepose_latent: torch.Tensor,
        agnostic_mask_latent: torch.Tensor,
        garment_latent: torch.Tensor,
        garment_mask_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the garment-to-person backward sampling grid.

        This is deliberately exposed separately from ``forward`` so the exact
        same learned warp can define the source endpoint of the residual flow.
        VTON is deterministic image translation; making a good warp available
        only as weak conditioning while starting from unrelated Gaussian noise
        needlessly asks the diffusion path to redraw logos and color blocks.
        """
        if not self.use_dense_transport:
            raise RuntimeError("predict_transport_grid requires use_dense_transport=true")
        cond_map = torch.cat(
            [agnostic_latent, densepose_latent, agnostic_mask_latent], dim=1
        )
        # Geometry must determine the deformation, not garment appearance. If
        # RGB/latent content is fed here, the paired-only training set lets the
        # flow memorize logo/color-specific offsets that do not transfer to a
        # shuffled garment. Keep the checkpoint-compatible four input channels
        # by repeating the silhouette; appearance is sampled only *after* the
        # geometry has selected source coordinates.
        if self.transport_geometry_only:
            if garment_mask_latent is None:
                raise ValueError(
                    "transport_geometry_only requires garment_mask_latent"
                )
            # Signed occupancy is centered and has the same per-location vector
            # norm as four unit-variance latent channels. Repeating it keeps the
            # checkpoint-compatible 13-channel convolution without feeding any
            # color or text information into coordinate prediction.
            garment_geometry = (
                garment_mask_latent.to(garment_latent.dtype) * 2.0 - 1.0
            ).expand(-1, garment_latent.shape[1], -1, -1)
            transport_input = torch.cat([cond_map, garment_geometry], dim=1)
        else:
            transport_input = torch.cat([cond_map, garment_latent], dim=1)
        if self.transport_learned_residual and self.transport_max_offset > 0:
            offset_field = torch.tanh(self.garment_transport_flow(transport_input))
            offset = (
                offset_field.permute(0, 2, 3, 1).float()
                * self.transport_max_offset
            )
        else:
            # Paired image reconstruction is not correspondence supervision: it
            # taught the old dense field to stretch logos for lower global L1.
            # A frozen zero residual preserves the verified affine source warp.
            offset_field = torch.zeros(
                garment_latent.shape[0],
                2,
                self.latent_hw[0],
                self.latent_hw[1],
                device=garment_latent.device,
                dtype=garment_latent.dtype,
            )
            offset = offset_field.permute(0, 2, 3, 1).float()
        if self.transport_mask_aligned_base and garment_mask_latent is not None:
            base_grid = mask_aligned_base_grid(
                garment_mask_latent.float(),
                agnostic_mask_latent.float(),
                self.latent_hw,
            )
        else:
            base_grid = self.transport_base_grid.float().expand(
                garment_latent.shape[0], -1, -1, -1
            )
        # Do not clamp: out-of-bounds samples should be invalid rather than
        # repeating a border pixel, which can smear text across the garment.
        grid = base_grid + offset
        return grid, offset_field

    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        agnostic_latent: torch.Tensor,
        densepose_latent: torch.Tensor,
        agnostic_mask_latent: torch.Tensor,
        garment_latent: torch.Tensor,
        edit_token_mask: torch.Tensor,
        garment_mask_latent: torch.Tensor | None = None,
        garment_rgb: torch.Tensor | None = None,
        garment_mask_rgb: torch.Tensor | None = None,
        source_latent: torch.Tensor | None = None,
        source_confidence_latent: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        return_uncertainty: bool = False,
        return_attention: bool = False,
    ):
        if xt.shape[-2:] != self.latent_hw:
            raise ValueError(f"Expected latent HxW={self.latent_hw}, got {tuple(xt.shape[-2:])}")
        if t.shape != (xt.shape[0], self.num_person_tokens):
            raise ValueError(f"Expected t [B,{self.num_person_tokens}], got {tuple(t.shape)}")

        x = self.x_embedder(xt) + self.pos_embed.to(dtype=xt.dtype)
        cond_map = torch.cat([agnostic_latent, densepose_latent, agnostic_mask_latent], dim=1)
        person_cond = self.person_condition_embedder(cond_map).flatten(2).transpose(1, 2)
        x = x + person_cond
        if source_latent is not None:
            if source_confidence_latent is None:
                source_confidence_latent = torch.ones_like(source_latent[:, :1])
            source_map = torch.cat(
                [source_latent, source_confidence_latent.to(source_latent.dtype)], dim=1
            )
            source_cond = self.source_condition_embedder(source_map).flatten(2).transpose(1, 2)
            x = x + self.source_condition_scale * source_cond.to(x.dtype)

        transport_coord = None
        transport_offset = None
        transport_grid = None
        warped_garment = None
        warped_mask = None
        transport_tokens = None
        rgb_transport = None
        transport_velocity = None
        if self.use_dense_transport:
            grid, transport_offset = self.predict_transport_grid(
                agnostic_latent,
                densepose_latent,
                agnostic_mask_latent,
                garment_latent,
                garment_mask_latent,
            )
            # CUDA grid_sample requires a float grid even when the transformer
            # itself runs under bf16 autocast.
            transport_grid = grid
            warped_garment = F.grid_sample(
                garment_latent, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            )
            if garment_mask_latent is None:
                warped_mask = torch.ones_like(warped_garment[:, :1])
            else:
                warped_mask = F.grid_sample(
                    garment_mask_latent.float(), grid.float(), mode="bilinear",
                    padding_mode="zeros", align_corners=True,
                ).to(warped_garment.dtype)
            warped_garment = warped_garment * warped_mask
            latent_transport = self.garment_transport_embedder(
                torch.cat([warped_garment, warped_mask], dim=1)
            ).flatten(2).transpose(1, 2)

            if garment_rgb is not None:
                quarter_hw = (self.image_hw[0] // 4, self.image_hw[1] // 4)
                source_rgb = F.interpolate(
                    garment_rgb.float(), size=quarter_hw, mode="bilinear", align_corners=True
                )
                if garment_mask_rgb is None:
                    source_mask = torch.ones_like(source_rgb[:, :1])
                else:
                    source_mask = F.interpolate(
                        garment_mask_rgb.float(), size=quarter_hw, mode="bilinear", align_corners=True
                    )
                quarter_grid = F.interpolate(
                    grid.permute(0, 3, 1, 2), size=quarter_hw,
                    mode="bilinear", align_corners=True,
                ).permute(0, 2, 3, 1)
                warped_rgb = F.grid_sample(
                    source_rgb, quarter_grid, mode="bilinear",
                    padding_mode="zeros", align_corners=True,
                )
                warped_rgb_mask = F.grid_sample(
                    source_mask, quarter_grid, mode="bilinear",
                    padding_mode="zeros", align_corners=True,
                )
                warped_rgb = warped_rgb * warped_rgb_mask
                rgb_transport = self.garment_rgb_transport_embedder(
                    torch.cat([warped_rgb, warped_rgb_mask], dim=1).to(xt.dtype)
                ).flatten(2).transpose(1, 2)

            transport_tokens = latent_transport
            if rgb_transport is not None:
                transport_tokens = transport_tokens + rgb_transport
            transport_tokens = self.garment_transport_norm(transport_tokens)
            edit = edit_token_mask.to(x.dtype)[..., None]
            # Small early context plus strong repeated late residuals. All scales
            # are fixed so optimization cannot silently close the copy route.
            x = x + edit * self.transport_fixed_gate * transport_tokens.to(x.dtype)

            token_grid = F.avg_pool2d(
                grid.permute(0, 3, 1, 2), kernel_size=self.patch_size, stride=self.patch_size
            )
            # grid_sample uses (x,y) in [-1,1]; CORAL uses (y,x) in [0,1].
            transport_coord = torch.stack(
                [(token_grid[:, 1] + 1.0) * 0.5, (token_grid[:, 0] + 1.0) * 0.5], dim=-1
            ).flatten(1, 2)

        attention_maps: Dict[int, torch.Tensor] = {}
        joint_attention_maps: Dict[int, torch.Tensor] = {}
        attention_transport_grid = None
        attention_transport_confidence = None
        key_valid = None
        garment_keys = garment_values = cross_garment_valid = None
        if self.joint_garment_tokens:
            garment_content = (
                self.x_embedder(garment_latent)
                + self.pos_embed.to(dtype=xt.dtype)
                + self.joint_garment_segment.to(dtype=xt.dtype)
            )
            garment_valid = pool_valid_mask(garment_mask_latent, self.token_hw)
            if garment_valid is None:
                garment_valid = torch.ones(
                    xt.shape[0], self.num_person_tokens,
                    device=xt.device, dtype=torch.bool,
                )
            person_valid = torch.ones_like(garment_valid)
            key_valid = torch.cat([person_valid, garment_valid], dim=1)
            x = torch.cat([x, garment_content], dim=1)
            # The reference garment is always clean data, independent of the
            # noising time of the generated person tokens.
            garment_t = torch.ones_like(t)
            cond_t = torch.cat([t, garment_t], dim=1)
            if self.joint_highres_cross_attention:
                # Preserve the native 64x48 garment latent grid for lettering
                # and narrow color boundaries. The joint 32x24 stream handles
                # global layout; this route supplies a sharp local payload.
                highres_content = self.garment_embedder(garment_latent)
                garment_keys = (
                    self.garment_content_norm(highres_content)
                    + self.garment_pos_embed.to(highres_content.dtype)
                )
                garment_values = highres_content
                cross_garment_valid = pool_valid_mask(
                    garment_mask_latent, self.garment_token_hw
                )
        else:
            garment_content = self.garment_embedder(garment_latent)
            garment_keys = (
                self.garment_content_norm(garment_content)
                + self.garment_pos_embed.to(garment_content.dtype)
            )
            # Deliberately position-free V: it carries visual payload, not routing coordinates.
            garment_values = garment_content
            garment_valid = pool_valid_mask(garment_mask_latent, self.garment_token_hw)
            cross_garment_valid = garment_valid
            cond_t = t

        t_emb = self.t_embedder(cond_t[..., None]).squeeze(1)
        if y is None:
            y = torch.full((xt.shape[0],), self.num_classes_vton, device=xt.device, dtype=torch.long)
        y_emb = self.y_embedder(y, self.training)
        y_emb = repeat(y_emb, "b c -> b n c", n=x.shape[1])
        cond = t_emb + y_emb
        edit = edit_token_mask.to(x.dtype)[..., None]

        for idx, block in enumerate(self.blocks):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaLN_modulation(cond).chunk(6, dim=-1)
            self_input = pf_modulate(block.norm1(x), shift_msa, scale_msa)
            if self.joint_garment_tokens:
                attn_out, joint_weights = self._masked_joint_attention(
                    block.attn,
                    self_input,
                    key_valid,
                    person_tokens=self.num_person_tokens,
                    need_weights=return_attention and idx in self.joint_attention_blocks,
                )
                x = x + gate_msa * attn_out
                if joint_weights is not None:
                    joint_attention_maps[idx] = joint_weights
            else:
                x = x + gate_msa * block.attn(self_input)

            if (
                idx in self.cross_blocks
                and (not self.joint_garment_tokens or self.joint_highres_cross_attention)
            ):
                need_transport = (
                    self.attention_transport
                    and not self.dedicated_router
                    and idx == self.attention_transport_block
                )
                need_weights = (
                    (
                        return_attention
                        and not self.dedicated_router
                        and idx in self.coral_blocks
                    )
                    or need_transport
                )
                cross_input = (
                    x[:, :self.num_person_tokens]
                    if self.joint_garment_tokens else x
                )
                cross, weights = self.garment_cross_attn[str(idx)](
                    person_tokens=cross_input,
                    garment_keys=garment_keys,
                    garment_values=garment_values,
                    garment_valid=cross_garment_valid,
                    need_weights=need_weights,
                )
                person_updated = (
                    cross_input + edit * self.cross_attention_scale * cross
                )
                if self.joint_garment_tokens:
                    x = torch.cat(
                        [person_updated, x[:, self.num_person_tokens:]], dim=1
                    )
                else:
                    x = person_updated
                if weights is not None:
                    if return_attention and idx in self.coral_blocks:
                        attention_maps[idx] = weights
                    if need_transport:
                        (
                            attention_transport_grid,
                            attention_transport_confidence,
                        ) = self._attention_transport_grid(
                            weights, garment_latent
                        )

            if transport_tokens is not None and idx in self.transport_blocks:
                x = x + edit * self.transport_block_gate * transport_tokens.to(x.dtype)

            x = x + gate_mlp * block.mlp(pf_modulate(block.norm2(x), shift_mlp, scale_mlp))

        # Garment tokens are a conditioning stream; only person tokens are decoded.
        person_x = x[:, :self.num_person_tokens]
        person_cond = cond[:, :self.num_person_tokens]
        if self.attention_transport and self.dedicated_router:
            if self.garment_router is None or garment_keys is None:
                raise RuntimeError("Dedicated garment router has no garment tokens")
            routing_weights = self.garment_router(
                person_x,
                garment_keys,
                cross_garment_valid,
            )
            (
                attention_transport_grid,
                attention_transport_confidence,
            ) = self._attention_transport_grid(routing_weights, garment_latent)
            if return_attention:
                attention_maps[-1] = routing_weights
        out = self.final_layer(person_x, person_cond)
        out = self._unpatchify_rect(out)
        logvar = None
        if self.predict_uncertainty:
            logvar = out[:, -1:]
            out = out[:, :-1]

        final_detail_velocity = None
        if self.final_detail_attention:
            if garment_keys is None or garment_values is None:
                raise RuntimeError("Final detail attention has no high-resolution garment stream")
            b = person_x.shape[0]
            p = self.patch_size
            detail_person = self.final_detail_person_proj(person_x)
            detail_queries = (
                detail_person[:, :, None, :]
                + self.final_detail_subpixel.to(detail_person.dtype)
            ).reshape(b, self.num_person_tokens * p * p, -1)
            detail_keys = self.final_detail_garment_proj(garment_keys)
            detail_values = self.final_detail_garment_proj(garment_values)
            detail_features, _ = self.final_detail_cross_attn(
                person_tokens=detail_queries,
                garment_keys=detail_keys,
                garment_values=detail_values,
                garment_valid=cross_garment_valid,
                need_weights=False,
            )
            detail_patches = self.final_detail_output(detail_features).reshape(
                b,
                self.token_hw[0],
                self.token_hw[1],
                p,
                p,
                4,
            )
            final_detail_velocity = torch.einsum(
                "nhwpqc->nchpwq", detail_patches
            ).reshape(b, 4, self.latent_hw[0], self.latent_hw[1])
            out = out + (
                agnostic_mask_latent.to(out.dtype)
                * final_detail_velocity.to(out.dtype)
            )

        attention_transport_velocity = None
        attention_transport_rgb = None
        if attention_transport_grid is not None and garment_rgb is not None:
            # Retrieve source RGB at 1/4 resolution, then convert it into a
            # latent correction conditioned on the generator's own endpoint,
            # person context and pose. This preserves lettering/color payload
            # without assuming a shop-image VAE latent is valid on the person.
            quarter_hw = (self.image_hw[0] // 4, self.image_hw[1] // 4)
            source_rgb = F.interpolate(
                garment_rgb.float(), size=quarter_hw,
                mode="bilinear", align_corners=True,
            )
            if garment_mask_rgb is None:
                source_mask = torch.ones_like(source_rgb[:, :1])
            else:
                source_mask = F.interpolate(
                    garment_mask_rgb.float(), size=quarter_hw,
                    mode="bilinear", align_corners=True,
                )
            quarter_grid = F.interpolate(
                attention_transport_grid.permute(0, 3, 1, 2),
                size=quarter_hw,
                mode="bilinear",
                align_corners=True,
            ).permute(0, 2, 3, 1)
            retrieved_mask = F.grid_sample(
                source_mask, quarter_grid, mode="bilinear",
                padding_mode="zeros", align_corners=True,
            )
            confidence_quarter = F.interpolate(
                attention_transport_confidence.float(), size=quarter_hw,
                mode="bilinear", align_corners=True,
            )
            retrieval_confidence = (
                retrieved_mask * confidence_quarter
            ).clamp(0.0, 1.0)
            attention_transport_rgb = F.grid_sample(
                source_rgb, quarter_grid, mode="bilinear",
                padding_mode="zeros", align_corners=True,
            ) * retrieved_mask
            rgb_features = self.attention_transport_rgb_encoder(
                torch.cat([attention_transport_rgb, retrieval_confidence], dim=1).to(out.dtype)
            )
            predicted_clean = xt.to(out.dtype) + out
            state_features = self.attention_transport_state_encoder(
                torch.cat([
                    predicted_clean,
                    agnostic_latent.to(out.dtype),
                    densepose_latent.to(out.dtype),
                    agnostic_mask_latent.to(out.dtype),
                ], dim=1)
            )
            attention_transport_velocity = self.attention_transport_output(
                torch.cat([rgb_features, state_features], dim=1)
            )
            out = out + (
                self.attention_transport_gate.to(out.dtype)
                * agnostic_mask_latent.to(out.dtype)
                * attention_transport_confidence.to(out.dtype)
                * attention_transport_velocity
            )

        if warped_garment is not None and warped_mask is not None:
            transport_velocity = self.garment_transport_output(
                torch.cat([warped_garment, warped_mask], dim=1).to(out.dtype)
            )
            out = out + (
                self.transport_output_scale
                * agnostic_mask_latent.to(out.dtype)
                * transport_velocity
            )

        if return_uncertainty or return_attention:
            return {
                "velocity": out,
                "logvar": logvar,
                "attention": attention_maps,
                "joint_attention": joint_attention_maps,
                "garment_valid": garment_valid,
                "cross_garment_valid": cross_garment_valid,
                "transport_coord": transport_coord,
                "transport_offset": transport_offset,
                "transport_grid": transport_grid,
                "warped_garment": warped_garment,
                "warped_mask": warped_mask,
                "transport_tokens": transport_tokens,
                "rgb_transport_tokens": rgb_transport,
                "transport_velocity": transport_velocity,
                "attention_transport_grid": attention_transport_grid,
                "attention_transport_confidence": attention_transport_confidence,
                "attention_transport_rgb": attention_transport_rgb,
                "attention_transport_velocity": attention_transport_velocity,
                "final_detail_velocity": final_detail_velocity,
            }
        return out
