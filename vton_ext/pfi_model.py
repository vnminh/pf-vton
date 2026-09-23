"""Patch-Forcing Inpainting DiT (PFI) for virtual try-on.

One pretrained PFT-XL/2 transformer, no warper, no transport, no pixel paste:

* Person stream: every latent patch of the target person. Patches inside the
  agnostic mask start from pure Gaussian noise and carry their own timestep;
  patches outside the mask are clean context (t=1), exactly like the
  conditioning frames of patch forcing.  Agnostic latent, mask and DensePose
  enter through a zero-initialised additive patch embedder, so step 0 equals
  the pretrained model.
* Garment stream: clean in-shop garment latent patches, also at t=1.  They
  run through the *same* blocks, attend only to each other, and expose their
  keys/values to the person stream.  Because garment tokens never see noisy
  person tokens, their per-block K/V are identical at every denoising step and
  are computed once per image at inference.
* CoRAL: selected blocks additionally return the person->garment attention
  distribution (softmax over garment keys) of a subset of heads, which the
  training loss aligns with DINOv3 correspondences.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from patch_flow.models.pf_transformer import PatchForcingDiT, pf_modulate
from vton_ext.utils import extract_state_dict, rectangular_pos_from_square


ROLE_EDIT, ROLE_KNOWN, ROLE_GARMENT = 0, 1, 2

GarmentKV = List[Tuple[torch.Tensor, torch.Tensor]]


class VTONInpaintDiT(PatchForcingDiT):
    def __init__(
        self,
        latent_hw: Tuple[int, int] = (64, 48),
        patch_size: int = 2,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        cond_channels: int = 9,
        garment_channels: int = 5,
        coral_blocks: Sequence[int] = (8, 12, 16, 20),
        coral_heads: int = 4,
        num_classes: int = 1000,
    ):
        super().__init__(
            input_size=32,
            patch_size=patch_size,
            in_channels=4,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            predict_uncertainty=True,
            num_classes=num_classes,
            compile=False,
        )
        self.latent_hw = tuple(latent_hw)
        self.token_hw = (latent_hw[0] // patch_size, latent_hw[1] // patch_size)
        self.num_tokens = self.token_hw[0] * self.token_hw[1]
        self.coral_blocks = tuple(int(i) for i in coral_blocks)
        self.coral_heads = min(int(coral_heads), num_heads)
        # Resolution-dependent schedule shift (SD3): t' = t / (t + a (1 - t)).
        # 1.0 = none. Set from cfg.flow.time_shift; used by training and samplers.
        self.time_shift = 1.0
        self.num_classes_ = num_classes
        self.gradient_checkpointing = False

        # Rectangular positions (initialised from the square pretrained table in
        # load_pretrained_pft). Garment tokens share them; the role embedding
        # tells the two streams apart.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_size), requires_grad=False)
        self.cond_embedder = nn.Conv2d(cond_channels, hidden_size, patch_size, patch_size)
        self.garment_embedder = nn.Conv2d(garment_channels, hidden_size, patch_size, patch_size)
        self.role_token = nn.Parameter(torch.zeros(3, hidden_size))
        self.role_cond = nn.Parameter(torch.zeros(3, hidden_size))
        nn.init.zeros_(self.cond_embedder.weight)
        nn.init.zeros_(self.cond_embedder.bias)

    # ------------------------------------------------------------------ setup

    def load_pretrained_pft(self, checkpoint_path: str) -> dict:
        state = extract_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
        source_pos = state.pop("pos_embed")
        own = self.state_dict()
        compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        result = self.load_state_dict(compatible, strict=False)
        with torch.no_grad():
            self.pos_embed.copy_(rectangular_pos_from_square(source_pos, self.token_hw))
            self.init_garment_embedder()
        return {"loaded": len(compatible), "missing": list(result.missing_keys)}

    def load_weights_any_resolution(self, checkpoint_path: str) -> dict:
        """Load a PFI checkpoint trained at another resolution (weights only).

        Every parameter is resolution-independent except the fixed position
        table, which is re-derived from the square pretrained PFT table exactly
        as at the source resolution (position interpolation), so the source and
        target models agree on the image-relative coordinate of every token.
        Call after load_pretrained_pft.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = dict(ckpt["model"] if "model" in ckpt else ckpt)
        state.pop("pos_embed", None)
        own = self.state_dict()
        bad = [k for k, v in state.items() if k in own and own[k].shape != v.shape]
        if bad:
            raise ValueError(f"shape mismatch beyond pos_embed: {bad[:5]}")
        result = self.load_state_dict(state, strict=False)
        missing = [k for k in result.missing_keys if k != "pos_embed"]
        if missing or result.unexpected_keys:
            raise ValueError(f"missing {missing[:5]} unexpected {result.unexpected_keys[:5]}")
        return {"loaded": len(state), "source_step": ckpt.get("step")}

    @torch.no_grad()
    def init_garment_embedder(self) -> None:
        """Garment latents are embedded exactly like clean image latents."""
        self.garment_embedder.weight.zero_()
        self.garment_embedder.weight[:, :4].copy_(self.x_embedder.proj.weight)
        self.garment_embedder.bias.copy_(self.x_embedder.proj.bias)

    def pretrained_parameters(self):
        new = {"cond_embedder", "garment_embedder", "role_token", "role_cond"}
        for name, p in self.named_parameters():
            if name.split(".")[0] not in new:
                yield name, p

    def new_parameters(self):
        new = {"cond_embedder", "garment_embedder", "role_token", "role_cond"}
        for name, p in self.named_parameters():
            if name.split(".")[0] in new:
                yield name, p

    # ---------------------------------------------------------------- helpers

    def _null_class(self, batch: int, device, dtype) -> torch.Tensor:
        # The pretrained CFG "null" class: keeps adaLN in its trained range.
        emb = self.y_embedder.embedding_table.weight[self.num_classes_]
        return emb.to(device=device, dtype=dtype).expand(batch, 1, -1)

    def _time_cond(self, t: torch.Tensor) -> torch.Tensor:
        b, n = t.shape
        return self.t_embedder(t.reshape(-1)).reshape(b, n, -1)

    def _patch(self, conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        return conv(x).flatten(2).transpose(1, 2)

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        p = self.patch_size
        th, tw = self.token_hw
        c = self.out_channels
        x = x.reshape(b, th, tw, p, p, c)
        return torch.einsum("nhwpqc->nchpwq", x).reshape(b, c, th * p, tw * p)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, -1).transpose(1, 2)

    def _qkv(self, attn, h: torch.Tensor):
        b, n, d = h.shape
        qkv = attn.qkv(h).reshape(b, n, 3, self.num_heads, d // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        return attn.q_norm(q), attn.k_norm(k), v

    def _out(self, attn, o: torch.Tensor) -> torch.Tensor:
        b, _, n, hd = o.shape
        o = o.transpose(1, 2).reshape(b, n, -1)
        if hasattr(attn, "norm"):
            o = attn.norm(o)
        return attn.proj(o)

    # ----------------------------------------------------------------- blocks

    def _garment_block(self, idx: int, x: torch.Tensor, c: torch.Tensor):
        blk = self.blocks[idx]
        sm, scm, gm, sp, scp, gp = blk.adaLN_modulation(c).chunk(6, dim=-1)
        q, k, v = self._qkv(blk.attn, pf_modulate(blk.norm1(x), sm, scm))
        x = x + gm * self._out(blk.attn, F.scaled_dot_product_attention(q, k, v))
        x = x + gp * blk.mlp(pf_modulate(blk.norm2(x), sp, scp))
        return x, k, v

    def _person_block(self, idx: int, x, c, kg, vg, want_attn: bool):
        blk = self.blocks[idx]
        sm, scm, gm, sp, scp, gp = blk.adaLN_modulation(c).chunk(6, dim=-1)
        q, k, v = self._qkv(blk.attn, pf_modulate(blk.norm1(x), sm, scm))
        o = F.scaled_dot_product_attention(q, torch.cat([k, kg], 2), torch.cat([v, vg], 2))
        x = x + gm * self._out(blk.attn, o)
        x = x + gp * blk.mlp(pf_modulate(blk.norm2(x), sp, scp))
        if not want_attn:
            return x, None
        h = self.coral_heads
        logits = torch.matmul(q[:, :h].float(), kg[:, :h].float().transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        return x, logits.softmax(dim=-1)

    def _run(self, fn, *args):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    # ---------------------------------------------------------------- forward

    def encode_garment(self, garment_latent: torch.Tensor, garment_mask: torch.Tensor) -> GarmentKV:
        """Per-block garment K/V. Independent of the noisy person state."""
        b = garment_latent.shape[0]
        x = self._patch(self.garment_embedder, torch.cat([garment_latent, garment_mask], 1))
        x = x + self.pos_embed + self.role_token[ROLE_GARMENT]
        ones = torch.ones(b, self.num_tokens, device=x.device, dtype=x.dtype)
        c = self._time_cond(ones) + self._null_class(b, x.device, x.dtype) + self.role_cond[ROLE_GARMENT]
        kv: GarmentKV = []
        for idx in range(len(self.blocks)):
            x, k, v = self._run(lambda x_, c_, i=idx: self._garment_block(i, x_, c_), x, c)
            kv.append((k, v))
        return kv

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        edit_tokens: torch.Tensor,
        garment_latent: Optional[torch.Tensor] = None,
        garment_mask: Optional[torch.Tensor] = None,
        garment_kv: Optional[GarmentKV] = None,
        return_uncertainty: bool = False,
        return_attention: bool = False,
    ):
        """
        x: (B,4,h,w) person latent state; t: (B,N) per-token time in [0,1];
        cond: (B,9,h,w) agnostic latent | latent mask | DensePose latent;
        edit_tokens: (B,N) bool, tokens inside the agnostic mask.
        """
        if garment_kv is None:
            garment_kv = self.encode_garment(garment_latent, garment_mask)
        b = x.shape[0]
        role = torch.where(edit_tokens, ROLE_EDIT, ROLE_KNOWN)
        h = self.x_embedder.proj(x).flatten(2).transpose(1, 2) + self._patch(self.cond_embedder, cond)
        h = h + self.pos_embed + self.role_token[role]
        c = self._time_cond(t) + self._null_class(b, x.device, h.dtype) + self.role_cond[role]

        attention: Dict[int, torch.Tensor] = {}
        for idx in range(len(self.blocks)):
            want = return_attention and idx in self.coral_blocks
            kg, vg = garment_kv[idx]
            h, a = self._run(
                lambda h_, c_, k_, v_, i=idx, w=want: self._person_block(i, h_, c_, k_, v_, w), h, c, kg, vg
            )
            if a is not None:
                attention[idx] = a
        out = self._unpatchify(self.final_layer(h, c))
        v, logvar = out[:, :-1], out[:, -1:]
        result = [v]
        if return_uncertainty:
            result.append(logvar)
        if return_attention:
            result.append(attention)
        return result[0] if len(result) == 1 else tuple(result)
