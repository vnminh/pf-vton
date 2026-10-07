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
* Fine garment K/V (optional, ``fine_garment_from``): from that block on, the
  person stream reads the garment at latent resolution (one token per 8 px
  latent cell, 4x the tokens) instead of per 16 px patch. Each fine token is
  its coarse garment token, normalised and modulated as the block's K/V input,
  plus a residual from a small conv encoder of its own latent neighbourhood and
  its sub-patch position, projected by the block's own pretrained K/V. The
  residual enters *after* LayerNorm: garment hidden states reach rms 15-30 in
  the late blocks, so a residual added before the norm (V48) changed the keys by
  under 1% and never learned. Fine keys carry a learnable per-head attention
  bias initialised to -log 4 and the encoder output is zero-initialised, so at
  initialisation the four copies weigh exactly one coarse token: the model
  starts unchanged.
* Attention flow (``forward(..., flow_blocks=...)``): for chosen blocks, the
  person->garment attention of every head (softmax over garment keys only)
  is reduced to the expected garment coordinate and its spatial variance per
  person token, averaged over heads. It is computed as a second attention
  with the key coordinates as values, so the (queries x keys) map is never
  stored. Training warps the garment image with it (Leffa, CVPR 2025).
"""
from __future__ import annotations

import math

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from patch_flow.models.pf_transformer import PatchForcingDiT, pf_modulate
from vton_ext.utils import extract_state_dict, rectangular_pos_from_square, sincos_pos_embed


ROLE_EDIT, ROLE_KNOWN, ROLE_GARMENT = 0, 1, 2

GarmentKV = List[Tuple[torch.Tensor, torch.Tensor]]

FINE_PARAMETERS = {"fine_encoder", "fine_scale", "fine_pos_scale", "fine_bias"}
# V48's pre-norm fine path; dropped when loading its checkpoints.
RETIRED_PARAMETERS = {"fine_garment_embedder", "fine_gain", "fine_pos_gain"}
NEW_PARAMETERS = {"cond_embedder", "garment_embedder", "role_token", "role_cond"} | FINE_PARAMETERS


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
        pos_embed: str = "interpolate",
        pos_embed_trainable: bool = False,
        fine_garment_from: Optional[int] = None,
        fine_channels: int = 256,
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
        # "interpolate": bicubic-resize the square pretrained table (all tokens
        # inside its 16x16 position range); "sincos": native table for this grid.
        if pos_embed not in ("interpolate", "sincos"):
            raise ValueError("pos_embed must be 'interpolate' or 'sincos'")
        self.pos_embed_mode = pos_embed
        self.gradient_checkpointing = False

        # Rectangular positions (initialised from the square pretrained table in
        # load_pretrained_pft). Garment tokens share them; the role embedding
        # tells the two streams apart.
        # Trainable: starts from that table, so outputs are unchanged at step 0,
        # but fine-tuning can separate neighbouring tokens that interpolation
        # squeezed together (4x at 1024x768 from the 16x16 table).
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_size),
                                      requires_grad=bool(pos_embed_trainable))
        self.cond_embedder = nn.Conv2d(cond_channels, hidden_size, patch_size, patch_size)
        self.garment_embedder = nn.Conv2d(garment_channels, hidden_size, patch_size, patch_size)
        self.role_token = nn.Parameter(torch.zeros(3, hidden_size))
        self.role_cond = nn.Parameter(torch.zeros(3, hidden_size))
        nn.init.zeros_(self.cond_embedder.weight)
        nn.init.zeros_(self.cond_embedder.bias)

        self.fine_garment_from = None if fine_garment_from is None else int(fine_garment_from)
        if self.fine_garment_from is not None:
            if patch_size != 2:
                raise ValueError("fine garment tokens assume patch_size 2 (one fine token per latent cell)")
            if not 0 <= self.fine_garment_from < depth:
                raise ValueError("fine_garment_from must be a block index")
            if any(i >= self.fine_garment_from for i in self.coral_blocks):
                raise ValueError("CoRAL blocks must precede fine_garment_from (CoRAL reads coarse keys)")
            n_fine = depth - self.fine_garment_from
            # 5x5 latent context (40 px) per cell: enough for a stroke and its
            # neighbours. Zero-initialised output keeps step 0 unchanged.
            self.fine_encoder = nn.Sequential(
                nn.Conv2d(garment_channels, fine_channels, 3, 1, 1), nn.GELU(),
                nn.Conv2d(fine_channels, fine_channels, 3, 1, 1), nn.GELU(),
                nn.Conv2d(fine_channels, hidden_size, 1),
            )
            nn.init.zeros_(self.fine_encoder[-1].weight)
            nn.init.zeros_(self.fine_encoder[-1].bias)
            # Per-block, per-channel scales in the normalised K/V input space.
            self.fine_scale = nn.Parameter(torch.ones(n_fine, hidden_size))
            self.fine_pos_scale = nn.Parameter(torch.zeros(n_fine, hidden_size))
            self.fine_bias = nn.Parameter(torch.full((n_fine, num_heads), -math.log(4.0)))

    # ------------------------------------------------------------------ setup

    def load_pretrained_pft(self, checkpoint_path: str) -> dict:
        state = extract_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
        source_pos = state.pop("pos_embed")
        own = self.state_dict()
        compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        result = self.load_state_dict(compatible, strict=False)
        with torch.no_grad():
            if self.pos_embed_mode == "sincos":
                self.pos_embed.copy_(sincos_pos_embed(self.pos_embed.shape[-1], self.token_hw))
            else:
                self.pos_embed.copy_(rectangular_pos_from_square(source_pos, self.token_hw))
            self.init_garment_embedder()
        return {"loaded": len(compatible), "missing": list(result.missing_keys)}

    def load_weights_any_resolution(self, checkpoint_path: str) -> dict:
        """Load PFI weights, retaining learned positions at the same resolution.

        Every parameter is resolution-independent except the position table.
        Across resolutions it is re-derived by load_pretrained_pft
        (``pos_embed`` mode). With "interpolate" the source and target models
        agree on the image-relative coordinate of every token.
        Call after load_pretrained_pft.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = dict(ckpt["model"] if "model" in ckpt else ckpt)
        state = {k: v for k, v in state.items() if k.split(".")[0] not in RETIRED_PARAMETERS}
        source_hw = ckpt.get("config", {}).get("model", {}).get("latent_hw")
        position_loaded = ("pos_embed" in state and state["pos_embed"].shape == self.pos_embed.shape
                           and (source_hw is None or tuple(source_hw) == self.latent_hw))
        if not position_loaded:
            state.pop("pos_embed", None)
        own = self.state_dict()
        bad = [k for k, v in state.items() if k in own and own[k].shape != v.shape]
        if bad:
            raise ValueError(f"shape mismatch beyond pos_embed: {bad[:5]}")
        result = self.load_state_dict(state, strict=False)
        # A coarse-only parent has no fine garment weights; they keep their zero init.
        missing = [k for k in result.missing_keys
                   if k != "pos_embed" and k.split(".")[0] not in FINE_PARAMETERS]
        if missing or result.unexpected_keys:
            raise ValueError(f"missing {missing[:5]} unexpected {result.unexpected_keys[:5]}")
        return {"loaded": len(state), "source_step": ckpt.get("step"), "position_loaded": position_loaded}

    @torch.no_grad()
    def init_garment_embedder(self) -> None:
        """Garment latents are embedded exactly like clean image latents."""
        self.garment_embedder.weight.zero_()
        self.garment_embedder.weight[:, :4].copy_(self.x_embedder.proj.weight)
        self.garment_embedder.bias.copy_(self.x_embedder.proj.bias)

    def pretrained_parameters(self):
        for name, p in self.named_parameters():
            if name.split(".")[0] not in NEW_PARAMETERS:
                yield name, p

    def new_parameters(self):
        for name, p in self.named_parameters():
            if name.split(".")[0] in NEW_PARAMETERS:
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

    def _up2(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) coarse tokens -> (B, 4N, D), each repeated over its 2x2 latent cells."""
        b, _, d = x.shape
        th, tw = self.token_hw
        x = x.view(b, th, tw, d).repeat_interleave(2, 1).repeat_interleave(2, 2)
        return x.reshape(b, 4 * th * tw, d)

    def _fine_position_delta(self) -> torch.Tensor:
        """Sub-patch offset of each fine token: positions interpolated to the latent grid
        minus the position of the coarse token that contains it, rescaled to unit rms."""
        th, tw = self.token_hw
        grid = self.pos_embed.view(1, th, tw, -1).permute(0, 3, 1, 2).float()
        fine = F.interpolate(grid, size=(2 * th, 2 * tw), mode="bicubic", align_corners=False)
        fine = fine.permute(0, 2, 3, 1).reshape(1, 4 * th * tw, -1)
        delta = fine - self._up2(self.pos_embed.float())
        return (delta / delta.pow(2).mean().sqrt().clamp_min(1e-6)).to(self.pos_embed.dtype)

    def _garment_block(self, idx: int, x: torch.Tensor, c: torch.Tensor, fine_r=None, fine_pos=None):
        blk = self.blocks[idx]
        sm, scm, gm, sp, scp, gp = blk.adaLN_modulation(c).chunk(6, dim=-1)
        q, k, v = self._qkv(blk.attn, pf_modulate(blk.norm1(x), sm, scm))
        exposed_k, exposed_v = k, v
        if fine_r is not None:
            # Garment conditioning is identical for every token (t=1, garment role).
            j = idx - self.fine_garment_from
            hf = pf_modulate(blk.norm1(self._up2(x)), sm[:, :1], scm[:, :1])
            hf = hf + self.fine_scale[j] * fine_r + self.fine_pos_scale[j] * fine_pos
            _, exposed_k, exposed_v = self._qkv(blk.attn, hf)
        x = x + gm * self._out(blk.attn, F.scaled_dot_product_attention(q, k, v))
        x = x + gp * blk.mlp(pf_modulate(blk.norm2(x), sp, scp))
        return x, exposed_k, exposed_v

    def _key_coords(self, n_keys: int, device) -> torch.Tensor:
        """(n_keys, 4) [x, y, x^2, y^2] of the garment key centres, in pixels / max(H, W)
        (isotropic), for the coarse (16 px) or fine (8 px) garment grid."""
        lh, lw = self.latent_hw
        th, tw = self.token_hw
        gh, gw = (th, tw) if n_keys == th * tw else (lh, lw)
        if gh * gw != n_keys:
            raise ValueError(f"no garment grid with {n_keys} keys")
        side = max(lh, lw)
        ys = (torch.arange(gh, device=device, dtype=torch.float32) + 0.5) * (lh / gh) / side
        xs = (torch.arange(gw, device=device, dtype=torch.float32) + 0.5) * (lw / gw) / side
        y, x = torch.meshgrid(ys, xs, indexing="ij")
        x, y = x.reshape(-1), y.reshape(-1)
        return torch.stack([x, y, x * x, y * y], 1)

    def _attention_flow(self, q, kg, queries=None) -> torch.Tensor:
        """(B, N, 3): expected garment [x, y] per person token and the variance
        E|c|^2 - |E c|^2 of its attention, averaged over heads. fp32: the
        variance is a difference of second moments. ``queries`` (B, M) long:
        only those person tokens, (B, M, 3)."""
        if queries is not None:
            q = q.gather(2, queries[:, None, :, None].expand(-1, q.shape[1], -1, q.shape[3]))
        b, h, n, d = q.shape
        coords = self._key_coords(kg.shape[2], q.device)
        values = F.pad(coords, (0, d - 4))[None, None].expand(b, h, -1, -1)
        with torch.autocast(q.device.type, enabled=False):
            m = F.scaled_dot_product_attention(q.float(), kg.float(), values)[..., :4].mean(1)
        mean = m[..., :2]
        var = (m[..., 2:].sum(-1) - mean.square().sum(-1)).clamp_min(0)
        return torch.cat([mean, var[..., None]], -1)

    def _person_block(self, idx: int, x, c, kg, vg, want_attn: bool, bias=None, want_flow: bool = False,
                      flow_queries=None):
        blk = self.blocks[idx]
        sm, scm, gm, sp, scp, gp = blk.adaLN_modulation(c).chunk(6, dim=-1)
        q, k, v = self._qkv(blk.attn, pf_modulate(blk.norm1(x), sm, scm))
        mask = None
        if bias is not None:
            # Person keys unbiased; per-head bias on the fine garment keys
            # (-log 4 at init: each covers a quarter patch).
            heads = bias.shape[0]
            mask = torch.cat([bias.new_zeros(heads, k.shape[2]), bias[:, None].expand(heads, kg.shape[2])], 1)
            mask = mask.to(q.dtype).view(1, heads, 1, -1)
        o = F.scaled_dot_product_attention(q, torch.cat([k, kg], 2), torch.cat([v, vg], 2), attn_mask=mask)
        x = x + gm * self._out(blk.attn, o)
        x = x + gp * blk.mlp(pf_modulate(blk.norm2(x), sp, scp))
        # A per-head bias shared by all garment keys cancels in the garment-only softmax.
        flow = self._attention_flow(q, kg, flow_queries) if want_flow else None
        if not want_attn:
            return x, None, flow
        h = self.coral_heads
        logits = torch.matmul(q[:, :h].float(), kg[:, :h].float().transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        return x, logits.softmax(dim=-1), flow

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
        fine_r = fine_pos = None
        if self.fine_garment_from is not None:
            fine_r = self.fine_encoder(torch.cat([garment_latent, garment_mask], 1)).flatten(2).transpose(1, 2)
            fine_pos = self._fine_position_delta().to(x.dtype)
        kv: GarmentKV = []
        for idx in range(len(self.blocks)):
            if fine_r is not None and idx >= self.fine_garment_from:
                x, k, v = self._run(lambda x_, c_, r_, p_, i=idx: self._garment_block(i, x_, c_, r_, p_),
                                    x, c, fine_r, fine_pos)
            else:
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
        flow_blocks: Sequence[int] = (),
        flow_queries: Optional[torch.Tensor] = None,
    ):
        """
        x: (B,4,h,w) person latent state; t: (B,N) per-token time in [0,1];
        cond: (B,9,h,w) agnostic latent | latent mask | DensePose latent;
        edit_tokens: (B,N) bool, tokens inside the agnostic mask.
        flow_blocks: blocks whose attention flow (B,N,3) is returned last, as {block: flow};
        flow_queries: (B,M) long, compute it for these person tokens only, (B,M,3).
        """
        if garment_kv is None:
            garment_kv = self.encode_garment(garment_latent, garment_mask)
        b = x.shape[0]
        role = torch.where(edit_tokens, ROLE_EDIT, ROLE_KNOWN)
        h = self.x_embedder.proj(x).flatten(2).transpose(1, 2) + self._patch(self.cond_embedder, cond)
        h = h + self.pos_embed + self.role_token[role]
        c = self._time_cond(t) + self._null_class(b, x.device, h.dtype) + self.role_cond[role]

        attention: Dict[int, torch.Tensor] = {}
        flows: Dict[int, torch.Tensor] = {}
        flow_blocks = set(int(i) for i in flow_blocks)
        for idx in range(len(self.blocks)):
            want = return_attention and idx in self.coral_blocks
            wf = idx in flow_blocks
            kg, vg = garment_kv[idx]
            if self.fine_garment_from is not None and idx >= self.fine_garment_from:
                h, a, f = self._run(
                    lambda h_, c_, k_, v_, b_, i=idx, w=want, wf=wf:
                        self._person_block(i, h_, c_, k_, v_, w, b_, wf, flow_queries),
                    h, c, kg, vg, self.fine_bias[idx - self.fine_garment_from],
                )
            else:
                h, a, f = self._run(
                    lambda h_, c_, k_, v_, i=idx, w=want, wf=wf:
                        self._person_block(i, h_, c_, k_, v_, w, None, wf, flow_queries),
                    h, c, kg, vg,
                )
            if a is not None:
                attention[idx] = a
            if f is not None:
                flows[idx] = f
        out = self._unpatchify(self.final_layer(h, c))
        v, logvar = out[:, :-1], out[:, -1:]
        result = [v]
        if return_uncertainty:
            result.append(logvar)
        if return_attention:
            result.append(attention)
        if flow_blocks:
            result.append(flows)
        return result[0] if len(result) == 1 else tuple(result)
