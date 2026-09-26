"""PFI model/sampler invariants on a tiny transformer, no dataset needed."""
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from vton_ext.coral import CoralTargets, coral_routing_loss
from vton_ext.pfi_model import VTONInpaintDiT
from omegaconf import OmegaConf

from vton_ext.pfi_sample import generate, open_mask, shift_time
from vton_ext.pfi_train import EditTimeSampler, decoded_detail_loss, load_cfg, lr_factor, resolve_resume, validate_config
from vton_ext.utils import expand_patch_values


def tiny():
    torch.manual_seed(0)
    m = VTONInpaintDiT(latent_hw=(8, 6), hidden_size=64, depth=3, num_heads=4, coral_blocks=[1], coral_heads=2)
    for p in m.parameters():  # break adaLN-zero so every path is active
        torch.nn.init.normal_(p, std=0.05)
    return m


def inputs(b=2):
    edit = torch.zeros(b, 12, dtype=torch.bool)
    edit[:, 3:9] = True
    return {
        "known": torch.randn(b, 4, 8, 6),
        "cond": torch.randn(b, 9, 8, 6),
        "garment": torch.randn(b, 4, 8, 6),
        "garment_mask": torch.ones(b, 1, 8, 6),
        "edit_tokens": edit,
        "edit_pixels": expand_patch_values(edit.float(), 2, (8, 6)),
    }


class PFITests(unittest.TestCase):
    def test_coral_checkpointed_loss_matches_gradients(self):
        torch.manual_seed(12)
        coords = torch.rand(2, 12, 2)
        targets = CoralTargets(torch.randint(12, (2, 12)), coords,
                               torch.rand(2, 12) > 0.3, torch.rand(2, 12))
        logits = [torch.randn(2, 2, 12, 12, requires_grad=True) for _ in range(2)]
        maps = {i: x.softmax(-1) for i, x in enumerate(logits)}
        a, b, metrics = coral_routing_loss(maps, targets, (4, 3))
        grads = torch.autograd.grad(a + 0.1 * b, logits, retain_graph=True)
        ac, bc, mc = coral_routing_loss(maps, targets, (4, 3), checkpoint_loss=True)
        grads_c = torch.autograd.grad(ac + 0.1 * bc, logits)
        torch.testing.assert_close(a, ac)
        torch.testing.assert_close(b, bc)
        for k in metrics:
            torch.testing.assert_close(metrics[k], mc[k])
        for g, gc in zip(grads, grads_c):
            torch.testing.assert_close(g, gc)

    def test_v43_dimensions_and_missing_resume(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_cfg(str(root / "configs/vton_v43_pfi_1024.yaml"))
        validate_config(cfg)
        cfg.model.latent_hw = [64, 48]
        with self.assertRaises(ValueError):
            validate_config(cfg)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                resolve_resume("auto", tmp)
            self.assertIsNone(resolve_resume(None, tmp))
            (Path(tmp) / "latest.pt").touch()
            self.assertEqual(resolve_resume("auto", tmp), Path(tmp) / "latest.pt")

    def test_cached_garment_kv_matches_direct(self):
        m, x = tiny().eval(), inputs()
        t = torch.rand(2, 12)
        direct = m(x["known"], t, x["cond"], x["edit_tokens"], x["garment"], x["garment_mask"])
        kv = m.encode_garment(x["garment"], x["garment_mask"])
        cached = m(x["known"], t, x["cond"], x["edit_tokens"], garment_kv=kv)
        torch.testing.assert_close(direct, cached)

    def test_checkpointing_matches_and_attention_is_distribution(self):
        m, x = tiny().train(), inputs()
        t = torch.rand(2, 12)
        args = (x["known"], t, x["cond"], x["edit_tokens"], x["garment"], x["garment_mask"])
        v1, lv1, a1 = m(*args, return_uncertainty=True, return_attention=True)
        g1 = torch.autograd.grad(v1.sum() + a1[1].square().sum(), m.blocks[0].attn.qkv.weight)[0]
        m.gradient_checkpointing = True
        v2, lv2, a2 = m(*args, return_uncertainty=True, return_attention=True)
        g2 = torch.autograd.grad(v2.sum() + a2[1].square().sum(), m.blocks[0].attn.qkv.weight)[0]
        torch.testing.assert_close(v1, v2)
        torch.testing.assert_close(g1, g2)
        self.assertEqual(tuple(a1[1].shape), (2, 2, 12, 12))
        torch.testing.assert_close(a1[1].sum(-1), torch.ones(2, 2, 12))

    def test_garment_changes_output(self):
        m, x = tiny().eval(), inputs()
        t = torch.zeros(2, 12)
        a = m(x["known"], t, x["cond"], x["edit_tokens"], x["garment"], x["garment_mask"])
        b = m(x["known"], t, x["cond"], x["edit_tokens"], -x["garment"], x["garment_mask"])
        self.assertGreater(float((a - b).abs().mean()), 1e-4)

    def test_samplers_budget_and_known_context(self):
        m, x = tiny().eval(), inputs()
        noise = torch.randn(2, 4, 8, 6)
        for name in ("euler", "dual_loop", "look_ahead"):
            out, stats = generate(m, x, nfe=8, sampler=name, noise=noise, return_stats=True)
            self.assertEqual(stats["nfe"], 8, name)
            keep = ~x["edit_pixels"].bool().expand_as(out)
            torch.testing.assert_close(out[keep], x["known"][keep])
            self.assertTrue(torch.isfinite(out).all())

    def test_cfg_interval_limits_guidance(self):
        m, x = tiny().eval(), inputs()
        noise = torch.randn(2, 4, 8, 6)
        plain, stats = generate(m, x, nfe=4, sampler="euler", noise=noise, return_stats=True)
        # Guidance restricted to an empty time interval must equal no guidance.
        off = generate(m, x, nfe=4, sampler="euler", noise=noise, cfg_scale=3.0, cfg_interval=(2.0, 3.0))
        torch.testing.assert_close(off, plain)
        self.assertEqual(stats["denoiser_calls"], 4)
        on, on_stats = generate(m, x, nfe=4, sampler="euler", noise=noise, cfg_scale=3.0, return_stats=True)
        self.assertEqual((on_stats["nfe"], on_stats["denoiser_calls"]), (4, 8))
        self.assertGreater(float((on - plain).abs().max()), 1e-4)
        keep = ~x["edit_pixels"].bool().expand_as(on)
        torch.testing.assert_close(on[keep], x["known"][keep])

    def test_time_shift_maps_endpoints_and_moves_toward_noise(self):
        t = torch.linspace(0, 1, 11)
        torch.testing.assert_close(shift_time(t, 1.0), t)
        s = shift_time(t, 2.0)
        self.assertEqual(float(s[0]), 0.0)
        self.assertEqual(float(s[-1]), 1.0)
        self.assertTrue(bool((s[1:-1] < t[1:-1]).all()))
        self.assertTrue(bool((s[1:] > s[:-1]).all()))

    def test_sampler_uses_model_time_shift_and_keeps_budget(self):
        m, x = tiny().eval(), inputs()
        noise = torch.randn(2, 4, 8, 6)
        base = generate(m, x, nfe=4, sampler="dual_loop", noise=noise)
        m.time_shift = 2.0
        shifted, stats = generate(m, x, nfe=4, sampler="dual_loop", noise=noise, return_stats=True)
        self.assertEqual(stats["nfe"], 4)
        self.assertGreater(float((shifted - base).abs().max()), 1e-5)
        torch.testing.assert_close(generate(m, x, nfe=4, sampler="dual_loop", noise=noise, time_shift=1.0), base)

    def test_weights_transfer_across_resolution(self):
        import tempfile
        small = tiny()
        big = VTONInpaintDiT(latent_hw=(16, 12), hidden_size=64, depth=3, num_heads=4, coral_blocks=[1], coral_heads=2)
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save({"model": small.state_dict(), "step": 7}, f.name)
            report = big.load_weights_any_resolution(f.name)
        self.assertEqual(report["source_step"], 7)
        torch.testing.assert_close(big.blocks[1].attn.qkv.weight, small.blocks[1].attn.qkv.weight)
        self.assertEqual(tuple(big.pos_embed.shape), (1, 48, 64))
        t = torch.rand(1, 48)
        edit = torch.zeros(1, 48, dtype=torch.bool); edit[:, 10:30] = True
        v = big(torch.randn(1, 4, 16, 12), t, torch.randn(1, 9, 16, 12), edit,
                torch.randn(1, 4, 16, 12), torch.ones(1, 1, 16, 12))
        self.assertEqual(tuple(v.shape), (1, 4, 16, 12))

    def test_decoded_crop_windows_stay_inside_and_cover_clothing(self):
        from vton_ext.pfi_train import _decode_windows
        torch.manual_seed(0)
        z = torch.zeros(2, 4, 128, 96)
        cloth = torch.zeros(2, 1, 1024, 768)
        cloth[0, :, 300:700, 200:600] = 1
        cloth[1, :, 900:1024, 0:100] = 1           # clothing at a corner
        for (iy0, iy1, ix0, ix1), (oy0, oy1, ox0, ox1) in _decode_windows(z, cloth, (64, 48), 4):
            self.assertEqual((iy1 - iy0, ix1 - ix0), (64, 48))
            self.assertTrue(0 <= oy0 <= iy0 and iy1 <= oy1 <= 128 and 0 <= ox0 <= ix0 and ix1 <= ox1 <= 96)

    def test_detail_branch_can_skip_resolution_shift(self):
        base = {"ltg_std": 0.6, "ltg_loc": 0.0, "ltg_scale": 1.0, "time_shift": 2.0, "curriculum": {
            "start": {"pure_noise": 0.0, "synchronous": 0.0, "detail": 1.0},
            "end": {"pure_noise": 0.0, "synchronous": 0.0, "detail": 1.0},
            "detail_range": [0.75, 0.98], "detail_std": 0.05, "gate_metric": "m", "gate_threshold": 1.0,
            "gate_min_steps": 0, "gate_max_steps": 10, "ramp_steps": 1}}
        torch.manual_seed(0)
        shifted = EditTimeSampler(OmegaConf.create({"flow": base}))(4096, 16, "cpu")
        base["curriculum"]["detail_unshifted"] = True
        torch.manual_seed(0)
        raw = EditTimeSampler(OmegaConf.create({"flow": base}))(4096, 16, "cpu")
        self.assertGreater(float(raw.min()), 0.5)          # stays near clean
        torch.testing.assert_close(shifted, shift_time(raw, 2.0))

    def test_prune_snapshots_keeps_newest_and_latest(self):
        import tempfile
        from vton_ext.pfi_train import prune_snapshots
        with tempfile.TemporaryDirectory() as tmp:
            for n in (1000, 2000, 3000, 4000):
                (Path(tmp) / f"step{n:07d}.pt").touch()
            (Path(tmp) / "latest.pt").touch()
            gone = prune_snapshots(Path(tmp), 2)
            self.assertEqual([p.name for p in gone], ["step0001000.pt", "step0002000.pt"])
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),
                             ["latest.pt", "step0003000.pt", "step0004000.pt"])
            self.assertEqual(prune_snapshots(Path(tmp), 0), [])

    def test_bf16_snapshot_loads_into_fp32_model(self):
        m = tiny()
        state = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in m.state_dict().items()}
        m2 = tiny()
        m2.load_state_dict(state)
        self.assertEqual(m2.blocks[0].attn.qkv.weight.dtype, torch.float32)
        torch.testing.assert_close(m2.blocks[0].attn.qkv.weight, m.blocks[0].attn.qkv.weight, atol=1e-2, rtol=1e-2)

    def test_coral_loss_prefers_target(self):
        target = torch.tensor([[5, 7]])
        coords = torch.stack(torch.meshgrid(torch.linspace(0, 1, 4), torch.linspace(0, 1, 3), indexing="ij"), -1)
        coords = coords.reshape(12, 2)
        tg = CoralTargets(target, coords[target], torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 2))
        good = torch.full((1, 1, 2, 12), 1e-3)
        good[0, 0, 0, 5] = good[0, 0, 1, 7] = 1.0
        bad = torch.full((1, 1, 2, 12), 1 / 12)
        lg, _, _ = coral_routing_loss({0: good}, tg, (4, 3))
        lb, _, _ = coral_routing_loss({0: bad}, tg, (4, 3))
        self.assertLess(float(lg), float(lb))

    def test_curriculum_gate_and_mixture(self):
        cfg = OmegaConf.create({"flow": {"ltg_std": 0.6, "ltg_loc": 0.0, "ltg_scale": 1.0, "curriculum": {
            "start": {"pure_noise": 0.5, "synchronous": 0.2, "detail": 0.0},
            "end": {"pure_noise": 0.1, "synchronous": 0.15, "detail": 0.4},
            "detail_range": [0.7, 0.98], "gate_metric": "m", "gate_threshold": 0.3,
            "gate_min_steps": 10, "gate_max_steps": 100, "ramp_steps": 20}}})
        s = EditTimeSampler(cfg)
        torch.manual_seed(0)
        t = s(20000, 16, "cpu", step=0)
        self.assertAlmostEqual(float((t == 0).all(1).float().mean()), 0.5, delta=0.02)
        s.observe(5, {"m": 0.9})           # good, but before min steps
        self.assertIsNone(s.ramp_start)
        s.metric_ema = 0.9
        s.observe(12, {"m": 0.9})
        self.assertEqual(s.ramp_start, 12)
        t = s(20000, 16, "cpu", step=12 + 20)
        self.assertAlmostEqual(float((t == 0).all(1).float().mean()), 0.1, delta=0.02)
        late = (t.max(1).values >= 0.7).float().mean()
        self.assertGreater(float(late), 0.4)
        s2 = EditTimeSampler(cfg)
        s2.observe(100, {"m": 0.0})        # max steps forces the gate
        self.assertEqual(s2.ramp_start, 100)

    def test_native_sincos_table_matches_pretrained_layout(self):
        import numpy as np
        from patch_flow.models.dit import get_2d_sincos_pos_embed
        from vton_ext.utils import sincos_pos_embed
        ref = torch.from_numpy(get_2d_sincos_pos_embed(64, 16)).float()[None]
        torch.testing.assert_close(sincos_pos_embed(64, (16, 16)), ref)
        rect = sincos_pos_embed(64, (8, 6))
        self.assertEqual(tuple(rect.shape), (1, 48, 64))
        # Row-major token order: token (r, c) of the rectangle equals token (r, c) of the square table.
        torch.testing.assert_close(rect[0].view(8, 6, 64), ref[0].view(16, 16, 64)[:8, :6])
        with self.assertRaises(ValueError):
            VTONInpaintDiT(latent_hw=(8, 6), hidden_size=64, depth=1, num_heads=4, pos_embed="rope")
        m = VTONInpaintDiT(latent_hw=(8, 6), hidden_size=64, depth=1, num_heads=4, pos_embed_trainable=True)
        self.assertIn("pos_embed", [n for n, p in m.pretrained_parameters() if p.requires_grad])
        self.assertFalse(tiny().pos_embed.requires_grad)

    def test_seam_correction_keeps_known_pixels_and_removes_border_offset(self):
        from vton_ext.pfi_sample import composite
        mask = torch.zeros(1, 1, 64, 64)
        mask[..., 16:48, 16:48] = 1
        truth = torch.full((1, 3, 64, 64), 0.8)
        agnostic = truth * (1 - mask)
        # One decode is continuous across the border; pasting the observed
        # pixels back exposes its brightness shift there as a step.
        decoded = truth + 0.06
        plain = composite(decoded, agnostic, mask)
        fixed = composite(decoded, agnostic, mask, seam_px=4.0)
        known = (1 - mask).bool().expand_as(truth)
        torch.testing.assert_close(fixed[known], agnostic[known])
        edge = (plain - truth).abs()[..., 16, 20:44].mean(), (fixed - truth).abs()[..., 16, 20:44].mean()
        self.assertLess(float(edge[1]), 0.5 * float(edge[0]))
        centre = (fixed - plain)[..., 32, 32].abs().max()
        self.assertLess(float(centre), 0.01)  # deep inside the mask stays as generated
        feathered = composite(decoded, agnostic, mask, seam_px=4.0, feather_px=2.0)
        torch.testing.assert_close(feathered[..., 0:4, 0:4], agnostic[..., 0:4, 0:4])

    def test_coarse_to_fine_location_moves_times_toward_clean(self):
        cfg = OmegaConf.create({"flow": {
            "ltg_std": 0.6, "ltg_loc": 0.0, "ltg_scale": 1.0,
            "curriculum": {"start": {"pure_noise": 0.3, "synchronous": 0.3, "detail": 0.0},
                           "end": {"pure_noise": 0.1, "synchronous": 0.15, "detail": 0.4},
                           "detail_range": [0.75, 0.98], "gate_metric": "m", "gate_threshold": 2.0,
                           "gate_min_steps": 0, "gate_max_steps": 0, "ramp_steps": 100,
                           "ltg_loc_start": -1.0, "ltg_loc_end": 0.8}}})
        ts = EditTimeSampler(cfg)
        ts.observe(0, {})
        self.assertEqual(ts.ltg_loc(0), -1.0)
        self.assertAlmostEqual(ts.ltg_loc(50), -0.1)
        self.assertEqual(ts.ltg_loc(500), 0.8)
        torch.manual_seed(0)
        early = ts(4096, 16, "cpu", step=0)
        late = ts(4096, 16, "cpu", step=100)
        self.assertLess(float(early.mean()), 0.3)
        self.assertGreater(float(late.mean()), float(early.mean()) + 0.25)
        self.assertLess(float((early > 0.8).float().mean()), 0.05)

    def test_open_mask_per_sample_radius(self):
        m = torch.zeros(2, 1, 21, 21)
        m[:, :, 10, 10] = 1
        out = open_mask(m, torch.tensor([0, 3]))
        self.assertEqual(int(out[0].sum()), 1)
        self.assertEqual(int(out[1].sum()), 49)


class DetailTimeSamplerTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.cfg = load_cfg(str(self.root / "configs/vton_v40_pfi_coral.yaml"))
        self.cfg.flow.curriculum.start = self.cfg.flow.curriculum.end

    def draw(self, cfg, b=4096, n=32, seed=12345):
        torch.manual_seed(seed)
        t = EditTimeSampler(cfg)(b, n, "cpu")
        return t, torch.get_rng_state()

    def test_explicit_legacy_width_matches_original_sampler_and_rng(self):
        legacy, legacy_rng = self.draw(self.cfg)
        self.cfg.flow.curriculum.detail_std = self.cfg.flow.ltg_std
        explicit, explicit_rng = self.draw(self.cfg)
        torch.testing.assert_close(explicit, legacy, rtol=0, atol=0)
        self.assertTrue(torch.equal(explicit_rng, legacy_rng))

    def test_detail_branch_has_late_patch_coverage(self):
        c = self.cfg.flow.curriculum
        c.start = c.end = {"pure_noise": 0.0, "synchronous": 0.0, "detail": 1.0}
        c.detail_range = [0.75, 0.98]
        c.detail_std = 0.05
        t, _ = self.draw(self.cfg)
        self.assertTrue(torch.isfinite(t).all())
        self.assertTrue(((t >= 0) & (t <= 0.98)).all())
        self.assertAlmostEqual(t.mean().item(), 0.825, delta=0.01)
        # Per-patch coverage, rather than a maximum over each example's tokens.
        self.assertAlmostEqual((t > 0.8).float().mean().item(), 0.609, delta=0.025)
        self.assertAlmostEqual((t > 0.9).float().mean().item(), 0.185, delta=0.025)
        self.assertTrue((t.std(dim=1) > 0).all())

    def test_other_branches_and_random_stream_are_unchanged(self):
        b, seed = 4096, 12345
        torch.manual_seed(seed)
        u = torch.rand(b)
        c = self.cfg.flow.curriculum
        p_noise, p_sync, p_detail = [float(c.start[k]) for k in EditTimeSampler.KINDS]
        detail = (u >= p_noise + p_sync) & (u < p_noise + p_sync + p_detail)
        legacy, legacy_rng = self.draw(self.cfg, b=b, seed=seed)
        c.detail_range, c.detail_std = [0.75, 0.98], 0.05
        narrow, narrow_rng = self.draw(self.cfg, b=b, seed=seed)
        torch.testing.assert_close(narrow[~detail], legacy[~detail], rtol=0, atol=0)
        self.assertTrue(torch.equal(narrow_rng, legacy_rng))
        self.assertGreater((narrow[detail] - legacy[detail]).mean().item(), 0.2)
        self.assertAlmostEqual((narrow == 0).all(1).float().mean().item(), p_noise, delta=0.02)

    def test_resumed_curriculum_retains_progress_and_new_detail_policy(self):
        cfg = load_cfg(str(self.root / "configs/vton_v41_pfi_detail.yaml"))
        sampler = EditTimeSampler(cfg)
        sampler.load_state_dict({"metric_ema": 0.5190847566, "ramp_start": 1500})
        self.assertEqual(sampler.progress(5000), 1.0)
        torch.testing.assert_close(sampler.probabilities(5000), torch.tensor([0.1, 0.15, 0.4]))
        torch.manual_seed(12345)
        t = sampler(4096, 32, "cpu", step=5000)
        self.assertAlmostEqual((t > 0.8).float().mean().item(), 0.259, delta=0.025)
        self.assertAlmostEqual((t > 0.9).float().mean().item(), 0.076, delta=0.015)

    def test_pilot_preserves_lr_horizon_and_other_model_settings(self):
        old = load_cfg(str(self.root / "configs/vton_v40_pfi_coral.yaml"))
        new = load_cfg(str(self.root / "configs/vton_v41_pfi_detail.yaml"))
        self.assertEqual(new.train.stop_at_step, 6000)
        self.assertEqual(new.train.max_steps, 20000)
        for step in (5000, 5500, 6000):
            self.assertEqual(lr_factor(step, old), lr_factor(step, new))
        new.flow.curriculum.detail_range = old.flow.curriculum.detail_range
        new.flow.curriculum.pop("detail_std")
        new.train.pop("stop_at_step")
        new.train.output_dir = old.train.output_dir
        new.train.keep_every = old.train.keep_every
        self.assertEqual(OmegaConf.to_container(old), OmegaConf.to_container(new))

    def test_invalid_detail_parameters_are_rejected(self):
        for std in (0.0, -0.1, float("inf"), float("nan")):
            with self.subTest(std=std):
                self.cfg.flow.curriculum.detail_std = std
                with self.assertRaises(ValueError):
                    EditTimeSampler(self.cfg)
        self.cfg.flow.curriculum.pop("detail_std")
        for bounds in ([], [0.98, 0.75], [0.75, 1.1]):
            with self.subTest(bounds=bounds):
                self.cfg.flow.curriculum.detail_range = bounds
                with self.assertRaises(ValueError):
                    EditTimeSampler(self.cfg)


class DecodedDetailTests(unittest.TestCase):
    def test_decoded_loss_reaches_only_selected_latents(self):
        class TinyVAE(torch.nn.Module):
            scale, shift = 1.0, 0.0

            def __init__(self):
                super().__init__()
                self.post_quant_conv = torch.nn.Identity()
                self.decoder = torch.nn.Sequential(torch.nn.Conv2d(4, 3, 1),
                                                   torch.nn.Upsample(scale_factor=2, mode="bilinear"))

        torch.manual_seed(4)
        vae = TinyVAE().requires_grad_(False)
        endpoint = torch.randn(2, 4, 8, 6, requires_grad=True)
        target = vae.decoder(endpoint.detach()).clone()
        target[:, :, 4:10, 4:8] += .2
        mask = torch.ones(2, 1, 16, 12)
        cfg = OmegaConf.create({"loss": {"decoded_rgb_weight": .5, "decoded_highpass_weight": 1.0,
                                         "decoded_graphic_boost": 2.0}})
        loss, measures = decoded_detail_loss(vae, endpoint, target, mask, mask, cfg,
                                             torch.tensor([True, False]))
        self.assertGreater(float(loss.detach()), 0)
        self.assertGreater(float(measures["loss_decoded_highpass"].detach()), 0)
        loss.backward()
        self.assertGreater(float(endpoint.grad[0].abs().sum()), 0)
        self.assertEqual(float(endpoint.grad[1].abs().sum()), 0)
        reference_grad = endpoint.grad.clone()
        endpoint.grad = None
        cfg.loss.decoded_checkpoint = True
        checked, _ = decoded_detail_loss(vae, endpoint, target, mask, mask, cfg,
                                        torch.tensor([True, False]))
        checked.backward()
        torch.testing.assert_close(checked, loss)
        torch.testing.assert_close(endpoint.grad, reference_grad)
        skipped, _ = decoded_detail_loss(vae, endpoint, target, mask, mask, cfg,
                                         torch.tensor([False, False]))
        self.assertEqual(float(skipped.detach()), 0)


if __name__ == "__main__":
    unittest.main()
