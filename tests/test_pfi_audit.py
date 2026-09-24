"""Validate diagnostic invariants without loading a checkpoint or using a GPU."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.audit_pfi_timesteps import ObservedModel, as_float, contrast_mask, per_image_mean, plot_profiles, rgb_metrics, summarize
from scripts.compare_pfi_audits import bin_time_rows, paired_summary
from vton_ext.pfi_sample import generate
from vton_ext.utils import expand_patch_values


class ConstantVelocity:
    patch_size, latent_hw = 2, (8, 6)

    def encode_garment(self, garment, mask):
        return []

    def __call__(self, x, t, cond, edit, **kwargs):
        return torch.ones_like(x), torch.zeros_like(x[:, :1])


class AuditTests(unittest.TestCase):
    def test_observer_does_not_change_generation(self):
        model = ConstantVelocity()
        edit = torch.zeros(2, 12, dtype=torch.bool)
        edit[:, 3:9] = True
        inputs = {"known": torch.zeros(2, 4, 8, 6), "cond": torch.zeros(2, 9, 8, 6),
                  "garment": torch.ones(2, 4, 8, 6), "garment_mask": torch.ones(2, 1, 8, 6),
                  "edit_tokens": edit, "edit_pixels": expand_patch_values(edit.float(), 2, (8, 6))}
        noise = torch.randn(2, 4, 8, 6)
        observed = []

        def observer(call, x, t, v, logvar, kv):
            observed.append((call, t.clone(), x.clone()))

        direct = generate(model, inputs, sampler="euler", nfe=8, noise=noise)
        wrapped, stats = generate(ObservedModel(model, observer), inputs, sampler="euler", nfe=8, noise=noise, return_stats=True)
        torch.testing.assert_close(wrapped, direct, rtol=0, atol=0)
        self.assertEqual(stats["nfe"], 8)
        self.assertEqual(len(observed), 8)
        for call, t, x in observed:
            torch.testing.assert_close(t[edit], torch.full_like(t[edit], call / 8))
            torch.testing.assert_close(t[~edit], torch.ones_like(t[~edit]))
            endpoint = x + expand_patch_values(1 - t, 2, (8, 6))
            selected = inputs["edit_pixels"].bool().expand_as(x)
            torch.testing.assert_close(endpoint[selected], (noise + 1)[selected])

    def test_empty_regions_are_excluded_and_json_is_finite(self):
        err = torch.ones(2, 3, 16, 16)
        mask = torch.ones(2, 1, 16, 16)
        mask[1] = 0
        values = per_image_mean(err, mask)
        self.assertEqual(values[0], 1)
        self.assertTrue(torch.isnan(values[1]))
        rows = [{"sampler": "x", "index": i, "cloth_l1": as_float(v)} for i, v in enumerate(values)]
        result = summarize(rows, ["sampler"])
        self.assertEqual(result[0]["metrics"]["cloth_l1"]["count"], 1)
        self.assertEqual(result[0]["metrics"]["cloth_l1"]["mean"], 1)
        json.dumps(result, allow_nan=False)

    def test_identity_rgb_metrics_and_plain_garment_proxy(self):
        rgb = torch.rand(2, 3, 32, 32) * 2 - 1
        mask = torch.ones(2, 1, 32, 32)
        measures = rgb_metrics(rgb, rgb, {"cloth": mask}, include_ssim=True)
        torch.testing.assert_close(measures["cloth_l1"], torch.zeros(2))
        torch.testing.assert_close(measures["cloth_highpass_l1"], torch.zeros(2))
        torch.testing.assert_close(measures["cloth_ssim"], torch.ones(2))
        self.assertEqual(contrast_mask(torch.ones_like(rgb), mask).sum(), 0)

    def test_summary_plot_is_written(self):
        rows = [{"sampler": "euler8", "state": state, "time": t, "index": i,
                 "cloth_l1": .2 - .1 * t + .01 * i} for state in ["teacher", "rollout"] for t in [0., .5] for i in [0, 1]]
        summary = summarize(rows, ["sampler", "state", "time"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "figure.png"
            plot_profiles(summary, path, "Synthetic fixture")
            self.assertGreater(path.stat().st_size, 0)
            self.assertGreater(path.with_suffix(".pdf").stat().st_size, 0)

    def test_comparison_pairs_same_cases_and_ignores_empty_regions(self):
        baseline = [{"index": 1, "sampler": "euler8", "cloth_l1": .3, "contrast_l1": None},
                    {"index": 2, "sampler": "euler8", "cloth_l1": .2, "contrast_l1": .4}]
        candidate = [{"index": 2, "sampler": "euler8", "cloth_l1": .3, "contrast_l1": .2},
                     {"index": 1, "sampler": "euler8", "cloth_l1": .1, "contrast_l1": None}]
        rows = paired_summary(baseline, candidate, ("index", "sampler"), ("cloth_l1", "contrast_l1"))
        self.assertAlmostEqual(rows[0]["metrics"]["cloth_l1"]["delta"], -.05)
        self.assertEqual(rows[0]["metrics"]["cloth_l1"]["n"], 2)
        self.assertEqual(rows[0]["metrics"]["contrast_l1"]["n"], 1)
        with self.assertRaises(ValueError):
            paired_summary(baseline, candidate[:1], ("index", "sampler"), ("cloth_l1",))

    def test_time_ranges_average_each_case_before_pairing(self):
        rows = [{"index": 1, "sampler": "euler50", "state": "teacher", "time": t,
                 "teacher_velocity_mse": v} for t, v in [(.02, 1.), (.10, 3.), (.22, 5.)]]
        binned = bin_time_rows(rows, ("teacher_velocity_mse",))
        by_range = {row["range"]: row["teacher_velocity_mse"] for row in binned}
        self.assertEqual(by_range, {"0.0-0.2": 2., "0.2-0.4": 5.})


if __name__ == "__main__":
    unittest.main()
