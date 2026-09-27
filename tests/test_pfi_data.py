"""Check that dataset and trainer reject contradictory paired supervision early."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from vton_ext.data import VitonHDDataset
from vton_ext.pfi_train import main


class DatasetPairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for folder in ("image", "cloth", "cloth-mask", "agnostic-v3.2", "agnostic-mask",
                       "image-densepose", "image-parse-v3"):
            (self.root / "train" / folder).mkdir(parents=True)
        (self.root / "train_pairs.txt").write_text("001_00.jpg 002_00.jpg\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_paired_dataset_rejects_raw_unpaired_rows(self):
        with self.assertRaisesRegex(ValueError, "Unpaired garment"):
            VitonHDDataset(str(self.root), order="paired")
        self.assertEqual(len(VitonHDDataset(str(self.root), order="unpaired")), 1)

    def test_paired_default_prefers_explicit_paired_file(self):
        paired = self.root / "train_pairs_paired.txt"
        paired.write_text("001_00.jpg 001_00.png\n")
        dataset = VitonHDDataset(str(self.root), order="paired")
        self.assertEqual(dataset.pair_path, paired)
        self.assertEqual(dataset.rows, [("001_00.jpg", "001_00.png")])

    def test_trainer_fails_before_model_allocation_and_run_output(self):
        out = self.root / "should-not-be-created"
        config = Path(__file__).resolve().parents[1] / "configs/vton_v46_pfi_1024_c2f.yaml"
        with patch("vton_ext.pfi_train.build_model") as build:
            with self.assertRaisesRegex(ValueError, "Unpaired garment"):
                main(["--config", str(config), f"data.root={self.root}",
                      f"data.pairs_file={self.root / 'train_pairs.txt'}", f"train.output_dir={out}"])
            build.assert_not_called()
        self.assertFalse(out.exists())

    def test_recovery_rejects_unverified_legacy_checkpoint_before_model_allocation(self):
        fit, dev = self.root / "fit.txt", self.root / "dev.txt"
        fit.write_text("001_00.jpg 001_00.jpg\n")
        dev.write_text("002_00.jpg 002_00.jpg\n")
        checkpoint = self.root / "legacy.pt"
        torch.save({"step": 3000, "optimizer": {}}, checkpoint)
        config = Path(__file__).resolve().parents[1] / "configs/vton_v46_pfi_1024_c2f.yaml"
        out = self.root / "should-not-be-created"
        with patch("vton_ext.pfi_train.build_model") as build:
            with self.assertRaisesRegex(ValueError, "requires checkpoint pair fingerprints"):
                main(["--config", str(config), "--resume", str(checkpoint),
                      f"data.root={self.root}", f"data.pairs_file={fit}", f"data.test_pairs_file={dev}",
                      "eval.indices=[0]", "train.require_pair_fingerprints=true", f"train.output_dir={out}"])
            build.assert_not_called()
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
