"""Prevent unpaired archives and changed splits from corrupting supervision."""
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_pfi_pairs import prepare
from vton_ext.pairs import read_pairs, split_manifest, validate_resume_pairs
from scripts.make_vton_dev_split import stable_key


class PairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for folder in ("image", "cloth"):
            d = self.root / "train" / folder
            d.mkdir(parents=True)
            for name in ("004_00.jpg", "001_00.jpg", "003_00.jpg", "002_00.jpg"):
                (d / name).touch()

    def tearDown(self):
        self.tmp.cleanup()

    def test_unpaired_archive_does_not_determine_supervision(self):
        raw = self.root / "train_pairs.txt"
        text = "004_00.jpg 001_00.jpg\n001_00.jpg 002_00.jpg\n"
        raw.write_text(text)
        with self.assertRaisesRegex(ValueError, "Unpaired garment"):
            read_pairs(raw)
        self.assertEqual(len(read_pairs(raw, paired=False)), 2)
        saved = prepare(self.root, dev_size=1)
        self.assertEqual(raw.read_text(), text)
        self.assertEqual((saved["fit"]["count"], saved["dev"]["count"]), (3, 1))
        self.assertEqual(prepare(self.root, dev_size=1), saved)

    def test_default_split_matches_historical_hash_recipe(self):
        prepare(self.root, dev_size=1)
        rows = [f"{i:03d}_00.jpg {i:03d}_00.jpg" for i in range(1, 5)]
        expected = min(rows, key=lambda row: stable_key(row, 2026))
        self.assertEqual((self.root / "train_dev_pairs.txt").read_text(), expected + "\n")

    def test_changed_existing_split_requires_explicit_repair_and_backup(self):
        fit = self.root / "train_fit_pairs.txt"
        fit.write_text("001_00.jpg 002_00.jpg\n")
        with self.assertRaises(FileExistsError):
            prepare(self.root, dev_size=1)
        self.assertEqual(fit.read_text(), "001_00.jpg 002_00.jpg\n")
        prepare(self.root, dev_size=1, replace=True)
        self.assertEqual(next(self.root.glob("*.backup-*")).read_text(), "001_00.jpg 002_00.jpg\n")

    def test_resume_detects_changed_order_but_accepts_line_endings(self):
        saved = prepare(self.root, dev_size=1)
        fit = self.root / "train_fit_pairs.txt"
        fit.write_bytes(fit.read_bytes().replace(b"\n", b"\r\n"))
        current = split_manifest(fit, self.root / "train_dev_pairs.txt")
        self.assertTrue(validate_resume_pairs(saved, current))
        fit.write_text("\n".join(reversed(fit.read_text().splitlines())) + "\n")
        with self.assertRaisesRegex(ValueError, "fit pairs differ"):
            validate_resume_pairs(saved, split_manifest(fit, self.root / "train_dev_pairs.txt"))
        self.assertFalse(validate_resume_pairs(None, current))

    def test_overlap_and_missing_matching_garment_are_rejected(self):
        prepare(self.root, dev_size=1)
        fit, dev = self.root / "train_fit_pairs.txt", self.root / "train_dev_pairs.txt"
        dev.write_text(fit.read_text().splitlines()[0] + "\n")
        with self.assertRaisesRegex(ValueError, "overlap"):
            split_manifest(fit, dev)
        (self.root / "train/cloth/004_00.jpg").unlink()
        with self.assertRaisesRegex(ValueError, "Missing matching garments"):
            prepare(self.root, dev_size=1)


if __name__ == "__main__":
    unittest.main()
