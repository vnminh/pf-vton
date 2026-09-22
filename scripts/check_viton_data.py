#!/usr/bin/env python3
import argparse
from vton_ext.data import VitonHDDataset

p = argparse.ArgumentParser()
p.add_argument("root")
p.add_argument("--phase", default="train", choices=["train", "test"])
p.add_argument("--pairs-file", default=None)
a = p.parse_args()

ds = VitonHDDataset(a.root, phase=a.phase, pairs_file=a.pairs_file)
print("pairs:", len(ds))
print("pair file:", ds.pair_path)
s = ds[0]
for k, v in s.items():
    print(k, tuple(v.shape) if hasattr(v, "shape") else v)
