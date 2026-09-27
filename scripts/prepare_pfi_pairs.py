"""Build paired fit/dev lists from VITON-HD identities, never raw unpaired rows."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from vton_ext.pairs import paired_split, split_manifest


def prepare(root: Path, seed=2026, dev_size=256, replace=False, method="stable-hash"):
    fit, dev = paired_split(root, seed, dev_size, method)
    outputs = {root / "train_fit_pairs.txt": fit, root / "train_dev_pairs.txt": dev}
    contents = {p: "".join(f"{person} {garment}\n" for person, garment in rows)
                for p, rows in outputs.items()}
    changed = [p for p, text in contents.items() if p.exists() and p.read_text() != text]
    if changed and not replace:
        raise FileExistsError(
            f"Existing split differs: {changed}. Audit it first with --check-only; "
            "--replace backs up both lists before an intentional repair."
        )
    if changed:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        for p in outputs:
            if p.exists():
                shutil.copy2(p, p.with_name(p.name + f".backup-{stamp}"))
    for p, text in contents.items():
        if not p.exists() or p.read_text() != text:
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_text(text)
            tmp.replace(p)
    return split_manifest(*outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--method", choices=("stable-hash", "random"), default="stable-hash")
    parser.add_argument("--dev-size", type=int, default=256)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = (split_manifest(args.root / "train_fit_pairs.txt", args.root / "train_dev_pairs.txt")
              if args.check_only else prepare(args.root, args.seed, args.dev_size, args.replace, args.method))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
