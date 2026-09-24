"""Compare two matched PFI dev audits, including paired confidence intervals."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def paired_stats(first, second):
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    delta = b[valid] - a[valid]
    if not len(delta):
        return None
    return {"n": int(len(delta)), "baseline": float(a[valid].mean()),
            "candidate": float(b[valid].mean()), "delta": float(delta.mean()),
            "sem": float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else 0.0,
            "median_delta": float(np.median(delta)),
            "fraction_improved": float(np.mean(delta < 0))}


def join_rows(rows, keys):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in result:
            raise ValueError(f"Duplicate audit row {key}")
        result[key] = row
    return result


def paired_summary(left, right, keys, metrics):
    a, b = join_rows(left, keys), join_rows(right, keys)
    if a.keys() != b.keys():
        raise ValueError(f"Audit cases differ: {len(a.keys()-b.keys())} baseline-only, {len(b.keys()-a.keys())} candidate-only")
    groups = defaultdict(list)
    for key in a:
        groups[key[1:]].append((a[key], b[key]))
    result = []
    for group, pairs in sorted(groups.items()):
        row = dict(zip(keys[1:], group))
        row["metrics"] = {}
        for metric in metrics:
            valid_pairs = [(x[metric], y[metric]) for x, y in pairs
                           if x.get(metric) is not None and y.get(metric) is not None]
            stats = paired_stats([x for x, _ in valid_pairs], [y for _, y in valid_pairs])
            if stats:
                row["metrics"][metric] = stats
        result.append(row)
    return result


def bin_time_rows(rows, metrics):
    """Average each case within fixed time ranges before comparing cases."""
    groups = defaultdict(list)
    for row in rows:
        left = min(int(float(row["time"]) * 5), 4) / 5
        label = f"{left:.1f}-{left + .2:.1f}"
        groups[(row["index"], row["sampler"], row["state"], label)].append(row)
    result = []
    for (index, sampler, state, label), parts in groups.items():
        record = {"index": index, "sampler": sampler, "state": state, "range": label}
        for metric in metrics:
            values = [p[metric] for p in parts if p.get(metric) is not None and np.isfinite(p[metric])]
            record[metric] = float(np.mean(values)) if values else None
        result.append(record)
    return result


def plot_comparison(final, time, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    samplers = [r for r in final if r["sampler"] != "vae_reconstruction"]
    labels = [r["sampler"] for r in samplers]
    for j, metric in enumerate(("cloth_l1", "contrast_highpass_l1")):
        vals = [r["metrics"].get(metric, {}).get("delta", np.nan) for r in samplers]
        errs = [1.96 * r["metrics"].get(metric, {}).get("sem", 0) for r in samplers]
        ax.errorbar(np.arange(len(labels)) + (j - .5) * .12, vals, yerr=errs, fmt="o", capsize=3, label=metric)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xticks(range(len(labels)), labels, rotation=30)
    ax.set(ylabel="Candidate minus baseline (lower is better)", title="Final quality; paired 95% normal intervals")
    ax.legend(fontsize=8)

    for ax, state, metric, title in [
        (axes[0, 1], "teacher", "teacher_velocity_mse", "Teacher velocity MSE by time"),
        (axes[1, 0], "rollout", "cloth_highpass_l1", "Rollout decoded detail error by time"),
        (axes[1, 1], "teacher", "contrast_highpass_l1", "Teacher contrast detail error by time"),
    ]:
        for sampler in ("euler8", "euler50"):
            rows = sorted((r for r in time if r["state"] == state and r["sampler"] == sampler and metric in r["metrics"]), key=lambda r: r["time"])
            if not rows:
                continue
            t = [r["time"] for r in rows]
            a = [r["metrics"][metric]["baseline"] for r in rows]
            b = [r["metrics"][metric]["candidate"] for r in rows]
            ax.plot(t, a, "--", label=f"baseline {sampler}")
            ax.plot(t, b, "-", label=f"candidate {sampler}")
        ax.set(title=title, xlabel="Time (0 = noise; 1 = clean)", xlim=(0, 1))
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.savefig(output, dpi=160)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_time_ranges(ranges, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    specs = (("teacher", "teacher_velocity_mse", "Teacher velocity MSE"),
             ("rollout", "cloth_highpass_l1", "Rollout decoded detail error"))
    for ax, (state, metric, title) in zip(axes, specs):
        rows = sorted((r for r in ranges if r["sampler"] == "euler50" and r["state"] == state
                       and metric in r["metrics"]), key=lambda r: r["range"])
        if not rows:
            continue
        x = np.arange(len(rows))
        baseline = [r["metrics"][metric]["baseline"] for r in rows]
        candidate = [r["metrics"][metric]["candidate"] for r in rows]
        ax.bar(x - .18, baseline, .35, label="baseline")
        ax.bar(x + .18, candidate, .35, label="candidate")
        ax.set_xticks(x, [r["range"] for r in rows])
        ax.set(xlabel="Time range", ylabel=metric, title=title)
        ax.legend(fontsize=8)
    fig.suptitle("Euler 50; each case contributes one mean per time range")
    fig.savefig(output, dpi=160)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if not (args.baseline / "complete.json").exists() or not (args.candidate / "complete.json").exists():
        raise ValueError("Both input audits must have complete.json")
    args.output.mkdir(parents=True, exist_ok=True)
    final_metrics = ("edit_l1", "cloth_l1", "contrast_l1", "known_l1", "edit_highpass_l1", "cloth_highpass_l1",
                     "contrast_highpass_l1", "image_ssim", "cloth_ssim")
    time_metrics = ("teacher_velocity_mse", "latent_cloth_endpoint_l1", "cloth_l1", "cloth_highpass_l1",
                    "contrast_l1", "contrast_highpass_l1", "uncertainty_spearman")
    final = paired_summary(read_jsonl(args.baseline / "final_metrics.jsonl"),
                           read_jsonl(args.candidate / "final_metrics.jsonl"), ("index", "sampler"), final_metrics)
    time = paired_summary(read_jsonl(args.baseline / "time_profiles.jsonl"),
                          read_jsonl(args.candidate / "time_profiles.jsonl"),
                          ("index", "sampler", "state", "time"), time_metrics)
    baseline_bins = bin_time_rows(read_jsonl(args.baseline / "time_profiles.jsonl"), time_metrics)
    candidate_bins = bin_time_rows(read_jsonl(args.candidate / "time_profiles.jsonl"), time_metrics)
    time_ranges = paired_summary(baseline_bins, candidate_bins,
                                 ("index", "sampler", "state", "range"), time_metrics)
    summary = {"baseline": str(args.baseline), "candidate": str(args.candidate),
               "direction": "candidate minus baseline; lower is better except SSIM and uncertainty correlation",
               "final": final, "time": time, "time_ranges": time_ranges}
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot_comparison(final, time, args.output / "comparison.png")
    plot_time_ranges(time_ranges, args.output / "time-ranges.png")
    for row in final:
        if row["sampler"] != "vae_reconstruction":
            print(row["sampler"], {key: round(row["metrics"][key]["delta"], 5)
                                    for key in ("cloth_l1", "contrast_highpass_l1", "cloth_ssim") if key in row["metrics"]})
    for row in time_ranges:
        if row["sampler"] == "euler50" and row["state"] == "teacher":
            metric = row["metrics"].get("teacher_velocity_mse")
            if metric:
                print("euler50 teacher", row["range"], "velocity MSE",
                      round(metric["baseline"], 5), "->", round(metric["candidate"], 5),
                      "delta", round(metric["delta"], 5))


if __name__ == "__main__":
    main()
