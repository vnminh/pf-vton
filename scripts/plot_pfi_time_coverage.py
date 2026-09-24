"""Relate the final training-time coverage to teacher velocity error by time."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vton_ext.pfi_train import EditTimeSampler, load_cfg


def sample_coverage(config, seed, step, examples=8192, tokens=64):
    cfg = load_cfg(str(config))
    sampler = EditTimeSampler(cfg)
    sampler.load_state_dict({"ramp_start": int(cfg.flow.curriculum.gate_min_steps), "metric_ema": None})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        values = sampler(examples, tokens, "cpu", step=step).numpy().reshape(-1)
    edges = np.linspace(0, 1, 21)
    counts, _ = np.histogram(values, bins=edges)
    return {"edges": edges.tolist(), "fraction": (counts / counts.sum()).tolist(),
            "above_08": float(np.mean(values > .8)), "above_09": float(np.mean(values > .9)),
            "examples": examples, "tokens_per_example": tokens, "seed": seed,
            "ramp_start_assumed": int(cfg.flow.curriculum.gate_min_steps), "step": step}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-config", required=True, type=Path)
    p.add_argument("--candidate-config", required=True, type=Path)
    p.add_argument("--comparison", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--step", type=int, default=6000)
    p.add_argument("--baseline-label", default="Baseline")
    p.add_argument("--candidate-label", default="Candidate")
    args = p.parse_args()
    baseline = sample_coverage(args.baseline_config, 12345, args.step)
    candidate = sample_coverage(args.candidate_config, 12345, args.step)
    comparison = json.loads(args.comparison.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "time-coverage.json").write_text(
        json.dumps({"baseline": baseline, "candidate": candidate,
                    "note": "Sampled final-mixture patch times, not the historical training batches"}, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    centers = (np.asarray(baseline["edges"][:-1]) + np.asarray(baseline["edges"][1:])) / 2
    axes[0].step(centers, baseline["fraction"], where="mid",
                 label=f"{args.baseline_label} final mixture", color="#1673b1")
    if args.baseline_config.resolve() != args.candidate_config.resolve():
        axes[0].step(centers, candidate["fraction"], where="mid",
                     label=f"{args.candidate_label} final mixture", color="#d35400")
    axes[0].set(ylabel="Patch-time fraction per 0.05 bin", title="Training coverage after curriculum ramp")
    axes[0].legend()
    for sampler in ("euler8", "euler50"):
        rows = sorted((r for r in comparison["time"] if r["sampler"] == sampler and
                       r["state"] == "teacher" and "teacher_velocity_mse" in r["metrics"]),
                      key=lambda r: r["time"])
        if not rows:
            continue
        x = [r["time"] for r in rows]
        for side, style, color in (("baseline", "--", "#1673b1"), ("candidate", "-", "#d35400")):
            y = [r["metrics"]["teacher_velocity_mse"][side] for r in rows]
            label = args.baseline_label if side == "baseline" else args.candidate_label
            axes[1].plot(x, y, style, color=color, marker="." if sampler == "euler8" else None,
                         label=f"{label} {sampler}")
    axes[1].set(xlabel="Time (0 = noise; 1 = clean)", ylabel="Teacher latent clothing velocity MSE",
                title="Matched 64-case teacher error")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_xlim(0, 1)
    fig.savefig(args.output, dpi=160)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
