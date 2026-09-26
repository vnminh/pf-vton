"""Exercise a worst-case high-resolution training microbatch without saving weights.

Forces detail times and the decoded objective to be active; uses AdamW lr=0
to allocate optimizer state while leaving model weights unchanged. A successful
grounding step alone would not test the later decoder-backward memory peak.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from vton_ext.coral import DINOv3CoralTeacher
from vton_ext.pfi_train import build_model, EditTimeSampler, load_cfg, make_dataset, training_loss, validate_config
from vton_ext.utils import seed_everything
from vton_ext.vae import load_sd_vae


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()
    cfg = load_cfg(args.config, args.overrides)
    validate_config(cfg)
    cfg.flow.curriculum.start = {"pure_noise": 0.0, "synchronous": 0.0, "detail": 1.0}
    cfg.flow.garment_dropout = 0.0
    cfg.loss.decoded_follow_curriculum = False
    cfg.loss.decoded_probability = 1.0
    seed_everything(int(cfg.seed))
    device = torch.device("cuda")
    result = {"config": args.config, "overrides": args.overrides,
              "gpu": torch.cuda.get_device_name(), "image_hw": list(cfg.model.image_hw),
              "batch_size": int(cfg.train.batch_size), "ok": False}
    started = time.time()
    try:
        model = build_model(cfg).to(device).train()
        model.gradient_checkpointing = bool(cfg.train.gradient_checkpointing)
        vae = load_sd_vae(cfg.weights.vae, device)
        teacher = DINOv3CoralTeacher(cfg.weights.dino, min_similarity=float(cfg.coral.min_similarity),
                                    cycle_radius=float(cfg.coral.cycle_radius)).to(device)
        dataset = make_dataset(cfg, cfg.data.test_pairs_file, augment=False)
        batch = next(iter(DataLoader(Subset(dataset, list(cfg.eval.indices)[:int(cfg.train.batch_size)]),
                                     batch_size=int(cfg.train.batch_size))))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0, fused=True)
        sampler = EditTimeSampler(cfg)
        # The second update includes resident Adam moments. Two microbatches
        # also exercise accumulation with an existing gradient allocation.
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(2):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, metrics = training_loss(model, teacher, vae, batch, cfg, sampler, device)
                assert float(metrics["decoded_selected"]) > 0, "Probe did not exercise decoded loss"
                (loss / 2).backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        torch.cuda.synchronize()
        result.update(ok=True, updates=2, accumulation=2, loss=float(loss.detach()), grad_norm=float(grad),
                      metrics={k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in metrics.items()})
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result.update(seconds=time.time() - started, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
