# Patch Forcing VTON: V43 at 1024 × 768

Run these commands from the **repository root**. V43 uses the released PFT-XL/2
weights to initialize a new try-on model. It trains on native-resolution
VITON-HD images with patch forcing, a DINOv3 correspondence teacher, and a
curriculum that adds decoded RGB/detail supervision. The configuration is
[`configs/vton_v43_pfi_1024.yaml`](../configs/vton_v43_pfi_1024.yaml).

For the existing V42 model, use the **fine-tuning recipe** below instead of
starting the VTON conditioning again. The reviewed parent is V42 step 16000;
its full optimizer checkpoint is preserved separately on the training server.
See [the high-resolution review](../docs/vton_v43_high_resolution_review.md)
for the choices and measured memory limits.

## 1. Environment and checkpoints

Use Python 3.12 and a CUDA/PyTorch installation appropriate for your GPU. The
repository's upstream and VTON dependencies are separate:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-vton.txt
mkdir -p checkpoints
```

V43 needs these three **different** pretrained artifacts:

| Config key | Local path | Source |
| --- | --- | --- |
| `weights.pft` | `checkpoints/pft-xl_step400k_ema.ckpt` | [Released PFT-XL/2](https://ommer-lab.com/files/pft/pft-xl_step400k_ema.ckpt) |
| `weights.vae` | `checkpoints/sd_ae.ckpt` | [CompVis first-stage autoencoder](https://github.com/CompVis/fm-boosting#-usage) |
| `weights.dino` | `checkpoints/dinov3-vits16-pretrain-lvd1689m/` | [Meta DINOv3 ViT-S/16](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) |

```bash
curl -fL --retry 3 -o checkpoints/pft-xl_step400k_ema.ckpt \
  'https://ommer-lab.com/files/pft/pft-xl_step400k_ema.ckpt'
curl -fL --retry 3 -o checkpoints/sd_ae.ckpt \
  'https://www.dropbox.com/scl/fi/lvfvy7qou05kxfbqz5d42/sd_ae.ckpt?rlkey=fvtu2o48namouu9x3w08olv3o&st=vahu44z5&dl=1'
```

Request access to the [gated DINOv3 model](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
with your Hugging Face account, then authenticate on the training machine and
download the **whole repository** (the teacher reads both its model and image
processor configuration):

```bash
hf auth login
hf download facebook/dinov3-vits16-pretrain-lvd1689m \
  --local-dir checkpoints/dinov3-vits16-pretrain-lvd1689m
test -s checkpoints/pft-xl_step400k_ema.ckpt
test -s checkpoints/sd_ae.ckpt
test -s checkpoints/dinov3-vits16-pretrain-lvd1689m/config.json
test -s checkpoints/dinov3-vits16-pretrain-lvd1689m/preprocessor_config.json
```

These checkpoint paths are ignored by Git. If you already have the exact files,
place them at the paths above instead of downloading them again. The SD VAE is
the `jutils` `AutoencoderKL` state dict expected by [`vae.py`](vae.py); an
arbitrary Diffusers VAE directory is not interchangeable with `sd_ae.ckpt`.

## 2. VITON-HD data and pair split

Download the [preprocessed 1024 × 768 VITON-HD dataset](https://github.com/shadow2496/VITON-HD#dataset)
and extract it as a sibling of this repository:

```text
../high-resolution-viton-zalando-dataset/
  train_pairs.txt
  train/{image,cloth,cloth-mask,agnostic-v3.2,agnostic-mask,image-densepose,image-parse-v3}/
  test/...
```

`train_fit_pairs.txt` and `train_dev_pairs.txt` are **local development** lists
used by V43; they are not supplied by the original dataset. Create a fixed
256-pair development split from the original training pairs, keeping the
official test split untouched:

```bash
python - <<'PY'
from pathlib import Path
from random import Random

root = Path('../high-resolution-viton-zalando-dataset')
for name in ('train_fit_pairs.txt', 'train_dev_pairs.txt'):
    assert not (root / name).exists(), f'Keep the existing split: {root / name}'
rows = [line for line in (root / 'train_pairs.txt').read_text().splitlines() if line.strip()]
assert len(rows) > 256, 'Expected the full VITON-HD training pair list'
dev = set(Random(42).sample(range(len(rows)), 256))
(root / 'train_fit_pairs.txt').write_text(''.join(f'{row}\n' for i, row in enumerate(rows) if i not in dev))
(root / 'train_dev_pairs.txt').write_text(''.join(f'{row}\n' for i, row in enumerate(rows) if i in dev))
print(f'{len(rows) - len(dev)} training pairs; {len(dev)} development pairs')
PY
```

The loader expects the listed image, mask, DensePose, garment and human-parse
folders under `train/`. If your dataset lives elsewhere, set `data.root` in
the config before training. The eight fixed preview indices must be valid in
`train_dev_pairs.txt` (256 rows satisfy this).

## 3. Train and resume V43

The default config targets full 1024 × 768 images, latent size 128 × 96,
batch 4 × gradient accumulation 4, and a 30,000-step learning-rate horizon.
It starts from PFT-XL/2 with a new optimizer (`weights.init_from: null`); it
does **not** require a V40–V42 checkpoint. Run it on your 48 GB GPU from the
repository root:

```bash
PYTHONPATH=. python -m vton_ext.pfi_train \
  --config configs/vton_v43_pfi_1024.yaml
```

Check `logs/vton-v43-pfi-1024/metrics.jsonl` for finite losses and `mem_gb`
after the first ten steps. `latest.pt` contains model, optimizer and curriculum
state; every 1,000 steps a model-only `stepXXXXXXX.pt` and an eight-case preview
are retained. If batch 4 exceeds GPU memory, preserve the effective batch of
16 with `train.batch_size=1 train.gradient_accumulation_steps=16
eval.batch_size=1`. If the full-image decoded loss is still too large, add
`loss.decoded_crop_latent=[64,48] loss.decoded_crop_margin=4`; this supervises
a clothing-centered window instead of the entire image. These are OmegaConf
`key=value` overrides appended after the config path.

Resume the **same-resolution V43 run** from its complete checkpoint:

```bash
test -s logs/vton-v43-pfi-1024/latest.pt
PYTHONPATH=. python -m vton_ext.pfi_train \
  --config configs/vton_v43_pfi_1024.yaml --resume auto
```

Use the same overrides again if you changed them for the first run. Before
resuming, verify `logs/vton-v43-pfi-1024/latest.pt` exists; `--resume auto`
loads that file. To stop after a short pilot without changing the 30,000-step
learning-rate schedule, add `train.stop_at_step=1000`; then remove this override
when resuming.

To **initialize weights** from a 512 × 384 PFI checkpoint such as V42 instead,
copy its model checkpoint to this machine and set `weights.init_from` to that
path on a *new* V43 run. The position table is rebuilt for 1024 × 768 and the
optimizer/curriculum start fresh. Do not pass a 512 × 384 checkpoint to
`--resume`, which requires matching parameter shapes and restores its optimizer:

```bash
PYTHONPATH=. python -m vton_ext.pfi_train \
  --config configs/vton_v43_pfi_1024.yaml \
  weights.init_from=checkpoints/v42-stepXXXXXXX.pt \
  train.output_dir=logs/vton-v43-from-v42-1024
```

The transfer checkpoint is optional and is **not** hosted with the three
pretrained artifacts above. `--resume` should only be used for a V43
`latest.pt` from the same architecture and resolution. Logs and checkpoints
stay under `logs/`, which is ignored by Git; back up wanted checkpoints
separately from the source repository.

## 4. Recommended continuation from V42

On the current server, `checkpoints/pfi-safety/v42-step16000.pt` holds the
weights extracted from the last saved V42 checkpoint. Copy this file when
moving machines; it is not included in Git or the public downloads.

For a GPU with enough memory for full-image decoded supervision:

```bash
PYTHONPATH=. python -m vton_ext.pfi_train \
  --config configs/vton_v43_pfi_1024_finetune.yaml
```

For the current 24 GB RTX 3090:

```bash
PYTHONPATH=. python -m vton_ext.pfi_train \
  --config configs/vton_v43_pfi_1024_24gb.yaml
```

Both recipes use 1024 × 768 inputs and the full 3,072-token person and garment
streams. The 24 GB variant uses smaller microbatches and a jittered 512 × 384
clothing crop for decoded loss, with 32 pixels of decoder context around it.
It evaluates whole images and accumulates an effective batch of 16. Cropped
decoder supervision is an approximation because decoder context is limited;
it is not identical to full-image decoded loss.

The fine-tuning pilot runs 1,000 **new** optimizer steps with a 10,000-step LR
horizon, 100 warmup steps, and a backbone peak LR of `5e-6`. It retains the
learned time-mixture probabilities and uses a new optimizer. Step 0, 500 and
1000 previews compare Euler 8, shifted dual-loop 8, and unshifted dual-loop 8.
Each uses eight person-denoiser calls (`cfg=1`). Evaluation noise is fixed per
case and does not change with evaluation batch size.

The server launcher [`run_pfi_v43_finetune.sh`](../scripts/run_pfi_v43_finetune.sh)
resumes the 24 GB run automatically if its `latest.pt` exists. Inspect it with
`supervisorctl status pfi-v43-1024` and
`tail -f logs/vton-v43-pfi-1024-24gb.log`. To continue past the pilot, increase
`train.stop_at_step` without shortening `train.max_steps`.

Before changing microbatch size or enabling full-image decoded loss, run the
memory probe; it forces the decoded objective on and tests backward,
accumulated gradients, and resident Adam state across two updates:

```bash
PYTHONPATH=. python scripts/check_pfi_high_resolution.py \
  --config configs/vton_v43_pfi_1024_24gb.yaml \
  --output logs/v43-memory-probe.json
```

It uses an optimizer learning rate of zero and saves no model checkpoint.
