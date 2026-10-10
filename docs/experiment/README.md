# PFI-VTON ablations and evaluation

The paper's existing nine-row sampler table predates this reproducibility
protocol. Do not turn its missing checkpoint step or KID into guessed values.
Use the `checkpoint_step` field in each new Modal evaluation's `metadata.json`.

Training ablations start from the released PFT-XL/2 checkpoint with a fresh
optimizer and the same paired fit/dev split. They retain the 10,000-step cosine
LR horizon even in the 1,000-step feasibility pilot. The generated YAML files
change sampling, loss coefficients, conditioning arithmetic or attention
routing; the explicitly named backbone-size runs change only constructor
hyperparameters. Default PFI-VTON parameters and checkpoint shapes are unchanged.

```bash
python3 scripts/pfi_ablation_matrix.py --write --stage pilot --seed 42
python3 scripts/pfi_ablation_matrix.py --run baseline --stage pilot --seed 42
python3 scripts/pfi_ablation_matrix.py --run shared_timestep --stage pilot --seed 42
python3 scripts/pfi_ablation_matrix.py --write --stage full --seed 42
python3 scripts/pfi_ablation_matrix.py --run baseline --stage full --seed 42
```

Generate `--stage full` for seeds 123 and 2026 only after a variant passes its
pilot. Run **one training variant at a time** on the disk-limited GPU server;
each full `latest.pt` may require about 8 GB plus temporary save space. Keep
each run's resolved config, pair-file fingerprints, `metrics.jsonl`,
`eval.jsonl`, checkpoint step and full test predictions. Compare model variants
at equal optimizer updates, with the same sampler/CFG/seed and test pairs.

`full_attention` allows garment queries to see person keys as well as garment
keys using the existing DiT block weights. It disables garment K/V caching at
inference and currently requires coarse garment tokens. `shared_timestep`
uses one patch's time draw for every editable patch, preserving the marginal
time distribution. `no_role_embedding` and `no_time_embedding` remove only
their additions to the existing activations; they add no new parameters.
The `heads_8` warm start cannot load 16-head Q/K normalization tensors, and
`backbone_b` compares a jointly smaller backbone, so neither is an isolated
causal test of head count. Confirm the PFT-B checkpoint exists before that run.
For the time-distribution study, compare every `curriculum_*`, `ltg_sigma_*`
and `*_only` run
against `no_rgb_hp`: all those configurations disable decoded RGB/HP because
the late-time eligibility gate would otherwise remove extra supervision from
the noise-only arm.

Inference sweep commands can be printed without submitting jobs:

```bash
python3 scripts/pfi_inference_sweep.py --case look8_r150 --mode sample --seeds 42
python3 scripts/pfi_inference_sweep.py --case look8_r150 --mode sample --seeds 42 --execute
python3 scripts/pfi_inference_sweep.py --case look8_r150 --mode full --execute
```

The evaluator saves separate paired/unpaired predictions for each checkpoint,
seed, NFE, sampler and look-ahead ratio. Full-test FID and KID share the same
Inception features. KID uses 100 subsets of 1,000 images and reports its
subset standard deviation; compute seed standard deviation across the three
separate runs. Full evaluation retains 16 deterministic prediction PNGs per
setting after FID/KID and the complete paired `detail.csv`; this keeps the
volume small enough for a sampler sweep. `sample_seconds_per_image` measures synchronized GPU sampling
only. `model_calls_per_image` equals NFE, while `denoiser_calls_per_image`
doubles when garment CFG is active. For model ablations, supply that variant's
checkpoint and config to the Modal command. Review paired clothing and logo
crops alongside aggregate metrics: FID/KID alone cannot show correct letters.

After exporting the small JSON summaries and metadata from the Modal volume,
aggregate seed variation separately from KID's subset variation:

```bash
python3 scripts/aggregate_pfi_inference.py --root /path/to/experiments \
  --case look8_r150 --cfg cfg1 --mode pair
```

Current execution status: the prior SSH endpoint `155.103.252.91:46507`
returns `connection refused`. The local `vt-env` supplies PyTorch and Modal;
44 CPU tests pass. A Modal smoke test was attempted on 8 NFE/CFG 1, but the
selected workspace has no volume named `VTON-HD`. The configured profiles have
older `viton-data`/`viton-run` volumes, not the requested V50 checkpoint.
Training, full image generation and KID need the current server or the correct
Modal workspace/volume with V50 weights and the VITON-HD test set.
