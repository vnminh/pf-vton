# V46 resume regression audit — 27 September 2026

The bad resume changed both **garment–target correspondence** and **fit/dev
membership**. Training continued with almost entirely unrelated garment
references. The step-3000 weights are substantially worse on correctly paired
cases. Restore a clean checkpoint from before that resume and the historical
paired split; do not continue the step-3000 optimizer.

For the active code paths, protected pair lists, and verified new checkpoint
after restoring the deleted logs, see
[the restart record](#restart-after-restoring-the-accidentally-deleted-logs).

## Evidence for the cause

The dataset ZIP's `train_pairs.txt` has 11,647 rows, of which **11,646 are
unpaired**. For example, `10224_00.jpg 03195_00.jpg` conditions the model on an
in-shop garment that is different from the garment in the target person image.
All 11,647 person identities have a same-identity garment in `train/cloth`.

The earlier README instructed splitting those raw rows with random seed 42.
Replaying that instruction exactly reproduces the restored files checked before
the previous resume:

| List used by the bad resume | Rows | Wrong garment identities | SHA-256 |
| --- | ---: | ---: | --- |
| Fit | 11,391 | 11,390 | `fc4f2eab83259905ba9cc7f4dd5452745cb5e97b6e0b8ca63e6632b1deab36a0` |
| Dev | 256 | 256 | `cf355ce929a484dbc37f99a3f94020665875ee34af608455d92e035487beb4ef` |

The previous check verified file existence and current fit/dev overlap, but
missed whether the input garment matched the training target. That was an error
in both the instructions and the verification.

Flow/RGB/detail losses still supervise the original person image. With an
unrelated garment condition, these objectives push the model to reproduce a
different shirt. CoRAL also receives inappropriate garment–person comparisons.
This encourages ignoring or confusing garment conditioning. Unpaired lists are
useful for try-on inference, but cannot supply this paired reconstruction target.

## Resume-state checks

The complete `latest.pt` loads and reports **step 3000**. All 298 populated Adam
parameter states also report step 3000; optimizer learning rates, model shapes,
and the saved curriculum state are consistent. Model tensors checked during
inference are finite. The last log reaches 3120, so the checkpoint is older than
the final logged update, as expected with periodic saving.

Logs contain an original segment through step 1910 and a resumed segment starting
at 1760, consistent with restoring the last saved step 1750. Compare the **same
16 logged steps, 1760–1910**, rather than different curriculum stages:

| Mean over matching steps | Original segment | After resume |
| --- | ---: | ---: |
| Flow loss | 0.401442 | 0.549159 |
| Gradient norm, before clipping | 0.370025 | 2.949379 |
| CoRAL local mass | 0.291711 | 0.227191 |
| Editable-token mean time | 0.220356 | 0.220958 |
| Backbone learning rate | 0.0000187924 | 0.0000187924 |

Flow loss increases about **37%** and gradient norm about **8×**, while learning
rate and sampled times remain comparable. The curriculum progresses rather than
resetting. The initial jump occurs on the old server, before the new migration.
The sampler and model source hashes on the old server match the local versions
before these audit fixes.

Checkpoints do not save RNG or dataloader position, so a resume is not bit-exact.
That is a reproducibility limitation, but does not explain the confirmed pairing
change and persistent regression. No evidence calls for changing the time
curriculum or learning rate before repairing the data.

## Recovering the historical split

The original [`make_vton_dev_split.py`](../scripts/make_vton_dev_split.py) ranks
paired rows by SHA-256 of `2026:<row>` and preserves source row order. It differs
from the later README's random seed-42 recipe.

The historical split was reconstructed from sorted matching person/cloth
identities. **All eight target panels** in the repository's saved V42 preview
match its selected images. A 24×32 thumbnail comparison gives mean absolute
pixel differences of only 0.14–0.27 on a 0–255 scale; the random seed-42 candidates
differ by 33.9–56.9. The reconstruction is supported by both the original splitter
and saved targets; legacy checkpoints themselves contain no split fingerprints.

Restored paired-list hashes:

```text
fit: 2e6e879bf0706a8f66dd26e491dd10fdf3e3596de1d514ee5408a32f202be1ad
dev: 2c5bb4147cc5da88f9139f6822c763e7754b7d6ee1754e0ab04b040e7106c5fd
```

The historical eight preview identities are `01234_00`, `03852_00`, `02298_00`,
`05560_00`, `08522_00`, `13034_00`, `04772_00`, and `07804_00`, in preview order.

**253 of the historical 256 dev people, including all eight preview people,
appear in the wrong resumed fit list.** They were eligible for training after
the bad resume; checkpoints do not record exactly which shuffled examples were
consumed. Conversely, 253 people in the changed dev list belonged to the original
fit set. A within-current-split overlap check misses this historical leakage.
Step 3000 therefore cannot be treated as having an untouched historical dev set.
Rolling back to clean weights removes the offending updates. These previews are
development diagnostics, not official test results.

## Matched inference measurements

The comparison uses the restored historical eight cases, native 1024×768
inputs, identical cached conditioning latents and per-case noise, no
augmentation, dual-loop **8 person-denoiser calls**, CFG 1, and time shift 1.
Both models use the same settings. L1 is measured in RGB `[-1,1]` with equal
per-image weighting; these means differ from the training preview's batchwise
mask-area weighting and must not be substituted directly into its old log.

| Checkpoint | Clothing L1 ↓ | Editable-region L1 ↓ | Clothing high-pass L1 ↓ |
| --- | ---: | ---: | ---: |
| V46 step 3000, bad resume | 0.773683 | 0.407023 | 0.038911 |
| V46 step 500, clean | 0.184981 | 0.137304 | 0.041480 |
| V46 step 1000, clean | 0.192918 | 0.143421 | 0.042735 |
| V46 step 1500, clean, old A6000 | 0.189674 | 0.143074 | 0.041394 |

Step 1500 was evaluated in place on the old server, avoiding a duplicate large
checkpoint transfer. Its damaged step-3000 control gives clothing L1 **0.774222**
and editable-region L1 **0.407716**, close to the new-server measurements. Both
environments report PyTorch 2.11.0+cu128. The small numerical difference between
GPUs does not explain the collapse.

Clothing reconstruction error at step 3000 is over **4×** that of the clean
step-500 snapshot. High-pass error alone fails to expose this regression: it is
slightly lower for the damaged model despite much worse clothing RGB error.
It does not measure text identity or garment correctness. These eight-case
measurements diagnose damage; they do not establish the best checkpoint over
the full development or official test sets.

Numeric outputs are under `/workspace/pfi-resume-audit/` on the new server.
The approved three-case comparison is downloaded into the ignored local
`logs/resume-audit-20260927/` directory. Its columns are **garment, agnostic
person, target, damaged step 3000, clean baseline**. Full resolution inference
was used before reducing panels to 512×384 for convenient viewing.

Visual inspection confirms that step 3000 produces unrelated colors and shirt
styles, even turning the white GAP shirt into a dark sleeveless garment. Clean
step 1000 retains the white graphic shirts and GAP lettering, though Lee and
other fine lettering remain imperfect. This is a loss of garment conditioning
in addition to the pre-existing text/detail limitation.

## Fixes and server state

- [`pairs.py`](../vton_ext/pairs.py) rejects mismatched garment identities in
  paired mode, duplicates, and fit/dev identity overlap.
- [`prepare_pfi_pairs.py`](../scripts/prepare_pfi_pairs.py) builds paired rows
  from actual image/cloth identities and defaults to the historical hash split.
  It preserves raw official lists, refuses differing existing splits, and backs
  up both lists before an explicit `--replace` repair.
- [`pfi_train.py`](../vton_ext/pfi_train.py) validates pairs before allocating
  the model or writing run output. New checkpoints include ordered split
  fingerprints; a changed split is rejected on resume. Legacy checkpoints warn
  that their original split cannot be verified. Pair fingerprints are checked
  again before each checkpoint save to detect external replacements.
- The damaged latest checkpoint is preserved and tagged with
  `latest.pt.invalid-data.json`. The patched trainer rejects its automatic
  resume. Inference remains available for diagnosis.
- [`pfi_model.py`](../vton_ext/pfi_model.py) now preserves learned positions
  during same-resolution weight initialization. Previously the weights-only
  loader dropped them even when the source grid matched. This is a recovery
  improvement, not the cause of the bad full-state resume; full-state resume
  already loads the position table strictly.
- The README and older splitter now guard against the unpaired-list mistake.
  **36 tests pass** on the new server, covering model/sampler behavior, pairing,
  early training rejection, split fingerprints, and position preservation.

The new server reports 49,140 MiB VRAM. Dataset image samples are native
768×1024, and model positions contain 3,072 tokens per stream. Seven required
training modalities were extracted with ZIP CRC checking: **81,529 files,
4.311 GB**. The ZIP and checkpoints were preserved. The old server has complete
step-500/1000/1500 clean snapshots, plus V43-pilot and V44 safety weights.

The new migration is still copying large snapshots before the source files;
the main source directories were mostly empty during the audit. Patched code
was uploaded into `/workspace/pfi-resume-audit/code`, separate from that copy.
The short audit processes used Supervisor and did not modify model or optimizer
weights. After diagnosis, the user explicitly requested retraining on the new
RTX 4090 server; that recovery run is described below.

## Initial recovery before accidental log deletion

Use the last verified clean V46 checkpoint at or before step 1750 if available.
Step 1500 exists on the old server and is already being copied to the new one.
A full clean optimizer checkpoint permits an exact-state continuation apart
from RNG/dataloader position. The retained numbered snapshots are weights only;
initialize a **new output directory with a fresh optimizer** from those weights.
Do not resume the damaged step-3000 optimizer. Do not restart from PFT alone while
useful clean VTON weights exist.

The step-1500 copy is complete and its SHA-256 matches on both servers:

```text
df3b35f8935fa91217f3265532727d16c88020294c404405b7eabd1c428984f5
```

The requested recovery was launched on **175.155.64.157:16377, RTX 4090** using
[`vton_v46_pfi_1024_paired_recovery.yaml`](../configs/vton_v46_pfi_1024_paired_recovery.yaml)
and [`run_pfi_v46_paired_recovery.sh`](../scripts/run_pfi_v46_paired_recovery.sh).
The initial resolved server configuration was
`/workspace/pfi-resume-audit/code/configs/v46-recovery-server.yaml`.

- Output: `/workspace/patch-forcing-vton/logs/vton-v46-pfi-1024-paired-recovery`.
- Parent weights: clean step 1500, including its learned position table.
- Fresh optimizer and curriculum: the new log counter starts at **0**, not 1500.
- Full 1024×768 inputs and decoded supervision; no decoder crop.
- Batch 4 × accumulation 4; 10,000 new-update LR horizon; 200 warmup updates.
- Full latest checkpoint every 250 updates; previews every 500; four retained
  numbered snapshots and a 24 GiB free-space threshold for optional snapshots.
- Read-only copies of the historical paired lists live inside this new run's
  own output directory, so the ongoing migration's original pair lists are not
  used by training. `recovery_provenance.json` records the parent hash and split
  fingerprints. Resume reads the new run's own latest checkpoint.

Before launch, a probe forced late detail times and decoded RGB/high-pass loss
active through two updates with resident Adam state and gradient accumulation.
It passed on the RTX 4090: **36.135 GiB allocated, 37.432 GiB reserved**, finite
loss and gradients. All 36 regression tests passed. The actual step-zero
training preview gives clothing L1 **0.182482** (dual-loop 8, CFG 1, shift 1),
consistent with the pre-corruption step-1500 logged preview, 0.183005, allowing
for the retained snapshot's bf16 precision and hardware numerics.

The first live check confirms Supervisor **RUNNING** on the RTX 4090 at recovery
step **10**, with flow loss **0.340751**, gradient norm **0.278228**, CoRAL local
mass **0.297532**, and finite logged metrics. GPU utilization is 100%; the logged
peak is 36.135 GiB. These are early health checks, not evidence of final quality
improvement; the restarted curriculum has a different time mixture from old
step 1500. Both run-specific pair files have permissions `0444`.

Monitor the run on the new server:

```bash
supervisorctl status pfi-v46-paired-recovery
tail -f /workspace/patch-forcing-vton/logs/vton-v46-pfi-1024-paired-recovery.log
```

After a period of healthy paired training, compare the full fixed dev set and
profile denoising time. Use the untouched official test split for final
generalization claims. First repair garment correspondence; extra logo/detail
losses cannot repair contradictory targets.

## Restart after restoring the accidentally deleted logs

The restored backup contains the original `vton-v46-pfi-1024-c2f` snapshots,
but no saved checkpoint from the newer paired-recovery run. Its deleted pair
files caused the old Supervisor entry to fail. The restored step-1500 snapshot
loads successfully and has the same SHA-256 recorded above. Recovery therefore
starts from those clean weights with a fresh optimizer and a counter of zero;
it cannot preserve updates that were never saved in the available backup.

Local source was restored to `/workspace/patch-forcing-vton`, replacing the
empty source folders. All **111 uploaded files** match their archive contents.
All **37 training regression tests** pass on the server. The active Supervisor
entry now uses this main repository rather than the temporary audit code.

The active server configuration is now
`/workspace/patch-forcing-vton/configs/v46-recovery-server.yaml`.
Read-only paired lists are stored **outside `logs/`**:

```text
/workspace/pfi-resume-audit/pairs/v46-paired-recovery/train_fit_pairs.txt
/workspace/pfi-resume-audit/pairs/v46-paired-recovery/train_dev_pairs.txt
```

Their counts and hashes exactly match the historical 11,391/256 split above.
All seven modalities exist for all 11,647 pairs; decoded dataset samples have
shape `[3, 1024, 768]`, with native image size 768×1024. Provenance is also
stored outside the log tree at
`/workspace/pfi-resume-audit/recovery-provenance-20260927.json`.

Two recovery settings were added: `train.save_first_step: 10` creates a full
resumable checkpoint before the regular 250-update save, and
`train.require_pair_fingerprints: true` rejects legacy resume checkpoints that
cannot verify the data split. The clean legacy step-1500 snapshot remains a
weights initializer. Later restarts automatically resume this new run's
`latest.pt`, including its optimizer and curriculum, after verifying its pair
fingerprints. The output path and monitoring commands above still apply.

The restored model's step-zero clothing L1 is **0.182482**, matching the initial
clean recovery baseline. At new recovery step **10**, flow loss is **0.340754**,
gradient norm **0.268112**, CoRAL local mass **0.297537**, and peak GPU allocation
**36.135 GiB**; all logged values are finite. This remains an early health check,
not a final assessment of VTON quality.

The new `latest.pt` at recovery step 10 was loaded and verified: **8,141,593,709
bytes**, 298 model tensors, 298 optimizer parameter states all at update 10,
curriculum state, and matching fit/dev fingerprints. Its SHA-256 is:

```text
2392bbbaae1cbd88099ac0fff6c9880fae19fcc5886e86c64cb87cf6b2574082
```

Supervisor remains **RUNNING** on the RTX 4090 after the save, with 100% GPU
utilization and approximately 45 GB disk space free. This checkpoint is now the
resumable state for this recovery run; its counter is separate from the parent
snapshot's original step-1500 counter.
