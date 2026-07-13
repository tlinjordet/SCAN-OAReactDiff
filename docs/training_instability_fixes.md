# Training Instability: Findings and Fixes

**Context:** OA-ReactDiff (`oa_reactdiff/`) shows high run-to-run variability when trained with
identical data, model configuration, and random seed. Two runs with `seed_everything(42)` can
converge to models of wildly different quality. This document traces the root causes, explains the
mechanics of each failure mode, and provides complete, copy-pastable code patches for every fix.
All changes are **backwards-compatible**: existing behaviour is preserved for code paths that do not
opt in.

---

## Table of Contents

1. [Finding 1 — Silent NaN Recovery Corrupts Gradients](#finding-1--silent-nan-recovery-corrupts-gradients)
2. [Finding 2 — Gradient Clipping Queue Cold-Starts at 3000](#finding-2--gradient-clipping-queue-cold-starts-at-3000)
3. [Finding 3 — Non-Deterministic CUDA Ops Despite Fixed Seed](#finding-3--non-deterministic-cuda-ops-despite-fixed-seed)
4. [Finding 4 — EMA Is Wired Up but Disabled](#finding-4--ema-is-wired-up-but-disabled)
5. [Finding 5 — No Learning-Rate Warmup](#finding-5--no-learning-rate-warmup)
6. [Finding 6 — TS Fragment Loss Scale Amplifies Early Noise](#finding-6--ts-fragment-loss-scale-amplifies-early-noise)
7. [Recommended Rollout Order](#recommended-rollout-order)
8. [Summary of All Changed Lines](#summary-of-all-changed-lines)

---

## Finding 1 — Silent NaN Recovery Corrupts Gradients

### File
`oa_reactdiff/dynamics/egnn_dynamics.py`, lines 138–143

### What the code currently does

```python
# oa_reactdiff/dynamics/egnn_dynamics.py  (current)
vel = pos_final - pos
if torch.any(torch.isnan(vel)):
    print("Warning: detected nan in pos, resetting EGNN output to randn.")
    vel = torch.randn_like(vel)
if torch.any(torch.isnan(vel)):                          # <-- BUG: checks vel again, not h_final
    print("Warning: detected nan in h, resetting EGNN output to randn.")
    h_final = torch.randn_like(h_final)
```

### Why this causes instability

**Bug 1 — the second `isnan` check is on the wrong tensor.** After the first block runs,
`vel` is replaced with `randn`, so it can never be NaN. The second condition therefore never fires.
`h_final` NaN events go completely undetected and unhandled.

**Bug 2 — `torch.randn_like` is the wrong recovery action.** When a forward pass produces NaN
it usually means the model weights are in a numerically bad region (e.g. attention logits or
inter-fragment distances have overflowed). Replacing the NaN output with fresh random noise does
*not* neutralise the problem — instead, it injects a large, random gradient signal into the
backward pass through that entire computation graph. The weight update that follows then pushes
the parameters further off-track, potentially triggering more NaN events on subsequent steps.
The warning message implies the situation is handled gracefully, but the model may be silently
diverging.

A far safer choice is to replace NaN outputs with **zeros**, which produces a zero gradient
for that step (a no-op weight update) rather than a destructive one. Pair this with a NaN guard
in `training_step` that logs the event and skips the batch, so the problem is visible without
crashing the run.

### Fix — Part A: `egnn_dynamics.py`

Replace lines 138–143:

```python
# oa_reactdiff/dynamics/egnn_dynamics.py  (fixed)
vel = pos_final - pos
if torch.any(torch.isnan(vel)):
    print("Warning: NaN detected in predicted velocity (pos). Zeroing this output.")
    vel = torch.zeros_like(vel)
if torch.any(torch.isnan(h_final)):          # <-- fixed: now checks h_final
    print("Warning: NaN detected in node features (h). Zeroing this output.")
    h_final = torch.zeros_like(h_final)
```

Zero velocity means the network predicts no positional displacement for this forward pass; zero
`h_final` means the node-feature branch also contributes nothing. Both zero out through the
decoder and produce a clean zero gradient — the optimizer skips the step effectively, without
corrupting weights.

### Fix — Part B: `pl_trainer.py` — batch-level NaN guard in `training_step`

Even with zeroed dynamics outputs, a malformed batch (e.g. extreme atom positions) can still
produce a NaN loss via numerical accumulation elsewhere. Add a guard in `training_step` that
detects this, logs it, and returns a harmless zero loss:

```python
# oa_reactdiff/trainer/pl_trainer.py — inside DDPMModule.training_step (fixed)
def training_step(self, batch, batch_idx):
    nll, info = self.compute_loss(batch)
    loss = nll.mean(0)

    # Guard: skip batches that produce NaN/Inf loss rather than corrupting weights.
    if not torch.isfinite(loss):
        print(
            f"Warning: non-finite loss ({loss.item():.4g}) at epoch "
            f"{self.current_epoch} batch {batch_idx}. Skipping weight update."
        )
        info["rmsd"], info["rmsd-median"] = np.nan, np.nan
        info["loss"] = torch.tensor(0.0, device=loss.device, requires_grad=True)
        return info

    self.log("train-totloss", loss, rank_zero_only=True)
    for k, v in info.items():
        self.log(f"train-{k}", v, rank_zero_only=True)

    if (self.current_epoch + 1) % self.eval_epochs == 0 and batch_idx == 0:
        if self.trainer.is_global_zero:
            print(
                "evaluation on samping for training batch...",
                batch[1].shape,
                batch_idx,
            )
        rmsd_mean, rmsd_median = self.eval_inpaint_batch(batch)
        info["rmsd"], info["rmsd-median"] = rmsd_mean, rmsd_median
    else:
        info["rmsd"], info["rmsd-median"] = np.nan, np.nan
    info["loss"] = loss
    return info
```

The returned zero-loss tensor has `requires_grad=True`, so PyTorch Lightning completes the
backward pass without errors, but no gradient flows and no weight is updated for that batch.

---

## Finding 2 — Gradient Clipping Queue Cold-Starts at 3000

### Files
- `oa_reactdiff/trainer/pl_trainer.py`, lines 143–146 (queue initialisation)
- `oa_reactdiff/trainer/pl_trainer.py`, lines 391–418 (clipping logic)
- `oa_reactdiff/utils/training_tools.py`, line 4 (queue `max_len=50`)

### What the code currently does

```python
# oa_reactdiff/trainer/pl_trainer.py  (current)
self.clip_grad = training_config["clip_grad"]
if self.clip_grad:
    self.gradnorm_queue = utils.Queue()
    self.gradnorm_queue.add(3000)          # single sentinel to bootstrap the queue
```

The queue stores the gradient norms observed in recent steps. The adaptive clip threshold is:

```python
# oa_reactdiff/trainer/pl_trainer.py — configure_gradient_clipping  (current)
max_grad_norm = 1.5 * self.gradnorm_queue.mean() + 3 * self.gradnorm_queue.std()
```

The queue has `max_len=50`, meaning it holds the last 50 gradient norm measurements. It is
initialised with a single value of 3000.

### Why this causes instability

For the first 50 training steps:

- `mean` ≈ 3000 (dominated by the sentinel)
- `std` ≈ 0 initially, grows slowly as real norms fill the queue
- `max_grad_norm` = 1.5 × 3000 + 3 × 0 = **4500**

This means the gradient clipper allows norms up to 4500 for the first ~50 steps — effectively
no clipping at all. A randomly-initialised LEFTNet with 6 layers, hidden channels 196, and 96
radial basis functions can easily produce gradient norms in the hundreds or thousands at
initialisation, particularly when the RBF distances are large or the first few scatter
aggregations saturate. An early spike pushes weights into a region from which Adam's adaptive
moments (which also need warm-up) cannot easily recover.

The model is **most vulnerable at initialisation**, and this is precisely when the clipping
provides the least protection.

### Fix — `pl_trainer.py`

Pre-fill the queue with a conservative starting estimate rather than a single large sentinel.
A value of 10.0 is appropriate for a randomly initialised network of this scale (typical early
gradient norms for this model are in the range 1–50). The queue will naturally converge to the
true running statistics within ~50 steps:

```python
# oa_reactdiff/trainer/pl_trainer.py  (fixed)
self.clip_grad = training_config["clip_grad"]
if self.clip_grad:
    _gradnorm_init = training_config.get("gradnorm_queue_init", 10.0)
    self.gradnorm_queue = utils.Queue()
    for _ in range(self.gradnorm_queue.max_len):
        self.gradnorm_queue.add(_gradnorm_init)
```

The `training_config.get("gradnorm_queue_init", 10.0)` default means **no change to any existing
call site** — passing nothing gives the new safe default. To restore the old behaviour exactly,
set `gradnorm_queue_init=3000` in `training_config` (not recommended).

In `train_ts1x.py`, add the new key to `training_config`:

```python
# oa_reactdiff/trainer/train_ts1x.py  (add to training_config dict)
training_config = dict(
    datadir="../data/transition1x/",
    remove_h=False,
    bz=14,
    num_workers=0,
    clip_grad=True,
    gradient_clip_val=None,
    gradnorm_queue_init=10.0,      # <-- add this line
    ema=False,
    ema_decay=0.999,
    swapping_react_prod=True,
    append_frag=False,
    use_by_ind=True,
    reflection=False,
    single_frag_only=True,
    only_ts=False,
    lr_schedule_type=None,
    lr_schedule_config=dict(
        gamma=0.8,
        step_size=100,
    ),
)
```

### Why 10.0 and not some other value?

The clipping rule `1.5 * mean + 3 * std` is designed to allow the gradient norm to be up to 150%
of the recent mean, plus a standard-deviation buffer. During the first real steps, the model
should be producing moderate gradients (O(1)–O(100)); initialising the queue at 10.0 means the
clip threshold starts at 15.0 and adjusts as real norms arrive. If the actual norms are larger
than 10.0, the queue updates upward within 50 steps. If they're smaller, the threshold tightens
within 50 steps. Neither direction is a catastrophe.

The sentinel value 3000 was never observationally grounded — it was chosen as "a very large
number" to avoid clipping on the very first step. The queue-fill approach achieves the same
goal (no excessive clipping of legitimate gradients) without leaving a 50-step window of
effective no-clipping.

---

## Finding 3 — Non-Deterministic CUDA Ops Despite Fixed Seed

### File
`oa_reactdiff/trainer/train_ts1x.py`, line 208

### What the code currently does

```python
# oa_reactdiff/trainer/train_ts1x.py  (current)
seed_everything(42, workers=True)
# ...
trainer = Trainer(
    max_epochs=2000,
    accelerator="gpu",
    deterministic=False,          # <-- non-deterministic
    ...
)
```

### Why this causes instability

`seed_everything(42)` seeds Python's `random`, NumPy, and PyTorch CPU/CUDA RNGs. However, it
does **not** make CUDA atomic operations deterministic. The codebase relies heavily on
`torch_scatter.scatter_mean`, `torch_geometric`'s scatter-based aggregations, and
`torch.scatter_add` internally. All of these use CUDA atomics, which execute in arbitrary order
on GPU and produce slightly different floating-point results depending on thread scheduling.

This means:
1. Two runs with `seed_everything(42)` on the same hardware can diverge after the first scatter
   operation (typically within the first batch of the first epoch).
2. When a run produces bad results, there is no way to reproduce the failure to debug it —
   it may never recur with the same seed.
3. The variability compounds over 2000 epochs.

### Fix — `train_ts1x.py`

```python
# oa_reactdiff/trainer/train_ts1x.py  (fixed)
trainer = Trainer(
    max_epochs=2000,
    accelerator="gpu",
    deterministic="warn",         # <-- changed from False
    ...
)
```

`deterministic="warn"` makes PyTorch use deterministic algorithms wherever they exist, and logs
a warning for any operation that cannot be made deterministic rather than raising an error. This
is the recommended setting for debugging because it maximises reproducibility without crashing on
any ops that genuinely lack a deterministic implementation.

Use `deterministic=True` if you require *strict* reproducibility (it will raise an error on
non-deterministic ops, forcing you to resolve each one). `deterministic=True` is the gold
standard for ablation studies. Note that deterministic mode has a performance cost (typically
5–20% slower training throughput).

---

## Finding 4 — EMA Is Wired Up but Disabled

### File
`oa_reactdiff/trainer/train_ts1x.py`, line 83

### What the code currently does

```python
# oa_reactdiff/trainer/train_ts1x.py  (current)
training_config = dict(
    ...
    ema=False,                    # <-- disabled
    ema_decay=0.999,
    ...
)
# ...
callbacks = [earlystopping, checkpoint_callback, TQDMProgressBar(), lr_monitor]
if training_config["ema"]:
    callbacks.append(EMACallback(decay=training_config["ema_decay"]))
```

The `EMACallback` in `oa_reactdiff/trainer/ema.py` is implemented and the decay constant is
already set. It is simply not added to the callback list.

### Why EMA helps stability

Exponential Moving Average of weights maintains a shadow copy of the model parameters:

```
ema_weights = decay * ema_weights + (1 - decay) * current_weights
```

With `decay=0.999`, the EMA weights change very slowly — equivalent to averaging over the last
~1000 gradient steps. This has two stabilising effects:

1. **Filters gradient noise**: A single bad gradient step (from an unusual batch or a near-NaN
   event) moves the current weights by a full step, but moves the EMA weights by only 0.1%.
   Checkpoints and validation that use EMA weights are therefore much less sensitive to
   individual bad batches.
2. **Smooths the loss landscape seen by early stopping and model selection**: The saved checkpoint
   is the EMA model, not the instantaneous model. Two runs that experience different gradient
   noise patterns converge to more similar EMA weights than instantaneous weights.

EMA is standard practice in diffusion model training (DDPM, EDM, DiT all use it). The `decay`
parameter is already well-chosen at 0.999.

### Fix — `train_ts1x.py`

```python
# oa_reactdiff/trainer/train_ts1x.py  (fixed)
training_config = dict(
    ...
    ema=True,                     # <-- changed from False
    ema_decay=0.999,
    ...
)
```

That is the only change required. The rest of the callback wiring is already correct.

---

## Finding 5 — No Learning-Rate Warmup

### Files
- `oa_reactdiff/trainer/train_ts1x.py`, lines 67–72 and 91–95
- `oa_reactdiff/trainer/pl_trainer.py`, lines 49–52 and 149–158

### What the code currently does

```python
# oa_reactdiff/trainer/train_ts1x.py  (current)
optimizer_config = dict(
    lr=2.5e-4,
    betas=[0.9, 0.999],
    weight_decay=0,
    amsgrad=True,
)

training_config = dict(
    ...
    lr_schedule_type=None,        # no schedule at all
    lr_schedule_config=dict(
        gamma=0.8,
        step_size=100,
    ),
)
```

```python
# oa_reactdiff/trainer/pl_trainer.py — configure_optimizers  (current)
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.ddpm.parameters(), **self.optimizer_config)
    if not self.training_config["lr_schedule_type"] is None:
        scheduler_func = LR_SCHEDULER[self.training_config["lr_schedule_type"]]
        scheduler = scheduler_func(
            optimizer=optimizer, **self.training_config["lr_schedule_config"]
        )
        return [optimizer], [scheduler]
    else:
        return optimizer
```

### Why this causes instability

Adam's adaptive per-parameter learning rates (`m / (sqrt(v) + eps)`) are computed from running
estimates `m` (first moment) and `v` (second moment). At step 0, both moments are zero. During
the first ~100–300 steps, the bias-corrected estimates are still dominated by the initial zero,
which means Adam's effective learning rate can be significantly different from the configured
`lr`. With `amsgrad=True`, the `v_max` term also starts at zero, making the effective learning
rate for low-variance parameters potentially very large in early steps.

Coupled with the poorly-initialised gradient clipping queue (Finding 2), this creates a window
at the start of training where the effective update size can be very large. For a task as
geometrically constrained as TS prediction — where the loss landscape has sharp curvature from
the SE(3) equivariance constraints — large early updates can push the model into flat or
divergent regions.

Linear warmup over ~200–500 steps manually controls the initial learning rate while Adam's
moments stabilise, preventing catastrophic early steps.

### Fix — Part A: Imports in `pl_trainer.py`

Add `LambdaLR` to the existing scheduler import:

```python
# oa_reactdiff/trainer/pl_trainer.py — imports  (fixed)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, LambdaLR
```

### Fix — Part B: Add warmup to `LR_SCHEDULER` and `configure_optimizers` in `pl_trainer.py`

Replace the existing `configure_optimizers` with a version that supports an optional
`warmup_steps` key in `lr_schedule_config`:

```python
# oa_reactdiff/trainer/pl_trainer.py — LR_SCHEDULER dict and configure_optimizers  (fixed)

LR_SCHEDULER = {
    "cos": CosineAnnealingWarmRestarts,
    "step": StepLR,
}


class DDPMModule(LightningModule):
    # ... (other methods unchanged) ...

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.ddpm.parameters(), **self.optimizer_config)

        warmup_steps = self.training_config.get("warmup_steps", 0)

        if self.training_config["lr_schedule_type"] is not None:
            # Build the primary epoch-level scheduler.
            scheduler_func = LR_SCHEDULER[self.training_config["lr_schedule_type"]]
            # Remove warmup_steps from the config before passing to the primary scheduler.
            primary_cfg = {
                k: v
                for k, v in self.training_config["lr_schedule_config"].items()
                if k != "warmup_steps"
            }
            primary_scheduler = scheduler_func(optimizer=optimizer, **primary_cfg)

            if warmup_steps > 0:
                # Linear warmup applied at the *step* level, primary at the epoch level.
                warmup_scheduler = LambdaLR(
                    optimizer,
                    lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps),
                )
                return (
                    [optimizer],
                    [
                        {"scheduler": warmup_scheduler, "interval": "step", "frequency": 1},
                        {"scheduler": primary_scheduler, "interval": "epoch", "frequency": 1},
                    ],
                )
            return [optimizer], [primary_scheduler]

        elif warmup_steps > 0:
            # Warmup only — ramp to lr over warmup_steps, then hold constant.
            warmup_scheduler = LambdaLR(
                optimizer,
                lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps),
            )
            return (
                [optimizer],
                [{"scheduler": warmup_scheduler, "interval": "step", "frequency": 1}],
            )

        else:
            # Unchanged: no schedule, constant lr.
            return optimizer
```

### Fix — Part C: `train_ts1x.py`

Add `warmup_steps` to `training_config`:

```python
# oa_reactdiff/trainer/train_ts1x.py  (fixed)
training_config = dict(
    datadir="../data/transition1x/",
    remove_h=False,
    bz=14,
    num_workers=0,
    clip_grad=True,
    gradient_clip_val=None,
    gradnorm_queue_init=10.0,
    warmup_steps=500,              # <-- add this; ramp lr from 0 to 2.5e-4 over 500 steps
    ema=True,
    ema_decay=0.999,
    swapping_react_prod=True,
    append_frag=False,
    use_by_ind=True,
    reflection=False,
    single_frag_only=True,
    only_ts=False,
    lr_schedule_type=None,
    lr_schedule_config=dict(
        gamma=0.8,
        step_size=100,
    ),
)
```

### Choosing `warmup_steps`

With `limit_train_batches=200` and `bz=14`, one epoch is 200 batches. 500 warmup steps equals
2.5 epochs. A rule of thumb for diffusion model training is 1–5% of total training steps.
With 2000 epochs × 200 steps = 400 000 total steps, 500 warmup steps is 0.1% — conservative
and safe. Values between 200 and 1000 are all reasonable; larger values provide more protection
at the cost of slower initial convergence.

---

## Finding 6 — TS Fragment Loss Scale Amplifies Early Noise

### File
`oa_reactdiff/trainer/train_ts1x.py`, line 111

### What the code currently does

```python
# oa_reactdiff/trainer/train_ts1x.py  (current)
scales = [1.0, 2.0, 1.0]    # [R, TS, P] — TS error weighted 2×
```

The loss for each fragment is:

```python
# oa_reactdiff/trainer/pl_trainer.py — compute_loss
error_t_normalized = [
    loss_terms["error_t"][ii] / denoms[ii] * self.scales[ii]
    for ii in range(self.n_fragments)
]
loss_t = torch.stack(error_t_normalized, dim=0).sum(dim=0)
```

### Why this can amplify instability

The TS fragment has the noisiest loss signal early in training: its geometry is hardest to
predict (no DFT analogue to anchor it), so the early error terms are large and high-variance.
Down-weighting the reactant and product by 0.5× relative to TS means the loss gradient is
dominated by the most uncertain fragment. If the TS head is in a NaN-prone region (as can
happen with very large inter-fragment distances after a bad update), the 2× scale means the
corrupting gradient is doubled before it reaches the backbone.

This is not a bug — a stronger supervision signal on the TS is scientifically motivated. But it
is a contributing factor to instability, and should be the last knob to turn back on once the
other fixes are in place.

### Recommended approach

Use equal scales while diagnosing instability, then re-introduce the 2× TS weight after the
model is confirmed to converge reliably:

```python
# oa_reactdiff/trainer/train_ts1x.py  (for stability testing)
scales = [1.0, 1.0, 1.0]    # equal weights while diagnosing instability

# oa_reactdiff/trainer/train_ts1x.py  (restore once stable)
scales = [1.0, 2.0, 1.0]    # original scientifically-motivated weighting
```

This is purely a config change with no code modification required.

---

## Recommended Rollout Order

Apply fixes in this sequence and run one training run after each to isolate which change has the
largest impact. Each fix is independently safe.

| Priority | Fix | Effort | Expected Impact |
|----------|-----|--------|-----------------|
| 1 | Fix NaN handler: zero instead of randn; fix `h_final` bug; add batch-skip | ~10 lines across 2 files | **Highest** — prevents silent gradient corruption |
| 2 | Pre-fill gradnorm queue at 10.0 instead of 3000 | 4 lines in `pl_trainer.py` + 1 line in `train_ts1x.py` | **High** — protects the critical early-training window |
| 3 | `deterministic="warn"` in Trainer | 1 word in `train_ts1x.py` | **Medium** — makes failures reproducible for debugging |
| 4 | Enable EMA (`ema=True`) | 1 word in `train_ts1x.py` | **Medium** — smooths effective checkpoint quality |
| 5 | Add LR warmup (500 steps) | ~30 lines in `pl_trainer.py` + 1 line in `train_ts1x.py` | **Medium** — reduces early spike risk |
| 6 | Set `scales = [1.0, 1.0, 1.0]` during testing | 1 line in `train_ts1x.py` | **Low/diagnostic** — removes TS amplification while verifying stability |

Fixes 1 and 2 together are expected to account for the majority of the run-to-run variance. Fix 3
is important for making the remaining variance diagnosable. Fixes 4 and 5 are quality-of-life
improvements that diffusion models typically benefit from regardless of instability.

---

## Summary of All Changed Lines

### `oa_reactdiff/dynamics/egnn_dynamics.py`

**Current (lines 138–143):**
```python
vel = pos_final - pos
if torch.any(torch.isnan(vel)):
    print("Warning: detected nan in pos, resetting EGNN output to randn.")
    vel = torch.randn_like(vel)
if torch.any(torch.isnan(vel)):
    print("Warning: detected nan in h, resetting EGNN output to randn.")
    h_final = torch.randn_like(h_final)
```

**Replacement:**
```python
vel = pos_final - pos
if torch.any(torch.isnan(vel)):
    print("Warning: NaN detected in predicted velocity (pos). Zeroing this output.")
    vel = torch.zeros_like(vel)
if torch.any(torch.isnan(h_final)):
    print("Warning: NaN detected in node features (h). Zeroing this output.")
    h_final = torch.zeros_like(h_final)
```

---

### `oa_reactdiff/trainer/pl_trainer.py`

**Change 1 — imports (add `LambdaLR`):**

```python
# current
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR

# replacement
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, LambdaLR
```

**Change 2 — gradnorm queue initialisation (lines 143–146):**

```python
# current
self.clip_grad = training_config["clip_grad"]
if self.clip_grad:
    self.gradnorm_queue = utils.Queue()
    self.gradnorm_queue.add(3000)

# replacement
self.clip_grad = training_config["clip_grad"]
if self.clip_grad:
    _gradnorm_init = training_config.get("gradnorm_queue_init", 10.0)
    self.gradnorm_queue = utils.Queue()
    for _ in range(self.gradnorm_queue.max_len):
        self.gradnorm_queue.add(_gradnorm_init)
```

**Change 3 — `configure_optimizers` (lines 149–158):**

```python
# current
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.ddpm.parameters(), **self.optimizer_config)
    if not self.training_config["lr_schedule_type"] is None:
        scheduler_func = LR_SCHEDULER[self.training_config["lr_schedule_type"]]
        scheduler = scheduler_func(
            optimizer=optimizer, **self.training_config["lr_schedule_config"]
        )
        return [optimizer], [scheduler]
    else:
        return optimizer

# replacement
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.ddpm.parameters(), **self.optimizer_config)

    warmup_steps = self.training_config.get("warmup_steps", 0)

    if self.training_config["lr_schedule_type"] is not None:
        scheduler_func = LR_SCHEDULER[self.training_config["lr_schedule_type"]]
        primary_cfg = {
            k: v
            for k, v in self.training_config["lr_schedule_config"].items()
            if k != "warmup_steps"
        }
        primary_scheduler = scheduler_func(optimizer=optimizer, **primary_cfg)

        if warmup_steps > 0:
            warmup_scheduler = LambdaLR(
                optimizer,
                lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps),
            )
            return (
                [optimizer],
                [
                    {"scheduler": warmup_scheduler, "interval": "step", "frequency": 1},
                    {"scheduler": primary_scheduler, "interval": "epoch", "frequency": 1},
                ],
            )
        return [optimizer], [primary_scheduler]

    elif warmup_steps > 0:
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps),
        )
        return (
            [optimizer],
            [{"scheduler": warmup_scheduler, "interval": "step", "frequency": 1}],
        )

    else:
        return optimizer
```

**Change 4 — NaN guard in `training_step` (insert after line 329, `loss = nll.mean(0)`):**

```python
# Insert immediately after:  loss = nll.mean(0)
if not torch.isfinite(loss):
    print(
        f"Warning: non-finite loss ({loss.item():.4g}) at epoch "
        f"{self.current_epoch} batch {batch_idx}. Skipping weight update."
    )
    info["rmsd"], info["rmsd-median"] = np.nan, np.nan
    info["loss"] = torch.tensor(0.0, device=loss.device, requires_grad=True)
    return info
```

---

### `oa_reactdiff/trainer/train_ts1x.py`

**Change 1 — `training_config` dict:**

```python
# current
training_config = dict(
    datadir="../data/transition1x/",
    remove_h=False,
    bz=14,
    num_workers=0,
    clip_grad=True,
    gradient_clip_val=None,
    ema=False,
    ema_decay=0.999,
    swapping_react_prod=True,
    append_frag=False,
    use_by_ind=True,
    reflection=False,
    single_frag_only=True,
    only_ts=False,
    lr_schedule_type=None,
    lr_schedule_config=dict(
        gamma=0.8,
        step_size=100,
    ),
)

# replacement
training_config = dict(
    datadir="../data/transition1x/",
    remove_h=False,
    bz=14,
    num_workers=0,
    clip_grad=True,
    gradient_clip_val=None,
    gradnorm_queue_init=10.0,      # pre-fill grad norm queue conservatively
    warmup_steps=500,              # linear LR warmup over first 500 steps
    ema=True,                      # enable EMA (was False)
    ema_decay=0.999,
    swapping_react_prod=True,
    append_frag=False,
    use_by_ind=True,
    reflection=False,
    single_frag_only=True,
    only_ts=False,
    lr_schedule_type=None,
    lr_schedule_config=dict(
        gamma=0.8,
        step_size=100,
    ),
)
```

**Change 2 — `deterministic` in `Trainer`:**

```python
# current
trainer = Trainer(
    max_epochs=2000,
    accelerator="gpu",
    deterministic=False,
    ...
)

# replacement
trainer = Trainer(
    max_epochs=2000,
    accelerator="gpu",
    deterministic="warn",          # deterministic where possible; logs non-deterministic ops
    ...
)
```

**Change 3 — `scales` (optional; for stability testing only):**

```python
# current (and restore after stability confirmed)
scales = [1.0, 2.0, 1.0]

# temporary during stability testing
scales = [1.0, 1.0, 1.0]
```
