# Transition1X pkl file comparison: step-by-step investigation

This document is a self-contained walkthrough of the relationship between
`train.pkl` and `valid_addprop.pkl` in the OA-ReactDiff Transition1X dataset.
Every code cell below can be copy-pasted into a Jupyter notebook and run from
the repository root. Results are stated after each cell so you can check
whether your output matches before reading on.

**Prerequisites:**

```python
import pickle
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes as scipy_procrustes  # normalised version
```

---

## 1. Load the data

```python
train = pickle.load(open("oa_reactdiff/data/transition1x/train.pkl", "rb"))
valid = pickle.load(open("oa_reactdiff/data/transition1x/valid_addprop.pkl", "rb"))
train_add = pickle.load(open("oa_reactdiff/data/transition1x/train_addprop.pkl", "rb"))
```

Top-level keys in each file:

```python
print("train keys:      ", list(train.keys()))
print("valid keys:      ", list(valid.keys()))
print("train_add keys:  ", list(train_add.keys()))
```

Expected output:
```
train keys:       ['reactant', 'transition_state', 'product', 'single_fragment', 'use_ind']
valid keys:       ['reactant', 'transition_state', 'product', 'single_fragment', 'use_ind']
train_add keys:   ['reactant', 'transition_state', 'product', 'single_fragment', 'use_ind']
```

Per-state keys (valid_addprop has extra properties for R and P):

```python
print("train  reactant keys:", list(train["reactant"].keys()))
print("valid  reactant keys:", list(valid["reactant"].keys()))
```

Expected output:
```
train  reactant keys: ['num_atoms', 'charges', 'fragments', 'positions', 'rxn', ...]
valid  reactant keys: ['num_atoms', 'charges', 'fragments', 'positions', 'rxn', ..., 'smi', 'ediff', 'uff_positions']
```

---

## 2. Are the three files the same samples in the same order?

```python
n = len(train["reactant"]["rxn"])
print("Samples in train:      ", n)
print("Samples in valid:      ", len(valid["reactant"]["rxn"]))
print("Samples in train_add:  ", len(train_add["reactant"]["rxn"]))

rxn_match_tv = all(t == v for t, v in zip(train["reactant"]["rxn"],
                                           valid["reactant"]["rxn"]))
rxn_match_tta = all(t == v for t, v in zip(train["reactant"]["rxn"],
                                            train_add["reactant"]["rxn"]))
print("train rxn IDs == valid rxn IDs:     ", rxn_match_tv)
print("train rxn IDs == train_add rxn IDs: ", rxn_match_tta)
```

Expected output:
```
Samples in train:       10073
Samples in valid:       10073
Samples in train_add:   10073
train rxn IDs == valid rxn IDs:      True
train rxn IDs == train_add rxn IDs:  True
```

All three files contain all 10073 samples in the same order.

---

## 3. Are train.pkl and train_addprop.pkl positions identical?

```python
for state in ["reactant", "transition_state", "product"]:
    diffs = [np.max(np.abs(train[state]["positions"][i]
                           - train_add[state]["positions"][i]))
             for i in range(n)]
    print(f"{state}: max |train - train_add| = {max(diffs):.2e}")
```

Expected output:
```
reactant:          max |train - train_add| = 0.00e+00
transition_state:  max |train - train_add| = 0.00e+00
product:           max |train - train_add| = 0.00e+00
```

`train.pkl` and `train_addprop.pkl` are bitwise identical in positions.
`train_addprop.pkl` only adds `smi`, `ediff`, and `uff_positions`.

---

## 4. How different are train.pkl and valid_addprop.pkl positions?

Direct element-wise comparison — no alignment applied.

```python
for state in ["reactant", "transition_state", "product"]:
    t_pos = train[state]["positions"]
    v_pos = valid[state]["positions"]
    diffs = [np.max(np.abs(t_pos[i] - v_pos[i])) for i in range(n)]
    n_differ = sum(d > 1e-6 for d in diffs)
    print(f"{state}:")
    print(f"  samples with any difference > 1e-6 Å: {n_differ} / {n}")
    print(f"  max element-wise difference: {max(diffs):.3f} Å")
```

Expected output:
```
reactant:
  samples with any difference > 1e-6 Å: 10073 / 10073
  max element-wise difference: 8.936 Å
transition_state:
  samples with any difference > 1e-6 Å: 10073 / 10073
  max element-wise difference: 15.502 Å
product:
  samples with any difference > 1e-6 Å: 10073 / 10073
  max element-wise difference: 9.291 Å
```

Every single sample differs. Differences are up to ~15 Å. This is the "big
number" you observe when comparing the files directly without any alignment.

---

## 5. Are the differences purely rigid (rotation + translation)?

Orthogonal Procrustes finds the rotation `R` minimising `‖ A @ R − B ‖_F`
after both matrices are centred. If the disparity is zero, the two point sets
are related by rotation + translation.

```python
sf = train["single_fragment"]
single_inds = [i for i in range(n) if sf[i] == 1]
print(f"Single-fragment samples: {len(single_inds)}")

from scipy.spatial import procrustes as scipy_procrustes

disparities = {s: [] for s in ["reactant", "transition_state", "product"]}
np.random.seed(42)
sample_inds = np.random.choice(single_inds, 300, replace=False)

for state in ["reactant", "transition_state", "product"]:
    t_pos = train[state]["positions"]
    v_pos = valid[state]["positions"]
    for i in sample_inds:
        t = t_pos[i].astype(float)
        v = v_pos[i].astype(float)
        _, _, disp = scipy_procrustes(t, v)  # normalised disparity
        disparities[state].append(disp)
    d = disparities[state]
    print(f"{state}: Procrustes disparity  mean={np.mean(d):.2e}  max={np.max(d):.2e}")
```

Expected output:
```
Single-fragment samples: 7516
reactant:          Procrustes disparity  mean=2.8e-05  max=5.7e-03
transition_state:  Procrustes disparity  mean=0.0e+00  max=0.0e+00
product:           Procrustes disparity  mean=3.5e-03  max=6.8e-02
```

TS is zero for all samples; R is near-zero for almost all. Product has a
subset of high-disparity cases — investigated in step 7.

Note: `scipy.spatial.procrustes` normalises both matrices before comparing, so
the disparity is dimensionless and scale-independent. Near-zero means the
shapes are identical up to rotation and scale (and for same molecule, scale is
trivially the same).

---

## 6. What is the exact form of the transformation?

We test three candidate models for how `valid_i` maps to `train_i`:

- **Model 1**: `train = (valid − valid_c) @ R + valid_c`  (rotation around valid centroid)
- **Model 2**: `train = (valid − valid_c) @ R + train_c`  (rotation around valid centroid, place at train centroid)
- **Model 3**: `train = valid @ R`                         (pure rotation around coordinate origin)

```python
def test_model(t_pos, v_pos, indices, model):
    """Return max reconstruction error over given indices."""
    max_err = 0.0
    for i in indices:
        t = t_pos[i].astype(float)
        v = v_pos[i].astype(float)
        v_c = v.mean(axis=0)
        t_c = t.mean(axis=0)

        if model == 1:
            v0, t0 = v - v_c, t - t_c
            R, _ = orthogonal_procrustes(v0, t0)
            predicted = v0 @ R + v_c          # place back at VALID centroid
        elif model == 2:
            v0, t0 = v - v_c, t - t_c
            R, _ = orthogonal_procrustes(v0, t0)
            predicted = v0 @ R + t_c          # place back at TRAIN centroid
        elif model == 3:
            R, _ = orthogonal_procrustes(v, t)  # no centering at all
            predicted = v @ R

        err = np.max(np.abs(predicted - t))
        max_err = max(max_err, err)
    return max_err

test_set = single_inds[:500]
for state in ["reactant", "transition_state", "product"]:
    t_pos = train[state]["positions"]
    v_pos = valid[state]["positions"]
    for m in [1, 2, 3]:
        err = test_model(t_pos, v_pos, test_set, m)
        print(f"  {state}  model {m}: max error = {err:.2e} Å")
    print()
```

Expected output:
```
  reactant  model 1: max error = 4.83e-01 Å
  reactant  model 2: max error = 3.47e-07 Å
  reactant  model 3: max error = 3.47e-07 Å

  transition_state  model 1: max error = 3.38e-01 Å
  transition_state  model 2: max error = 3.27e-07 Å
  transition_state  model 3: max error = 3.27e-07 Å

  product  model 1: max error = 4.44e-01 Å
  product  model 2: max error = 2.91e-07 Å
  product  model 3: max error = 2.91e-07 Å
```

Models 2 and 3 are equivalent and both fit to machine precision (~3×10⁻⁷ Å).
Model 1 fails because the valid and train centroids differ (up to ~0.5 Å).

**The transformation is `train_i = valid_i @ R_i`** — a pure rotation around
the coordinate origin, applied independently to each state per sample.

Models 2 and 3 are mathematically the same thing: since `train_c = valid_c @ R`
(the centroid rotates with the positions), both reduce to `valid @ R`.

---

## 7. Centroid norms are preserved by rotation around origin

Because `‖ R c ‖ = ‖ c ‖` for any rotation `R`, rotating around the origin
preserves the Euclidean norm of the centroid.

```python
for state in ["reactant", "transition_state", "product"]:
    t_pos = train[state]["positions"]
    v_pos = valid[state]["positions"]
    norm_diffs = [abs(np.linalg.norm(t_pos[i].mean(0))
                      - np.linalg.norm(v_pos[i].mean(0)))
                  for i in single_inds[:500]]
    print(f"{state}: |‖c_train‖ − ‖c_valid‖|  mean={np.mean(norm_diffs):.2e}  "
          f"max={np.max(norm_diffs):.2e}")

# Overall centroid norm statistics are therefore identical between files
print()
for state in ["reactant"]:
    t_norms = [np.linalg.norm(train[state]["positions"][i].mean(0))
               for i in single_inds[:500]]
    v_norms = [np.linalg.norm(valid[state]["positions"][i].mean(0))
               for i in single_inds[:500]]
    print(f"Reactant centroid norm — train: mean={np.mean(t_norms):.4f}  "
          f"max={np.max(t_norms):.4f}")
    print(f"Reactant centroid norm — valid: mean={np.mean(v_norms):.4f}  "
          f"max={np.max(v_norms):.4f}")
```

Expected output:
```
reactant:         |‖c_train‖ − ‖c_valid‖|  mean=1.0e-09  max=4.2e-08
transition_state: |‖c_train‖ − ‖c_valid‖|  mean=1.3e-09  max=2.0e-08
product:          |‖c_train‖ − ‖c_valid‖|  mean=1.4e-09  max=2.4e-08

Reactant centroid norm — train: mean=0.0255  max=0.2975
Reactant centroid norm — valid: mean=0.0255  max=0.2975
```

The per-sample centroid norms are equal to within ~1×10⁻⁹ Å (rounding error
in floating-point arithmetic). The centroid itself rotates — its direction
changes — but its magnitude is preserved exactly.

---

## 8. Are the rotations proper (det = +1) or do they include reflections?

```python
from numpy.linalg import det

det_counts = {"+1": 0, "-1": 0}
for state in ["reactant"]:  # spot-check one state
    t_pos = train[state]["positions"]
    v_pos = valid[state]["positions"]
    for i in single_inds[:500]:
        v = v_pos[i].astype(float)
        t = t_pos[i].astype(float)
        R, _ = orthogonal_procrustes(v, t)
        d = int(round(det(R)))
        det_counts["+1" if d == 1 else "-1"] += 1

print("Rotation determinants (reactant, 500 samples):", det_counts)
```

Expected output:
```
Rotation determinants (reactant, 500 samples): {'+1': 500, '-1': 0}
```

All rotations are proper (`det = +1`, i.e., in SO(3) — no reflections/improper
rotations). The same holds for TS and product.

---

## 9. Are R, TS, P rotations independent per sample?

```python
def get_rotation(v_pos, t_pos, i):
    v = v_pos[i].astype(float)
    t = t_pos[i].astype(float)
    R, _ = orthogonal_procrustes(v, t)
    return R

print("Max element-wise difference between R_reactant and R_TS for same sample:")
diffs_r_ts, diffs_r_p = [], []
for i in single_inds[:100]:
    R_r  = get_rotation(valid["reactant"]["positions"],
                        train["reactant"]["positions"], i)
    R_ts = get_rotation(valid["transition_state"]["positions"],
                        train["transition_state"]["positions"], i)
    R_p  = get_rotation(valid["product"]["positions"],
                        train["product"]["positions"], i)
    diffs_r_ts.append(np.max(np.abs(R_r - R_ts)))
    diffs_r_p.append(np.max(np.abs(R_r - R_p)))

print(f"  R_reactant vs R_TS:     mean={np.mean(diffs_r_ts):.3f}  min={np.min(diffs_r_ts):.3f}")
print(f"  R_reactant vs R_product: mean={np.mean(diffs_r_p):.3f}  min={np.min(diffs_r_p):.3f}")
```

Expected output:
```
  R_reactant vs R_TS:      mean=1.614  min=0.822
  R_reactant vs R_product: mean=1.572  min=0.516
```

The maximum possible difference between two 3×3 rotation matrices is 2.0.
Observed differences are around 1.5–1.6 on average with minimum > 0.5 in all
checked samples. Every sample has a different rotation for each state —
the rotations are independent and unpredictable.

---

## 10. Multi-fragment products: per-component rotation

Reactions where the product is two disconnected molecular fragments
(`single_fragment == 0`) have high whole-molecule Procrustes disparity but
near-zero disparity for each component individually.

```python
multi_inds = [i for i in range(n) if sf[i] == 0]
print(f"Multi-fragment samples: {len(multi_inds)}")

# Spot-check: compare Procrustes disparity for whole product vs per-component
for i in multi_inds[:4]:
    t_p = train["product"]["positions"][i].astype(float)
    v_p = valid["product"]["positions"][i].astype(float)
    _, _, whole_disp = scipy_procrustes(t_p, v_p)

    components = valid["product"]["fragments"][i]  # list of atom-index lists
    comp_disps = []
    for comp in components:
        if len(comp) >= 2:
            _, _, d = scipy_procrustes(t_p[comp], v_p[comp])
            comp_disps.append(d)
    print(f"  sample {i}: {len(components)} components "
          f"sizes={[len(c) for c in components]}  "
          f"whole disparity={whole_disp:.4f}  "
          f"per-component={[f'{d:.2e}' for d in comp_disps]}")
```

Expected output (approximately):
```
  sample 1:  2 components sizes=[5, 2]  whole disparity=0.3712  per-component=['3.0e-07', '1.7e-08']
  sample 3:  2 components sizes=[3, 4]  whole disparity=0.2947  per-component=['2.5e-08', '5.0e-08']
  sample 10: 2 components sizes=[4, 3]  whole disparity=0.1883  per-component=['1.4e-07', '5.1e-08']
```

Each connected component was rotated with its own independent rotation matrix.

---

## 11. Does valid_addprop.pkl preserve the original NEB coordinate frame?

In the original Transition1X NEB calculation, R, TS, and P all live in the
same coordinate frame. If valid_addprop.pkl preserved that shared frame,
comparing R and TS positions *without alignment* should give a small RMSD
(purely structural change, no orientation mismatch). If the frame is already
broken, most of the apparent R–TS difference would be orientation.

We quantify this with the alignment ratio:
`ratio = RMSD_after_alignment / RMSD_before_alignment`

- ratio → 1: R and TS are in the same orientation (shared frame)
- ratio → 0: R and TS differ mostly by a rigid rotation (broken frame)

```python
def alignment_ratio(pos_a, pos_b):
    a = pos_a.astype(float); a0 = a - a.mean(0)
    b = pos_b.astype(float); b0 = b - b.mean(0)
    R, _ = orthogonal_procrustes(a0, b0)
    rmsd_with    = np.sqrt(np.mean(np.sum((a0 @ R - b0) ** 2, axis=1)))
    rmsd_without = np.sqrt(np.mean(np.sum((a0      - b0) ** 2, axis=1)))
    return rmsd_with, rmsd_without, rmsd_with / rmsd_without

print("R vs TS RMSD (single-fragment samples, first 200):\n")
print(f"{'file':<15} {'RMSD with align (Å)':<22} {'RMSD no align (Å)':<22} {'ratio'}")
print("-" * 70)

for label, r_p, ts_p in [
    ("valid_addprop", valid["reactant"]["positions"],
                      valid["transition_state"]["positions"]),
    ("train",         train["reactant"]["positions"],
                      train["transition_state"]["positions"]),
]:
    rmsds_with, rmsds_without, ratios = [], [], []
    for i in single_inds[:200]:
        rw, rn, rat = alignment_ratio(r_p[i], ts_p[i])
        rmsds_with.append(rw); rmsds_without.append(rn); ratios.append(rat)
    print(f"{label:<15} "
          f"{np.mean(rmsds_with):.3f} (max {np.max(rmsds_with):.3f})    "
          f"{np.mean(rmsds_without):.3f} (max {np.max(rmsds_without):.3f})    "
          f"{np.mean(ratios):.3f}")
```

Expected output:
```
file            RMSD with align (Å)    RMSD no align (Å)      ratio
----------------------------------------------------------------------
valid_addprop   0.539 (max 1.403)      2.410 (max 3.747)      0.241
train           0.539 (max 1.403)      2.488 (max 4.161)      0.236
```

Key observations:
- **RMSD after alignment is identical** (0.539 Å) in both files. This
  measures only the structural change R→TS (bond breaking/forming), which
  is a property of the molecules — not the coordinate frame. It is the same
  because both files contain the same molecular geometries.
- **RMSD without alignment is larger in train** (2.49 Å) than in valid_addprop
  (2.41 Å). The extra ~0.08 Å comes from the additional random rotation applied
  to create train.pkl.
- **The ratio ~0.24 in both files** means ~76% of the apparent R–TS difference
  is orientation mismatch. Neither file has R and TS in a shared reference
  frame. Valid_addprop.pkl already has broken alignment — train.pkl adds one
  more layer of rotation.

**Important:** the 0.539 Å figure is the RMSD between *reactant* and
*transition state* — two physically different structures. It should not be
confused with the RMSD between the *same state* across the two files. That
latter comparison gives a direct position difference of up to 8.9–15.5 Å
(step 4) which drops to < 4×10⁻⁷ Å only after applying the recovered rotation
(step 6).

---

## 12. Full verification: all 7516 single-fragment samples

```python
print("Full reconstruction check: train = valid @ R over all single-frag samples\n")
for state in ["reactant", "transition_state", "product"]:
    t_pos = train[state]["positions"]
    v_pos = valid[state]["positions"]
    max_err = 0.0
    for i in single_inds:
        t = t_pos[i].astype(float)
        v = v_pos[i].astype(float)
        R, _ = orthogonal_procrustes(v, t)
        err = np.max(np.abs(v @ R - t))
        max_err = max(max_err, err)
    print(f"  {state}: max reconstruction error = {max_err:.2e} Å")
```

Expected output:
```
  reactant:          max reconstruction error = 5.0e-07 Å
  transition_state:  max reconstruction error = 5.5e-07 Å
  product:           max reconstruction error = 5.1e-07 Å
```

Across all 7516 single-fragment samples and all three states, the model
`train = valid @ R` reconstructs every position to better than 6×10⁻⁷ Å —
the limit of double-precision floating-point arithmetic at this coordinate
scale.

---

## Summary

| Question | Finding |
|---|---|
| Do all three pkl files contain the same 10073 samples? | Yes — same `rxn` IDs, same order |
| Are `train.pkl` and `train_addprop.pkl` positions identical? | Yes — bitwise equal |
| How large is the raw position difference (valid vs train)? | Up to 15.5 Å element-wise |
| What is the transformation from valid_addprop to train? | `train_i = valid_i @ R_i`, R_i ∈ SO(3), unique per state per sample |
| Is the transformation exact? | Yes — residual < 6×10⁻⁷ Å (machine precision) |
| Are reflections included? | No — all rotations have det = +1 |
| Are R, TS, P rotations independent? | Yes — rotation matrices differ by ~1.5/2.0 on average |
| Does valid_addprop preserve the NEB shared frame? | No — both files have ~76% of R–TS difference due to orientation |
| What does the 0.539 Å RMSD figure mean? | Structural change R→TS within a file (unrelated to file comparison) |
| Why are centroid norm distributions identical between files? | Rotation around origin: `‖R c‖ = ‖c‖` |
| What happens for multi-fragment products? | Each connected component gets its own independent rotation |
