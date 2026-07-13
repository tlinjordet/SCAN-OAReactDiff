# Transition1X Position Preprocessing: Findings and Open Questions

**Terminology note:** Throughout the codebase, "fragment" is overloaded — it means both "reaction state (R/TS/P)" and "connected molecular component." This document uses "state" for R/TS/P and "molecular fragment" or "connected component" for the chemistry sense. See `docs/terminology_fragment_misuse.md` for the full catalogue of misuse.

---

## Context and goal

OA-ReactDiff is trained on a pkl-format version of the Transition1X dataset. The paper claims the model requires neither atom mapping nor fragment alignment. Understanding exactly what the pkl files contain — and where/how alignment and ordering were broken — is necessary before modifying the data pipeline for new experiments.

This document records verified findings from code and data inspection and maps them against the paper's prose.

---

## Paper claim (verbatim)

> "Transition1x dataset has atom mapping and aligned product molecules as constructed. To showcase the capability of OA-ReactDiff not relying on the atom mapping and fragment alignment, we intentionally swapped the atom ordering and broke the alignment in Transition1x beforehand and verified that OA-ReactDiff functions well without these pre-processing requirements (Supplementary Figure 1)."

---

## What the pkl data actually contains

### Data files

| File | Samples | Extra keys (R/P only) |
|---|---|---|
| `oa_reactdiff/data/transition1x/train.pkl` | 10073 | — |
| `oa_reactdiff/data/transition1x/train_addprop.pkl` | 10073 | `smi`, `ediff`, `uff_positions` |
| `oa_reactdiff/data/transition1x/valid_addprop.pkl` | 10073 | same as train_addprop |

Top-level keys in each pkl: `reactant`, `transition_state`, `product`, `single_fragment`, `use_ind`. Each fragment dict: `num_atoms`, `charges`, `fragments`, `positions`, `rxn`, `wB97x_6-31G(d).energy`, `wB97x_6-31G(d).atomization_energy`, `wB97x_6-31G(d).forces`, `formula`.

`uff_positions` (UFF-optimized, exactly centered geometries) exists in `train_addprop.pkl` and `valid_addprop.pkl` for R and P but **not** for TS. It is never read anywhere in the codebase — `position_key` defaults to `"positions"` in every calling context.

### Atom ordering

Verified across 500 single-fragment samples: **the `charges` (atomic number) array is in the same order for R and P in every sample**. Atom mapping has NOT been broken at the pkl level. The `swapping_react_prod=True` flag used in training is where the paper's "swapped atom ordering" claim is implemented — it doubles the dataset by swapping R↔P, which changes which molecule's atom ordering appears as "reactant" and which as "product."

### Spatial alignment ("breaking the alignment")

The original Transition1X dataset from climbing-image NEB stores R, TS, P in a **shared coordinate frame** (all three structures come from the same NEB calculation). The OA-ReactDiff pkl files have each molecule in **its own independent DFT reference frame** — the shared NEB frame has been discarded. This was done as a one-time preprocessing step **before** the pkl files were created; the code that performed this transformation is not in the repository.

Evidence from data: DFT centroid norms for single-fragment samples reach up to ~0.30 Å (not pre-centered), and centroids differ independently between R, TS, and P for the same reaction. The dataset class applies `pos - mean(pos)` independently to each fragment at load time (`center=True`, default).

Procrustes disparity (orthogonal, R aligned to P, single-fragment samples): mean **0.84 Å**, range 0.42–1.61 Å. This is physically expected — R and P are genuinely different geometries, not rigid transforms of each other.

### Multi-fragment samples (~25%)

2548 of 10073 samples (25.3%) are dissociation reactions where the product splits into two separate molecular fragments. The `fragments` key encodes this, e.g.:

```python
data['reactant']['fragments'][1]  # [[0, 1, 2, 3, 4, 5, 6]]
data['product']['fragments'][1]   # [[0, 1, 4, 5, 6], [2, 3]]
```

These produce physically impossible-looking product geometries (inter-atomic distances up to 55 Å) but are not corrupted — they represent legitimate bond-breaking reactions. All 2548 have `single_fragment = 0`. None of the `single_fragment = 1` samples have this issue.

---

## Where the paper's claims map to code

### "Swapped atom ordering"

**`transition1x.py:64–71`** — the `swapping_react_prod` flag:

```python
FRAG_MAPPING = {
    "reactant": "product",
    "transition_state": "transition_state",
    "product": "reactant",
}
# ...
if swapping_react_prod:
    mapped_val = data_duplicated[mapped_k][v]
    self.raw_dataset[k][v] += [mapped_val[ii] for ii in single_frag_inds]
```

For every reaction R→TS→P, this appends P→TS→R (treating the product as the "reactant" input and vice versa). **`train_ts1x.py` line 85** sets `swapping_react_prod=True`. This is the mechanism the paper calls "swapping atom ordering" — the atom ordering of the product becomes the ordering of the reactant in the augmented sample.

The TS maps to itself (`"transition_state": "transition_state"`), so the TS geometry is unchanged but is now conditioned on R=product and P=reactant.

**`reflection` option** (`transition1x.py:73–78`) provides a further 2× augmentation by flipping the z-coordinate. Not enabled in `train_ts1x.py` (`reflection=False`).

### "Broke the alignment"

**No code in the repository does this.** The pkl files already contain misaligned positions (each molecule in its own DFT reference frame). This was done as an offline preprocessing step whose code is not provided.

The dataset class re-applies centering at load time (`base_dataset.py:215–218`):
```python
if self.center:
    self.data[f"pos_{idx}"] = [
        pos - torch.mean(pos, dim=0) for pos in self.data[f"pos_{idx}"]
    ]
```
This centers each of R, TS, P independently — discarding whatever relative spatial relationship remains between fragments.

### "Not relying on atom mapping" — RMSD evaluation

**`analyze/rmsd.py:30–51`** uses pymatgen's `BruteForceOrderMatcher`, `GeneticOrderMatcher`, or `HungarianOrderMatcher` to compute RMSD between generated and reference TS structures. These algorithms search over atom permutations, so the RMSD metric itself does not assume consistent atom ordering. This is consistent with the paper's claim — evaluation also does not require atom mapping.

---

## File relationships: train.pkl, train_addprop.pkl, valid_addprop.pkl

All three pkl files contain **all 10073 samples** (same `rxn` IDs, same order). They differ as follows:

- `train.pkl` and `train_addprop.pkl` are **bitwise identical in positions** (max diff = 0). `train_addprop.pkl` adds extra keys: `smi`, `ediff`, and `uff_positions` (R and P only).
- `valid_addprop.pkl` has the **same keys as `train_addprop.pkl`** but different positions.

The `train.pkl` positions were derived from `valid_addprop.pkl` by applying an **independent random rotation around the coordinate origin** to each reaction state (and to each disconnected component within multi-fragment products):

```
train_i = valid_i @ R_i
```

where `R_i ∈ SO(3)` is a unique proper rotation (det = +1; no reflections) per state per sample. This is a pure rotation around the origin — no translation, no centering. Reconstruction error: max < 4×10⁻⁷ Å (machine precision) across all 7516 single-fragment samples.

Consequence: **centroid norms are preserved exactly** (`|train_c_i| = |valid_c_i|`), so both files have identical centroid-norm distributions (mean 0.025 Å, max 0.298 Å), but individual centroid directions differ.

For multi-fragment products (2557 samples with `single_fragment=0`): each disconnected component was rotated independently, explaining why whole-product Procrustes disparity is high but per-component disparity is machine-precision zero.

**Clarification on RMSD numbers.** Comparing the *same state* between the two files (e.g., reactant in valid_addprop vs reactant in train) gives a direct position difference of up to 8.9 Å — they look completely different. After recovering `R_i` via orthogonal Procrustes and applying it, the residual is < 4×10⁻⁷ Å (machine precision), confirming the positions are geometrically identical up to the rotation. The 0.539 Å figure is an unrelated quantity: the per-atom RMSD between R and TS *within the same file* after optimal alignment — a measure of the structural change along the reaction coordinate, not a file comparison.

**Neither file has R, TS, P in a shared NEB reference frame.** Within valid_addprop.pkl: comparing R to TS after optimal alignment gives mean 0.539 Å RMSD (actual structural change). Without alignment, that rises to mean 2.41 Å — meaning ~76% of the apparent R–TS difference is orientation, not structure. Both files show essentially the same picture; valid_addprop.pkl retains marginally more inter-state orientation consistency (ratio 0.241 vs 0.236 in train), consistent with it being the pre-rotation source, but neither preserves the original NEB coordinate frame.

The small inter-state centroid distances (mean 0.048 Å in valid_addprop, 0.046 Å in train) reflect only that all DFT geometries happen to be placed close to the coordinate origin — not any shared spatial frame. See `docs/transition1x_pkl_comparison_walkthrough.md` for reproducible code verifying all these numbers.

This is the offline "breaking alignment" step described in the paper: `valid_addprop.pkl` is the source; `train.pkl` is the result after applying one independent random rotation per state. The code that performed this transformation is not in the repository.

---

## Inconsistencies between paper and code

### 1. Split labeling: "test" vs "valid"

The paper describes a 9000/1073 partition as training+validation / test. All three pkl files contain all 10073 samples — the split is encoded by the `use_ind` field (a list of 9000 indices). The 1073 samples not in `use_ind` are the held-out test set; the file name `valid_addprop.pkl` is misleading (it is not a validation-only split).

### 2. Actual training set size: 6733, not 9000

`train_ts1x.py` applies `single_frag_only=True` and `use_by_ind=True` simultaneously. Their intersection in `train.pkl` is **6733 samples** — not 9000. The `use_ind` field alone selects 9000; the `single_fragment` filter then removes 2267 multi-fragment reactions from that set. The paper reports results on 9000 reactions (unclear whether this refers to the `use_ind` set or the final filtered training set).

### 3. TS always uses DFT positions; R/P accept `position_key`

In `ProcessedTS1x.__init__`, TS is processed without `position_key`:
```python
self.process_molecules("transition_state", n_samples, idx=1)          # always "positions"
self.process_molecules("reactant", n_samples, idx=0, position_key=position_key)
self.process_molecules("product", n_samples, idx=2, position_key=position_key)
```
This asymmetry exists but has no practical effect because `position_key` always defaults to `"positions"` in every calling context.

---

## Open questions for new experiments

1. **Does the 6733 vs 9000 sample discrepancy affect published metrics?** If the paper trained on 9000 (`use_by_ind=True`, `single_frag_only=False`), then 2267 training samples had multi-fragment products — the model was exposed to physically fragmented geometries.

3. **Would realigning R and P (e.g., RMSD-aligning P onto R before training) improve performance?** The object-aware SE(3) design discards relative R–P spatial context; it is possible a model trained on aligned data would be more accurate for reactions where geometry transfer matters.

4. **Can `uff_positions` improve training?** UFF geometries are exactly centered and physically reasonable (no fragmentation artefacts). They would give consistent starting geometries for the model.

---

## Session status

**Completed:**
- Verified pkl data structure and all field types
- Confirmed multi-fragment issue (25.3% of data) and its relationship to `single_fragment` flag
- Verified atom-ordering (charges) is preserved between R and P in pkl files
- Mapped paper claims to specific code locations
- Identified inconsistencies between paper prose and implemented code
- Fully characterized the `valid_addprop.pkl` → `train.pkl` transformation: independent SO(3) rotation per fragment (per connected component for multi-fragment products), exact to machine precision
- Confirmed `train.pkl` and `train_addprop.pkl` are bitwise identical in positions
- Confirmed all three pkl files contain all 10073 samples; split is via `use_ind`

**Next step:** Read the published paper and SI (to be added to `scratch/papers/`) to resolve the 6733 vs 9000 training sample discrepancy and confirm the test/validation split convention. Then decide whether to clean/restructure the data pipeline or add experiment scripts on top of the existing code.
