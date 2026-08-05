# Terminology

Project-specific and overloaded terms, one term per `##` section. Add new
terms as new sections in this file rather than new files — see
`docs/documentation_policy.md`.

## Fragment

### The problem

"Fragment" has two distinct meanings in chemistry and in this codebase, and they are conflated throughout.

**Correct meaning — molecular fragment:** a connected component within a single chemical state. When a bond breaks in a dissociation reaction, the product consists of two molecular fragments. This is what the `single_fragment` flag, `data['product']['fragments']`, and `xyz2mol.py` mean.

**Incorrect (overloaded) meaning — reaction state:** R, TS, and P are called "fragments" throughout the code, even though each is a complete molecular species (or set of species). The terms `fragment_names`, `fragments_nodes`, `fragments_masks`, `n_fragments`, and `n_frag_switch` all refer to the three reaction states, not to disconnected molecular components.

Preferred terms for R/TS/P: **state**, **species**, or **reaction component**. In the model's graph-theoretic view, **subgraph** or **subgraph type** are also acceptable.

---

### Where each meaning appears

#### Correct use (molecular fragment = connected component)

| Location | Symbol | Meaning |
|---|---|---|
| `data/transition1x/*.pkl` | `data[state]['fragments']` | List of atom-index lists per connected component (e.g. `[[0,1,4,5], [2,3]]` for a two-fragment product) |
| `dataset/transition1x.py:54` | `single_fragment` | Bool flag: all three states (R, TS, P) are single connected-component molecules |
| `dataset/transition1x.py:54,57` | `single_frag_inds` | Indices of reactions where all states are single-component |
| `utils/xyz2mol.py` | `Chem.GetMolFrags()`, `allow_charged_fragments` | RDKit API — connected-component sense |

#### Incorrect use (reaction state = R, TS, or P)

| Location | Symbol | What it actually means |
|---|---|---|
| `dynamics/_base.py:13,30,55` | `fragment_names` | List of state names, e.g. `["R", "TS", "P"]` |
| `dynamics/egnn_dynamics.py:17,34,49` | `fragment_names` | Same |
| `dynamics/confidence.py:25,41,59` | `fragment_names` | Same |
| `trainer/train_ts1x.py:102` | `fragment_names = ["R", "TS", "P"]` | Three reaction states |
| `trainer/pl_trainer.py:64,126,458` | `fragment_names`, `n_fragments` | Three reaction states |
| `diffusion/en_diffusion.py:49,76,477–480` | `fragment_names`, `fragments_nodes`, `fragments_masks` | Per-state node counts and batch masks |
| `diffusion/en_diffusion.py:482` | `n_frag_switch` | Per-node integer indicating which state (R=0, TS=1, P=2) the atom belongs to |
| `utils/_graph_tools.py:18,21,40,51,63,67,85,89` | `fragments_nodes`, `n_frag_switch`, `fragments_masks`, docstrings | State-level graph quantities |
| `analyze/rmsd.py:79,88` | `fragments_nodes` | List of per-state atom-count tensors |
| `evaluate/evaluate_ts_w_rp.py:98,130,140` | `fragments_nodes`, `pad_fragments` | State-level |
| `dataset/base_dataset.py:26,41,60,61` | `n_fragment` | Number of states (3 for R/TS/P) |
| `dataset/transition1x.py:85,131` | `n_fragments`, `pad_fragments` | Number of states; padding adds dummy zero-atom states |
| `dataset/qm9.py:85,94,126,138,213,226` | `n_fragments`, `pad_fragments` | Same pattern |
| `model/block.py:265`, `model/egnn.py:131` | "subgraph (i.e., fragment) level equivariance" | Equivariance is enforced per state (R/TS/P), not per molecular fragment |
| `tests/dynamics/test_egnn_dynamics.py:62,100,105,110` | `fragment_names`, `fragments_nodes`, `fragments_masks` | State-level |
| `tests/dynamics/test_switch_fragments.py:48,84,96,112,157` | `fragment_names`, `fragments_nodes`, `test_switch_fragments` | Tests switching R and P states |
| `tests/model/test_subgraphs.py:250` | "Change the geometry of one fragment" | Means changing R or P |

#### Specifically ambiguous / collision point

`n_frag_switch` (per-node, integer 0/1/2 mapping each atom to its state) vs. `data[state]['fragments']` (per-state, list of connected-component atom indices) — both use "frag" but mean entirely different things.

---

### Impact on new code

When modifying or extending this codebase:

- `fragments_nodes` — list of length 3, each element is a 1-D tensor of per-sample atom counts for one **state** (R, TS, P). Rename to `states_nodes` in new code.
- `n_frag_switch` — per-atom integer index indicating **state** membership. Rename to `state_idx` or `species_idx`.
- `fragment_names` — ordered list of **state** names. Rename to `state_names`.
- `n_fragments` / `n_fragment` — number of **states** (= 3 for R/TS/P). Rename to `n_states`.
- `pad_fragments` — number of dummy zero-atom states appended for padding. Rename to `pad_states`.
- `single_fragment`, `single_frag_inds`, `data[state]['fragments']` — these already use "fragment" correctly; keep as-is.
