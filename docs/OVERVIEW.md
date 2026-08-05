# Documentation overview

Index of every active document in `docs/`. Read this first; open only the
row(s) relevant to the task at hand rather than the whole directory. Rules
for maintaining this index and the files it lists are in
`docs/documentation_policy.md` — update both together.

## Meta

| File | Scope | Status |
|---|---|---|
| `docs/documentation_policy.md` | Rules for this documentation and task-tracking system itself: prose style, file organization, archiving, `todo.org` conventions. | active |
| `docs/OVERVIEW.md` | This index. | active |

## Reference

| File | Scope | Status |
|---|---|---|
| `docs/architecture.md` | The four runtime layers (diffusion, dynamics, GNN backbone, data/training), key data structures, atom type encoding, pretrained checkpoint. | active |
| `docs/terminology.md` | Glossary of project-specific and overloaded terms, one `##` section per term. Currently: "Fragment". | active |

## Environment and infrastructure

| File | Scope | Status |
|---|---|---|
| `docs/apptainer_image.md` | Design of `environment/apptainer/oa_reactdiff.def`: base image, CUDA/PyTorch version choice, micromamba, code-outside-image + bind-mount model, wandb credential handling, build/run reference. | active |

## Investigations and findings

| File | Scope | Status |
|---|---|---|
| `docs/training_instability_fixes.md` | Six findings on run-to-run training variability (NaN recovery, gradient-clipping cold start, non-determinism, disabled EMA, missing LR warmup, TS loss scale) with patches. | **archive candidate** — all six findings appear to match commits already on `dev/stability` (`d94ce32`, `aa2b5c0`, `8d50ba1`, `ac1ab59`, `7439d77`); not yet archived pending confirmation the document's rationale isn't still needed as active reference. |
| `docs/transition1x_position_preprocessing.md` | What the Transition1X `.pkl` files actually contain vs. the paper's claims about atom mapping and fragment alignment. | active |
| `docs/transition1x_pkl_comparison_walkthrough.md` | Step-by-step, copy-pastable comparison of `train.pkl` vs. `valid_addprop.pkl`. | active |

## Archive

Nothing archived yet. See `docs/archive/README.md`.

## Outside `docs/`

| File | Scope |
|---|---|
| `CLAUDE.md` | Repository root. Always-loaded Claude Code guidance; kept terse, points here for detail. |
| `README.md` | Repository root. Upstream public-facing project description (exempt from the prose-style policy — see `docs/documentation_policy.md`). |
| `todo.org` | Repository root. Active task and decision log. |
| `todo_archive.org` | Repository root. Closed task/decision entries moved out of `todo.org`. |
