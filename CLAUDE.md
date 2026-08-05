# CLAUDE.md

Guidance for Claude Code in this repository. **Documentation index:
`docs/OVERVIEW.md`** — open it before nontrivial work and read only the
file(s) it points to, rather than re-deriving context that is already
written down. Rules for maintaining the documentation and task-tracking
system itself: `docs/documentation_policy.md`.

## Operating policies

- **Advisory only.** Claude may not Write/Edit implementation files
  (source, notebooks, configs, data, checkpoints) in this repository.
  Permitted writes: markdown documents under `docs/` (including its own
  index, policy, and archive), and `todo.org`/`todo_archive.org` at the
  repository root. When asked to implement, fix, or change something,
  write the proposal to `docs/` instead, filed per
  `docs/documentation_policy.md`, for a human to review and apply.
  Read-only work (inspection, running tests to check something) is
  unaffected. Why: this is a research codebase where subtle correctness
  — physics, equivariance, numerics — is the point, and an LLM's direct
  edits can be confidently, silently wrong.
- **Use and maintain the documentation system as a matter of course.**
  Check `docs/OVERVIEW.md` before starting nontrivial work. Update the
  relevant document(s) and `todo.org` as part of finishing a task, not as
  a follow-up. Flag — don't silently perform — archiving of documents
  that look obviated.
- Claude Code's own settings (permissions, hooks, allowed tools) live
  under this repository's `.claude/`, never in user-level/global config.
- Don't address the maintainer by name; refer to them by role if needed.
- **Never suggest or run a command that blindly overwrites a file outside
  git version control** (`/etc/**`, `sysctl.d`, dotfiles, cron/systemd
  units, ...) — check whether the target already exists and read it
  first, whether Claude or the maintainer is the one to run the command.
  Scope `sudo` tightly: only propose it when it's clear the maintainer is
  acting as their admin account; this machine is shared (`/etc/subuid`
  lists both `trond` and `icredd`), so don't assume the active shell has
  admin rights — ask if it's ambiguous.

## What this project is

OA-ReactDiff is a diffusion-based generative model for 3D chemical reaction generation — specifically generating reactant, transition state, and product structures simultaneously, using object-aware SE(3) equivariance (symmetry enforced per fragment/state rather than per whole system). Primary use case: transition state generation from known reactant/product pairs (inpainting mode). Full architecture: `docs/architecture.md`.

## Project lineage and naming

This repository descends from Chenru Duan's original OA-ReactDiff work; a lot of code, notebooks, and structure still carry that legacy. The current project built on top of it — generating transition states from reactant/product pairs — has gone by several working titles across notebooks and branches: **TS-Diffusion**, **TS-diff**, and formerly **SCAN-OAReactDiff**. Treat these as the same project when they show up in filenames, notebook titles, or commit history. The current main developer of this downstream project is the repository's maintainer (not Chenru Duan).

## Environment setup

```bash
conda env create -f env.yaml
conda activate oa_reactdiff
pip install -e .
```

Python 3.10, PyTorch 1.12.1, PyTorch Geometric, pytorch-lightning 1.8.6. GPU expected for training; CPU works for testing. For an Apptainer image with a current CUDA/PyTorch stack instead, see `docs/apptainer_image.md`.

## Commands

```bash
pytest oa_reactdiff/tests/                                              # all tests
pytest oa_reactdiff/tests/dynamics/test_egnn_dynamics.py -v             # single file
pytest -v --cov=oa_reactdiff --cov-report=xml --color=yes oa_reactdiff/tests/  # with coverage, as CI does
cd oa_reactdiff/trainer && python train_ts1x.py                         # train on Transition1x (must run from this directory)
```

## Further reading

- `docs/architecture.md` — the four runtime layers, key data structures, atom type encoding, pretrained checkpoint.
- `docs/terminology.md` — project-specific/overloaded terms; read before touching "fragment"-related code.
- `docs/apptainer_image.md` — the Apptainer image.
- `docs/OVERVIEW.md` — index of everything else.
