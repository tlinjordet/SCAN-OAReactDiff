# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

OA-ReactDiff is a diffusion-based generative model for 3D chemical reaction generation — specifically generating reactant, transition state, and product structures simultaneously. The key innovation is **object-aware SE(3) equivariance**: unlike standard SE(3) models, this enforces symmetry at the fragment (molecule) level rather than the whole system, because any rotation of an individual molecule in a reaction should not change the reaction's identity.

The primary use case is transition state generation from known reactant/product pairs (inpainting mode), reducing cost from days of DFT computation to seconds.

## Environment setup

```bash
conda env create -f env.yaml
conda activate oa_reactdiff
pip install -e .
```

The environment uses Python 3.10, PyTorch 1.12.1, PyTorch Geometric, and pytorch-lightning 1.8.6. GPU is expected for training; CPU works for testing.

## Commands

Run all tests:
```bash
pytest oa_reactdiff/tests/
```

Run a single test file:
```bash
pytest oa_reactdiff/tests/dynamics/test_egnn_dynamics.py -v
```

Run with coverage (as CI does):
```bash
pytest -v --cov=oa_reactdiff --cov-report=xml --color=yes oa_reactdiff/tests/
```

Train on Transition1x dataset (run from `oa_reactdiff/trainer/`):
```bash
cd oa_reactdiff/trainer
python train_ts1x.py
```

## Architecture overview

The system has four layers that compose at runtime:

**1. Diffusion (`oa_reactdiff/diffusion/`)**
- `EnVariationalDiffusion` (`en_diffusion.py`) — the top-level DDPM module. Owns the noise schedule, normalizer, and dynamics. Handles forward (training: add noise, predict epsilon) and reverse (sampling: iterative denoising). Also implements RePaint-style inpainting for conditioning on known fragments.
- `DiffSchedule` / `PredefinedNoiseSchedule` (`_schedule.py`) — wraps cosine/polynomial/linear noise schedules and computes alpha/sigma/SNR values.
- `Normalizer` (`_normalizer.py`) — normalizes the three node feature types: `pos` (3D coords), `one_hot` (atom type, 5 elements: H/C/N/O/F), `charge` (integer).

**2. Dynamics (`oa_reactdiff/dynamics/`)**
- `EGNNDynamics` (`egnn_dynamics.py`) — the denoising network wrapper. Takes a list of per-fragment `xh` tensors, concatenates them into a single graph, runs the GNN, then splits results back per-fragment. Enforces object-aware SE(3) by removing the center of gravity of each fragment's predicted velocities separately (`remove_mean_batch`).
- `BaseDynamics` (`_base.py`) — builds per-fragment MLPs (`encoders`/`decoders`) that project raw node features into/out of a shared embedding dimension. All fragments share one GNN backbone but have their own encode/decode heads (with optional weight-sharing via `enforce_same_encoding`).
- `Confidence` (`confidence.py`) — a separate dynamics head for scoring generated samples; same architecture as `EGNNDynamics` but outputs a scalar confidence per sample.

**3. GNN backbone (`oa_reactdiff/model/`)**
- `LEFTNet` (`leftnet.py`) — the default and recommended backbone. A SOTA SE(3)-equivariant GNN. Set `object_aware=True` in its config to enable the mixed update/scalar-pass that achieves object-wise equivariance.
- `EGNN` (`egnn.py`) — the fallback backbone from e3_diffusion_for_molecules. Simpler but less expressive.
- `MLP` / `core.py` — shared building blocks used in encoders/decoders.
- The `subgraph_mask` (from `_graph_tools.get_subgraph_mask`) is passed to the GNN to distinguish intra-fragment edges from inter-fragment edges. This is how object-aware updates are implemented: LEFTNet uses it to apply equivariant updates only within fragments, and scalar updates across fragments.

**4. Data and training (`oa_reactdiff/dataset/`, `oa_reactdiff/trainer/`)**
- `BaseDataset` loads `.npz` or `.pkl` files; `ProcessedTS1x` handles the Transition1x dataset with support for reactant/product swapping augmentation and reflection augmentation.
- `DDPMModule` (`pl_trainer.py`) — PyTorch Lightning module wrapping the full diffusion stack. Configures datasets, optimizer (Adam + optional cosine/step LR schedule), EMA, and validation metrics (RMSD).
- Training entry point is `train_ts1x.py` (run from `oa_reactdiff/trainer/`); it directly imports `pl_trainer` via a relative path (not as a package), so it must be run from that directory.

## Key data structures

Every molecule/fragment is represented as a dict with keys `pos` (Tensor `[n_atoms, 3]`), `one_hot` (Tensor `[n_atoms, 5]`), `charge` (Tensor `[n_atoms, 1]`), `size`, and `mask`. The `mask` tensor maps each atom to its batch index (which sample in the batch it belongs to). The combined graph for all fragments in a batch is built by concatenating these masks; `n_frag_switch` maps each atom to its fragment type (0=reactant, 1=TS, 2=product).

For the TS1x task, `fragment_names = ["R", "TS", "P"]`, `node_nfs = [9, 9, 9]` (3 pos + 5 one_hot + 1 charge), and `pos_only=True` (atom identities are fixed; only positions are diffused).

## Atom type encoding

Only five elements are supported: H(1), C(6), N(7), O(8), F(9), mapped to indices 0–4 in `ATOM_MAPPING` (`base_dataset.py`). This is a hard constraint throughout the codebase.

## Pretrained checkpoint

`pretrained-ts1x-diff.ckpt` in the repo root is the published model for TS generation. Load it via `DDPMModule` using pytorch-lightning's checkpoint loading. The notebook `OA-ReactDiff.ipynb` shows the full inference workflow.
