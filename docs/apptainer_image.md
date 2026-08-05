# Apptainer image for OA-ReactDiff

This documents the design of `environment/apptainer/oa_reactdiff.def`, which
builds an Apptainer/Singularity image able to run everything in this repo
(tests, training, inference) with working CUDA on current GPUs.

## Starting point and the problem with it

`environment/apptainer/environment.yaml` is an archived `conda env export`
(379 packages) from an earlier working setup: PyTorch 1.12.1, CUDA 11.6,
CUDA-toolkit-linked `pytorch-scatter`/`pytorch-sparse` from the `pyg`
channel, plus a broad scientific/notebook stack (ASE, PyMatGen, PySCF,
OpenBabel, e3nn, Jupyter, TensorFlow, wandb, ...).

Before writing the `.def` file, the build host was checked directly:

```
$ nvidia-smi --query-gpu=driver_version,name,compute_cap --format=csv
driver_version, name, compute_cap
595.71.05, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 12.0

$ cat /etc/os-release
PRETTY_NAME="Ubuntu 26.04 LTS"
```

Compute capability 12.0 is NVIDIA's Blackwell architecture. PyTorch 1.12.1
was built years before Blackwell existed and ships no `sm_120` kernels, so
`torch.cuda.is_available()`/matmul on this GPU would fail (or silently run
on no device) no matter how faithfully the old pins were reproduced.
Decision (confirmed with the user): keep `environment.yaml`'s pins for
everything *except* the CUDA/PyTorch/PyG stack, which is bumped to
whatever currently supports this hardware.

## Package versions and why

- **Base image**: `nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04`
  - **CUDA 13.0, not 12.8**: as of PyTorch's 2.12 release (spring 2026),
    upstream PyTorch dropped prebuilt CUDA 12.8 wheels from its release
    pipeline; CUDA 13.0 is the current stable production track with full
    Blackwell support, and it's within this host's driver's supported
    range (driver 595.x supports up to CUDA 13.2).
  - **Ubuntu 24.04, not 22.04, and not 26.04**: NVIDIA's `nvidia/cuda`
    images currently top out at `ubuntu24.04` — there is no published
    image yet for this host's Ubuntu 26.04. This is fine: `apptainer run
    --nv` passes through the *host's* kernel-level GPU driver at runtime,
    so it's only the CUDA userspace libraries baked into the image that
    need to be new enough for Blackwell — the container's own OS/kernel
    version is otherwise decoupled from the host's. Matching the host
    Ubuntu release exactly is not a requirement for GPU support.
  - **`-devel`, not `-runtime`**: `pip install torch_scatter` falls back
    to compiling from source with `nvcc` whenever no prebuilt wheel
    matches the exact torch+CUDA combination (checked at build time
    against `data.pyg.org`, but not guaranteed for every future
    combination); the devel image keeps that fallback working without
    reaching outside the image for a compiler.
  - **`-cudnn`**: bundles cuDNN instead of a separate manual install.
- **PyTorch**: pinned to `2.13.0+cu130` (the current PyPI stable at time
  of writing). `torchvision`/`torchaudio` are left unpinned in the pip
  install so pip resolves whatever versions the PyTorch package index
  declares compatible with that exact torch release, rather than us
  guessing a matching triplet by hand.
- **`torch_geometric`**: installed via plain pip (pure Python + optional
  compiled extras it manages itself); left unpinned to track torch.
- **`torch_scatter`**: installed from PyG's wheel index
  (`https://data.pyg.org/whl/torch-2.13.0+cu130.html`, confirmed to exist
  before committing to this combination). This repo only imports
  `torch_scatter.segment_csr` (`oa_reactdiff/model/util_funcs.py`) — the
  `pytorch-sparse` conda package in the original lockfile is not actually
  imported anywhere and is dropped rather than replaced.
- **`torchtext`**: dropped outright. Not imported anywhere in this repo,
  and upstream `torchtext` was discontinued around torch 2.2 — there is
  no version of it compatible with torch 2.13.
- **`pytorch-lightning`**: kept at its original pin, **1.8.6**, per "keep
  the rest pinned." It's installed via `pip install --no-deps` *after*
  the conda/micromamba environment is built, not as part of it. Reason:
  conda-forge's `pytorch-lightning` package declares a conda-level
  dependency on `pytorch`; if it were left in the `micromamba create`
  step, the solver would pull in *some* conda-forge `pytorch` build to
  satisfy that dependency — reintroducing an old, CUDA-11.6-linked torch
  install underneath the one installed explicitly via pip afterwards.
  `--no-deps` is safe here because every other dependency PL 1.8.6 needs
  (`torchmetrics`, `fsspec`, `tqdm`, `packaging`, `pyyaml`, ...) is
  already present, explicitly pinned, in the untouched parts of
  `environment.yaml`.
  - **Known risk, left as-is per the pinning decision**: PyTorch
    Lightning 1.8.6 was released in late 2022, before PyTorch 2.0. It has
    not been tested against torch 2.13 here. If `pl_trainer.py` fails to
    import or errors on trainer construction, the most likely fix is
    bumping this pin — flagged here rather than silently changed.
- **`wandb`**: bumped to **0.22.3** (minimum), overriding
  `environment.yaml`'s originally-pinned `0.19.10`. Same category of
  exception as the CUDA/PyTorch stack: not a repo-internal dependency,
  but one that has to interoperate with a live external service
  (wandb.ai) whose behavior moved on since the pin was set. wandb.ai now
  issues 86-character API keys; client versions before 0.22.3
  hard-validate keys as exactly 40 characters and reject the new format
  with `API key must be 40 characters long, yours was 86` at `wandb
  login` — discovered when logging in through this image (2026-08-05),
  confirmed against upstream reports, not a container or paste issue.

  Resolved at the conda level rather than with a pip override (the
  `pytorch-lightning`/`torchmetrics` pattern): `wandb=0.22.3` moved into
  `environment.yaml`'s top-level `dependencies:` list, `wandb==0.19.10`
  removed from its `pip:` section. Preferred here because wandb, unlike
  `pytorch-lightning`/`torchmetrics`, has no conda-level dependency on
  `pytorch` — nothing forces it out of the conda solve — so folding it
  back into the same lockfile the rest of the environment comes from is
  less to read later than a second pip install. The one open risk is
  `protobuf`/`libprotobuf` (`environment.yaml` still pins
  `libprotobuf=3.20.1`): unconfirmed whether conda-forge's `wandb=0.22.3`
  needs a newer one, so not yet verified to actually solve — see
  `todo.org`. If it doesn't, the fallback is the same treatment
  `torchmetrics` already gets: exclude from the conda solve, pip-install
  `--no-deps` afterward.

## Package manager: micromamba, not Miniforge/mamba

The image uses [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)
— a single ~100 MB static binary — instead of installing a full
Miniforge/Mambaforge base distribution. For a single-environment image
this avoids carrying a base conda install that's only ever used once, and
avoids `conda activate`/shell-hook ceremony entirely: since the image only
ever has one environment (`oa_reactdiff`), its `bin/` directory is just
prepended to `PATH` directly, both in `%post` (for the pip install steps)
and in `%environment` (for runtime).

`environment.yaml` is used as-is as the input lockfile (its `channels:`
key — `conda-forge`, `pyg`, `pytorch` — is respected the same way by
`micromamba create -f` as it would be by conda/mamba); a filtered copy
with the GPU-stack lines stripped out (see `%post`) is what actually gets
solved.

## Code lives outside the image

The repository is **not** copied into the image. Only
`environment/apptainer/environment.yaml` is (for the conda/micromamba
solve at build time). At run time, bind-mount the repo instead:

```bash
apptainer shell --nv --bind "$PWD":/workspace \
    environment/apptainer/oa_reactdiff.sif
```

`%environment` puts `/workspace` on `PYTHONPATH`, so `import oa_reactdiff`
works directly from the bind-mounted source — no editable install step,
no rebuild when the code changes. This keeps the image itself small,
long-lived, and reusable across the many experiment variants in this
repo, while the (large, frequently-changing) repository — notebooks,
checkpoints, training-script variants — stays exactly as it is on disk
and under git, never duplicated into a build artifact.

## Secrets: wandb credentials stay on the host, not in the image

Training scripts (e.g. `train_ts1x.py`) use PyTorch Lightning's
`WandbLogger`, which authenticates however the `wandb` client normally
does — by reading `~/.netrc` (written once by running `wandb login` on
the host) or `~/.config/wandb/settings`. Nothing in this repo hardcodes a
key, so nothing in the image needs to either.

Apptainer bind-mounts the host's `$HOME` into the container **by
default**, at the same path, whenever `--no-home`/`--contain` are not
passed. `%environment` in this image deliberately leaves `$HOME`
untouched, so:

1. Run `wandb login` once, on the host (not in the container).
2. Run training normally, e.g.:
   ```bash
   apptainer exec --nv --bind "$PWD":/workspace --pwd \
       /workspace/oa_reactdiff/trainer \
       environment/apptainer/oa_reactdiff.sif \
       python train_ts1x.py
   ```
   No `--env WANDB_API_KEY=...` is needed. That form was deliberately
   avoided: it would put the key in plaintext in shell history and in
   the process list (visible to anything reading `/proc/<pid>/environ`
   on a shared machine) for no benefit over the credentials Apptainer
   already exposes safely through the home-directory bind mount.

**This breaks if a run command adds `--no-home` or `--contain`** — those
disable the default home bind mount. If you need either flag for other
reasons, bind `~/.netrc` explicitly instead:
`--bind "$HOME/.netrc:/root/.netrc"` (adjust the target path to whatever
`$HOME` resolves to inside the container).

## Build-time vs run-time validation

- **`%test`** (`apptainer build` / `apptainer test`) — runs *without*
  `--nv` (no GPU is available at build time), so it only checks that the
  environment resolved correctly: imports `torch`, `torchvision`,
  `torch_geometric`, `torch_scatter`, `pytorch_lightning`, `ase`, `e3nn`,
  `pymatgen`, `rmsd` and prints versions. This is what would catch a
  broken build in CI, before ever touching a GPU.
- **`%runscript`** (`apptainer run --nv ...`) — the GPU-specific test
  requested: prints `torch.__version__`, `torch.version.cuda`,
  `torch.cuda.is_available()`, the device name and compute capability,
  then allocates two random `4096x4096` tensors on the CUDA device,
  multiplies them, calls `torch.cuda.synchronize()`, and prints a
  checksum of the result. It exits non-zero with an actionable message
  (missing `--nv`?) if CUDA isn't available. Any arguments passed to
  `apptainer run` are `exec`'d after the self-test succeeds, so the same
  entrypoint doubles as a normal run wrapper
  (`apptainer run --nv image.sif python train_ts1x.py` runs the self-test
  then the training script); with no arguments it drops into `bash`.

## Provenance

`%post` writes `/opt/build/BUILD_INFO.txt` inside the image (build
timestamp, base image tag, torch version, and a checksum of the
`environment.yaml` it was built from), and `%labels` records the same
key facts for `apptainer inspect`. The full `environment.yaml` is also
left at `/opt/build/environment.yaml` inside the image, so the exact
lockfile a given `.sif` was built from travels with it.

## Build & run reference

```bash
# Build (from the repository root):
apptainer build --fakeroot \
    environment/apptainer/oa_reactdiff.sif \
    environment/apptainer/oa_reactdiff.def

# CUDA self-test:
apptainer run --nv environment/apptainer/oa_reactdiff.sif

# CUDA self-test, then a shell with the repo mounted (`run` triggers the
# self-test; `shell` below does not, since it skips %runscript):
apptainer run --nv --bind "$PWD":/workspace \
    environment/apptainer/oa_reactdiff.sif

# Interactive shell, repo mounted at /workspace, no self-test:
apptainer shell --nv --bind "$PWD":/workspace \
    environment/apptainer/oa_reactdiff.sif

# Test suite:
apptainer exec --nv --bind "$PWD":/workspace \
    environment/apptainer/oa_reactdiff.sif \
    pytest /workspace/oa_reactdiff/tests/

# Training (train_ts1x.py must run from oa_reactdiff/trainer/):
apptainer exec --nv --bind "$PWD":/workspace --pwd \
    /workspace/oa_reactdiff/trainer \
    environment/apptainer/oa_reactdiff.sif \
    python train_ts1x.py
```

### Background training runs, and picking a GPU

```bash
# Check what's free first (this host is shared — see CLAUDE.md):
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

CUDA_VISIBLE_DEVICES=0 nohup apptainer exec --nv --bind "$PWD":/workspace --pwd \
    /workspace/oa_reactdiff/trainer \
    environment/apptainer/oa_reactdiff.sif \
    python train_ts1x.py \
    > train_ts1x_$(date +%Y%m%d_%H%M%S).log 2>&1 &
disown
```

`CUDA_VISIBLE_DEVICES` set on the host side reaches the container:
Apptainer inherits the calling shell's environment by default (no
`--cleanenv` in any documented invocation here), and `--nv` itself
exposes all host GPUs rather than restricting to one. This is not just
device selection: `train_ts1x.py` builds
`devices = list(range(torch.cuda.device_count()))` and hands that to the
Lightning `Trainer`, so however many devices are visible, it uses all of
them — `CUDA_VISIBLE_DEVICES` is what keeps a run to a single GPU on a
host with more than one. `disown` after backgrounding fully detaches the
job from the shell (closing the SSH session won't touch it), beyond what
`nohup` alone (SIGHUP-immunity only) provides.

## Open risks to watch for on first build

- **`pytorch-lightning==1.8.6` on `torch==2.13.0`**: import confirmed
  working (`%test`, 2026-08-05). Actual training behavior not yet run.
  If `pl_trainer.py` breaks at runtime, bump this pin first.
- **`torch_scatter` wheel availability**: confirmed against
  `data.pyg.org` for `torch-2.13.0+cu130` at the time this was written,
  but PyG's wheel builds can lag a new torch release. If the prebuilt
  wheel is gone by the time you build, pip transparently falls back to
  compiling from source (the devel base image has `nvcc` for exactly
  this reason) — slower, but should still work.
- **`pillow=8.4.0`** is pinned (via the untouched part of
  `environment.yaml`) from the original 2021-era export; `pip install
  torchvision` may pull a newer `pillow` to satisfy its own requirement,
  which is an intentional, accepted deviation from that one pin.

## Known build issues (first build attempt, 2026-08-05)

Build succeeded and `apptainer run --nv` passed the CUDA self-test on
this host: `torch 2.13.0+cu130`, device `NVIDIA RTX PRO 6000 Blackwell
Max-Q Workstation Edition`, compute capability `(12, 0)`, `a @ b` matmul
correct. Issues below are the trail that got there.

- **`--fakeroot` prerequisites (uidmap, subuid/subgid, AppArmor
  unprivileged user namespaces)**: hit and resolved on this host. See
  `todo.org` for the diagnostic trail; not a `.def`-file problem, so
  nothing here needed to change.
- **`%post` fails immediately**: `/.post.script: N: set: Illegal option
  -o pipefail` (exit status 2), before any of the section's actual
  commands run. `-o pipefail` is a bash-only `set` option and Apptainer
  executes `%post` by running a fixed shell (`/bin/sh`, dash on Ubuntu)
  on the generated script file — it does not exec the file directly and
  does not honor a shebang line written inside the section body. This
  was tested directly: adding `#!/bin/bash` as the first line of `%post`
  did not change the outcome, including after correcting that line to
  have no leading whitespace (`#!` as the first two characters) — the
  error reproduced identically both times, only the reported line number
  changed. `%post`, in this respect, differs from `%runscript`, which
  becomes a file inside the built container that `apptainer run` execs
  directly at container-run time, where a shebang is meaningful.

  **Fix:** remove `-o pipefail` from `%post`; keep `set -eu`, which is
  POSIX and works under `/bin/sh`. The one pipe in the section
  (`curl ... | tar -xj ...`) does not need `pipefail` for correctness
  here: a `curl` failure feeds `tar` truncated or empty input, which
  `tar` itself rejects with a nonzero exit status, and `set -e` catches
  that. The `#!/bin/bash` line in `%post` has no effect and should be
  removed to avoid suggesting otherwise. `%runscript`'s `#!/bin/bash`
  can stay — harmless there, though also not load-bearing, since nothing
  in that section uses bash-only syntax. `%test` uses only `set -e` and
  is unaffected.

  Applied and confirmed (2026-08-05): `%post` now runs under bash and
  reaches `micromamba create`. Resolved.

- **`micromamba create` failed to solve the environment**: `opt_einsum_fx`
  and `torchmetrics`, both then present in `environment.yaml`, each
  declared a conda-level dependency on `pytorch` — the same failure mode
  the `pytorch-lightning` exclusion exists to avoid, reached here through
  two packages that exclusion didn't cover. Once any `pytorch` variant
  entered the solve, its transitive dependencies (`libabseil`, `mkl`,
  `tbb`, `setuptools` version caps) collided with this environment's
  older pins (`abseil-cpp=20210324.2`, `libprotobuf=3.20.1`), and every
  candidate failed.

  Resolution differed by package, decided by checking what the codebase
  actually imports rather than reinstalling both at their original
  lockfile versions by default: `opt_einsum_fx` is not imported anywhere
  in `oa_reactdiff/` (only in a few root-level scratch notebooks,
  alongside `e3nn`, likewise unused by the core package) and was removed
  from `environment.yaml` entirely rather than reinstalled.
  `torchmetrics` is a hard, direct dependency of
  `oa_reactdiff/trainer/pl_trainer.py` and was excluded from the conda
  solve, then pip-installed `--no-deps` at its original `1.5.2` — the
  same treatment `pytorch-lightning` already gets, for the same reason
  (it's the version this exact import pattern was validated against, and
  `torchmetrics` has no CUDA-specific code, so there's no reason tied to
  this image's purpose to move it off that pin).

  Applied and confirmed (2026-08-05): build completed;
  `Successfully installed pytorch-lightning-1.8.6 torchmetrics-1.5.2`
  (no `opt_einsum_fx`); `%test` passed. Resolved.

- **`wandb login` rejects a valid key**: `ValueError: API key must be 40
  characters long, yours was 86`, run via
  `apptainer exec environment/apptainer/oa_reactdiff.sif wandb login`
  (2026-08-05). Not a paste or container issue: wandb.ai now issues
  86-character API keys, and the pinned `wandb==0.19.10` client
  hard-validates keys as exactly 40 characters. Fixed upstream in
  `wandb==0.22.3`. See "Package versions and why", above, for the fix
  (`wandb=0.22.3` moved into `environment.yaml`'s conda `dependencies:`,
  removed from its `pip:` section). Not yet rebuilt/verified — see
  `todo.org` for the one open risk (`protobuf`/`libprotobuf`).
