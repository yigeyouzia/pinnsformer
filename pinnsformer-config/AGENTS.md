# AGENTS.md — pinnsformer-config

This file is the handoff contract for ChatGPT, Codex, and other agents working in `pinnsformer-config/`.

## Scope

- Repository: `yigeyouzia/pinnsformer`
- Working area: `pinnsformer-config/`
- Purpose: executable PINN/PINNsFormer experiments, reproduction, diagnostics, and method development.
- Paper notes, knowledge-base material, and agent/project management belong in `yigeyouzia/ts-pinn-agent`.

## Git policy — main only

This repository uses a **main-only workflow**.

- Work directly on `main`.
- Do **not** create feature, experiment, temporary, or agent branches.
- Do **not** open a PR for ordinary automated changes in this repository unless the user explicitly asks for one.
- Before writing, read the latest `main`; after validation, commit/push directly to `main`.
- If an old temporary branch exists from historical work, it is not a valid target for new changes and should be removed after its contents are present on `main`.
- Never force-push or rewrite `main` history unless the user explicitly requests it.

## Stable shared modules

Canonical shared modules stay at the root of `pinnsformer-config/`:

- `navier_stokes_common.py`
- `gradient_diagnostics.py`

Do not move or rename them without updating all imports and compatibility links.

## Notebook layout

Navier–Stokes notebooks are grouped by method name:

```text
notebooks/navier_stokes/
├── adam/
├── config/
├── mconfig/
├── autobalance/
├── harmonic/
├── chebyshev/
└── analysis/
```

Each method directory may contain compatibility symlinks back to the root shared modules and `outputs/`. Do not replace those links with divergent copied implementations.

### Adam

- `notebooks/navier_stokes/adam/navier_stokes_adam_baseline.ipynb`

### ConFIG

- `notebooks/navier_stokes/config/navier_stokes_config.ipynb`
- `notebooks/navier_stokes/config/navier_stokes_config2.ipynb`
- `notebooks/navier_stokes/config/navier_stokes_multiseed_runner.ipynb`
- `notebooks/navier_stokes/config/navier_stokes_4loss_multiseed_runner.ipynb`

### M-ConFIG

- `notebooks/navier_stokes/mconfig/navier_stokes_mconfig4_fixed_roundrobin_runner.ipynb`
- `notebooks/navier_stokes/mconfig/navier_stokes_adaptive_mconfig4_runner.ipynb`
- `notebooks/navier_stokes/mconfig/navier_stokes_adaptive_mconfig4_rr_override_v2_runner.ipynb`

### AutoBalance

- `notebooks/navier_stokes/autobalance/navier_stokes_autobalance4_runner.ipynb`

### HARMONIC

- `notebooks/navier_stokes/harmonic/navier_stokes_harmonic4_runner_serverpath.ipynb`

### Chebyshev center

Reusable implementation:

- `notebooks/navier_stokes/chebyshev/chebyshev_runner.py`

Runnable server notebooks:

- `notebooks/navier_stokes/chebyshev/navier_stokes_chebyshev2_runner_serverpath.ipynb`
- `notebooks/navier_stokes/chebyshev/navier_stokes_chebyshev4_runner_serverpath.ipynb`

Both use the same implementation. `p=2` is the primary candidate; `p=4` is a required norm-geometry ablation until experiments justify changing that status.

### Analysis / diagnostics

- `notebooks/navier_stokes/analysis/navier_stokes_gradient_diagnostic.ipynb`
- `notebooks/navier_stokes/analysis/navier_stokes_adam_vs_config_analysis.ipynb`
- `notebooks/navier_stokes/analysis/cylinder_nektar_wake_dataset_explorer.ipynb`
- `notebooks/navier_stokes/analysis/plot.ipynb`

## Reference runners for new methods

When adding a new gradient-balancing method, first inspect:

1. `notebooks/navier_stokes/harmonic/navier_stokes_harmonic4_runner_serverpath.ipynb`
2. `notebooks/navier_stokes/mconfig/navier_stokes_mconfig4_fixed_roundrobin_runner.ipynb`

Follow their server-path, multi-seed, 4-loss, output, and comparison conventions unless there is a concrete reason not to.

## Server defaults

```text
PINNSFORMER_ROOT=/home/simplexity/cyt/pinnsformer-main
CONFIG_ROOT=/home/simplexity/cyt/ConFIG-main
PINNSFORMER_CONFIG_ROOT=/home/simplexity/cyt/pinnsformer-main/pinnsformer-config
```

Prefer these environment variables instead of adding new machine-specific paths.

## Output policy

Generated experiment artifacts belong in:

```text
pinnsformer-config/outputs/
```

Do not commit routine generated artifacts:

- `*.pt`, `*.pth`, `*.ckpt`
- `*.npz`, `*.npy`
- generated `*.png`, `*.jpg`, `*.jpeg`, `*.svg`
- per-run logs, raw histories, predictions, and temporary CSV files
- per-seed CSV files under `outputs/` or `results/raw/`

Promote only compact, checked aggregate tables needed for scientific comparison into:

```text
pinnsformer-config/results/summaries/
```

When adding/removing an important notebook or curated result, update both `README.md` and this inventory.

## Fair-comparison rules

Unless an experiment explicitly tests an ablation, keep fixed:

- PINNsFormer architecture and initialization procedure
- dataset and sampled training points
- PDE and loss definitions
- seed set
- optimizer family and learning rate when a fair match is intended
- update/function-evaluation budget
- evaluation code and metric definitions

For multi-seed experiments report mean ± std and keep component losses plus gradient diagnostics. Do not claim a method is better or SOTA from code presence or one favorable seed.

## Chebyshev implementation contract

The current implementation follows Yoon et al. (2026), arXiv:2605.09975:

1. compute fresh gradients for `u_data`, `v_data`, `f_u`, `f_v`;
2. normalize each task gradient with the selected `l_p` norm;
3. solve the simplex dual `min ||sum alpha_i ghat_i||_p` with Frank–Wolfe;
4. recover the primal `l_q`-unit direction;
5. apply the paper's adaptive scalar `(sum_i g_i^T v) v`;
6. feed that final gradient direction to Adam.

The implementation supports `p=2` and `p=4` through one parameterized code path. Do not duplicate the algorithm into separate Chebyshev-2 and Chebyshev-4 Python implementations.
