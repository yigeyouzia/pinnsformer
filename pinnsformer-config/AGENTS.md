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
- Do **not** open a PR for ordinary automated changes unless the user explicitly asks for one.
- Before writing, read the latest `main`; after validation, commit/push directly to `main`.
- Never force-push or rewrite `main` history unless the user explicitly requests it.

## Notebook-first experiment policy

For experiment methods in `pinnsformer-config/notebooks/`, the default deliverable is a **complete runnable notebook**, following the existing HARMONIC / M-ConFIG style.

A method notebook must keep the full workflow visible in the `.ipynb` itself:

1. server paths and experiment configuration;
2. method/optimizer implementation needed for the experiment;
3. training loop and multi-seed execution;
4. raw metric/history export;
5. aggregate mean ± std tables;
6. diagnostic and result plotting cells.

Do **not** replace an experiment notebook with a thin wrapper that only imports an external runner module unless the user explicitly asks for that architecture. Shared, stable domain helpers such as `navier_stokes_common.py` and `gradient_diagnostics.py` may remain external; method-specific algorithm/training/plotting logic should remain visible in the notebook.

When adding a new method, first inspect:

1. `notebooks/navier_stokes/harmonic/navier_stokes_harmonic4_runner_serverpath.ipynb`
2. `notebooks/navier_stokes/mconfig/navier_stokes_mconfig4_fixed_roundrobin_runner.ipynb`

Preserve their server-path, multi-seed, 4-loss, output, summary, and plotting conventions unless there is a concrete reason not to.

## Stable shared modules

Canonical shared modules stay at the root of `pinnsformer-config/`:

- `navier_stokes_common.py`
- `gradient_diagnostics.py`

Do not move or rename them without updating all notebook imports.

## Notebook layout

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
- `notebooks/navier_stokes/chebyshev/navier_stokes_chebyshev2_runner_serverpath.ipynb`
- `notebooks/navier_stokes/chebyshev/navier_stokes_chebyshev4_runner_serverpath.ipynb`

Both Chebyshev notebooks are self-contained experiment notebooks. `p=2` is the primary candidate; `p=4` is the norm-geometry ablation. Current memory-constrained default is `N_TRAIN=600`; an 800-point benchmark must explicitly change that config and be reported separately.

### Analysis / diagnostics
- `notebooks/navier_stokes/analysis/navier_stokes_gradient_diagnostic.ipynb`
- `notebooks/navier_stokes/analysis/navier_stokes_adam_vs_config_analysis.ipynb`
- `notebooks/navier_stokes/analysis/cylinder_nektar_wake_dataset_explorer.ipynb`
- `notebooks/navier_stokes/analysis/plot.ipynb`

## Server defaults

```text
PINNSFORMER_ROOT=/home/simplexity/cyt/pinnsformer-main
CONFIG_ROOT=/home/simplexity/cyt/ConFIG-main
PINNSFORMER_CONFIG_ROOT=/home/simplexity/cyt/pinnsformer-main/pinnsformer-config
```

Prefer existing environment variables/server conventions instead of inventing new machine-specific paths.

## Output policy

Generated experiment artifacts belong in `pinnsformer-config/outputs/` and must not be committed routinely:

- `*.pt`, `*.pth`, `*.ckpt`
- `*.npz`, `*.npy`
- generated `*.png`, `*.jpg`, `*.jpeg`, `*.svg`
- per-run logs, histories, predictions, and temporary/per-seed CSV files

Promote only compact, checked aggregate tables needed for scientific comparison into `pinnsformer-config/results/summaries/`.

## Fair-comparison rules

Unless an experiment explicitly tests an ablation, keep fixed:

- PINNsFormer architecture and initialization;
- dataset and sampled training points;
- PDE and loss definitions;
- seed set;
- optimizer family and learning rate where a fair match is intended;
- update/function-evaluation budget;
- evaluation code and metric definitions.

Report mean ± std for multi-seed experiments and retain component losses plus gradient diagnostics. Do not claim a method is better or SOTA from one favorable run.

## Chebyshev implementation contract

The Chebyshev notebooks follow Yoon et al. (2026), arXiv:2605.09975:

1. compute fresh gradients for `u_data`, `v_data`, `f_u`, `f_v`;
2. normalize each task gradient with the selected `l_p` norm;
3. solve the simplex dual `min ||sum alpha_i ghat_i||_p` with Frank–Wolfe;
4. recover the primal `l_q`-unit direction;
5. apply the adaptive scalar `(sum_i g_i^T v) v`;
6. feed that final gradient direction to Adam.

The p=2 and p=4 notebooks should stay structurally parallel so differences are attributable to the selected norm, not to unrelated code changes. Each must retain plotting for losses, dual weights, gradient geometry, direction alignment, solver diagnostics, and final multi-seed Relative-L2 summary.
