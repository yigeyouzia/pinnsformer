# AGENTS.md — pinnsformer-config

This file is the handoff contract for ChatGPT/Codex/other agents working in `pinnsformer-config/`.

## Scope

- Repository: `yigeyouzia/pinnsformer`
- Working area: `pinnsformer-config/`
- Purpose: executable PINN/PINNsFormer experiments, reproduction, diagnostics, and method development.
- Do not use this repository as the main paper/knowledge-base store. That role belongs to `yigeyouzia/ts-pinn-agent`.

## Stable paths

Canonical shared modules stay at the root of `pinnsformer-config/`:

- `navier_stokes_common.py`
- `gradient_diagnostics.py`

Do not move or rename them without updating every notebook/import and the compatibility symlinks under `notebooks/navier_stokes/`.

Navier–Stokes notebooks live under:

```text
pinnsformer-config/notebooks/navier_stokes/
```

Generated experiment artifacts live under:

```text
pinnsformer-config/outputs/
```

Curated small tables that should remain versioned live under:

```text
pinnsformer-config/results/summaries/
```

## Current file inventory

### Shared code

- `gradient_diagnostics.py`
- `navier_stokes_common.py`

### Baseline

- `notebooks/navier_stokes/navier_stokes_adam_baseline.ipynb`

### ConFIG / multi-loss

- `notebooks/navier_stokes/navier_stokes_config.ipynb`
- `notebooks/navier_stokes/navier_stokes_config2.ipynb`
- `notebooks/navier_stokes/navier_stokes_multiseed_runner.ipynb`
- `notebooks/navier_stokes/navier_stokes_4loss_multiseed_runner.ipynb`

### M-ConFIG

- `notebooks/navier_stokes/navier_stokes_mconfig4_fixed_roundrobin_runner.ipynb`
- `notebooks/navier_stokes/navier_stokes_adaptive_mconfig4_runner.ipynb`
- `notebooks/navier_stokes/navier_stokes_adaptive_mconfig4_rr_override_v2_runner.ipynb`

### Other balancing methods

- `notebooks/navier_stokes/navier_stokes_autobalance4_runner.ipynb`
- `notebooks/navier_stokes/navier_stokes_harmonic4_runner_serverpath.ipynb`

### Diagnostics / analysis

- `notebooks/navier_stokes/navier_stokes_gradient_diagnostic.ipynb`
- `notebooks/navier_stokes/navier_stokes_adam_vs_config_analysis.ipynb`
- `notebooks/navier_stokes/cylinder_nektar_wake_dataset_explorer.ipynb`
- `notebooks/navier_stokes/plot.ipynb`

### Curated summaries

- `results/summaries/adam_vs_config/field_metrics_corrected.csv`
- `results/summaries/adam_vs_config/history_summary.csv`
- `results/summaries/adam_vs_config/physics_threshold_steps.csv`
- `results/summaries/navier_stokes_multiseed/all_field_metrics.csv`
- `results/summaries/navier_stokes_multiseed/all_history_metrics.csv`
- `results/summaries/navier_stokes_multiseed/all_threshold_steps.csv`
- `results/summaries/navier_stokes_multiseed/mean_std_summary.csv`
- `results/summaries/navier_stokes_multiseed/paired_improvement.csv`
- `results/summaries/navier_stokes_multiseed/threshold_mean_std.csv`

## Reference notebooks for new methods

When adding a new gradient-balancing method, first inspect these two notebooks and follow their conventions unless there is a concrete reason not to:

1. `notebooks/navier_stokes/navier_stokes_harmonic4_runner_serverpath.ipynb`
2. `notebooks/navier_stokes/navier_stokes_mconfig4_fixed_roundrobin_runner.ipynb`

They are the preferred references for server paths, runner organization, multi-loss handling, output directories, and repeatable experiment structure.

## Server defaults

Current Linux-server defaults:

```text
PINNSFORMER_ROOT=/home/simplexity/cyt/pinnsformer-main
CONFIG_ROOT=/home/simplexity/cyt/ConFIG-main
PINNSFORMER_CONFIG_ROOT=/home/simplexity/cyt/pinnsformer-main/pinnsformer-config
```

Prefer reading these from environment variables. Do not add a new absolute path when an existing variable can be reused.

## Git/output policy

Never commit routine generated artifacts from experiments:

- checkpoints or model weights: `*.pt`, `*.pth`, `*.ckpt`;
- raw arrays/history/predictions: `*.npz`, `*.npy`;
- generated plots: `*.png`, `*.jpg`, `*.jpeg`, `*.svg`;
- per-run logs and temporary outputs;
- raw/per-seed CSV files under `outputs/` or `results/raw/`.

Keep `outputs/` as a local/generated workspace. Promote only compact aggregate tables needed for scientific comparison into `results/summaries/`.

When adding or removing an important notebook or curated result, update both `README.md` and this file inventory in the same change.

## Fair-comparison rules

A method comparison is valid only when non-method variables are controlled. Unless explicitly testing an ablation, keep fixed:

- network architecture;
- data sampling/training points;
- PDE/loss definitions;
- seed set;
- optimizer and learning rate where the method permits a fair match;
- update/function-evaluation budget;
- evaluation code and metric definitions.

Report mean ± std across seeds when the runner supports multiple seeds. Also retain component losses and gradient diagnostics so a result can be explained, not only ranked.

Do not claim a method is SOTA/better because code exists or because one run looks favorable. Distinguish clearly between code-present, run-complete, and statistically supported conclusions.

## Chebyshev next-step contract

The next planned method is Chebyshev-center gradient direction selection.

Implementation guidance:

- implement one parameterized method path, not separate duplicated Chebyshev-2 and Chebyshev-4 codebases;
- run both `p=2` and `p=4`;
- treat `p=2` as the primary candidate and `p=4` as a required geometry/norm ablation initially;
- compare against at least Adam/simple-sum, ConFIG, M-ConFIG/HARMONIC where budgets are compatible;
- save raw outputs under `outputs/`;
- save aggregate comparison tables under `results/summaries/chebyshev/` only after real runs.

Suggested future naming:

```text
notebooks/navier_stokes/navier_stokes_chebyshev_runner.ipynb
```

If a standalone reusable implementation is introduced later, prefer a small Python module rather than duplicating algorithm code across notebooks.

## Compatibility symlinks

`notebooks/navier_stokes/` contains symlinks back to the root shared modules and `outputs/`. They exist so reorganizing notebook files does not silently change imports or relative output locations on Linux/macOS. Do not replace them with divergent copied code.