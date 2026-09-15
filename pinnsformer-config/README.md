# PINNsFormer multi-objective PINN experiments

`pinnsformer-config/` is the experiment-development area for multi-loss / multi-objective optimization on PINNsFormer. The separate `ts-pinn-agent` repository is used for papers, notes, knowledge-base material, and agent/project management; this repository is for executable experiments, reproductions, diagnostics, and method development.

## Current research line

The current Navier–Stokes line compares gradient/loss balancing methods under a shared PINNsFormer setup. Existing code covers Adam/simple-sum, ConFIG, 4-loss and multi-seed variants, M-ConFIG variants, AutoBalance, and HARMONIC. Chebyshev-center training is the next method to add; use `p=2` as the primary configuration and `p=4` as an ablation unless experiments show otherwise.

## Layout

```text
pinnsformer-config/
├── README.md
├── AGENTS.md
├── .gitignore
├── gradient_diagnostics.py          # canonical shared gradient diagnostics
├── navier_stokes_common.py          # canonical shared PINNsFormer/NS helpers
├── notebooks/
│   └── navier_stokes/               # runnable and analysis notebooks
├── outputs/                         # generated raw runs; ignored by git
│   └── .gitkeep
└── results/
    ├── README.md
    └── summaries/                   # curated small CSV summaries kept in git
```

`notebooks/navier_stokes/` contains compatibility symlinks for `navier_stokes_common.py`, `gradient_diagnostics.py`, and `outputs/`. This keeps existing imports and relative `./outputs/...` paths working after the notebook reorganization on Linux/macOS.

## Notebook inventory

### Baseline

- `notebooks/navier_stokes/navier_stokes_adam_baseline.ipynb` — Adam + simple-sum baseline.

### ConFIG / multi-loss baselines

- `notebooks/navier_stokes/navier_stokes_config.ipynb` — 2-loss ConFIG experiment.
- `notebooks/navier_stokes/navier_stokes_config2.ipynb` — alternate/iterated ConFIG notebook retained for reproducibility.
- `notebooks/navier_stokes/navier_stokes_multiseed_runner.ipynb` — multi-seed Adam/simple-sum vs ConFIG runner.
- `notebooks/navier_stokes/navier_stokes_4loss_multiseed_runner.ipynb` — 4-loss multi-seed runner.

### M-ConFIG family

- `notebooks/navier_stokes/navier_stokes_mconfig4_fixed_roundrobin_runner.ipynb` — fixed round-robin M-ConFIG runner; preferred reference for the runner structure.
- `notebooks/navier_stokes/navier_stokes_adaptive_mconfig4_runner.ipynb` — adaptive M-ConFIG variant.
- `notebooks/navier_stokes/navier_stokes_adaptive_mconfig4_rr_override_v2_runner.ipynb` — adaptive round-robin/override v2 variant.

### Other balancing methods

- `notebooks/navier_stokes/navier_stokes_autobalance4_runner.ipynb` — AutoBalance 4-loss runner.
- `notebooks/navier_stokes/navier_stokes_harmonic4_runner_serverpath.ipynb` — HARMONIC-4 runner with server-path conventions; preferred reference together with the fixed-round-robin M-ConFIG runner.

### Diagnostics / analysis

- `notebooks/navier_stokes/navier_stokes_gradient_diagnostic.ipynb` — gradient norms, cosine similarity, and conflict diagnosis.
- `notebooks/navier_stokes/navier_stokes_adam_vs_config_analysis.ipynb` — Adam vs ConFIG analysis.
- `notebooks/navier_stokes/cylinder_nektar_wake_dataset_explorer.ipynb` — dataset inspection/reconstruction helper.
- `notebooks/navier_stokes/plot.ipynb` — plotting/inspection notebook.

## Curated results

Raw run artifacts are not source code and should not be committed. Existing useful aggregate CSVs were moved out of `outputs/` into:

```text
results/summaries/adam_vs_config/
results/summaries/navier_stokes_multiseed/
```

Keep only compact, human-readable tables needed for comparison/reproducibility. Per-seed histories, predictions, checkpoints, `.npz`, generated figures, and temporary CSVs belong in `outputs/` and are ignored.

## Server paths and environment

Existing experiments use these defaults on the Linux server:

```bash
export PINNSFORMER_ROOT=/home/simplexity/cyt/pinnsformer-main
export CONFIG_ROOT=/home/simplexity/cyt/ConFIG-main
export PINNSFORMER_CONFIG_ROOT=$PINNSFORMER_ROOT/pinnsformer-config
```

The Navier–Stokes dataset is expected at:

```text
$PINNSFORMER_ROOT/demo/navier_stokes/cylinder_nektar_wake.mat
```

Prefer environment variables instead of introducing new hard-coded machine paths.

## Comparison invariants

When comparing gradient-combination methods, keep the following fixed unless the experiment is explicitly an ablation:

- PINNsFormer architecture and initialization procedure;
- dataset and sampled training points;
- loss definitions and 2-loss/4-loss decomposition;
- optimizer family, learning rate, step/function-evaluation budget;
- seed set;
- evaluation definition and metric implementation.

Record at least relative L2 error, component losses, gradient norms, pairwise gradient cosine/conflict statistics, wall time, and peak memory when practical. Do not infer a method is better from one seed or from loss curves alone.

## Next method: Chebyshev center

For the next implementation, add a runner under `notebooks/navier_stokes/` and follow the server/path/output conventions used by:

1. `navier_stokes_harmonic4_runner_serverpath.ipynb`
2. `navier_stokes_mconfig4_fixed_roundrobin_runner.ipynb`

Start with both `p=2` and `p=4` under the same implementation/configurable code path. Treat `p=2` as the main candidate and `p=4` as a required norm-geometry ablation. Save raw outputs to `outputs/` and commit only aggregate summaries to `results/summaries/`.

See `AGENTS.md` before making automated changes.