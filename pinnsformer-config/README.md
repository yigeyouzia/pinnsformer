# PINNsFormer multi-objective PINN experiments

`pinnsformer-config/` is the executable experiment area for multi-loss / multi-objective optimization on PINNsFormer. The separate `ts-pinn-agent` repository remains the paper/knowledge-base/agent-management repository.

## Workflow rule

This repository uses **`main` only** for ongoing work. Agents should not create feature or experiment branches unless the user explicitly overrides this rule.

## Layout

```text
pinnsformer-config/
├── README.md
├── AGENTS.md
├── .gitignore
├── gradient_diagnostics.py
├── navier_stokes_common.py
├── notebooks/
│   └── navier_stokes/
│       ├── adam/
│       ├── config/
│       ├── mconfig/
│       ├── autobalance/
│       ├── harmonic/
│       ├── chebyshev/
│       └── analysis/
├── outputs/                 # generated runs; ignored
└── results/
    └── summaries/           # curated checked aggregate CSVs
```

## Current method inventory

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

- `notebooks/navier_stokes/chebyshev/chebyshev_runner.py` — shared implementation for `p=2` and `p=4`.
- `notebooks/navier_stokes/chebyshev/navier_stokes_chebyshev2_runner_serverpath.ipynb`
- `notebooks/navier_stokes/chebyshev/navier_stokes_chebyshev4_runner_serverpath.ipynb`

The Chebyshev runner uses four objectives (`u_data`, `v_data`, `f_u`, `f_v`) to match the current HARMONIC/M-ConFIG comparison line. It follows the Chebyshev dual formulation and adaptive direction scaling from Yoon et al. (2026), arXiv:2605.09975. `p=2` is the primary candidate; `p=4` is retained as a required geometry ablation.

### Analysis / diagnostics

- `notebooks/navier_stokes/analysis/navier_stokes_gradient_diagnostic.ipynb`
- `notebooks/navier_stokes/analysis/navier_stokes_adam_vs_config_analysis.ipynb`
- `notebooks/navier_stokes/analysis/cylinder_nektar_wake_dataset_explorer.ipynb`
- `notebooks/navier_stokes/analysis/plot.ipynb`

## Fair comparison defaults

Current HARMONIC/Chebyshev comparison defaults are:

```text
PINNsFormer(d_out=2, d_hidden=512, d_model=32, N=1, heads=2)
N_TRAIN=800
pseudo sequence=5
TIME_STEP=1e-2
seeds=0,1,2
optimizer=Adam
lr=1e-4
steps=1000
objectives=u_data,v_data,f_u,f_v
```

The Chebyshev notebooks intentionally inherit this protocol instead of reproducing the paper's MLP benchmark architecture, because the purpose here is a controlled solver comparison on the existing PINNsFormer Navier–Stokes experiment.

## Server paths

```bash
export PINNSFORMER_ROOT=/home/simplexity/cyt/pinnsformer-main
export CONFIG_ROOT=/home/simplexity/cyt/ConFIG-main
export PINNSFORMER_CONFIG_ROOT=$PINNSFORMER_ROOT/pinnsformer-config
```

Dataset:

```text
$PINNSFORMER_ROOT/demo/navier_stokes/cylinder_nektar_wake.mat
```

## Running Chebyshev

Notebook entry points:

```text
notebooks/navier_stokes/chebyshev/navier_stokes_chebyshev2_runner_serverpath.ipynb
notebooks/navier_stokes/chebyshev/navier_stokes_chebyshev4_runner_serverpath.ipynb
```

Or run the shared Python runner directly:

```bash
python pinnsformer-config/notebooks/navier_stokes/chebyshev/chebyshev_runner.py --p 2
python pinnsformer-config/notebooks/navier_stokes/chebyshev/chebyshev_runner.py --p 4
```

Raw runs are written to `outputs/navier_stokes_chebyshev2/` and `outputs/navier_stokes_chebyshev4/` and are ignored by git. Promote only checked aggregate tables to `results/summaries/`.

See `AGENTS.md` before making automated changes.
