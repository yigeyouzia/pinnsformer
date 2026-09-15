# PINNsFormer multi-objective PINN experiments

`pinnsformer-config/` is the executable experiment area for multi-loss / multi-objective optimization on PINNsFormer. The separate `ts-pinn-agent` repository remains the paper/knowledge-base/agent-management repository.

## Workflow rule

This repository uses **`main` only** for ongoing work. Agents should not create feature or experiment branches unless the user explicitly overrides this rule.

Experiment methods are notebook-first: method-specific algorithm code, training, multi-seed aggregation and plotting should remain visible in the runnable `.ipynb`, following the existing HARMONIC / M-ConFIG style. Stable shared helpers such as `navier_stokes_common.py` and `gradient_diagnostics.py` may remain external.

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
- `notebooks/navier_stokes/chebyshev/navier_stokes_chebyshev2_runner_serverpath.ipynb`
- `notebooks/navier_stokes/chebyshev/navier_stokes_chebyshev4_runner_serverpath.ipynb`

Both Chebyshev notebooks are self-contained. They include the Chebyshev solver, 4-loss training loop, 3-seed aggregation and plotting directly inside the notebook. `p=2` is the primary candidate; `p=4` is the norm-geometry ablation.

Current memory-constrained Chebyshev setting is:

```text
PINNsFormer(d_out=2, d_hidden=512, d_model=32, N=1, heads=2)
N_TRAIN=600
pseudo sequence=5
TIME_STEP=1e-2
seeds=0,1,2
optimizer=Adam
lr=1e-4
steps=1000
objectives=u_data,v_data,f_u,f_v
```

The previous ConFIG/HARMONIC 800-point results remain a separate benchmark. Do not present 600-point Chebyshev vs 800-point baselines as a strict method-only comparison without labeling the sampling difference.

## Chebyshev outputs and plots

Raw runs are written to:

```text
outputs/navier_stokes_chebyshev2/
outputs/navier_stokes_chebyshev4/
```

Each notebook produces per-seed history/metrics plus aggregate `mean_std_summary.csv`. The plotting cell generates:

- component-loss + physics curves;
- dual `alpha` trajectories;
- raw gradient cosine geometry;
- Chebyshev direction alignment;
- Frank-Wolfe gap / dual-norm diagnostics;
- final 3-seed Relative-L2 mean ± std.

Generated CSV/PNG/checkpoint files under `outputs/` are ignored by git. Promote only checked aggregate tables to `results/summaries/`.

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

See `AGENTS.md` before making automated changes.
