# Chebyshev-center Navier–Stokes experiments

Paper: **Chebyshev Center-Based Direction Selection for Multi-Objective Optimization and Training PINNs** (Yoon et al., 2026), arXiv:2605.09975.

Files:

- `chebyshev_runner.py` — reusable 4-loss implementation for both `p=2` and `p=4`.
- `navier_stokes_chebyshev2_runner_serverpath.ipynb` — server entry point for Chebyshev-2.
- `navier_stokes_chebyshev4_runner_serverpath.ipynb` — server entry point for Chebyshev-4.

The runner uses the same PINNsFormer Navier–Stokes setup as the existing HARMONIC/M-ConFIG comparison: seeds 0/1/2, 800 sampled training points, sequence length 5, 1000 Adam steps, `lr=1e-4`, and four objectives (`u_data`, `v_data`, `f_u`, `f_v`).

Algorithm path:

```text
fresh task gradients
  -> l_p normalization
  -> simplex dual Frank-Wolfe solve
  -> primal direction recovery
  -> adaptive scalar from sum_i g_i^T v
  -> Adam
```

The Frank-Wolfe solver uses exact one-dimensional line search for the supported `p=2` and `p=4` cases and warm-starts the simplex weights from the previous training step. Raw outputs are written below `pinnsformer-config/outputs/` and should not be committed.
