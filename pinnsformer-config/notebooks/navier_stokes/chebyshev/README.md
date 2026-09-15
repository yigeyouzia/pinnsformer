# Chebyshev-center Navier–Stokes experiments

Paper: **Chebyshev Center-Based Direction Selection for Multi-Objective Optimization and Training PINNs** (Yoon et al., 2026), arXiv:2605.09975.

## Files

- `navier_stokes_chebyshev2_runner_serverpath.ipynb` — self-contained Chebyshev-2 experiment notebook.
- `navier_stokes_chebyshev4_runner_serverpath.ipynb` — self-contained Chebyshev-4 experiment notebook.

Both notebooks follow the existing HARMONIC / M-ConFIG notebook style: configuration, Chebyshev implementation, 3-seed training, CSV aggregation, diagnostics, and plotting all live in the notebook. Do not replace them with thin wrapper notebooks that only import an external runner.

Current memory-constrained default is `N_TRAIN=600`, with seeds `0/1/2`, sequence length 5, 1000 Adam steps, `lr=1e-4`, and four objectives (`u_data`, `v_data`, `f_u`, `f_v`). Change `N_TRAIN` in the configuration cell when running an 800-point benchmark.

Algorithm path:

```text
fresh task gradients
  -> l_p normalization
  -> simplex dual Frank-Wolfe solve
  -> primal direction recovery
  -> adaptive scalar from sum_i g_i^T v
  -> Adam
```

Each notebook writes raw experiment artifacts below `pinnsformer-config/outputs/navier_stokes_chebyshev{2|4}/`. The plotting cell generates per-seed loss, dual-alpha, gradient-geometry, direction-alignment and solver curves, plus final 3-seed Relative-L2 mean±std.
