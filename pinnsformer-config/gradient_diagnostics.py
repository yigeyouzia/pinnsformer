"""Gradient diagnostics for multi-loss PINN / PINNsFormer experiments.

This module is intentionally optimizer-agnostic. It measures parameter-gradient
norms, pairwise cosine similarities and conflict rates. It can also flatten
loss-specific gradients in a stable parameter order for ConFIG.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch


def gradient_vector(
    loss: torch.Tensor,
    model: torch.nn.Module,
    *,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> torch.Tensor:
    """Return d(loss)/d(theta) as one flat vector, padding unused params with zero."""
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )
    flat = []
    for p, g in zip(params, grads):
        flat.append(torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1))
    return torch.cat(flat)


def cosine_similarity(g1: torch.Tensor, g2: torch.Tensor, eps: float = 1e-12) -> float:
    """Cosine similarity between two flat gradient vectors."""
    with torch.no_grad():
        denom = g1.norm() * g2.norm()
        if denom <= eps:
            return float("nan")
        return float(torch.dot(g1, g2).div(denom).item())


def pairwise_cosine_matrix(gradients: Mapping[str, torch.Tensor], eps: float = 1e-12):
    """Return (names, cosine_matrix) for a mapping of loss name -> gradient vector."""
    names = list(gradients.keys())
    n = len(names)
    matrix = np.full((n, n), np.nan, dtype=np.float64)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i == j:
                matrix[i, j] = 1.0
            else:
                matrix[i, j] = cosine_similarity(gradients[ni], gradients[nj], eps=eps)
    return names, matrix


def summarize_gradients(gradients: Mapping[str, torch.Tensor]):
    """Return gradient norms and pairwise cosine diagnostics."""
    names, matrix = pairwise_cosine_matrix(gradients)
    norms = {name: float(gradients[name].detach().norm().item()) for name in names}
    off_diag = [
        matrix[i, j]
        for i in range(len(names))
        for j in range(i + 1, len(names))
        if np.isfinite(matrix[i, j])
    ]
    conflict_pairs = sum(v < 0 for v in off_diag)
    total_pairs = len(off_diag)
    return {
        "names": names,
        "norms": norms,
        "cosine_matrix": matrix,
        "conflict_pairs": int(conflict_pairs),
        "total_pairs": int(total_pairs),
        "pair_conflict_rate": float(conflict_pairs / total_pairs) if total_pairs else float("nan"),
    }


class ConflictTracker:
    """Accumulate step-wise cosine/conflict statistics for a chosen pair of losses."""

    def __init__(self, left: str, right: str):
        self.left = left
        self.right = right
        self.cosines: list[float] = []

    def update(self, gradients: Mapping[str, torch.Tensor]) -> float:
        value = cosine_similarity(gradients[self.left], gradients[self.right])
        self.cosines.append(value)
        return value

    @property
    def conflict_rate(self) -> float:
        valid = [v for v in self.cosines if np.isfinite(v)]
        if not valid:
            return float("nan")
        return float(np.mean(np.asarray(valid) < 0))

    def summary(self):
        valid = np.asarray([v for v in self.cosines if np.isfinite(v)], dtype=np.float64)
        if len(valid) == 0:
            return {
                "n": 0,
                "mean_cosine": float("nan"),
                "min_cosine": float("nan"),
                "max_cosine": float("nan"),
                "conflict_rate": float("nan"),
            }
        return {
            "n": int(len(valid)),
            "mean_cosine": float(valid.mean()),
            "min_cosine": float(valid.min()),
            "max_cosine": float(valid.max()),
            "conflict_rate": float(np.mean(valid < 0)),
        }
