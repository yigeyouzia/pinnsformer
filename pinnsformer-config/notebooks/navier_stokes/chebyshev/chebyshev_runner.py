from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

DEFAULT_EXPERIMENT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(os.environ.get("PINNSFORMER_CONFIG_ROOT", str(DEFAULT_EXPERIMENT_ROOT))).resolve()
PINNSFORMER_ROOT = Path(os.environ.get("PINNSFORMER_ROOT", str(EXPERIMENT_ROOT.parent))).resolve()
DATA_PATH = PINNSFORMER_ROOT / "demo" / "navier_stokes" / "cylinder_nektar_wake.mat"

if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from gradient_diagnostics import gradient_vector, pairwise_cosine_matrix
from navier_stokes_common import (
    PINNsformer,
    compute_ns_losses,
    evaluation_tensors,
    get_n_params,
    init_weights,
    load_training_data,
)

LOSS_KEYS = ("u_data", "v_data", "f_u", "f_v")
PAPER_URL = "https://arxiv.org/abs/2605.09975"


@dataclass
class ExperimentConfig:
    p: int
    seeds: Tuple[int, ...] = (0, 1, 2)
    epochs: int = 1000
    lr: float = 1e-4
    n_train: int = 800
    num_step: int = 5
    time_step: float = 1e-2
    snapshot: int = 100
    metric_interval: int = 10
    fw_max_iters: int = 100
    fw_tol: float = 1e-6
    stop_tol: float = 1e-6
    eps: float = 1e-12
    force_rerun: bool = False
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        if self.p not in (2, 4):
            raise ValueError("This experiment runner intentionally supports p=2 or p=4 only.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.fw_max_iters <= 0:
            raise ValueError("fw_max_iters must be positive")

    @property
    def method_key(self) -> str:
        return f"chebyshev{self.p}"

    @property
    def method_name(self) -> str:
        return f"PINNsFormer / Chebyshev-{self.p}"

    @property
    def output_root(self) -> Path:
        return EXPERIMENT_ROOT / "outputs" / f"navier_stokes_chebyshev{self.p}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_model(device: str) -> PINNsformer:
    model = PINNsformer(d_out=2, d_hidden=512, d_model=32, N=1, heads=2).to(device)
    model.apply(init_weights)
    return model


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rel_l2(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> float:
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)
    denom = float(np.linalg.norm(true))
    return float(np.linalg.norm(pred - true) / max(denom, eps))


def _lp_norm(x: torch.Tensor, p: int, eps: float = 0.0) -> torch.Tensor:
    if p == 2:
        out = torch.linalg.vector_norm(x, ord=2)
    elif p == 4:
        out = torch.linalg.vector_norm(x, ord=4)
    else:
        raise ValueError(f"Unsupported p={p}")
    return out.clamp_min(eps) if eps > 0 else out


def _recover_primal(w: torch.Tensor, p: int, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Recover the l_q-unit primal direction from the dual aggregate w."""
    w_norm = _lp_norm(w, p)
    if float(w_norm.detach().item()) <= eps:
        return torch.zeros_like(w), w_norm
    if p == 2:
        v = w / w_norm
    elif p == 4:
        v = torch.sign(w) * torch.abs(w).pow(3) / w_norm.pow(3)
    else:
        raise ValueError(f"Unsupported p={p}")
    return v, w_norm


def _exact_fw_line_search(w: torch.Tensor, vertex: torch.Tensor, p: int, eps: float) -> float:
    """Exact 1-D Frank-Wolfe line search for p=2 and p=4."""
    d = vertex - w
    if p == 2:
        denom = torch.dot(d, d)
        if float(denom.detach().item()) <= eps:
            return 0.0
        gamma = -torch.dot(w, d) / denom
        return float(gamma.clamp(0.0, 1.0).detach().item())

    if p == 4:
        # Minimize ||w + gamma d||_4^4. Its derivative/4 is a monotone cubic.
        c3 = float(torch.sum(d.pow(4)).detach().item())
        c2 = float((3.0 * torch.sum(w * d.pow(3))).detach().item())
        c1 = float((3.0 * torch.sum(w.pow(2) * d.pow(2))).detach().item())
        c0 = float(torch.sum(w.pow(3) * d).detach().item())

        def h(gamma: float) -> float:
            return ((c3 * gamma + c2) * gamma + c1) * gamma + c0

        if h(0.0) >= 0.0:
            return 0.0
        if h(1.0) <= 0.0:
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if h(mid) <= 0.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    raise ValueError(f"Unsupported p={p}")


def solve_chebyshev_dual(
    normalized_grads: torch.Tensor,
    p: int,
    *,
    max_iters: int = 100,
    tol: float = 1e-6,
    eps: float = 1e-12,
    alpha0: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Solve min_{alpha in simplex} ||sum_i alpha_i ghat_i||_p by Frank-Wolfe."""
    if normalized_grads.ndim != 2:
        raise ValueError("normalized_grads must have shape [m, n]")
    m = normalized_grads.shape[0]
    if m < 2:
        raise ValueError("At least two objectives are required")

    if alpha0 is None:
        alpha = torch.full(
            (m,), 1.0 / m, device=normalized_grads.device, dtype=normalized_grads.dtype
        )
    else:
        alpha = alpha0.to(device=normalized_grads.device, dtype=normalized_grads.dtype).clone()
        alpha = alpha.clamp_min(0)
        alpha = alpha / alpha.sum().clamp_min(eps)

    final_gap = float("inf")
    iterations = 0
    for k in range(max_iters):
        iterations = k + 1
        w = torch.sum(alpha[:, None] * normalized_grads, dim=0)
        v, w_norm = _recover_primal(w, p, eps)
        if float(w_norm.detach().item()) <= eps:
            final_gap = 0.0
            break

        grad_alpha = normalized_grads @ v
        vertex_index = int(torch.argmin(grad_alpha).item())
        final_gap = float((torch.dot(alpha, grad_alpha) - grad_alpha[vertex_index]).detach().item())
        if final_gap <= tol:
            break

        gamma = _exact_fw_line_search(w, normalized_grads[vertex_index], p, eps)
        if gamma <= eps:
            break
        alpha.mul_(1.0 - gamma)
        alpha[vertex_index] += gamma

    w = torch.sum(alpha[:, None] * normalized_grads, dim=0)
    _, w_norm = _recover_primal(w, p, eps)
    return alpha, w, {
        "fw_iters": float(iterations),
        "fw_gap": float(final_gap),
        "dual_norm": float(w_norm.detach().item()),
    }


def chebyshev_direction(
    raw_grads: torch.Tensor,
    p: int,
    *,
    fw_max_iters: int,
    fw_tol: float,
    stop_tol: float,
    eps: float,
    alpha0: torch.Tensor | None = None,
) -> Tuple[torch.Tensor | None, torch.Tensor, Dict[str, object]]:
    """Algorithm 1 from Yoon et al. (2026), including the adaptive scalar."""
    if raw_grads.ndim != 2:
        raise ValueError("raw_grads must have shape [m, n]")
    task_norms = torch.linalg.vector_norm(raw_grads, ord=p, dim=1)
    normalized = raw_grads / task_norms.clamp_min(eps)[:, None]

    alpha, w, solver_info = solve_chebyshev_dual(
        normalized,
        p,
        max_iters=fw_max_iters,
        tol=fw_tol,
        eps=eps,
        alpha0=alpha0,
    )
    v, w_norm = _recover_primal(w, p, eps)
    if float(w_norm.detach().item()) <= stop_tol:
        return None, alpha, {
            **solver_info,
            "pareto_stationary": True,
            "adaptive_scalar": 0.0,
            "min_raw_alignment": 0.0,
            "min_normalized_alignment": 0.0,
        }

    raw_alignment = raw_grads @ v
    normalized_alignment = normalized @ v
    adaptive_scalar = raw_alignment.sum()
    direction = adaptive_scalar * v

    info: Dict[str, object] = {
        **solver_info,
        "pareto_stationary": False,
        "adaptive_scalar": float(adaptive_scalar.detach().item()),
        "min_raw_alignment": float(raw_alignment.min().detach().item()),
        "min_normalized_alignment": float(normalized_alignment.min().detach().item()),
        "mean_normalized_alignment": float(normalized_alignment.mean().detach().item()),
        "direction_l2_norm": float(torch.linalg.vector_norm(direction, ord=2).detach().item()),
        "alpha": [float(x) for x in alpha.detach().cpu().tolist()],
        "task_p_norms": [float(x) for x in task_norms.detach().cpu().tolist()],
    }
    return direction, alpha, info


@torch.no_grad()
def apply_gradient_vector(model: torch.nn.Module, grad_vec: torch.Tensor) -> None:
    offset = 0
    for param in model.parameters():
        if not param.requires_grad:
            continue
        n = param.numel()
        param.grad = grad_vec[offset : offset + n].view_as(param).clone()
        offset += n
    if offset != grad_vec.numel():
        raise RuntimeError(f"Gradient vector length mismatch: used {offset}, got {grad_vec.numel()}")


def compute_four_gradients(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]):
    losses = compute_ns_losses(
        model, batch["x"], batch["y"], batch["t"], batch["u"], batch["v"]
    )
    grads: Dict[str, torch.Tensor] = {}
    for i, key in enumerate(LOSS_KEYS):
        grads[key] = gradient_vector(
            losses[key], model, retain_graph=(i < len(LOSS_KEYS) - 1)
        ).detach()
    return losses, grads


def gradient_stats(grads: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    names, matrix = pairwise_cosine_matrix(grads)
    pairs: List[float] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            value = float(matrix[i, j])
            if np.isfinite(value):
                pairs.append(value)
    return {
        "raw_cosine_mean": float(np.mean(pairs)) if pairs else float("nan"),
        "raw_cosine_min": float(np.min(pairs)) if pairs else float("nan"),
        "raw_conflict_rate": float(np.mean(np.asarray(pairs) < 0)) if pairs else float("nan"),
    }


def evaluate_model(model: torch.nn.Module, reference: Mapping[str, np.ndarray], cfg: ExperimentConfig):
    model.eval()
    tensors, truth = evaluation_tensors(
        reference,
        cfg.device,
        snap=cfg.snapshot,
        num_step=cfg.num_step,
        time_step=cfg.time_step,
    )
    psi_and_p = model(tensors["x"], tensors["y"], tensors["t"])
    psi = psi_and_p[:, :, 0:1]
    p_pred = psi_and_p[:, 0, 1].detach().cpu().numpy().reshape(-1)
    u_pred = torch.autograd.grad(
        psi,
        tensors["y"],
        grad_outputs=torch.ones_like(psi),
        retain_graph=True,
        create_graph=False,
    )[0][:, 0].detach().cpu().numpy().reshape(-1)
    v_pred = -torch.autograd.grad(
        psi,
        tensors["x"],
        grad_outputs=torch.ones_like(psi),
        retain_graph=False,
        create_graph=False,
    )[0][:, 0].detach().cpu().numpy().reshape(-1)

    u_true = np.asarray(truth["u"]).reshape(-1)
    v_true = np.asarray(truth["v"]).reshape(-1)
    p_true = np.asarray(truth["p"]).reshape(-1)
    p_aligned = p_pred - float(np.mean(p_pred - p_true))
    return {
        "rel_l2_u": rel_l2(u_pred, u_true),
        "rel_l2_v": rel_l2(v_pred, v_true),
        "rel_l2_p_aligned": rel_l2(p_aligned, p_true),
    }


def run_seed(cfg: ExperimentConfig, seed: int) -> Dict[str, object]:
    run_dir = cfg.output_root / f"seed_{seed}"
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists() and not cfg.force_rerun:
        with metrics_path.open("r", newline="", encoding="utf-8") as f:
            cached = list(csv.DictReader(f))
        if cached:
            row = dict(cached[-1])
            row["cached"] = True
            return row

    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    batch, reference = load_training_data(
        DATA_PATH,
        cfg.device,
        seed=seed,
        n_train=cfg.n_train,
        num_step=cfg.num_step,
        time_step=cfg.time_step,
    )
    model = make_model(cfg.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8
    )

    history: List[Dict[str, object]] = []
    alpha_prev: torch.Tensor | None = None
    stationary_step: int | None = None
    solver_seconds = 0.0

    sync_cuda()
    started = time.perf_counter()
    for step in tqdm(range(cfg.epochs), desc=f"Chebyshev-{cfg.p} seed={seed}"):
        optimizer.zero_grad(set_to_none=True)
        losses, grads = compute_four_gradients(model, batch)
        gstack = torch.stack([grads[k] for k in LOSS_KEYS], dim=0)

        t0 = time.perf_counter()
        direction, alpha, info = chebyshev_direction(
            gstack,
            cfg.p,
            fw_max_iters=cfg.fw_max_iters,
            fw_tol=cfg.fw_tol,
            stop_tol=cfg.stop_tol,
            eps=cfg.eps,
            alpha0=alpha_prev,
        )
        sync_cuda()
        solver_seconds += time.perf_counter() - t0
        alpha_prev = alpha.detach()

        if direction is None:
            stationary_step = step
        else:
            if not torch.isfinite(direction).all():
                raise FloatingPointError(f"Non-finite Chebyshev direction at step={step}, seed={seed}")
            apply_gradient_vector(model, direction)
            optimizer.step()

        if step % cfg.metric_interval == 0 or step == cfg.epochs - 1 or direction is None:
            stats = gradient_stats(grads)
            row: Dict[str, object] = {
                "step": step,
                "p": cfg.p,
                "loss_total": float(losses["total"].detach().item()),
                "loss_u_data": float(losses["u_data"].detach().item()),
                "loss_v_data": float(losses["v_data"].detach().item()),
                "loss_f_u": float(losses["f_u"].detach().item()),
                "loss_f_v": float(losses["f_v"].detach().item()),
                "loss_physics": float(losses["physics"].detach().item()),
                **stats,
                "fw_iters": info["fw_iters"],
                "fw_gap": info["fw_gap"],
                "dual_norm": info["dual_norm"],
                "adaptive_scalar": info["adaptive_scalar"],
                "min_raw_alignment": info["min_raw_alignment"],
                "min_normalized_alignment": info["min_normalized_alignment"],
            }
            for i, key in enumerate(LOSS_KEYS):
                row[f"alpha_{key}"] = float(alpha[i].detach().item())
                row[f"grad_l2_{key}"] = float(torch.linalg.vector_norm(grads[key], ord=2).item())
            history.append(row)

        if direction is None:
            break

    sync_cuda()
    wall_seconds = time.perf_counter() - started

    eval_metrics = evaluate_model(model, reference, cfg)
    final_losses = compute_ns_losses(
        model, batch["x"], batch["y"], batch["t"], batch["u"], batch["v"]
    )
    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
    )
    summary: Dict[str, object] = {
        "method": cfg.method_key,
        "p": cfg.p,
        "seed": seed,
        "steps_completed": (stationary_step + 1) if stationary_step is not None else cfg.epochs,
        "stationary_step": stationary_step if stationary_step is not None else "",
        "wall_seconds": wall_seconds,
        "solver_seconds": solver_seconds,
        "peak_vram_mb": peak_vram_mb,
        "n_params": get_n_params(model),
        "final_total_loss": float(final_losses["total"].detach().item()),
        "final_physics_loss": float(final_losses["physics"].detach().item()),
        **eval_metrics,
        "cached": False,
    }

    write_rows(run_dir / "history.csv", history)
    write_rows(metrics_path, [summary])
    torch.save(model.state_dict(), run_dir / "model.pt")
    with (run_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "paper": PAPER_URL,
                "config": asdict(cfg),
                "summary": summary,
                "loss_keys": list(LOSS_KEYS),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return summary


def _as_float(row: Mapping[str, object], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def aggregate_rows(rows: Sequence[Mapping[str, object]], cfg: ExperimentConfig) -> List[Dict[str, object]]:
    keys = [
        "rel_l2_u",
        "rel_l2_v",
        "rel_l2_p_aligned",
        "final_total_loss",
        "final_physics_loss",
        "wall_seconds",
        "solver_seconds",
        "peak_vram_mb",
    ]
    out: List[Dict[str, object]] = []
    for key in keys:
        values = np.asarray([_as_float(r, key) for r in rows], dtype=float)
        values = values[np.isfinite(values)]
        out.append(
            {
                "method": cfg.method_key,
                "p": cfg.p,
                "metric": key,
                "n": int(values.size),
                "mean": float(values.mean()) if values.size else float("nan"),
                "std": float(values.std(ddof=1)) if values.size > 1 else (0.0 if values.size == 1 else float("nan")),
                "min": float(values.min()) if values.size else float("nan"),
                "max": float(values.max()) if values.size else float("nan"),
            }
        )
    return out


def run_multiseed(cfg: ExperimentConfig) -> List[Dict[str, object]]:
    if not EXPERIMENT_ROOT.exists():
        raise FileNotFoundError(f"Experiment root not found: {EXPERIMENT_ROOT}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Navier-Stokes dataset not found: {DATA_PATH}")
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    print("METHOD:", cfg.method_name)
    print("PINNSFORMER_ROOT:", PINNSFORMER_ROOT)
    print("EXPERIMENT_ROOT:", EXPERIMENT_ROOT)
    print("DATA_PATH:", DATA_PATH)
    print("OUTPUT_ROOT:", cfg.output_root)
    print("DEVICE:", cfg.device)
    print("SEEDS:", cfg.seeds)
    print("EPOCHS:", cfg.epochs, "LR:", cfg.lr)
    print("FW max_iters/tol:", cfg.fw_max_iters, cfg.fw_tol)

    rows = [run_seed(cfg, seed) for seed in cfg.seeds]
    write_rows(cfg.output_root / "all_seed_metrics.csv", rows)
    aggregate = aggregate_rows(rows, cfg)
    write_rows(cfg.output_root / "mean_std_summary.csv", aggregate)
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PINNsFormer Navier-Stokes Chebyshev-center runner")
    parser.add_argument("--p", type=int, choices=(2, 4), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fw-max-iters", type=int, default=100)
    parser.add_argument("--fw-tol", type=float, default=1e-6)
    parser.add_argument("--stop-tol", type=float, default=1e-6)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        p=args.p,
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        fw_max_iters=args.fw_max_iters,
        fw_tol=args.fw_tol,
        stop_tol=args.stop_tol,
        force_rerun=args.force_rerun,
    )
    run_multiseed(cfg)


if __name__ == "__main__":
    main()
