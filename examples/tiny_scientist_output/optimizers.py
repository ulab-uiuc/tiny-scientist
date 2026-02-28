"""Step-size policies + optimizer loop (Step 4/6).

Implements a shared optimizer API for convex quadratics with an oracle that
accounts for gradient/value calls.

Policies implemented (10-12):
- constant step (grid on eta)
- diminishing eta0/sqrt(t)
- exact line search (quadratic, deterministic)
- Armijo backtracking (counts value calls)
- (optional) Wolfe line search (not implemented; stub with explicit message)
- Barzilai-Borwein BB1 / BB2 (with safeguards/clipping)
- Polyak step using f* in {true, true+delta} with delta in {1e-6,1e-3,1e-1}
- RMSProp scalar (global 2nd moment)
- Adam-like scalar (global moments)
- diagonal RMSProp preconditioner (per-coordinate 2nd moment)

Also includes hyperparameter grids EXACTLY per spec and a tuning routine that
selects best hyperparameters under equal oracle-budget validation per regime.

Step 5 addition:
- run_optimizer_under_budget now returns the final iterate xT for secondary
  diagnostics (Q-basis error) and meta-analysis.

Note: This step focuses on the quadratic benchmark; SST-2 training comes later.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from oracle import NoisyQuadraticOracle


PolicyName = Literal[
    "constant",
    "diminishing",
    "exact_linesearch",
    "armijo",
    "bb1",
    "bb2",
    "polyak",
    "rmsprop_scalar",
    "adam_scalar",
    "diag_rmsprop",
]


def _logspace(a: float, b: float, n: int) -> List[float]:
    return [float(x) for x in np.logspace(np.log10(a), np.log10(b), int(n), dtype=np.float64).tolist()]


def hyperparameter_grids() -> Dict[str, List[Dict[str, Any]]]:
    """Hyperparameter grids EXACTLY as specified in the blueprint."""

    grids: Dict[str, List[Dict[str, Any]]] = {}

    # constant η ∈ logspace(1e-6, 1e1, 30)
    grids["constant"] = [{"eta": eta} for eta in _logspace(1e-6, 1e1, 30)]

    # diminishing η_t = η0 / sqrt(t)
    grids["diminishing"] = [{"eta0": eta0} for eta0 in _logspace(1e-6, 1e1, 30)]

    # Armijo c1∈{1e-4,1e-2}, shrink β∈{0.5,0.8}
    grids["armijo"] = [
        {"c1": c1, "beta": beta, "eta_init": 1.0, "max_backtracks": 50}
        for c1 in [1e-4, 1e-2]
        for beta in [0.5, 0.8]
    ]

    # BB safeguard clip η∈[1e-12,1e12]
    grids["bb1"] = [{"clip_min": 1e-12, "clip_max": 1e12, "eta_fallback": 1.0}]
    grids["bb2"] = [{"clip_min": 1e-12, "clip_max": 1e12, "eta_fallback": 1.0}]

    # Polyak f* variants
    grids["polyak"] = [{"fstar_delta": dlt} for dlt in [0.0, 1e-6, 1e-3, 1e-1]]

    # Adam/RMSProp base lr grids
    base_lrs = _logspace(1e-6, 1e0, 20)
    grids["rmsprop_scalar"] = [{"lr": lr, "beta2": beta2, "eps": 1e-8} for beta2 in [0.99, 0.999] for lr in base_lrs]
    grids["adam_scalar"] = [{"lr": lr, "beta1": 0.9, "beta2": beta2, "eps": 1e-8} for beta2 in [0.99, 0.999] for lr in base_lrs]
    grids["diag_rmsprop"] = [{"lr": lr, "beta2": beta2, "eps": 1e-8} for beta2 in [0.99, 0.999] for lr in base_lrs]

    grids["exact_linesearch"] = [{}]

    return grids


@dataclasses.dataclass
class OptimizerState:
    t: int = 0
    x_prev: Optional[np.ndarray] = None
    g_prev: Optional[np.ndarray] = None
    # moments
    m: float = 0.0
    v: float = 0.0
    # diagonal moments
    v_diag: Optional[np.ndarray] = None


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _exact_linesearch_eta(inst_H: np.ndarray, g: np.ndarray) -> float:
    Hg = inst_H @ g
    num = float(g @ g)
    den = float(g @ Hg)
    if den <= 0 or not np.isfinite(den) or num <= 0 or not np.isfinite(num):
        return 0.0
    return num / den


def _armijo_backtracking(
    oracle: NoisyQuadraticOracle,
    x: np.ndarray,
    g: np.ndarray,
    *,
    eta_init: float,
    c1: float,
    beta: float,
    max_backtracks: int,
    budget_units: float,
) -> Tuple[float, bool]:
    """Return (eta, success). Charges value calls through oracle.value."""

    if oracle.budget.total + oracle.budget.alpha_value > budget_units:
        return 0.0, False
    fx = oracle.value(x)
    g2 = float(g @ g)
    if g2 <= 0 or not np.isfinite(g2):
        return 0.0, False

    eta = float(eta_init)
    for _ in range(int(max_backtracks)):
        if oracle.budget.total + oracle.budget.alpha_value > budget_units:
            return 0.0, False
        ftrial = oracle.value(x - eta * g)
        if ftrial <= fx - float(c1) * eta * g2:
            return float(eta), True
        eta *= float(beta)
        if eta == 0.0:
            break
    return float(eta), False


def _bb_stepsize(
    s: np.ndarray,
    y: np.ndarray,
    variant: Literal["bb1", "bb2"],
    *,
    clip_min: float,
    clip_max: float,
    eta_fallback: float,
) -> float:
    sty = float(s @ y)
    if not np.isfinite(sty) or sty <= 0:
        return float(eta_fallback)

    if variant == "bb1":
        sts = float(s @ s)
        if not np.isfinite(sts) or sts <= 0:
            return float(eta_fallback)
        eta = sts / sty
    else:
        yty = float(y @ y)
        if not np.isfinite(yty) or yty <= 0:
            return float(eta_fallback)
        eta = sty / yty

    return _clip(float(eta), float(clip_min), float(clip_max))


def run_optimizer_under_budget(
    *,
    oracle: NoisyQuadraticOracle,
    x0: np.ndarray,
    policy: PolicyName,
    hparams: Dict[str, Any],
    budget_units: float,
    eps_stop: Optional[float],
    log_every: int = 1,
) -> Dict[str, Any]:
    """Run a single optimizer instance under a budget.

    Returns summary + lightweight trajectory stats + final iterate xT.
    """

    inst = oracle.inst
    x = x0.copy()
    st = OptimizerState(t=0)

    f0 = oracle.value(x)
    gap0 = float(f0 - inst.f_star)
    best_gap = gap0

    diverged = False
    div_reason = "ok"
    best_seen_decrease = False

    traj_budget: List[float] = []
    traj_gap: List[float] = []
    traj_eta: List[float] = []

    while oracle.budget.total + 1.0 <= budget_units:
        st.t += 1

        g = oracle.grad(x)

        eta = 0.0

        if policy == "constant":
            eta = float(hparams["eta"])
            x_new = x - eta * g

        elif policy == "diminishing":
            eta0 = float(hparams["eta0"])
            eta = eta0 / np.sqrt(float(st.t))
            x_new = x - eta * g

        elif policy == "exact_linesearch":
            eta = _exact_linesearch_eta(inst.H, g)
            x_new = x - eta * g

        elif policy == "armijo":
            eta_init = float(hparams.get("eta_init", 1.0))
            c1 = float(hparams["c1"])
            beta = float(hparams["beta"])
            max_backtracks = int(hparams.get("max_backtracks", 50))
            eta, _ = _armijo_backtracking(
                oracle,
                x,
                g,
                eta_init=eta_init,
                c1=c1,
                beta=beta,
                max_backtracks=max_backtracks,
                budget_units=budget_units,
            )
            x_new = x - eta * g

        elif policy in ("bb1", "bb2"):
            if st.x_prev is None or st.g_prev is None:
                eta = float(hparams.get("eta_fallback", 1.0))
            else:
                s = x - st.x_prev
                y = g - st.g_prev
                eta = _bb_stepsize(
                    s,
                    y,
                    "bb1" if policy == "bb1" else "bb2",
                    clip_min=float(hparams.get("clip_min", 1e-12)),
                    clip_max=float(hparams.get("clip_max", 1e12)),
                    eta_fallback=float(hparams.get("eta_fallback", 1.0)),
                )
            x_new = x - eta * g

        elif policy == "polyak":
            delta = float(hparams["fstar_delta"])
            fstar_hat = float(inst.f_star + delta)
            if oracle.budget.total + oracle.budget.alpha_value > budget_units:
                break
            fx = oracle.value(x)
            numer = float(fx - fstar_hat)
            denom = float(g @ g)
            if denom <= 0 or not np.isfinite(denom):
                eta = 0.0
            else:
                eta = max(0.0, numer / denom)
            x_new = x - eta * g

        elif policy == "rmsprop_scalar":
            lr = float(hparams["lr"])
            beta2 = float(hparams["beta2"])
            eps = float(hparams["eps"])
            g2 = float(g @ g) / float(g.shape[0])
            st.v = beta2 * st.v + (1.0 - beta2) * g2
            denom = np.sqrt(st.v) + eps
            eta = lr / denom
            x_new = x - eta * g

        elif policy == "adam_scalar":
            lr = float(hparams["lr"])
            beta1 = float(hparams["beta1"])
            beta2 = float(hparams["beta2"])
            eps = float(hparams["eps"])
            gmean = float(np.mean(g))
            g2 = float(g @ g) / float(g.shape[0])
            st.m = beta1 * st.m + (1.0 - beta1) * gmean
            st.v = beta2 * st.v + (1.0 - beta2) * g2
            vhat = st.v / (1.0 - beta2**st.t)
            eta = lr / (np.sqrt(vhat) + eps)
            x_new = x - eta * g

        elif policy == "diag_rmsprop":
            lr = float(hparams["lr"])
            beta2 = float(hparams["beta2"])
            eps = float(hparams["eps"])
            if st.v_diag is None:
                st.v_diag = np.zeros_like(x)
            st.v_diag = beta2 * st.v_diag + (1.0 - beta2) * (g * g)
            precond = 1.0 / (np.sqrt(st.v_diag) + eps)
            eta = float(lr)
            x_new = x - lr * (precond * g)

        else:
            raise ValueError(f"Unknown policy: {policy}")

        st.x_prev = x
        st.g_prev = g
        x = x_new

        if not np.all(np.isfinite(x)) or float(np.linalg.norm(x)) > 1e12:
            diverged = True
            div_reason = "non_finite" if not np.all(np.isfinite(x)) else "norm_exceeded"
            break

        if (st.t % int(log_every)) == 0:
            if oracle.budget.total + oracle.budget.alpha_value > budget_units:
                break
            f = oracle.value(x)
            gap = float(f - inst.f_star)
            if gap < gap0:
                best_seen_decrease = True
            best_gap = min(best_gap, gap)
            traj_budget.append(float(oracle.budget.total))
            traj_gap.append(float(gap))
            traj_eta.append(float(eta))

            if eps_stop is not None and best_gap <= float(eps_stop):
                break

    fT = None
    if not diverged and oracle.budget.total + oracle.budget.alpha_value <= budget_units:
        fT = float(oracle.value(x))

    return {
        "policy": str(policy),
        "hparams": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in hparams.items()},
        "iterations": int(st.t),
        "budget": oracle.budget.state_dict(),
        "f0": float(f0),
        "gap0": float(gap0),
        "fT": fT,
        "gapT": None if fT is None else float(fT - inst.f_star),
        "best_gap": float(best_gap),
        "diverged": bool(diverged),
        "divergence_reason": str(div_reason),
        "saw_decrease_first_K": bool(best_seen_decrease),
        "traj": {"budget": traj_budget, "gap": traj_gap, "eta": traj_eta},
        "xT": x.astype(float).tolist(),
    }


def tune_hyperparameters_under_budget(
    *,
    make_oracle_fn,
    x0: np.ndarray,
    policy: PolicyName,
    grid: List[Dict[str, Any]],
    validation_budget_units: float,
    validation_seeds: List[int],
    eps_stop: Optional[float],
) -> Dict[str, Any]:
    """Select best hyperparams by median best_gap across validation seeds."""

    cand_results: List[Dict[str, Any]] = []

    for hp in grid:
        gaps: List[float] = []
        diverged_any = False
        for sd in validation_seeds:
            oracle = make_oracle_fn(sd)
            info = run_optimizer_under_budget(
                oracle=oracle,
                x0=x0,
                policy=policy,
                hparams=hp,
                budget_units=validation_budget_units,
                eps_stop=eps_stop,
                log_every=5,
            )
            diverged_any = diverged_any or bool(info["diverged"])
            gaps.append(float(info["best_gap"]))
        score = float(np.median(np.asarray(gaps, dtype=np.float64)))
        cand_results.append({"hparams": hp, "median_best_gap": score, "diverged_any": diverged_any})

    cand_results_sorted = sorted(cand_results, key=lambda r: (r["diverged_any"], r["median_best_gap"]))
    best = cand_results_sorted[0]
    return {
        "policy": str(policy),
        "best_hparams": best["hparams"],
        "best_median_best_gap": float(best["median_best_gap"]),
        "num_candidates": int(len(grid)),
    }
