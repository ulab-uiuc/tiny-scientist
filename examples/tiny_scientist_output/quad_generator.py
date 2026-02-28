"""Spectral Ladder Quadratic Generator (Step 2/6).

Generates SPD quadratic objectives in R^d:
    f(x) = 0.5 x^T H x + b^T x
with H = Q^T diag(lam) Q.

Knobs:
- condition number kappa in {10, 1e2, 1e3, 1e4}
- eigenvalue pattern: {two_cluster, power_law, flat_spike}
- rotation structure: {identity, block8, dense}

This module is deterministic given an explicit numpy Generator.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Literal, Tuple

import numpy as np


EigenPattern = Literal["two_cluster", "power_law", "flat_spike"]
RotationType = Literal["identity", "block8", "dense"]


@dataclasses.dataclass
class QuadraticInstance:
    d: int
    kappa: float
    pattern: EigenPattern
    rotation: RotationType

    lam: np.ndarray  # shape (d,)
    Q: np.ndarray  # shape (d,d) orthogonal
    H: np.ndarray  # shape (d,d) SPD
    b: np.ndarray  # shape (d,)

    x_star: np.ndarray  # minimizer
    f_star: float

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x)
        return 0.5 * float(x @ (self.H @ x)) + float(self.b @ x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return (self.H @ x) + self.b

    def diagnostics(self) -> Dict[str, float]:
        eigs = np.linalg.eigvalsh(self.H)
        return {
            "mu": float(np.min(eigs)),
            "L": float(np.max(eigs)),
            "kappa_empirical": float(np.max(eigs) / np.min(eigs)),
            "x_star_norm": float(np.linalg.norm(self.x_star)),
            "f_star": float(self.f_star),
        }


def _orthogonal_from_qr(A: np.ndarray) -> np.ndarray:
    # Deterministic QR-based orthogonalization.
    Q, R = np.linalg.qr(A)
    # Fix sign ambiguity by forcing diagonal of R positive.
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s
    return Q


def build_eigenvalues(d: int, kappa: float, pattern: EigenPattern, *, dtype=np.float64) -> np.ndarray:
    """Construct eigenvalues with min=1 and max=kappa."""

    if d <= 0:
        raise ValueError("d must be positive")
    if kappa < 1:
        raise ValueError("kappa must be >= 1")

    mu = 1.0
    L = float(kappa)

    if pattern == "two_cluster":
        # Half small, half large.
        n_small = d // 2
        n_large = d - n_small
        lam = np.concatenate([mu * np.ones(n_small), L * np.ones(n_large)]).astype(dtype)
    elif pattern == "power_law":
        # Smooth monotone spectrum from L down to mu.
        # Use exponent p>0: lam_i = L * (i/(d-1))^p mapped to [mu, L].
        # We want lam[0]=L, lam[-1]=mu.
        p = 2.0
        t = np.linspace(0.0, 1.0, d, dtype=dtype)
        # decreasing curve
        curve = (1.0 - t) ** p
        lam = mu + (L - mu) * curve
    elif pattern == "flat_spike":
        # Mostly flat at mu, with a handful of spikes at L.
        k = max(1, d // 16)  # 16 spikes for d=256
        lam = mu * np.ones(d, dtype=dtype)
        lam[:k] = L
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    # Sort descending (conventional) but not required.
    lam = np.asarray(lam, dtype=dtype)
    lam = np.sort(lam)[::-1]

    # Ensure strict positivity.
    if float(np.min(lam)) <= 0:
        raise RuntimeError("Non-positive eigenvalue produced")

    # Normalize so min=1 exactly, max=kappa exactly (within float error)
    lam = lam / float(np.min(lam))
    lam = lam * (L / float(np.max(lam)))
    lam = np.clip(lam, 1.0, L)

    return lam


def build_rotation(d: int, rotation: RotationType, rng: np.random.Generator, *, block_size: int = 8, dtype=np.float64) -> np.ndarray:
    if rotation == "identity":
        return np.eye(d, dtype=dtype)

    if rotation == "dense":
        A = rng.standard_normal((d, d), dtype=dtype)
        return _orthogonal_from_qr(A).astype(dtype)

    if rotation == "block8":
        if d % block_size != 0:
            raise ValueError(f"d={d} must be divisible by block_size={block_size}")
        Q = np.zeros((d, d), dtype=dtype)
        for i in range(0, d, block_size):
            A = rng.standard_normal((block_size, block_size), dtype=dtype)
            Qi = _orthogonal_from_qr(A).astype(dtype)
            Q[i : i + block_size, i : i + block_size] = Qi
        return Q

    raise ValueError(f"Unknown rotation: {rotation}")


def make_quadratic_instance(
    *,
    d: int = 256,
    kappa: float,
    pattern: EigenPattern,
    rotation: RotationType,
    rng: np.random.Generator,
    dtype=np.float64,
) -> QuadraticInstance:
    lam = build_eigenvalues(d=d, kappa=kappa, pattern=pattern, dtype=dtype)
    Q = build_rotation(d=d, rotation=rotation, rng=rng, dtype=dtype)

    # Construct H = Q^T diag(lam) Q.
    H = (Q.T @ (lam[:, None] * Q)).astype(dtype)

    # Sample x* with ||x*|| = 1 exactly (deterministic given rng).
    x_star = rng.standard_normal((d,), dtype=dtype)
    x_star = x_star / float(np.linalg.norm(x_star))

    # Set b = -H x* so that x* is the minimizer.
    b = -(H @ x_star)

    # Compute f* = f(x*)
    f_star = 0.5 * float(x_star @ (H @ x_star)) + float(b @ x_star)

    inst = QuadraticInstance(
        d=d,
        kappa=float(kappa),
        pattern=pattern,
        rotation=rotation,
        lam=lam,
        Q=Q,
        H=H,
        b=b,
        x_star=x_star,
        f_star=float(f_star),
    )

    # Basic correctness checks.
    g_star = inst.grad(inst.x_star)
    if float(np.linalg.norm(g_star)) > 1e-8:
        raise RuntimeError(f"Generator bug: ||grad(x*)||={float(np.linalg.norm(g_star))} too large")

    return inst
