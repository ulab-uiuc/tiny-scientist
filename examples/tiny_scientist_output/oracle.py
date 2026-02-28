"""Noise/Oracle model + budget accounting (Step 3/6).

Implements:
- Stochastic gradient oracle: g(x) = grad(x) + xi, xi ~ N(0, Sigma)
- Sigma families:
    * isotropic: Sigma = (sigma^2) I
    * aligned:   Sigma = Q^T diag(s) Q
    * misaligned:Sigma = Q^T P^T diag(s) P Q, with random orthogonal P
- Spectra s choices:
    * proportional to lambda (H eigenvalues)
    * proportional to 1/lambda
    * independent two-cluster
- Noise scale sweep: sigma in logspace(1e-6, 1e0)
- OracleBudget counter: +1 per gradient call, +alpha per value call.

All code is deterministic given explicit numpy Generator seeds.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Literal, Optional

import numpy as np

from quad_generator import QuadraticInstance


NoiseFamily = Literal["isotropic", "aligned", "misaligned"]
NoiseSpectrum = Literal["prop_lambda", "prop_inv_lambda", "two_cluster"]


@dataclasses.dataclass
class OracleBudget:
    """Tracks oracle usage in budget units."""

    alpha_value: float = 1.0
    grad_calls: int = 0
    value_calls: int = 0

    def reset(self) -> None:
        self.grad_calls = 0
        self.value_calls = 0

    def charge_grad(self, n: int = 1) -> None:
        self.grad_calls += int(n)

    def charge_value(self, n: int = 1) -> None:
        self.value_calls += int(n)

    @property
    def total(self) -> float:
        return float(self.grad_calls) + float(self.alpha_value) * float(self.value_calls)

    def state_dict(self) -> Dict[str, float]:
        return {
            "alpha_value": float(self.alpha_value),
            "grad_calls": float(self.grad_calls),
            "value_calls": float(self.value_calls),
            "total_budget": float(self.total),
        }


def _orthogonal_from_qr(A: np.ndarray) -> np.ndarray:
    Q, R = np.linalg.qr(A)
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s
    return Q


def build_noise_spectrum(
    lam: np.ndarray,
    spectrum: NoiseSpectrum,
    rng: np.random.Generator,
    *,
    dtype=np.float64,
) -> np.ndarray:
    """Return base per-eigendirection variances s (shape d,) before global scaling.

    Note: The overall noise scale is applied separately via sigma.
    """

    lam = np.asarray(lam, dtype=dtype)
    d = lam.shape[0]

    if spectrum == "prop_lambda":
        s = lam.copy()
    elif spectrum == "prop_inv_lambda":
        s = 1.0 / lam
    elif spectrum == "two_cluster":
        # Independent of lambda: half small, half large with a fixed ratio.
        ratio = 100.0
        n_small = d // 2
        n_large = d - n_small
        s = np.concatenate([
            np.ones(n_small, dtype=dtype),
            ratio * np.ones(n_large, dtype=dtype),
        ])
        # Shuffle deterministically using rng to avoid coupling to coordinate order.
        perm = rng.permutation(d)
        s = s[perm]
    else:
        raise ValueError(f"Unknown spectrum: {spectrum}")

    # Normalize to mean 1 so sigma meaning is comparable.
    s = s / float(np.mean(s))
    s = np.maximum(s, np.finfo(dtype).tiny)
    return s.astype(dtype)


def build_noise_cov_factor(
    inst: QuadraticInstance,
    family: NoiseFamily,
    spectrum: NoiseSpectrum,
    sigma: float,
    rng: np.random.Generator,
    *,
    dtype=np.float64,
) -> np.ndarray:
    """Return matrix A such that xi = A z, z~N(0, I).

    Then Cov[xi] = A A^T = Sigma.

    For aligned/misaligned we build Sigma in the *H eigenbasis*.
    """

    d = inst.d
    sigma = float(sigma)
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")

    if family == "isotropic":
        return (sigma * np.eye(d, dtype=dtype))

    s = build_noise_spectrum(inst.lam, spectrum=spectrum, rng=rng, dtype=dtype)
    # Ensure overall scale: Sigma = sigma^2 * (Q^T diag(s) Q) with mean diag = sigma^2.
    # So factor in eigenbasis is sigma * diag(sqrt(s)).
    Dhalf = (sigma * np.sqrt(s)).astype(dtype)

    if family == "aligned":
        # A = Q^T diag(Dhalf)
        return (inst.Q.T @ (Dhalf[:, None] * np.eye(d, dtype=dtype))).astype(dtype)

    if family == "misaligned":
        # Build additional random orthogonal P, then Sigma = Q^T P^T diag(s) P Q.
        A = rng.standard_normal((d, d), dtype=dtype)
        P = _orthogonal_from_qr(A).astype(dtype)
        # Factor: Q^T P^T diag(Dhalf)
        return (inst.Q.T @ (P.T @ (Dhalf[:, None] * np.eye(d, dtype=dtype)))).astype(dtype)

    raise ValueError(f"Unknown family: {family}")


@dataclasses.dataclass
class NoisyQuadraticOracle:
    """Wraps a QuadraticInstance with noisy gradient and budget accounting."""

    inst: QuadraticInstance
    budget: OracleBudget
    rng: np.random.Generator

    noise_family: NoiseFamily = "isotropic"
    noise_spectrum: NoiseSpectrum = "prop_lambda"
    sigma: float = 0.0

    _A: Optional[np.ndarray] = None  # factor for sampling xi

    def __post_init__(self) -> None:
        self._refresh_factor()

    def _refresh_factor(self) -> None:
        if float(self.sigma) == 0.0:
            self._A = None
        else:
            self._A = build_noise_cov_factor(
                self.inst,
                family=self.noise_family,
                spectrum=self.noise_spectrum,
                sigma=float(self.sigma),
                rng=self.rng,
                dtype=self.inst.H.dtype,
            )

    def set_noise(self, *, family: NoiseFamily, spectrum: NoiseSpectrum, sigma: float) -> None:
        self.noise_family = family
        self.noise_spectrum = spectrum
        self.sigma = float(sigma)
        self._refresh_factor()

    def value(self, x: np.ndarray) -> float:
        self.budget.charge_value(1)
        return self.inst.value(x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        self.budget.charge_grad(1)
        g = self.inst.grad(x)
        if self._A is None:
            return g
        z = self.rng.standard_normal((self.inst.d,), dtype=self.inst.H.dtype)
        xi = self._A @ z
        return g + xi

    def diagnostics(self) -> Dict[str, float]:
        base = self.inst.diagnostics()
        base.update(
            {
                "noise_family": str(self.noise_family),
                "noise_spectrum": str(self.noise_spectrum),
                "sigma": float(self.sigma),
                "oracle_alpha_value": float(self.budget.alpha_value),
            }
        )
        return base


def sigma_sweep(num: int = 13) -> np.ndarray:
    return np.logspace(-6, 0, int(num), dtype=np.float64)
