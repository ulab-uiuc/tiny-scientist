"""QStepBench experiment script (Step 3/5).

Step 3 adds the synthetic quadratic generator with eigenbasis diagnostics hooks:
- Generate strongly convex quadratics f(x)=0.5 x^T H x + b^T x with H=Q diag(λ) Q^T.
- Q is Haar-random orthogonal; λ follows one of four spectrum families:
  (i) log-uniform in [1, κ]
  (ii) power-law λ_i ∝ i^{-p} scaled to [1, κ]
  (iii) k-cluster with controlled gap ratios
  (iv) spiked spectra
- Build a canonical 30-instance suite over (family, κ, d). Allow seed expansion.
- Generate x* with energy concentrated in selected eigenspaces; set b=-H x*.
- Provide oracle: x*, f(x), grad(x) with optional gradient noise injection:
  isotropic N(0, σ^2 I) and top-eigenvector-aligned noise.
- Expose eigenpairs (Q, λ) for mechanistic diagnostics (Steps 5).

This file remains runnable and retains Step 2 (IMDB ridge LS quadratic) functionality.
Steps 4-5 (optimizers, training loop, metrics/aggregation) are NOT implemented yet.

Hard constraints honored:
- Script is self-contained and runnable.
- Accepts --out_dir and writes final_info.json there.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------
# Utilities
# ---------------------------


def stable_hash(data: Any, n_chars: int = 12) -> str:
    s = json.dumps(data, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return h[:n_chars]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def set_global_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def json_dump(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


# ---------------------------
# Step 3: Synthetic Quadratic Generator
# ---------------------------


def haar_orthogonal(d: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample a Haar-random orthogonal matrix via QR of a standard normal matrix."""
    a = rng.normal(loc=0.0, scale=1.0, size=(d, d)).astype(np.float64)
    q, r = np.linalg.qr(a)
    # Make Q uniform by enforcing positive diagonal of R
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    q = q * signs  # broadcast columnwise
    return q.astype(np.float32)


def spectrum_log_uniform(d: int, kappa: float, rng: np.random.RandomState) -> np.ndarray:
    if kappa < 1.0:
        raise ValueError("kappa must be >= 1")
    u = rng.uniform(low=0.0, high=1.0, size=d)
    lam = np.exp(np.log(1.0) + u * (np.log(kappa) - np.log(1.0)))
    lam = np.sort(lam)[::-1]  # descending: λ_1 = λ_max
    return lam.astype(np.float32)


def spectrum_power_law(d: int, kappa: float, p: float) -> np.ndarray:
    if kappa < 1.0:
        raise ValueError("kappa must be >= 1")
    if p <= 0:
        raise ValueError("p must be > 0")
    i = np.arange(1, d + 1, dtype=np.float64)
    base = i ** (-p)
    # Scale to [1, kappa] with λ_1 = kappa, λ_d = 1
    lam = 1.0 + (kappa - 1.0) * (base - base[-1]) / (base[0] - base[-1])
    lam = np.sort(lam)[::-1]
    return lam.astype(np.float32)


def _alloc_cluster_sizes(d: int, n_clusters: int) -> List[int]:
    base = d // n_clusters
    rem = d - base * n_clusters
    sizes = [base] * n_clusters
    for i in range(rem):
        sizes[i] += 1
    return sizes


def spectrum_k_cluster(
    d: int,
    kappa: float,
    n_clusters: int,
    gap_ratio: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Clustered eigenvalues.

    We construct n_clusters clusters. Each cluster has a constant eigenvalue.
    Cluster means decay geometrically by gap_ratio, then we rescale so that
    λ_max=kappa and λ_min=1 exactly.
    """
    if n_clusters < 2:
        raise ValueError("n_clusters must be >= 2")
    if gap_ratio <= 1.0:
        raise ValueError("gap_ratio must be > 1")
    if kappa < 1.0:
        raise ValueError("kappa must be >= 1")

    sizes = _alloc_cluster_sizes(d, n_clusters)
    # provisional means: m_0=1, m_{j}=m_{j-1}/gap_ratio
    means = np.array([gap_ratio ** (-j) for j in range(n_clusters)], dtype=np.float64)
    # rescale means to achieve exact condition number
    means = 1.0 + (kappa - 1.0) * (means - means[-1]) / (means[0] - means[-1])
    means = means[::-1]  # ascending? we'll sort later anyway

    lam_list: List[float] = []
    for m, sz in zip(means, sizes):
        lam_list.extend([float(m)] * sz)
    lam = np.array(lam_list, dtype=np.float64)
    lam = np.sort(lam)[::-1]
    meta = {
        "n_clusters": int(n_clusters),
        "gap_ratio": float(gap_ratio),
        "cluster_sizes": [int(x) for x in sizes],
        "cluster_means_desc": [float(x) for x in np.sort(means)[::-1]],
    }
    return lam.astype(np.float32), meta


def spectrum_spiked(
    d: int,
    kappa: float,
    spike_rank: int,
    spike_ratio: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Spiked spectrum: a low-rank set of large eigenvalues on top of a bulk.

    Construction:
    - bulk eigenvalues are linearly spaced in [1, bulk_max]
    - top spike_rank eigenvalues are set to kappa
    - bulk_max is chosen so that bulk_max = kappa/spike_ratio (clipped to >=1)

    Ensures λ_max=kappa and λ_min=1.
    """
    if kappa < 1.0:
        raise ValueError("kappa must be >= 1")
    if not (1 <= spike_rank < d):
        raise ValueError("spike_rank must satisfy 1 <= spike_rank < d")
    if spike_ratio <= 1.0:
        raise ValueError("spike_ratio must be > 1")

    bulk_max = max(1.0, kappa / spike_ratio)
    bulk_size = d - spike_rank
    if bulk_size == 1:
        bulk = np.array([1.0], dtype=np.float64)
    else:
        bulk = np.linspace(1.0, bulk_max, num=bulk_size, dtype=np.float64)
    spikes = np.full((spike_rank,), kappa, dtype=np.float64)
    lam = np.concatenate([spikes, bulk], axis=0)
    lam = np.sort(lam)[::-1]
    meta = {"spike_rank": int(spike_rank), "spike_ratio": float(spike_ratio), "bulk_max": float(bulk_max)}
    return lam.astype(np.float32), meta


def sample_xstar_in_eigenbasis(
    lam: np.ndarray,
    rng: np.random.RandomState,
    concentrate: str,
    k: int,
    cluster_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Sample x* coefficients in eigenbasis with energy concentrated.

    Returns z* (coefficients in eigenbasis) and a dict describing the concentration.
    """
    d = lam.shape[0]
    k = int(max(1, min(k, d)))
    if concentrate == "topk":
        idx = np.arange(0, k)
    elif concentrate == "bottomk":
        idx = np.arange(d - k, d)
    elif concentrate == "randomk":
        idx = rng.choice(np.arange(d), size=k, replace=False)
        idx = np.sort(idx)
    elif concentrate == "cluster" and cluster_meta is not None:
        # choose the largest-eigenvalue cluster by construction: it occupies the first cluster_sizes[0] indices
        sizes = cluster_meta.get("cluster_sizes", None)
        if sizes is None:
            raise ValueError("cluster_meta missing cluster_sizes")
        top_cluster_size = int(sizes[0])
        kk = min(k, top_cluster_size)
        idx = np.arange(0, kk)
    else:
        raise ValueError(f"Unknown concentrate mode: {concentrate}")

    z = np.zeros((d,), dtype=np.float32)
    # Put nonzero mass in selected modes using deterministic RNG (not hardcoded values)
    z_sel = rng.normal(loc=0.0, scale=1.0, size=idx.shape[0]).astype(np.float32)
    # normalize so ||z||_2 = 1 for comparability
    norm = float(np.linalg.norm(z_sel))
    if norm == 0.0:
        # Extremely unlikely; resample deterministically by advancing RNG
        z_sel = rng.normal(loc=0.0, scale=1.0, size=idx.shape[0]).astype(np.float32)
        norm = float(np.linalg.norm(z_sel))
    z_sel /= (norm + 1e-12)
    z[idx] = z_sel
    meta = {"mode": concentrate, "k": int(k), "active_indices": idx.tolist()}
    return z, meta


@dataclasses.dataclass
class NoiseConfig:
    sigma_iso: float = 0.0
    sigma_top: float = 0.0
    top_m: int = 1


class SyntheticQuadraticOracle:
    """Oracle for synthetic strongly convex quadratic with diagnostics hooks."""

    def __init__(
        self,
        Q: np.ndarray,
        lam: np.ndarray,
        b: np.ndarray,
        x_star: np.ndarray,
        noise: NoiseConfig,
        rng: np.random.RandomState,
    ):
        self.Q = np.asarray(Q, dtype=np.float32)
        self.lam = np.asarray(lam, dtype=np.float32)
        self.b = np.asarray(b, dtype=np.float32)
        self.x_star = np.asarray(x_star, dtype=np.float32)
        self.noise = noise
        self.rng = rng

        d = self.lam.shape[0]
        if self.Q.shape != (d, d):
            raise ValueError("Q shape mismatch")
        if self.b.shape != (d,):
            raise ValueError("b shape mismatch")
        if self.x_star.shape != (d,):
            raise ValueError("x_star shape mismatch")

    @property
    def dim(self) -> int:
        return int(self.lam.shape[0])

    @property
    def lambda_max(self) -> float:
        return float(self.lam[0])

    @property
    def lambda_min(self) -> float:
        return float(self.lam[-1])

    @property
    def kappa(self) -> float:
        return float(self.lambda_max / self.lambda_min)

    def matvec_H(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        z = self.Q.T @ x
        z = (self.lam * z).astype(np.float32)
        return (self.Q @ z).astype(np.float32)

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float32)
        Hx = self.matvec_H(x)
        return 0.5 * float(np.dot(x, Hx)) + float(np.dot(self.b, x))

    def grad(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        g = self.matvec_H(x) + self.b

        if self.noise.sigma_iso > 0.0:
            g = g + self.rng.normal(0.0, self.noise.sigma_iso, size=g.shape).astype(np.float32)

        if self.noise.sigma_top > 0.0:
            m = int(max(1, min(self.noise.top_m, self.dim)))
            # sample in eigenbasis supported on top m directions, then map back
            coeff = np.zeros((self.dim,), dtype=np.float32)
            coeff[:m] = self.rng.normal(0.0, self.noise.sigma_top, size=(m,)).astype(np.float32)
            g = g + (self.Q @ coeff).astype(np.float32)

        return g.astype(np.float32)

    def f_star(self) -> float:
        # By construction b = -H x*, so minimum value is -0.5 x*^T H x*
        Hx = self.matvec_H(self.x_star)
        return 0.5 * float(np.dot(self.x_star, Hx)) + float(np.dot(self.b, self.x_star))


@dataclasses.dataclass
class SyntheticInstanceSpec:
    family: str
    d: int
    kappa: float
    seed: int
    # family knobs
    power_p: float = 1.0
    n_clusters: int = 3
    gap_ratio: float = 10.0
    spike_rank: int = 1
    spike_ratio: float = 10.0
    # x* concentration
    concentrate: str = "topk"
    concentrate_k: int = 8
    # noise
    sigma_iso: float = 0.0
    sigma_top: float = 0.0
    top_m: int = 1


def build_synthetic_instance(spec: SyntheticInstanceSpec) -> Tuple[SyntheticQuadraticOracle, Dict[str, Any]]:
    rng = np.random.RandomState(spec.seed)
    Q = haar_orthogonal(spec.d, rng=rng)

    fam_meta: Dict[str, Any] = {}
    if spec.family == "log_uniform":
        lam = spectrum_log_uniform(spec.d, spec.kappa, rng=rng)
    elif spec.family == "power_law":
        lam = spectrum_power_law(spec.d, spec.kappa, p=spec.power_p)
        fam_meta["p"] = float(spec.power_p)
    elif spec.family == "k_cluster":
        lam, meta = spectrum_k_cluster(
            spec.d, spec.kappa, n_clusters=spec.n_clusters, gap_ratio=spec.gap_ratio
        )
        fam_meta.update(meta)
    elif spec.family == "spiked":
        lam, meta = spectrum_spiked(
            spec.d, spec.kappa, spike_rank=spec.spike_rank, spike_ratio=spec.spike_ratio
        )
        fam_meta.update(meta)
    else:
        raise ValueError(f"Unknown family: {spec.family}")

    # Choose z* in eigenbasis and map x*=Q z*
    z_star, z_meta = sample_xstar_in_eigenbasis(
        lam=lam,
        rng=rng,
        concentrate=spec.concentrate,
        k=spec.concentrate_k,
        cluster_meta=fam_meta if spec.family == "k_cluster" else None,
    )
    x_star = (Q @ z_star).astype(np.float32)

    # Set b = -H x* = -Q diag(lam) Q^T x* = -Q (lam * z*)
    b = -(Q @ (lam * z_star).astype(np.float32)).astype(np.float32)

    noise = NoiseConfig(sigma_iso=float(spec.sigma_iso), sigma_top=float(spec.sigma_top), top_m=int(spec.top_m))
    oracle = SyntheticQuadraticOracle(Q=Q, lam=lam, b=b, x_star=x_star, noise=noise, rng=rng)

    inst_meta: Dict[str, Any] = {
        "family": spec.family,
        "d": int(spec.d),
        "kappa_target": float(spec.kappa),
        "kappa_actual": float(oracle.kappa),
        "lambda_max": float(oracle.lambda_max),
        "lambda_min": float(oracle.lambda_min),
        "seed": int(spec.seed),
        "family_meta": fam_meta,
        "xstar_concentration": z_meta,
        "noise": dataclasses.asdict(noise),
    }
    return oracle, inst_meta


def canonical_30_instance_suite(base_seed: int = 0) -> List[SyntheticInstanceSpec]:
    """Canonical 30 instances over (family, kappa, d).

    We use:
    - families: 4
    - kappas: {10, 100, 1000}
    - d: {256, 1024}
    Total combos: 4*3*2 = 24. Add 6 extra instances by varying x* concentration
    for selected (family, kappa, d) tuples.
    """
    families = ["log_uniform", "power_law", "k_cluster", "spiked"]
    kappas = [10.0, 100.0, 1000.0]
    ds = [256, 1024]

    specs: List[SyntheticInstanceSpec] = []
    idx = 0
    for fam in families:
        for kappa in kappas:
            for d in ds:
                seed = int(base_seed + 1000 + idx)
                spec = SyntheticInstanceSpec(
                    family=fam,
                    d=d,
                    kappa=kappa,
                    seed=seed,
                    power_p=1.0,
                    n_clusters=3,
                    gap_ratio=10.0,
                    spike_rank=2 if d >= 256 else 1,
                    spike_ratio=20.0,
                    concentrate="topk",
                    concentrate_k=8,
                )
                specs.append(spec)
                idx += 1

    # Add 6 additional challenging directions variants
    extras = [
        ("log_uniform", 1000.0, 1024, "bottomk"),
        ("power_law", 1000.0, 1024, "bottomk"),
        ("k_cluster", 1000.0, 1024, "cluster"),
        ("spiked", 1000.0, 1024, "randomk"),
        ("k_cluster", 100.0, 256, "cluster"),
        ("spiked", 100.0, 256, "randomk"),
    ]
    for fam, kappa, d, conc in extras:
        seed = int(base_seed + 5000 + idx)
        spec = SyntheticInstanceSpec(
            family=fam,
            d=d,
            kappa=kappa,
            seed=seed,
            power_p=1.2,
            n_clusters=4,
            gap_ratio=20.0,
            spike_rank=4 if d >= 1024 else 2,
            spike_ratio=50.0,
            concentrate=conc,
            concentrate_k=16,
        )
        specs.append(spec)
        idx += 1

    assert len(specs) == 30
    return specs


def expand_suite_seeds(suite: List[SyntheticInstanceSpec], n_seeds: int, seed_offset: int = 0) -> List[SyntheticInstanceSpec]:
    """Expand each canonical instance with n_seeds different seeds (deterministically)."""
    expanded: List[SyntheticInstanceSpec] = []
    for j in range(n_seeds):
        for s in suite:
            s2 = dataclasses.replace(s, seed=int(s.seed + seed_offset + 100000 * j))
            expanded.append(s2)
    return expanded


def run_synthetic_step(out_dir: Path, base_seed: int, noise_sigma: float, noise_top_sigma: float) -> Dict[str, Any]:
    """Step 3 runnable check: build suite, instantiate one oracle, and evaluate."""
    set_global_seeds(base_seed)

    suite = canonical_30_instance_suite(base_seed=base_seed)
    # Use first instance for a quick sanity run.
    spec0 = dataclasses.replace(
        suite[0],
        sigma_iso=float(noise_sigma),
        sigma_top=float(noise_top_sigma),
        top_m=4,
    )
    oracle, meta = build_synthetic_instance(spec0)

    x0 = np.zeros((oracle.dim,), dtype=np.float32)
    f0 = oracle.value(x0)
    g0 = oracle.grad(x0)
    f_star = oracle.f_star()
    subopt0 = float(f0 - f_star)

    # Verify oracle consistency: grad at x* should be noise-free ~0; check with noise disabled.
    spec_clean = dataclasses.replace(spec0, sigma_iso=0.0, sigma_top=0.0)
    oracle_clean, _ = build_synthetic_instance(spec_clean)
    g_star = oracle_clean.grad(oracle_clean.x_star)
    g_star_norm = float(np.linalg.norm(g_star))

    info: Dict[str, Any] = {
        "step": 3,
        "suite": "synthetic",
        "what": "synthetic_quadratic_generator",
        "base_seed": int(base_seed),
        "suite_size": int(len(suite)),
        "first_instance_spec": dataclasses.asdict(spec0),
        "first_instance_meta": meta,
        "sanity": {
            "f(x0)": float(f0),
            "f_star": float(f_star),
            "suboptimality_x0": float(subopt0),
            "grad_norm_x0": float(np.linalg.norm(g0)),
            "grad_norm_at_xstar_noise_free": float(g_star_norm),
        },
        "notes": {
            "eigenpairs_exposed": True,
            "oracle_supports_noise": True,
            "diagnostics_hooks": ["Q", "lam", "x_star"],
        },
    }

    ensure_dir(out_dir / "meta")
    json_dump(out_dir / "meta" / "synthetic_suite_meta.json", {
        "base_seed": int(base_seed),
        "canonical_suite": [dataclasses.asdict(s) for s in suite],
    })
    json_dump(out_dir / "meta" / "synthetic_first_instance.json", info)

    return info


# ---------------------------
# Step 2: IMDB TF-IDF dataset builder
# ---------------------------


def stratified_subsample_indices(labels: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """Deterministic stratified subsample of indices."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be 1D")

    classes, counts = np.unique(labels, return_counts=True)
    if classes.size < 2:
        raise ValueError("Need at least 2 classes for stratification")

    if n_samples > labels.size:
        raise ValueError(f"n_samples={n_samples} exceeds dataset size={labels.size}")

    props = counts / counts.sum()
    raw = props * n_samples
    base = np.floor(raw).astype(int)
    remainder = n_samples - int(base.sum())
    frac_order = np.argsort(-(raw - base))
    for k in range(remainder):
        base[frac_order[k % len(base)]] += 1

    rs = np.random.RandomState(seed)
    selected: List[int] = []
    for cls, k in zip(classes, base):
        cls_idx = np.flatnonzero(labels == cls)
        chosen = rs.choice(cls_idx, size=int(k), replace=False)
        selected.extend(chosen.tolist())

    selected_arr = np.array(selected, dtype=np.int64)
    rs.shuffle(selected_arr)
    return selected_arr


@dataclasses.dataclass
class ImdbTfidfSplits:
    X_train: "scipy.sparse.csr_matrix"
    X_val: "scipy.sparse.csr_matrix"
    X_test: "scipy.sparse.csr_matrix"
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    yls_train: np.ndarray
    yls_val: np.ndarray
    yls_test: np.ndarray
    vectorizer_vocab_size: int


def load_imdb_tfidf_splits(
    seed: int,
    n_train: int = 5000,
    n_val: int = 2000,
    n_test: int = 2000,
    max_features: int = 20000,
) -> Tuple[ImdbTfidfSplits, Dict[str, Any]]:
    """Load IMDB and build TF-IDF features with specified subsampling."""
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: `datasets`. Install with `pip install datasets`."
        ) from e

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: `scikit-learn`. Install with `pip install scikit-learn`."
        ) from e

    import scipy.sparse as sp  # type: ignore

    ds = load_dataset("imdb")
    train = ds["train"]
    test = ds["test"]

    y_train_full = np.asarray(train["label"], dtype=np.int64)
    y_test_full = np.asarray(test["label"], dtype=np.int64)

    tr_idx = stratified_subsample_indices(y_train_full, n_train, seed=seed)
    test_4k_idx = stratified_subsample_indices(y_test_full, n_val + n_test, seed=seed)

    rs = np.random.RandomState(seed)
    perm = rs.permutation(test_4k_idx.shape[0])
    val_idx = test_4k_idx[perm[:n_val]]
    te_idx = test_4k_idx[perm[n_val : n_val + n_test]]

    tr_text = [train[int(i)]["text"] for i in tr_idx]
    va_text = [test[int(i)]["text"] for i in val_idx]
    te_text = [test[int(i)]["text"] for i in te_idx]

    y_tr = y_train_full[tr_idx]
    y_va = y_test_full[val_idx]
    y_te = y_test_full[te_idx]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        norm="l2",
        dtype=np.float32,
    )
    X_tr = vectorizer.fit_transform(tr_text)
    X_va = vectorizer.transform(va_text)
    X_te = vectorizer.transform(te_text)

    X_tr = sp.csr_matrix(X_tr, dtype=np.float32)
    X_va = sp.csr_matrix(X_va, dtype=np.float32)
    X_te = sp.csr_matrix(X_te, dtype=np.float32)

    yls_tr = (2.0 * y_tr.astype(np.float32) - 1.0).astype(np.float32)
    yls_va = (2.0 * y_va.astype(np.float32) - 1.0).astype(np.float32)
    yls_te = (2.0 * y_te.astype(np.float32) - 1.0).astype(np.float32)

    meta = {
        "dataset": "imdb",
        "subsample_seed": int(seed),
        "sizes": {
            "train": int(X_tr.shape[0]),
            "val": int(X_va.shape[0]),
            "test": int(X_te.shape[0]),
        },
        "tfidf": {
            "max_features": int(max_features),
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.9,
            "norm": "l2",
        },
        "n_features": int(X_tr.shape[1]),
        "label_distribution": {
            "train": {"neg": int((y_tr == 0).sum()), "pos": int((y_tr == 1).sum())},
            "val": {"neg": int((y_va == 0).sum()), "pos": int((y_va == 1).sum())},
            "test": {"neg": int((y_te == 0).sum()), "pos": int((y_te == 1).sum())},
        },
    }

    splits = ImdbTfidfSplits(
        X_train=X_tr,
        X_val=X_va,
        X_test=X_te,
        y_train=y_tr,
        y_val=y_va,
        y_test=y_te,
        yls_train=yls_tr,
        yls_val=yls_va,
        yls_test=yls_te,
        vectorizer_vocab_size=len(vectorizer.vocabulary_),
    )
    return splits, meta


class RidgeLeastSquaresOracle:
    """Full-batch ridge least squares oracle with bias term."""

    def __init__(self, X: "scipy.sparse.csr_matrix", y: np.ndarray, lam: float):
        import scipy.sparse as sp  # type: ignore

        if not sp.isspmatrix_csr(X):
            X = sp.csr_matrix(X)
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be 1D")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y have inconsistent n")

        self.X = X
        self.y = y.astype(np.float32)
        self.lam = float(lam)
        self.n, self.d = X.shape

    @property
    def dim(self) -> int:
        return self.d + 1

    def value(self, theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=np.float32)
        if theta.shape != (self.dim,):
            raise ValueError(f"theta must have shape ({self.dim},)")
        w = theta[: self.d]
        b = float(theta[self.d])

        r = self.X @ w
        r = np.asarray(r).reshape(-1).astype(np.float32)
        r = (r + b) - self.y

        data_fid = 0.5 * float(np.dot(r, r))
        reg = 0.5 * self.lam * float(np.dot(w, w))
        return data_fid + reg

    def grad(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float32)
        if theta.shape != (self.dim,):
            raise ValueError(f"theta must have shape ({self.dim},)")
        w = theta[: self.d]
        b = float(theta[self.d])

        r = self.X @ w
        r = np.asarray(r).reshape(-1).astype(np.float32)
        r = (r + b) - self.y

        grad_w = self.X.T @ r
        grad_w = np.asarray(grad_w).reshape(-1).astype(np.float32)
        grad_w = grad_w + (self.lam * w)

        grad_b = np.array([float(r.sum())], dtype=np.float32)
        return np.concatenate([grad_w, grad_b], axis=0)


def run_imdb_quadratic_step(out_dir: Path, seed: int, lam: float) -> Dict[str, Any]:
    set_global_seeds(seed)

    splits, ds_meta = load_imdb_tfidf_splits(seed=seed)
    oracle = RidgeLeastSquaresOracle(splits.X_train, splits.yls_train, lam=lam)

    theta0 = np.zeros((oracle.dim,), dtype=np.float32)
    f0 = oracle.value(theta0)
    g0 = oracle.grad(theta0)
    g0_norm = float(np.linalg.norm(g0))

    oracle_val = RidgeLeastSquaresOracle(splits.X_val, splits.yls_val, lam=lam)
    oracle_test = RidgeLeastSquaresOracle(splits.X_test, splits.yls_test, lam=lam)
    f0_val = oracle_val.value(theta0)
    f0_test = oracle_test.value(theta0)

    assert splits.X_train.shape[0] == 5000
    assert splits.X_val.shape[0] == 2000
    assert splits.X_test.shape[0] == 2000
    assert splits.X_train.shape[1] <= 20000

    info: Dict[str, Any] = {
        "step": 2,
        "suite": "imdb_quadratic",
        "what": "dataset_induced_quadratic_oracle",
        "seed": int(seed),
        "lambda": float(lam),
        "dataset_meta": ds_meta,
        "oracle": {
            "n": int(oracle.n),
            "d": int(oracle.d),
            "dim_with_bias": int(oracle.dim),
            "objective": "0.5*||Xw + b - y||^2 + 0.5*lambda*||w||^2",
            "y_transform": "y_ls = 2*y - 1",
        },
        "initial_point": {
            "theta0": "zeros",
            "f_train": float(f0),
            "grad_norm_train": g0_norm,
            "f_val": float(f0_val),
            "f_test": float(f0_test),
        },
    }

    ensure_dir(out_dir / "meta")
    json_dump(out_dir / "meta" / "imdb_quadratic_meta.json", info)
    return info


# ---------------------------
# CLI
# ---------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QStepBench (Step 3/5): synthetic generator + IMDB oracle")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory")

    p.add_argument(
        "--suite",
        type=str,
        default="synthetic",
        choices=["synthetic", "imdb_quadratic"],
        help="Which suite to run.",
    )
    p.add_argument("--seed", type=int, default=12345, help="Base seed")

    # IMDB quadratic args
    p.add_argument("--ridge_lambda", type=float, default=1e-2)

    # Synthetic noise args (Step 3)
    p.add_argument("--noise_sigma", type=float, default=0.0, help="Isotropic grad noise sigma")
    p.add_argument("--noise_top_sigma", type=float, default=0.0, help="Top-eigenspace grad noise sigma")

    # Kept for compatibility; not used until Step 4
    p.add_argument("--max_runs", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if args.suite == "imdb_quadratic":
        info = run_imdb_quadratic_step(out_dir=out_dir, seed=args.seed, lam=args.ridge_lambda)
        info["out_dir"] = str(out_dir)
        info["config_id"] = stable_hash(
            {"suite": args.suite, "seed": args.seed, "lambda": args.ridge_lambda}
        )
        json_dump(out_dir / "final_info.json", info)
        print(json.dumps(info, indent=2, sort_keys=True))
        return

    # synthetic
    info = run_synthetic_step(
        out_dir=out_dir,
        base_seed=args.seed,
        noise_sigma=args.noise_sigma,
        noise_top_sigma=args.noise_top_sigma,
    )
    info["out_dir"] = str(out_dir)
    info["config_id"] = stable_hash(
        {
            "suite": args.suite,
            "seed": args.seed,
            "noise_sigma": args.noise_sigma,
            "noise_top_sigma": args.noise_top_sigma,
        }
    )
    json_dump(out_dir / "final_info.json", info)
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
