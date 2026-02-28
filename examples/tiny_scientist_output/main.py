"""QuadStep-Ladder experiment workspace.

Step 5/6: Benchmark Runner + Metrics + Phase Maps + Meta-Model

Adds:
- Full benchmark runner over ladder configs + budgets + seeds
- Metrics aggregation (median time-to-eps, stability probability, tail risk)
- Phase-transition map data + stability boundary estimation
- Interpretable meta-model predicting stability from spectral/noise features

Quadratic benchmark remains the scientific core. SST-2 smoke test remains a
lightweight dataset IO check until Step 6.

All results are saved to: <out_dir>/final_info.json
Additional logs are written to: <out_dir>/logs/{events.jsonl,metrics.csv}
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from benchmark import run_full_benchmark
from oracle import NoisyQuadraticOracle, OracleBudget, sigma_sweep
from optimizers import hyperparameter_grids, run_optimizer_under_budget, tune_hyperparameters_under_budget
from quad_generator import make_quadratic_instance


# -------------------------
# Config + IO
# -------------------------


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_config_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in [".json"]:
        return json.loads(text)
    if suffix in [".yml", ".yaml"]:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("YAML config requires `pyyaml` (pip install pyyaml).") from e
        return yaml.safe_load(text)
    raise ValueError(f"Unsupported config file suffix: {suffix}")


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def json_dump(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


# -------------------------
# Logging
# -------------------------


class JsonlLogger:
    def __init__(self, path: Path):
        ensure_dir(path.parent)
        self.path = path

    def log(self, row: Dict[str, Any]) -> None:
        row = dict(row)
        row["ts"] = time.time()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")


class CsvLogger:
    def __init__(self, path: Path, fieldnames: List[str]):
        ensure_dir(path.parent)
        self.path = path
        self.fieldnames = list(fieldnames)
        self._wrote_header = path.exists() and path.stat().st_size > 0

    def log(self, row: Dict[str, Any]) -> None:
        r = {k: row.get(k, "") for k in self.fieldnames}
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._wrote_header:
                w.writeheader()
                self._wrote_header = True
            w.writerow(r)


# -------------------------
# Determinism
# -------------------------


def set_global_determinism(seed: int) -> Dict[str, Any]:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch_info: Dict[str, Any] = {"torch_available": False}
    try:
        import torch

        torch_info["torch_available"] = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
            torch_info["torch_deterministic_algorithms"] = True
        except Exception:
            torch_info["torch_deterministic_algorithms"] = False

        torch_info["torch_version"] = torch.__version__
    except Exception:
        pass

    return {"seed": int(seed), **torch_info}


# -------------------------
# Divergence guards (kept for Step 6 and external use)
# -------------------------


@dataclasses.dataclass
class DivergenceStatus:
    ok: bool
    reason: str
    x_norm: float


def check_divergence(x: np.ndarray, norm_limit: float = 1e12) -> DivergenceStatus:
    x = np.asarray(x)
    if not np.all(np.isfinite(x)):
        return DivergenceStatus(ok=False, reason="non_finite", x_norm=float("inf"))
    n = float(np.linalg.norm(x))
    if n > norm_limit:
        return DivergenceStatus(ok=False, reason="norm_exceeded", x_norm=n)
    return DivergenceStatus(ok=True, reason="ok", x_norm=n)


# -------------------------
# Mode: quadratic_benchmark
# -------------------------


def run_quadratic_benchmark(cfg: Dict[str, Any], out_dir: Path, events: JsonlLogger, metrics: CsvLogger) -> Dict[str, Any]:
    """Two behaviors:

    - If cfg["quadratic"]["benchmark"]["enabled"] == True, run Step-5 full benchmark sweep.
    - Else, run the Step-4 single-instance sanity run (kept for quick checks).
    """

    quad_cfg = cfg["quadratic"]

    bench_cfg = quad_cfg.get("benchmark", {})
    if bool(bench_cfg.get("enabled", True)):
        events.log({"mode": "quadratic_benchmark", "event": "start_full_benchmark"})
        res = run_full_benchmark(cfg=cfg, base_seed=int(cfg["seed"]), events_logger=events)
        events.log({"mode": "quadratic_benchmark", "event": "end_full_benchmark"})

        # Write a few summary rows to metrics.csv (not exhaustive)
        for row in res.get("phase_map_rows", [])[:50]:
            metrics.log(
                {
                    "mode": "quadratic_benchmark",
                    "t": "",
                    "budget": "",
                    "grad_calls": "",
                    "value_calls": "",
                    "f": "",
                    "best_f": "",
                    "x_norm": "",
                    "gap": "",
                    "best_gap": "",
                    "x_err": json.dumps({"stability": row.get("stability_prob"), "sigma": row.get("sigma"), "policy": row.get("policy")}),
                }
            )
        return {"mode": "quadratic_benchmark", "benchmark": res}

    # legacy single-run mode (should not be default for Step 5)
    d = int(quad_cfg["d"])
    kappa = float(quad_cfg["kappa"])
    pattern = str(quad_cfg["pattern"])
    rotation = str(quad_cfg["rotation"])

    oracle_cfg = quad_cfg.get("oracle", {})
    noise_family = str(oracle_cfg.get("noise_family", "isotropic"))
    noise_spectrum = str(oracle_cfg.get("noise_spectrum", "prop_lambda"))
    sigma_mode = str(oracle_cfg.get("sigma_mode", "sweep"))
    sigma_fixed = float(oracle_cfg.get("sigma", 0.0))

    budget_units = float(oracle_cfg.get("budget_units", 200.0))
    alpha_value = float(oracle_cfg.get("alpha_value", 1.0))
    eps_stop = oracle_cfg.get("eps_stop", None)
    eps_stop_f = None if eps_stop is None else float(eps_stop)

    policy_cfg = quad_cfg.get("policy", {})
    policy = str(policy_cfg.get("name", "constant"))
    tune = bool(policy_cfg.get("tune", True))
    validation_budget_units = float(policy_cfg.get("validation_budget_units", min(200.0, budget_units)))
    validation_seeds = policy_cfg.get("validation_seeds", [0, 1, 2])

    dtype = np.float64
    base_rng = np.random.default_rng(int(cfg["seed"]))
    inst = make_quadratic_instance(d=d, kappa=kappa, pattern=pattern, rotation=rotation, rng=base_rng, dtype=dtype)  # type: ignore[arg-type]

    x0 = np.zeros((d,), dtype=dtype)

    sigmas: np.ndarray
    if sigma_mode == "fixed":
        sigmas = np.asarray([sigma_fixed], dtype=np.float64)
    elif sigma_mode == "sweep":
        sigmas = sigma_sweep(int(oracle_cfg.get("sigma_sweep_num", 13)))
    else:
        raise ValueError(f"Unknown sigma_mode: {sigma_mode}")

    grids = hyperparameter_grids()
    if policy not in grids:
        raise ValueError(f"Unknown policy: {policy}. Available: {sorted(grids.keys())}")

    runs = []
    for i, sigma in enumerate(sigmas.tolist()):
        def make_oracle_fn(seed_offset: int) -> NoisyQuadraticOracle:
            budget = OracleBudget(alpha_value=alpha_value)
            rng = np.random.default_rng(int(cfg["seed"]) + 1000 + i * 100 + int(seed_offset))
            oracle = NoisyQuadraticOracle(inst=inst, budget=budget, rng=rng)
            oracle.set_noise(family=noise_family, spectrum=noise_spectrum, sigma=float(sigma))
            return oracle

        tuning_info = None
        if tune:
            tuning_info = tune_hyperparameters_under_budget(
                make_oracle_fn=make_oracle_fn,
                x0=x0,
                policy=policy,  # type: ignore[arg-type]
                grid=grids[policy],
                validation_budget_units=validation_budget_units,
                validation_seeds=[int(s) for s in validation_seeds],
                eps_stop=eps_stop_f,
            )
            chosen_hparams = dict(tuning_info["best_hparams"])
        else:
            chosen_hparams = dict(grids[policy][0])

        oracle = make_oracle_fn(9999)
        mode = "quadratic_benchmark"
        events.log({"mode": mode, "event": "start", "sigma": float(sigma), "policy": policy, **oracle.diagnostics()})

        run_info = run_optimizer_under_budget(
            oracle=oracle,
            x0=x0,
            policy=policy,  # type: ignore[arg-type]
            hparams=chosen_hparams,
            budget_units=budget_units,
            eps_stop=eps_stop_f,
            log_every=int(policy_cfg.get("log_every", 1)),
        )

        metrics.log(
            {
                "mode": mode,
                "t": run_info["iterations"],
                "budget": run_info["budget"]["total_budget"],
                "grad_calls": run_info["budget"]["grad_calls"],
                "value_calls": run_info["budget"]["value_calls"],
                "f": run_info["fT"],
                "best_f": float(inst.f_star + run_info["best_gap"]),
                "x_norm": float(np.linalg.norm(np.asarray(run_info["xT"], dtype=np.float64))),
                "gap": run_info["gapT"],
                "best_gap": run_info["best_gap"],
                "x_err": "",
            }
        )

        run_info.update({"sigma": float(sigma), "tuning": tuning_info})
        runs.append(run_info)
        events.log({"mode": mode, "event": "end", "sigma": float(sigma), **run_info})

    info: Dict[str, Any] = {
        "mode": "quadratic_benchmark",
        "dtype": "float64",
        "d": d,
        "kappa": float(kappa),
        "pattern": pattern,
        "rotation": rotation,
        "policy": {"name": policy, "tune": tune, "validation_budget_units": float(validation_budget_units), "validation_seeds": validation_seeds},
        "oracle": {
            "noise_family": noise_family,
            "noise_spectrum": noise_spectrum,
            "sigma_mode": sigma_mode,
            "sigma_fixed": float(sigma_fixed),
            "sigmas": [float(s) for s in sigmas.tolist()],
            "budget_units": float(budget_units),
            "alpha_value": float(alpha_value),
            "eps_stop": eps_stop_f,
        },
        "generator_diagnostics": inst.diagnostics(),
        "runs": runs,
    }
    return info


# -------------------------
# Mode: sst2_smoke_test (unchanged)
# -------------------------


def load_sst2_subsample(seed: int, n_train: int, n_val: int, n_test: int) -> Dict[str, Any]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: `datasets` (pip install datasets).") from e

    ds = load_dataset("glue", "sst2")

    tr = ds["train"].select(range(min(n_train, len(ds["train"]))))
    va_full = ds["validation"].select(range(min(n_val + n_test, len(ds["validation"]))))

    va = va_full.select(range(min(n_val, len(va_full))))
    te = va_full.select(range(min(n_val, len(va_full)), min(n_val + n_test, len(va_full))))

    def label_counts(split) -> Dict[str, int]:
        y = np.asarray(split["label"], dtype=np.int64)
        uniq, cnt = np.unique(y, return_counts=True)
        out = {"0": 0, "1": 0}
        for u, c in zip(uniq.tolist(), cnt.tolist()):
            out[str(int(u))] = int(c)
        return out

    meta = {
        "dataset": "glue/sst2",
        "seed": int(seed),
        "sizes": {"train": int(len(tr)), "validation": int(len(va)), "test": int(len(te))},
        "label_counts": {"train": label_counts(tr), "validation": label_counts(va), "test": label_counts(te)},
    }

    assert len(tr) <= 5000
    assert len(va) <= 2000
    assert len(te) <= 2000

    return meta


def run_sst2_smoke_test(cfg: Dict[str, Any], out_dir: Path, events: JsonlLogger, metrics: CsvLogger) -> Dict[str, Any]:
    seed = int(cfg["seed"])
    n_train = int(cfg["sst2"]["n_train"])
    n_val = int(cfg["sst2"]["n_val"])
    n_test = int(cfg["sst2"]["n_test"])

    events.log({"mode": "sst2_smoke_test", "event": "start"})
    meta = load_sst2_subsample(seed=seed, n_train=n_train, n_val=n_val, n_test=n_test)

    metrics.log({"mode": "sst2_smoke_test", "t": 0, "budget": "", "grad_calls": "", "value_calls": "", "f": "", "best_f": "", "x_norm": "", "gap": "", "best_gap": "", "x_err": ""})

    info: Dict[str, Any] = {"mode": "sst2_smoke_test", "dataset_meta": meta, "note": "Step 5 does not change SST-2 smoke test. Training is implemented in Step 6."}
    events.log({"mode": "sst2_smoke_test", "event": "end", **info})
    return info


# -------------------------
# CLI
# -------------------------


def build_default_config() -> Dict[str, Any]:
    # Default Step-5 run is a *small* sweep to keep runtime reasonable.
    # The code supports the full N=100 protocol by editing config.
    return {
        "mode": "quadratic_benchmark",
        "seed": 12345,
        "quadratic": {
            "d": 256,
            "kappa": 100.0,
            "pattern": "two_cluster",
            "rotation": "dense",
            "oracle": {
                "noise_family": "aligned",
                "noise_spectrum": "prop_lambda",
                "sigma_mode": "fixed",
                "sigma": 1e-4,
                "sigma_sweep_num": 5,
                "budget_units": 200.0,
                "alpha_value": 1.0,
                "eps_stop": 1e-8,
            },
            "policy": {"name": "constant", "tune": True, "validation_budget_units": 120.0, "validation_seeds": [0, 1, 2], "log_every": 5},
            "benchmark": {
                "enabled": True,
                "kappas": [10.0, 100.0],
                "patterns": ["two_cluster"],
                "rotations": ["identity", "dense"],
                "noise_families": ["aligned", "misaligned"],
                "noise_spectra": ["prop_lambda"],
                "sigmas": [0.0, 1e-4, 1e-2],
                "budgets": [200, 500],
                "num_seeds": 10,
                "methods": ["constant", "exact_linesearch", "bb1"],
                "tune": True,
                "validation_budget_units": 200.0,
                "validation_seeds": [0, 1, 2],
                "log_every": 5,
                "heldout_frac": 0.25,
            },
        },
        "sst2": {"n_train": 5000, "n_val": 2000, "n_test": 2000},
        "logging": {"write_jsonl": True, "write_csv": True},
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QuadStep-Ladder workspace (Step 5/6)")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for logs and final_info.json")
    p.add_argument("--mode", type=str, choices=["quadratic_benchmark", "sst2_smoke_test"], default=None)
    p.add_argument("--config", type=str, default=None, help="Optional JSON/YAML config file")
    p.add_argument("--seed", type=int, default=None)

    return p.parse_args()


def merge_cli_into_config(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    if args.mode is not None:
        out["mode"] = args.mode
    if args.seed is not None:
        out["seed"] = int(args.seed)
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    cfg = build_default_config()
    if args.config is not None:
        file_cfg = read_config_file(Path(args.config))
        cfg = deep_update(cfg, file_cfg)
    cfg = merge_cli_into_config(args, cfg)

    det_info = set_global_determinism(int(cfg["seed"]))

    logs_dir = out_dir / "logs"
    events = JsonlLogger(logs_dir / "events.jsonl")
    metrics = CsvLogger(logs_dir / "metrics.csv", fieldnames=["mode", "t", "budget", "grad_calls", "value_calls", "f", "best_f", "x_norm", "gap", "best_gap", "x_err"])

    events.log({"event": "config", "config": cfg, "determinism": det_info})

    mode = str(cfg["mode"])
    if mode == "quadratic_benchmark":
        run_info = run_quadratic_benchmark(cfg=cfg, out_dir=out_dir, events=events, metrics=metrics)
    elif mode == "sst2_smoke_test":
        run_info = run_sst2_smoke_test(cfg=cfg, out_dir=out_dir, events=events, metrics=metrics)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    final_info = {"step": 5, "config": cfg, "determinism": det_info, "run": run_info, "out_dir": str(out_dir)}
    json_dump(out_dir / "final_info.json", final_info)
    print(json.dumps(final_info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
