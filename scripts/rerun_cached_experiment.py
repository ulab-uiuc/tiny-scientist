#!/usr/bin/env python3
"""
Quick-and-dirty helper to re-run cached experiments after manually editing
the generated experiment code. It deliberately skips regenerating
experiment.py/run_*.py and only executes the existing run scripts, then
refreshes the aggregated artefacts.

Typical usage:
    python scripts/rerun_cached_experiment.py \
        --experiment-dir generated/experiments
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run cached experiment runs without regenerating code."
    )
    parser.add_argument(
        "--experiment-dir",
        default="generated/experiments",
        help=(
            "Directory containing experiment.py, run_*.py, etc. "
            "If it only contains idea-* subfolders, all of them are processed."
        ),
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Optional explicit list of run script stems (e.g. run_1 run_2). "
        "Defaults to discovering all run_*.py files.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable to use when invoking the run scripts (default: current interpreter).",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete pre-existing run_* output folders before re-running.",
    )
    return parser.parse_args()


def discover_runs(exp_dir: Path) -> Iterable[Tuple[str, Path]]:
    """Yield (run_name, script_path) pairs sorted by numeric suffix."""
    candidates = sorted(exp_dir.glob("run_*.py"))

    def sort_key(path: Path) -> Tuple[int, str]:
        stem = path.stem
        try:
            suffix = int(stem.split("_", 1)[1])
        except (IndexError, ValueError):
            suffix = sys.maxsize
        return suffix, stem

    for script_path in sorted(candidates, key=sort_key):
        yield script_path.stem, script_path


def run_single(
    run_name: str,
    script_path: Path,
    exp_dir: Path,
    python_bin: str,
    keep_existing: bool,
) -> None:
    run_dir = exp_dir / run_name
    if run_dir.exists() and not keep_existing:
        shutil.rmtree(run_dir)

    cmd = [python_bin, script_path.name, f"--out_dir={run_name}"]
    proc = subprocess.run(
        cmd,
        cwd=exp_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{script_path.name} failed (exit {proc.returncode}).\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )

    final_info = run_dir / "final_info.json"
    if not final_info.exists():
        raise FileNotFoundError(
            f"{final_info} not written. The run script likely crashed or skipped saving results."
        )

    run_notes = run_dir / "notes.txt"
    metrics = _load_json(final_info)
    write_run_notes(run_notes, run_name, metrics)


def _load_json(path: Path) -> Union[Dict, list]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def flatten_metrics(metrics: Union[Dict, list]) -> Union[Dict, list]:
    """Strip helper fields the coder normally removes before aggregation."""
    if isinstance(metrics, dict):
        cleaned: Dict[str, Union[float, int, str, dict, list]] = {}
        for key, value in metrics.items():
            if isinstance(value, dict) and "means" in value:
                cleaned[key] = value["means"]
            else:
                cleaned[key] = value
        return cleaned
    return metrics


def aggregate_results(exp_dir: Path, runs: Iterable[str]) -> Dict[str, Union[Dict, list]]:
    summary: Dict[str, Union[Dict, list]] = {}
    for run_name in runs:
        final_info_path = exp_dir / run_name / "final_info.json"
        metrics = _load_json(final_info_path)
        summary[run_name] = flatten_metrics(metrics)
    return summary


def write_results(exp_dir: Path, summary: Dict[str, Union[Dict, list]]) -> None:
    results_path = exp_dir / "experiment_results.txt"
    with results_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)


def write_run_notes(path: Path, run_name: str, metrics: Union[Dict, list]) -> None:
    lines = [
        f"Auto-generated notes for {run_name}",
        f"Updated: {datetime.utcnow().isoformat()}Z",
        "",
    ]
    if isinstance(metrics, dict):
        for key in sorted(metrics.keys()):
            lines.append(f"- {key}: {metrics[key]}")
    else:
        lines.append("Metrics:")
        for idx, entry in enumerate(metrics, start=1):
            lines.append(f"- entry_{idx}: {entry}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_root_notes(exp_dir: Path, summary: Dict[str, Union[Dict, list]]) -> None:
    lines = [
        "Auto-generated experiment notes",
        f"Updated: {datetime.utcnow().isoformat()}Z",
        "",
    ]
    for run_name in sorted(summary.keys()):
        lines.append(f"{run_name}:")
        data = summary[run_name]
        if isinstance(data, dict):
            for key in sorted(data.keys()):
                lines.append(f"  - {key}: {data[key]}")
        else:
            for idx, entry in enumerate(data, start=1):
                lines.append(f"  - entry_{idx}: {entry}")
        lines.append("")
    notes_path = exp_dir / "notes.txt"
    notes_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def detect_experiment_dirs(base_dir: Path) -> List[Path]:
    """Return a list of experiment directories that contain run scripts."""
    if list(base_dir.glob("run_*.py")):
        return [base_dir]

    experiment_dirs: List[Path] = []
    for sub_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        if list(sub_dir.glob("run_*.py")):
            experiment_dirs.append(sub_dir)

    return experiment_dirs


def rerun_experiment(exp_dir: Path, runs: List[str], python_bin: str, keep_existing: bool) -> None:
    if runs:
        run_pairs = []
        for run in runs:
            script = exp_dir / f"{run}.py"
            if not script.exists():
                raise FileNotFoundError(f"Requested run script {script} missing.")
            run_pairs.append((run, script))
    else:
        run_pairs = list(discover_runs(exp_dir))
        if not run_pairs:
            raise FileNotFoundError(f"No run_*.py files found under {exp_dir}.")

    executed_runs = []
    for run_name, script_path in run_pairs:
        run_single(run_name, script_path, exp_dir, python_bin, keep_existing)
        executed_runs.append(run_name)

    summary = aggregate_results(exp_dir, executed_runs)
    write_results(exp_dir, summary)
    write_root_notes(exp_dir, summary)
    print(f"[{exp_dir}] Re-generated runs: {', '.join(executed_runs)}")
    print(f"[{exp_dir}] Wrote {exp_dir / 'experiment_results.txt'} and {exp_dir / 'notes.txt'}")


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.experiment_dir).resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory {exp_dir} does not exist.")

    experiment_dirs = detect_experiment_dirs(exp_dir)
    if not experiment_dirs:
        raise FileNotFoundError(
            f"No experiment directories found under {exp_dir}. "
            "Expect either run_*.py files directly or idea-* subdirectories."
        )

    for target_dir in experiment_dirs:
        rerun_experiment(target_dir, args.runs or [], args.python_bin, args.keep_existing)


if __name__ == "__main__":
    main()
