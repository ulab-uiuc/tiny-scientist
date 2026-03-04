"""
Integration test: generate-initial → coder → writer → reviewer

Run against a live backend:
    python tests/test_backend_workflow.py [--url URL] [--model MODEL] [--intent INTENT]

Environment variables (alternatively):
    OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY
    TEST_MODEL   (default: deepseek-chat)
    TEST_INTENT  (default: "applying GNNs to improve LLM reasoning")
    TEST_S2_KEY  (Semantic Scholar key, required for /api/write)
    BACKEND_URL  (default: http://localhost:5000)

Exit code 0 = all required steps passed, 1 = any required step failed.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_PATH = os.path.join(os.path.dirname(__file__), "test_backend_workflow.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="w"),
    ],
)
log = logging.getLogger(__name__)

SEPARATOR = "=" * 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _post(url: str, payload: Dict[str, Any], timeout: int = 600) -> Tuple[int, Dict]:
    resp = requests.post(url, json=payload, timeout=timeout)
    try:
        body = resp.json()
    except Exception:
        body = {"_raw": resp.text}
    return resp.status_code, body


def _get(url: str, timeout: int = 30) -> Tuple[int, Dict]:
    resp = requests.get(url, timeout=timeout)
    try:
        body = resp.json()
    except Exception:
        body = {"_raw": resp.text}
    return resp.status_code, body


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Individual test steps
# ---------------------------------------------------------------------------


def step_health_check(base: str) -> None:
    """Backend must be reachable; /api/generate-initial without config returns 400."""
    code, body = _post(f"{base}/api/generate-initial", {"intent": "test"}, timeout=10)
    _assert(
        code in (400, 200),
        f"Expected 400 (not configured) or 200, got {code}: {body}",
    )
    log.info("  backend reachable, status_code=%d", code)


def step_configure(base: str, model: str, api_key: str) -> None:
    code, body = _post(
        f"{base}/api/configure",
        {"model": model, "api_key": api_key},
    )
    _assert(code == 200, f"configure failed {code}: {body}")
    _assert(body.get("status") == "configured", f"unexpected body: {body}")
    log.info("  model=%s  budget=%s", body.get("model"), body.get("budget"))


def step_generate_initial(base: str, intent: str) -> Dict[str, Any]:
    """
    Call /api/generate-initial and validate the idea shape required by coder.
    Returns the idea node dict (with id, title, content, originalData).
    """
    log.info("  intent: %r", intent)
    code, body = _post(f"{base}/api/generate-initial", {"intent": intent}, timeout=300)
    _assert(code == 200, f"generate-initial failed {code}: {body.get('error', body)}")

    ideas = body.get("ideas", [])
    _assert(len(ideas) > 0, "generate-initial returned no ideas")

    node = ideas[0]
    log.info("  idea id=%s  title=%r", node.get("id"), node.get("title"))

    original = node.get("originalData", {})
    _assert(isinstance(original, dict), "originalData is not a dict")
    _assert(bool(original), "originalData is empty")

    # ---- Fields required by coder ----------------------------------------
    missing_top = [
        k
        for k in ("Title", "Problem", "Approach", "NoveltyComparison")
        if not original.get(k)
    ]
    if missing_top:
        log.warning("  WARN: missing top-level fields: %s", missing_top)

    is_exp = original.get("is_experimental", True)
    log.info("  is_experimental=%s", is_exp)

    if is_exp:
        experiment = original.get("Experiment")
        _assert(
            isinstance(experiment, dict),
            f"is_experimental=True but Experiment field is {type(experiment).__name__!r}: {experiment}",
        )
        missing_exp = [
            k for k in ("Model", "Dataset", "Metric") if not experiment.get(k)
        ]
        _assert(
            not missing_exp,
            f"Experiment dict missing required keys: {missing_exp}  got: {list(experiment.keys())}",
        )
        log.info("  Experiment keys: %s", list(experiment.keys()))
    else:
        log.info("  non-experimental idea — skipping Experiment field check")

    return node


def step_coder(base: str, node: Dict[str, Any]) -> Optional[str]:
    """
    POST to /api/code.  Returns experiment_dir (relative) on success, None on failure.
    A coder failure is reported but does NOT fail the overall test suite
    (it is an LLM-dependent step that can fail for environment reasons).
    """
    original = node.get("originalData", {})
    if not original.get("is_experimental", True):
        log.info("  non-experimental idea — skipping coder step")
        return None

    log.info("  sending idea %r to coder …", node.get("title"))
    code, body = _post(
        f"{base}/api/code",
        {"idea": node, "baseline_results": {}},
        timeout=600,
    )
    _assert(code == 200, f"/api/code HTTP error {code}: {body.get('error', body)}")

    success = body.get("success", False)
    exp_dir = body.get("experiment_dir")
    log.info("  coder success=%s  experiment_dir=%s", success, exp_dir)
    if not success:
        log.warning("  coder reported failure: %s", body.get("error_details", "")[:300])

    return exp_dir if success else None


def step_writer(
    base: str, node: Dict[str, Any], exp_dir: Optional[str], s2_key: str
) -> Optional[str]:
    """
    POST to /api/write.  Returns the API pdf_path on success, None otherwise.
    Also a soft step — writer depends on LaTeX and can fail outside CI.
    """
    log.info("  sending idea %r to writer …", node.get("title"))
    payload: Dict[str, Any] = {"idea": node, "s2_api_key": s2_key}
    if exp_dir:
        payload["experiment_dir"] = exp_dir

    code, body = _post(f"{base}/api/write", payload, timeout=600)
    _assert(code == 200, f"/api/write HTTP error {code}: {body.get('error', body)}")

    success = body.get("success", False)
    pdf_path = body.get("pdf_path")
    log.info("  writer success=%s  pdf_path=%s", success, pdf_path)
    if not success:
        log.warning("  writer error: %s", body.get("error", "")[:300])

    return pdf_path if success else None


def step_reviewer(base: str, pdf_path: str, s2_key: str) -> None:
    """POST to /api/review and validate the review structure."""
    log.info("  reviewing pdf_path=%s …", pdf_path)
    code, body = _post(
        f"{base}/api/review",
        {"pdf_path": pdf_path, "s2_api_key": s2_key},
        timeout=300,
    )
    _assert(code == 200, f"/api/review HTTP error {code}: {body.get('error', body)}")
    _assert(body.get("success"), f"review reported failure: {body.get('error', body)}")

    review = body.get("review", {})
    _assert(isinstance(review, dict), f"review is not a dict: {type(review)}")
    log.info("  review keys: %s", list(review.keys()))


def step_clear_session(base: str) -> None:
    code, body = _post(f"{base}/api/clear-session", {})
    _assert(code == 200, f"clear-session failed {code}: {body}")
    log.info("  session cleared")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class Result:
    def __init__(self, name: str, required: bool = True):
        self.name = name
        self.required = required
        self.status: str = "pending"  # pass | fail | skip
        self.detail: str = ""
        self.elapsed: float = 0.0


def run_step(result: Result, fn, *args, **kwargs) -> bool:
    log.info(SEPARATOR)
    log.info("STEP: %s%s", result.name, "" if result.required else "  [optional]")
    log.info(SEPARATOR)
    t0 = time.time()
    try:
        fn(*args, **kwargs)
        result.status = "pass"
        result.elapsed = time.time() - t0
        log.info("✓ PASS  (%.1fs)", result.elapsed)
        return True
    except AssertionError as exc:
        result.status = "fail"
        result.detail = str(exc)
        result.elapsed = time.time() - t0
        log.info("✗ FAIL  %s  (%.1fs)", exc, result.elapsed)
        return False
    except Exception as exc:
        result.status = "fail"
        result.detail = str(exc)
        result.elapsed = time.time() - t0
        log.info("✗ ERROR  %s  (%.1fs)", exc, result.elapsed)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Backend workflow integration test")
    parser.add_argument(
        "--url", default=os.environ.get("BACKEND_URL", "http://localhost:5000")
    )
    parser.add_argument(
        "--model", default=os.environ.get("TEST_MODEL", "deepseek-chat")
    )
    parser.add_argument(
        "--intent",
        default=os.environ.get("TEST_INTENT", "applying GNNs to improve LLM reasoning"),
    )
    parser.add_argument("--s2-key", default=os.environ.get("TEST_S2_KEY", ""))
    parser.add_argument(
        "--skip-coder", action="store_true", help="Skip the /api/code step"
    )
    parser.add_argument(
        "--skip-writer", action="store_true", help="Skip the /api/write step"
    )
    parser.add_argument(
        "--skip-reviewer", action="store_true", help="Skip the /api/review step"
    )
    args = parser.parse_args()

    # Resolve API key from env
    key_map = {
        "deepseek-chat": "DEEPSEEK_API_KEY",
        "deepseek-reasoner": "DEEPSEEK_API_KEY",
        "claude-opus-4-6": "ANTHROPIC_API_KEY",
        "claude-sonnet-4-5": "ANTHROPIC_API_KEY",
        "gpt-5.2": "OPENAI_API_KEY",
        "gpt-5-mini": "OPENAI_API_KEY",
    }
    env_var = key_map.get(args.model, "OPENAI_API_KEY")
    api_key = os.environ.get(env_var, "")
    if not api_key:
        log.error("No API key found in $%s — set it before running.", env_var)
        return 1

    log.info("Backend Workflow Integration Test")
    log.info("Intent : %r", args.intent)
    log.info("URL    : %s", args.url)
    log.info("Started: %s", datetime.now().isoformat())
    log.info("Model  : %s", args.model)
    log.info("")

    results = []
    node: Optional[Dict[str, Any]] = None
    exp_dir: Optional[str] = None
    pdf_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Required steps
    # ------------------------------------------------------------------
    r = Result("health_check", required=True)
    results.append(r)
    if not run_step(r, step_health_check, args.url):
        log.error("Backend unreachable — aborting.")
        _print_summary(results)
        return 1

    r = Result("configure", required=True)
    results.append(r)
    if not run_step(r, step_configure, args.url, args.model, api_key):
        _print_summary(results)
        return 1

    # generate-initial is the core required step
    r = Result("generate_initial", required=True)
    results.append(r)

    def _generate_and_capture():
        nonlocal node
        node = step_generate_initial(args.url, args.intent)

    if not run_step(r, _generate_and_capture):
        _print_summary(results)
        return 1

    # ------------------------------------------------------------------
    # Optional pipeline steps (failures logged but don't kill exit code)
    # ------------------------------------------------------------------
    if not args.skip_coder and node is not None:
        r = Result("coder", required=False)
        results.append(r)

        def _code_and_capture():
            nonlocal exp_dir
            exp_dir = step_coder(args.url, node)

        run_step(r, _code_and_capture)

    s2_key = args.s2_key
    if not args.skip_writer and node is not None and s2_key:
        r = Result("writer", required=False)
        results.append(r)

        def _write_and_capture():
            nonlocal pdf_path
            pdf_path = step_writer(args.url, node, exp_dir, s2_key)

        run_step(r, _write_and_capture)
    elif not args.skip_writer and not s2_key:
        log.info("")
        log.info("Skipping writer — no TEST_S2_KEY / --s2-key provided.")

    if not args.skip_reviewer and pdf_path and s2_key:
        r = Result("reviewer", required=False)
        results.append(r)
        run_step(r, step_reviewer, args.url, pdf_path, s2_key)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    r = Result("clear_session", required=False)
    results.append(r)
    run_step(r, step_clear_session, args.url)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _print_summary(results)

    # Persist results JSON
    results_path = os.path.join(
        os.path.dirname(__file__), "test_backend_workflow_results.json"
    )
    with open(results_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "model": args.model,
                "intent": args.intent,
                "results": [
                    {
                        "name": r.name,
                        "required": r.required,
                        "status": r.status,
                        "elapsed": round(r.elapsed, 2),
                        "detail": r.detail,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )
    log.info("Results → %s", results_path)
    log.info("Log     → %s", LOG_PATH)
    log.info(SEPARATOR)

    # Fail if any required step failed
    failed_required = [r for r in results if r.required and r.status == "fail"]
    return 1 if failed_required else 0


def _print_summary(results) -> None:
    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    log.info("")
    log.info(SEPARATOR)
    log.info("SUMMARY  %d passed / %d failed  (total %d)", passed, failed, len(results))
    for r in results:
        icon = "✓" if r.status == "pass" else ("✗" if r.status == "fail" else "·")
        opt = "" if r.required else "  [optional]"
        detail = f"  — {r.detail}" if r.detail else ""
        log.info(
            "  %s [%-5s] %s  (%.1fs)%s%s",
            icon,
            r.status.upper(),
            r.name,
            r.elapsed,
            opt,
            detail,
        )


if __name__ == "__main__":
    sys.exit(main())
