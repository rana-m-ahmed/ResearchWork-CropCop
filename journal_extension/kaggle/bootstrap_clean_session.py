from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.g1 import validate_dependency_environment, validate_dependency_lock_object
from cropcop_je.session import SessionBudget
from cropcop_je.source_state import verify_clean_source

OUTPUT_ENV_KEYS = (
    "CROPCOP_OUTPUT_ROOT",
    "CROPCOP_SYNTHETIC_SMOKE_ROOT",
    "CROPCOP_SMOKE_A_EXPORT_ROOT",
    "CROPCOP_SMOKE_B_EXPORT_ROOT",
    "CROPCOP_G1_BUNDLE_DIR",
    "CROPCOP_G2_SUMMARIES_DIR",
    "CROPCOP_TERMINAL_EVIDENCE_DIR",
)
SMOKE_PHASES = {"smoke-write", "smoke-restore"}
NON_SMOKE_PHASES = {"g1", "calibration", "principal"}


def _required_secret_names(phase: str) -> tuple[str, ...]:
    if phase in SMOKE_PHASES:
        return ("CROPCOP_GITHUB_TOKEN",)
    if phase in NON_SMOKE_PHASES:
        return ("CROPCOP_GITHUB_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY")
    raise ValueError(f"unsupported execution phase: {phase}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--authorized-source-sha", required=True)
    ap.add_argument("--phase", required=True, choices=sorted(SMOKE_PHASES | NON_SMOKE_PHASES))
    ap.add_argument("--dependency-lock", default="journal_extension/locks/execution_dependency_lock.json")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    dep_path = Path(args.dependency_lock)
    if not dep_path.is_absolute():
        dep_path = repo / dep_path
    dep = json.loads(dep_path.read_text(encoding="utf-8"))
    errors = validate_dependency_lock_object(dep) + validate_dependency_environment(dep)

    roots = [os.environ[k] for k in OUTPUT_ENV_KEYS if os.environ.get(k)]
    try:
        source = verify_clean_source(repo, authorized_source_sha=args.authorized_source_sha, output_roots=roots)
    except Exception as exc:
        source = {"status": "FAIL", "error": str(exc)}
        errors.append(str(exc))

    if os.environ.get("CROPCOP_SOURCE_GIT_COMMIT") != args.authorized_source_sha:
        errors.append("CROPCOP_SOURCE_GIT_COMMIT differs from authorized source SHA")
    if os.environ.get("CROPCOP_LANE") not in {"K1", "K2", "K3"}:
        errors.append("CROPCOP_LANE is missing/invalid")
    for name in _required_secret_names(args.phase):
        if not os.environ.get(name):
            errors.append(f"required secret missing: {name}")

    try:
        budget = SessionBudget.from_environment(require_global_clock=True)
        session = budget.snapshot()
    except Exception as exc:
        session = {"status": "FAIL", "error": str(exc)}
        errors.append(str(exc))

    report = {
        "schema_version": "2.0",
        "status": "PASS" if not errors else "FAIL",
        "phase": args.phase,
        "source_state": source,
        "dependency_lock_sha256": dep.get("dependency_lock_sha256"),
        "notebook_session": session,
        "required_secret_names": list(_required_secret_names(args.phase)),
        "secret_presence": {name: bool(os.environ.get(name)) for name in _required_secret_names(args.phase)},
        "errors": errors,
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 3


if __name__ == "__main__":
    raise SystemExit(main())
