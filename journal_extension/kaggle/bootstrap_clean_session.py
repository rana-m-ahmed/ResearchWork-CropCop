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
    "CROPCOP_G1_BUNDLE_DIR",
    "CROPCOP_G2_SUMMARIES_DIR",
    "CROPCOP_TERMINAL_EVIDENCE_DIR",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--authorized-source-sha", required=True)
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
    if not os.environ.get("CROPCOP_GITHUB_TOKEN"):
        errors.append("private Git credential was not retrieved into the process environment")
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        errors.append("Kaggle API credentials were not retrieved into the process environment")

    try:
        budget = SessionBudget.from_environment(require_global_clock=True)
        session = budget.snapshot()
    except Exception as exc:
        session = {"status": "FAIL", "error": str(exc)}
        errors.append(str(exc))

    report = {
        "schema_version": "1.0",
        "status": "PASS" if not errors else "FAIL",
        "source_state": source,
        "dependency_lock_sha256": dep.get("dependency_lock_sha256"),
        "notebook_session": session,
        "secret_presence": {
            "private_git": bool(os.environ.get("CROPCOP_GITHUB_TOKEN")),
            "kaggle_username": bool(os.environ.get("KAGGLE_USERNAME")),
            "kaggle_key": bool(os.environ.get("KAGGLE_KEY")),
        },
        "errors": errors,
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 3


if __name__ == "__main__":
    raise SystemExit(main())
