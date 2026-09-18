from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path

SCIENCE_BASE = "56023042e57758591df9babb3438f191dbe10312"
FROZEN_SCIENCE_FILES = (
    "journal_extension/scripts/run_tracka_v12_direct_evidence.py",
    "journal_extension/scripts/run_tracka_v12_auxiliary_evidence.py",
    "journal_extension/scripts/run_tracka_v12_xai.py",
    "journal_extension/scripts/seal_tracka_v12_selection.py",
    "journal_extension/scripts/seal_tracka_v12_auxiliary_analysis.py",
    "journal_extension/scripts/seal_tracka_v12_comprehensive_closure.py",
    "journal_extension/src/cropcop_je/tracka_v12_analysis.py",
    "journal_extension/src/cropcop_je/tracka_v12_evidence.py",
    "journal_extension/src/cropcop_je/tracka_v12_posttraining.py",
    "journal_extension/src/cropcop_je/tracka_v12_xai.py",
)
INFRA_SCRIPTS = (
    "journal_extension/scripts/recover_tracka_v12_terminal_record.py",
    "journal_extension/scripts/probe_tracka_v12_historical_artifacts.py",
    "journal_extension/scripts/freeze_tracka_v12_posttraining_placement.py",
    "journal_extension/scripts/prepare_tracka_v12_posttraining_account.py",
    "journal_extension/scripts/build_tracka_v12_posttraining_account_inventory.py",
    "journal_extension/scripts/build_tracka_v12_posttraining_account_readiness.py",
    "journal_extension/scripts/build_tracka_v12_posttraining_readiness.py",
    "journal_extension/scripts/run_tracka_v12_posttraining_account.py",
    "journal_extension/scripts/sync_tracka_v12_posttraining_evidence.py",
    "journal_extension/scripts/publish_tracka_v12_posttraining_state.py",
    "journal_extension/scripts/audit_tracka_v12_posttraining_evidence.py",
    "journal_extension/scripts/close_tracka_v12.py",
    "journal_extension/scripts/generate_tracka_v12_posttraining_notebooks.py",
)
NOTEBOOKS = (
    "notebooks/tracka_posttraining_preflight.ipynb",
    "notebooks/tracka_posttraining_execute.ipynb",
    "notebooks/tracka_posttraining_global.ipynb",
)
SECRET_PATTERN = re.compile(r"(gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|KAGGLE_KEY\s*=\s*['\"][^'\"]+)", re.I)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def git_blob(repo: Path, ref: str, path: str) -> str:
    return git(repo, "rev-parse", f"{ref}:{path}")


def validate_notebook(path: Path) -> list[str]:
    errors = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("nbformat") != 4:
        errors.append(f"{path}: nbformat")
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        errors.append(f"{path}: empty cells")
        return errors
    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            if cell.get("execution_count") is not None:
                errors.append(f"{path}: code cell {index} execution_count")
            if cell.get("outputs") != []:
                errors.append(f"{path}: code cell {index} has outputs")
    text = path.read_text(encoding="utf-8")
    if SECRET_PATTERN.search(text):
        errors.append(f"{path}: live-looking secret")
    for forbidden in ("DS-V1-TEST-CONSUMED", "TRACK-B-PREDICTIONS", "TRACK-C-CANDIDATE-RESULTS"):
        if forbidden in text and path.name != "tracka_posttraining_preflight.ipynb":
            pass
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--expected-head", required=True)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    errors = []
    observed = git(repo, "rev-parse", "HEAD")
    if observed != args.expected_head:
        errors.append("exact head mismatch")
    try:
        subprocess.check_call(["git", "-C", str(repo), "merge-base", "--is-ancestor", SCIENCE_BASE, observed])
    except subprocess.CalledProcessError:
        errors.append("science base is not ancestor of analysis head")

    for path in FROZEN_SCIENCE_FILES:
        if git_blob(repo, SCIENCE_BASE, path) != git_blob(repo, observed, path):
            errors.append(f"frozen science drift: {path}")

    for path in INFRA_SCRIPTS:
        full = repo / path
        if not full.is_file():
            errors.append(f"missing infrastructure script: {path}")
            continue
        try:
            ast.parse(full.read_text(encoding="utf-8"), filename=path)
        except SyntaxError as exc:
            errors.append(f"syntax error {path}:{exc.lineno}:{exc.msg}")

    for path in NOTEBOOKS:
        full = repo / path
        if not full.is_file():
            errors.append(f"missing canonical notebook: {path}")
        else:
            errors.extend(validate_notebook(full))

    campaign = json.loads((repo / "journal_extension/locks/track_a_posttraining_campaign_v1.json").read_text(encoding="utf-8"))
    authority = json.loads((repo / "journal_extension/locks/track_a_posttraining_closure_authority_v1.json").read_text(encoding="utf-8"))
    if campaign.get("scientific_semantics_change_authorized") is not False:
        errors.append("campaign authorizes scientific semantics change")
    if len(authority.get("state_inventory", {})) != 21:
        errors.append("closure authority state count")
    if set(campaign.get("continuation_account_binding", {})) != {
        state for state, row in authority["state_inventory"].items()
        if row.get("terminal_metadata_recovery_required") is True
    }:
        errors.append("continuation binding differs from closure authority")
    protected = campaign.get("protected_surfaces", {})
    if not protected or any(value != "CLOSED" for value in protected.values()):
        errors.append("campaign protected surface open")

    operator = (repo / "journal_extension/scripts/run_tracka_v12_posttraining_account.py").read_text(encoding="utf-8")
    for token in (
        "--global-readiness",
        "DEFERRED_SESSION_BUDGET",
        "REUSED_COMPLETE",
        "CUDA_VISIBLE_DEVICES",
        "ThreadPoolExecutor(max_workers=2)",
        "sync_tracka_v12_posttraining_evidence.py",
        "publish_tracka_v12_posttraining_state.py",
    ):
        if token not in operator:
            errors.append(f"account operator missing control: {token}")
    for forbidden in ("DistributedDataParallel", "DataParallel(", "FullyShardedDataParallel"):
        if forbidden in operator:
            errors.append(f"account operator contains forbidden multi-GPU training primitive: {forbidden}")

    global_audit = (repo / "journal_extension/scripts/audit_tracka_v12_posttraining_evidence.py").read_text(encoding="utf-8")
    if "validate_direct_state_evidence_bundle" not in global_audit:
        errors.append("global audit does not independently validate direct bundles")
    if "FULL_TRACK_A" not in global_audit or "selector_authorized" not in global_audit:
        errors.append("global audit lacks 21-state selector gate")

    closer = (repo / "journal_extension/scripts/close_tracka_v12.py").read_text(encoding="utf-8")
    for token in ("global evidence audit does not authorize Track-A closure", "CO_PRIMARY_TIE", "track_a_closed"):
        if token not in closer:
            errors.append(f"final closure controller missing control: {token}")

    result = {
        "schema_version": "1.0",
        "status": "PASS" if not errors else "FAIL",
        "validation_kind": "track_a_posttraining_infrastructure",
        "analysis_source_git_commit": observed,
        "training_science_source_git_commit": SCIENCE_BASE,
        "frozen_science_files_checked": len(FROZEN_SCIENCE_FILES),
        "infrastructure_scripts_checked": len(INFRA_SCRIPTS),
        "canonical_notebooks_checked": len(NOTEBOOKS),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
