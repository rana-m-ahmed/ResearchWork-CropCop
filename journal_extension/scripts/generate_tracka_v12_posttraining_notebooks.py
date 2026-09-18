from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip().splitlines()]}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_BOOTSTRAP = r"""
import json, os, subprocess, sys, time
from pathlib import Path

repo = Path(os.environ["CROPCOP_REPO_ROOT"]).resolve()
analysis_sha = os.environ["CROPCOP_ANALYSIS_SHA"].strip()
account_id = os.environ.get("CROPCOP_ACCOUNT_ID", "").strip()
if len(analysis_sha) != 40:
    raise RuntimeError("CROPCOP_ANALYSIS_SHA must be a full 40-character SHA")
observed = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
if observed != analysis_sha:
    raise RuntimeError(f"exact analysis checkout mismatch: expected={analysis_sha}, observed={observed}")

environment_gate = repo / "journal_extension/scripts/ensure_tracka_locked_environment.py"
subprocess.run([
    sys.executable,
    str(environment_gate),
    "--repo-root", str(repo),
    "--repair",
], cwd=repo, check=True)

os.environ.setdefault("CROPCOP_NOTEBOOK_STARTED_MONOTONIC", repr(time.monotonic()))
os.environ.setdefault("CROPCOP_NOTEBOOK_HARD_LIMIT_SECONDS", "43200")
os.environ.setdefault("CROPCOP_NOTEBOOK_FINALIZATION_MARGIN_SECONDS", "3600")
print({"analysis_sha": analysis_sha, "account_id": account_id or None, "repo": str(repo)})
"""


def build_preflight() -> dict:
    return notebook([
        md("""
# CropCop Track A — Post-Training Preflight

This notebook performs only pre-metric recovery and artifact-availability checks. It does not run validation replay, robustness, XAI, selection, V1-test inference, Track-B predictions, or Track-C candidate evaluation.
"""),
        code(COMMON_BOOTSTRAP),
        code(r"""
required = ["CROPCOP_ACCOUNT_ID", "CROPCOP_PREFLIGHT_SPEC", "CROPCOP_PREFLIGHT_OUTPUT"]
missing = [name for name in required if not os.environ.get(name, "").strip()]
if missing:
    raise RuntimeError(f"missing required environment variables: {missing}")
if account_id not in {"K1", "K2", "K3"}:
    raise RuntimeError("CROPCOP_ACCOUNT_ID must be K1, K2, or K3")
subprocess.run([
    sys.executable,
    str(repo / "journal_extension/scripts/prepare_tracka_v12_posttraining_account.py"),
    "--repo-root", str(repo),
    "--spec", os.environ["CROPCOP_PREFLIGHT_SPEC"],
    "--analysis-source-git-commit", analysis_sha,
    "--output-dir", os.environ["CROPCOP_PREFLIGHT_OUTPUT"],
], cwd=repo, check=True)
"""),
        code(r"""
manifest = Path(os.environ["CROPCOP_PREFLIGHT_OUTPUT"]) / f"{account_id}_POSTTRAINING_PREFLIGHT_MANIFEST.json"
payload = json.loads(manifest.read_text(encoding="utf-8"))
if payload.get("status") != "PASS":
    raise RuntimeError("preflight manifest is not PASS")
print(json.dumps({
    "status": payload["status"],
    "account_id": payload["account_id"],
    "continuation_recovery_count": payload["continuation_recovery_count"],
    "historical_availability_report_sha256": payload["historical_availability_report_sha256"],
    "manifest_sha256": payload["manifest_sha256"],
}, indent=2))
"""),
    ])


def build_execute() -> dict:
    return notebook([
        md("""
# CropCop Track A — Account Readiness / Evidence Execution

Run with CROPCOP_ACCOUNT_PHASE=readiness first on K1/K2/K3. After the global readiness gate is sealed, rerun with CROPCOP_ACCOUNT_PHASE=evidence. The evidence phase uses two isolated T4 workers and never uses DDP/DataParallel/FSDP.
"""),
        code(COMMON_BOOTSTRAP),
        code(r"""
required = [
    "CROPCOP_ACCOUNT_ID", "CROPCOP_ACCOUNT_PHASE", "CROPCOP_MATERIALIZATION_CATALOG",
    "CROPCOP_PLACEMENT_FREEZE", "CROPCOP_ACCOUNT_CONTROL_DIR"
]
missing = [name for name in required if not os.environ.get(name, "").strip()]
if missing:
    raise RuntimeError(f"missing required environment variables: {missing}")
phase = os.environ["CROPCOP_ACCOUNT_PHASE"].strip().lower()
if phase not in {"readiness", "evidence"}:
    raise RuntimeError("CROPCOP_ACCOUNT_PHASE must be readiness or evidence")
control = Path(os.environ["CROPCOP_ACCOUNT_CONTROL_DIR"]).resolve()
control.mkdir(parents=True, exist_ok=True)
inventory = control / f"{account_id}_POSTTRAINING_INVENTORY.json"
target_preflight = control / f"{account_id}_POSTTRAINING_EVIDENCE_TARGETS.json"
readiness = control / f"{account_id}_POSTTRAINING_ACCOUNT_READINESS.json"
subprocess.run([
    sys.executable,
    str(repo / "journal_extension/scripts/build_tracka_v12_posttraining_account_inventory.py"),
    "--repo-root", str(repo),
    "--account-id", account_id,
    "--materialization-catalog", os.environ["CROPCOP_MATERIALIZATION_CATALOG"],
    "--placement-freeze", os.environ["CROPCOP_PLACEMENT_FREEZE"],
    "--analysis-source-git-commit", analysis_sha,
    "--output", str(inventory),
], cwd=repo, check=True)
subprocess.run([
    sys.executable,
    str(repo / "journal_extension/scripts/bootstrap_tracka_v12_posttraining_evidence_targets.py"),
    "--repo-root", str(repo),
    "--account-id", account_id,
    "--account-inventory", str(inventory),
    "--analysis-source-git-commit", analysis_sha,
    "--allow-create", os.environ.get("CROPCOP_POSTTRAINING_ALLOW_CREATE_PRIVATE_DATASET", "0"),
    "--output", str(target_preflight),
], cwd=repo, check=True)
subprocess.run([
    sys.executable,
    str(repo / "journal_extension/scripts/build_tracka_v12_posttraining_account_readiness.py"),
    "--repo-root", str(repo),
    "--inventory", str(inventory),
    "--analysis-source-git-commit", analysis_sha,
    "--evidence-target-preflight", str(target_preflight),
    "--output", str(readiness),
], cwd=repo, check=True)
gate = json.loads(readiness.read_text(encoding="utf-8"))
if gate.get("status") != "PASS":
    raise RuntimeError("account readiness gate is not PASS")
print(json.dumps({"phase": phase, "account_id": account_id, "readiness_gate_sha256": gate["gate_sha256"]}, indent=2))
"""),
        code(r"""
if phase == "evidence":
    required = ["CROPCOP_GLOBAL_READINESS", "CROPCOP_ACCOUNT_WORK_ROOT", "CROPCOP_ACCOUNT_COMPLETION"]
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    if missing:
        raise RuntimeError(f"evidence phase missing required environment variables: {missing}")
    global_gate = json.loads(Path(os.environ["CROPCOP_GLOBAL_READINESS"]).read_text(encoding="utf-8"))
    if global_gate.get("status") != "PASS" or global_gate.get("ready_for_posttraining_evidence") is not True:
        raise RuntimeError("global readiness must be PASS before evidence execution")
    cp = subprocess.run([
        sys.executable,
        str(repo / "journal_extension/scripts/run_tracka_v12_posttraining_account.py"),
        "--repo-root", str(repo),
        "--account-id", account_id,
        "--account-inventory", str(inventory),
        "--account-readiness", str(readiness),
        "--global-readiness", os.environ["CROPCOP_GLOBAL_READINESS"],
        "--placement-freeze", os.environ["CROPCOP_PLACEMENT_FREEZE"],
        "--analysis-source-git-commit", analysis_sha,
        "--work-root", os.environ["CROPCOP_ACCOUNT_WORK_ROOT"],
        "--output", os.environ["CROPCOP_ACCOUNT_COMPLETION"],
    ], cwd=repo, check=False)
    if cp.returncode not in {0, 20}:
        raise RuntimeError(f"post-training account operator failed rc={cp.returncode}")
    result = json.loads(Path(os.environ["CROPCOP_ACCOUNT_COMPLETION"]).read_text(encoding="utf-8"))
    print(json.dumps({
        "status": result["status"],
        "account_id": result["account_id"],
        "assigned_state_count": result["assigned_state_count"],
        "completed_state_count": result["completed_state_count"],
        "manifest_sha256": result["manifest_sha256"],
    }, indent=2))
else:
    print("Readiness-only phase complete. Do not start evidence until the global readiness gate is sealed.")
"""),
    ])


def build_global() -> dict:
    return notebook([
        md("""
# CropCop Track A — Global Control / Closure

Use CROPCOP_GLOBAL_PHASE=placement, then readiness, then closure. Each phase is fail-closed and consumes only artifacts from the preceding frozen gates.
"""),
        code(COMMON_BOOTSTRAP),
        code(r"""
phase = os.environ.get("CROPCOP_GLOBAL_PHASE", "").strip().lower()
if phase not in {"placement", "readiness", "closure"}:
    raise RuntimeError("CROPCOP_GLOBAL_PHASE must be placement, readiness, or closure")
global_dir = Path(os.environ["CROPCOP_GLOBAL_CONTROL_DIR"]).resolve()
global_dir.mkdir(parents=True, exist_ok=True)
placement = global_dir / "TRACKA_POSTTRAINING_PLACEMENT_FREEZE.json"
global_readiness = global_dir / "POSTTRAINING_READINESS_GATE.json"
print({"global_phase": phase, "global_control_dir": str(global_dir)})
"""),
        code(r"""
if phase == "placement":
    reports = [os.environ[f"CROPCOP_{account}_AVAILABILITY_REPORT"] for account in ("K1", "K2", "K3")]
    command = [
        sys.executable,
        str(repo / "journal_extension/scripts/freeze_tracka_v12_posttraining_placement.py"),
        "--repo-root", str(repo),
        "--analysis-source-git-commit", analysis_sha,
        "--output", str(placement),
    ]
    for report in reports:
        command.extend(["--availability-report", report])
    subprocess.run(command, cwd=repo, check=True)
    payload = json.loads(placement.read_text(encoding="utf-8"))
    print(json.dumps({"status": payload["status"], "placement_freeze_sha256": payload["placement_freeze_sha256"]}, indent=2))
"""),
        code(r"""
if phase == "readiness":
    gates = [os.environ[f"CROPCOP_{account}_READINESS"] for account in ("K1", "K2", "K3")]
    command = [
        sys.executable,
        str(repo / "journal_extension/scripts/build_tracka_v12_posttraining_readiness.py"),
        "--repo-root", str(repo),
        "--analysis-source-git-commit", analysis_sha,
        "--output", str(global_readiness),
    ]
    for gate in gates:
        command.extend(["--account-readiness", gate])
    subprocess.run(command, cwd=repo, check=True)
    payload = json.loads(global_readiness.read_text(encoding="utf-8"))
    print(json.dumps({
        "status": payload["status"],
        "state_count": payload["state_count"],
        "unique_selected_checkpoint_count": payload["unique_selected_checkpoint_count"],
        "gate_sha256": payload["gate_sha256"],
    }, indent=2))
"""),
        code(r"""
if phase == "closure":
    account_completions = [os.environ[f"CROPCOP_{account}_COMPLETION"] for account in ("K1", "K2", "K3")]
    audit_dir = global_dir / "evidence_audit"
    if audit_dir.exists():
        raise RuntimeError("evidence_audit directory already exists; closure phase is append-only")
    command = [
        sys.executable,
        str(repo / "journal_extension/scripts/audit_tracka_v12_posttraining_evidence.py"),
        "--repo-root", str(repo),
        "--analysis-source-git-commit", analysis_sha,
        "--global-readiness", str(global_readiness),
        "--output-dir", str(audit_dir),
    ]
    for manifest in account_completions:
        command.extend(["--account-completion", manifest])
    subprocess.run(command, cwd=repo, check=True)
    closure_dir = global_dir / "final_closure"
    subprocess.run([
        sys.executable,
        str(repo / "journal_extension/scripts/close_tracka_v12.py"),
        "--repo-root", str(repo),
        "--analysis-source-git-commit", analysis_sha,
        "--global-evidence-audit", str(audit_dir / "TRACKA_POSTTRAINING_GLOBAL_EVIDENCE_AUDIT.json"),
        "--direct-evidence-index", str(audit_dir / "TRACKA_DIRECT_EVIDENCE_INDEX.json"),
        "--auxiliary-evidence-index", str(audit_dir / "TRACKA_AUXILIARY_EVIDENCE_INDEX.json"),
        "--output-dir", str(closure_dir),
    ], cwd=repo, check=True)
    final = json.loads((closure_dir / "TRACKA_FINAL_CLOSURE_AUDIT.json").read_text(encoding="utf-8"))
    if final.get("status") != "PASS" or final.get("track_a_closed") is not True:
        raise RuntimeError("final Track-A closure audit is not PASS")
    print(json.dumps(final, indent=2, sort_keys=True))
"""),
    ])


def write_notebook(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    write_notebook(root / "notebooks" / "tracka_posttraining_preflight.ipynb", build_preflight())
    write_notebook(root / "notebooks" / "tracka_posttraining_execute.ipynb", build_execute())
    write_notebook(root / "notebooks" / "tracka_posttraining_global.ipynb", build_global())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
