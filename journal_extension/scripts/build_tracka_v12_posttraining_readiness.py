from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.checkpointing import verify_selected
from cropcop_je.hashing import sha256_file, sha256_json
from cropcop_je.tracka_v12 import CLASS_MAP_SHA256, EXPERIMENT_SPECS, MANIFEST_SHA256
from cropcop_je.tracka_v12_evidence import ALL_DIRECT_STATES
from cropcop_je.tracka_v12_historical import HISTORICAL_TRACKA_CLOSURE_SPECS, validate_historical_closure_identity
from cropcop_je.tracka_v12_recovery import SCIENCE_SOURCE_SHA, validate_recovered_terminal_record
from cropcop_je.tracka_v12_runtime import load_and_validate_g1a_bundle
from cropcop_je.train import _identity as checkpoint_identity

AUXILIARY_STATES = {
    *(f"R05-MNV4-TEACHER-{seed}" for seed in ("S1", "S2", "S3")),
    *(f"R12-MNV4-LOGITS-{seed}" for seed in ("S1", "S2", "S3")),
    *(f"R12-MNV4-FEATURE-{seed}" for seed in ("S1", "S2", "S3")),
}
ALL_STATES = set(ALL_DIRECT_STATES) | AUXILIARY_STATES
NEW_STATES = set(EXPERIMENT_SPECS)
HISTORICAL_STATES = set(HISTORICAL_TRACKA_CLOSURE_SPECS)


def load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def file_required(path: str | Path, label: str) -> Path:
    p = Path(path).resolve()
    if not p.is_file():
        raise RuntimeError(f"required {label} file missing: {p}")
    return p


def directory_required(path: str | Path, label: str) -> Path:
    p = Path(path).resolve()
    if not p.is_dir():
        raise RuntimeError(f"required {label} directory missing: {p}")
    return p


def validate_inventory(inventory: dict, repo: Path, analysis_source: str) -> dict:
    if inventory.get("schema_version") != "1.0":
        raise RuntimeError("post-training inventory schema must be 1.0")
    if inventory.get("analysis_source_git_commit") != analysis_source:
        raise RuntimeError("inventory analysis-source SHA differs from current exact checkout")
    if inventory.get("training_science_source_git_commit") != SCIENCE_SOURCE_SHA:
        raise RuntimeError("inventory training science source mismatch")

    manifest = file_required(inventory["manifest"], "frozen manifest")
    class_map = file_required(inventory["class_map"], "class map")
    directory_required(inventory["image_root"], "image root")
    if sha256_file(manifest) != MANIFEST_SHA256:
        raise RuntimeError("frozen manifest SHA mismatch")
    if sha256_file(class_map) != CLASS_MAP_SHA256:
        raise RuntimeError("class-map SHA mismatch")

    g1a_bundle = directory_required(inventory["g1a_bundle"], "G1A bundle")
    _seal, g1a_errors = load_and_validate_g1a_bundle(g1a_bundle, expected_source_sha=SCIENCE_SOURCE_SHA)
    if g1a_errors:
        raise RuntimeError("G1A bundle failed validation: " + "; ".join(g1a_errors))

    states = inventory.get("states") or {}
    if set(states) != ALL_STATES:
        missing = sorted(ALL_STATES - set(states))
        extra = sorted(set(states) - ALL_STATES)
        raise RuntimeError(f"inventory must contain exact 21-state Track-A set; missing={missing}, extra={extra}")

    rows = {}
    selected_hashes = set()
    analysis_accounts = set()
    for experiment_id in sorted(ALL_STATES):
        spec = states[experiment_id]
        role = spec.get("role")
        expected_role = "direct" if experiment_id in set(ALL_DIRECT_STATES) else "auxiliary"
        if role != expected_role:
            raise RuntimeError(f"role mismatch for {experiment_id}: {role!r}")
        account_id = str(spec.get("analysis_account_id", ""))
        slot_id = str(spec.get("analysis_slot_id", ""))
        if account_id not in {"K1", "K2", "K3"} or slot_id not in {f"{account_id}/GPU0", f"{account_id}/GPU1"}:
            raise RuntimeError(f"invalid frozen analysis placement for {experiment_id}")
        analysis_accounts.add(account_id)

        run_record_path = file_required(spec["run_record"], f"run record for {experiment_id}")
        record = load_json(run_record_path)
        if record.get("experiment_id") != experiment_id:
            raise RuntimeError(f"run-record experiment mismatch for {experiment_id}")
        if record.get("status") != "PASS" or record.get("continuation_required") not in {None, False}:
            raise RuntimeError(f"run record is not terminal PASS for {experiment_id}")
        if record.get("v1_test_accessed") not in {None, False}:
            raise RuntimeError(f"run record indicates V1-test access for {experiment_id}")
        if record.get("protected_external_surface_accessed") not in {None, False}:
            raise RuntimeError(f"run record indicates protected external access for {experiment_id}")

        if experiment_id in NEW_STATES:
            recovery_errors = validate_recovered_terminal_record(record)
            if recovery_errors:
                raise RuntimeError(f"recovered run record invalid for {experiment_id}: {recovery_errors}")
            certificate_path = file_required(spec["recovery_certificate"], f"recovery certificate for {experiment_id}")
            certificate = load_json(certificate_path)
            if certificate.get("status") != "PASS" or certificate.get("experiment_id") != experiment_id:
                raise RuntimeError(f"recovery certificate invalid for {experiment_id}")
            if certificate.get("recovered_run_record_sha256") != sha256_file(run_record_path):
                raise RuntimeError(f"recovery certificate/run-record hash mismatch for {experiment_id}")
        else:
            historical_errors = validate_historical_closure_identity(record)
            if historical_errors:
                raise RuntimeError(f"historical run record invalid for {experiment_id}: {historical_errors}")

        checkpoint_root = directory_required(spec["checkpoint_root"], f"checkpoint root for {experiment_id}")
        selected_expected = ((record.get("artifact_locators") or {}).get("selected_checkpoint") or {}).get("sha256")
        if not selected_expected:
            selected_expected = (record.get("result_summary") or {}).get("selected_checkpoint_sha256")
        checkpoint_path, checkpoint_payload = verify_selected(
            checkpoint_root,
            expected_identity=checkpoint_identity(record),
            expected_sha256=str(selected_expected),
        )
        if selected_expected in selected_hashes:
            raise RuntimeError(f"selected-checkpoint SHA collision across scientific states: {selected_expected}")
        selected_hashes.add(selected_expected)

        for field in spec.get("required_executor_files", []):
            file_required(field, f"executor dependency for {experiment_id}")
        for field in spec.get("required_executor_directories", []):
            directory_required(field, f"executor dependency for {experiment_id}")

        rows[experiment_id] = {
            "role": expected_role,
            "run_id": record.get("run_id"),
            "scientific_source_git_commit": record.get("source_git_commit"),
            "selected_checkpoint_sha256": selected_expected,
            "selected_checkpoint_file_sha256": sha256_file(checkpoint_path),
            "selected_checkpoint_epoch": int(checkpoint_payload["epoch"]),
            "run_record_sha256": sha256_file(run_record_path),
            "analysis_account_id": account_id,
            "analysis_slot_id": slot_id,
            "recovered_terminal_metadata": experiment_id in NEW_STATES,
        }

    if analysis_accounts != {"K1", "K2", "K3"}:
        raise RuntimeError("frozen analysis placement must use all three Kaggle accounts")

    return {
        "schema_version": "1.0",
        "status": "PASS",
        "gate_kind": "track_a_v12_posttraining_readiness",
        "analysis_source_git_commit": analysis_source,
        "training_science_source_git_commit": SCIENCE_SOURCE_SHA,
        "inventory_sha256": sha256_json(inventory),
        "manifest_sha256": MANIFEST_SHA256,
        "class_map_sha256": CLASS_MAP_SHA256,
        "state_count": len(rows),
        "direct_state_count": sum(row["role"] == "direct" for row in rows.values()),
        "auxiliary_state_count": sum(row["role"] == "auxiliary" for row in rows.values()),
        "new_recovered_state_count": sum(row["recovered_terminal_metadata"] for row in rows.values()),
        "historical_state_count": sum(not row["recovered_terminal_metadata"] for row in rows.values()),
        "unique_selected_checkpoint_count": len(selected_hashes),
        "states": rows,
        "v1_test_closed": True,
        "external_predictions_closed": True,
        "track_c_candidate_runtime_closed": True,
        "training_or_adaptation_authorized": False,
        "model_pool_change_authorized": False,
        "seed_change_authorized": False,
        "selector_change_authorized": False,
        "ready_for_posttraining_evidence": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--analysis-source-git-commit", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    observed = git_head(repo)
    if observed != args.analysis_source_git_commit:
        raise SystemExit(f"exact checkout mismatch: expected={args.analysis_source_git_commit}, observed={observed}")
    inventory_path = Path(args.inventory).resolve()
    inventory = load_json(inventory_path)
    gate = validate_inventory(inventory, repo, observed)
    gate["inventory_file_sha256"] = sha256_file(inventory_path)
    gate["gate_sha256"] = sha256_json(gate)
    atomic_write_json(args.output, gate)
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
