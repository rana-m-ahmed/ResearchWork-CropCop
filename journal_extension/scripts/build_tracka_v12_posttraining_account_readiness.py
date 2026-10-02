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
from cropcop_je.tracka_v12_historical import validate_historical_closure_identity
from cropcop_je.tracka_v12_recovery import SCIENCE_SOURCE_SHA, validate_recovered_terminal_record
from cropcop_je.tracka_v12_runtime import load_and_validate_g1a_bundle
from cropcop_je.tracka_v12_posttraining_operator import validate_state_operator_spec
from cropcop_je.train import _identity as checkpoint_identity

ACCOUNTS = {"K1", "K2", "K3"}


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


def validate_account_inventory(*, inventory: dict, authority: dict, repo: Path, analysis_source: str) -> dict:
    if inventory.get("schema_version") != "1.0":
        raise RuntimeError("account post-training inventory schema must be 1.0")
    inventory_hash = inventory.get("inventory_sha256")
    inventory_clean = dict(inventory)
    inventory_clean.pop("inventory_sha256", None)
    if inventory_hash != sha256_json(inventory_clean):
        raise RuntimeError("account post-training inventory self-hash mismatch")
    account_id = str(inventory.get("account_id", ""))
    if account_id not in ACCOUNTS:
        raise RuntimeError(f"invalid account_id: {account_id!r}")
    if inventory.get("analysis_source_git_commit") != analysis_source:
        raise RuntimeError("account inventory analysis-source SHA differs from current exact checkout")
    if inventory.get("training_science_source_git_commit") != SCIENCE_SOURCE_SHA:
        raise RuntimeError("account inventory training science source mismatch")

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

    authority_states = authority.get("state_inventory") or {}
    states = inventory.get("states") or {}
    if not states:
        raise RuntimeError(f"account {account_id} has no assigned Track-A post-training states")
    unknown = set(states) - set(authority_states)
    if unknown:
        raise RuntimeError(f"account inventory contains states outside frozen closure authority: {sorted(unknown)}")

    rows = {}
    selected_hashes = set()
    for experiment_id in sorted(states):
        spec = states[experiment_id]
        auth = authority_states[experiment_id]
        operator_errors = validate_state_operator_spec(experiment_id, spec, check_paths=True)
        if operator_errors:
            raise RuntimeError(
                f"post-training operator contract invalid for {experiment_id}: "
                + "; ".join(operator_errors)
            )
        if spec.get("role") != auth.get("role"):
            raise RuntimeError(f"role mismatch for {experiment_id}")
        if auth.get("terminal_metadata_recovery_required") is True and auth.get("terminal_account_id") != account_id:
            raise RuntimeError(
                f"continuation state {experiment_id} must remain on its terminal private-data account "
                f"{auth.get('terminal_account_id')}, not {account_id}"
            )

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

        recovered = auth.get("terminal_metadata_recovery_required") is True
        certificate_sha = None
        if recovered:
            recovery_errors = validate_recovered_terminal_record(record)
            if recovery_errors:
                raise RuntimeError(f"recovered run record invalid for {experiment_id}: {recovery_errors}")
            certificate_path = file_required(spec["recovery_certificate"], f"recovery certificate for {experiment_id}")
            certificate = load_json(certificate_path)
            if certificate.get("status") != "PASS" or certificate.get("experiment_id") != experiment_id:
                raise RuntimeError(f"recovery certificate invalid for {experiment_id}")
            if certificate.get("recovered_run_record_sha256") != sha256_file(run_record_path):
                raise RuntimeError(f"recovery certificate/run-record hash mismatch for {experiment_id}")
            if certificate.get("selected_checkpoint_sha256") != auth.get("selected_checkpoint_sha256"):
                raise RuntimeError(f"recovery certificate selected-checkpoint mismatch for {experiment_id}")
            certificate_sha = sha256_file(certificate_path)
        else:
            historical_errors = validate_historical_closure_identity(record)
            if historical_errors:
                raise RuntimeError(f"historical run record invalid for {experiment_id}: {historical_errors}")

        selected_expected = ((record.get("artifact_locators") or {}).get("selected_checkpoint") or {}).get("sha256")
        if not selected_expected:
            selected_expected = (record.get("result_summary") or {}).get("selected_checkpoint_sha256")
        if selected_expected != auth.get("selected_checkpoint_sha256"):
            raise RuntimeError(f"run record selected-checkpoint differs from closure authority for {experiment_id}")

        checkpoint_root = directory_required(spec["checkpoint_root"], f"checkpoint root for {experiment_id}")
        checkpoint_path, checkpoint_payload = verify_selected(
            checkpoint_root,
            expected_identity=checkpoint_identity(record),
            expected_sha256=str(selected_expected),
        )
        if selected_expected in selected_hashes:
            raise RuntimeError(f"selected-checkpoint SHA collision within account inventory: {selected_expected}")
        selected_hashes.add(selected_expected)

        for path in spec.get("required_executor_files", []):
            file_required(path, f"executor dependency for {experiment_id}")
        for path in spec.get("required_executor_directories", []):
            directory_required(path, f"executor dependency for {experiment_id}")

        evidence_locator = str(spec.get("evidence_dataset_locator", "")).strip()
        if not evidence_locator:
            raise RuntimeError(f"evidence dataset locator missing for {experiment_id}")

        rows[experiment_id] = {
            "role": auth["role"],
            "run_id": record.get("run_id"),
            "scientific_source_git_commit": record.get("source_git_commit"),
            "selected_checkpoint_sha256": selected_expected,
            "selected_checkpoint_file_sha256": sha256_file(checkpoint_path),
            "selected_checkpoint_epoch": int(checkpoint_payload["epoch"]),
            "run_record_sha256": sha256_file(run_record_path),
            "recovery_certificate_sha256": certificate_sha,
            "recovered_terminal_metadata": recovered,
            "private_selected_checkpoint_verified": True,
            "evidence_dataset_locator": evidence_locator,
        }

    result = {
        "schema_version": "1.0",
        "status": "PASS",
        "gate_kind": "track_a_v12_posttraining_account_readiness",
        "account_id": account_id,
        "analysis_source_git_commit": analysis_source,
        "training_science_source_git_commit": SCIENCE_SOURCE_SHA,
        "closure_authority_sha256": sha256_json(authority),
        "inventory_sha256": sha256_json(inventory),
        "placement_freeze_sha256": inventory.get("placement_freeze_sha256"),
        "campaign_lock_sha256": inventory.get("campaign_lock_sha256"),
        "manifest_sha256": MANIFEST_SHA256,
        "class_map_sha256": CLASS_MAP_SHA256,
        "state_count": len(rows),
        "states": rows,
        "v1_test_closed": True,
        "external_predictions_closed": True,
        "track_c_candidate_runtime_closed": True,
        "training_or_adaptation_performed": False,
        "optimizer_state_advanced": False,
        "private_material_verified_locally": True,
    }
    result["gate_sha256"] = sha256_json(result)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--authority", default="journal_extension/locks/track_a_posttraining_closure_authority_v1.json")
    ap.add_argument("--analysis-source-git-commit", required=True)
    ap.add_argument("--evidence-target-preflight", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    observed = git_head(repo)
    if observed != args.analysis_source_git_commit:
        raise SystemExit(f"exact checkout mismatch: expected={args.analysis_source_git_commit}, observed={observed}")
    authority_path = Path(args.authority)
    if not authority_path.is_absolute():
        authority_path = repo / authority_path
    authority = load_json(authority_path)
    inventory_path = Path(args.inventory).resolve()
    inventory = load_json(inventory_path)
    target_path = Path(args.evidence_target_preflight).resolve()
    targets = load_json(target_path)
    if (
        targets.get("status") != "PASS"
        or targets.get("gate_kind") != "track_a_posttraining_private_evidence_targets"
        or targets.get("account_id") != inventory.get("account_id")
        or targets.get("analysis_source_git_commit") != observed
        or targets.get("account_inventory_sha256") != sha256_file(inventory_path)
        or targets.get("all_targets_private") is not True
        or targets.get("all_targets_owner_bound") is not True
    ):
        raise SystemExit("private evidence target preflight does not authorize account readiness")
    target_states = targets.get("states") or {}
    inventory_states = inventory.get("states") or {}
    if set(target_states) != set(inventory_states):
        raise SystemExit("private evidence target preflight state inventory mismatch")
    for experiment_id, spec in inventory_states.items():
        row = target_states[experiment_id]
        if row.get("dataset_slug") != spec.get("evidence_dataset_locator"):
            raise SystemExit(f"private evidence target locator mismatch: {experiment_id}")
        if row.get("authoritative_is_private") is not True:
            raise SystemExit(f"private evidence target is not authoritatively private: {experiment_id}")

    gate = validate_account_inventory(inventory=inventory, authority=authority, repo=repo, analysis_source=observed)
    gate["inventory_file_sha256"] = sha256_file(inventory_path)
    gate["authority_file_sha256"] = sha256_file(authority_path)
    gate["evidence_target_preflight_sha256"] = sha256_file(target_path)
    gate["evidence_targets_ready"] = True
    gate["gate_sha256"] = sha256_json({key: value for key, value in gate.items() if key != "gate_sha256"})
    atomic_write_json(args.output, gate)
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
