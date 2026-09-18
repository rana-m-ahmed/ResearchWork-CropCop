from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file, sha256_json
from cropcop_je.tracka_v12 import CLASS_MAP_SHA256, MANIFEST_SHA256
from cropcop_je.tracka_v12_recovery import SCIENCE_SOURCE_SHA

ACCOUNTS = {"K1", "K2", "K3"}


def load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def verify_gate_hash(payload: dict) -> bool:
    observed = payload.get("gate_sha256")
    if not isinstance(observed, str) or len(observed) != 64:
        return False
    clean = {key: value for key, value in payload.items() if key != "gate_sha256"}
    return observed == sha256_json(clean)


def validate_global_readiness(*, authority: dict, account_gates: list[tuple[Path, dict]], analysis_source: str) -> dict:
    if authority.get("status") != "PRE_EXECUTION_LOCK" or authority.get("scientific_semantics_change_authorized") is not False:
        raise RuntimeError("post-training closure authority is not the frozen pre-execution lock")
    auth_states = authority.get("state_inventory") or {}
    if len(auth_states) != 21:
        raise RuntimeError("closure authority must contain exactly 21 states")

    account_ids = {gate.get("account_id") for _, gate in account_gates}
    if account_ids != ACCOUNTS or len(account_gates) != 3:
        raise RuntimeError(f"global readiness requires exact K1/K2/K3 account gates, got {sorted(account_ids)}")

    authority_sha = sha256_json(authority)
    union: dict[str, dict] = {}
    account_hashes = {}
    for path, gate in account_gates:
        account_id = str(gate.get("account_id", ""))
        if gate.get("status") != "PASS" or gate.get("gate_kind") != "track_a_v12_posttraining_account_readiness":
            raise RuntimeError(f"account readiness is not PASS: {account_id}")
        if not verify_gate_hash(gate):
            raise RuntimeError(f"account readiness self-hash mismatch: {account_id}")
        if gate.get("analysis_source_git_commit") != analysis_source:
            raise RuntimeError(f"account analysis-source mismatch: {account_id}")
        if gate.get("training_science_source_git_commit") != SCIENCE_SOURCE_SHA:
            raise RuntimeError(f"account training-source mismatch: {account_id}")
        if gate.get("closure_authority_sha256") != authority_sha:
            raise RuntimeError(f"account closure-authority mismatch: {account_id}")
        if gate.get("manifest_sha256") != MANIFEST_SHA256 or gate.get("class_map_sha256") != CLASS_MAP_SHA256:
            raise RuntimeError(f"account frozen dataset identity mismatch: {account_id}")
        if gate.get("v1_test_closed") is not True or gate.get("external_predictions_closed") is not True or gate.get("track_c_candidate_runtime_closed") is not True:
            raise RuntimeError(f"account protected-surface gate is open: {account_id}")
        if gate.get("training_or_adaptation_performed") is not False or gate.get("optimizer_state_advanced") is not False:
            raise RuntimeError(f"account post-training readiness advanced scientific state: {account_id}")
        if gate.get("private_material_verified_locally") is not True:
            raise RuntimeError(f"account did not locally verify private material: {account_id}")
        if gate.get("evidence_targets_ready") is not True:
            raise RuntimeError(f"account private evidence targets are not ready: {account_id}")
        for experiment_id, row in (gate.get("states") or {}).items():
            if experiment_id in union:
                raise RuntimeError(f"state appears in more than one account readiness gate: {experiment_id}")
            union[experiment_id] = row
        account_hashes[account_id] = {
            "file_sha256": sha256_file(path),
            "gate_sha256": gate["gate_sha256"],
            "state_count": int(gate.get("state_count", 0)),
        }

    if set(union) != set(auth_states):
        missing = sorted(set(auth_states) - set(union))
        extra = sorted(set(union) - set(auth_states))
        raise RuntimeError(f"account readiness union must equal exact 21-state authority; missing={missing}, extra={extra}")

    selected = set()
    direct = 0
    auxiliary = 0
    recovered = 0
    historical = 0
    for experiment_id, row in sorted(union.items()):
        auth = auth_states[experiment_id]
        if row.get("role") != auth.get("role"):
            raise RuntimeError(f"role mismatch in global readiness: {experiment_id}")
        if row.get("selected_checkpoint_sha256") != auth.get("selected_checkpoint_sha256"):
            raise RuntimeError(f"selected-checkpoint mismatch in global readiness: {experiment_id}")
        if row.get("private_selected_checkpoint_verified") is not True:
            raise RuntimeError(f"selected checkpoint was not privately verified: {experiment_id}")
        sha = row["selected_checkpoint_sha256"]
        if sha in selected:
            raise RuntimeError(f"selected-checkpoint collision across Track-A states: {sha}")
        selected.add(sha)
        direct += int(row["role"] == "direct")
        auxiliary += int(row["role"] == "auxiliary")
        expected_recovered = auth.get("terminal_metadata_recovery_required") is True
        if bool(row.get("recovered_terminal_metadata")) != expected_recovered:
            raise RuntimeError(f"terminal-metadata recovery role mismatch: {experiment_id}")
        recovered += int(expected_recovered)
        historical += int(not expected_recovered)

    if (direct, auxiliary, recovered, historical, len(selected)) != (12, 9, 11, 10, 21):
        raise RuntimeError(
            "global Track-A readiness counts mismatch: "
            f"direct={direct}, auxiliary={auxiliary}, recovered={recovered}, historical={historical}, selected={len(selected)}"
        )

    result = {
        "schema_version": "1.0",
        "status": "PASS",
        "gate_kind": "track_a_v12_posttraining_global_readiness",
        "analysis_source_git_commit": analysis_source,
        "training_science_source_git_commit": SCIENCE_SOURCE_SHA,
        "closure_authority_sha256": authority_sha,
        "account_readiness": account_hashes,
        "state_count": 21,
        "direct_state_count": 12,
        "auxiliary_state_count": 9,
        "new_recovered_state_count": 11,
        "historical_state_count": 10,
        "unique_selected_checkpoint_count": 21,
        "states": union,
        "v1_test_closed": True,
        "external_predictions_closed": True,
        "track_c_candidate_runtime_closed": True,
        "training_or_adaptation_authorized": False,
        "model_pool_change_authorized": False,
        "seed_change_authorized": False,
        "selector_change_authorized": False,
        "all_private_evidence_targets_ready": True,
        "ready_for_posttraining_evidence": True,
    }
    result["gate_sha256"] = sha256_json(result)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--authority", default="journal_extension/locks/track_a_posttraining_closure_authority_v1.json")
    ap.add_argument("--account-readiness", action="append", required=True)
    ap.add_argument("--analysis-source-git-commit", required=True)
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
    gates = []
    for raw in args.account_readiness:
        path = Path(raw).resolve()
        gates.append((path, load_json(path)))
    result = validate_global_readiness(authority=authority, account_gates=gates, analysis_source=observed)
    result["authority_file_sha256"] = sha256_file(authority_path)
    result["gate_sha256"] = sha256_json({key: value for key, value in result.items() if key != "gate_sha256"})
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
