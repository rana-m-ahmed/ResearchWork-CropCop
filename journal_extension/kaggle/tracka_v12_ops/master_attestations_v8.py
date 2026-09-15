from __future__ import annotations

from pathlib import Path

from tracka_v12_kaggle_operator_v8 import (
    SCIENCE_SHA,
    R13_PARITY_CONTRACT_ID_V8,
    R13_PARITY_REQUIRED_MAX_ABS_V8,
    OperatorError,
    load_json,
    sha256_file,
)

RELEASE_MANIFEST_FILE_SHA256 = "22a675bf82f53cc6b6ce4996332d4c442a65fc9293200affc2750ad2446f93d4"
CODE_ATTESTATION_MEMBER_SHA256 = "d2f2aa7f3f830829986071473b636079281583899152cbddc4e2562c7370e896"
LOCK_RUNTIME_ATTESTATION_MEMBER_SHA256 = "3f69d7606ab13e1539e98234789fd847d822196adbd62f651dc42aced3f2a863"


def verified_release_attestation_manifest() -> tuple[Path, dict]:
    path = Path(__file__).resolve().parent / "attestations_v8r2" / "RELEASE_ATTESTATION_MANIFEST.json"
    if not path.is_file():
        raise OperatorError("v8r2 release attestation manifest is missing")
    if sha256_file(path) != RELEASE_MANIFEST_FILE_SHA256:
        raise OperatorError("v8r2 release attestation manifest bytes changed")

    payload = load_json(path)
    if payload.get("schema_version") != "1.0" or payload.get("status") != "LOCKED_PRE_SCIENCE":
        raise OperatorError("v8r2 release attestation manifest status/schema mismatch")
    if payload.get("science_source_sha") != SCIENCE_SHA:
        raise OperatorError("v8r2 release attestation manifest science source mismatch")
    if payload.get("science_authorized") is not False:
        raise OperatorError("v8r2 release attestation manifest unexpectedly authorizes science")
    if payload.get("protected_test_accessed") is not False or payload.get("external_surface_accessed") is not False:
        raise OperatorError("v8r2 release attestation manifest reports protected-surface access")
    if int(payload.get("remaining_scientific_states", -1)) != 11:
        raise OperatorError("v8r2 release attestation manifest does not bind exactly 11 remaining states")

    parity = payload.get("r13_parity_contract") or {}
    if parity.get("contract_id") != R13_PARITY_CONTRACT_ID_V8:
        raise OperatorError("v8r2 release attestation manifest parity-contract mismatch")
    if float(parity.get("required_max_abs_difference", -1.0)) != R13_PARITY_REQUIRED_MAX_ABS_V8:
        raise OperatorError("v8r2 release attestation manifest parity tolerance mismatch")
    if float(parity.get("historical_v1_2_required_max_abs_difference", -1.0)) != 1e-5:
        raise OperatorError("v8r2 release attestation manifest lost historical parity provenance")
    if parity.get("exact_pretrained_gate_pass") is not True:
        raise OperatorError("v8r2 release attestation manifest lacks exact-pretrained parity PASS")

    code = payload.get("code_attestation") or {}
    lock = payload.get("lock_runtime_attestation") or {}
    if code.get("status") != "PASS" or code.get("member_sha256") != CODE_ATTESTATION_MEMBER_SHA256:
        raise OperatorError("v8r2 code-attestation provenance mismatch")
    if lock.get("status") != "PASS" or lock.get("member_sha256") != LOCK_RUNTIME_ATTESTATION_MEMBER_SHA256:
        raise OperatorError("v8r2 lock/runtime-attestation provenance mismatch")
    if int(code.get("workflow_run_id", -1)) != 34960774118:
        raise OperatorError("v8r2 code-attestation workflow provenance mismatch")
    if int(lock.get("workflow_run_id", -1)) != 34960774095:
        raise OperatorError("v8r2 lock/runtime-attestation workflow provenance mismatch")

    required = set(payload.get("required_static_pre_science_gates") or [])
    expected = {
        "exact_head_ci",
        "kaggle_generation_durability_contract",
        "g1a_runtime_global_resolution_contract",
        "r13_parity_amendment_contract",
        "principal_science_diff",
        "secondary_science_diff",
        "checkpoint_recovery_contract",
        "cross_slot_recovery_contract",
        "six_gpu_parent_orchestration",
        "training_runner_contract",
    }
    if required != expected:
        raise OperatorError("v8r2 release attestation manifest required-gate set drifted")

    return path, payload


# Backward-compatible helper name retained for neutral inherited tests only.
def verified_attestation_paths() -> tuple[Path, Path]:
    path, _ = verified_release_attestation_manifest()
    return path, path
