from __future__ import annotations

import os
import shutil
from pathlib import Path

from tracka_v12_kaggle_operator_v8 import (
    SCIENCE_SHA,
    R13_PARITY_CONTRACT_ID_V8,
    R13_PARITY_REQUIRED_MAX_ABS_V8,
    OperatorError,
    load_json,
    sha256_file,
)

RELEASE_MANIFEST_FILE_SHA256 = "0777ce9f040ada279dfe31b85ce8a0498926a10c78ef3b7aa6297c64ea9d0c43"
CODE_ATTESTATION_MEMBER_SHA256 = "4f29548f5ba56c2969fb1b0c7234d66ce15a2747d104e8e27579af08f9be7010"
LOCK_RUNTIME_ATTESTATION_MEMBER_SHA256 = "d2a1f4303ce73f1f92f2a3835b6300230ab566686da0f3a2e58f743bf1366c4d"
ATTESTATION_DIR = Path(__file__).resolve().parent / "attestations_v9"


def verified_release_attestation_manifest() -> tuple[Path, dict]:
    path = ATTESTATION_DIR / "RELEASE_ATTESTATION_MANIFEST.json"
    if not path.is_file():
        raise OperatorError("v9 release attestation manifest is missing")
    if sha256_file(path) != RELEASE_MANIFEST_FILE_SHA256:
        raise OperatorError("v9 release attestation manifest bytes changed")

    payload = load_json(path)
    if payload.get("schema_version") != "1.1" or payload.get("status") != "LOCKED_PRE_SCIENCE":
        raise OperatorError("v9 release attestation manifest status/schema mismatch")
    if payload.get("science_source_sha") != SCIENCE_SHA:
        raise OperatorError("v9 release attestation manifest science source mismatch")
    if payload.get("science_authorized") is not False:
        raise OperatorError("v9 release attestation manifest unexpectedly authorizes science")
    if payload.get("protected_test_accessed") is not False or payload.get("external_surface_accessed") is not False:
        raise OperatorError("v9 release attestation manifest reports protected-surface access")
    if int(payload.get("remaining_scientific_states", -1)) != 11:
        raise OperatorError("v9 release attestation manifest does not bind exactly 11 remaining states")

    materialization = payload.get("attestation_materialization") or {}
    if materialization.get("mode") != "packaged_exact_bytes":
        raise OperatorError("v9 attestation materialization mode mismatch")
    if materialization.get("runtime_requires_github_actions_api") is not False:
        raise OperatorError("v8r3 unexpectedly requires GitHub Actions API access at runtime")
    if materialization.get("runtime_requires_actions_read_permission") is not False:
        raise OperatorError("v8r3 unexpectedly requires Actions-read permission at runtime")

    parity = payload.get("r13_parity_contract") or {}
    if parity.get("contract_id") != R13_PARITY_CONTRACT_ID_V8:
        raise OperatorError("v9 release attestation manifest parity-contract mismatch")
    if float(parity.get("required_max_abs_difference", -1.0)) != R13_PARITY_REQUIRED_MAX_ABS_V8:
        raise OperatorError("v9 release attestation manifest parity tolerance mismatch")
    if float(parity.get("historical_v1_2_required_max_abs_difference", -1.0)) != 1e-5:
        raise OperatorError("v9 release attestation manifest lost historical parity provenance")
    if parity.get("exact_pretrained_gate_pass") is not True:
        raise OperatorError("v9 release attestation manifest lacks exact-pretrained parity PASS")

    code = payload.get("code_attestation") or {}
    lock = payload.get("lock_runtime_attestation") or {}
    if code.get("status") != "PASS" or code.get("member_sha256") != CODE_ATTESTATION_MEMBER_SHA256:
        raise OperatorError("v9 code-attestation provenance mismatch")
    if lock.get("status") != "PASS" or lock.get("member_sha256") != LOCK_RUNTIME_ATTESTATION_MEMBER_SHA256:
        raise OperatorError("v9 lock/runtime-attestation provenance mismatch")

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
        raise OperatorError("v9 release attestation manifest required-gate set drifted")
    return path, payload


def _validate_attestation_payload(payload: dict, *, kind: str, required_gates: set[str]) -> None:
    if payload.get("status") != "PASS" or payload.get("github_actions") is not True:
        raise OperatorError(f"packaged {kind} attestation is not a GitHub Actions PASS")
    if payload.get("source_git_commit") != SCIENCE_SHA or payload.get("pull_request_head_sha") != SCIENCE_SHA:
        raise OperatorError(f"packaged {kind} attestation source mismatch")
    if payload.get("r13_parity_contract_id") != R13_PARITY_CONTRACT_ID_V8:
        raise OperatorError(f"packaged {kind} attestation parity-contract mismatch")
    if float(payload.get("r13_parity_required_max_abs_difference", -1.0)) != R13_PARITY_REQUIRED_MAX_ABS_V8:
        raise OperatorError(f"packaged {kind} attestation parity tolerance mismatch")

    if kind == "code":
        gates = payload.get("static_pre_science_gates") or {}
        missing = [name for name in sorted(required_gates) if gates.get(name) != "PASS"]
        if missing:
            raise OperatorError("packaged code attestation lacks required PASS gates: " + ", ".join(missing))
        implementation = payload.get("implementation_sha256") or {}
        for required_path in (
            "journal_extension/kaggle/run_tracka_v12_account_v121.py",
            "journal_extension/scripts/seal_tracka_v12_science_go_v124.py",
            "journal_extension/scripts/seal_tracka_v12_g1a_v121.py",
            "journal_extension/scripts/run_tracka_v12_training_v121.py",
        ):
            value = str(implementation.get(required_path, ""))
            if len(value) != 64:
                raise OperatorError(f"packaged code attestation does not bind {required_path}")
    else:
        if payload.get("science_authorized") is not False:
            raise OperatorError("packaged lock/runtime attestation unexpectedly authorizes science")
        if payload.get("r13_exact_pretrained_parity") != "PASS" or payload.get("r13_v121_parity_amendment") != "PASS":
            raise OperatorError("packaged lock/runtime attestation lacks parity qualification PASS")
        exact = payload.get("exact_pretrained_parity_report") or {}
        if exact.get("v1_2_1_gate_pass") is not True or exact.get("historical_v1_2_gate_pass") is not False:
            raise OperatorError("packaged lock/runtime attestation does not prove superseding parity behavior")
        if exact.get("scientific_metric_computed") is not False or exact.get("v1_test_accessed") is not False:
            raise OperatorError("packaged lock/runtime attestation touched a protected/scientific surface")


def verified_attestation_paths() -> tuple[Path, Path]:
    _manifest_path, manifest = verified_release_attestation_manifest()
    required_gates = set(manifest["required_static_pre_science_gates"])
    code = ATTESTATION_DIR / str(manifest["code_attestation"]["member"])
    lock = ATTESTATION_DIR / str(manifest["lock_runtime_attestation"]["member"])

    for kind, path, expected_sha in (
        ("code", code, CODE_ATTESTATION_MEMBER_SHA256),
        ("lock", lock, LOCK_RUNTIME_ATTESTATION_MEMBER_SHA256),
    ):
        if not path.is_file():
            raise OperatorError(f"packaged {kind} attestation is missing")
        observed = sha256_file(path)
        if observed != expected_sha:
            raise OperatorError(
                f"packaged {kind} attestation byte hash mismatch: expected {expected_sha}, got {observed}"
            )
        _validate_attestation_payload(load_json(path), kind=kind, required_gates=required_gates)

    return code, lock


def materialize_verified_attestations(destination: str | Path) -> tuple[Path, Path]:
    """Copy already-verified release evidence into the run workspace without network access."""
    code, lock = verified_attestation_paths()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for source in (code, lock):
        target = destination / source.name
        tmp = target.with_suffix(target.suffix + ".tmp")
        shutil.copyfile(source, tmp)
        os.replace(tmp, target)
        if sha256_file(target) != sha256_file(source):
            raise OperatorError(f"packaged attestation copy verification failed: {source.name}")
        outputs.append(target)

    return outputs[0], outputs[1]
