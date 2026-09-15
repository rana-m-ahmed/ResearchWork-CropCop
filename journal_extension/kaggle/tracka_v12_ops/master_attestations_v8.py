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

CODE_ATTESTATION_FILE_SHA256 = "d2f2aa7f3f830829986071473b636079281583899152cbddc4e2562c7370e896"
LOCK_RUNTIME_ATTESTATION_FILE_SHA256 = "3f69d7606ab13e1539e98234789fd847d822196adbd62f651dc42aced3f2a863"


def verified_attestation_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parent / "attestations_v8"
    code = root / "tracka-v12-pre-science-code-attestation.json"
    lock = root / "tracka-v12-exact-head-lock-runtime-attestation.json"
    if sha256_file(code) != CODE_ATTESTATION_FILE_SHA256:
        raise OperatorError("v8r2 packaged code-attestation bytes changed")
    if sha256_file(lock) != LOCK_RUNTIME_ATTESTATION_FILE_SHA256:
        raise OperatorError("v8r2 packaged lock/runtime-attestation bytes changed")

    cp = load_json(code)
    lp = load_json(lock)
    for label, payload in (("code", cp), ("lock", lp)):
        if payload.get("status") != "PASS" or payload.get("github_actions") is not True:
            raise OperatorError(f"v8r2 packaged {label} attestation is not a GitHub Actions PASS")
        if payload.get("source_git_commit") != SCIENCE_SHA or payload.get("pull_request_head_sha") != SCIENCE_SHA:
            raise OperatorError(f"v8r2 packaged {label} attestation source mismatch")
        if payload.get("r13_parity_contract_id") != R13_PARITY_CONTRACT_ID_V8:
            raise OperatorError(f"v8r2 packaged {label} attestation parity-contract mismatch")
        if float(payload.get("r13_parity_required_max_abs_difference", -1.0)) != R13_PARITY_REQUIRED_MAX_ABS_V8:
            raise OperatorError(f"v8r2 packaged {label} attestation parity tolerance mismatch")

    gates = cp.get("static_pre_science_gates") or {}
    required_gates = (
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
    )
    missing = [name for name in required_gates if gates.get(name) != "PASS"]
    if missing:
        raise OperatorError("v8r2 source mitigation gates are absent or non-PASS: " + ", ".join(missing))

    if lp.get("science_authorized") is not False:
        raise OperatorError("v8r2 lock/runtime attestation unexpectedly authorizes science")
    if lp.get("r13_exact_pretrained_parity") != "PASS" or lp.get("r13_v121_parity_amendment") != "PASS":
        raise OperatorError("v8r2 exact-pretrained/parity-amendment proof is not PASS")
    exact = lp.get("exact_pretrained_parity_report") or {}
    if exact.get("v1_2_1_gate_pass") is not True or exact.get("historical_v1_2_gate_pass") is not False:
        raise OperatorError("v8r2 parity attestation does not prove superseding gate behavior")
    if exact.get("scientific_metric_computed") is not False or exact.get("v1_test_accessed") is not False:
        raise OperatorError("v8r2 parity proof touched a protected/scientific surface")

    return code, lock
