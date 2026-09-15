from __future__ import annotations

from pathlib import Path

from tracka_v12_kaggle_operator_v8 import SCIENCE_SHA, OperatorError, load_json, sha256_file

CODE_ATTESTATION_FILE_SHA256 = "e3c38aa2de9db070200732e8b67463956476413ee2899c2db462e6bca02a3d7e"
LOCK_RUNTIME_ATTESTATION_FILE_SHA256 = "1bc81be25b38cd7bf8490858c7d4931b811e993702210d2f856b56a96d7decbe"

def verified_attestation_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parent / "attestations_v8"
    code = root / "tracka-v12-pre-science-code-attestation.json"
    lock = root / "tracka-v12-exact-head-lock-runtime-attestation.json"
    if sha256_file(code) != CODE_ATTESTATION_FILE_SHA256:
        raise OperatorError("v8 packaged code-attestation bytes changed")
    if sha256_file(lock) != LOCK_RUNTIME_ATTESTATION_FILE_SHA256:
        raise OperatorError("v8 packaged lock/runtime-attestation bytes changed")
    cp = load_json(code); lp = load_json(lock)
    for label, payload in (("code", cp), ("lock", lp)):
        if payload.get("status") != "PASS" or payload.get("github_actions") is not True:
            raise OperatorError(f"v8 packaged {label} attestation is not a GitHub Actions PASS")
        if payload.get("source_git_commit") != SCIENCE_SHA or payload.get("pull_request_head_sha") != SCIENCE_SHA:
            raise OperatorError(f"v8 packaged {label} attestation source mismatch")
    gates = cp.get("static_pre_science_gates") or {}
    if gates.get("kaggle_generation_durability_contract") != "PASS" or gates.get("g1a_runtime_global_resolution_contract") != "PASS":
        raise OperatorError("v8 source mitigation gates are absent from code attestation")
    if lp.get("science_authorized") is not False:
        raise OperatorError("v8 lock/runtime attestation unexpectedly authorizes science")
    return code, lock
