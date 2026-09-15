from __future__ import annotations

from pathlib import Path

from tracka_v12_kaggle_operator_v3 import SCIENCE_SHA, OperatorError, load_json, sha256_file

CODE_ATTESTATION_FILE_SHA256 = "29075babe0f898165c47140436239199f48909585d0caddaf9e2f06c129806fa"
LOCK_RUNTIME_ATTESTATION_FILE_SHA256 = "ed5c76d062b8f23703e0e02d1984bc6daeab811d94819c6a1804697837a4f138"


def verified_attestation_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parent / "attestations"
    code = root / "tracka-v12-pre-science-code-attestation.json"
    lock = root / "tracka-v12-exact-head-lock-runtime-attestation.json"
    if sha256_file(code) != CODE_ATTESTATION_FILE_SHA256:
        raise OperatorError("packaged code-attestation bytes changed")
    if sha256_file(lock) != LOCK_RUNTIME_ATTESTATION_FILE_SHA256:
        raise OperatorError("packaged lock/runtime-attestation bytes changed")
    code_payload = load_json(code)
    lock_payload = load_json(lock)
    if code_payload.get("status") != "PASS" or code_payload.get("github_actions") is not True:
        raise OperatorError("packaged code attestation is not a GitHub Actions PASS")
    if code_payload.get("pull_request_head_sha") != SCIENCE_SHA or code_payload.get("source_git_commit") != SCIENCE_SHA:
        raise OperatorError("packaged code attestation source mismatch")
    if lock_payload.get("status") != "PASS" or lock_payload.get("github_actions") is not True:
        raise OperatorError("packaged lock/runtime attestation is not a GitHub Actions PASS")
    if lock_payload.get("pull_request_head_sha") != SCIENCE_SHA or lock_payload.get("source_git_commit") != SCIENCE_SHA:
        raise OperatorError("packaged lock/runtime attestation source mismatch")
    if lock_payload.get("science_authorized") is not False:
        raise OperatorError("packaged lock/runtime attestation unexpectedly authorizes science")
    return code, lock
