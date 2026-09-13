from __future__ import annotations

from pathlib import Path

from tracka_v12_kaggle_operator_v3 import SCIENCE_SHA, OperatorError, load_json, sha256_file

CODE_ATTESTATION_FILE_SHA256 = "a618bda23fb0be0e7e52f968c08bde47c135429d719c4f4cc7539d2e67abe03d"
LOCK_RUNTIME_ATTESTATION_FILE_SHA256 = "8e77489ecd159f396cb189b7de93529f99063b9d1ffe3183afd1602d274a4223"


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
