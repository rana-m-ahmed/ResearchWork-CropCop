from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


class ReleaseAuthorityError(RuntimeError):
    pass


def _load(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ReleaseAuthorityError(f"JSON object required: {path}")
    return obj


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _blob_sha(repo: Path, rel: str) -> str:
    cp = subprocess.run(
        ["git", "-C", str(repo), "hash-object", rel],
        check=True,
        capture_output=True,
        text=True,
    )
    return cp.stdout.strip()


def _require_hex(value: str, length: int, label: str) -> None:
    if len(value) != length or any(ch not in "0123456789abcdef" for ch in value.lower()):
        raise ReleaseAuthorityError(f"{label} must be {length}-character hexadecimal")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate the single Track-B v5 release authority.")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--require-executable", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    authority_path = repo / "journal_extension/track_b_r07/TRACKB_RELEASE_AUTHORITY_v1.json"
    source_lock_path = repo / "journal_extension/track_b_r07/TRACKB_EXTERNAL_SOURCE_LOCK_v2.json"
    execution_lock_path = repo / "journal_extension/track_b_r07/TRACKB_R07_EXECUTION_LOCK_v4.json"
    code_attestation_path = repo / "journal_extension/track_b_r07/TRACKB_CODE_ATTESTATION_v4.json"
    materialization_lock_path = repo / "journal_extension/track_b_r07/TRACKB_INPUT_MATERIALIZATION_LOCK_v2.json"
    requirements_lock_path = repo / "journal_extension/track_b_r07/requirements-trackb.lock.txt"
    orchestration_lock_path = repo / "journal_extension/track_b_r07/TRACKB_ORCHESTRATION_LOCK_v5.json"
    operator_attestation_path = repo / "journal_extension/track_b_r07/TRACKB_V5_OPERATOR_ATTESTATION_v1.json"

    authority = _load(authority_path)
    source_lock = _load(source_lock_path)
    execution_lock = _load(execution_lock_path)
    operator_attestation = _load(operator_attestation_path)
    orchestration_lock = _load(orchestration_lock_path)

    if authority.get("release_id") != "TRACKB_V5_RELEASE_AUTHORITY_v1":
        raise ReleaseAuthorityError("unexpected Track-B release authority identity")
    if authority.get("scientific_change") is not False:
        raise ReleaseAuthorityError("release authority must explicitly preserve frozen science")
    if source_lock.get("lock_id") != "TRACKB_EXTERNAL_SOURCE_LOCK_v2":
        raise ReleaseAuthorityError("unexpected external-source lock identity")
    if source_lock.get("protected_external_predictions_allowed") is not False:
        raise ReleaseAuthorityError("external-source lock must prohibit protected predictions")

    if args.require_executable:
        if authority.get("status") != "SEALED_PRE_RESULTS_RELEASE" or authority.get("executable") is not True:
            raise ReleaseAuthorityError("Track-B release is not sealed/executable")
        source_candidates = source_lock.get("candidates") or {}
        blocked = [
            role for role, row in source_candidates.items()
            if not isinstance(row, dict) or row.get("materialization_permitted") is not True
        ]
        if blocked:
            raise ReleaseAuthorityError(f"external-source authority remains blocked: {blocked}")

        runtime = str(authority.get("runtime_source_commit", "")).lower()
        _require_hex(runtime, 40, "runtime_source_commit")

        bindings = authority.get("bindings") or {}
        sha_bindings = {
            "code_attestation_sha256",
            "execution_lock_sha256",
            "materialization_lock_sha256",
            "external_source_lock_sha256",
            "requirements_lock_sha256",
            "orchestration_lock_sha256",
            "operator_attestation_sha256",
        }
        for field in sha_bindings:
            _require_hex(str(bindings.get(field, "")).lower(), 64, field)
        for field in ("operator_notebook_00_blob_sha1", "operator_notebook_01_blob_sha1"):
            _require_hex(str(bindings.get(field, "")).lower(), 40, field)

        expected_sha = {
            "code_attestation_sha256": _sha256(code_attestation_path),
            "execution_lock_sha256": _sha256(execution_lock_path),
            "materialization_lock_sha256": _sha256(materialization_lock_path),
            "external_source_lock_sha256": _sha256(source_lock_path),
            "requirements_lock_sha256": _sha256(requirements_lock_path),
            "orchestration_lock_sha256": _sha256(orchestration_lock_path),
            "operator_attestation_sha256": _sha256(operator_attestation_path),
        }
        mismatches = {
            field: {"expected": expected_sha[field], "bound": str(bindings.get(field, ""))}
            for field in expected_sha
            if expected_sha[field] != str(bindings.get(field, ""))
        }
        if mismatches:
            raise ReleaseAuthorityError(
                f"release authority hash binding mismatch: {mismatches}"
            )

        if execution_lock.get("code_attestation_sha256") != bindings["code_attestation_sha256"]:
            raise ReleaseAuthorityError(
                "execution lock does not bind the same code attestation as the release"
            )
        if orchestration_lock.get("runtime_source_commit") != runtime:
            raise ReleaseAuthorityError(
                "orchestration lock runtime source differs from release authority"
            )
        if operator_attestation.get("runtime_source_commit") != runtime:
            raise ReleaseAuthorityError(
                "operator attestation runtime source differs from release authority"
            )
        if operator_attestation.get("release_id") != authority.get("release_id"):
            raise ReleaseAuthorityError(
                "operator attestation release identity mismatch"
            )
        if operator_attestation.get("status") != "FROZEN_PRE_RESULTS_OPERATOR_SURFACE":
            raise ReleaseAuthorityError("operator attestation is not frozen pre-results")
        if operator_attestation.get("scientific_change") is not False:
            raise ReleaseAuthorityError("operator attestation must preserve frozen science")

        notebook_paths = authority.get("supported_operator_notebooks") or []
        if len(notebook_paths) != 2:
            raise ReleaseAuthorityError("release authority must bind exactly two operator notebooks")
        observed_blobs = [_blob_sha(repo, str(rel)) for rel in notebook_paths]
        expected_blobs = [
            bindings["operator_notebook_00_blob_sha1"],
            bindings["operator_notebook_01_blob_sha1"],
        ]
        if observed_blobs != expected_blobs:
            raise ReleaseAuthorityError(
                f"operator notebook blob drift: expected={expected_blobs}, observed={observed_blobs}"
            )

        operator_rows = operator_attestation.get("operator_files") or []
        operator_blobs = {
            str(row.get("path", "")): str(row.get("git_blob_sha1", ""))
            for row in operator_rows
            if isinstance(row, dict)
        }
        for rel, observed in zip(notebook_paths, observed_blobs):
            if operator_blobs.get(str(rel)) != observed:
                raise ReleaseAuthorityError(
                    f"operator attestation notebook binding mismatch: {rel}"
                )

    result = {
        "status": "PASS_RELEASE_AUTHORITY_STRUCTURE",
        "release_status": authority.get("status"),
        "executable": bool(authority.get("executable")),
        "external_source_status": source_lock.get("status"),
        "runtime_source_commit": authority.get("runtime_source_commit"),
        "operator_attestation_status": operator_attestation.get("status"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
