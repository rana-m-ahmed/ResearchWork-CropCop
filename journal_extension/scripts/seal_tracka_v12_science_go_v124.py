from __future__ import annotations

"""Durability-bound SCIENCE_GO sealer for the R13 parity v1.2.1 contract.

Historical SCIENCE_GO implementations remain unchanged. This wrapper accepts
only the exact-head v1.1 attestation schema emitted by the superseding parity
amendment and rebinds G1A validation to the v1.2.1 seal contract for this call.
"""

from typing import Any

import _bootstrap  # noqa: F401
import seal_tracka_v12_science_go as legacy
import seal_tracka_v12_science_go_v123 as base
from cropcop_je.tracka_v12_g1a_v121 import validate_g1a_seal_object as validate_g1a_seal_object_v121


def _validate_ci_attestation_provenance_v11(
    payload: dict[str, Any],
    *,
    expected_kind: str,
    source_git_commit: str,
    label: str,
) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != "1.1":
        errors.append(f"{label} schema version mismatch")
    if payload.get("attestation_kind") != expected_kind:
        errors.append(f"{label} kind mismatch")
    if payload.get("github_actions") is not True:
        errors.append(f"{label} is not marked as GitHub Actions emitted")
    if payload.get("pull_request_head_sha") != source_git_commit:
        errors.append(f"{label} PR-head SHA mismatch")
    if payload.get("source_git_commit") != source_git_commit:
        errors.append(f"{label} source SHA mismatch")
    if not legacy._valid_actions_run_id(payload.get("workflow_run_id")):
        errors.append(f"{label} workflow run ID is missing/invalid")
    if not legacy._valid_actions_run_id(payload.get("workflow_run_attempt")):
        errors.append(f"{label} workflow run attempt is missing/invalid")
    return errors


def main() -> int:
    previous_validator = legacy.validate_g1a_seal_object
    previous_provenance = legacy._validate_ci_attestation_provenance
    legacy.validate_g1a_seal_object = validate_g1a_seal_object_v121
    legacy._validate_ci_attestation_provenance = _validate_ci_attestation_provenance_v11
    try:
        return base.main()
    finally:
        legacy.validate_g1a_seal_object = previous_validator
        legacy._validate_ci_attestation_provenance = previous_provenance


if __name__ == "__main__":
    raise SystemExit(main())
