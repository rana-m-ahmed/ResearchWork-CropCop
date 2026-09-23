from __future__ import annotations

import os
from typing import Any

HOTFIX_ID = "TRACKB_V5_V1_GUARD_FALSE_POSITIVE_FIX_v1"
ACTIVE_ENV = "TRACKB_V1_GUARD_HOTFIX_ACTIVE"


class GuardHotfixError(RuntimeError):
    pass


def _path_is_forbidden(value: str) -> bool:
    normalized = "/" + str(value).replace("\\", "/").strip().strip("/").lower() + "/"
    if "/ds-v1-test-consumed/" in normalized or "/ds-v1-test/" in normalized:
        return True
    if "/v1_test/" in normalized or "/v1-test/" in normalized:
        return True
    if "/test_consumed/" in normalized:
        return True
    return False


def _iter_path_values(obj: Any):
    if isinstance(obj, dict):
        for child_key, value in obj.items():
            child_lower = str(child_key).lower()
            if isinstance(value, str) and (
                child_lower == "path"
                or child_lower in {
                    "data_root",
                    "repository_root",
                    "v1_validation_root",
                    "validation_root",
                    "image_root",
                    "source_root",
                }
                or child_lower.endswith(("_path", "_root", "_dir"))
            ):
                yield str(child_key), value
            else:
                yield from _iter_path_values(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_path_values(value)


def _iter_access_flags(obj: Any):
    if isinstance(obj, dict):
        for child_key, value in obj.items():
            child_lower = str(child_key).lower()
            if child_lower in {
                "v1_test_accessed",
                "v1_test_image_bytes_accessed",
                "consumed_v1_test_accessed",
            }:
                yield str(child_key), value
            yield from _iter_access_flags(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_access_flags(value)


def validate_manifest_safety(role: str, manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise GuardHotfixError(f"Track-B input manifest is not an object: role={role!r}")

    for key, value in _iter_access_flags(manifest):
        if value is not False:
            raise GuardHotfixError(
                f"consumed V1-test access flag is not explicitly false: role={role!r}, "
                f"field={key!r}, value={value!r}"
            )

    for key, value in _iter_path_values(manifest):
        if _path_is_forbidden(value):
            raise GuardHotfixError(
                f"forbidden consumed V1-test path referenced by input role {role!r}: "
                f"field={key!r}, value={value!r}"
            )

    if role == "historical_compare":
        if manifest.get("coverage_scope") != "V1_TRAIN_VAL_ONLY":
            raise GuardHotfixError(
                "historical_compare must remain on the V1_TRAIN_VAL_ONLY safe surface"
            )
        if manifest.get("v1_test_image_bytes_accessed") is not False:
            raise GuardHotfixError(
                "historical_compare must explicitly assert v1_test_image_bytes_accessed=false"
            )
        if manifest.get("ext_i_eligible") is not False:
            raise GuardHotfixError(
                "historical_compare must remain ineligible for EXT-I under the safe route"
            )
        if manifest.get("maximum_evidence_grade") != "EXT-S":
            raise GuardHotfixError(
                "historical_compare safe route must remain capped at EXT-S"
            )


def install(expected_source_sha256: str) -> dict[str, str]:
    expected_source_sha256 = str(expected_source_sha256).strip().lower()
    if len(expected_source_sha256) != 64 or any(
        ch not in "0123456789abcdef" for ch in expected_source_sha256
    ):
        raise GuardHotfixError("operator hotfix source SHA-256 is missing or malformed")

    from cropcop_je import trackb_r07

    def _strict_no_v1_test_surface(inputs):
        for bundle in inputs.values():
            validate_manifest_safety(str(bundle.role), bundle.manifest)

    trackb_r07.assert_no_v1_test_surface = _strict_no_v1_test_surface
    os.environ[ACTIVE_ENV] = expected_source_sha256
    return {
        "status": "PASS_TRACKB_V1_GUARD_HOTFIX_INSTALLED",
        "hotfix_id": HOTFIX_ID,
        "source_sha256": expected_source_sha256,
    }
