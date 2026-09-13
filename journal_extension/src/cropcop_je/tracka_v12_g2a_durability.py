from __future__ import annotations

from typing import Any

from .hashing import sha256_json
from .tracka_v12_g2a import REQUIRED_PROFILES

G2A_DURABILITY_SCHEMA_VERSION = "1.0"
G2A_DURABILITY_KIND = "track_a_v12_g2a_kaggle_private_durability"
G2A_DURABLE_STORE_KIND = "kaggle-dataset"
G2A_DURABLE_BACKEND = "kaggle_private_dataset"


class TrackAV12G2ADurabilityError(RuntimeError):
    pass


def durability_contract_hash(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("durability_contract_sha256", None)
    return sha256_json(clean)


def _valid_kaggle_locator(locator: Any) -> bool:
    text = str(locator or "").strip()
    parts = text.split("/")
    return len(parts) == 2 and all(part and part == part.strip() for part in parts)


def validate_calibration_durability(summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    calibration_id = str(summary.get("calibration_id", ""))
    if calibration_id not in REQUIRED_PROFILES:
        errors.append("G2A durability calibration ID is not frozen")
    if summary.get("status") != "PASS":
        errors.append("G2A durability summary is not terminal PASS")
    if summary.get("durable_roundtrip_success") is not True:
        errors.append("G2A durability roundtrip did not PASS")
    if summary.get("durable_store_kind") != G2A_DURABLE_STORE_KIND:
        errors.append("G2A v1.2.2 qualification must use kaggle-dataset durability")

    run_id = str(summary.get("run_id", "")).strip()
    if not run_id:
        errors.append("G2A durability run ID missing")

    durability = summary.get("durability") or {}
    if durability.get("backend") != G2A_DURABLE_BACKEND:
        errors.append("G2A durability backend is not kaggle_private_dataset")
    locator = durability.get("locator")
    if not _valid_kaggle_locator(locator):
        errors.append("G2A durability locator is not a valid owner/dataset slug")

    preflight = durability.get("preflight_private_access") or {}
    if preflight.get("status") != "PASS" or preflight.get("errors"):
        errors.append("G2A private-dataset durability preflight is not PASS")
    if preflight.get("kind") != G2A_DURABLE_STORE_KIND:
        errors.append("G2A durability preflight kind mismatch")
    if int(preflight.get("resolved_locator_count", 0) or 0) != 1:
        errors.append("G2A durability preflight must cover exactly one calibration locator")
    checks = preflight.get("checks") or {}
    row = checks.get(run_id) if run_id else None
    if not isinstance(row, dict):
        errors.append("G2A durability preflight does not bind the calibration run ID")
    else:
        if row.get("locator") != locator:
            errors.append("G2A durability preflight locator mismatch")
        for field in (
            "owner_matches_authenticated_user",
            "authenticated_read",
            "authoritative_is_private",
            "private_metadata_verified",
        ):
            if row.get(field) is not True:
                errors.append(f"G2A durability preflight missing {field}")
        if row.get("write_generation_mutated_by_preflight") is not False:
            errors.append("G2A durability preflight unexpectedly mutated durable history")
    return errors


def build_g2a_durability_contract(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(row.get("calibration_id", "")): row for row in summaries}
    errors: list[str] = []
    if set(by_id) != set(REQUIRED_PROFILES) or len(by_id) != len(summaries):
        errors.append("G2A durability contract requires each frozen profile exactly once")

    locators: dict[str, str] = {}
    input_hashes: dict[str, str] = {}
    for calibration_id in REQUIRED_PROFILES:
        row = by_id.get(calibration_id)
        if row is None:
            continue
        errors.extend(f"{calibration_id}: {error}" for error in validate_calibration_durability(row))
        locators[calibration_id] = str((row.get("durability") or {}).get("locator", ""))
        input_hashes[calibration_id] = sha256_json(row)

    if len(locators) == len(REQUIRED_PROFILES) and len(set(locators.values())) != len(REQUIRED_PROFILES):
        errors.append("G2A calibration profiles must use five distinct private Kaggle durable datasets")

    contract = {
        "schema_version": G2A_DURABILITY_SCHEMA_VERSION,
        "contract_kind": G2A_DURABILITY_KIND,
        "status": "PASS" if not errors else "FAIL",
        "science_authorized": False,
        "required_profiles": list(REQUIRED_PROFILES),
        "required_store_kind": G2A_DURABLE_STORE_KIND,
        "required_backend": G2A_DURABLE_BACKEND,
        "profile_locators": locators,
        "unique_locator_count": len(set(locators.values())),
        "input_summary_sha256": input_hashes,
        "errors": errors,
    }
    contract["durability_contract_sha256"] = durability_contract_hash(contract)
    return contract


def validate_g2a_durability_contract(
    contract: dict[str, Any],
    *,
    expected_input_summary_sha256: dict[str, str] | None = None,
) -> list[str]:
    errors: list[str] = []
    if contract.get("schema_version") != G2A_DURABILITY_SCHEMA_VERSION or contract.get("contract_kind") != G2A_DURABILITY_KIND:
        errors.append("unsupported G2A durability contract schema/kind")
    if contract.get("status") != "PASS" or contract.get("errors"):
        errors.append("G2A durability contract is not terminal PASS")
    if contract.get("science_authorized") is not False:
        errors.append("G2A durability contract may not authorize science")
    if contract.get("required_store_kind") != G2A_DURABLE_STORE_KIND or contract.get("required_backend") != G2A_DURABLE_BACKEND:
        errors.append("G2A durability contract backend requirement drift")
    if contract.get("durability_contract_sha256") != durability_contract_hash(contract):
        errors.append("G2A durability contract self-hash mismatch")

    profiles = contract.get("required_profiles", [])
    if list(profiles) != list(REQUIRED_PROFILES):
        errors.append("G2A durability required-profile inventory mismatch")
    locators = contract.get("profile_locators") or {}
    if set(locators) != set(REQUIRED_PROFILES):
        errors.append("G2A durability locator inventory mismatch")
    else:
        values = [str(locators[profile]) for profile in REQUIRED_PROFILES]
        if any(not _valid_kaggle_locator(value) for value in values):
            errors.append("G2A durability contract contains an invalid Kaggle locator")
        if len(set(values)) != len(REQUIRED_PROFILES):
            errors.append("G2A durability locators are not unique across five profiles")
        if int(contract.get("unique_locator_count", 0) or 0) != len(REQUIRED_PROFILES):
            errors.append("G2A durability unique-locator count mismatch")

    input_hashes = contract.get("input_summary_sha256") or {}
    if set(input_hashes) != set(REQUIRED_PROFILES) or any(len(str(value)) != 64 for value in input_hashes.values()):
        errors.append("G2A durability input-summary hash inventory invalid")
    if expected_input_summary_sha256 is not None and input_hashes != expected_input_summary_sha256:
        errors.append("G2A durability contract is not bound to the G2A barrier input summaries")
    return errors
