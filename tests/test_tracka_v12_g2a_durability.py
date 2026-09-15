from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.hashing import sha256_json  # noqa: E402
from cropcop_je.tracka_v12_g2a import REQUIRED_PROFILES  # noqa: E402
from cropcop_je.tracka_v12_g2a_durability import (  # noqa: E402
    G2A_RESTORE_CONTRACT,
    G2A_RESTORE_STORE_CLASS,
    build_g2a_durability_contract,
    validate_calibration_durability,
    validate_g2a_durability_contract,
)


def generation(previous: int, confirmed: int, marker: str) -> dict:
    return {
        "previous_version_number": previous,
        "confirmed_version_number": confirmed,
        "generation_marker_sha256": marker,
        "generation_roundtrip_verified": True,
    }


def summary(calibration_id: str, index: int) -> dict:
    run_id = f"G2A-{calibration_id}"
    locator = f"owner{index}/cropcop-{calibration_id.lower()}"
    return {
        "schema_version": "1.2.2",
        "calibration_id": calibration_id,
        "status": "PASS",
        "durable_roundtrip_success": True,
        "durable_store_kind": "kaggle-dataset",
        "run_id": run_id,
        "durability": {
            "backend": "kaggle_private_dataset",
            "locator": locator,
            "restore_status": "PASS",
            "restore_contract": G2A_RESTORE_CONTRACT,
            "restore_store_class": G2A_RESTORE_STORE_CLASS,
            "initial_sync_generation": generation(index, index + 1, f"{index:064x}"),
            "resumed_sync_generation": generation(index + 1, index + 2, f"{index + 100:064x}"),
            "preflight_private_access": {
                "schema_version": "1.0",
                "status": "PASS",
                "kind": "kaggle-dataset",
                "resolved_locator_count": 1,
                "errors": [],
                "checks": {
                    run_id: {
                        "locator": locator,
                        "owner_matches_authenticated_user": True,
                        "authenticated_read": True,
                        "authoritative_is_private": True,
                        "private_metadata_verified": True,
                        "write_generation_mutated_by_preflight": False,
                    }
                },
            },
        },
    }


class TrackAV12G2ADurabilityTests(unittest.TestCase):
    def summaries(self):
        return [summary(calibration_id, index) for index, calibration_id in enumerate(REQUIRED_PROFILES, start=1)]

    def test_valid_five_profile_private_durability_contract_passes(self):
        rows = self.summaries()
        for row in rows:
            self.assertEqual(validate_calibration_durability(row), [])
        contract = build_g2a_durability_contract(rows)
        expected = {row["calibration_id"]: sha256_json(row) for row in rows}
        self.assertEqual(contract["status"], "PASS")
        self.assertEqual(contract["required_restore_contract"], G2A_RESTORE_CONTRACT)
        self.assertEqual(validate_g2a_durability_contract(contract, expected_input_summary_sha256=expected), [])

    def test_filesystem_backend_is_rejected(self):
        row = self.summaries()[0]
        row["durable_store_kind"] = "filesystem"
        row["durability"]["backend"] = "filesystem"
        self.assertTrue(validate_calibration_durability(row))

    def test_duplicate_private_dataset_locator_is_rejected(self):
        rows = self.summaries()
        duplicate = rows[0]["durability"]["locator"]
        rows[1]["durability"]["locator"] = duplicate
        run_id = rows[1]["run_id"]
        rows[1]["durability"]["preflight_private_access"]["checks"][run_id]["locator"] = duplicate
        contract = build_g2a_durability_contract(rows)
        self.assertEqual(contract["status"], "FAIL")
        self.assertTrue(any("distinct" in error for error in contract["errors"]))

    def test_non_private_or_unauthenticated_preflight_is_rejected(self):
        row = self.summaries()[0]
        check = row["durability"]["preflight_private_access"]["checks"][row["run_id"]]
        check["authoritative_is_private"] = False
        check["authenticated_read"] = False
        errors = validate_calibration_durability(row)
        self.assertTrue(any("authoritative_is_private" in error for error in errors))
        self.assertTrue(any("authenticated_read" in error for error in errors))

    def test_legacy_restore_contract_is_rejected(self):
        row = self.summaries()[0]
        row["durability"]["restore_contract"] = "legacy_kaggle_restore"
        row["durability"]["restore_store_class"] = "KagglePrivateDatasetStore"
        errors = validate_calibration_durability(row)
        self.assertTrue(any("generation-aware v8 contract" in error for error in errors))
        self.assertTrue(any("store class" in error for error in errors))

    def test_nonadvancing_or_unverified_generation_is_rejected(self):
        row = self.summaries()[0]
        generation_row = row["durability"]["initial_sync_generation"]
        generation_row["confirmed_version_number"] = generation_row["previous_version_number"]
        generation_row["generation_roundtrip_verified"] = False
        errors = validate_calibration_durability(row)
        self.assertTrue(any("did not strictly advance" in error for error in errors))
        self.assertTrue(any("roundtrip is not verified" in error for error in errors))

    def test_invalid_generation_marker_hash_is_rejected(self):
        row = self.summaries()[0]
        row["durability"]["resumed_sync_generation"]["generation_marker_sha256"] = "not-a-sha"
        errors = validate_calibration_durability(row)
        self.assertTrue(any("marker SHA-256" in error for error in errors))

    def test_contract_must_bind_exact_input_summary_hashes(self):
        rows = self.summaries()
        contract = build_g2a_durability_contract(rows)
        expected = {row["calibration_id"]: sha256_json(row) for row in rows}
        broken = copy.deepcopy(expected)
        broken[REQUIRED_PROFILES[0]] = "0" * 64
        self.assertTrue(validate_g2a_durability_contract(contract, expected_input_summary_sha256=broken))

    def test_tampered_contract_self_hash_is_rejected(self):
        contract = build_g2a_durability_contract(self.summaries())
        contract["profile_locators"][REQUIRED_PROFILES[0]] = "other/locator"
        self.assertTrue(any("self-hash" in error for error in validate_g2a_durability_contract(contract)))

    def test_restore_contract_drift_is_rejected(self):
        contract = build_g2a_durability_contract(self.summaries())
        contract["required_restore_contract"] = "legacy"
        errors = validate_g2a_durability_contract(contract)
        self.assertTrue(any("restore requirement drift" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
