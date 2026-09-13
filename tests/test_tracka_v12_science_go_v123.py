from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
SCRIPTS = ROOT / "journal_extension" / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cropcop_je.hashing import sha256_json  # noqa: E402
from cropcop_je.tracka_v12_g2a import REQUIRED_PROFILES  # noqa: E402
from cropcop_je.tracka_v12_g2a_durability import build_g2a_durability_contract  # noqa: E402
import seal_tracka_v12_science_go_v123 as go123  # noqa: E402


def calibration(calibration_id: str, index: int) -> dict:
    run_id = f"G2A-{calibration_id}"
    locator = f"owner{index}/cropcop-{calibration_id.lower()}"
    return {
        "calibration_id": calibration_id,
        "status": "PASS",
        "durable_roundtrip_success": True,
        "durable_store_kind": "kaggle-dataset",
        "run_id": run_id,
        "durability": {
            "backend": "kaggle_private_dataset",
            "locator": locator,
            "preflight_private_access": {
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


class TrackAV12ScienceGoV123Tests(unittest.TestCase):
    def g2a(self):
        rows = [calibration(profile, index) for index, profile in enumerate(REQUIRED_PROFILES, start=1)]
        contract = build_g2a_durability_contract(rows)
        return {
            "barrier_sha256": "b" * 64,
            "input_summary_sha256": {row["calibration_id"]: sha256_json(row) for row in rows},
            "durability_contract": contract,
        }

    @patch.object(go123.base, "compose_gate_inputs")
    def test_valid_durability_hash_is_injected_into_g2a_binding(self, compose):
        compose.return_value = (
            {"g2a": "PASS"},
            {"g2a": {"kind": "track_a_v12_g2a_full_run_forecast", "barrier_sha256": "b" * 64}},
        )
        g2a = self.g2a()
        _gates, bindings = go123.compose_durability_bound_gate_inputs(
            source_git_commit="1" * 40,
            code_attestation={},
            lock_runtime_attestation={},
            g1a={},
            g2a=g2a,
            scheduler={},
            file_hashes={},
        )
        self.assertEqual(
            bindings["g2a"]["durability_contract_sha256"],
            g2a["durability_contract"]["durability_contract_sha256"],
        )

    @patch.object(go123.base, "compose_gate_inputs")
    def test_missing_durability_contract_fails_closed(self, compose):
        compose.return_value = (
            {"g2a": "PASS"},
            {"g2a": {"kind": "track_a_v12_g2a_full_run_forecast", "barrier_sha256": "b" * 64}},
        )
        with self.assertRaises(ValueError):
            go123.compose_durability_bound_gate_inputs(
                source_git_commit="1" * 40,
                code_attestation={},
                lock_runtime_attestation={},
                g1a={},
                g2a={"barrier_sha256": "b" * 64, "input_summary_sha256": {}},
                scheduler={},
                file_hashes={},
            )


if __name__ == "__main__":
    unittest.main()
