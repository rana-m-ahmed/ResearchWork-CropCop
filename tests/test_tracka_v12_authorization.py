from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.tracka_v12_authorization import (  # noqa: E402
    build_science_authorization,
    validate_science_authorization,
)


class TrackAV12AuthorizationTests(unittest.TestCase):
    def gates(self):
        return {
            "exact_head_ci": "PASS",
            "immutable_v12_lock": "PASS",
            "v121_runtime_qualification": "PASS",
            "config_contract_and_failure_injection": "PASS",
            "principal_science_diff": "PASS",
            "secondary_science_diff": "PASS",
            "g1a": "PASS",
            "g2a": "PASS",
            "scheduler_freeze": "PASS",
            "analysis_selector_implementation": "PASS",
            "checkpoint_recovery_contract": "PASS",
        }

    def build(self):
        return build_science_authorization(
            source_git_commit="1" * 40,
            g1a_seal_sha256="2" * 64,
            g2a_barrier_sha256="3" * 64,
            scheduler_freeze_sha256="4" * 64,
            pre_science_gates=self.gates(),
            evidence_bindings={"qa": "synthetic-test"},
        )

    def validate(self, payload):
        return validate_science_authorization(
            payload,
            expected_source_sha="1" * 40,
            expected_g1a_seal_sha256="2" * 64,
            expected_g2a_barrier_sha256="3" * 64,
            expected_scheduler_freeze_sha256="4" * 64,
        )

    def test_valid_authorization_passes(self):
        self.assertEqual(self.validate(self.build()), [])

    def test_non_pass_gate_cannot_authorize_science(self):
        gates = self.gates()
        gates["g2a"] = "FAIL"
        with self.assertRaises(Exception):
            build_science_authorization(
                source_git_commit="1" * 40,
                g1a_seal_sha256="2" * 64,
                g2a_barrier_sha256="3" * 64,
                scheduler_freeze_sha256="4" * 64,
                pre_science_gates=gates,
                evidence_bindings={},
            )

    def test_test_surface_must_remain_closed(self):
        payload = self.build()
        payload["v1_test_closed"] = False
        self.assertTrue(self.validate(payload))

    def test_external_predictions_must_remain_closed(self):
        payload = self.build()
        payload["external_predictions_closed"] = False
        self.assertTrue(self.validate(payload))

    def test_track_c_results_must_remain_closed(self):
        payload = self.build()
        payload["track_c_candidate_runtime_closed"] = False
        self.assertTrue(self.validate(payload))

    def test_exact_identity_bindings_are_mandatory(self):
        payload = self.build()
        self.assertTrue(validate_science_authorization(
            payload,
            expected_source_sha="9" * 40,
            expected_g1a_seal_sha256="2" * 64,
            expected_g2a_barrier_sha256="3" * 64,
            expected_scheduler_freeze_sha256="4" * 64,
        ))


if __name__ == "__main__":
    unittest.main()
