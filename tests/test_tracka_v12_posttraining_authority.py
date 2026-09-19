from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTH = ROOT / "journal_extension" / "locks" / "track_a_posttraining_closure_authority_v1.json"
SCIENCE = "56023042e57758591df9babb3438f191dbe10312"


class TrackAV12PosttrainingAuthorityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(AUTH.read_text(encoding="utf-8"))
        cls.states = cls.payload["state_inventory"]

    def test_exact_21_state_inventory(self):
        self.assertEqual(self.payload["state_count"], 21)
        self.assertEqual(len(self.states), 21)
        direct = [row for row in self.states.values() if row["role"] == "direct"]
        auxiliary = [row for row in self.states.values() if row["role"] == "auxiliary"]
        self.assertEqual(len(direct), 12)
        self.assertEqual(len(auxiliary), 9)

    def test_selected_checkpoint_hashes_are_unique_and_complete(self):
        hashes = [row["selected_checkpoint_sha256"] for row in self.states.values()]
        self.assertEqual(len(hashes), 21)
        self.assertEqual(len(set(hashes)), 21)
        self.assertTrue(all(len(value) == 64 for value in hashes))

    def test_exact_eleven_continuation_states_require_recovery(self):
        recovered = {
            experiment_id
            for experiment_id, row in self.states.items()
            if row.get("terminal_metadata_recovery_required") is True
        }
        self.assertEqual(len(recovered), 11)
        for experiment_id in recovered:
            row = self.states[experiment_id]
            self.assertEqual(row["scientific_source_git_commit"], SCIENCE)
            self.assertIn(row["terminal_account_id"], {"K1", "K2", "K3"})
            self.assertIn("durable_checkpoint_locator", row)

    def test_protected_surfaces_remain_closed(self):
        self.assertEqual(
            self.payload["protected_surfaces"],
            {"v1_test": "CLOSED", "external_predictions": "CLOSED", "track_c_candidate_runtime": "CLOSED"},
        )
        forbidden = set(self.payload["forbidden_actions"])
        self.assertIn("V1-test access", forbidden)
        self.assertIn("Track-B prediction access", forbidden)
        self.assertIn("Track-C candidate-result access", forbidden)
        self.assertIn("training or optimizer advance during posttraining evidence", forbidden)

    def test_no_scientific_semantics_change_authorized(self):
        self.assertFalse(self.payload["scientific_semantics_change_authorized"])
        self.assertFalse(self.payload["pre_analysis_gate"]["analysis_may_start_before_pass"])
        self.assertEqual(self.payload["parent_training_science_source_git_commit"], SCIENCE)


if __name__ == "__main__":
    unittest.main()
