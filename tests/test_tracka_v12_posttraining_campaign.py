from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "journal_extension" / "locks" / "track_a_posttraining_campaign_v1.json"
AUTHORITY = ROOT / "journal_extension" / "locks" / "track_a_posttraining_closure_authority_v1.json"


class TrackAPosttrainingCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.campaign = json.loads(CAMPAIGN.read_text(encoding="utf-8"))
        cls.authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))

    def test_campaign_is_preexecution_and_science_immutable(self):
        self.assertEqual(self.campaign["status"], "PRE_EXECUTION_LOCK")
        self.assertFalse(self.campaign["scientific_semantics_change_authorized"])
        self.assertEqual(
            self.campaign["parent_training_science_source_git_commit"],
            "56023042e57758591df9babb3438f191dbe10312",
        )

    def test_continuation_binding_exactly_matches_authority(self):
        binding = self.campaign["continuation_account_binding"]
        authority_states = self.authority["state_inventory"]
        recovered = {
            experiment_id
            for experiment_id, row in authority_states.items()
            if row.get("terminal_metadata_recovery_required") is True
        }
        self.assertEqual(set(binding), recovered)
        self.assertEqual(len(binding), 11)
        for experiment_id, account in binding.items():
            self.assertEqual(authority_states[experiment_id]["terminal_account_id"], account)

    def test_historical_preference_inventory_exact(self):
        authority_states = set(self.authority["state_inventory"])
        historical = authority_states - set(self.campaign["continuation_account_binding"])
        prefs = self.campaign["historical_placement_preferences"]
        self.assertEqual(set(prefs), historical)
        self.assertEqual(len(historical), 10)
        for order in prefs.values():
            self.assertEqual(set(order), {"K1", "K2", "K3"})
            self.assertEqual(len(order), 3)

    def test_relocation_is_pre_metric_only(self):
        policy = self.campaign["historical_relocation_policy"]
        self.assertTrue(policy["authorized"])
        self.assertIn("before any new post-training metric", policy["timing"])
        forbidden = set(policy["forbidden_reasons"])
        self.assertEqual(
            forbidden,
            {
                "validation metric",
                "robustness result",
                "XAI result",
                "runtime speed result",
                "model quality impression",
            },
        )

    def test_execution_is_two_isolated_gpus_without_ddp(self):
        execution = self.campaign["execution_model"]
        self.assertEqual(execution["gpus_per_account"], 2)
        self.assertFalse(execution["cross_gpu_gradient_synchronization"])
        self.assertFalse(execution["distributed_data_parallel"])
        self.assertIn("static pre-results", execution["scheduling_basis"])

    def test_all_protected_surfaces_are_closed(self):
        self.assertEqual(
            set(self.campaign["protected_surfaces"].values()),
            {"CLOSED"},
        )
        self.assertEqual(
            set(self.campaign["protected_surfaces"]),
            {"DS-V1-TEST-CONSUMED", "TRACK-B-PREDICTIONS", "TRACK-C-CANDIDATE-RESULTS"},
        )

    def test_final_exit_criteria_are_complete(self):
        exit_gate = self.campaign["final_exit_criteria"]
        self.assertEqual(exit_gate["exact_scientific_state_count"], 21)
        self.assertEqual(exit_gate["direct_state_count"], 12)
        self.assertEqual(exit_gate["auxiliary_state_count"], 9)
        self.assertEqual(exit_gate["recovered_continuation_state_count"], 11)
        self.assertEqual(exit_gate["historical_state_count"], 10)
        self.assertEqual(exit_gate["unique_selected_checkpoint_count"], 21)
        self.assertEqual(exit_gate["direct_evidence_complete"], 12)
        self.assertEqual(exit_gate["xai_evidence_complete"], 12)
        self.assertEqual(exit_gate["auxiliary_evidence_complete"], 9)
        self.assertEqual(exit_gate["account_completion_manifests"], 3)
        self.assertEqual(exit_gate["final_closure_audit"], "PASS")


if __name__ == "__main__":
    unittest.main()
