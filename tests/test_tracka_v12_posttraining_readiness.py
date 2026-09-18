from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
SCRIPTS = ROOT / "journal_extension" / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cropcop_je.hashing import sha256_json  # noqa: E402
import build_tracka_v12_posttraining_readiness as global_gate  # noqa: E402

AUTH_PATH = ROOT / "journal_extension" / "locks" / "track_a_posttraining_closure_authority_v1.json"
SCIENCE = "56023042e57758591df9babb3438f191dbe10312"
ANALYSIS = "a" * 40


def seal_account(account_id: str, states: dict, authority_sha: str) -> dict:
    payload = {
        "schema_version": "1.0",
        "status": "PASS",
        "gate_kind": "track_a_v12_posttraining_account_readiness",
        "account_id": account_id,
        "analysis_source_git_commit": ANALYSIS,
        "training_science_source_git_commit": SCIENCE,
        "closure_authority_sha256": authority_sha,
        "manifest_sha256": "bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2",
        "class_map_sha256": "46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2",
        "state_count": len(states),
        "states": states,
        "v1_test_closed": True,
        "external_predictions_closed": True,
        "track_c_candidate_runtime_closed": True,
        "training_or_adaptation_performed": False,
        "optimizer_state_advanced": False,
        "private_material_verified_locally": True,
        "evidence_targets_ready": True,
    }
    payload["gate_sha256"] = sha256_json(payload)
    return payload


class TrackAV12PosttrainingReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.authority = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
        cls.authority_sha = sha256_json(cls.authority)

    def gates(self):
        buckets = {"K1": {}, "K2": {}, "K3": {}}
        historical_cycle = iter(["K1", "K2", "K3"] * 4)
        for experiment_id, auth in self.authority["state_inventory"].items():
            account = auth.get("terminal_account_id") if auth.get("terminal_metadata_recovery_required") else next(historical_cycle)
            buckets[account][experiment_id] = {
                "role": auth["role"],
                "run_id": auth["run_id"],
                "scientific_source_git_commit": auth["scientific_source_git_commit"],
                "selected_checkpoint_sha256": auth["selected_checkpoint_sha256"],
                "selected_checkpoint_file_sha256": auth["selected_checkpoint_sha256"],
                "selected_checkpoint_epoch": int(auth.get("selected_epoch", 30)),
                "run_record_sha256": "1" * 64,
                "recovery_certificate_sha256": "2" * 64 if auth.get("terminal_metadata_recovery_required") else None,
                "recovered_terminal_metadata": bool(auth.get("terminal_metadata_recovery_required")),
                "private_selected_checkpoint_verified": True,
            }
        return [
            (AUTH_PATH, seal_account(account, states, self.authority_sha))
            for account, states in buckets.items()
        ]

    def test_three_account_union_closes_exact_21_state_readiness(self):
        result = global_gate.validate_global_readiness(
            authority=self.authority,
            account_gates=self.gates(),
            analysis_source=ANALYSIS,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["state_count"], 21)
        self.assertEqual(result["direct_state_count"], 12)
        self.assertEqual(result["auxiliary_state_count"], 9)
        self.assertEqual(result["new_recovered_state_count"], 11)
        self.assertEqual(result["historical_state_count"], 10)
        self.assertEqual(result["unique_selected_checkpoint_count"], 21)
        self.assertTrue(result["ready_for_posttraining_evidence"])

    def test_duplicate_state_across_accounts_fails(self):
        gates = self.gates()
        experiment_id = next(iter(gates[0][1]["states"]))
        gates[1][1]["states"][experiment_id] = copy.deepcopy(gates[0][1]["states"][experiment_id])
        gates[1][1]["state_count"] = len(gates[1][1]["states"])
        gates[1][1].pop("gate_sha256")
        gates[1][1]["gate_sha256"] = sha256_json(gates[1][1])
        with self.assertRaises(RuntimeError):
            global_gate.validate_global_readiness(authority=self.authority, account_gates=gates, analysis_source=ANALYSIS)

    def test_wrong_selected_checkpoint_fails(self):
        gates = self.gates()
        experiment_id = next(iter(gates[0][1]["states"]))
        gates[0][1]["states"][experiment_id]["selected_checkpoint_sha256"] = "0" * 64
        gates[0][1].pop("gate_sha256")
        gates[0][1]["gate_sha256"] = sha256_json(gates[0][1])
        with self.assertRaises(RuntimeError):
            global_gate.validate_global_readiness(authority=self.authority, account_gates=gates, analysis_source=ANALYSIS)

    def test_open_protected_surface_fails(self):
        gates = self.gates()
        gates[2][1]["v1_test_closed"] = False
        gates[2][1].pop("gate_sha256")
        gates[2][1]["gate_sha256"] = sha256_json(gates[2][1])
        with self.assertRaises(RuntimeError):
            global_gate.validate_global_readiness(authority=self.authority, account_gates=gates, analysis_source=ANALYSIS)


if __name__ == "__main__":
    unittest.main()
