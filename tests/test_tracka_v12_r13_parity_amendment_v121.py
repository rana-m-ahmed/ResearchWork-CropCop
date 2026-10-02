from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.tracka_v12_g1a import R13_PARITY_TOLERANCE as HISTORICAL_TOLERANCE
from cropcop_je.tracka_v12_g1a_v121 import (
    R13_PARITY_CONTRACT_ID,
    R13_PARITY_HISTORICAL_V12_TOLERANCE,
    R13_PARITY_TOLERANCE,
    validate_g1a_seal_object,
)

AMEND = ROOT / "journal_extension" / "amendments" / "track_a_strengthening_v1"
OLD_CONTRACT = AMEND / "r13_pretrained_identity_and_normalization_contract_v1_2.json"
NEW_CONTRACT = AMEND / "r13_pretrained_identity_and_normalization_contract_v1_2_1.json"
EVIDENCE = AMEND / "r13_parity_preexecution_evidence_v1_2_1.json"


class R13ParityAmendmentV121Tests(unittest.TestCase):
    def test_historical_and_superseding_thresholds_are_explicit(self):
        old = json.loads(OLD_CONTRACT.read_text(encoding="utf-8"))
        new = json.loads(NEW_CONTRACT.read_text(encoding="utf-8"))
        evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        self.assertEqual(float(old["parity_gate"]["required_max_abs_difference"]), 1e-5)
        self.assertEqual(new["contract_id"], R13_PARITY_CONTRACT_ID)
        self.assertEqual(new["supersedes_contract_id"], old["contract_id"])
        self.assertEqual(float(new["parity_gate"]["original_v1_2_required_max_abs_difference"]), 1e-5)
        self.assertEqual(float(new["parity_gate"]["required_max_abs_difference"]), 5e-5)
        self.assertEqual(float(evidence["kaggle_observation"]["observed_max_abs_difference"]), 3.4332275390625e-05)
        self.assertLess(1e-5, float(evidence["kaggle_observation"]["observed_max_abs_difference"]))
        self.assertLessEqual(float(evidence["kaggle_observation"]["observed_max_abs_difference"]), 5e-5)

    def test_versioned_source_constant_matches_contract(self):
        # Importing v1.2.1 intentionally patches the historical implementation
        # module's call-time tolerance for this execution process.
        self.assertEqual(R13_PARITY_CONTRACT_ID, "TRACKA-A1-R13-PRETRAINED-IDENTITY-V1.2.1")
        self.assertEqual(R13_PARITY_HISTORICAL_V12_TOLERANCE, 1e-5)
        self.assertEqual(R13_PARITY_TOLERANCE, 5e-5)
        self.assertEqual(HISTORICAL_TOLERANCE, 1e-5)

    def test_seal_validator_rejects_missing_or_wrong_contract_identity(self):
        base = {
            "schema_version": "1.0",
            "status": "PASS",
            "science_authorized": False,
            "authority": {},
            "dataset": {},
            "r12_reused_pairs": {},
            "baselines": {},
            "r13": {
                "parity_contract_id": "wrong",
                "required_max_abs_difference": 5e-5,
                "historical_v1_2_required_max_abs_difference": 1e-5,
            },
        }
        errors = validate_g1a_seal_object(base)
        self.assertIn("G1A R13 parity contract identity mismatch", errors)
        base["r13"]["parity_contract_id"] = R13_PARITY_CONTRACT_ID
        base["r13"]["required_max_abs_difference"] = 1e-5
        errors = validate_g1a_seal_object(base)
        self.assertIn("G1A R13 required parity tolerance mismatch", errors)

    def test_no_scientific_or_protected_output_informed_amendment(self):
        evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        self.assertFalse(evidence["science_authorized"])
        self.assertFalse(evidence["scientific_metric_computed"])
        self.assertFalse(evidence["classifier_prediction_opened"])
        self.assertFalse(evidence["v1_test_accessed"])
        self.assertFalse(evidence["external_surface_accessed"])
        self.assertFalse(evidence["kaggle_observation"]["scientific_training_started"])


if __name__ == "__main__":
    unittest.main()
