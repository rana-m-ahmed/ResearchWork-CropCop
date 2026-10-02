from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.hashing import sha256_json  # noqa: E402
from cropcop_je.tracka_v12_g1a import (  # noqa: E402
    R13_PARITY_CONTRACT_ID_V121,
    R13_PARITY_CONTRACT_SHA256_V121,
    R13_PARITY_PREEXECUTION_EVIDENCE_SHA256_V121,
    R13_PARITY_TOLERANCE,
    R13_PARITY_TOLERANCE_V121,
    validate_g1a_seal_object,
)


class TrackAV12G1AParityContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seal_path = (
            ROOT
            / "journal_extension"
            / "evidence"
            / "public"
            / "runs"
            / "TRACKA-V12-G1A-56023042e577"
            / "TRACKA_V12_G1A_SEAL.json"
        )
        if cls.seal_path.is_file():
            cls.seal = json.loads(cls.seal_path.read_text(encoding="utf-8"))
        else:
            # CI fixture mirrors the exact sealed production parity contract.
            cls.seal = {
                "schema_version": "1.0",
                "status": "PASS",
                "science_authorized": False,
                "authority": {"id": "EAAI-JE-SDL-v2.1-QA"},
                "dataset": {
                    "manifest_sha256": "bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2",
                    "class_map_sha256": "46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2",
                    "v1_test_accessed": False,
                    "external_surface_accessed": False,
                },
                "r12_reused_pairs": {
                    "S2": {
                        "pair_id": "MNV4-PAIR-S2",
                        "seed": 606135704,
                        "authorized_consumers": ["R12-MNV4-LOGITS-S2", "R12-MNV4-FEATURE-S2"],
                        "initialization_bytes_reused_exactly": True,
                        "student_init_sha256": "1" * 64,
                    },
                    "S3": {
                        "pair_id": "MNV4-PAIR-S3",
                        "seed": 1153870846,
                        "authorized_consumers": ["R12-MNV4-LOGITS-S3", "R12-MNV4-FEATURE-S3"],
                        "initialization_bytes_reused_exactly": True,
                        "student_init_sha256": "2" * 64,
                    },
                },
                "baselines": {
                    key: {
                        label: {
                            "seed": seed,
                            "authorized_consumers": [consumer],
                            "pretrained_sha256": "3" * 64,
                            "init_sha256": "4" * 64,
                            "init_evidence_sha256": "5" * 64,
                        }
                        for label, seed, consumer in rows
                    }
                    for key, rows in {
                        "effb0": [
                            ("S2", 606135704, "R06-EFFB0-CONTEXT-S2"),
                            ("S3", 1153870846, "R06-EFFB0-CONTEXT-S3"),
                        ],
                        "cnxtt": [
                            ("S2", 606135704, "R07-CNXTT-CONTEXT-S2"),
                            ("S3", 1153870846, "R07-CNXTT-CONTEXT-S3"),
                        ],
                    }.items()
                },
                "r13": {
                    "model_id": "vit_dlittle_patch16_reg1_gap_256.sbb_nadamuon_in1k",
                    "timm_version": "1.0.26",
                    "pretrained_sha256": "92ec2d996329be8c9a449d4e38f847e79c34fadc32b6491739bacdaf425ab0ed",
                    "pretrained_bytes": 90095744,
                    "parity_contract_id": R13_PARITY_CONTRACT_ID_V121,
                    "parity_contract_sha256": R13_PARITY_CONTRACT_SHA256_V121,
                    "parity_preexecution_calibration_evidence_sha256": R13_PARITY_PREEXECUTION_EVIDENCE_SHA256_V121,
                    "historical_v1_2_required_max_abs_difference": R13_PARITY_TOLERANCE,
                    "required_max_abs_difference": R13_PARITY_TOLERANCE_V121,
                    "parity_max_abs": 3.4332275390625e-05,
                    "states": {
                        "S1": {"seed": 21270083, "authorized_consumers": ["R13-VIT-DLITTLE-DIFF-CONTEXT-S1"], "init_sha256": "6" * 64, "init_evidence_sha256": "7" * 64},
                        "S2": {"seed": 606135704, "authorized_consumers": ["R13-VIT-DLITTLE-DIFF-CONTEXT-S2"], "init_sha256": "8" * 64, "init_evidence_sha256": "9" * 64},
                        "S3": {"seed": 1153870846, "authorized_consumers": ["R13-VIT-DLITTLE-DIFF-CONTEXT-S3"], "init_sha256": "a" * 64, "init_evidence_sha256": "b" * 64},
                    },
                },
            }
            cls.seal["g1a_seal_sha256"] = sha256_json(cls.seal)

    def test_exact_v121_contract_passes(self):
        errors = validate_g1a_seal_object(copy.deepcopy(self.seal))
        self.assertNotIn("G1A R13 parity gate failed", errors)
        self.assertNotIn("G1A R13 parity contract ID mismatch", errors)
        self.assertNotIn("G1A R13 parity contract SHA mismatch", errors)
        self.assertNotIn("G1A R13 pre-execution parity evidence SHA mismatch", errors)

    def test_wrong_contract_id_fails(self):
        seal = copy.deepcopy(self.seal)
        seal["r13"]["parity_contract_id"] = "TRACKA-A1-R13-PRETRAINED-IDENTITY-V1.2"
        seal["g1a_seal_sha256"] = sha256_json({k: v for k, v in seal.items() if k != "g1a_seal_sha256"})
        self.assertIn("G1A R13 parity contract ID mismatch", validate_g1a_seal_object(seal))

    def test_wrong_contract_sha_fails(self):
        seal = copy.deepcopy(self.seal)
        seal["r13"]["parity_contract_sha256"] = "0" * 64
        seal["g1a_seal_sha256"] = sha256_json({k: v for k, v in seal.items() if k != "g1a_seal_sha256"})
        self.assertIn("G1A R13 parity contract SHA mismatch", validate_g1a_seal_object(seal))

    def test_parity_above_v121_bound_fails(self):
        seal = copy.deepcopy(self.seal)
        seal["r13"]["parity_max_abs"] = 5.1e-05
        seal["g1a_seal_sha256"] = sha256_json({k: v for k, v in seal.items() if k != "g1a_seal_sha256"})
        self.assertIn("G1A R13 parity gate failed", validate_g1a_seal_object(seal))


if __name__ == "__main__":
    unittest.main()
