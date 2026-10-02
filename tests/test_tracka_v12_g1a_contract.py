from __future__ import annotations

import ast
import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.tracka_v12_g1a import (  # noqa: E402
    BASELINE_CONSUMERS,
    R12_CONSUMERS,
    R13_CONSUMERS,
    R13_MODEL_ID,
    R13_PARITY_CONTRACT_ID_V121,
    R13_PARITY_CONTRACT_SHA256_V121,
    R13_PARITY_PREEXECUTION_EVIDENCE_SHA256_V121,
    R13_PARITY_TOLERANCE,
    R13_PARITY_TOLERANCE_V121,
    R13_PRETRAINED_BYTES,
    R13_PRETRAINED_SHA256,
    SEEDS,
    g1a_seal_hash,
    validate_g1a_seal_object,
)


def valid_seal() -> dict:
    seal = {
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
        "r12_reused_pairs": {},
        "baselines": {"effb0": {}, "cnxtt": {}},
        "r13": {
            "model_id": R13_MODEL_ID,
            "timm_version": "1.0.26",
            "pretrained_sha256": R13_PRETRAINED_SHA256,
            "pretrained_bytes": R13_PRETRAINED_BYTES,
            "parity_contract_id": R13_PARITY_CONTRACT_ID_V121,
            "parity_contract_sha256": R13_PARITY_CONTRACT_SHA256_V121,
            "parity_preexecution_calibration_evidence_sha256": R13_PARITY_PREEXECUTION_EVIDENCE_SHA256_V121,
            "historical_v1_2_required_max_abs_difference": R13_PARITY_TOLERANCE,
            "required_max_abs_difference": R13_PARITY_TOLERANCE_V121,
            "parity_max_abs": 3.4332275390625e-05,
            "states": {},
        },
    }
    for label in ("S2", "S3"):
        seal["r12_reused_pairs"][label] = {
            "pair_id": f"MNV4-PAIR-{label}",
            "seed": SEEDS[label],
            "authorized_consumers": R12_CONSUMERS[label],
            "initialization_bytes_reused_exactly": True,
            "student_init_sha256": "a" * 64,
        }
        for key in ("effb0", "cnxtt"):
            seal["baselines"][key][label] = {
                "seed": SEEDS[label],
                "authorized_consumers": [BASELINE_CONSUMERS[key][label]],
                "pretrained_sha256": "b" * 64,
                "init_sha256": "c" * 64,
                "init_evidence_sha256": "d" * 64,
            }
    for label in ("S1", "S2", "S3"):
        seal["r13"]["states"][label] = {
            "seed": SEEDS[label],
            "authorized_consumers": [R13_CONSUMERS[label]],
            "init_sha256": "e" * 64,
            "init_evidence_sha256": "f" * 64,
        }
    seal["g1a_seal_sha256"] = g1a_seal_hash(seal)
    return seal


class TrackAV12G1AContractTests(unittest.TestCase):
    def test_valid_seal_passes(self):
        self.assertEqual(validate_g1a_seal_object(valid_seal()), [])

    def assert_rejected(self, mutator):
        seal = valid_seal()
        mutator(seal)
        seal["g1a_seal_sha256"] = g1a_seal_hash(seal)
        self.assertTrue(validate_g1a_seal_object(seal))

    def test_g1a_cannot_authorize_science(self):
        self.assert_rejected(lambda x: x.__setitem__("science_authorized", True))

    def test_test_surface_access_marker_fails(self):
        self.assert_rejected(lambda x: x["dataset"].__setitem__("v1_test_accessed", True))

    def test_external_surface_access_marker_fails(self):
        self.assert_rejected(lambda x: x["dataset"].__setitem__("external_surface_accessed", True))

    def test_wrong_r12_seed_fails(self):
        self.assert_rejected(lambda x: x["r12_reused_pairs"]["S2"].__setitem__("seed", SEEDS["S3"]))

    def test_r12_must_reuse_exact_bytes(self):
        self.assert_rejected(lambda x: x["r12_reused_pairs"]["S3"].__setitem__("initialization_bytes_reused_exactly", False))

    def test_wrong_baseline_consumer_fails(self):
        self.assert_rejected(lambda x: x["baselines"]["effb0"]["S2"].__setitem__("authorized_consumers", ["wrong"]))

    def test_wrong_r13_model_fails(self):
        self.assert_rejected(lambda x: x["r13"].__setitem__("model_id", R13_MODEL_ID + "-drift"))

    def test_r13_parity_above_tolerance_fails(self):
        self.assert_rejected(lambda x: x["r13"].__setitem__("parity_max_abs", R13_PARITY_TOLERANCE_V121 * 1.01))

    def test_wrong_r13_seed_consumer_fails(self):
        self.assert_rejected(lambda x: x["r13"]["states"]["S1"].__setitem__("authorized_consumers", [R13_CONSUMERS["S2"]]))

    def test_self_hash_is_mandatory(self):
        seal = valid_seal()
        seal["g1a_seal_sha256"] = "0" * 64
        self.assertTrue(any("self-hash" in x for x in validate_g1a_seal_object(seal)))

    def test_g1a_sealer_imports_frozen_torchvision_version(self):
        path = ROOT / "journal_extension" / "scripts" / "seal_tracka_v12_g1a.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "cropcop_je.secondary"
            for alias in node.names
        }
        self.assertIn("TORCHVISION_VERSION", imports)


if __name__ == "__main__":
    unittest.main()
