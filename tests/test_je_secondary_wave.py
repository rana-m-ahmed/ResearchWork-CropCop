from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "journal_extension" / "src"))

from cropcop_je.secondary import (
    BASELINE_SPECS,
    PRINCIPAL_BACKFILL_ENVELOPES,
    PRINCIPAL_G1_SEAL_SHA256,
    PRINCIPAL_MNV4_PRETRAINED_SHA256,
    PRINCIPAL_S1_INIT_SHA256,
    SECONDARY_CONFIG_SPECS,
    SECONDARY_ENVELOPES,
    experiment_config_path,
    validate_backfill_envelope_config,
    validate_secondary_config,
    validate_secondary_envelope_config,
)
from cropcop_je.secondary_g2 import REQUIRED_SECONDARY_CALIBRATIONS


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class SecondaryWaveContractTests(unittest.TestCase):
    def test_secondary_configs_are_exactly_frozen(self):
        for eid, spec in SECONDARY_CONFIG_SPECS.items():
            cfg = load(ROOT / experiment_config_path(eid))
            self.assertEqual(cfg["experiment_id"], eid)
            self.assertEqual(validate_secondary_config(cfg), [])
            self.assertEqual(cfg["seed"], 21270083)
            self.assertEqual(cfg["objective"], spec["objective"])
            self.assertEqual(cfg["train_surface"], "DS-V1-TRAIN")
            self.assertEqual(cfg["validation_surface"], "DS-V1-VAL")
            self.assertIn("DS-V1-TEST-CONSUMED", cfg["forbidden_surfaces"])

    def test_secondary_registry_is_separate_and_exactly_four_states(self):
        registry = load(ROOT / "journal_extension/locks/secondary_experiment_registry.json")
        ids = [row["experiment_id"] for row in registry["experiments"]]
        self.assertEqual(set(ids), set(SECONDARY_CONFIG_SPECS))
        self.assertEqual(len(ids), 4)
        self.assertFalse(any("*" in eid for eid in ids))
        self.assertEqual(tuple(registry["calibrations"]), REQUIRED_SECONDARY_CALIBRATIONS)
        self.assertIn("CAL-EFFB0", registry["calibrations"])

    def test_principal_exact_anchor_constants_are_terminal_evidence_values(self):
        self.assertEqual(PRINCIPAL_G1_SEAL_SHA256, "442d9e7708749efedcb82ef9f4fd131211549770eecad985177eaaf4117052cd")
        self.assertEqual(PRINCIPAL_S1_INIT_SHA256, "7040e48fe697c539dbbdfa8f57ffd20e9327def82f6485267f224dcf4c34cce1")
        self.assertEqual(PRINCIPAL_MNV4_PRETRAINED_SHA256, "35ca23dc46c0075d9acdcab06d30e5395c4d97e422d8cb5e5e627823aeb7c1fa")

    def test_official_torchvision_baseline_identities_are_frozen(self):
        self.assertEqual(BASELINE_SPECS["effb0"]["weight_enum"], "EfficientNet_B0_Weights.IMAGENET1K_V1")
        self.assertEqual(BASELINE_SPECS["effb0"]["official_filename"], "efficientnet_b0_rwightman-7f5810bc.pth")
        self.assertEqual(BASELINE_SPECS["effb0"]["official_sha256_prefix"], "7f5810bc")
        self.assertEqual(BASELINE_SPECS["cnxtt"]["weight_enum"], "ConvNeXt_Tiny_Weights.IMAGENET1K_V1")
        self.assertEqual(BASELINE_SPECS["cnxtt"]["official_filename"], "convnext_tiny-983f1562.pth")
        self.assertEqual(BASELINE_SPECS["cnxtt"]["official_sha256_prefix"], "983f1562")

    def test_secondary_dual_t4_envelopes_are_fixed_and_disjoint(self):
        files = {
            "SEC-MECHANISM-T4X2-V1": "SEC_MECHANISM_T4X2.json",
            "SEC-CONTEXT-T4X2-V1": "SEC_CONTEXT_T4X2.json",
        }
        seen = set()
        for envelope_id, filename in files.items():
            config = load(ROOT / "journal_extension/kaggle/envelopes" / filename)
            self.assertEqual(config["envelope_id"], envelope_id)
            self.assertEqual(validate_secondary_envelope_config(config), [])
            self.assertEqual([c["slot"] for c in config["children"]], [0, 1])
            child_experiments = {c["experiment_id"] for c in config["children"]}
            self.assertFalse(seen & child_experiments)
            seen |= child_experiments
        self.assertEqual(seen, set(SECONDARY_CONFIG_SPECS))

    def test_principal_backfill_envelopes_cover_all_six_once(self):
        files = ["VAL_BACKFILL_S1_T4X2.json", "VAL_BACKFILL_S2_T4X2.json", "VAL_BACKFILL_S3_T4X2.json"]
        observed = []
        for filename in files:
            config = load(ROOT / "journal_extension/kaggle/envelopes" / filename)
            self.assertEqual(validate_backfill_envelope_config(config), [])
            self.assertEqual([c["slot"] for c in config["children"]], [0, 1])
            observed.extend(c["experiment_id"] for c in config["children"])
        self.assertEqual(len(observed), 6)
        self.assertEqual(len(set(observed)), 6)

    def test_backfill_is_inference_only_and_verifies_selected_checkpoint(self):
        source = (ROOT / "journal_extension/scripts/export_principal_validation.py").read_text(encoding="utf-8")
        self.assertIn("verify_selected", source)
        self.assertIn('surface="DS-V1-VAL"', source)
        self.assertIn('"training_performed": False', source)
        self.assertIn('"v1_test_accessed": False', source)
        self.assertNotIn("run_training(", source)

    def test_secondary_runtime_has_no_distributed_scientific_path(self):
        paths = [ROOT / "journal_extension/scripts/run_secondary_training.py", ROOT / "journal_extension/kaggle/run_secondary_envelope.py"]
        text = "\n".join(p.read_text(encoding="utf-8") for p in paths)
        for token in ("DistributedDataParallel", "nn.DataParallel", "SyncBatchNorm", "FullyShardedDataParallel"):
            self.assertNotIn(token, text)
        runner = paths[0].read_text(encoding="utf-8")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", runner)

    def test_restore_failure_is_fail_closed_not_fresh_restart(self):
        runner = (ROOT / "journal_extension/scripts/run_secondary_training.py").read_text(encoding="utf-8")
        self.assertIn("durable checkpoint restore failed; fresh scientific restart is forbidden", runner)
        self.assertNotIn("except Exception:\n                if resume_mode == \"required\":\n                    raise", runner)

    def test_scientific_envelope_resolves_fresh_vs_resume_before_launch(self):
        source = (ROOT / "journal_extension/kaggle/run_secondary_envelope.py").read_text(encoding="utf-8")
        self.assertIn("def durable_resume_mode", source)
        self.assertIn('"checkpoint_index.json" in cp.stdout', source)
        self.assertIn('return "required"', source)
        self.assertIn('return "never"', source)
        self.assertIn("fresh restart is forbidden", source)

    def test_secondary_g2_qualifies_all_model_families_and_durable_roundtrip(self):
        g2 = (ROOT / "journal_extension/kaggle/run_secondary_g2_envelope.py").read_text(encoding="utf-8")
        cal = (ROOT / "journal_extension/scripts/calibrate_secondary.py").read_text(encoding="utf-8")
        self.assertEqual(REQUIRED_SECONDARY_CALIBRATIONS, ("CAL-MNV4-DIRECT", "CAL-MNV4-TEACHER", "CAL-EFFB0", "CAL-CNXTT"))
        self.assertIn('"CAL-EFFB0"', g2)
        self.assertIn('"CAL-EFFB0"', cal)
        self.assertIn("store.sync", cal)
        self.assertIn("store.restore", cal)
        self.assertIn("run_checkpoint_contract_probe", g2)
        self.assertIn("publication_idempotency", g2)

    def test_capacity_plan_is_four_science_plus_two_backfill_initially(self):
        science_slots = sum(len(v["children"]) for v in SECONDARY_ENVELOPES.values())
        s3_backfill_slots = len(PRINCIPAL_BACKFILL_ENVELOPES["VAL-BACKFILL-S3-T4X2-V1"]["children"])
        self.assertEqual(science_slots, 4)
        self.assertEqual(s3_backfill_slots, 2)
        self.assertEqual(science_slots + s3_backfill_slots, 6)


if __name__ == "__main__":
    unittest.main()
