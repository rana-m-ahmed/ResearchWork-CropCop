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

from cropcop_je.tracka_v12 import (  # noqa: E402
    EXPERIMENT_SPECS,
    PAIR_IDS,
    R13_MODEL_ID,
    SEEDS,
    experiment_config_path,
    validate_materialized_configs,
    validate_tracka_v12_config,
)


class TrackAV12ContractTests(unittest.TestCase):
    def load(self, experiment_id: str) -> dict:
        path = ROOT / experiment_config_path(experiment_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def test_exact_frozen_experiment_inventory(self):
        self.assertEqual(len(EXPERIMENT_SPECS), 11)
        self.assertEqual(
            set(EXPERIMENT_SPECS),
            {
                "R06-EFFB0-CONTEXT-S2",
                "R06-EFFB0-CONTEXT-S3",
                "R07-CNXTT-CONTEXT-S2",
                "R07-CNXTT-CONTEXT-S3",
                "R12-MNV4-LOGITS-S2",
                "R12-MNV4-LOGITS-S3",
                "R12-MNV4-FEATURE-S2",
                "R12-MNV4-FEATURE-S3",
                "R13-VIT-DLITTLE-DIFF-CONTEXT-S1",
                "R13-VIT-DLITTLE-DIFF-CONTEXT-S2",
                "R13-VIT-DLITTLE-DIFF-CONTEXT-S3",
            },
        )

    def test_all_materialized_configs_validate(self):
        self.assertEqual(validate_materialized_configs(ROOT), [])

    def test_seed_labels_are_globally_consistent(self):
        for spec in EXPERIMENT_SPECS.values():
            self.assertEqual(spec["seed"], SEEDS[spec["seed_label"]])

    def test_mnv4_pair_identity_tracks_seed(self):
        for experiment_id, spec in EXPERIMENT_SPECS.items():
            if spec["model_family"] == "mnv4":
                self.assertEqual(spec["pair_id"], PAIR_IDS[spec["seed_label"]], experiment_id)

    def test_wrong_seed_fails_closed(self):
        config = copy.deepcopy(self.load("R06-EFFB0-CONTEXT-S2"))
        config["seed"] = SEEDS["S3"]
        self.assertTrue(any("seed" in x for x in validate_tracka_v12_config(config)))

    def test_wrong_objective_fails_closed(self):
        config = copy.deepcopy(self.load("R12-MNV4-LOGITS-S2"))
        config["objective"]["kd"] = 0.34
        self.assertTrue(any("objective" in x for x in validate_tracka_v12_config(config)))

    def test_missing_protected_surface_fails_closed(self):
        config = copy.deepcopy(self.load("R07-CNXTT-CONTEXT-S3"))
        config["forbidden_surfaces"].remove("DS-V1-TEST-CONSUMED")
        self.assertTrue(any("forbidden-surface" in x for x in validate_tracka_v12_config(config)))

    def test_unauthorized_experiment_fails_closed(self):
        config = copy.deepcopy(self.load("R06-EFFB0-CONTEXT-S2"))
        config["experiment_id"] = "R99-UNAUTHORIZED-S1"
        self.assertTrue(validate_tracka_v12_config(config))
        with self.assertRaises(ValueError):
            experiment_config_path("R99-UNAUTHORIZED-S1")

    def test_wrong_pair_fails_closed(self):
        config = copy.deepcopy(self.load("R12-MNV4-FEATURE-S3"))
        config["pair_id"] = PAIR_IDS["S2"]
        self.assertTrue(any("pair_id" in x for x in validate_tracka_v12_config(config)))

    def test_wrong_r13_model_fails_closed(self):
        config = copy.deepcopy(self.load("R13-VIT-DLITTLE-DIFF-CONTEXT-S1"))
        config["model_id"] = R13_MODEL_ID + "-drift"
        self.assertTrue(any("model_id" in x for x in validate_tracka_v12_config(config)))

    def test_r13_xai_qualification_is_mandatory(self):
        config = copy.deepcopy(self.load("R13-VIT-DLITTLE-DIFF-CONTEXT-S2"))
        config["required_xai_interface_qualification"] = "wrong.json"
        self.assertTrue(any("required_xai_interface_qualification" in x for x in validate_tracka_v12_config(config)))


if __name__ == "__main__":
    unittest.main()
