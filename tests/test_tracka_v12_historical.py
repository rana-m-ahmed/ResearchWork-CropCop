from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.tracka_v12_recovery import SCIENCE_SOURCE_SHA  # noqa: E402
import cropcop_je.tracka_v12_source_materialization as source_materialization  # noqa: E402
from cropcop_je.tracka_v12_source_materialization import (  # noqa: E402
    PROFILE_CONTINUATION_V8,
    PROFILE_HISTORICAL_LEGACY,
    persistence_profile,
    restore_for_historical_availability,
)
from cropcop_je.tracka_v12_historical import (  # noqa: E402
    HISTORICAL_SECONDARY_DIRECT_SPECS,
    HISTORICAL_TRACKA_CLOSURE_SPECS,
    validate_historical_closure_identity,
)


class TrackAV12HistoricalLineageTests(unittest.TestCase):
    def test_registry_contains_exact_ten_completed_states(self):
        expected = {
            "R04-MNV4-DIRECT-S1", "R04-MNV4-DIRECT-S2", "R04-MNV4-DIRECT-S3",
            "R05-MNV4-TEACHER-S1", "R05-MNV4-TEACHER-S2", "R05-MNV4-TEACHER-S3",
            "R12-MNV4-LOGITS-S1", "R12-MNV4-FEATURE-S1",
            "R06-EFFB0-CONTEXT-S1", "R07-CNXTT-CONTEXT-S1",
        }
        self.assertEqual(set(HISTORICAL_TRACKA_CLOSURE_SPECS), expected)
        for experiment_id, row in HISTORICAL_TRACKA_CLOSURE_SPECS.items():
            self.assertEqual(len(row["source_git_commit"]), 40, experiment_id)
            self.assertEqual(len(row["selected_checkpoint_sha256"]), 64, experiment_id)
            self.assertGreater(int(row["selected_epoch"]), 0, experiment_id)

    def test_registry_matches_canonical_wave_closure_files(self):
        evidence = ROOT / "journal_extension" / "evidence" / "public" / "track_a"
        wave1 = json.loads((evidence / "WAVE1_PRINCIPAL_VALIDATION_CLOSURE.json").read_text(encoding="utf-8"))
        wave2 = json.loads((evidence / "WAVE2_SECONDARY_VALIDATION_CLOSURE.json").read_text(encoding="utf-8"))
        observed = {}
        for payload, wave in ((wave1, "WAVE1"), (wave2, "WAVE2")):
            for row in payload["runs"]:
                observed[row["experiment_id"]] = {
                    "wave": wave,
                    "run_id": row["run_id"],
                    "selected_epoch": row["selected_epoch"],
                    "selected_checkpoint_sha256": row["selected_checkpoint_sha256"],
                }
        self.assertEqual(set(observed), set(HISTORICAL_TRACKA_CLOSURE_SPECS))
        for experiment_id, canonical in HISTORICAL_TRACKA_CLOSURE_SPECS.items():
            self.assertEqual(observed[experiment_id]["wave"], canonical["wave"])
            self.assertEqual(observed[experiment_id]["run_id"], canonical["run_id"])
            self.assertEqual(observed[experiment_id]["selected_epoch"], canonical["selected_epoch"])
            self.assertEqual(observed[experiment_id]["selected_checkpoint_sha256"], canonical["selected_checkpoint_sha256"])

    def test_historical_identity_rejects_source_run_and_checkpoint_tampering(self):
        experiment_id = "R04-MNV4-DIRECT-S1"
        spec = HISTORICAL_TRACKA_CLOSURE_SPECS[experiment_id]
        record = {
            "experiment_id": experiment_id,
            "status": "PASS",
            "run_id": spec["run_id"],
            "source_git_commit": spec["source_git_commit"],
            "continuation_required": False,
            "v1_test_accessed": False,
            "protected_external_surface_accessed": False,
            "artifact_locators": {"selected_checkpoint": {"sha256": spec["selected_checkpoint_sha256"]}},
            "result_summary": {
                "selected_checkpoint_sha256": spec["selected_checkpoint_sha256"],
                "selected_epoch": spec["selected_epoch"],
            },
        }
        self.assertEqual(validate_historical_closure_identity(record), [])
        for field, bad_value, expected_fragment in (
            ("source_git_commit", "f" * 40, "source_git_commit"),
            ("run_id", "wrong-run", "run_id"),
        ):
            broken = json.loads(json.dumps(record))
            broken[field] = bad_value
            self.assertTrue(any(expected_fragment in error for error in validate_historical_closure_identity(broken)))
        broken = json.loads(json.dumps(record))
        broken["artifact_locators"]["selected_checkpoint"]["sha256"] = "0" * 64
        broken["result_summary"]["selected_checkpoint_sha256"] = "0" * 64
        self.assertIn("historical selected checkpoint differs from immutable Wave closure", validate_historical_closure_identity(broken))

    def test_historical_run_records_bind_legacy_kaggle_persistence_profile(self):
        experiment_id = "R04-MNV4-DIRECT-S1"
        spec = HISTORICAL_TRACKA_CLOSURE_SPECS[experiment_id]
        selected = spec["selected_checkpoint_sha256"]
        record = {
            "experiment_id": experiment_id,
            "status": "PASS",
            "run_id": spec["run_id"],
            "source_git_commit": spec["source_git_commit"],
            "continuation_required": False,
            "artifact_locators": {
                "selected_checkpoint": {
                    "sha256": selected,
                    "durable_locator": "owner/historical-private",
                }
            },
            "result_summary": {
                "selected_checkpoint_sha256": selected,
                "selected_epoch": spec["selected_epoch"],
            },
            "persistence_status": {
                "backend": "kaggle_private_dataset",
                "files": [
                    "checkpoint_index.json",
                    f"objects/selected.g00000001.{selected[:16]}.ckpt",
                ],
            },
        }
        self.assertEqual(persistence_profile(record), PROFILE_HISTORICAL_LEGACY)

    def test_recovered_continuation_binds_generation_aware_v8_profile(self):
        experiment_id = "R13-VIT-DLITTLE-DIFF-CONTEXT-S1"
        selected = "b" * 64
        record = {
            "experiment_id": experiment_id,
            "status": "PASS",
            "mode": "scientific",
            "source_git_commit": SCIENCE_SOURCE_SHA,
            "continuation_required": False,
            "allowed_surfaces": ["DS-V1-TRAIN", "DS-V1-VAL"],
            "v1_test_accessed": False,
            "external_protected_surface_accessed": False,
            "protected_external_surface_accessed": False,
            "artifact_locators": {
                "selected_checkpoint": {
                    "sha256": selected,
                    "durable_locator": "owner/continuation-v8",
                }
            },
            "result_summary": {
                "selected_checkpoint_sha256": selected,
                "selected_metrics": {
                    "validation_accuracy": 0.98,
                    "validation_balanced_accuracy": 0.95,
                    "validation_macro_f1": 0.96,
                    "validation_nll": 0.1,
                },
            },
            "recovery_provenance": {
                "kind": "cryptographic_terminal_record_recovery",
                "scientific_training_reperformed": False,
                "optimizer_state_advanced": False,
                "surface_safety": {
                    "claim_basis": "frozen_training_runner_and_config_contract",
                    "checkpoint_identity_contains_surface_flags": False,
                    "allowed_surfaces": ["DS-V1-TRAIN", "DS-V1-VAL"],
                },
            },
        }
        self.assertEqual(persistence_profile(record), PROFILE_CONTINUATION_V8)

    def test_historical_profile_rejects_fake_generation_aware_inventory(self):
        experiment_id = "R04-MNV4-DIRECT-S1"
        spec = HISTORICAL_TRACKA_CLOSURE_SPECS[experiment_id]
        selected = spec["selected_checkpoint_sha256"]
        record = {
            "experiment_id": experiment_id,
            "status": "PASS",
            "run_id": spec["run_id"],
            "source_git_commit": spec["source_git_commit"],
            "continuation_required": False,
            "artifact_locators": {"selected_checkpoint": {"sha256": selected}},
            "result_summary": {
                "selected_checkpoint_sha256": selected,
                "selected_epoch": spec["selected_epoch"],
            },
            "persistence_status": {
                "backend": "generation_aware_v8",
                "files": ["checkpoint_index.json", "durable_sync.json"],
            },
        }
        with self.assertRaises(Exception):
            persistence_profile(record)

    def test_historical_availability_routes_only_to_legacy_store(self):
        experiment_id = "R04-MNV4-DIRECT-S1"
        spec = HISTORICAL_TRACKA_CLOSURE_SPECS[experiment_id]
        selected = spec["selected_checkpoint_sha256"]
        record = {
            "experiment_id": experiment_id,
            "status": "PASS",
            "run_id": spec["run_id"],
            "source_git_commit": spec["source_git_commit"],
            "continuation_required": False,
            "artifact_locators": {
                "selected_checkpoint": {
                    "sha256": selected,
                    "durable_locator": "owner/historical-private",
                }
            },
            "result_summary": {
                "selected_checkpoint_sha256": selected,
                "selected_epoch": spec["selected_epoch"],
            },
            "persistence_status": {
                "backend": "kaggle_private_dataset",
                "files": [
                    "checkpoint_index.json",
                    f"objects/selected.g00000001.{selected[:16]}.ckpt",
                ],
            },
        }
        calls = []
        class FakeStore:
            def restore(self, destination, *, run_id):
                calls.append(("legacy_restore", run_id))
                Path(destination).mkdir(parents=True, exist_ok=True)
                return True
        def legacy_factory(kind, locator):
            calls.append(("legacy_factory", kind, locator))
            return FakeStore()
        with tempfile.TemporaryDirectory() as td, patch.object(
            source_materialization,
            "shallow_verify_selected_checkpoint",
            return_value={
                "checkpoint_index_sha256": "1" * 64,
                "selected_checkpoint_sha256": selected,
                "selected_checkpoint_file_sha256": selected,
                "selected_checkpoint_epoch": spec["selected_epoch"],
                "selected_checkpoint_relative_path": "objects/selected.ckpt",
                "payload_deserialized": False,
                "scientific_metrics_opened": False,
            },
        ):
            result = restore_for_historical_availability(
                record=record,
                checkpoint_root=Path(td) / "checkpoint",
                verify_private=False,
                legacy_store_factory=legacy_factory,
            )
        self.assertEqual(result["profile"], PROFILE_HISTORICAL_LEGACY)
        self.assertFalse(result["payload_deserialized"])
        self.assertFalse(result["scientific_metrics_opened"])
        self.assertEqual(calls[0], ("legacy_factory", "kaggle-dataset", "owner/historical-private"))
        self.assertEqual(calls[1], ("legacy_restore", spec["run_id"]))

    def test_secondary_direct_specs_agree_with_canonical_closure_registry(self):
        for experiment_id, spec in HISTORICAL_SECONDARY_DIRECT_SPECS.items():
            self.assertEqual(spec["selected_checkpoint_sha256"], HISTORICAL_TRACKA_CLOSURE_SPECS[experiment_id]["selected_checkpoint_sha256"])
            self.assertEqual(spec["run_id"], HISTORICAL_TRACKA_CLOSURE_SPECS[experiment_id]["run_id"])


if __name__ == "__main__":
    unittest.main()
