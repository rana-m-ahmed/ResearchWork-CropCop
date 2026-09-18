from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.hashing import sha256_json  # noqa: E402
from cropcop_je.tracka_v12 import (  # noqa: E402
    AUTHORITY_ID,
    CLASS_MAP_SHA256,
    EXPERIMENT_SPECS,
    MANIFEST_SHA256,
    experiment_config_path,
)
from cropcop_je.tracka_v12_recovery import (  # noqa: E402
    SCIENCE_AUTHORIZATION_SHA256,
    SCIENCE_SOURCE_SHA,
    SCHEDULER_FREEZE_SHA256,
    TerminalRecoveryError,
    account_result,
    build_recovered_terminal_record,
    validate_checkpoint_payload,
    validate_recovered_terminal_record,
)

EXP = "R13-VIT-DLITTLE-DIFF-CONTEXT-S1"
RUN = "JE-R13-VIT-DLITTLE-DIFF-CONTEXT-S1-56023042e577-A01"
SELECTED = "b" * 64


def account_report():
    return {
        "status": "PASS",
        "science_complete": True,
        "science_source_sha": SCIENCE_SOURCE_SHA,
        "science_authorization_sha256": SCIENCE_AUTHORIZATION_SHA256,
        "scheduler_freeze_sha256": SCHEDULER_FREEZE_SHA256,
        "cross_gpu_gradient_synchronization": False,
        "child_git_credentials_removed": True,
        "slot_results": {
            "K2/GPU0": [
                {
                    "experiment_id": EXP,
                    "run_id": RUN,
                    "return_code": 0,
                    "run_status": "PASS",
                    "continuation_required": False,
                    "selected_checkpoint_sha256": SELECTED,
                }
            ]
        },
    }


class TrackAV12RecoveryTests(unittest.TestCase):
    def test_account_report_binds_terminal_result(self):
        row = account_result(account_report(), EXP)
        self.assertEqual(row["run_id"], RUN)
        self.assertEqual(row["selected_checkpoint_sha256"], SELECTED)

    def test_account_report_rejects_authorization_drift(self):
        report = account_report()
        report["science_authorization_sha256"] = "0" * 64
        with self.assertRaises(TerminalRecoveryError):
            account_result(report, EXP)

    def test_checkpoint_payload_can_recover_selection_metrics_without_original_run_record(self):
        config = json.loads((ROOT / experiment_config_path(EXP)).read_text(encoding="utf-8"))
        ctc = json.loads((ROOT / config["ctc_config"]).read_text(encoding="utf-8"))
        identity = {
            "experiment_id": EXP,
            "authority_id": AUTHORITY_ID,
            "source_git_commit": SCIENCE_SOURCE_SHA,
            "config_sha256": sha256_json(config),
            "ctc_v2_sha256": sha256_json(ctc),
            "manifest_sha256": MANIFEST_SHA256,
            "class_map_sha256": CLASS_MAP_SHA256,
            "seed": int(EXPERIMENT_SPECS[EXP]["seed"]),
            "student_init_sha256": "1" * 64,
            "pretrained_sha256": "2" * 64,
            "teacher_sha256": None,
            "teacher_factory_sha256": None,
            "teacher_factory_bundle_sha256": None,
            "software_stack_sha256": "3" * 64,
            "dependency_lock_sha256": "4" * 64,
            "g1_seal_sha256": "5" * 64,
            "g2_barrier_sha256": "6" * 64,
            "lane_id": f"TRACKA-V12:{EXP}",
            "allowed_surfaces": ["DS-V1-TRAIN", "DS-V1-VAL"],
            "v1_test_accessed": False,
            "external_protected_surface_accessed": False,
        }
        payload = {
            "identity": identity,
            "identity_sha256": sha256_json(identity),
            "epoch": 17,
            "optimizer_step": 20000,
            "selection_state": {
                "best": {
                    "epoch": 17,
                    "checkpoint_sha256": None,
                    "metrics": {
                        "validation_accuracy": 0.98,
                        "validation_balanced_accuracy": 0.95,
                        "validation_macro_f1": 0.96,
                        "validation_nll": 0.1,
                    },
                }
            },
        }
        selected_ref = {"sha256": SELECTED, "epoch": 17, "optimizer_step": 20000}
        evidence = validate_checkpoint_payload(
            repo_root=ROOT,
            experiment_id=EXP,
            run_id=RUN,
            selected_ref=selected_ref,
            payload=payload,
            expected_selected_sha256=SELECTED,
        )
        self.assertEqual(evidence["selected_checkpoint_sha256"], SELECTED)
        self.assertEqual(evidence["selected_epoch"], 17)
        self.assertEqual(evidence["selected_metrics"]["validation_macro_f1"], 0.96)

    def test_recovered_record_is_explicitly_recovery_not_retraining(self):
        checkpoint = {
            "identity": {
                "experiment_id": EXP,
                "source_git_commit": SCIENCE_SOURCE_SHA,
                "v1_test_accessed": False,
                "external_protected_surface_accessed": False,
            },
            "identity_sha256": "a" * 64,
            "selected_epoch": 17,
            "selected_metrics": {
                "validation_accuracy": 0.98,
                "validation_balanced_accuracy": 0.95,
                "validation_macro_f1": 0.96,
                "validation_nll": 0.1,
            },
            "selected_checkpoint_sha256": SELECTED,
            "checkpoint_index_sha256": "c" * 64,
            "selected_checkpoint_file_sha256": SELECTED,
            "run_id": RUN,
        }
        record = build_recovered_terminal_record(
            experiment_id=EXP,
            account_report=account_report(),
            checkpoint_evidence=checkpoint,
            account_report_sha256="d" * 64,
            durable_locator="owner/private-durable",
        )
        self.assertEqual(validate_recovered_terminal_record(record), [])
        self.assertFalse(record["recovery_provenance"]["scientific_training_reperformed"])
        self.assertEqual(record["artifact_locators"]["selected_checkpoint"]["sha256"], SELECTED)

    def test_recovered_record_rejects_missing_metric(self):
        checkpoint = {
            "identity": {
                "experiment_id": EXP,
                "source_git_commit": SCIENCE_SOURCE_SHA,
                "v1_test_accessed": False,
                "external_protected_surface_accessed": False,
            },
            "identity_sha256": "a" * 64,
            "selected_epoch": 17,
            "selected_metrics": {
                "validation_accuracy": 0.98,
                "validation_balanced_accuracy": 0.95,
                "validation_macro_f1": 0.96,
                "validation_nll": 0.1,
            },
            "selected_checkpoint_sha256": SELECTED,
            "checkpoint_index_sha256": "c" * 64,
            "selected_checkpoint_file_sha256": SELECTED,
            "run_id": RUN,
        }
        record = build_recovered_terminal_record(
            experiment_id=EXP,
            account_report=account_report(),
            checkpoint_evidence=checkpoint,
            account_report_sha256="d" * 64,
            durable_locator=None,
        )
        record["result_summary"]["selected_metrics"].pop("validation_nll")
        self.assertIn("selected_metrics:validation_nll", validate_recovered_terminal_record(record))


if __name__ == "__main__":
    unittest.main()
