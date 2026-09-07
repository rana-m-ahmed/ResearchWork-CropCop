from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.frozen_v1_manifest import (
    FROZEN_V1_COLUMNS,
    load_frozen_v1_rows,
    validate_frozen_v1_column_contract,
)
from cropcop_je.persistence import validate_durable_access_plan


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FrozenV1ManifestRemediationTests(unittest.TestCase):
    def _fixture(self, root: Path):
        labels = {f"class_{i:03d}": i for i in range(120)}
        class_map = root / "class_to_idx.json"
        class_map.write_text(json.dumps(labels, sort_keys=True) + "\n", encoding="utf-8")
        manifest = root / "final_manifest.csv"
        manifest.write_text(
            "record_key,source,relative_path,filename,label,original_label,sha256,format,mode,width,height,file_bytes,"
            "leakage_group,dedup_representative,split,cv_fold,visual_label_audited,knn_label_agreement,"
            "nearest_neighbor_label,nearest_neighbor_similarity,flag_visual_label_outlier_v5,portable_relpath\n"
            "rk-train,main,legacy/train.jpg,train.jpg,class_007,class_007,"
            + "a" * 64
            + ",JPG,RGB,10,10,100,lg1,legacy/train.jpg,train,0,True,1.0,class_007,1.0,False,"
            "train/class_007/train.jpg\n"
            "rk-val,main,legacy/val.jpg,val.jpg,class_009,class_009,"
            + "b" * 64
            + ",JPG,RGB,10,10,100,lg2,legacy/val.jpg,val,0,True,1.0,class_009,1.0,False,"
            "val/class_009/val.jpg\n",
            encoding="utf-8",
        )
        return manifest, class_map

    def test_01_certified_schema_contract_is_exact(self):
        self.assertEqual(
            FROZEN_V1_COLUMNS,
            {
                "row_id": "record_key",
                "relative_path": "portable_relpath",
                "split": "split",
                "label": "label",
            },
        )

    def test_02_targets_are_derived_from_locked_class_map(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest, class_map = self._fixture(root)
            rows = load_frozen_v1_rows(
                manifest,
                class_map,
                expected_manifest_sha256=_sha(manifest),
                expected_class_map_sha256=_sha(class_map),
                surface="DS-V1-TRAIN",
                expected_count=1,
            )
            self.assertEqual(rows[0].stable_row_id, "rk-train")
            self.assertEqual(rows[0].relative_path, "train/class_007/train.jpg")
            self.assertEqual(rows[0].class_index, 7)

    def test_03_numeric_manifest_class_index_is_forbidden(self):
        with self.assertRaises(ValueError):
            validate_frozen_v1_column_contract(
                row_id_column="record_key",
                path_column="portable_relpath",
                split_column="split",
                label_column="label",
                class_index_column="cv_fold",
            )

    def test_04_historical_relative_path_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_frozen_v1_column_contract(
                row_id_column="record_key",
                path_column="relative_path",
                split_column="split",
                label_column="label",
            )

    def test_05_non_bijective_class_map_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest, class_map = self._fixture(root)
            bad = {f"class_{i:03d}": (0 if i == 119 else i) for i in range(120)}
            class_map.write_text(json.dumps(bad, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_frozen_v1_rows(
                    manifest,
                    class_map,
                    expected_manifest_sha256=_sha(manifest),
                    expected_class_map_sha256=_sha(class_map),
                    surface="DS-V1-TRAIN",
                    expected_count=1,
                )

    def test_06_lane_uses_label_column_not_class_index_column(self):
        source = (ROOT / "journal_extension/kaggle/run_lane.py").read_text(encoding="utf-8")
        self.assertIn('"label_column": "CROPCOP_LABEL_COLUMN"', source)
        self.assertNotIn('"class_index_column": "CROPCOP_CLASS_INDEX_COLUMN"', source)

    def test_07_training_parser_keeps_legacy_index_override_fail_closed(self):
        source = (ROOT / "journal_extension/scripts/run_training.py").read_text(encoding="utf-8")
        self.assertIn('ap.add_argument("--label-column", required=True)', source)
        self.assertIn('ap.add_argument("--class-index-column", default="")', source)
        self.assertIn("load_frozen_v1_rows(", source)

    def test_08_cnxtt_uses_same_frozen_manifest_adapter(self):
        source = (ROOT / "journal_extension/scripts/calibrate_cnxtt.py").read_text(encoding="utf-8")
        self.assertIn("load_frozen_v1_rows(", source)
        self.assertIn('ap.add_argument("--label-column", required=True)', source)

    def test_09_private_durable_target_is_required(self):
        resolved = {"CAL": "owner/cropcop-je-cal"}
        env = {"KAGGLE_USERNAME": "owner", "KAGGLE_KEY": "fixture"}
        with patch(
            "cropcop_je.g1_publication.preflight_private_target",
            side_effect=RuntimeError("target is not private"),
        ), patch(
            "cropcop_je.persistence.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
        ):
            report = validate_durable_access_plan("kaggle-dataset", resolved, env=env)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(
            any("private-target preflight failed" in e for e in report["errors"])
        )

    def test_10_private_durable_target_and_cli_read_pass(self):
        resolved = {"CAL": "owner/cropcop-je-cal"}
        env = {"KAGGLE_USERNAME": "owner", "KAGGLE_KEY": "fixture"}
        with patch(
            "cropcop_je.g1_publication.preflight_private_target",
            return_value={
                "status": "PASS",
                "authoritative_is_private": True,
                "current_version_number": 1,
            },
        ), patch(
            "cropcop_je.persistence.subprocess.run",
            return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
        ):
            report = validate_durable_access_plan("kaggle-dataset", resolved, env=env)
        self.assertEqual(report["status"], "PASS", report["errors"])
        self.assertTrue(report["checks"]["CAL"]["authoritative_is_private"])
        self.assertTrue(report["checks"]["CAL"]["authenticated_read"])

    def test_11_first_run_envelope_publication_assignment_is_reachable(self):
        source = (ROOT / "journal_extension/kaggle/run_envelope.py").read_text(encoding="utf-8")
        guard = source.index(
            'if prior_bundle is not None and prior_control_before != _prior_control_fingerprint(prior_bundle):'
        )
        publish = source.index(
            'envelope_branch = _publish(envelope_id, source_sha, [manifest_path, evidence_path, state_path])',
            guard,
        )
        evidence = source.index('evidence["envelope_publication_branch"] = envelope_branch', publish)
        block = source[guard:publish]
        self.assertIn('raise EnvelopeError("attached prior envelope control files changed during continuation")', block)
        line_start = source.rfind("\n", 0, publish) + 1
        publish_line = source[line_start:source.find("\n", publish)]
        self.assertEqual(
            publish_line,
            "    envelope_branch = _publish(envelope_id, source_sha, [manifest_path, evidence_path, state_path])",
        )
        self.assertLess(publish, evidence)


class OperatorRecoveryHardeningTests(unittest.TestCase):
    def test_12_canonical_wrapper_fails_early_when_cnxtt_path_is_missing(self):
        source = (ROOT / "journal_extension/kaggle/generate_canonical_notebook.py").read_text(encoding="utf-8")
        self.assertIn('CROPCOP_CNXTT_PRETRAINED is required for G2 calibration and P3 principal.', source)
        self.assertIn('CROPCOP_CNXTT_PRETRAINED must point to convnext_tiny-983f1562.pth', source)

    def test_13_wrapper_repairs_only_terminal_envelope_publication_tail(self):
        source = (ROOT / "journal_extension/kaggle/generate_canonical_notebook.py").read_text(encoding="utf-8")
        self.assertIn('Envelope terminal publication repair: PASS', source)
        self.assertIn('_state.get("state") in {"PASS", "CONTINUATION_REQUIRED"}', source)
        self.assertIn('files=[_evidence_path]', source)
        self.assertIn('scientific_execution_relaunched=false', source)


if __name__ == "__main__":
    unittest.main()
