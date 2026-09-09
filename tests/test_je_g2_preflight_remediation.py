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


    def test_14_principal_wrapper_hydrates_exact_locked_g2_summaries(self):
        source = (ROOT / "journal_extension/kaggle/generate_canonical_notebook.py").read_text(encoding="utf-8")
        self.assertIn('if EXECUTION_PHASE == "principal-dual":', source)
        self.assertIn('_required_g2 = ("CAL-MNV4-DIRECT", "CAL-MNV4-TEACHER", "CAL-CNXTT")', source)
        self.assertIn('authenticated-github-contents-api', source)
        self.assertIn('Principal G2 authenticated evidence hydration: PASS', source)
        self.assertNotIn('_run_lane.collect_g2_summaries_from_evidence_branches(_shared_g2)', source)

    def test_15_principal_hydration_precedes_frozen_envelope_launch(self):
        source = (ROOT / "journal_extension/kaggle/generate_canonical_notebook.py").read_text(encoding="utf-8")
        hydrate = source.index('Principal G2 authenticated evidence hydration: PASS')
        launch = source.index('_envelope_cp = _run_frozen_envelope_with_parent_emergency_guard()')
        self.assertLess(hydrate, launch)
        self.assertIn('repo_workdir / "journal_extension/kaggle/run_envelope.py"', source)

    def test_16_principal_envelopes_keep_seed_pairs_and_slots_exact(self):
        expected = {
            "P1_S1_PAIR.json": ("K1", "R04-MNV4-DIRECT-S1", "R05-MNV4-TEACHER-S1"),
            "P2_S2_PAIR.json": ("K2", "R04-MNV4-DIRECT-S2", "R05-MNV4-TEACHER-S2"),
            "P3_S3_PAIR.json": ("K3", "R04-MNV4-DIRECT-S3", "R05-MNV4-TEACHER-S3"),
        }
        root = ROOT / "journal_extension/kaggle/envelopes"
        for name, (lane, direct, teacher) in expected.items():
            payload = json.loads((root / name).read_text(encoding="utf-8"))
            self.assertEqual(payload["phase"], "principal-dual")
            self.assertEqual(payload["hardware_profile"], "T4X2")
            self.assertEqual(
                [(row["lane"], row["experiment_id"], row["slot"]) for row in payload["children"]],
                [(lane, direct, 0), (lane, teacher, 1)],
            )


    def test_17_principal_handoff_pins_qualified_g2_hashes(self):
        source = (ROOT / "journal_extension/kaggle/generate_canonical_notebook.py").read_text(encoding="utf-8")
        self.assertIn("3be3ef666b1e9b479e0173cd74909c43023093e320bd0b3f2731f11e97d21a0a", source)
        self.assertIn("c89293bc4d0390737160b9aa39cd8aa973d5a54da3b7b83d8c21056e1942106d", source)
        self.assertIn("8439e003934d58d0d278145376fc452a7afbb93c599a34146f5e545cd6a65260", source)
        self.assertIn("00d3518b3229ab16786d5d97d19ce0e16c966930740b2bf432ee81bc059a06e0", source)
        self.assertIn("_validate_g2_barrier_object(", source)
        self.assertIn("_validate_calibration_summary(_summary)", source)
        self.assertIn("_sha256_json(_summary)", source)

    def test_18_principal_handoff_does_not_silence_transport_failure(self):
        source = (ROOT / "journal_extension/kaggle/generate_canonical_notebook.py").read_text(encoding="utf-8")
        self.assertIn("GitHub evidence fetch failed with HTTP", source)
        self.assertIn("GitHub evidence fetch network failure", source)
        self.assertIn("GitHub evidence response identity/encoding mismatch", source)
        self.assertNotIn("if not _git_fetch_branch(branch):\\n            continue", source)

    def test_20_principal_continuation_is_explicit_and_fail_closed(self):
        source = (ROOT / "journal_extension/kaggle/generate_canonical_notebook.py").read_text(encoding="utf-8")
        self.assertIn('CROPCOP_CONTINUATION_POLICY', source)
        self.assertIn('Continuation requires explicit CROPCOP_ENVELOPE_INPUT_ROOT', source)
        self.assertIn('fresh_restart_possible_after_preflight": False', source)
        self.assertIn('Principal continuation checkpoint recovery/prestage: PASS', source)
        self.assertNotIn('Path("/kaggle/input").rglob', source)

    def test_21_continuation_recovery_prefers_durable_then_exact_prior_checkpoint(self):
        source = (ROOT / "journal_extension/kaggle/generate_canonical_notebook.py").read_text(encoding="utf-8")
        durable = source.index('_restore_verified_durable(')
        rescue = source.index('_store.sync(', durable)
        prestage = source.index('_export_recovery_bundle(_verified_root, _new_checkpoint_root)', rescue)
        self.assertLess(durable, rescue)
        self.assertLess(rescue, prestage)
        self.assertIn('_recover_latest(', source)
        self.assertIn('prior run record surface contract changed', source)
        self.assertIn('durable rescue changed checkpoint identity/progress', source)

    def test_22_parent_guard_preserves_child_safe_deadline_and_delays_only_emergency_cutoff(self):
        source = (ROOT / "journal_extension/kaggle/generate_canonical_notebook.py").read_text(encoding="utf-8")
        self.assertIn('def _run_frozen_envelope_with_parent_emergency_guard()', source)
        self.assertIn('return self._budget.remaining_hard_seconds', source)
        self.assertIn('_module.SessionBudget = _ParentSessionBudgetProxy', source)
        self.assertIn('child_safe_deadline_unchanged=true', source)
        self.assertIn('parent_emergency_cutoff=hard_limit_minus_300s', source)

    def test_23_continuation_remediation_does_not_edit_frozen_scientific_runner(self):
        source = (ROOT / "journal_extension/kaggle/generate_canonical_notebook.py").read_text(encoding="utf-8")
        self.assertIn('scientific_configuration_changed": False', source)
        self.assertIn('scientific_source_modified": False', source)
        self.assertIn('AUTHORIZED_SOURCE_SHA = "f171309fc7e9dc22241ecc137ebbb8e4bcdc5433"', source)
        self.assertIn('importlib.util.spec_from_file_location(', source)

    def test_19_canonical_notebook_contains_same_hardened_handoff(self):
        notebook = json.loads(
            (ROOT / "journal_extension/kaggle/canonical_lane.ipynb").read_text(encoding="utf-8")
        )
        code = "\\n".join(
            "".join(cell.get("source", [])) if isinstance(cell.get("source"), list)
            else str(cell.get("source", ""))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertIn("authenticated-github-contents-api", code)
        self.assertIn("Principal G2 authenticated evidence hydration: PASS", code)
        self.assertIn("3be3ef666b1e9b479e0173cd74909c43023093e320bd0b3f2731f11e97d21a0a", code)
        self.assertNotIn("_run_lane.collect_g2_summaries_from_evidence_branches(_shared_g2)", code)


if __name__ == "__main__":
    unittest.main()
