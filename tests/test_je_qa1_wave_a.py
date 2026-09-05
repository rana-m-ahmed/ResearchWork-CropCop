import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "journal_extension" / "src"))
sys.path.insert(0, str(ROOT / "journal_extension" / "scripts"))

from cropcop_je.envelope import AMENDMENT_ID, AMENDMENT_SHA256
from cropcop_je.smoke_handoff import (
    SmokeHandoffError,
    locate_exact_evidence_file,
    validate_terminal_dual_gpu_smoke_evidence,
)
from validate_operator_docs import validate as validate_operator_docs

SOURCE = "a" * 40
DEP = "b" * 64
SMOKE_B_SHA = "c" * 64


def valid_dual():
    return {
        "schema_version": "1.0",
        "status": "PASS",
        "qualification_id": "MGPU-DUAL-SMOKE-T4X2-V1",
        "scientific": False,
        "synthetic_unprotected_data_only": True,
        "source_git_sha": SOURCE,
        "dependency_lock_sha256": DEP,
        "amendment_id": AMENDMENT_ID,
        "amendment_sha256": AMENDMENT_SHA256,
        "smoke_b_evidence_sha256": SMOKE_B_SHA,
        "kaggle_run_type": "Batch",
        "notebook_started_monotonic": 100.0,
        "parent_gpu_inventory": [
            {"index": 0, "uuid": "GPU-A", "name": "Tesla T4", "memory_total_mib": 15360},
            {"index": 1, "uuid": "GPU-B", "name": "Tesla T4", "memory_total_mib": 15360},
        ],
        "children": [
            {
                "child_id": "DUAL-SMOKE-A",
                "requested_physical_slot": 0,
                "physical_gpu_uuid": "GPU-A",
                "visible_cuda_device_count": 1,
                "visible_gpu_name": "Tesla T4",
                "optimizer_step": 4,
                "checkpoint_sha256": "d" * 64,
                "checkpoint_bytes": 100,
                "notebook_started_monotonic": 100.0,
                "git_credentials_present_in_child": False,
            },
            {
                "child_id": "DUAL-SMOKE-B",
                "requested_physical_slot": 1,
                "physical_gpu_uuid": "GPU-B",
                "visible_cuda_device_count": 1,
                "visible_gpu_name": "Tesla T4",
                "optimizer_step": 4,
                "checkpoint_sha256": "e" * 64,
                "checkpoint_bytes": 101,
                "notebook_started_monotonic": 100.0,
                "git_credentials_present_in_child": False,
            },
        ],
        "overlap_duration_seconds": 1.5,
        "no_output_collision": True,
        "no_git_child_publication": True,
        "common_session_clock": True,
        "parent_finalized_both": True,
        "science_diff_status": "PASS",
        "restricted_cropcop_data_accessed": False,
        "g1_executed": False,
        "g2_executed": False,
        "r04_r05_executed": False,
        "errors": [],
        "git_publication_status": "PASS",
        "public_safe_evidence_branch": "run-evidence/DUAL-GPU-SMOKE",
    }


def validate(evidence, **overrides):
    kwargs = {
        "expected_source_sha": SOURCE,
        "expected_dependency_lock_sha256": DEP,
        "expected_amendment_id": AMENDMENT_ID,
        "expected_amendment_sha256": AMENDMENT_SHA256,
        "expected_smoke_b_evidence_sha256": SMOKE_B_SHA,
        "require_batch": True,
    }
    kwargs.update(overrides)
    return validate_terminal_dual_gpu_smoke_evidence(evidence, **kwargs)


class QA1WaveATests(unittest.TestCase):
    def test_01_valid_dual_smoke_accepted(self):
        self.assertEqual(validate(valid_dual()), [])

    def test_02_wrong_source_rejected(self):
        self.assertTrue(validate(valid_dual(), expected_source_sha="9" * 40))

    def test_03_wrong_dependency_rejected(self):
        self.assertTrue(validate(valid_dual(), expected_dependency_lock_sha256="9" * 64))

    def test_04_wrong_amendment_rejected(self):
        self.assertTrue(validate(valid_dual(), expected_amendment_sha256="9" * 64))

    def test_05_interactive_rejected(self):
        row = valid_dual()
        row["kaggle_run_type"] = "Interactive"
        self.assertTrue(validate(row))

    def test_06_failed_dual_smoke_rejected(self):
        row = valid_dual()
        row["status"] = "FAIL"
        self.assertTrue(validate(row))

    def test_07_invalid_child_count_rejected(self):
        row = valid_dual()
        row["children"] = row["children"][:1]
        self.assertTrue(validate(row))

    def test_08_duplicate_gpu_uuid_rejected(self):
        row = valid_dual()
        row["parent_gpu_inventory"][1]["uuid"] = "GPU-A"
        row["children"][1]["physical_gpu_uuid"] = "GPU-A"
        self.assertTrue(validate(row))

    def test_09_no_overlap_rejected(self):
        row = valid_dual()
        row["overlap_duration_seconds"] = 0
        self.assertTrue(validate(row))

    def test_10_missing_publication_rejected(self):
        row = valid_dual()
        row["git_publication_status"] = "FAIL"
        self.assertTrue(validate(row))

    def test_11_smoke_b_digest_binding_rejected_on_drift(self):
        self.assertTrue(validate(valid_dual(), expected_smoke_b_evidence_sha256="9" * 64))

    def test_12_exact_root_lookup_requires_exactly_one(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "x").mkdir()
            p = root / "x" / "SMOKE_B_EVIDENCE.json"
            p.write_text("{}")
            self.assertEqual(
                locate_exact_evidence_file(root, "SMOKE_B_EVIDENCE.json", label="Smoke-B"),
                p.resolve(),
            )
            (root / "y").mkdir()
            (root / "y" / "SMOKE_B_EVIDENCE.json").write_text("{}")
            with self.assertRaises(SmokeHandoffError):
                locate_exact_evidence_file(root, "SMOKE_B_EVIDENCE.json", label="Smoke-B")

    def test_13_g1_requires_dual_smoke_environment_before_artifact_work(self):
        path = ROOT / "journal_extension/kaggle/run_g1.py"
        spec = importlib.util.spec_from_file_location("qa1_run_g1", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        env = {
            "KAGGLE_KERNEL_RUN_TYPE": "Batch",
            "CROPCOP_SOURCE_GIT_COMMIT": SOURCE,
            "CROPCOP_INFRA_SMOKE_EVIDENCE": "/tmp/smoke-b.json",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "CROPCOP_DUAL_GPU_SMOKE_EVIDENCE"):
                module.main()

    def test_14_envelope_dual_preflight_requires_path(self):
        path = ROOT / "journal_extension/kaggle/run_envelope.py"
        spec = importlib.util.spec_from_file_location("qa1_run_envelope", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(Exception, "CROPCOP_DUAL_GPU_SMOKE_EVIDENCE"):
                module._dual_smoke_preflight(SOURCE, {"dependency_lock_sha256": DEP}, {})

    def test_15_wrapper_has_explicit_handoff_roots_and_no_global_input_search(self):
        generator = (ROOT / "journal_extension/kaggle/generate_canonical_notebook.py").read_text()
        self.assertIn("CROPCOP_SMOKE_B_INPUT_ROOT", generator)
        self.assertIn("CROPCOP_DUAL_GPU_SMOKE_INPUT_ROOT", generator)
        self.assertIn("must contain exactly one", generator)
        self.assertNotIn('Path("/kaggle/input").rglob', generator)

    def test_16_active_operator_docs_are_synchronized(self):
        report = validate_operator_docs(ROOT)
        self.assertEqual(report["status"], "PASS", report["errors"])

    def test_17_active_docs_do_not_name_superseded_source(self):
        old = "67370145c9104edd52330b788c3b41b28f5cab87"
        self.assertNotIn(old, (ROOT / "journal_extension/kaggle/README_SMOKE.md").read_text())
        self.assertNotIn(old, (ROOT / "journal_extension/kaggle/smoke_inputs.example.json").read_text())

    def test_18_seal_g1_has_terminal_dual_smoke_gate_before_model_creation(self):
        source = (ROOT / "journal_extension/scripts/seal_g1.py").read_text()
        self.assertIn("--dual-gpu-smoke-evidence", source)
        self.assertLess(
            source.index("validate_terminal_dual_gpu_smoke_evidence("),
            source.index("create_student_from_pretrained("),
        )


if __name__ == "__main__":
    unittest.main()
