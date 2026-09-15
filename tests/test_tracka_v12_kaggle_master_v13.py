from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "journal_extension" / "kaggle" / "tracka_v12_ops"
SRC = ROOT / "journal_extension" / "src"
for path in (OPS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import master_science_v13 as science  # noqa: E402


class TrackAV12MasterV13Tests(unittest.TestCase):
    def test_exact_initial_readme_is_pristine_first_run(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "README.txt").write_text(science.BOOTSTRAP_README, encoding="utf-8")
            self.assertEqual(
                science._classify_downloaded_generation(root, version=1, locator="owner/dataset"),
                "PRISTINE_FIRST_RUN",
            )

    def test_version_advanced_without_recovery_generation_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "README.txt").write_text(science.BOOTSTRAP_README, encoding="utf-8")
            with self.assertRaisesRegex(Exception, "no recovery marker/index"):
                science._classify_downloaded_generation(root, version=2, locator="owner/dataset")

    def test_partial_recovery_generation_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "checkpoint_index.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(Exception, "partial durable recovery generation"):
                science._classify_downloaded_generation(root, version=1, locator="owner/dataset")

    def test_unexpected_initial_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "README.txt").write_text(science.BOOTSTRAP_README, encoding="utf-8")
            (root / "unexpected.bin").write_bytes(b"x")
            with self.assertRaisesRegex(Exception, "unexpected files"):
                science._classify_downloaded_generation(root, version=1, locator="owner/dataset")

    def test_existing_marker_and_index_remain_auto_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "durable_sync.json").write_text("{}\n", encoding="utf-8")
            (root / "checkpoint_index.json").write_text("{}\n", encoding="utf-8")
            self.assertEqual(
                science._classify_downloaded_generation(root, version=1, locator="owner/dataset"),
                "AUTO_RESUME",
            )

    def test_failure_tail_is_bounded_and_redacts_named_secrets(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "console.log"
            prefix = "x" * (science.MAX_DIAGNOSTIC_CHARS + 1000)
            path.write_text(prefix + "\nKAGGLE_KEY=secret-value\n", encoding="utf-8")
            tail = science._tail_text(path)
            self.assertLessEqual(len(tail), science.MAX_DIAGNOSTIC_CHARS + 64)
            self.assertNotIn("secret-value", tail)
            self.assertIn("[REDACTED]", tail)

    def test_runtime_wrapper_switches_only_qualified_auto_resume(self):
        text = (OPS / "science_account_runner_v13.py").read_text(encoding="utf-8")
        self.assertIn('command[resume_index] != "auto"', text)
        self.assertIn('command[resume_index] = "never"', text)
        self.assertIn("experiment_id in pristine", text)
        self.assertIn("run_tracka_v12_account_v121", text)

    def test_v13_driver_preserves_v12_control_bridge_and_rebinds_science(self):
        text = (OPS / "master_account_driver_v13.py").read_text(encoding="utf-8")
        self.assertIn("master_account_driver_v12", text)
        self.assertIn("base_v11.run_science = run_science", text)

    def test_v13_guard_has_fresh_status_namespace_and_driver(self):
        text = (OPS / "master_launch_guard_v13.py").read_text(encoding="utf-8")
        self.assertIn(".cropcop_tracka_master_guard_v13", text)
        self.assertIn("master_account_driver_v13.py", text)
        self.assertIn('STATUS_SCHEMA = "3.4"', text)

    def test_v13_does_not_modify_scientific_source_or_protected_surface_contract(self):
        account_wrapper = (OPS / "science_account_runner_v13.py").read_text(encoding="utf-8")
        science_wrapper = (OPS / "master_science_v13.py").read_text(encoding="utf-8")
        joined = account_wrapper + science_wrapper
        self.assertNotIn("DS-V1-TEST-CONSUMED", joined)
        self.assertNotIn("DS-EXT-", joined)
        self.assertNotIn("optimizer", account_wrapper.lower())
        self.assertNotIn("objective", account_wrapper.lower())
        self.assertNotIn("seed", account_wrapper.lower())


if __name__ == "__main__":
    unittest.main()
