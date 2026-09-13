from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "journal_extension" / "kaggle" / "tracka_v12_ops"
SRC = ROOT / "journal_extension" / "src"
for path in (OPS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import tracka_v12_kaggle_operator_v3 as v3
import master_preflight
from master_attestations import verified_attestation_paths
from master_control import CONTROL_FILES
from master_g2a import REQUIRED_G2A, g2a_profile_command
from master_science import science_command


class MasterOperatorV3Tests(unittest.TestCase):
    def test_frozen_science_identity(self):
        self.assertEqual(v3.SCIENCE_SHA, "9a72e9466a9a3e7429e0e36a028edac662f83146")
        self.assertEqual(v3.MASTER_ACCOUNTS, ("K1", "K2", "K3"))

    def test_g2a_coverage_is_exact_and_uses_five_of_six_gpus(self):
        rows = [row for account in v3.MASTER_ACCOUNTS for row in v3.G2A_ACCOUNT_PROFILES[account]]
        ids = [row[0] for row in rows]
        slots = [row[2] for row in rows]
        self.assertEqual(set(ids), REQUIRED_G2A)
        self.assertEqual(len(ids), 5)
        self.assertEqual(len(set(ids)), 5)
        self.assertEqual(len(set(slots)), 5)
        self.assertEqual(set(slots), {"K1/GPU0", "K1/GPU1", "K2/GPU0", "K2/GPU1", "K3/GPU0"})

    def test_public_coordination_ids_are_source_bound(self):
        for kind in ("G1A-HANDOFF", "CONTROL", "ACCOUNT-K1", "G2A-CAL-R13"):
            run_id = v3.public_run_id(kind)
            self.assertTrue(run_id.endswith(v3.SCIENCE_SHA[:12]))
            self.assertNotIn("/", run_id)

    def test_packaged_exact_head_attestations(self):
        code, lock = verified_attestation_paths()
        code_payload = v3.load_json(code)
        lock_payload = v3.load_json(lock)
        self.assertEqual(code_payload["attestation_kind"], "track_a_v12_pre_science_code")
        self.assertEqual(code_payload["pull_request_head_sha"], v3.SCIENCE_SHA)
        self.assertEqual(code_payload["source_git_commit"], v3.SCIENCE_SHA)
        self.assertEqual(lock_payload["attestation_kind"], "track_a_v12_exact_head_lock_runtime")
        self.assertEqual(lock_payload["pull_request_head_sha"], v3.SCIENCE_SHA)
        self.assertEqual(lock_payload["source_git_commit"], v3.SCIENCE_SHA)
        self.assertFalse(lock_payload["science_authorized"])

    def test_control_inventory_is_minimal_public_coordination_bundle(self):
        self.assertEqual(
            CONTROL_FILES,
            [
                "TRACKA_V12_G2A_BARRIER.json",
                "TRACKA_V12_SCHEDULER_FREEZE.json",
                "TRACKA_V12_SCIENCE_GO.json",
                "TRACKA_V12_DURABLE_MAP.json",
                "TRACKA_V12_ACCOUNT_OWNERS.json",
                "TRACKA_V12_CONTROL_PUBLIC_REPORT.json",
            ],
        )

    def test_science_command_has_single_bindings_and_auto_publication(self):
        cmd = science_command(
            Path("/repo"), account_id="K1", manifest=Path("/manifest.csv"),
            class_map=Path("/class.json"), image_root=Path("/images"),
            g1a_bundle=Path("/g1a"), control_dir=Path("/control"),
            master_root=Path("/master"),
        )
        singleton_flags = (
            "--account-id", "--repo-root", "--source-git-commit", "--manifest", "--class-map",
            "--image-root", "--g1a-bundle", "--g2a-barrier", "--scheduler-freeze",
            "--science-authorization", "--durable-map", "--output-root",
        )
        for flag in singleton_flags:
            self.assertEqual(cmd.count(flag), 1, flag)
        self.assertIn("--publish-evidence", cmd)
        self.assertEqual(cmd[cmd.index("--source-git-commit") + 1], v3.SCIENCE_SHA)

    def test_g2a_command_is_non_scientific_and_durable(self):
        cmd = g2a_profile_command(
            repo=Path("/repo"), calibration_id="CAL-R13",
            experiment_id="R13-VIT-DLITTLE-DIFF-CONTEXT-S1", slot_id="K3/GPU0",
            manifest=Path("/manifest.csv"), class_map=Path("/class.json"),
            image_root=Path("/images"), g1a_bundle=Path("/g1a"),
            output_dir=Path("/out"), summary_path=Path("/summary.json"),
            durable_locator="owner/private-dataset",
        )
        self.assertEqual(cmd[cmd.index("--mode") + 1], "calibration")
        self.assertEqual(cmd[cmd.index("--durable-store-kind") + 1], "kaggle-dataset")
        self.assertIn("--durable-required", cmd)
        self.assertEqual(cmd.count("--g1a-bundle"), 1)
        self.assertEqual(cmd.count("--summary-out"), 1)

    def test_rollover_classifier_accepts_only_explicit_technical_continuation(self):
        valid = {
            "status": "ATTENTION_REQUIRED",
            "science_complete": False,
            "worker_errors": {},
            "slot_results": {
                "K1/GPU0": [{
                    "return_code": 0,
                    "run_status": "LAUNCHED",
                    "continuation_required": True,
                    "session_rollover_required": True,
                }],
                "K1/GPU1": [{
                    "return_code": None,
                    "run_status": None,
                    "status": "NOT_STARTED_SESSION_BUDGET",
                }],
            },
        }
        self.assertTrue(v3.technical_rollover_only(valid))
        malformed = dict(valid)
        malformed["slot_results"] = {}
        self.assertFalse(v3.technical_rollover_only(malformed))
        failed = dict(valid)
        failed["slot_results"] = {"K1/GPU0": [{"return_code": 1, "run_status": "FAIL"}]}
        self.assertFalse(v3.technical_rollover_only(failed))
        worker_error = dict(valid)
        worker_error["worker_errors"] = {"K1/GPU0": {"type": "RuntimeError"}}
        self.assertFalse(v3.technical_rollover_only(worker_error))

    def test_locked_stack_reuses_exact_environment_before_install(self):
        expected = {"dependency_lock_sha256": "a" * 64}
        with mock.patch.object(v3, "verify_locked_stack", return_value=expected) as verify, \
             mock.patch.object(v3, "install_locked_stack") as install:
            self.assertEqual(v3.ensure_locked_stack("/repo"), expected)
            verify.assert_called_once()
            install.assert_not_called()

    def test_locked_stack_repairs_once_then_reverifies(self):
        expected = {"dependency_lock_sha256": "b" * 64}
        with mock.patch.object(
            v3, "verify_locked_stack", side_effect=[v3.OperatorError("drift"), expected]
        ) as verify, mock.patch.object(v3, "install_locked_stack") as install:
            self.assertEqual(v3.ensure_locked_stack("/repo"), expected)
            self.assertEqual(verify.call_count, 2)
            install.assert_called_once_with("/repo")

    def test_publication_retries_transient_failure(self):
        import cropcop_je.publication as publication
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "evidence.json"
            f.write_text("{}", encoding="utf-8")
            with mock.patch.object(v3, "load_github_token", return_value="token"), \
                 mock.patch.object(publication, "publish_to_github_branch", side_effect=[RuntimeError("transient"), "run-evidence/x"]) as pub, \
                 mock.patch.object(v3.time, "sleep") as sleep:
                branch = v3.publish_public_files(td, "X", [f], attempts=2)
                self.assertEqual(branch, "run-evidence/x")
                self.assertEqual(pub.call_count, 2)
                sleep.assert_called_once_with(5)

    def test_dataset_slug_is_bounded_and_deterministic(self):
        identities = [row[1] for account in v3.MASTER_ACCOUNTS for row in v3.G2A_ACCOUNT_PROFILES[account]]
        slugs = [v3.kaggle_safe_dataset_slug("cropcop", identity) for identity in identities]
        self.assertEqual(len(slugs), len(set(slugs)))
        self.assertTrue(all(3 <= len(slug) <= 50 for slug in slugs))
        self.assertEqual(slugs[0], v3.kaggle_safe_dataset_slug("cropcop", identities[0]))

    def _fake_manifest(self, path: Path) -> list[str]:
        rels = [f"train/class-a/image-{index}.jpg" for index in range(16)]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("portable_relpath\n" + "\n".join(rels) + "\n", encoding="utf-8")
        return rels

    def test_duplicate_exact_manifest_selects_structurally_nearest_dataset_copy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            package = root / "datasets" / "owner" / "cropcop"
            actual_root = package / "CropCop_Final_v1"
            report_root = package / "CropCop_Final_v1_CERTIFICATION_REPORTS"
            actual_manifest = actual_root / "audit" / "final_manifest.csv"
            report_manifest = report_root / "audit" / "final_manifest.csv"
            rels = self._fake_manifest(actual_manifest)
            report_manifest.parent.mkdir(parents=True, exist_ok=True)
            report_manifest.write_bytes(actual_manifest.read_bytes())
            for rel in rels:
                image = actual_root / rel
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(b"x")
            actual_class = actual_root / "audit" / "classes.json"
            report_class = report_root / "audit" / "classes.json"
            actual_class.write_text("{}\n", encoding="utf-8")
            report_class.write_bytes(actual_class.read_bytes())

            real_sha = master_preflight.sha256_file
            def frozen_sha(path):
                path = Path(path)
                if path.name == "final_manifest.csv":
                    return v3.MANIFEST_SHA256
                if path.name == "classes.json":
                    return v3.CLASS_MAP_SHA256
                return real_sha(path)

            with mock.patch.object(master_preflight, "sha256_file", side_effect=frozen_sha):
                manifest, class_map, image_root = master_preflight.resolve_master_inputs("K2", root)
            self.assertEqual(manifest, actual_manifest.resolve())
            self.assertEqual(class_map, actual_class.resolve())
            self.assertEqual(image_root, actual_root.resolve())

    def test_conflicting_equally_qualified_manifest_roots_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            roots = [root / "datasets" / "a" / "d1", root / "datasets" / "b" / "d2"]
            manifests = []
            for data_root in roots:
                manifest = data_root / "audit" / "final_manifest.csv"
                rels = self._fake_manifest(manifest)
                for rel in rels:
                    image = data_root / rel
                    image.parent.mkdir(parents=True, exist_ok=True)
                    image.write_bytes(b"x")
                manifests.append(manifest)
            real_sha = master_preflight.sha256_file
            def frozen_sha(path):
                path = Path(path)
                if path.name == "final_manifest.csv":
                    return v3.MANIFEST_SHA256
                return real_sha(path)
            with mock.patch.object(master_preflight, "sha256_file", side_effect=frozen_sha):
                with self.assertRaises(v3.OperatorError) as ctx:
                    master_preflight._resolve_manifest_and_image_root(root)
            self.assertIn("different image roots", str(ctx.exception))

    def test_input_diagnostics_report_wrong_candidate_hash(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mounted = root / "wrong-v1"
            mounted.mkdir()
            candidate = mounted / "dataset_manifest.csv"
            candidate.write_text("wrong revision\n", encoding="utf-8")
            message = master_preflight._dataset_error_message(v3.OperatorError("found 0"), root)
            self.assertIn(v3.MANIFEST_SHA256, message)
            self.assertIn(v3.CLASS_MAP_SHA256, message)
            self.assertIn(str(candidate), message)
            self.assertIn(v3.sha256_file(candidate), message)
            self.assertIn("wrong-v1", message)

    def test_master_input_failure_is_fail_fast_and_actionable(self):
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(master_preflight, "_resolve_manifest_and_image_root", side_effect=v3.OperatorError("found 0")):
            with self.assertRaises(v3.OperatorError) as ctx:
                master_preflight.resolve_master_inputs("K1", td)
            text = str(ctx.exception)
            self.assertIn("before dependency installation", text)
            self.assertIn("attach the frozen CropCop V1 dataset", text)

    def test_k1_principal_g1_preflight_is_required(self):
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(master_preflight, "_resolve_manifest_and_image_root", return_value=(Path("/m"), Path("/images"))), \
             mock.patch.object(master_preflight, "_resolve_class_map", return_value=Path("/c")), \
             mock.patch.object(master_preflight, "_resolve_principal_g1_fast", side_effect=v3.OperatorError("missing G1")):
            with self.assertRaises(v3.OperatorError) as ctx:
                master_preflight.resolve_master_inputs("K1", td)
            self.assertIn("historical principal-G1 preflight failed", str(ctx.exception))

    def test_worker_does_not_require_principal_g1_mount(self):
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(master_preflight, "_resolve_manifest_and_image_root", return_value=(Path("/m"), Path("/images"))), \
             mock.patch.object(master_preflight, "_resolve_class_map", return_value=Path("/c")), \
             mock.patch.object(master_preflight, "_resolve_principal_g1_fast") as principal:
            result = master_preflight.resolve_master_inputs("K2", td)
            self.assertEqual(result, (Path("/m"), Path("/c"), Path("/images")))
            principal.assert_not_called()

    def test_master_driver_checks_inputs_before_stack_install(self):
        source = (OPS / "master_account_driver.py").read_text(encoding="utf-8")
        self.assertLess(source.index("resolve_master_inputs(account_id)"), source.index("ensure_science_checkout()"))
        self.assertLess(source.index("resolve_master_inputs(account_id)"), source.index("ensure_locked_stack(repo)"))


if __name__ == "__main__":
    unittest.main()
