from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "journal_extension" / "scripts"
SRC = ROOT / "journal_extension" / "src"
for entry in (str(SCRIPTS), str(SRC)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

SPEC = importlib.util.spec_from_file_location(
    "trackb_v4_materialize_under_test",
    SCRIPTS / "trackb_v4_materialize.py",
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)

CORE_SPEC = importlib.util.spec_from_file_location(
    "build_trackb_core_package_under_test",
    SCRIPTS / "build_trackb_core_package.py",
)
core_module = importlib.util.module_from_spec(CORE_SPEC)
assert CORE_SPEC.loader is not None
CORE_SPEC.loader.exec_module(core_module)


EXEC_SPEC = importlib.util.spec_from_file_location(
    "trackb_v4_execute_attached_under_test",
    SCRIPTS / "trackb_v4_execute_attached.py",
)
exec_module = importlib.util.module_from_spec(EXEC_SPEC)
assert EXEC_SPEC.loader is not None
EXEC_SPEC.loader.exec_module(exec_module)


class TrackBV4MaterializationTests(unittest.TestCase):
    def test_core_builder_disables_python_bytecode_during_snapshot_preparation(self):
        source = (SCRIPTS / "build_trackb_core_package.py").read_text(encoding="utf-8")
        self.assertIn('sys.executable, "-B", str(prep)', source)
        self.assertIn('prep_env["PYTHONDONTWRITEBYTECODE"] = "1"', source)
        self.assertIn("_purge_transient_repository_artifacts(repo_copy)", source)
        self.assertIn("_assert_repository_transport_clean(repo_copy)", source)

    def test_stage4_transport_scrub_precedes_role_validation_and_identity(self):
        source = (SCRIPTS / "trackb_v4_materialize.py").read_text(encoding="utf-8")
        main_start = source.index("def main() -> int:")
        main = source[main_start:]
        scrub = main.index('transport_hygiene = {')
        validate = main.index("_validate_roles(", scrub)
        pair = main.index("_write_pair_receipts(", validate)
        self.assertLess(scrub, validate)
        self.assertLess(validate, pair)

    def test_core_repository_hygiene_removes_post_copy_python_bytecode(self):
        import importlib.util as _iu

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "repository"
            scripts = root / "journal_extension" / "scripts"
            scripts.mkdir(parents=True)
            source = scripts / "synthetic_module.py"
            source.write_text("VALUE = 7\n", encoding="utf-8")

            spec = _iu.spec_from_file_location("synthetic_trackb_cache_module", source)
            loaded = _iu.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(loaded)
            self.assertTrue(any(root.rglob("*.pyc")))

            receipt = core_module._purge_transient_repository_artifacts(root)
            core_module._assert_repository_transport_clean(root)

            self.assertEqual(receipt["status"], "PASS_REPOSITORY_TRANSPORT_HYGIENE")
            self.assertFalse(any(root.rglob("*.pyc")))
            self.assertFalse(any(p.name == "__pycache__" for p in root.rglob("*")))

    def test_stage4_role_hygiene_scrubs_runtime_artifacts_before_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "payload.txt").write_text("stable\n", encoding="utf-8")
            cache = root / "sub" / "__pycache__"
            cache.mkdir(parents=True)
            (cache / "x.cpython-312.pyc").write_bytes(b"ephemeral")
            (root / ".pytest_cache").mkdir()
            (root / ".pytest_cache" / "state").write_text("junk", encoding="utf-8")

            receipt = module._scrub_role_transport_artifacts(root)
            identity = module._role_content_identity(root)

            self.assertEqual(receipt["status"], "PASS_ROLE_TRANSPORT_HYGIENE")
            self.assertGreaterEqual(receipt["removed_count"], 2)
            self.assertEqual(identity["file_count"], 1)
            self.assertEqual(identity["total_bytes"], len(b"stable\n"))

    def test_stage4_role_identity_rejects_unscrubbed_runtime_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "payload.txt").write_text("stable\n", encoding="utf-8")
            cache = root / "__pycache__"
            cache.mkdir()
            (cache / "x.pyc").write_bytes(b"ephemeral")
            with self.assertRaises(module.TrackBOpsError):
                module._role_content_identity(root)

    def test_kaggle_publication_forces_keep_tabular_exact_byte_mode(self):
        source = (
            ROOT / "journal_extension" / "src" / "cropcop_je" / "trackb_r07_ops.py"
        ).read_text(encoding="utf-8")
        start = source.index("def publish_private_kaggle_dataset(")
        end = source.index("def verify_kaggle_publication_capability(", start)
        helper = source[start:end]
        self.assertIn('"-r", "zip", "-t"', helper)

    def test_publication_probe_exercises_nested_multitype_exact_bytes(self):
        source = (
            ROOT / "journal_extension" / "src" / "cropcop_je" / "trackb_r07_ops.py"
        ).read_text(encoding="utf-8")
        start = source.index("def verify_kaggle_publication_capability(")
        end = source.index("def acquire_claim_lease(", start)
        helper = source[start:end]
        self.assertIn("NESTED_JSON_PYTHON_CSV_BINARY_EXACT_BYTES", helper)
        self.assertIn("probe_source.py", helper)
        self.assertIn("probe_table.csv", helper)
        self.assertIn("probe_binary.bin", helper)
        self.assertIn("full_roundtrip=True", helper)

    def test_final_publication_uses_exact_handoff_sentinels_not_download_all_archive(self):
        source = (SCRIPTS / "trackb_v4_materialize.py").read_text(encoding="utf-8")
        main = source[source.index("def main() -> int:"):]

        self.assertNotIn("_verify_published_archive_roundtrip(", source)
        self.assertNotIn('"kaggle", "datasets", "download", "-d"', source)
        self.assertIn("verify_kaggle_published_file_roundtrip(", main)
        self.assertIn('"TRACKB_INFRASTRUCTURE_BUNDLE.json"', main)
        self.assertIn('"TRACKB_EXTERNAL_BUNDLE.json"', main)
        self.assertIn('"DEFERRED_TO_ATTACHED_FULL_BYTE_VERIFICATION"', main)
        self.assertIn('"NOTEBOOK_01_FULL_ATTACHED_ROLE_CONTENT_IDENTITY"', main)

    def test_final_publication_preserves_local_bundles_until_readiness_receipt(self):
        source = (SCRIPTS / "trackb_v4_materialize.py").read_text(encoding="utf-8")
        main = source[source.index("def main() -> int:"):]

        self.assertNotIn("shutil.rmtree(infra_root", main)
        self.assertNotIn("shutil.rmtree(external_root", main)
        self.assertLess(
            main.index("verify_kaggle_published_file_roundtrip(\n            infra_slug"),
            main.index("verify_kaggle_published_file_roundtrip(\n            external_slug"),
        )
        self.assertLess(
            main.index("verify_kaggle_published_file_roundtrip(\n            external_slug"),
            main.index('readiness["status"] = "PASS_TRACKB_INPUT_MATERIALIZATION"'),
        )

    def test_exact_handoff_sentinel_roundtrip_verifies_bytes_and_sha(self):
        from cropcop_je import trackb_r07_ops

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            expected = root / "TRACKB_INFRASTRUCTURE_BUNDLE.json"
            expected.write_bytes(b'{"materialization_id":"abc"}\n')

            def fake_wait(slug, relative_path, destination, *, timeout_seconds):
                self.assertEqual(slug, "owner/test-dataset")
                self.assertEqual(relative_path, expected.name)
                self.assertEqual(timeout_seconds, 7)
                destination.mkdir(parents=True, exist_ok=True)
                observed = destination / expected.name
                observed.write_bytes(expected.read_bytes())
                return observed

            with mock.patch.object(
                trackb_r07_ops,
                "_wait_download_kaggle_file",
                side_effect=fake_wait,
            ):
                receipt = trackb_r07_ops.verify_kaggle_published_file_roundtrip(
                    "owner/test-dataset",
                    expected.name,
                    expected,
                    timeout_seconds=7,
                )

        self.assertEqual(receipt["status"], "PASS_EXACT_HANDOFF_SENTINEL_ROUNDTRIP")
        self.assertEqual(receipt["bytes"], len(b'{"materialization_id":"abc"}\n'))
        self.assertEqual(receipt["authority"], "EXACT_SINGLE_FILE_ROUNDTRIP")

    def test_exact_handoff_sentinel_roundtrip_rejects_remote_byte_drift(self):
        from cropcop_je import trackb_r07_ops

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            expected = root / "TRACKB_EXTERNAL_BUNDLE.json"
            expected.write_bytes(b"expected")

            def fake_wait(_slug, relative_path, destination, *, timeout_seconds):
                destination.mkdir(parents=True, exist_ok=True)
                observed = destination / relative_path
                observed.write_bytes(b"DIFFERENT")
                return observed

            with mock.patch.object(
                trackb_r07_ops,
                "_wait_download_kaggle_file",
                side_effect=fake_wait,
            ):
                with self.assertRaises(trackb_r07_ops.TrackBOpsError):
                    trackb_r07_ops.verify_kaggle_published_file_roundtrip(
                        "owner/test-dataset",
                        expected.name,
                        expected,
                        timeout_seconds=7,
                    )

    def test_manifest_only_publication_does_not_overclaim_full_roundtrip(self):
        source = (
            ROOT / "journal_extension" / "src" / "cropcop_je" / "trackb_r07_ops.py"
        ).read_text(encoding="utf-8")
        start = source.index("def _verify_remote_kaggle_content(")
        end = source.index("def publish_private_kaggle_dataset(", start)
        helper = source[start:end]
        self.assertIn(
            '"BOUND_MANIFEST_ONLY_PENDING_ATTACHED_BYTE_VERIFICATION"',
            helper,
        )
        self.assertIn('"attached_full_byte_verification_required"', helper)

    def test_attached_execution_rehashes_every_role_before_qualification(self):
        source = (SCRIPTS / "trackb_v4_execute_attached.py").read_text(encoding="utf-8")
        pairing_start = source.index("def _validate_pairing(")
        pairing_end = source.index("def _run_qualification(", pairing_start)
        pairing = source[pairing_start:pairing_end]
        self.assertIn("_role_content_identity(path.parent)", pairing)
        self.assertIn("declared_content != observed_content", pairing)
        self.assertIn("attached role payload bytes differ", pairing)

    def test_final_v1_resolver_prefers_hash_valid_image_backed_authority_pair(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            real = root / "CropCop_Final_v1"
            reports = root / "CropCop_Final_v1_CERTIFICATION_REPORTS"

            for base in (real, reports):
                (base / "audit").mkdir(parents=True)
                (base / "audit" / "final_manifest.csv").write_text("manifest\n", encoding="utf-8")
                (base / "audit" / "class_to_idx.json").write_text("{}\n", encoding="utf-8")

            (real / "dataset" / "train").mkdir(parents=True)
            (real / "dataset" / "val").mkdir(parents=True)

            def fake_sha(path: Path) -> str:
                if Path(path).name == "final_manifest.csv":
                    return module.DATASET_MANIFEST_SHA256
                if Path(path).name == "class_to_idx.json":
                    return module.CLASS_MAP_SHA256
                return "0" * 64

            with mock.patch.object(module, "sha256_file", side_effect=fake_sha):
                manifest, class_map, image_root = module._find_v1(root)

            self.assertEqual(manifest, (real / "audit" / "final_manifest.csv").resolve())
            self.assertEqual(class_map, (real / "audit" / "class_to_idx.json").resolve())
            self.assertEqual(image_root, (real / "dataset").resolve())

    def test_final_v1_resolver_fails_if_two_image_backed_authorities_exist(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ("A", "B"):
                base = root / name
                (base / "audit").mkdir(parents=True)
                (base / "audit" / "final_manifest.csv").write_text("manifest\n", encoding="utf-8")
                (base / "audit" / "class_to_idx.json").write_text("{}\n", encoding="utf-8")
                (base / "dataset" / "train").mkdir(parents=True)
                (base / "dataset" / "val").mkdir(parents=True)

            def fake_sha(path: Path) -> str:
                if Path(path).name == "final_manifest.csv":
                    return module.DATASET_MANIFEST_SHA256
                if Path(path).name == "class_to_idx.json":
                    return module.CLASS_MAP_SHA256
                return "0" * 64

            with mock.patch.object(module, "sha256_file", side_effect=fake_sha):
                with self.assertRaises(module.TrackBOpsError):
                    module._find_v1(root)


    def test_core_builder_accepts_exact_s1_canonical_validation_metric_schema(self):
        record = (
            ROOT
            / "journal_extension"
            / "track_b_r07"
            / "replay_authority"
            / "R07_S1_ORIGINAL_RUN_RECORD.json"
        )
        parsed = core_module._verify_run_record(record, "S1")
        metrics = parsed["result_summary"]["selected_metrics"]
        self.assertIn("validation_accuracy", metrics)
        self.assertIn("validation_balanced_accuracy", metrics)
        self.assertIn("validation_macro_f1", metrics)
        self.assertIn("validation_nll", metrics)

    def test_core_builder_rejects_obsolete_short_metric_schema(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run_record.json"
            path.write_text(
                json.dumps({
                    "run_id": core_module.R07_RUN_RECORDS["S1"]["run_id"],
                    "artifact_locators": {
                        "selected_checkpoint": {
                            "sha256": core_module.R07_CHECKPOINTS["S1"]
                        }
                    },
                    "result_summary": {
                        "selected_checkpoint_sha256": core_module.R07_CHECKPOINTS["S1"],
                        "selected_metrics": {
                            "accuracy": 0.98,
                            "balanced_accuracy": 0.96,
                            "macro_f1": 0.96,
                            "nll": 0.10,
                        },
                    },
                }),
                encoding="utf-8",
            )
            with self.assertRaises(core_module.TrackBError):
                core_module._verify_run_record(path, "S1")

    def test_core_builder_accepts_recovered_continuation_record_shape(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "recovered.json"
            path.write_text(
                json.dumps({
                    "run_id": core_module.R07_RUN_RECORDS["S2"]["run_id"],
                    "artifact_locators": {
                        "selected_checkpoint": {
                            "sha256": core_module.R07_CHECKPOINTS["S2"]
                        }
                    },
                    "result_summary": {
                        "selected_epoch": 23,
                        "selected_checkpoint_sha256": core_module.R07_CHECKPOINTS["S2"],
                        "selected_metrics": {
                            "validation_accuracy": 0.9846041055718475,
                            "validation_balanced_accuracy": 0.9680297463314904,
                            "validation_macro_f1": 0.9674376113284783,
                            "validation_nll": 0.10054967464716538,
                        },
                    },
                    "recovery_provenance": {
                        "kind": "cryptographic_terminal_record_recovery"
                    },
                }),
                encoding="utf-8",
            )
            parsed = core_module._verify_run_record(path, "S2")
            self.assertEqual(
                parsed["result_summary"]["selected_checkpoint_sha256"],
                core_module.R07_CHECKPOINTS["S2"],
            )


    def test_trackb_core_builder_binds_dino_factory_to_dedicated_source_root(self):
        source = (SCRIPTS / "build_trackb_core_package.py").read_text(encoding="utf-8")
        self.assertIn(
            '"--dino-factory-source-root", "repository/journal_extension/teacher_factory"',
            source,
        )
        self.assertNotIn(
            '"--dino-factory-source-root", "repository",',
            source,
        )

    def test_trackb_dino_prequalification_validates_sealed_factory_sources(self):
        source = (SCRIPTS / "trackb_v4_materialize.py").read_text(encoding="utf-8")
        self.assertIn("validate_teacher_factory_bundle(", source)
        self.assertIn(
            'repo_root / "journal_extension" / "teacher_factory"',
            source,
        )
        self.assertIn('"factory_source_validation": "PASS"', source)


    def test_dino_factory_manifest_validates_only_against_dedicated_source_root(self):
        from cropcop_je.g1 import factory_bundle_hash, validate_teacher_factory_bundle

        source_root = ROOT / "journal_extension" / "teacher_factory"
        source = source_root / "historical_dino_tiny.py"
        self.assertTrue(source.is_file())
        manifest = {
            "schema_version": "1.0",
            "entrypoint": "historical_dino_tiny:build_teacher",
            "output_order_transform": "none",
            "files": [{
                "path": "historical_dino_tiny.py",
                "sha256": module.sha256_file(source),
                "bytes": source.stat().st_size,
            }],
        }
        manifest["bundle_sha256"] = factory_bundle_hash(manifest)

        wrong = validate_teacher_factory_bundle(
            manifest,
            source_root=ROOT,
            expected_entrypoint=manifest["entrypoint"],
        )
        self.assertIn("teacher factory source file missing: historical_dino_tiny.py", wrong)

        correct = validate_teacher_factory_bundle(
            manifest,
            source_root=source_root,
            expected_entrypoint=manifest["entrypoint"],
        )
        self.assertEqual(correct, [])


    def test_historical_builder_has_exact_batch_and_output_disk_preflight(self):
        source = (
            ROOT / "journal_extension" / "scripts" / "build_trackb_historical_compare.py"
        ).read_text(encoding="utf-8")
        self.assertIn("dino_batch_probe_size", source)
        self.assertIn("[:32]", source)
        self.assertIn("len(dino_samples) != 64", source)
        self.assertIn("shutil.disk_usage(output_root).free", source)
        self.assertIn("_historical_output_budget_bytes()", source)
        self.assertLess(
            source.index("_preflight_historical_components("),
            source.index("_pack_orb("),
        )

    def test_kaggle_publication_never_uses_list_files_as_existence_gate(self):
        source = (
            ROOT / "journal_extension" / "src" / "cropcop_je" / "trackb_r07_ops.py"
        ).read_text(encoding="utf-8")
        start = source.index("def publish_private_kaggle_dataset(")
        end = source.index("def verify_kaggle_publication_capability(", start)
        publish = source[start:end]
        self.assertNotIn('["kaggle", "datasets", "files"', publish)
        self.assertNotIn("kaggle_dataset_exists(slug)", publish)
        self.assertIn('"kaggle", "datasets", "create",', publish)
        self.assertIn("_read_remote_kaggle_content_manifest(slug, strict=False)", publish)
        self.assertIn("_kaggle_publication_failure_kind", publish)

    def test_kaggle_dataset_exists_uses_status_not_list_files(self):
        source = (
            ROOT / "journal_extension" / "src" / "cropcop_je" / "trackb_r07_ops.py"
        ).read_text(encoding="utf-8")
        start = source.index("def kaggle_dataset_exists(")
        end = source.index("def _metadata_slug(", start)
        helper = source[start:end]
        self.assertIn('["kaggle", "datasets", "status", slug, "--format", "json"]', helper)
        self.assertNotIn('"datasets", "files"', helper)

    def test_publication_capability_probe_precedes_expensive_stage_one(self):
        source = (SCRIPTS / "trackb_v4_materialize.py").read_text(encoding="utf-8")
        probe = source.index("publication_probe = verify_kaggle_publication_capability(early_owner)")
        stage_one = source.index('stage("1 :: authoritative external cohorts')
        self.assertLess(probe, stage_one)
        self.assertIn('"kaggle_publication_capability": kaggle_publication_probe', source)

    def test_publication_capability_probe_is_content_addressed_and_roundtripped(self):
        source = (
            ROOT / "journal_extension" / "src" / "cropcop_je" / "trackb_r07_ops.py"
        ).read_text(encoding="utf-8")
        start = source.index("def verify_kaggle_publication_capability(")
        end = source.index("def acquire_claim_lease(", start)
        helper = source[start:end]
        self.assertIn("probe_digest = sha256_json(payload)", helper)
        self.assertIn("cropcop-trackb-pubprobe-v5-", helper)
        self.assertIn("full_roundtrip=True", helper)
        self.assertIn("allow_version=False", helper)
        self.assertIn("PASS_KAGGLE_PUBLICATION_CAPABILITY", helper)

    def test_kaggle_publication_authority_distinguishes_probe_from_final_handoff(self):
        source = (
            ROOT / "journal_extension" / "src" / "cropcop_je" / "trackb_r07_ops.py"
        ).read_text(encoding="utf-8")
        verify_start = source.index("def _verify_remote_kaggle_content(")
        verify_end = source.index("def publish_private_kaggle_dataset(", verify_start)
        verify = source[verify_start:verify_end]
        self.assertIn("_wait_remote_kaggle_content_manifest(", verify)
        self.assertIn("_wait_download_kaggle_file(", verify)
        self.assertIn('"BOUND_MANIFEST_AND_EXACT_BYTE_ROUNDTRIP"', verify)
        self.assertIn('"BOUND_MANIFEST_ONLY_PENDING_ATTACHED_BYTE_VERIFICATION"', verify)
        self.assertIn('"attached_full_byte_verification_required"', verify)
        self.assertIn("_kaggle_dataset_status_best_effort(slug)", verify)
        self.assertNotIn("_wait_kaggle_dataset_ready(", verify)

    def test_kaggle_visibility_wait_retries_processing_403_and_404(self):
        source = (
            ROOT / "journal_extension" / "src" / "cropcop_je" / "trackb_r07_ops.py"
        ).read_text(encoding="utf-8")
        start = source.index("def _wait_download_kaggle_file(")
        end = source.index("def _download_kaggle_file(", start)
        helper = source[start:end]
        self.assertIn('"403"', helper)
        self.assertIn('"404"', helper)
        self.assertIn('"processing"', helper)
        self.assertIn('"pending"', helper)
        self.assertIn("timeout_seconds: int = 3600", helper)

    def test_claim_lease_and_attempt_state_do_not_require_status_endpoint(self):
        source = (
            ROOT / "journal_extension" / "src" / "cropcop_je" / "trackb_r07_ops.py"
        ).read_text(encoding="utf-8")
        lease_start = source.index("def acquire_claim_lease(")
        lease_end = source.index("def read_latest_attempt_state(", lease_start)
        lease = source[lease_start:lease_end]
        self.assertNotIn("kaggle_dataset_exists(slug)", lease)
        self.assertNotIn("_wait_kaggle_dataset_ready(", lease)
        self.assertIn("_wait_download_kaggle_file(", lease)

        attempt_start = source.index("def read_latest_attempt_state(")
        attempt_end = source.index("def publish_attempt_state(", attempt_start)
        attempt = source[attempt_start:attempt_end]
        self.assertNotIn("kaggle_dataset_exists(slug)", attempt)
        self.assertNotIn("_kaggle_dataset_status(", attempt)
        self.assertIn("_download_kaggle_file(", attempt)

    def test_large_download_all_archive_is_not_a_readiness_primitive(self):
        source = (SCRIPTS / "trackb_v4_materialize.py").read_text(encoding="utf-8")
        self.assertNotIn("def _verify_published_archive_roundtrip(", source)
        self.assertNotIn("Kaggle whole-archive visibility pending", source)
        self.assertNotIn('"kaggle", "datasets", "download", "-d"', source)
        self.assertIn("verify_kaggle_published_file_roundtrip(", source)
        self.assertIn("DEFERRED_TO_ATTACHED_FULL_BYTE_VERIFICATION", source)

    def test_ambiguous_create_waits_for_deterministic_slug_manifest(self):
        source = (
            ROOT / "journal_extension" / "src" / "cropcop_je" / "trackb_r07_ops.py"
        ).read_text(encoding="utf-8")
        start = source.index("def publish_private_kaggle_dataset(")
        end = source.index("def verify_kaggle_publication_capability(", start)
        helper = source[start:end]
        self.assertIn('kind in {"CONFLICT", "PERMISSION", "TRANSIENT"}', helper)
        self.assertIn("_wait_remote_kaggle_content_manifest(", helper)
        self.assertIn("timeout_seconds=900", helper)
        self.assertIn('"action": "reuse_after_ambiguous_write"', helper)

    def test_publication_retains_local_bundle_through_readiness_closure(self):
        source = (SCRIPTS / "trackb_v4_materialize.py").read_text(encoding="utf-8")
        self.assertNotIn("shutil.rmtree(infra_root, ignore_errors=True)", source)
        self.assertNotIn("shutil.rmtree(external_root, ignore_errors=True)", source)

        infra_manifest = source.index(
            'infra_manifest = load_json(infra_root / "TRACKB_KAGGLE_CONTENT_MANIFEST.json")'
        )
        infra_verify = source.index(
            'infra_pub["handoff_sentinel_roundtrip"] = verify_kaggle_published_file_roundtrip(',
            infra_manifest,
        )
        external_manifest = source.index(
            'external_manifest = load_json(external_root / "TRACKB_KAGGLE_CONTENT_MANIFEST.json")'
        )
        external_verify = source.index(
            'external_pub["handoff_sentinel_roundtrip"] = verify_kaggle_published_file_roundtrip(',
            external_manifest,
        )
        pass_marker = source.index(
            'readiness["status"] = "PASS_TRACKB_INPUT_MATERIALIZATION"',
            external_verify,
        )
        self.assertLess(infra_manifest, infra_verify)
        self.assertLess(external_manifest, external_verify)
        self.assertLess(external_verify, pass_marker)

    def test_external_transport_probe_requires_url_checksum_and_size(self):
        source = (
            ROOT / "journal_extension" / "src" / "cropcop_je" / "trackb_r07_ops.py"
        ).read_text(encoding="utf-8")
        self.assertIn("Irish Potato source probe found invalid download URL", source)
        self.assertIn("Irish Potato source probe found missing checksum", source)
        self.assertIn("Irish Potato source probe found invalid declared size", source)
        self.assertIn("GVLiD public API probe reported no positive declared payload bytes", source)
        self.assertIn("download_urls_valid", source)

    def test_archive_extraction_has_member_size_symlink_and_disk_guards(self):
        source = (
            ROOT / "journal_extension" / "src" / "cropcop_je" / "trackb_r07_ops.py"
        ).read_text(encoding="utf-8")
        self.assertIn("max_members: int = 120000", source)
        self.assertIn("max_uncompressed_bytes: int = 20 * 1024 * 1024 * 1024", source)
        self.assertIn("archive symlink member is not permitted", source)
        self.assertIn("insufficient disk before archive extraction", source)


    def test_embedded_replay_authority_matches_frozen_trackb_contract(self):
        authority = ROOT / "journal_extension" / "track_b_r07" / "replay_authority"
        s1 = authority / "R07_S1_ORIGINAL_RUN_RECORD.json"
        k3 = authority / "TRACKA_V12_K3_PUBLIC_REPORT.json"

        module._verify_run_record_contract(s1, "S1")
        report = json.loads(k3.read_text(encoding="utf-8"))
        self.assertEqual(report.get("status"), "PASS")
        self.assertIs(report.get("science_complete"), True)
        self.assertEqual(
            report.get("science_source_sha"),
            "56023042e57758591df9babb3438f191dbe10312",
        )

        observed = {}
        for rows in (report.get("slot_results") or {}).values():
            for row in rows or []:
                if row.get("experiment_id") in {
                    "R07-CNXTT-CONTEXT-S2",
                    "R07-CNXTT-CONTEXT-S3",
                }:
                    observed[row["experiment_id"]] = row

        for seed in ("S2", "S3"):
            experiment_id = f"R07-CNXTT-CONTEXT-{seed}"
            self.assertIn(experiment_id, observed)
            self.assertEqual(
                observed[experiment_id]["run_id"],
                module.R07_RUN_RECORDS[seed]["run_id"],
            )
            self.assertEqual(
                observed[experiment_id]["selected_checkpoint_sha256"],
                module.R07_CHECKPOINTS[seed],
            )
            self.assertEqual(observed[experiment_id]["run_status"], "PASS")
            self.assertFalse(observed[experiment_id]["continuation_required"])

    def test_selected_checkpoint_index_resolution_is_hash_bound(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            selected = root / "objects" / "selected.ckpt"
            selected.parent.mkdir(parents=True)
            selected.write_bytes(b"checkpoint-bytes")
            expected = "a" * 64
            (root / "checkpoint_index.json").write_text(
                json.dumps({
                    "selected": {
                        "sha256": expected,
                        "relative_path": "objects/selected.ckpt",
                    }
                }),
                encoding="utf-8",
            )
            with mock.patch.object(module, "sha256_file", return_value=expected):
                index_root, checkpoint = module._selected_checkpoint_from_index(root, expected)
            self.assertEqual(index_root, root.resolve())
            self.assertEqual(checkpoint, selected.resolve())

    def test_selected_checkpoint_index_resolution_rejects_wrong_version(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            selected = root / "objects" / "selected.ckpt"
            selected.parent.mkdir(parents=True)
            selected.write_bytes(b"checkpoint-bytes")
            (root / "checkpoint_index.json").write_text(
                json.dumps({
                    "selected": {
                        "sha256": "b" * 64,
                        "relative_path": "objects/selected.ckpt",
                    }
                }),
                encoding="utf-8",
            )
            with self.assertRaises(module.TrackBOpsError):
                module._selected_checkpoint_from_index(root, "a" * 64)


    def test_role_content_identity_binds_every_payload_byte(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "data").mkdir()
            payload = root / "data" / "sample.bin"
            payload.write_bytes(b"alpha")
            first = module._role_content_identity(root)
            second = exec_module._role_content_identity(root)
            self.assertEqual(first, second)
            payload.write_bytes(b"beta")
            changed = module._role_content_identity(root)
            self.assertNotEqual(first["content_sha256"], changed["content_sha256"])

    def test_role_content_identity_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "target.bin"
            target.write_bytes(b"x")
            link = root / "linked.bin"
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable on this platform")
            with self.assertRaises(module.TrackBOpsError):
                module._role_content_identity(root)
            with self.assertRaises(exec_module.TrackBOpsError):
                exec_module._role_content_identity(root)

    def test_external_cohorts_are_acquired_before_historical_compute(self):
        source = (SCRIPTS / "trackb_v4_materialize.py").read_text(encoding="utf-8")
        external = source.index('stage("1 :: authoritative external cohorts')
        historical = source.index('stage("3 :: safe historical comparison")')
        self.assertLess(external, historical)
        self.assertIn("expected_source_manifest_sha256", source)

    def test_attached_execution_refuses_to_delete_authoritative_outputs(self):
        source = (SCRIPTS / "trackb_v4_execute_attached.py").read_text(encoding="utf-8")
        self.assertIn(
            "refusing to delete or overwrite an existing Track-B authoritative output root",
            source,
        )
        self.assertNotIn("shutil.rmtree(output_root)", source)

    def test_notebook01_fail_fast_operator_preflight_precedes_bootstrap_and_science(self):
        notebook = json.loads(
            (
                ROOT
                / "journal_extension"
                / "kaggle"
                / "TrackB_01_Final_Execution.ipynb"
            ).read_text(encoding="utf-8")
        )
        source = "\n".join(
            "".join(cell.get("source") or [])
            for cell in notebook.get("cells") or []
        )
        attached_trust = source.index("PASS_PRECONTROLLER_ATTACHED_AUTHORITY")
        gpu_preflight = source.index("T4 x2 preflight could not query nvidia-smi")
        bootstrap = source.index(
            "BOOTSTRAP = REPO / 'journal_extension/scripts/bootstrap_trackb_runtime.py'"
        )
        operator_preflight = source.index("PASS_OPERATOR_PREFLIGHT")
        controller = source.index(
            "CONTROLLER = REPO / 'journal_extension/scripts/trackb_v4_execute_attached.py'"
        )
        self.assertLess(attached_trust, gpu_preflight)
        self.assertLess(gpu_preflight, bootstrap)
        self.assertLess(bootstrap, operator_preflight)
        self.assertLess(operator_preflight, controller)
        self.assertIn("verify_kaggle_publication_capability(observed_owner)", source)
        self.assertIn("BOUND_MANIFEST_AND_EXACT_BYTE_ROUNDTRIP", source)
        self.assertIn("roundtrip_verified_file_count", source)
        self.assertIn("verify_authenticated_kaggle_owner(KAGGLE_OWNER)", source)
        self.assertIn("Track-B v5 qualification requires Kaggle T4 x2", source)
        self.assertIn("KAGGLE_OWNER = 'ranamuhammadahmed6'", source)

    def test_notebook01_scrubs_publication_credentials_before_scientific_controller(self):
        notebook = json.loads(
            (
                ROOT
                / "journal_extension"
                / "kaggle"
                / "TrackB_01_Final_Execution.ipynb"
            ).read_text(encoding="utf-8")
        )
        source = "\n".join(
            "".join(cell.get("source") or [])
            for cell in notebook.get("cells") or []
        )
        probe = source.index(
            "publication_probe = verify_kaggle_publication_capability(observed_owner)"
        )
        scrub = source.index("os.environ.pop('KAGGLE_API_TOKEN', None)", probe)
        controller = source.index(
            "CONTROLLER = REPO / 'journal_extension/scripts/trackb_v4_execute_attached.py'"
        )
        self.assertLess(probe, scrub)
        self.assertLess(scrub, controller)
        self.assertIn("finally:", source)
        self.assertIn("os.environ.pop('CROPCOP_GITHUB_TOKEN', None)", source)

    def test_notebook01_rejects_attachment_mixups_before_science(self):
        notebook = json.loads(
            (
                ROOT
                / "journal_extension"
                / "kaggle"
                / "TrackB_01_Final_Execution.ipynb"
            ).read_text(encoding="utf-8")
        )
        source = "\n".join(
            "".join(cell.get("source") or [])
            for cell in notebook.get("cells") or []
        )
        self.assertIn(
            "Qualification mode must attach only infrastructure + external handoffs",
            source,
        )
        self.assertIn(
            "Claim mode requires exactly one immutable qualification bundle",
            source,
        )
        self.assertIn(
            "Infrastructure and external handoffs must be distinct Kaggle datasets",
            source,
        )
        self.assertIn("is outside its paired dataset root", source)
        self.assertIn("bundle receipt role mismatch", source)

    def test_v1_guard_hotfix_allows_explicit_false_historical_provenance(self):
        hotfix_path = (
            ROOT
            / "journal_extension"
            / "operator_hotfixes"
            / "trackb_v5_v1_guard_hotfix.py"
        )
        spec = importlib.util.spec_from_file_location(
            "trackb_v5_v1_guard_hotfix_under_test",
            hotfix_path,
        )
        hotfix = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(hotfix)

        manifest = {
            "schema_version": "1.1",
            "role": "historical_compare",
            "coverage_scope": "V1_TRAIN_VAL_ONLY",
            "v1_test_image_bytes_accessed": False,
            "ext_i_eligible": False,
            "maximum_evidence_grade": "EXT-S",
            "files": {
                "historical_manifest": {
                    "path": "historical_manifest.csv",
                    "sha256": "a" * 64,
                }
            },
        }
        hotfix.validate_manifest_safety("historical_compare", manifest)

    def test_v1_guard_hotfix_rejects_true_or_forbidden_test_surface(self):
        hotfix_path = (
            ROOT
            / "journal_extension"
            / "operator_hotfixes"
            / "trackb_v5_v1_guard_hotfix.py"
        )
        spec = importlib.util.spec_from_file_location(
            "trackb_v5_v1_guard_hotfix_reject_under_test",
            hotfix_path,
        )
        hotfix = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(hotfix)

        base = {
            "role": "historical_compare",
            "coverage_scope": "V1_TRAIN_VAL_ONLY",
            "v1_test_image_bytes_accessed": False,
            "ext_i_eligible": False,
            "maximum_evidence_grade": "EXT-S",
            "files": {},
        }
        bad_flag = dict(base)
        bad_flag["v1_test_image_bytes_accessed"] = True
        with self.assertRaises(hotfix.GuardHotfixError):
            hotfix.validate_manifest_safety("historical_compare", bad_flag)

        bad_path = dict(base)
        bad_path["files"] = {
            "x": {"path": "safe/v1_test/images.bin", "sha256": "a" * 64}
        }
        with self.assertRaises(hotfix.GuardHotfixError):
            hotfix.validate_manifest_safety("historical_compare", bad_path)

        bad_surface = dict(base)
        bad_surface["surface"] = "DS-V1-TEST-CONSUMED"
        with self.assertRaises(hotfix.GuardHotfixError):
            hotfix.validate_manifest_safety("historical_compare", bad_surface)

    def test_notebook01_uses_fast_guard_smoke_and_single_full_byte_gate(self):
        notebook = json.loads(
            (
                ROOT
                / "journal_extension"
                / "kaggle"
                / "TrackB_01_Final_Execution.ipynb"
            ).read_text(encoding="utf-8")
        )
        source = "\n".join(
            "".join(cell.get("source") or [])
            for cell in notebook.get("cells") or []
        )
        self.assertIn("PASS_FAST_INPUT_DISCOVERY_SMOKE", source)
        self.assertIn("TRACKB_V5_V1_GUARD_FALSE_POSITIVE_FIX_v1", source)
        self.assertIn("sitecustomize.py", source)
        self.assertIn("TRACKB_V1_GUARD_HOTFIX_ACTIVE", source)
        self.assertIn("EXACT_EQUIVALENT_VECTORIZED_CUTOFF_TIE_DETECTION", source)
        self.assertIn("DINO top-k", source)
        self.assertIn("science_subprocess_live_streaming", source)
        self.assertIn("DEFERRED_TO_CONTROLLER_BEFORE_SCIENCE", source)
        self.assertNotIn(
            "observed_content = {role: _role_identity(path.parent)",
            source,
        )
        controller = (
            ROOT
            / "journal_extension"
            / "scripts"
            / "trackb_v4_execute_attached.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "observed_content = {\n        role: _role_content_identity(path.parent)",
            controller,
        )

    def test_v1_guard_hotfix_activates_in_fresh_sitecustomize_subprocess(self):
        hotfix_path = (
            ROOT
            / "journal_extension"
            / "operator_hotfixes"
            / "trackb_v5_v1_guard_hotfix.py"
        )
        hotfix_sha = "1ee18ea621a269c211100800c78bd25a21ef40bb150abdf47f17e49549f13711"

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            hotfix_root = root / "hotfix"
            hotfix_root.mkdir()
            (hotfix_root / hotfix_path.name).write_bytes(hotfix_path.read_bytes())
            (hotfix_root / "sitecustomize.py").write_text(
                "from trackb_v5_v1_guard_hotfix import install\n"
                f"install('{hotfix_sha}')\n",
                encoding="utf-8",
            )

            input_root = root / "input"
            for role in ("core", "gvlid_v5", "irish_potato"):
                role_root = input_root / role
                role_root.mkdir(parents=True)
                (role_root / "TRACKB_INPUT_MANIFEST.json").write_text(
                    json.dumps({"schema_version": "1.0", "role": role}),
                    encoding="utf-8",
                )
            historical_root = input_root / "historical"
            historical_root.mkdir(parents=True)
            historical_manifest = {
                "schema_version": "1.1",
                "role": "historical_compare",
                "coverage_scope": "V1_TRAIN_VAL_ONLY",
                "v1_test_image_bytes_accessed": False,
                "ext_i_eligible": False,
                "maximum_evidence_grade": "EXT-S",
            }
            historical_path = historical_root / "TRACKB_INPUT_MANIFEST.json"
            historical_path.write_text(json.dumps(historical_manifest), encoding="utf-8")

            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                [str(hotfix_root), str(SRC), env.get("PYTHONPATH", "")]
            )
            code = (
                "import json, os; "
                "from cropcop_je.trackb_r07 import discover_kaggle_inputs; "
                f"found=discover_kaggle_inputs({str(input_root)!r}); "
                f"assert os.environ.get('TRACKB_V1_GUARD_HOTFIX_ACTIVE') == {hotfix_sha!r}; "
                "print(json.dumps(sorted(found)))"
            )
            ok = subprocess.run(
                [sys.executable, "-c", code],
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(ok.returncode, 0, ok.stderr or ok.stdout)
            self.assertIn('"historical_compare"', ok.stdout)

            historical_manifest["v1_test_image_bytes_accessed"] = True
            historical_path.write_text(json.dumps(historical_manifest), encoding="utf-8")
            bad = subprocess.run(
                [sys.executable, "-c", code],
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("consumed V1-test access flag is not explicitly false", bad.stderr)

    def test_science_stream_wrapper_enforces_timeout_when_child_is_silent(self):
        import types

        hotfix_path = (
            ROOT
            / "journal_extension"
            / "operator_hotfixes"
            / "trackb_v5_v1_guard_hotfix.py"
        )
        spec = importlib.util.spec_from_file_location(
            "trackb_v5_stream_timeout_under_test",
            hotfix_path,
        )
        hotfix = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(hotfix)

        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "run_trackb_r07.py"
            script.write_text(
                "import time\ntime.sleep(5)\n",
                encoding="utf-8",
            )
            ops = types.SimpleNamespace(
                redact=lambda text: text,
                TrackBOpsError=RuntimeError,
            )
            def original_run_checked(args, *, cwd=None, timeout=3600):
                raise AssertionError("science runner must use streamed wrapper")
            wrapped = hotfix._stream_science_command(original_run_checked, ops)

            started = time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired):
                wrapped(
                    [sys.executable, str(script)],
                    cwd=td,
                    timeout=1,
                )
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 4.0)

    def test_claim_path_requires_single_writer_lease_and_durable_state_transitions(self):
        runner = (SCRIPTS / "run_trackb_r07.py").read_text(encoding="utf-8")
        executor = (SCRIPTS / "trackb_v4_execute_attached.py").read_text(encoding="utf-8")
        ops = (
            ROOT / "journal_extension" / "src" / "cropcop_je" / "trackb_r07_ops.py"
        ).read_text(encoding="utf-8")
        self.assertIn("acquire_claim_lease(", runner)
        self.assertIn("--claim-lease-dataset-slug", runner)
        self.assertIn("PASS_CLAIM_LEASE_ACQUIRED", ops)
        self.assertIn('"status": "PRIVATE_ARCHIVE_VERIFIED"', executor)
        self.assertIn('"status": "TRACK_B_CLOSED"', executor)
        self.assertIn("publish_attempt_state(args.attempt_dataset_slug, attempt_state)", runner)

    def test_external_transport_has_toc_tou_binding_and_signed_url_refresh(self):
        source = (
            ROOT / "journal_extension" / "src" / "cropcop_je" / "trackb_r07_ops.py"
        ).read_text(encoding="utf-8")
        self.assertIn("expected_source_manifest_sha256", source)
        self.assertIn("source changed between readiness resolution and acquisition", source)
        self.assertIn("_download_mendeley_record(", source)
        self.assertIn("signed-URL refresh", source)
        self.assertIn("source byte-size mismatch", source)
        self.assertIn("duplicate/case-colliding member path", source)




    def test_vendored_gvlid_checksum_authority_is_complete_and_pinned(self):
        from cropcop_je import trackb_r07_ops

        ledger = (
            ROOT
            / "journal_extension"
            / "track_b_r07"
            / "external_authority"
            / "gvlid_v5_checksums.csv"
        )
        entries = trackb_r07_ops._parse_gvlid_checksum_authority(ledger)
        self.assertEqual(len(entries), 3477)
        self.assertEqual(
            trackb_r07_ops._git_blob_sha1(ledger),
            "5c953cf0381614d8e3744737bfeed98ea1162f9c",
        )
        support = {}
        for label, _name in entries:
            support[label] = support.get(label, 0) + 1
        self.assertEqual(
            support,
            {
                "Black Rot": 808,
                "Esca": 888,
                "Healthy": 1109,
                "Leaf Blight": 672,
            },
        )

    def test_gvlid_checksum_authority_rejects_duplicate_identity(self):
        from cropcop_je import trackb_r07_ops

        with tempfile.TemporaryDirectory() as td:
            ledger = Path(td) / "checksums.csv"
            digest = "a" * 64
            ledger.write_text(
                "filename,sha256\n"
                f"GVLiD\\healthy\\x.jpg,{digest}\n"
                f"GVLiD\\healthy\\x.jpg,{digest}\n",
                encoding="utf-8",
            )
            with self.assertRaises(trackb_r07_ops.TrackBOpsError):
                trackb_r07_ops._parse_gvlid_checksum_authority(ledger)

    def test_gvlid_opaque_mendeley_transport_has_pinned_companion_fallback(self):
        source = (
            ROOT / "journal_extension" / "src" / "cropcop_je" / "trackb_r07_ops.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'GVLID_OFFICIAL_COMPANION_COMMIT = "878a1c7098964bb52c4c4a5c30e5b53340565258"',
            source,
        )
        self.assertIn(
            'GVLID_OFFICIAL_COMPANION_ARCHIVE_URL = (',
            source,
        )
        self.assertIn(
            'skip_single_opaque_container = (',
            source,
        )
        self.assertIn(
            'materialization_route = "PINNED_OFFICIAL_COMPANION_ARCHIVE_FALLBACK"',
            source,
        )
        self.assertIn(
            '_safe_extract_gvlid_companion_tar(',
            source,
        )
        self.assertIn(
            '_verify_gvlid_checksum_authority(',
            source,
        )
        self.assertLess(
            source.index('_safe_extract_gvlid_companion_tar('),
            source.index('supports, checksum_integrity = _normalize_gvlid_tree('),
        )

    def test_irish_streaming_normalizer_allows_identical_exact_duplicate_zip_path(self):
        import warnings
        import zipfile
        from cropcop_je.trackb_r07_ops import _normalize_irish_zip_streaming

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "irish.zip"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with zipfile.ZipFile(archive, "w") as zf:
                    zf.writestr("earlyblt/earlyblt1.jpg", b"same")
                    zf.writestr("earlyblt/earlyblt1.jpg", b"same")

            out = root / "out"
            receipt = _normalize_irish_zip_streaming(
                archive,
                out,
                expected_count=1,
                max_members=20,
                max_uncompressed_bytes=4096,
            )
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(receipt["duplicate_full_path_members"], 1)
            self.assertEqual(receipt["duplicate_members_collapsed"], 1)
            self.assertEqual(receipt["normalized_image_count"], 1)

    def test_irish_streaming_normalizer_rejects_conflicting_exact_duplicate_zip_path(self):
        import warnings
        import zipfile
        from cropcop_je.trackb_r07_ops import TrackBOpsError, _normalize_irish_zip_streaming

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "irish.zip"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with zipfile.ZipFile(archive, "w") as zf:
                    zf.writestr("earlyblt/earlyblt1.jpg", b"first")
                    zf.writestr("earlyblt/earlyblt1.jpg", b"DIFFERENT")

            with self.assertRaises(TrackBOpsError):
                _normalize_irish_zip_streaming(
                    archive,
                    root / "out",
                    expected_count=1,
                    max_members=20,
                    max_uncompressed_bytes=4096,
                )

    def test_irish_streaming_normalizer_ignores_macos_appledouble_jpg_metadata(self):
        import zipfile
        from cropcop_je.trackb_r07_ops import _normalize_irish_zip_streaming

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "irish.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("earlyblt/earlyblt1.jpg", b"real-one")
                zf.writestr("earlyblt/earlyblt2.jpg", b"real-two")
                zf.writestr("__MACOSX/earlyblt/._earlyblt1.jpg", b"appledouble-one")
                zf.writestr("__MACOSX/earlyblt/._earlyblt2.jpg", b"appledouble-two")

            out = root / "out"
            receipt = _normalize_irish_zip_streaming(
                archive,
                out,
                expected_count=2,
                max_members=20,
                max_uncompressed_bytes=4096,
            )
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(receipt["raw_image_member_count"], 2)
            self.assertEqual(receipt["transport_metadata_members_ignored"], 2)
            self.assertEqual(receipt["logical_unique_filename_count"], 2)
            self.assertEqual(receipt["normalized_image_count"], 2)
            self.assertEqual(len(list(out.glob("*.jpg"))), 2)

    def test_irish_streaming_normalizer_collapses_identical_duplicate_filenames(self):
        import zipfile
        from cropcop_je.trackb_r07_ops import _normalize_irish_zip_streaming

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "irish.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("copyA/img001.jpg", b"same")
                zf.writestr("copyB/img001.jpg", b"same")
                zf.writestr("copyA/img002.jpg", b"other")
                zf.writestr("copyB/img002.jpg", b"other")

            out = root / "out"
            receipt = _normalize_irish_zip_streaming(
                archive,
                out,
                expected_count=2,
                max_members=20,
                max_uncompressed_bytes=4096,
            )
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(receipt["raw_image_member_count"], 4)
            self.assertEqual(receipt["logical_unique_filename_count"], 2)
            self.assertEqual(receipt["normalized_image_count"], 2)
            self.assertEqual(receipt["duplicate_filename_groups"], 2)
            self.assertEqual(receipt["duplicate_members_collapsed"], 2)
            self.assertEqual(len(list(out.glob("*.jpg"))), 2)

    def test_irish_streaming_normalizer_rejects_conflicting_duplicate_filenames(self):
        import zipfile
        from cropcop_je.trackb_r07_ops import (
            TrackBOpsError,
            _normalize_irish_zip_streaming,
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "irish.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("copyA/img001.jpg", b"first")
                zf.writestr("copyB/img001.jpg", b"DIFFERENT")

            with self.assertRaises(TrackBOpsError):
                _normalize_irish_zip_streaming(
                    archive,
                    root / "out",
                    expected_count=1,
                    max_members=20,
                    max_uncompressed_bytes=4096,
                )

    def test_irish_streaming_normalizer_rejects_wrong_logical_count(self):
        import zipfile
        from cropcop_je.trackb_r07_ops import (
            TrackBOpsError,
            _normalize_irish_zip_streaming,
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "irish.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("a/img001.jpg", b"x")
                zf.writestr("b/img001.jpg", b"x")
            with self.assertRaises(TrackBOpsError):
                _normalize_irish_zip_streaming(
                    archive,
                    root / "out",
                    expected_count=2,
                    max_members=20,
                    max_uncompressed_bytes=4096,
                )

    def test_gvlid_companion_tar_extractor_handles_canonical_tree(self):
        import io
        import tarfile
        from cropcop_je.trackb_r07_ops import _safe_extract_gvlid_companion_tar

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "companion.tar.gz"
            with tarfile.open(archive, "w:gz") as tf:
                for name, payload in (
                    ("DNN-fixed/GVLiD/Black rot/a.jpg", b"aaa"),
                    ("DNN-fixed/GVLiD/healthy/b.jpg", b"bbb"),
                ):
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    tf.addfile(info, io.BytesIO(payload))

            destination = root / "out"
            receipt = _safe_extract_gvlid_companion_tar(
                archive,
                destination,
                expected_image_count=2,
                max_members=10,
                max_uncompressed_bytes=1024,
            )
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(receipt["selected_image_count"], 2)
            self.assertEqual(
                (destination / "GVLiD" / "Black rot" / "a.jpg").read_bytes(),
                b"aaa",
            )
            self.assertEqual(
                (destination / "GVLiD" / "healthy" / "b.jpg").read_bytes(),
                b"bbb",
            )

    def test_gvlid_companion_tar_extractor_rejects_traversal(self):
        import io
        import tarfile
        from cropcop_je.trackb_r07_ops import _safe_extract_gvlid_companion_tar

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as tf:
                payload = b"bad"
                info = tarfile.TarInfo("../GVLiD/healthy/bad.jpg")
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))
            with self.assertRaises(module.TrackBOpsError):
                _safe_extract_gvlid_companion_tar(
                    archive,
                    root / "out",
                    expected_image_count=1,
                    max_members=10,
                    max_uncompressed_bytes=1024,
                )

    def test_gvlid_companion_tar_extraction_is_bounded_and_path_safe(self):
        source = (
            ROOT / "journal_extension" / "src" / "cropcop_je" / "trackb_r07_ops.py"
        ).read_text(encoding="utf-8")
        start = source.index("def _safe_extract_gvlid_companion_tar(")
        end = source.index("def _load_lineage_review", start)
        helper = source[start:end]
        self.assertIn('expected_image_count: int = 3477', helper)
        self.assertIn('max_members: int = 5000', helper)
        self.assertIn('max_uncompressed_bytes: int = 4 * 1024 * 1024 * 1024', helper)
        self.assertIn('member_path.is_absolute()', helper)
        self.assertIn('".." in member_path.parts', helper)
        self.assertIn('if not member.isfile():', helper)
        self.assertIn('duplicate/case-colliding GVLiD companion image path', helper)
        self.assertIn('shutil.disk_usage(destination).free', helper)
        self.assertIn('with src, target.open("xb") as out:', helper)

    def test_gvlid_fallback_still_requires_full_3477_checksum_authority(self):
        source = (
            ROOT / "journal_extension" / "src" / "cropcop_je" / "trackb_r07_ops.py"
        ).read_text(encoding="utf-8")
        verifier_start = source.index("def _verify_gvlid_checksum_authority(")
        verifier_end = source.index("def _normalize_gvlid_tree(", verifier_start)
        verifier = source[verifier_start:verifier_end]
        self.assertIn("if verified != 3477", verifier)
        self.assertIn("len(consumed) != len(authority)", verifier)
        self.assertIn("missing", verifier)
        self.assertIn("observed_sha != row[\"sha256\"]", verifier)

    def test_external_source_lock_uses_pinned_gvlid_companion_ledger(self):
        lock = json.loads(
            (
                ROOT
                / "journal_extension"
                / "track_b_r07"
                / "TRACKB_EXTERNAL_SOURCE_LOCK_v2.json"
            ).read_text(encoding="utf-8")
        )
        row = lock["candidates"]["gvlid_v5"]
        self.assertTrue(row["materialization_permitted"])
        self.assertEqual(row["checksum_authority"]["official_repository"], "MilindGayakwad/DNN")
        self.assertEqual(
            row["checksum_authority"]["official_commit"],
            "878a1c7098964bb52c4c4a5c30e5b53340565258",
        )
        self.assertEqual(
            row["checksum_authority"]["official_blob_sha1"],
            "5c953cf0381614d8e3744737bfeed98ea1162f9c",
        )


if __name__ == "__main__":
    unittest.main()
