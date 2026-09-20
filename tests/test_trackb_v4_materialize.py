from __future__ import annotations

import importlib.util
import sys
import json
import tempfile
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


class TrackBV4MaterializationTests(unittest.TestCase):
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

    def test_publication_releases_local_bundle_before_archive_roundtrip(self):
        source = (SCRIPTS / "trackb_v4_materialize.py").read_text(encoding="utf-8")
        infra_manifest = source.index(
            'infra_manifest = load_json(infra_root / "TRACKB_KAGGLE_CONTENT_MANIFEST.json")'
        )
        infra_delete = source.index("shutil.rmtree(infra_root, ignore_errors=True)", infra_manifest)
        infra_verify = source.index(
            'infra_pub["archive_roundtrip"] = _verify_published_archive_roundtrip(',
            infra_delete,
        )
        self.assertLess(infra_manifest, infra_delete)
        self.assertLess(infra_delete, infra_verify)

        external_manifest = source.index(
            'external_manifest = load_json(external_root / "TRACKB_KAGGLE_CONTENT_MANIFEST.json")'
        )
        external_delete = source.index(
            "shutil.rmtree(external_root, ignore_errors=True)",
            external_manifest,
        )
        external_verify = source.index(
            'external_pub["archive_roundtrip"] = _verify_published_archive_roundtrip(',
            external_delete,
        )
        self.assertLess(external_manifest, external_delete)
        self.assertLess(external_delete, external_verify)

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


if __name__ == "__main__":
    unittest.main()
