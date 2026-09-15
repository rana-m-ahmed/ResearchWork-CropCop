from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.g1 import factory_bundle_hash, validate_teacher_factory_bundle
from cropcop_je.hashing import sha256_file


class TrackAV12TeacherFactoryRootBindingTests(unittest.TestCase):
    @staticmethod
    def _manifest() -> dict:
        source = ROOT / "journal_extension" / "teacher_factory" / "historical_dino_tiny.py"
        if not source.is_file():
            raise AssertionError(f"frozen historical teacher source missing: {source}")
        manifest = {
            "schema_version": "1.0",
            "entrypoint": "historical_dino_tiny:build_teacher",
            "output_order_transform": "none",
            "files": [
                {
                    "path": "historical_dino_tiny.py",
                    "sha256": sha256_file(source),
                    "bytes": source.stat().st_size,
                }
            ],
        }
        manifest["bundle_sha256"] = factory_bundle_hash(manifest)
        return manifest

    def test_incident_reproduction_repo_root_is_wrong_for_sealed_relative_factory_path(self):
        errors = validate_teacher_factory_bundle(
            self._manifest(),
            source_root=ROOT,
            expected_entrypoint="historical_dino_tiny:build_teacher",
        )
        self.assertIn("teacher factory source file missing: historical_dino_tiny.py", errors)

    def test_frozen_teacher_factory_bundle_validates_against_dedicated_source_root(self):
        source_root = ROOT / "journal_extension" / "teacher_factory"
        errors = validate_teacher_factory_bundle(
            self._manifest(),
            source_root=source_root,
            expected_entrypoint="historical_dino_tiny:build_teacher",
        )
        self.assertEqual(errors, [])

    def test_versioned_training_wrapper_binds_exact_dedicated_factory_root(self):
        path = ROOT / "journal_extension" / "scripts" / "run_tracka_v12_training_v121.py"
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('source_root = repo / "journal_extension" / "teacher_factory"', source)
        self.assertIn("factory_source_root=source_root", source)
        self.assertIn("tracka_runtime.load_exact_teacher = load_exact_teacher_v121", source)
        self.assertIn("_historical_load_exact_teacher = tracka_runtime.load_exact_teacher", source)


if __name__ == "__main__":
    unittest.main()
