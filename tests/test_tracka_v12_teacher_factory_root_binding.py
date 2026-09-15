from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
SCRIPTS = ROOT / "journal_extension" / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cropcop_je.g1 import factory_bundle_hash, validate_teacher_factory_bundle
from cropcop_je.hashing import sha256_file
import run_tracka_v12_training_v121 as v121


class TrackAV12TeacherFactoryRootBindingTests(unittest.TestCase):
    def test_frozen_teacher_factory_bundle_validates_against_dedicated_source_root(self):
        source_root = ROOT / "journal_extension" / "teacher_factory"
        source = source_root / "historical_dino_tiny.py"
        self.assertTrue(source.is_file())
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
        errors = validate_teacher_factory_bundle(
            manifest,
            source_root=source_root,
            expected_entrypoint="historical_dino_tiny:build_teacher",
        )
        self.assertEqual(errors, [])

    def test_versioned_training_wrapper_rebinds_repo_root_to_frozen_factory_directory(self):
        captured = {}
        original = v121._historical_load_exact_teacher

        def fake_loader(
            checkpoint_path,
            *,
            factory_spec,
            factory_bundle_manifest=None,
            repo_root=".",
            factory_source_root=None,
        ):
            captured.update(
                checkpoint_path=checkpoint_path,
                factory_spec=factory_spec,
                factory_bundle_manifest=factory_bundle_manifest,
                repo_root=Path(repo_root).resolve(),
                factory_source_root=Path(factory_source_root).resolve(),
            )
            return "teacher", {"bundle_sha256": "b" * 64}

        v121._historical_load_exact_teacher = fake_loader
        try:
            result = v121.load_exact_teacher_v121(
                "teacher.pt",
                factory_spec="historical_dino_tiny:build_teacher",
                factory_bundle_manifest="TEACHER_FACTORY_BUNDLE.json",
                repo_root=ROOT,
                factory_source_root=ROOT,
            )
        finally:
            v121._historical_load_exact_teacher = original

        self.assertEqual(result[0], "teacher")
        self.assertEqual(captured["repo_root"], ROOT.resolve())
        self.assertEqual(
            captured["factory_source_root"],
            (ROOT / "journal_extension" / "teacher_factory").resolve(),
        )
        self.assertNotEqual(captured["factory_source_root"], ROOT.resolve())


if __name__ == "__main__":
    unittest.main()
