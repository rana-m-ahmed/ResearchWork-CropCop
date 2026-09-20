from __future__ import annotations

import importlib.util
import sys
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


if __name__ == "__main__":
    unittest.main()
