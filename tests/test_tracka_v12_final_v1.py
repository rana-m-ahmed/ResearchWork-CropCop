from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.hashing import sha256_file  # noqa: E402
from cropcop_je.tracka_v12_final_v1 import (  # noqa: E402
    CLASS_MAP_SHA256,
    MANIFEST_SHA256,
    FinalV1ResolutionError,
    _class_map_contract,
    _ordered_remote_candidates,
    _safe_extract_zip,
)


class FinalV1ResolverTests(unittest.TestCase):
    def test_constants_match_frozen_identity(self):
        self.assertEqual(
            MANIFEST_SHA256,
            "bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2",
        )
        self.assertEqual(
            CLASS_MAP_SHA256,
            "46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2",
        )

    def test_class_map_cardinality_guard(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "class_map.json"
            path.write_text(json.dumps({"num_classes": 119}), encoding="utf-8")
            with self.assertRaises(FinalV1ResolutionError):
                _class_map_contract(path)

    def test_candidate_ranking_prioritizes_finalized_dataset_over_cicps_and_models(self):
        kernel_refs = [
            "ranamuhammadahmed6/cropcop-cicps-kaggle-3-inputs-v1-1",
            "ranamuhammadahmed6/cropcop-finalized-v8-11-2026-1",
            "ranamuhammadahmed6/lastruncheckpoints-cropcop-prodrfdv",
            "ranamuhammadahmed6/cropcop-model-rfdv",
            "ranamuhammadahmed6/cropcop-120-class-dataset",
        ]
        ordered = _ordered_remote_candidates(kernel_refs, [])
        self.assertEqual(
            ordered[:2],
            [
                "ranamuhammadahmed6/cropcop-finalized-v8-11-2026-1",
                "ranamuhammadahmed6/cropcop-120-class-dataset",
            ],
        )
        self.assertGreater(
            ordered.index("ranamuhammadahmed6/cropcop-cicps-kaggle-3-inputs-v1-1"),
            1,
        )

    def test_zip_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../escape.txt", "blocked")
            with self.assertRaises(FinalV1ResolutionError):
                _safe_extract_zip(archive, root / "out")
            self.assertFalse((root.parent / "escape.txt").exists())

    def test_hash_function_is_byte_exact(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "x.bin"
            path.write_bytes(b"cropcop-final-v1")
            self.assertEqual(
                sha256_file(path),
                "fd64e31d2710c645d9065b6e07474ebfa76fb8eaeb06b54cc5ca70242bf9b40f",
            )


if __name__ == "__main__":
    unittest.main()
