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

import master_g1a_v8 as g1a
import tracka_v12_kaggle_operator_v8 as v8


class TrackAV12KaggleMasterV8R4Tests(unittest.TestCase):
    def test_exact_generation_wait_retries_old_placeholder_and_stale_valid_generation(self):
        expected = "a" * 64
        stale = "b" * 64
        calls = [
            v8.OperatorError("G1A bundle resolution must be unique, found 0: []"),
            (Path("/tmp/stale"), {"g1a_seal_sha256": stale}),
            (Path("/tmp/exact"), {"g1a_seal_sha256": expected}),
        ]

        def fake_download(*args, **kwargs):
            item = calls.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(g1a, "download_and_validate_g1a_dataset", side_effect=fake_download) as download, \
             mock.patch.object(g1a, "dependency_wait_expired", return_value=False), \
             mock.patch.object(g1a.time, "sleep") as sleep:
            bundle, seal = g1a.wait_for_exact_g1a_generation(
                Path(td),
                "owner/canonical-g1a",
                Path(td) / "download",
                env={},
                expected_seal_sha256=expected,
            )

        self.assertEqual(bundle, Path("/tmp/exact"))
        self.assertEqual(seal["g1a_seal_sha256"], expected)
        self.assertEqual(download.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_exact_generation_wait_fails_closed_on_session_budget(self):
        expected = "c" * 64
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(
                 g1a,
                 "download_and_validate_g1a_dataset",
                 side_effect=v8.OperatorError("G1A bundle resolution must be unique, found 0: []"),
             ), \
             mock.patch.object(g1a, "dependency_wait_expired", return_value=True), \
             mock.patch.object(g1a.time, "sleep") as sleep:
            with self.assertRaises(TimeoutError):
                g1a.wait_for_exact_g1a_generation(
                    Path(td),
                    "owner/canonical-g1a",
                    Path(td) / "download",
                    env={},
                    expected_seal_sha256=expected,
                )
        sleep.assert_not_called()

    def test_exact_generation_wait_rejects_invalid_expected_seal(self):
        with self.assertRaises(v8.OperatorError):
            g1a.wait_for_exact_g1a_generation(
                Path("/tmp/repo"),
                "owner/canonical-g1a",
                Path("/tmp/download"),
                env={},
                expected_seal_sha256="short",
            )

    def test_k1_g1a_path_binds_roundtrip_to_newly_built_seal(self):
        text = (OPS / "master_g1a_v8.py").read_text(encoding="utf-8")
        build = text.index("seal = build_g1a_once")
        version = text.index("version_private_dataset(locator, bundle")
        exact_wait = text.index("roundtrip_bundle, roundtrip_seal = wait_for_exact_g1a_generation")
        expected_binding = text.index('expected_seal_sha256=seal["g1a_seal_sha256"]')
        self.assertLess(build, version)
        self.assertLess(version, exact_wait)
        self.assertLess(exact_wait, expected_binding)

    def test_science_authority_is_unchanged(self):
        self.assertEqual(v8.SCIENCE_SHA, "4ced2fd7c764c07fa47fb57fbea38376d2ce61a4")


if __name__ == "__main__":
    unittest.main()
