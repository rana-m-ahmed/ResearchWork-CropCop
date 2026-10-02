from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je import persistence_v8 as p8  # noqa: E402
from cropcop_je.persistence import PersistenceError  # noqa: E402


def write_generation(root: Path, *, run_id: str, segment_id: str, nonce: str, corrupt_index_hash: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    index = root / "checkpoint_index.json"
    index.write_text('{"schema_version":"2.0"}\n', encoding="utf-8")
    marker = {
        "schema_version": "2.0",
        "run_id": run_id,
        "segment_id": segment_id,
        "sync_nonce": nonce,
        "checkpoint_index_sha256": "0" * 64 if corrupt_index_hash else p8.sha256_file(index),
        "previous_version_number": 4,
        "complete": True,
    }
    (root / "durable_sync.json").write_text(json.dumps(marker), encoding="utf-8")


class TrackAV12MitigationV9Tests(unittest.TestCase):
    def test_exact_roundtrip_retries_transient_download_failure(self):
        store = p8.GenerationAwareKagglePrivateDatasetStore("owner/cropcop-run")
        calls = {"count": 0}

        def fake_download(destination: Path) -> None:
            calls["count"] += 1
            if calls["count"] == 1:
                raise subprocess.CalledProcessError(1, ["kaggle", "datasets", "download"])
            write_generation(destination, run_id="RUN-1", segment_id="SEG-1", nonce="nonce-new")

        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(store, "_download_latest", side_effect=fake_download), \
             mock.patch.object(p8.time, "sleep"):
            marker = store._wait_for_exact_downloadable_generation(
                Path(td) / "roundtrip",
                run_id="RUN-1",
                segment_id="SEG-1",
                expected_nonce="nonce-new",
                timeout_seconds=2.0,
                poll_interval_seconds=0.0,
            )
        self.assertEqual(calls["count"], 2)
        self.assertEqual(marker["sync_nonce"], "nonce-new")

    def test_exact_roundtrip_retries_stale_generation_until_expected_nonce(self):
        store = p8.GenerationAwareKagglePrivateDatasetStore("owner/cropcop-run")
        calls = {"count": 0}

        def fake_download(destination: Path) -> None:
            calls["count"] += 1
            nonce = "nonce-old" if calls["count"] == 1 else "nonce-new"
            write_generation(destination, run_id="RUN-1", segment_id="SEG-1", nonce=nonce)

        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(store, "_download_latest", side_effect=fake_download), \
             mock.patch.object(p8.time, "sleep"):
            marker = store._wait_for_exact_downloadable_generation(
                Path(td) / "roundtrip",
                run_id="RUN-1",
                segment_id="SEG-1",
                expected_nonce="nonce-new",
                timeout_seconds=2.0,
                poll_interval_seconds=0.0,
            )
        self.assertEqual(calls["count"], 2)
        self.assertEqual(marker["sync_nonce"], "nonce-new")

    def test_corrupt_expected_generation_fails_closed_without_retry(self):
        store = p8.GenerationAwareKagglePrivateDatasetStore("owner/cropcop-run")
        calls = {"count": 0}

        def fake_download(destination: Path) -> None:
            calls["count"] += 1
            write_generation(
                destination,
                run_id="RUN-1",
                segment_id="SEG-1",
                nonce="nonce-new",
                corrupt_index_hash=True,
            )

        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(store, "_download_latest", side_effect=fake_download), \
             mock.patch.object(p8.time, "sleep"):
            with self.assertRaisesRegex(PersistenceError, "checkpoint-index hash mismatch"):
                store._wait_for_exact_downloadable_generation(
                    Path(td) / "roundtrip",
                    run_id="RUN-1",
                    segment_id="SEG-1",
                    expected_nonce="nonce-new",
                    timeout_seconds=2.0,
                    poll_interval_seconds=0.0,
                )
        self.assertEqual(calls["count"], 1)

    def test_restore_uses_bounded_exact_downloadable_generation_path(self):
        text = (SRC / "cropcop_je" / "persistence_v8.py").read_text(encoding="utf-8")
        self.assertIn("self._wait_for_exact_downloadable_generation(staging, run_id=run_id)", text)
        self.assertNotIn("self._download_latest(staging)\n            self._validate_generation_payload(staging, run_id=run_id)", text)


if __name__ == "__main__":
    unittest.main()
