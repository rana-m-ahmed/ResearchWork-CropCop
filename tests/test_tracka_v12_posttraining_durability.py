from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
SCRIPTS = ROOT / "journal_extension" / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cropcop_je.hashing import sha256_file, sha256_json  # noqa: E402
from sync_tracka_v12_posttraining_evidence import file_manifest, verify_download  # noqa: E402


class TrackAPosttrainingDurabilityTests(unittest.TestCase):
    def make_generation(self, root: Path, *, kind: str = "partial"):
        evidence = root / "attempts" / "direct" / "attempt-001" / "public_evidence"
        evidence.mkdir(parents=True)
        gate = evidence / "DIRECT_STATE_EVIDENCE_GATE.json"
        gate.write_text('{"status":"PASS"}\n', encoding="utf-8")
        private = root / "attempts" / "direct" / "attempt-001" / "private_predictions.jsonl"
        private.write_text('{"row":"x"}\n', encoding="utf-8")
        manifest = file_manifest(root)
        manifest_sha = sha256_json(manifest)
        marker = {
            "schema_version": "1.0",
            "run_id": "TRACKA-POST-R04",
            "experiment_id": "R04-MNV4-DIRECT-S1",
            "analysis_source_git_commit": "a" * 40,
            "generation_kind": kind,
            "sync_nonce": "nonce",
            "previous_version_number": 1,
            "files": manifest,
            "evidence_manifest_sha256": manifest_sha,
            "complete": True,
        }
        marker_path = root / "POSTTRAINING_DURABILITY_MARKER.json"
        marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest, manifest_sha

    def test_partial_generation_roundtrip_verifies(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _manifest, manifest_sha = self.make_generation(root, kind="partial")
            marker = verify_download(
                root,
                run_id="TRACKA-POST-R04",
                experiment_id="R04-MNV4-DIRECT-S1",
                nonce="nonce",
                manifest_sha=manifest_sha,
                generation_kind="partial",
            )
            self.assertEqual(marker["generation_kind"], "partial")

    def test_final_generation_roundtrip_verifies(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _manifest, manifest_sha = self.make_generation(root, kind="final")
            marker = verify_download(
                root,
                run_id="TRACKA-POST-R04",
                experiment_id="R04-MNV4-DIRECT-S1",
                nonce="nonce",
                manifest_sha=manifest_sha,
                generation_kind="final",
            )
            self.assertEqual(marker["generation_kind"], "final")

    def test_generation_kind_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _manifest, manifest_sha = self.make_generation(root, kind="partial")
            with self.assertRaises(RuntimeError):
                verify_download(
                    root,
                    run_id="TRACKA-POST-R04",
                    experiment_id="R04-MNV4-DIRECT-S1",
                    nonce="nonce",
                    manifest_sha=manifest_sha,
                    generation_kind="final",
                )

    def test_file_tampering_fails_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest, manifest_sha = self.make_generation(root, kind="partial")
            rel = next(iter(manifest))
            (root / rel).write_bytes(b"tampered")
            with self.assertRaises(RuntimeError):
                verify_download(
                    root,
                    run_id="TRACKA-POST-R04",
                    experiment_id="R04-MNV4-DIRECT-S1",
                    nonce="nonce",
                    manifest_sha=manifest_sha,
                    generation_kind="partial",
                )

    def test_marker_is_excluded_from_source_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.make_generation(root, kind="partial")
            manifest = file_manifest(root)
            self.assertNotIn("POSTTRAINING_DURABILITY_MARKER.json", manifest)
            self.assertTrue(all(len(row["sha256"]) == 64 for row in manifest.values()))
            self.assertTrue(all(sha256_file(root / rel) == row["sha256"] for rel, row in manifest.items()))


if __name__ == "__main__":
    unittest.main()
