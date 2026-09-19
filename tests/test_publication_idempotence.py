from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.publication import (  # noqa: E402
    PublicationError,
    _verify_destination_snapshot,
    _verify_staged_subset,
)


class PublicationIdempotenceTests(unittest.TestCase):
    def test_restored_branch_allows_unchanged_approved_gate_to_be_absent_from_diff(self):
        prefix = (
            "journal_extension/evidence/public/track_a/posttraining/"
            "TRACKA-POST-r05-mnv4-teacher-s3-e08e471cc033"
        )
        allowed = [
            f"{prefix}/AUXILIARY_STATE_EVIDENCE_GATE.json",
            f"{prefix}/POSTTRAINING_PRIVATE_SYNC_CERTIFICATE.json",
            f"{prefix}/POSTTRAINING_STATE_COMPLETION.json",
            f"{prefix}/POSTTRAINING_PUBLICATION_MANIFEST.json",
        ]
        changed = [
            f"{prefix}/POSTTRAINING_PRIVATE_SYNC_CERTIFICATE.json",
            f"{prefix}/POSTTRAINING_STATE_COMPLETION.json",
            f"{prefix}/POSTTRAINING_PUBLICATION_MANIFEST.json",
        ]

        _verify_staged_subset(changed, allowed)

    def test_staged_subset_rejects_any_path_outside_explicit_allowlist(self):
        with self.assertRaises(PublicationError):
            _verify_staged_subset(
                ["journal_extension/evidence/public/runs/SAFE.json", "private/secret.json"],
                ["journal_extension/evidence/public/runs/SAFE.json"],
            )

    def test_destination_snapshot_requires_exact_allowlisted_file_set_and_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()

            gate = source / "AUXILIARY_STATE_EVIDENCE_GATE.json"
            sync = source / "POSTTRAINING_PRIVATE_SYNC_CERTIFICATE.json"
            gate.write_text('{"status":"PASS"}\n', encoding="utf-8")
            sync.write_text('{"status":"PASS","generation_kind":"final"}\n', encoding="utf-8")

            (destination / gate.name).write_bytes(gate.read_bytes())
            (destination / sync.name).write_bytes(sync.read_bytes())

            _verify_destination_snapshot(destination, [gate, sync])

            (destination / "UNEXPECTED.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(PublicationError):
                _verify_destination_snapshot(destination, [gate, sync])

    def test_destination_snapshot_rejects_same_name_with_different_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()

            gate = source / "AUXILIARY_STATE_EVIDENCE_GATE.json"
            gate.write_text('{"status":"PASS"}\n', encoding="utf-8")
            (destination / gate.name).write_text('{"status":"FAIL"}\n', encoding="utf-8")

            with self.assertRaises(PublicationError):
                _verify_destination_snapshot(destination, [gate])


if __name__ == "__main__":
    unittest.main()
