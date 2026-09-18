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
from audit_tracka_v12_posttraining_evidence import validate_published_completion_chain  # noqa: E402

EXPERIMENT = "R13-VIT-DLITTLE-DIFF-CONTEXT-S1"
ANALYSIS = "a" * 40


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class TrackAPosttrainingEvidenceChainTests(unittest.TestCase):
    def build(self, root: Path):
        sync = {
            "schema_version": "1.0",
            "status": "PASS",
            "sync_kind": "track_a_posttraining_private_evidence",
            "experiment_id": EXPERIMENT,
            "run_id": "TRACKA-POST-R13",
            "analysis_source_git_commit": ANALYSIS,
            "dataset_locator": "owner/private-evidence",
            "owner_matches_authenticated_user": True,
            "previous_version_number": 1,
            "confirmed_version_number": 2,
            "evidence_file_count": 4,
            "evidence_manifest_sha256": "1" * 64,
            "roundtrip_marker_sha256": "2" * 64,
            "generation_roundtrip_verified": True,
            "private_dataset_verified": True,
        }
        sync["sync_certificate_sha256"] = sha256_json(sync)
        sync_path = root / "POSTTRAINING_PRIVATE_SYNC_CERTIFICATE.json"
        write_json(sync_path, sync)

        completion = {
            "schema_version": "1.0",
            "status": "PASS",
            "completion_kind": "track_a_posttraining_state",
            "experiment_id": EXPERIMENT,
            "run_id": "JE-R13",
            "posttraining_public_run_id": "TRACKA-POST-R13",
            "analysis_source_git_commit": ANALYSIS,
            "role": "direct",
            "stage_gates": {},
            "private_sync_certificate_sha256": sha256_file(sync_path),
            "private_evidence_dataset_locator": "owner/private-evidence",
            "private_generation_roundtrip_verified": True,
            "publication_branch": "run-evidence/TRACKA-POST-R13",
            "training_or_adaptation_performed": False,
            "optimizer_state_advanced": False,
            "v1_test_accessed": False,
            "external_surface_accessed": False,
        }
        completion["completion_sha256"] = sha256_json(completion)
        completion_path = root / "POSTTRAINING_STATE_COMPLETION.json"
        write_json(completion_path, completion)
        return completion, {
            "POSTTRAINING_STATE_COMPLETION.json": completion_path,
            "POSTTRAINING_PRIVATE_SYNC_CERTIFICATE.json": sync_path,
        }

    def test_valid_chain_passes(self):
        with tempfile.TemporaryDirectory() as td:
            completion, files = self.build(Path(td))
            result = validate_published_completion_chain(
                files=files,
                account_completion=completion,
                experiment_id=EXPERIMENT,
                analysis_sha=ANALYSIS,
            )
            self.assertTrue(result["private_generation_roundtrip_verified"])

    def test_published_completion_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            completion, files = self.build(Path(td))
            tampered = dict(completion)
            tampered["role"] = "auxiliary"
            with self.assertRaises(RuntimeError):
                validate_published_completion_chain(
                    files=files,
                    account_completion=tampered,
                    experiment_id=EXPERIMENT,
                    analysis_sha=ANALYSIS,
                )

    def test_sync_certificate_tampering_fails(self):
        with tempfile.TemporaryDirectory() as td:
            completion, files = self.build(Path(td))
            sync = json.loads(files["POSTTRAINING_PRIVATE_SYNC_CERTIFICATE.json"].read_text(encoding="utf-8"))
            sync["generation_roundtrip_verified"] = False
            write_json(files["POSTTRAINING_PRIVATE_SYNC_CERTIFICATE.json"], sync)
            with self.assertRaises(RuntimeError):
                validate_published_completion_chain(
                    files=files,
                    account_completion=completion,
                    experiment_id=EXPERIMENT,
                    analysis_sha=ANALYSIS,
                )

    def test_dataset_locator_tampering_fails(self):
        with tempfile.TemporaryDirectory() as td:
            completion, files = self.build(Path(td))
            sync = json.loads(files["POSTTRAINING_PRIVATE_SYNC_CERTIFICATE.json"].read_text(encoding="utf-8"))
            sync["dataset_locator"] = "other/private-evidence"
            sync.pop("sync_certificate_sha256")
            sync["sync_certificate_sha256"] = sha256_json(sync)
            write_json(files["POSTTRAINING_PRIVATE_SYNC_CERTIFICATE.json"], sync)
            with self.assertRaises(RuntimeError):
                validate_published_completion_chain(
                    files=files,
                    account_completion=completion,
                    experiment_id=EXPERIMENT,
                    analysis_sha=ANALYSIS,
                )


if __name__ == "__main__":
    unittest.main()
