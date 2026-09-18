from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "journal_extension" / "scripts"
SRC = ROOT / "journal_extension" / "src"
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cropcop_je.hashing import sha256_file, sha256_json  # noqa: E402
import run_tracka_v12_posttraining_account as operator  # noqa: E402

ANALYSIS = "a" * 40
EXPERIMENT = "R13-VIT-DLITTLE-DIFF-CONTEXT-S1"
RUN = "JE-R13-VIT-DLITTLE-DIFF-CONTEXT-S1-56023042e577-A01"


def write_gate(root: Path, stage: str, payload: dict) -> Path:
    attempt = root / "attempt-001"
    gate = attempt / operator.STAGE_GATE[stage]
    gate.parent.mkdir(parents=True, exist_ok=True)
    gate.write_text(json.dumps(payload), encoding="utf-8")
    return attempt



def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_local_completion_chain(root: Path) -> tuple[dict, dict, dict]:
    cert_dir = root / "certificates"
    sync = {
        "schema_version": "1.0",
        "status": "PASS",
        "generation_kind": "final",
        "experiment_id": EXPERIMENT,
        "run_id": "TRACKA-POST-R13",
        "analysis_source_git_commit": ANALYSIS,
        "dataset_locator": "owner/private-evidence",
        "generation_roundtrip_verified": True,
        "private_dataset_verified": True,
    }
    sync["sync_certificate_sha256"] = sha256_json(sync)
    sync_path = cert_dir / "POSTTRAINING_PRIVATE_SYNC_CERTIFICATE.json"
    write_json(sync_path, sync)

    completion = {
        "schema_version": "1.0",
        "status": "PASS",
        "completion_kind": "track_a_posttraining_state",
        "experiment_id": EXPERIMENT,
        "run_id": RUN,
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
    write_json(cert_dir / "POSTTRAINING_STATE_COMPLETION.json", completion)

    publication = {
        "schema_version": "1.0",
        "status": "PASS",
        "publication_kind": "track_a_posttraining_state",
        "experiment_id": EXPERIMENT,
        "run_id": "TRACKA-POST-R13",
        "analysis_source_git_commit": ANALYSIS,
        "public_file_sha256": {},
        "public_file_count": 0,
        "private_material_published": False,
        "publication_manifest_sha256": "1" * 64,
        "publication_branch": "run-evidence/TRACKA-POST-R13",
        "publication_manifest_file_sha256": "2" * 64,
    }
    publication["publication_certificate_sha256"] = sha256_json(publication)
    write_json(cert_dir / "POSTTRAINING_PUBLICATION_CERTIFICATE.json", publication)
    return completion, publication, sync


class TrackAPosttrainingOperatorTests(unittest.TestCase):
    def base_gate(self, stage: str) -> dict:
        payload = {
            "status": "PASS",
            "experiment_id": EXPERIMENT,
            "run_id": RUN,
            "analysis_source_git_commit": ANALYSIS,
            "v1_test_accessed": False,
            "external_surface_accessed": False,
        }
        if stage in {"direct", "auxiliary"}:
            payload.update(
                {
                    "training_performed": False,
                    "optimizer_state_advanced": False,
                    "replay_gate": {"status": "PASS"},
                }
            )
        else:
            payload["training_or_adaptation_performed"] = False
        return payload

    def test_terminal_direct_attempt_reused_only_when_all_gates_hold(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            expected = write_gate(root, "direct", self.base_gate("direct"))
            self.assertEqual(operator.terminal_stage(root, "direct", EXPERIMENT, RUN, ANALYSIS), expected)

    def test_wrong_analysis_sha_is_not_reused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gate = self.base_gate("direct")
            gate["analysis_source_git_commit"] = "b" * 40
            write_gate(root, "direct", gate)
            self.assertIsNone(operator.terminal_stage(root, "direct", EXPERIMENT, RUN, ANALYSIS))

    def test_open_v1_surface_is_not_reused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gate = self.base_gate("direct")
            gate["v1_test_accessed"] = True
            write_gate(root, "direct", gate)
            self.assertIsNone(operator.terminal_stage(root, "direct", EXPERIMENT, RUN, ANALYSIS))

    def test_training_marker_is_not_reused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gate = self.base_gate("auxiliary")
            gate["training_performed"] = True
            write_gate(root, "auxiliary", gate)
            self.assertIsNone(operator.terminal_stage(root, "auxiliary", EXPERIMENT, RUN, ANALYSIS))

    def test_xai_warning_is_terminal_but_failed_xai_is_not(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gate = self.base_gate("xai")
            gate["status"] = "WARNING_NONFINITE_MAPS"
            expected = write_gate(root, "xai", gate)
            self.assertEqual(operator.terminal_stage(root, "xai", EXPERIMENT, RUN, ANALYSIS), expected)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gate = self.base_gate("xai")
            gate["status"] = "FAIL"
            write_gate(root, "xai", gate)
            self.assertIsNone(operator.terminal_stage(root, "xai", EXPERIMENT, RUN, ANALYSIS))

    def test_executor_args_are_allowlisted(self):
        self.assertEqual(
            operator.cli_args({"manifest": "/m.csv", "batch_size": 16}, operator.DIRECT_ALLOWED),
            ["--batch-size", "16", "--manifest", "/m.csv"],
        )
        with self.assertRaises(RuntimeError):
            operator.cli_args({"v1_test": "/forbidden"}, operator.DIRECT_ALLOWED)


    def test_local_completion_reuse_requires_full_durability_publication_chain(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            completion, _publication, _sync = build_local_completion_chain(root)
            observed = operator.load_local_completed_state(root, EXPERIMENT, ANALYSIS)
            self.assertEqual(observed, completion)

    def test_local_completion_without_publication_certificate_is_not_reused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            build_local_completion_chain(root)
            (root / "certificates" / "POSTTRAINING_PUBLICATION_CERTIFICATE.json").unlink()
            self.assertIsNone(operator.load_local_completed_state(root, EXPERIMENT, ANALYSIS))

    def test_local_completion_with_tampered_sync_certificate_is_not_reused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            build_local_completion_chain(root)
            sync_path = root / "certificates" / "POSTTRAINING_PRIVATE_SYNC_CERTIFICATE.json"
            payload = json.loads(sync_path.read_text(encoding="utf-8"))
            payload["generation_roundtrip_verified"] = False
            write_json(sync_path, payload)
            self.assertIsNone(operator.load_local_completed_state(root, EXPERIMENT, ANALYSIS))

    def test_local_completion_with_wrong_publication_branch_is_not_reused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            build_local_completion_chain(root)
            publication_path = root / "certificates" / "POSTTRAINING_PUBLICATION_CERTIFICATE.json"
            payload = json.loads(publication_path.read_text(encoding="utf-8"))
            payload["publication_branch"] = "run-evidence/wrong"
            payload.pop("publication_certificate_sha256")
            payload["publication_certificate_sha256"] = sha256_json(payload)
            write_json(publication_path, payload)
            self.assertIsNone(operator.load_local_completed_state(root, EXPERIMENT, ANALYSIS))

    def test_attempt_numbers_are_append_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "attempt-001").mkdir(parents=True)
            (root / "attempt-004").mkdir()
            self.assertEqual(operator.next_attempt(root).name, "attempt-005")


if __name__ == "__main__":
    unittest.main()
