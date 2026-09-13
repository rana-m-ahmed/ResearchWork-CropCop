from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
SCRIPTS = ROOT / "journal_extension" / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cropcop_je.hashing import sha256_json  # noqa: E402
from cropcop_je.tracka_v12_authorization import REQUIRED_PRE_SCIENCE_GATES  # noqa: E402
import seal_tracka_v12_science_go as go  # noqa: E402


def attest(kind: str, source: str, extra=None):
    payload = {
        "schema_version": "1.0",
        "attestation_kind": kind,
        "status": "PASS",
        "source_git_commit": source,
        "github_actions": True,
        "workflow_run_id": "34740000000",
        "workflow_run_attempt": "1",
        "pull_request_head_sha": source,
        **(extra or {}),
    }
    payload["attestation_sha256"] = sha256_json(payload)
    return payload


class TrackAV12ScienceGoTests(unittest.TestCase):
    def fixtures(self):
        source = "1" * 40
        static_names = sorted(go.STATIC_GATES)
        code = attest(
            "track_a_v12_pre_science_code",
            source,
            {
                "static_pre_science_gates": {name: "PASS" for name in static_names},
                "science_diff_sha256": {"principal": "e" * 64, "secondary": "f" * 64},
                "science_diff_reports": {"principal": {"status": "PASS"}, "secondary": {"status": "PASS"}},
            },
        )
        lock = attest(
            "track_a_v12_exact_head_lock_runtime",
            source,
            {
                "science_authorized": False,
                "immutable_v12_lock": "PASS",
                "v121_runtime_qualification": "PASS",
                "candidate_claim_boundary_lock": "PASS",
                "content_lock_report_sha256": "a" * 64,
                "runtime_report_sha256": "b" * 64,
                "candidate_claim_boundary_sha256": "c" * 64,
                "xai_operationalization_sha256": "d" * 64,
                "content_lock_report": {"overall_status": "PASS", "static": {"status": "PASS", "science_authorized": False}},
                "runtime_report": {"status": "PASS", "science_authorized": False, "qualified_target": "blocks.13.norm1"},
            },
        )
        g1a = {
            "schema_version": "1.0",
            "status": "PASS",
            "science_authorized": False,
            "source_git_sha": source,
            "g1a_seal_sha256": "2" * 64,
            "dependency_lock_sha256": "3" * 64,
        }
        g2a = {
            "status": "PASS",
            "source_git_commit": source,
            "g1a_seal_sha256": "2" * 64,
            "dependency_lock_sha256": "3" * 64,
            "barrier_sha256": "4" * 64,
        }
        scheduler = {
            "status": "PASS",
            "source_git_commit": source,
            "g1a_seal_sha256": "2" * 64,
            "g2a_barrier_sha256": "4" * 64,
            "scheduler_freeze_sha256": "5" * 64,
        }
        hashes = {name: (str(index) * 64)[:64] for index, name in enumerate(("code_attestation", "lock_runtime_attestation", "g1a", "g2a", "scheduler"), start=6)}
        return source, code, lock, g1a, g2a, scheduler, hashes

    def compose(self, source, code, lock, g1a, g2a, scheduler, hashes):
        return go.compose_gate_inputs(
            source_git_commit=source,
            code_attestation=code,
            lock_runtime_attestation=lock,
            g1a=g1a,
            g2a=g2a,
            scheduler=scheduler,
            file_hashes=hashes,
        )

    @patch.object(go, "validate_scheduler_freeze_v122", return_value=[])
    @patch.object(go, "validate_g2a_v122_barrier", return_value=[])
    @patch.object(go, "validate_g1a_seal_object", return_value=[])
    def test_exact_valid_inventory_composes(self, _g1, _g2, _scheduler):
        source, code, lock, g1a, g2a, scheduler, hashes = self.fixtures()
        gates, bindings = self.compose(source, code, lock, g1a, g2a, scheduler, hashes)
        self.assertEqual(set(gates), set(REQUIRED_PRE_SCIENCE_GATES))
        self.assertEqual(set(bindings), set(REQUIRED_PRE_SCIENCE_GATES))
        self.assertEqual(set(gates.values()), {"PASS"})
        self.assertEqual({row.get("workflow_run_id") for name, row in bindings.items() if name in go.STATIC_GATES}, {code["workflow_run_id"]})

    @patch.object(go, "validate_scheduler_freeze_v122", return_value=[])
    @patch.object(go, "validate_g2a_v122_barrier", return_value=[])
    @patch.object(go, "validate_g1a_seal_object", return_value=[])
    def test_stale_code_attestation_is_rejected(self, _g1, _g2, _scheduler):
        source, code, lock, g1a, g2a, scheduler, hashes = self.fixtures()
        code["source_git_commit"] = "9" * 40
        code["attestation_sha256"] = sha256_json({k: v for k, v in code.items() if k != "attestation_sha256"})
        with self.assertRaises(ValueError):
            self.compose(source, code, lock, g1a, g2a, scheduler, hashes)

    @patch.object(go, "validate_scheduler_freeze_v122", return_value=[])
    @patch.object(go, "validate_g2a_v122_barrier", return_value=[])
    @patch.object(go, "validate_g1a_seal_object", return_value=[])
    def test_non_ci_shaped_code_attestation_is_rejected(self, _g1, _g2, _scheduler):
        source, code, lock, g1a, g2a, scheduler, hashes = self.fixtures()
        code["github_actions"] = False
        code["workflow_run_id"] = None
        code["attestation_sha256"] = sha256_json({k: v for k, v in code.items() if k != "attestation_sha256"})
        with self.assertRaises(ValueError):
            self.compose(source, code, lock, g1a, g2a, scheduler, hashes)

    @patch.object(go, "validate_scheduler_freeze_v122", return_value=[])
    @patch.object(go, "validate_g2a_v122_barrier", return_value=[])
    @patch.object(go, "validate_g1a_seal_object", return_value=[])
    def test_failed_embedded_science_diff_is_rejected(self, _g1, _g2, _scheduler):
        source, code, lock, g1a, g2a, scheduler, hashes = self.fixtures()
        code["science_diff_reports"]["principal"]["status"] = "FAIL"
        code["attestation_sha256"] = sha256_json({k: v for k, v in code.items() if k != "attestation_sha256"})
        with self.assertRaises(ValueError):
            self.compose(source, code, lock, g1a, g2a, scheduler, hashes)

    @patch.object(go, "validate_scheduler_freeze_v122", return_value=[])
    @patch.object(go, "validate_g2a_v122_barrier", return_value=[])
    @patch.object(go, "validate_g1a_seal_object", return_value=[])
    def test_lock_runtime_kind_or_target_drift_is_rejected(self, _g1, _g2, _scheduler):
        source, code, lock, g1a, g2a, scheduler, hashes = self.fixtures()
        lock["attestation_kind"] = "wrong"
        lock["runtime_report"]["qualified_target"] = "wrong.target"
        lock["attestation_sha256"] = sha256_json({k: v for k, v in lock.items() if k != "attestation_sha256"})
        with self.assertRaises(ValueError):
            self.compose(source, code, lock, g1a, g2a, scheduler, hashes)

    @patch.object(go, "validate_scheduler_freeze_v122", return_value=[])
    @patch.object(go, "validate_g2a_v122_barrier", return_value=[])
    @patch.object(go, "validate_g1a_seal_object", return_value=[])
    def test_dependency_lock_mismatch_is_rejected(self, _g1, _g2, _scheduler):
        source, code, lock, g1a, g2a, scheduler, hashes = self.fixtures()
        g2a["dependency_lock_sha256"] = "9" * 64
        with self.assertRaises(ValueError):
            self.compose(source, code, lock, g1a, g2a, scheduler, hashes)

    @patch.object(go, "validate_scheduler_freeze_v122", return_value=[])
    @patch.object(go, "validate_g2a_v122_barrier", return_value=[])
    @patch.object(go, "validate_g1a_seal_object", return_value=[])
    def test_scheduler_g1a_binding_mismatch_is_rejected(self, _g1, _g2, _scheduler):
        source, code, lock, g1a, g2a, scheduler, hashes = self.fixtures()
        scheduler["g1a_seal_sha256"] = "9" * 64
        with self.assertRaises(ValueError):
            self.compose(source, code, lock, g1a, g2a, scheduler, hashes)


if __name__ == "__main__":
    unittest.main()
