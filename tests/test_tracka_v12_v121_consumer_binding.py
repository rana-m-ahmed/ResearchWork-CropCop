from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
KAGGLE = ROOT / "journal_extension" / "kaggle"
SCRIPTS = ROOT / "journal_extension" / "scripts"
for path in (SRC, KAGGLE, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cropcop_je import tracka_v12_g1a_v121 as g1a_v121
from cropcop_je import tracka_v12_runtime as runtime
import run_tracka_v12_account as account_base
import run_tracka_v12_account_v121 as account_v121
import seal_tracka_v12_science_go as go_legacy
import seal_tracka_v12_science_go_v123 as go_v123
import seal_tracka_v12_science_go_v124 as go_v124


class TrackAV121ConsumerBindingTests(unittest.TestCase):
    def test_account_parent_preflight_uses_versioned_g1a_validator_and_restores_runtime(self):
        original = runtime.validate_g1a_seal_object
        observed = {}

        def fake_loader(bundle_dir, *, expected_source_sha=None):
            observed["versioned_validator_active"] = runtime.validate_g1a_seal_object is g1a_v121.validate_g1a_seal_object
            observed["source"] = expected_source_sha
            return {"status": "PASS"}, []

        with mock.patch.object(runtime, "load_and_validate_g1a_bundle", side_effect=fake_loader):
            payload, errors = account_v121.load_and_validate_g1a_bundle_v121("dummy", expected_source_sha="a" * 40)
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(errors, [])
        self.assertTrue(observed["versioned_validator_active"])
        self.assertEqual(observed["source"], "a" * 40)
        self.assertIs(runtime.validate_g1a_seal_object, original)

    def test_account_base_global_is_rebound_and_training_runner_is_versioned(self):
        self.assertIs(account_base.load_and_validate_g1a_bundle, account_v121.load_and_validate_g1a_bundle_v121)
        self.assertEqual(account_base.RUNNER.name, "run_tracka_v12_training_v121.py")

    def test_science_go_accepts_only_exact_head_attestation_schema_v11(self):
        payload = {
            "schema_version": "1.1",
            "attestation_kind": "track_a_v12_pre_science_code",
            "github_actions": True,
            "pull_request_head_sha": "a" * 40,
            "source_git_commit": "a" * 40,
            "workflow_run_id": "123",
            "workflow_run_attempt": "1",
        }
        self.assertEqual(
            go_v124._validate_ci_attestation_provenance_v11(
                payload,
                expected_kind="track_a_v12_pre_science_code",
                source_git_commit="a" * 40,
                label="code attestation",
            ),
            [],
        )
        payload["schema_version"] = "1.0"
        errors = go_v124._validate_ci_attestation_provenance_v11(
            payload,
            expected_kind="track_a_v12_pre_science_code",
            source_git_commit="a" * 40,
            label="code attestation",
        )
        self.assertIn("code attestation schema version mismatch", errors)

    def test_science_go_scopes_versioned_rebinding_and_restores_historical_globals(self):
        old_validator = go_legacy.validate_g1a_seal_object
        old_provenance = go_legacy._validate_ci_attestation_provenance
        observed = {}

        def fake_main():
            observed["validator"] = go_legacy.validate_g1a_seal_object is g1a_v121.validate_g1a_seal_object
            observed["provenance"] = go_legacy._validate_ci_attestation_provenance is go_v124._validate_ci_attestation_provenance_v11
            return 0

        with mock.patch.object(go_v123, "main", side_effect=fake_main):
            self.assertEqual(go_v124.main(), 0)
        self.assertTrue(observed["validator"])
        self.assertTrue(observed["provenance"])
        self.assertIs(go_legacy.validate_g1a_seal_object, old_validator)
        self.assertIs(go_legacy._validate_ci_attestation_provenance, old_provenance)


if __name__ == "__main__":
    unittest.main()
