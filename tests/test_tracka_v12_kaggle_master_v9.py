from __future__ import annotations

import os
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

import tracka_v12_kaggle_operator as v1
import tracka_v12_kaggle_operator_v2 as v2
import tracka_v12_kaggle_operator_v3 as v3
import tracka_v12_kaggle_operator_v8 as v8
import master_account_driver_v8 as driver
import master_attestations_v8 as attest
import master_g1a_v8 as g1a
import master_g2a_v8 as g2a
import master_launch_guard_v8 as guard
import master_wait_v8 as wait8

SCIENCE_SHA = "c65a5082809603155fa80a6eb0152fae33dbd9e0"
MANIFEST_SHA256 = "0777ce9f040ada279dfe31b85ce8a0498926a10c78ef3b7aa6297c64ea9d0c43"
CODE_MEMBER_SHA256 = "4f29548f5ba56c2969fb1b0c7234d66ce15a2747d104e8e27579af08f9be7010"
LOCK_MEMBER_SHA256 = "d2a1f4303ce73f1f92f2a3835b6300230ab566686da0f3a2e58f743bf1366c4d"


class TrackAV12KaggleMasterV9Tests(unittest.TestCase):
    def test_exact_science_authority_rebinds_all_inherited_modules(self):
        self.assertEqual(v8.SCIENCE_SHA, SCIENCE_SHA)
        self.assertEqual(v1.SCIENCE_SHA, SCIENCE_SHA)
        self.assertEqual(v2.SCIENCE_SHA, SCIENCE_SHA)
        self.assertEqual(v3.SCIENCE_SHA, SCIENCE_SHA)
        self.assertEqual(v8.OPERATOR_SCHEMA_VERSION, "4.3")
        self.assertEqual(v8.R13_PARITY_CONTRACT_ID_V8, "TRACKA-A1-R13-PRETRAINED-IDENTITY-V1.2.1")
        self.assertEqual(v8.R13_PARITY_REQUIRED_MAX_ABS_V8, 5e-5)

    def test_release_manifest_is_packaged_exact_bytes_and_non_authorizing(self):
        path, payload = attest.verified_release_attestation_manifest()
        self.assertEqual(v8.sha256_file(path), MANIFEST_SHA256)
        self.assertEqual(payload["schema_version"], "1.1")
        self.assertEqual(payload["science_source_sha"], SCIENCE_SHA)
        self.assertEqual(payload["code_attestation"]["member_sha256"], CODE_MEMBER_SHA256)
        self.assertEqual(payload["lock_runtime_attestation"]["member_sha256"], LOCK_MEMBER_SHA256)
        self.assertEqual(payload["remaining_scientific_states"], 11)
        self.assertFalse(payload["science_authorized"])
        self.assertFalse(payload["protected_test_accessed"])
        self.assertFalse(payload["external_surface_accessed"])
        materialization = payload["attestation_materialization"]
        self.assertEqual(materialization["mode"], "packaged_exact_bytes")
        self.assertFalse(materialization["runtime_requires_github_actions_api"])
        self.assertFalse(materialization["runtime_requires_actions_read_permission"])

    def test_packaged_attestation_bytes_are_exact_and_semantically_valid(self):
        code, lock = attest.verified_attestation_paths()
        self.assertEqual(v8.sha256_file(code), CODE_MEMBER_SHA256)
        self.assertEqual(v8.sha256_file(lock), LOCK_MEMBER_SHA256)
        code_payload = v8.load_json(code)
        lock_payload = v8.load_json(lock)
        self.assertEqual(code_payload["source_git_commit"], SCIENCE_SHA)
        self.assertEqual(code_payload["pull_request_head_sha"], SCIENCE_SHA)
        self.assertEqual(lock_payload["source_git_commit"], SCIENCE_SHA)
        self.assertEqual(lock_payload["pull_request_head_sha"], SCIENCE_SHA)
        self.assertEqual((code_payload.get("static_pre_science_gates") or {}).get("kaggle_generation_durability_contract"), "PASS")

    def test_materialization_succeeds_without_github_token_or_network(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {}, clear=True):
            code, lock = attest.materialize_verified_attestations(td)
            self.assertEqual(v8.sha256_file(code), CODE_MEMBER_SHA256)
            self.assertEqual(v8.sha256_file(lock), LOCK_MEMBER_SHA256)
            self.assertTrue(Path(code).is_file())
            self.assertTrue(Path(lock).is_file())

    def test_attestation_runtime_contains_no_actions_download_path(self):
        text = (OPS / "master_attestations_v8.py").read_text(encoding="utf-8")
        for forbidden in (
            "urllib.request",
            "urllib.error",
            "api.github.com",
            "/actions/artifacts/",
            "_download_artifact_zip",
            "CROPCOP_GITHUB_TOKEN",
        ):
            self.assertNotIn(forbidden, text)
        self.assertIn("packaged_exact_bytes", text)
        self.assertIn("attestations_v9", text)

    def test_inherited_g1a_helpers_resolve_versioned_validator(self):
        self.assertIs(v1.validate_g1a_bundle_with_science, v8.validate_g1a_bundle_with_science)
        self.assertIs(v2.validate_g1a_bundle_with_science, v8.validate_g1a_bundle_with_science)
        self.assertIs(v3.validate_g1a_bundle_with_science, v8.validate_g1a_bundle_with_science)

    def test_active_g1a_and_science_routes_are_versioned(self):
        g1a_text = (OPS / "master_g1a_v8.py").read_text(encoding="utf-8")
        science_text = (OPS / "master_science_v8.py").read_text(encoding="utf-8")
        control_text = (OPS / "master_control_v8.py").read_text(encoding="utf-8")
        self.assertIn("seal_tracka_v12_g1a_v121.py", g1a_text)
        self.assertIn("run_tracka_v12_account_v121.py", science_text)
        self.assertIn("seal_tracka_v12_science_go_v124.py", control_text)
        self.assertIn("materialize_verified_attestations", control_text)

    def test_exact_attestation_validation_precedes_stack_and_g1a(self):
        text = (OPS / "master_account_driver_v8.py").read_text(encoding="utf-8")
        source_stage = text.index('stage("SCIENCE_SOURCE_GITHUB_AND_ATTESTATION_PREFLIGHT"')
        materialize = text.index("lambda: materialize_verified_attestations")
        stack_stage = text.index('stage("EXACT_EXECUTION_STACK"')
        g1a_stage = text.index('stage("CANONICAL_G1A"')
        self.assertLess(source_stage, materialize)
        self.assertLess(materialize, stack_stage)
        self.assertLess(stack_stage, g1a_stage)

    def test_active_runtime_has_no_legacy_compatibility_injection(self):
        names = (
            "tracka_v12_kaggle_operator_v8.py",
            "master_attestations_v8.py",
            "master_g1a_v8.py",
            "master_g2a_v8.py",
            "master_control_v8.py",
            "master_science_v8.py",
            "master_account_driver_v8.py",
            "master_launch_guard_v8.py",
        )
        combined = "\n".join((OPS / name).read_text(encoding="utf-8") for name in names)
        for forbidden in (
            "master_g1a_sealer_compat_v6",
            "init_globals",
            "runpy.run_path",
            "DS-V1-TEST-CONSUMED",
            "--test",
        ):
            self.assertNotIn(forbidden, combined)

    def test_expected_account_binding_is_mandatory(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(v8.OperatorError):
                driver.assert_expected_account("K2", "some-user")
        with mock.patch.dict(os.environ, {"CROPCOP_EXPECTED_KAGGLE_USERNAME": "correct-user"}, clear=True):
            with self.assertRaises(v8.OperatorError):
                driver.assert_expected_account("K2", "wrong-user")
            driver.assert_expected_account("K2", "correct-user")

    def test_guard_terminal_reuse_requires_exact_science_runtime_account(self):
        status = {
            "schema_version": guard.STATUS_SCHEMA,
            "state": "FINISHED",
            "account_id": "K1",
            "science_sha": SCIENCE_SHA,
            "operator_runtime_sha": "r" * 40,
            "return_code": 0,
        }
        self.assertEqual(guard._matching_terminal_code(status, account_id="K1", runtime_sha="r" * 40), 0)
        self.assertIsNone(guard._matching_terminal_code(status, account_id="K2", runtime_sha="r" * 40))
        self.assertIsNone(guard._matching_terminal_code(status, account_id="K1", runtime_sha="x" * 40))

    def test_wait_budget_uses_global_session_deadline(self):
        with mock.patch.dict(os.environ, {
            "CROPCOP_NOTEBOOK_STARTED_MONOTONIC": "1000",
            "CROPCOP_NOTEBOOK_HARD_LIMIT_SECONDS": "43200",
            "CROPCOP_NOTEBOOK_FINALIZATION_MARGIN_SECONDS": "3600",
        }, clear=False), mock.patch.object(wait8.time, "monotonic", return_value=2000):
            self.assertEqual(wait8.dependency_deadline_monotonic(), 40600.0)
            self.assertEqual(wait8.remaining_dependency_seconds(), 38600.0)
            self.assertFalse(wait8.dependency_wait_expired())

    def test_failed_handoffs_remain_non_authorizing(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(g1a, "operator_runtime_head", return_value="r" * 40):
            payload = g1a.write_handoff(Path(td) / "g1a.json", status="FAILED", locator="owner/dataset", failure_code=g1a.G1A_FAILURE_CODE)
            self.assertEqual(payload["science_source_sha"], SCIENCE_SHA)
            self.assertFalse(payload["science_authorized"])
        with tempfile.TemporaryDirectory() as td, mock.patch.object(g2a, "operator_runtime_head", return_value="r" * 40):
            payload = g2a.write_account_status(Path(td) / "g2a.json", account_id="K2", status="FAILED", g1a_seal_sha256="a" * 64, failure_code=g2a.G2A_FAILURE_CODE)
            self.assertEqual(payload["science_source_sha"], SCIENCE_SHA)
            self.assertFalse(payload["science_authorized"])

    def test_g1a_exact_generation_wait_retries_placeholder_and_stale_generation(self):
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
                Path(td), "owner/canonical-g1a", Path(td) / "download", env={}, expected_seal_sha256=expected,
            )
        self.assertEqual(bundle, Path("/tmp/exact"))
        self.assertEqual(seal["g1a_seal_sha256"], expected)
        self.assertEqual(download.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_g1a_exact_generation_wait_fails_closed_on_session_budget(self):
        expected = "c" * 64
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(g1a, "download_and_validate_g1a_dataset", side_effect=v8.OperatorError("G1A bundle resolution must be unique, found 0: []")), \
             mock.patch.object(g1a, "dependency_wait_expired", return_value=True), \
             mock.patch.object(g1a.time, "sleep") as sleep:
            with self.assertRaises(TimeoutError):
                g1a.wait_for_exact_g1a_generation(
                    Path(td), "owner/canonical-g1a", Path(td) / "download", env={}, expected_seal_sha256=expected,
                )
        sleep.assert_not_called()

    def test_k1_g1a_path_binds_roundtrip_to_newly_built_seal(self):
        text = (OPS / "master_g1a_v8.py").read_text(encoding="utf-8")
        build = text.index("seal = build_g1a_once")
        version = text.index("version_private_dataset(locator, bundle")
        exact_wait = text.index("roundtrip_bundle, roundtrip_seal = wait_for_exact_g1a_generation")
        expected_binding = text.index('expected_seal_sha256=seal["g1a_seal_sha256"]')
        self.assertLess(build, version)
        self.assertLess(version, exact_wait)
        self.assertLess(exact_wait, expected_binding)


if __name__ == "__main__":
    unittest.main()
