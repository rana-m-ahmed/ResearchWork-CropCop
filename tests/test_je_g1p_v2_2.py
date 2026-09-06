from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.g1 import validate_v1_test_access_identity
from cropcop_je.g1_package import G1Package
from cropcop_je.g1_publication import (
    G1PublicationError,
    G1RemoteMismatch,
    _require_transport_inventory,
    _version_ref,
    ensure_private_target,
    publish_and_roundtrip,
    wait_for_version_advance,
)
from cropcop_je.tensor_identity import (
    TENSOR_IDENTITY_ALGORITHM,
    tensor_identity_sha256,
)


class FakeArray:
    def __init__(self, data: bytes):
        self.data = data

    def tobytes(self, order="C"):
        if order != "C":
            raise AssertionError(order)
        return self.data


class FakeTensor:
    def __init__(self, data: bytes, *, dtype="torch.float32", shape=(1,)):
        self._data = data
        self.dtype = dtype
        self.shape = shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return FakeArray(self._data)


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        self.value += 1.0
        return self.value


class VersionSequenceApi:
    def __init__(self, states):
        self.states = list(states)
        self.calls = 0

    def dataset_status(self, slug, format="json"):
        self.calls += 1
        state = self.states[min(self.calls - 1, len(self.states) - 1)]
        return json.dumps(state)


class TargetApi:
    def __init__(
        self,
        root: Path,
        states,
        *,
        private=True,
        in_mine=True,
        metadata_failures=0,
    ):
        self.root = root
        self.states = list(states)
        self.private = private
        self.in_mine = in_mine
        self.metadata_failures = metadata_failures
        self.status_calls = 0
        self.metadata_calls = 0

    def dataset_metadata(self, slug, path):
        self.metadata_calls += 1
        if self.metadata_calls <= self.metadata_failures:
            raise RuntimeError("transient 403")
        target = Path(path) / "dataset-metadata.json"
        target.write_text(json.dumps({"id": slug, "isPrivate": self.private}))
        return str(target)

    def dataset_list_with_response(self, **kwargs):
        rows = (
            [SimpleNamespace(ref="owner/private-dataset")]
            if self.in_mine and self.metadata_calls > self.metadata_failures
            else []
        )
        return SimpleNamespace(datasets=rows)

    def dataset_status(self, slug, format="json"):
        self.status_calls += 1
        state = self.states[min(self.status_calls - 1, len(self.states) - 1)]
        if isinstance(state, Exception):
            raise state
        return json.dumps(state)


def fake_package() -> G1Package:
    return G1Package(
        Path("/tmp/G1_PACKAGE.tar"),
        Path("/tmp/G1_PACKAGE_MANIFEST.json"),
        {
            "package_sha256": "1" * 64,
            "package_bytes": 123,
            "manifest_sha256": "2" * 64,
        },
    )


def fake_seal() -> dict:
    return {
        "source_git_sha": "a" * 40,
        "dependency_lock_sha256": "b" * 64,
        "g1_seal_sha256": "c" * 64,
    }


def fake_preflight() -> dict:
    return {
        "current_version_number": 1,
        "owner_match": True,
        "authoritative_is_private": True,
    }


def fake_roundtrip(version: int) -> dict:
    return {
        "published_version_number": version,
        "published_version_ref": f"owner/dataset/{version}",
        "remote_required_file_inventory": [
            {"name": "G1_PACKAGE.tar", "bytes": 123},
            {"name": "G1_PACKAGE_MANIFEST.json", "bytes": 456},
        ],
        "roundtrip_package_sha256": "1" * 64,
        "roundtrip_member_count": 3,
        "roundtrip_verified": True,
    }


class TensorIdentityTests(unittest.TestCase):
    def test_01_algorithm_is_single_versioned_domain(self):
        self.assertEqual(
            TENSOR_IDENTITY_ALGORITHM,
            "cropcop-tensor-identity-v1",
        )

    def test_02_same_state_same_hash(self):
        state = {"b": FakeTensor(b"bbb"), "a": FakeTensor(b"aaa")}
        self.assertEqual(
            tensor_identity_sha256(state),
            tensor_identity_sha256(dict(reversed(list(state.items())))),
        )

    def test_03_tensor_mutation_changes_hash(self):
        self.assertNotEqual(
            tensor_identity_sha256({"x": FakeTensor(b"abc")}),
            tensor_identity_sha256({"x": FakeTensor(b"abd")}),
        )

    def test_04_key_dtype_shape_are_bound(self):
        base = tensor_identity_sha256(
            {"x": FakeTensor(b"abc", dtype="d1", shape=(3,))}
        )
        variants = [
            tensor_identity_sha256(
                {"y": FakeTensor(b"abc", dtype="d1", shape=(3,))}
            ),
            tensor_identity_sha256(
                {"x": FakeTensor(b"abc", dtype="d2", shape=(3,))}
            ),
            tensor_identity_sha256(
                {"x": FakeTensor(b"abc", dtype="d1", shape=(1, 3))}
            ),
        ]
        self.assertTrue(all(value != base for value in variants))

    def test_05_old_delimiter_implementations_are_removed(self):
        prep = (
            ROOT / "journal_extension/scripts/prepare_mnv4_pretrained.py"
        ).read_text()
        capture = (
            ROOT / "journal_extension/scripts/capture_mnv4_pretrained_provenance.py"
        ).read_text()
        for source in (prep, capture):
            self.assertIn("tensor_identity_sha256", source)
            self.assertIn("TENSOR_IDENTITY_ALGORITHM", source)
            self.assertNotIn("def _tensor_identity_sha256", source)
        self.assertNotIn("def tensor_identity_sha256(state", prep)


class V1SchemaTests(unittest.TestCase):
    def test_06_canonical_false_passes(self):
        errors, mode = validate_v1_test_access_identity(
            {"v1_test_accessed": False}
        )
        self.assertEqual(errors, [])
        self.assertEqual(mode, "canonical")

    def test_07_legacy_false_passes_only_as_reported_alias(self):
        errors, mode = validate_v1_test_access_identity(
            {"protected_test_accessed_during_g1": False}
        )
        self.assertEqual(errors, [])
        self.assertEqual(mode, "legacy_alias_false")

    def test_08_true_missing_and_contradiction_fail(self):
        for case in (
            {"v1_test_accessed": True},
            {},
            {
                "v1_test_accessed": False,
                "protected_test_accessed_during_g1": True,
            },
        ):
            errors, _ = validate_v1_test_access_identity(case)
            self.assertTrue(errors, case)

    def test_09_new_sealer_emits_only_canonical_field(self):
        source = (
            ROOT / "journal_extension/scripts/seal_g1.py"
        ).read_text()
        self.assertIn('"v1_test_accessed": False', source)
        self.assertNotIn(
            '"protected_test_accessed_during_g1": False',
            source,
        )


class VersionBarrierTests(unittest.TestCase):
    def test_10_ready_old_version_does_not_satisfy_transition(self):
        api = VersionSequenceApi(
            [
                {"status": "ready", "current_version_number": 1},
                {"status": "ready", "current_version_number": 1},
                {"status": "ready", "current_version_number": 2},
            ]
        )
        result = wait_for_version_advance(
            api,
            "owner/dataset",
            1,
            timeout_seconds=20,
            poll_interval_seconds=0,
            clock=FakeClock(),
            sleep_fn=lambda _: None,
        )
        self.assertEqual(result["current_version_number"], 2)
        self.assertEqual(api.calls, 3)

    def test_11_failed_processing_fails(self):
        api = VersionSequenceApi(
            [{"status": "failed", "current_version_number": 1}]
        )
        with self.assertRaises(G1PublicationError):
            wait_for_version_advance(
                api,
                "owner/dataset",
                1,
                timeout_seconds=10,
                poll_interval_seconds=0,
                clock=FakeClock(),
                sleep_fn=lambda _: None,
            )

    def test_12_exact_version_ref_is_explicit(self):
        self.assertEqual(
            _version_ref("owner/dataset", 2),
            "owner/dataset/2",
        )

    def test_13_stale_placeholder_inventory_is_rejected(self):
        with self.assertRaises(G1RemoteMismatch):
            _require_transport_inventory(
                [{"name": "README.txt", "bytes": 10}],
                "owner/dataset/1",
            )

    def test_14_ambiguous_nonzero_upload_adopts_advanced_version(self):
        calls = []
        with tempfile.TemporaryDirectory() as td,              patch(
                 "cropcop_je.g1_publication.validate_local_g1_bundle",
                 return_value=fake_seal(),
             ),              patch(
                 "cropcop_je.g1_publication.preflight_private_target",
                 return_value=fake_preflight(),
             ),              patch(
                 "cropcop_je.g1_publication._prepare_transport",
                 return_value=fake_package(),
             ),              patch(
                 "cropcop_je.g1_publication.wait_for_version_advance",
                 return_value={
                     "status": "ready",
                     "current_version_number": 2,
                 },
             ),              patch(
                 "cropcop_je.g1_publication._roundtrip_exact_version",
                 return_value=fake_roundtrip(2),
             ):
            receipt = publish_and_roundtrip(
                "/unused",
                "owner/dataset",
                Path(td) / "receipt.json",
                env={
                    "KAGGLE_USERNAME": "owner",
                    "KAGGLE_KEY": "fixture",
                },
                api_factory=lambda: object(),
                run=lambda *a, **k: (
                    calls.append(a[0])
                    or SimpleNamespace(
                        returncode=1,
                        stdout="",
                        stderr="timeout",
                    )
                ),
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(receipt["published_version_number"], 2)
        self.assertEqual(
            receipt["mutation_outcome"],
            "COMMAND_NONZERO_AMBIGUOUS",
        )

    def test_15_real_upload_failure_without_advance_has_no_receipt(self):
        with tempfile.TemporaryDirectory() as td,              patch(
                 "cropcop_je.g1_publication.validate_local_g1_bundle",
                 return_value=fake_seal(),
             ),              patch(
                 "cropcop_je.g1_publication.preflight_private_target",
                 return_value=fake_preflight(),
             ),              patch(
                 "cropcop_je.g1_publication._prepare_transport",
                 return_value=fake_package(),
             ),              patch(
                 "cropcop_je.g1_publication.wait_for_version_advance",
                 side_effect=G1PublicationError("no advance"),
             ),              patch(
                 "cropcop_je.g1_publication._status_snapshot",
                 return_value={
                     "status": "ready",
                     "current_version_number": 1,
                 },
             ):
            receipt_path = Path(td) / "receipt.json"
            with self.assertRaises(G1PublicationError):
                publish_and_roundtrip(
                    "/unused",
                    "owner/dataset",
                    receipt_path,
                    env={
                        "KAGGLE_USERNAME": "owner",
                        "KAGGLE_KEY": "fixture",
                    },
                    api_factory=lambda: object(),
                    run=lambda *a, **k: SimpleNamespace(
                        returncode=1,
                        stdout="",
                        stderr="failed",
                    ),
                )
            self.assertFalse(receipt_path.exists())
            attempt = json.loads(
                (
                    Path(td) / "G1_PUBLICATION_ATTEMPT.json"
                ).read_text()
            )
            self.assertEqual(attempt["status"], "FAILED")

    def test_16_repair_reuses_exact_current_version_no_upload(self):
        calls = []
        with tempfile.TemporaryDirectory() as td,              patch(
                 "cropcop_je.g1_publication.validate_local_g1_bundle",
                 return_value=fake_seal(),
             ),              patch(
                 "cropcop_je.g1_publication.preflight_private_target",
                 return_value=fake_preflight(),
             ),              patch(
                 "cropcop_je.g1_publication._prepare_transport",
                 return_value=fake_package(),
             ),              patch(
                 "cropcop_je.g1_publication._roundtrip_exact_version",
                 return_value=fake_roundtrip(1),
             ),              patch(
                 "cropcop_je.g1_publication._status_snapshot",
                 return_value={
                     "status": "ready",
                     "current_version_number": 1,
                 },
             ):
            receipt = publish_and_roundtrip(
                "/unused",
                "owner/dataset",
                Path(td) / "receipt.json",
                mode="repair",
                env={
                    "KAGGLE_USERNAME": "owner",
                    "KAGGLE_KEY": "fixture",
                },
                api_factory=lambda: object(),
                run=lambda *a, **k: calls.append(a[0]),
            )
        self.assertEqual(calls, [])
        self.assertTrue(receipt["reused_existing_exact_version"])
        self.assertEqual(
            receipt["published_version_ref"],
            "owner/dataset/1",
        )

    def test_17_repair_stale_placeholder_uploads_once(self):
        calls = []
        seen = []

        def roundtrip(api, slug, version, package, seal, root):
            seen.append(version)
            if version == 1:
                raise G1RemoteMismatch("README-only placeholder")
            return fake_roundtrip(version)

        with tempfile.TemporaryDirectory() as td,              patch(
                 "cropcop_je.g1_publication.validate_local_g1_bundle",
                 return_value=fake_seal(),
             ),              patch(
                 "cropcop_je.g1_publication.preflight_private_target",
                 return_value=fake_preflight(),
             ),              patch(
                 "cropcop_je.g1_publication._prepare_transport",
                 return_value=fake_package(),
             ),              patch(
                 "cropcop_je.g1_publication.wait_for_version_advance",
                 return_value={
                     "status": "ready",
                     "current_version_number": 2,
                 },
             ),              patch(
                 "cropcop_je.g1_publication._roundtrip_exact_version",
                 side_effect=roundtrip,
             ):
            receipt = publish_and_roundtrip(
                "/unused",
                "owner/dataset",
                Path(td) / "receipt.json",
                mode="repair",
                env={
                    "KAGGLE_USERNAME": "owner",
                    "KAGGLE_KEY": "fixture",
                },
                api_factory=lambda: object(),
                run=lambda *a, **k: (
                    calls.append(a[0])
                    or SimpleNamespace(
                        returncode=0,
                        stdout="",
                        stderr="",
                    )
                ),
            )
        self.assertEqual(seen, [1, 2])
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            receipt["published_version_ref"],
            "owner/dataset/2",
        )


class TargetSettlementTests(unittest.TestCase):
    def test_18_transient_visibility_after_create_settles(self):
        with tempfile.TemporaryDirectory() as td:
            api = TargetApi(
                Path(td),
                [RuntimeError("403")],
                metadata_failures=1,
            )
            calls = []

            def run(*args, **kwargs):
                calls.append(args[0])
                return SimpleNamespace(
                    returncode=0,
                    stdout="",
                    stderr="",
                )

            with patch(
                "cropcop_je.g1_publication.wait_until_target_settled",
                return_value={
                    "metadata_visible": True,
                    "mine_membership": True,
                    "dataset_status": "ready",
                    "current_version_number": 1,
                },
            ):
                result = ensure_private_target(
                    "owner/private-dataset",
                    env={
                        "KAGGLE_USERNAME": "owner",
                        "KAGGLE_KEY": "fixture",
                        "CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET": "1",
                    },
                    api_factory=lambda: api,
                    run=run,
                )
            self.assertTrue(result["created_this_run"])
            self.assertEqual(len(calls), 1)

    def test_19_nonzero_create_can_be_adopted_after_settle(self):
        with tempfile.TemporaryDirectory() as td:
            api = TargetApi(
                Path(td),
                [RuntimeError("403")],
                metadata_failures=1,
            )
            with patch(
                "cropcop_je.g1_publication.wait_until_target_settled",
                return_value={
                    "metadata_visible": True,
                    "mine_membership": True,
                    "dataset_status": "ready",
                    "current_version_number": 1,
                },
            ):
                result = ensure_private_target(
                    "owner/private-dataset",
                    env={
                        "KAGGLE_USERNAME": "owner",
                        "KAGGLE_KEY": "fixture",
                        "CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET": "1",
                    },
                    api_factory=lambda: api,
                    run=lambda *a, **k: SimpleNamespace(
                        returncode=1,
                        stdout="ambiguous",
                        stderr="",
                    ),
                )
            self.assertTrue(
                result["creation_command_nonzero_but_target_settled"]
            )

    def test_20_owner_mismatch_fails_before_mutation(self):
        with self.assertRaises(G1PublicationError):
            ensure_private_target(
                "other/private-dataset",
                env={
                    "KAGGLE_USERNAME": "owner",
                    "KAGGLE_KEY": "fixture",
                    "CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET": "1",
                },
                api_factory=lambda: (_ for _ in ()).throw(
                    AssertionError("API called")
                ),
            )


class SourceArchitectureTests(unittest.TestCase):
    def test_21_publication_is_version_bound_not_bare_download(self):
        source = (
            ROOT
            / "journal_extension/src/cropcop_je/g1_publication.py"
        ).read_text()
        self.assertIn("wait_for_version_advance", source)
        self.assertIn("current > pre", source)
        self.assertIn("dataset_list_files", source)
        self.assertIn("dataset_download_file", source)
        self.assertNotIn("api.dataset_download_files(slug", source)

    def test_22_attempt_is_written_before_version_mutation(self):
        source = (
            ROOT
            / "journal_extension/src/cropcop_je/g1_publication.py"
        ).read_text()
        self.assertLess(
            source.index("atomic_write_json(attempt_path, attempt)"),
            source.index("command = _version_command"),
        )

    def test_23_run_g1_explicit_repair_and_no_hidden_boolean(self):
        source = (
            ROOT / "journal_extension/kaggle/run_g1.py"
        ).read_text()
        self.assertIn('"g1-publication-repair"', source)
        self.assertIn("CROPCOP_G1_REPAIR_INPUT_ROOT", source)
        self.assertNotIn("CROPCOP_G1_PUBLICATION_REPAIR", source)
        start = source.index("def _publication_only_repair")
        end = source.index("\ndef main()", start)
        repair = source[start:end]
        self.assertNotIn("seal_g1.py", repair)
        self.assertIn(
            '"pair_initializations_regenerated": False',
            repair,
        )
        self.assertIn('"seal_rewritten": False', repair)

    def test_24_stage_markers_cover_terminal_path(self):
        source = (
            ROOT / "journal_extension/kaggle/run_g1.py"
        ).read_text()
        for stage_name in (
            "SOURCE_PREFLIGHT",
            "INPUT_RESOLUTION",
            "TARGET_PREFLIGHT",
            "MNV4_PROVENANCE",
            "TEACHER_IDENTITY",
            "PAIR_SEAL",
            "G1_BARRIER",
            "PUBLICATION_PREPARED",
            "PUBLICATION_VERSION_ADVANCED",
            "ROUNDTRIP_VERIFY",
            "TERMINAL_PUBLICATION",
        ):
            self.assertIn(f'stage("{stage_name}")', source)

    def test_25_terminal_branch_is_bound_before_publication(self):
        source = (
            ROOT / "journal_extension/kaggle/run_g1.py"
        ).read_text()
        branch = source.index(
            'terminal["public_evidence_branch"] = "run-evidence/G1"'
        )
        publish = source.index(
            "branch = _publish_terminal_evidence",
            branch,
        )
        self.assertLess(branch, publish)
        publication = (
            ROOT / "journal_extension/src/cropcop_je/publication.py"
        ).read_text()
        self.assertIn(
            'branch = f"run-evidence/{run_id}"',
            publication,
        )
        self.assertIn("merge-base", publication)
        self.assertIn("--is-ancestor", publication)


if __name__ == "__main__":
    unittest.main()
