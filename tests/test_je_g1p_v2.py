from __future__ import annotations

import io
import json
import sys
import tarfile
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.g1 import g1_seal_hash, validate_g1_seal_object
from cropcop_je.g1_package import (
    G1Package,
    G1PackageError,
    create_g1_package,
    mount_g1_input,
    readiness_transport_dry_run,
    safe_extract_g1_package,
)
from cropcop_je.g1_publication import G1PublicationError, preflight_private_target, wait_until_ready
from cropcop_je.hashing import sha256_file


class G1PDependencyAndReadinessTests(unittest.TestCase):
    def test_01_transformers_exact_lock_is_synchronized(self):
        lock = json.loads(
            (ROOT / "journal_extension/locks/execution_dependency_lock.json").read_text()
        )
        self.assertEqual(lock["packages"]["transformers"], "5.0.0")
        for rel in (
            "journal_extension/requirements-training.txt",
            "journal_extension/requirements-training.lock.txt",
        ):
            lines = {
                x.strip()
                for x in (ROOT / rel).read_text().splitlines()
                if x.strip() and not x.lstrip().startswith("#")
            }
            self.assertIn("transformers==5.0.0", lines)

    def test_02_readiness_entrypoint_is_explicitly_non_qualifying(self):
        source = (ROOT / "journal_extension/scripts/validate_g1_inputs.py").read_text()
        self.assertNotIn('execute("seal_g1.py"', source)
        self.assertNotIn("prepare_pair_init.py", source)
        self.assertNotIn("run_training.py", source)
        self.assertIn('"qualifying_phase": False', source)
        self.assertIn('"g1_seal_created": False', source)
        self.assertIn('"pair_initializations_created": False', source)
        self.assertIn('"optimizer_steps_performed": 0', source)
        self.assertIn('"v1_test_accessed": False', source)
        self.assertIn('"scientific_metric_computed": False', source)

    def test_03_readiness_runs_real_teacher_and_mnv4_checks(self):
        source = (ROOT / "journal_extension/scripts/validate_g1_inputs.py").read_text()
        for token in (
            "capture_teacher_canonical_state.py",
            "verify_teacher_adapter_parity.py",
            "verify_teacher_class_order.py",
            "prepare_mnv4_pretrained.py",
            "capture_mnv4_pretrained_provenance.py",
            "readiness_private_target_probe",
            "readiness_transport_dry_run",
        ):
            self.assertIn(token, source)


class G1PTransportTests(unittest.TestCase):
    @staticmethod
    def _fixture_bundle(root: Path) -> Path:
        bundle = root / "bundle"
        (bundle / "private").mkdir(parents=True)
        (bundle / "evidence").mkdir()
        (bundle / "G1_MODEL_IDENTITY_SEAL.json").write_text(
            json.dumps({
                "g1_seal_sha256": "a" * 64,
                "source_git_sha": "b" * 40,
                "dependency_lock_sha256": "c" * 64,
            }, sort_keys=True) + "\n"
        )
        (bundle / "private" / "PAIR_INIT_S1.pt").write_bytes(b"pair-s1")
        (bundle / "evidence" / "fixture.json").write_text('{"status":"PASS"}\n')
        return bundle

    def test_04_production_tar_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle = self._fixture_bundle(root)
            one = create_g1_package(bundle, root / "one")
            two = create_g1_package(bundle, root / "two")
            self.assertEqual(one.manifest["package_sha256"], two.manifest["package_sha256"])
            self.assertEqual(one.package_path.read_bytes(), two.package_path.read_bytes())

    def test_05_package_mount_roundtrip_verifies_members(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle = self._fixture_bundle(root)
            package = create_g1_package(bundle, root / "transport")
            mounted, observed = mount_g1_input(root / "transport", root / "mounted")
            self.assertEqual(observed.manifest["package_sha256"], package.manifest["package_sha256"])
            self.assertEqual(
                (mounted / "private/PAIR_INIT_S1.pt").read_bytes(),
                b"pair-s1",
            )

    def test_06_tar_traversal_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tar_path = root / "G1_PACKAGE.tar"
            with tarfile.open(tar_path, "w") as archive:
                data = b"bad"
                info = tarfile.TarInfo("../escape")
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            manifest = {
                "schema_version": "1.0",
                "package_basename": "G1_PACKAGE.tar",
                "package_sha256": sha256_file(tar_path),
                "package_bytes": tar_path.stat().st_size,
                "members": [{
                    "path": "safe.txt",
                    "type": "file",
                    "sha256": "0" * 64,
                    "bytes": 3,
                }],
                "manifest_sha256": "0" * 64,
            }
            package = G1Package(tar_path, root / "unused.json", manifest)
            with self.assertRaises(G1PackageError):
                safe_extract_g1_package(package, root / "extract")

    def test_07_readiness_transport_uses_same_deterministic_tar_rules(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            staging = root / "staging"
            (staging / "private").mkdir(parents=True)
            (staging / "evidence").mkdir()
            (staging / "private/model.bin").write_bytes(b"real-input-fixture")
            (staging / "evidence/check.json").write_text('{"ok":true}\n')
            one = readiness_transport_dry_run(staging, root / "one")
            two = readiness_transport_dry_run(staging, root / "two")
            self.assertEqual(one["package_sha256"], two["package_sha256"])
            self.assertTrue(one["safe_extract_verified"])
            self.assertFalse(one["production_g1_seal_created"])
            self.assertFalse(one["pair_initializations_created"])


class G1PDownstreamAndRepairTests(unittest.TestCase):
    def test_08_parent_mounts_single_g1_input_and_validates_before_gpu_inventory(self):
        source = (ROOT / "journal_extension/kaggle/run_envelope.py").read_text()
        self.assertIn('req("CROPCOP_G1_INPUT_ROOT")', source)
        self.assertIn("mount_g1_input", source)
        self.assertIn("validate_g1_barrier(inputs)", source)
        call = source.index("_mount_and_validate_g1(source_sha, dependency)")
        gpu = source.index("inventory = gpu_inventory()")
        self.assertLess(call, gpu)

    def test_09_downstream_uses_sealed_teacher_and_mnv4_not_rfdv(self):
        envelope = (ROOT / "journal_extension/kaggle/run_envelope.py").read_text()
        lane = (ROOT / "journal_extension/kaggle/run_lane.py").read_text()
        self.assertNotIn("CROPCOP_RFDV_ROOT", envelope)
        self.assertNotIn("CROPCOP_RFDV_ROOT", lane)
        self.assertNotIn('env_path("CROPCOP_MNV4_PRETRAINED")', lane)
        self.assertNotIn('env_path("CROPCOP_TEACHER_CHECKPOINT")', lane)
        self.assertIn('"pretrained": private / pretrained_name', lane)
        self.assertIn('"teacher": private / teacher_name', lane)

    def test_10_publication_repair_does_not_regenerate_g1(self):
        source = (ROOT / "journal_extension/kaggle/run_g1.py").read_text()
        start = source.index("def _publication_only_repair")
        end = source.index("\ndef main()", start)
        repair = source[start:end]
        self.assertNotIn("seal_g1.py", repair)
        self.assertNotIn("prepare_pair_init.py", repair)
        self.assertIn('"pair_initializations_regenerated": False', repair)
        self.assertIn('"seal_rewritten": False', repair)

    def test_11_g1_pair_initialization_is_cpu_defined(self):
        models = (ROOT / "journal_extension/src/cropcop_je/models.py").read_text()
        g1 = (ROOT / "journal_extension/kaggle/run_g1.py").read_text()
        self.assertIn("torch.random.fork_rng(devices=[])", models)
        self.assertIn("torch.manual_seed(int(seed))", models)
        self.assertIn('"accelerator_required": False', g1)



class G1PSealAndTeacherTests(unittest.TestCase):
    def _valid_seal(self):
        pairs = {}
        specs = {
            "S1": ("MNV4-PAIR-S1", 21270083, ["R04-MNV4-DIRECT-S1", "R05-MNV4-TEACHER-S1"]),
            "S2": ("MNV4-PAIR-S2", 606135704, ["R04-MNV4-DIRECT-S2", "R05-MNV4-TEACHER-S2"]),
            "S3": ("MNV4-PAIR-S3", 1153870846, ["R04-MNV4-DIRECT-S3", "R05-MNV4-TEACHER-S3"]),
        }
        for i, (key, (pair_id, seed, consumers)) in enumerate(specs.items(), 1):
            pairs[key] = {
                "pair_id": pair_id,
                "seed": seed,
                "authorized_consumers": consumers,
                "pretrained_sha256": "1" * 64,
                "sha256": str(i) * 64,
                "bytes": 100 + i,
            }
        seal = {
            "schema_version": "2.0",
            "authority": {
                "id": "EAAI-JE-SDL-v2.1-QA",
                "sha256": "aab17b65b0873dcb1ecedb061eb02ff60ccb09f8b830184f5e2231a600278f74",
            },
            "source_git_sha": "a" * 40,
            "dataset": {
                "manifest_sha256": "bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2",
                "class_map_sha256": "46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2",
                "identity_evidence_sha256": "2" * 64,
            },
            "student": {
                "model_name": "mobilenetv4_conv_medium.e500_r256_in1k",
                "timm_version": "1.0.26",
                "pretrained": {
                    "sha256": "1" * 64,
                    "bytes": 123,
                    "source_kind": "timm_pretrained_cfg_hf_hub",
                    "source_locator": "fixture/model",
                    "provenance_sha256": "3" * 64,
                    "tensor_identity_sha256": "4" * 64,
                    "candidate_serialization_format": "safetensors_state_dict",
                },
            },
            "pair_initializations": pairs,
            "teacher": {
                "checkpoint_sha256": "74b4701b8931976c9227845ead50788ae47e3596f575f2817b7352a715f53b79",
                "checkpoint_bytes": 456,
                "class_map_sha256": "46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2",
                "canonical_state": "EMA",
                "byte_evidence_sha256": "5" * 64,
                "factory_bundle_sha256": "6" * 64,
                "factory_manifest_sha256": "7" * 64,
                "class_order_evidence_sha256": "8" * 64,
                "canonical_state_evidence_sha256": "9" * 64,
                "adapter_parity_evidence_sha256": "c" * 64,
                "factory_entrypoint": "historical_dino_tiny:build_teacher",
            },
            "dependency_lock_sha256": "d" * 64,
            "infra_smoke_evidence_sha256": "e" * 64,
            "dual_gpu_smoke_evidence_sha256": "f" * 64,
        }
        seal["g1_seal_sha256"] = g1_seal_hash(seal)
        return seal

    def test_12_v2_seal_requires_dual_smoke_digest(self):
        seal = self._valid_seal()
        seal.pop("dual_gpu_smoke_evidence_sha256")
        seal["g1_seal_sha256"] = g1_seal_hash(seal)
        errors = validate_g1_seal_object(seal)
        self.assertTrue(any("dual-GPU-smoke" in e for e in errors), errors)

    def test_13_teacher_factory_is_offline_and_has_no_pretrained_fallback(self):
        source = (ROOT / "journal_extension/teacher_factory/historical_dino_tiny.py").read_text()
        self.assertIn("DINOv3ConvNextConfig()", source)
        self.assertIn("DINOv3ConvNextModel(config)", source)
        self.assertNotIn("from_pretrained(", source)
        self.assertIn("load_state_dict(raw_state, strict=True)", source)
        self.assertIn("ema_exact_complete_coverage", source)


class G1PPrivateTargetTests(unittest.TestCase):
    class FakeApi:
        def __init__(self, root: Path, *, private=True, status="ready", fail_metadata=False):
            self.root = root
            self.private = private
            self.status = status
            self.fail_metadata = fail_metadata

        def dataset_metadata(self, slug, path):
            if self.fail_metadata:
                raise RuntimeError("missing")
            target = Path(path) / "dataset-metadata.json"
            target.write_text(json.dumps({"id": slug, "isPrivate": self.private}))
            return str(target)

        def dataset_list_with_response(self, **kwargs):
            return SimpleNamespace(datasets=[SimpleNamespace(ref="owner/private-dataset")])

        def dataset_status(self, slug, format="json"):
            return json.dumps({"status": self.status, "current_version_number": 1})

    def test_14_private_target_owner_mismatch_fails_before_api(self):
        with self.assertRaises(G1PublicationError):
            preflight_private_target(
                "other/private-dataset",
                env={"KAGGLE_USERNAME": "owner", "KAGGLE_KEY": "fixture"},
                api_factory=lambda: (_ for _ in ()).throw(AssertionError("API should not be called")),
            )

    def test_15_public_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            api = self.FakeApi(Path(td), private=False)
            with self.assertRaises(G1PublicationError):
                preflight_private_target(
                    "owner/private-dataset",
                    env={"KAGGLE_USERNAME": "owner", "KAGGLE_KEY": "fixture"},
                    api_factory=lambda: api,
                )

    def test_16_missing_target_fails_without_explicit_create(self):
        with tempfile.TemporaryDirectory() as td:
            api = self.FakeApi(Path(td), fail_metadata=True)
            with self.assertRaises(G1PublicationError):
                preflight_private_target(
                    "owner/private-dataset",
                    env={"KAGGLE_USERNAME": "owner", "KAGGLE_KEY": "fixture"},
                    api_factory=lambda: api,
                )

    def test_17_failed_dataset_processing_blocks_terminal_ready(self):
        with tempfile.TemporaryDirectory() as td:
            api = self.FakeApi(Path(td), status="failed")
            with self.assertRaises(G1PublicationError):
                wait_until_ready(api, "owner/private-dataset", timeout_seconds=0.1)

    def test_18_private_create_code_defaults_private_and_never_public(self):
        source = (ROOT / "journal_extension/src/cropcop_je/g1_publication.py").read_text()
        start = source.index("def create_private_target_if_missing")
        end = source.index("\ndef wait_until_ready", start)
        create = source[start:end]
        self.assertIn('"isPrivate": True', create)
        self.assertNotIn("--public", create)


class G1PMoreTransportAndOrderingTests(unittest.TestCase):
    def test_19_package_corruption_is_detected_before_mount(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle = G1PTransportTests._fixture_bundle(root)
            package = create_g1_package(bundle, root / "transport")
            data = bytearray(package.package_path.read_bytes())
            data[len(data) // 2] ^= 0x01
            package.package_path.write_bytes(bytes(data))
            with self.assertRaises(G1PackageError):
                mount_g1_input(root / "transport", root / "mounted")

    def test_20_child_revalidates_g1_before_any_calibration_or_principal(self):
        source = (ROOT / "journal_extension/kaggle/run_lane.py").read_text()
        validate = source.index("g1_seal = validate_g1(")
        calibration = source.index("if args.phase == \"calibration\":")
        principal = source.index("item = select_principal(lane)")
        self.assertLess(validate, calibration)
        self.assertLess(validate, principal)



class G1PLineageHashHotfixTests(unittest.TestCase):
    def test_21_teacher_lineage_manifest_hashes_every_historical_source(self):
        root = (
            ROOT
            / "journal_extension"
            / "evidence"
            / "historical"
            / "teacher_stage1"
        ).resolve()
        lineage = json.loads(
            (root / "teacher_lineage_manifest.json").read_text(encoding="utf-8")
        )
        sources = lineage.get("historical_evidence_sources", [])
        self.assertTrue(sources, "historical lineage must enumerate evidence sources")
        for row in sources:
            rel = str(row.get("path", ""))
            path = (root / rel).resolve()
            self.assertIn(root, path.parents, f"historical evidence path escapes root: {rel}")
            self.assertTrue(path.is_file(), f"historical evidence source missing: {rel}")
            self.assertEqual(
                sha256_file(path),
                row.get("sha256"),
                f"historical evidence SHA mismatch: {rel}",
            )

    def test_22_class_order_verifier_hashes_lineage_before_model_loading(self):
        source = (
            ROOT / "journal_extension/scripts/verify_teacher_class_order.py"
        ).read_text(encoding="utf-8")
        hash_gate = source.index("require_sha256(p, expected")
        model_load = source.index("teacher, factory_identity = load_exact_teacher(")
        self.assertLess(hash_gate, model_load)


if __name__ == "__main__":
    unittest.main()
