from __future__ import annotations

import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.g1_package import (
    G1Package,
    G1PackageError,
    create_g1_package,
    mount_g1_input,
    readiness_transport_dry_run,
    safe_extract_g1_package,
)
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
            "preflight_private_target",
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


if __name__ == "__main__":
    unittest.main()
