import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "journal_extension" / "src"))

from cropcop_je.g1 import (
    AUTHORITY_ID, AUTHORITY_SHA256, CLASS_MAP_SHA256, MANIFEST_SHA256,
    MNV4_MODEL_NAME, PAIR_SPECS, TEACHER_SHA256, TIMM_VERSION,
    assert_g1_creation_target_fresh, dependency_lock_hash, factory_bundle_hash,
    g1_seal_hash, validate_dependency_environment, validate_dependency_lock_object,
    validate_g1_seal_object, validate_teacher_class_order_evidence,
    validate_teacher_factory_bundle,
)
from cropcop_je.g2 import build_g2_barrier, validate_principal_gate_bindings
from cropcop_je.persistence import FilesystemStore, validate_durable_locator_template
from cropcop_je.runlog import assert_resume_identity
from cropcop_je.session import (
    FINALIZATION_MARGIN_ENV, HARD_LIMIT_ENV, START_ENV, SessionBudget,
)
from cropcop_je.source_state import SourceStateError, verify_clean_source


PRETRAINED = "1" * 64
FACTORY = "2" * 64
DEP = "3" * 64
STACK = "4" * 64
ORDER = "5" * 64
BYTE_EV = "6" * 64
PROV = "7" * 64
SMOKE = "8" * 64


def valid_seal(source="a" * 40):
    pairs = {}
    for i, (key, spec) in enumerate(PAIR_SPECS.items(), 1):
        pairs[key] = {
            "pair_id": spec["pair_id"],
            "seed": spec["seed"],
            "authorized_consumers": list(spec["consumers"]),
            "pretrained_sha256": PRETRAINED,
            "sha256": str(i) * 64,
            "bytes": 100 + i,
            "basename": f"PAIR_INIT_{key}.pt",
            "evidence_sha256": chr(96 + i) * 64,
        }
    seal = {
        "schema_version": "1.0",
        "authority": {"id": AUTHORITY_ID, "sha256": AUTHORITY_SHA256},
        "source_git_sha": source,
        "dataset": {"manifest_sha256": MANIFEST_SHA256, "class_map_sha256": CLASS_MAP_SHA256},
        "student": {
            "model_name": MNV4_MODEL_NAME,
            "timm_version": TIMM_VERSION,
            "pretrained": {
                "sha256": PRETRAINED,
                "bytes": 12345,
                "basename": "model.safetensors",
                "source_kind": "timm_pretrained_cfg_hf_hub",
                "source_locator": "timm/mobilenetv4_conv_medium.e500_r256_in1k",
                "timm_pretrained_cfg_sha256": "9" * 64,
                "provenance_sha256": PROV,
            },
        },
        "pair_initializations": pairs,
        "teacher": {
            "checkpoint_sha256": TEACHER_SHA256,
            "checkpoint_bytes": 54321,
            "class_map_sha256": CLASS_MAP_SHA256,
            "byte_evidence_sha256": BYTE_EV,
            "factory_entrypoint": "trusted_factory:build",
            "factory_bundle_sha256": FACTORY,
            "factory_manifest_sha256": "b" * 64,
            "class_order_evidence_sha256": ORDER,
        },
        "dependency_lock_sha256": DEP,
        "infra_smoke_evidence_sha256": SMOKE,
        "sealed_at_utc": "2026-09-05T00:00:00+00:00",
    }
    seal["g1_seal_sha256"] = g1_seal_hash(seal)
    return seal


def measured():
    return {
        "sec_per_optimizer_step": 1.0,
        "examples_per_second": 64.0,
        "dataloader_wait_seconds": 2.0,
        "dataloader_examples_per_wait_second": 1000.0,
        "peak_gpu_memory_bytes": 1024,
        "checkpoint_save_seconds": 3.0,
        "checkpoint_load_seconds": 1.0,
        "durable_sync_seconds": 4.0,
        "validation_forward_benchmark": {"end_to_end_examples_per_second": 512.0},
    }


def calibration(cid, seal, *, g1=None, pretrained=PRETRAINED, dep=DEP, stack=STACK, source=None, factory=FACTORY):
    row = {
        "schema_version": "3.0",
        "status": "PASS",
        "calibration_id": cid,
        "resume_success": True,
        "calibration_weights_scientific": False,
        "source_git_commit": source or seal["source_git_sha"],
        "software_stack_sha256": stack,
        "g1_seal_sha256": g1 or seal["g1_seal_sha256"],
        "dependency_lock_sha256": dep,
        "mnv4_pretrained_sha256": pretrained,
        "teacher_checkpoint_sha256": None,
        "teacher_factory_bundle_sha256": None,
        "measured": measured(),
    }
    if cid == "CAL-MNV4-TEACHER":
        row["teacher_checkpoint_sha256"] = TEACHER_SHA256
        row["teacher_factory_bundle_sha256"] = factory
    return row


class PreG1IntegrityTests(unittest.TestCase):
    def test_dependency_lock_self_hash_is_valid(self):
        lock = json.loads((ROOT / "journal_extension/locks/execution_dependency_lock.json").read_text())
        self.assertEqual(lock["dependency_lock_sha256"], dependency_lock_hash(lock))
        self.assertEqual(validate_dependency_lock_object(lock), [])

    def test_non_core_dependency_drift_is_rejected(self):
        lock = json.loads((ROOT / "journal_extension/locks/execution_dependency_lock.json").read_text())
        versions = dict(lock["packages"])
        versions["numpy"] = "0.0.0"
        with mock.patch("cropcop_je.g1.platform.python_version", return_value=lock["python"]), \
             mock.patch("cropcop_je.g1.importlib.metadata.version", side_effect=lambda name: versions[name]):
            errors = validate_dependency_environment(lock)
        self.assertTrue(any("numpy" in x for x in errors), errors)

    def test_valid_g1_seal_contract(self):
        self.assertEqual(validate_g1_seal_object(valid_seal()), [])

    def test_g1_rejects_pair_with_different_pretrained_sha(self):
        seal = valid_seal()
        seal["pair_initializations"]["S2"]["pretrained_sha256"] = "f" * 64
        seal["g1_seal_sha256"] = g1_seal_hash(seal)
        self.assertTrue(any("S2" in x and "pretrained" in x for x in validate_g1_seal_object(seal)))

    def test_teacher_factory_bundle_drift_is_rejected(self):
        manifest = {
            "schema_version": "1.0",
            "entrypoint": "trusted_factory:build",
            "output_order_transform": "none",
            "files": [{"path": "factory.py", "sha256": "a" * 64, "bytes": 1}],
        }
        manifest["bundle_sha256"] = factory_bundle_hash(manifest)
        self.assertEqual(validate_teacher_factory_bundle(manifest), [])
        manifest["files"][0]["sha256"] = "b" * 64
        self.assertTrue(any("bundle hash" in x for x in validate_teacher_factory_bundle(manifest)))

    def test_teacher_class_order_requires_historical_evidence_not_shape_only(self):
        evidence = {
            "status": "PASS",
            "class_map_sha256": CLASS_MAP_SHA256,
            "teacher_output_width": 120,
            "classifier_out_features": 120,
            "historical_index_semantics": "class_map_index",
            "factory_output_order_transform": "none",
            "teacher_factory_bundle_sha256": FACTORY,
            "not_inferred_from_shape_only": False,
            "historical_evidence_sources": [],
        }
        errors = validate_teacher_class_order_evidence(evidence, factory_bundle_sha256=FACTORY)
        self.assertTrue(any("shape" in x for x in errors))
        self.assertTrue(any("historical source" in x for x in errors))

    def test_teacher_class_order_rejects_factory_bundle_mismatch(self):
        evidence = {
            "status": "PASS",
            "class_map_sha256": CLASS_MAP_SHA256,
            "teacher_output_width": 120,
            "classifier_out_features": 120,
            "historical_index_semantics": "class_map_index",
            "factory_output_order_transform": "none",
            "teacher_factory_bundle_sha256": FACTORY,
            "not_inferred_from_shape_only": True,
            "historical_evidence_sources": [{"kind": "historical_config", "sha256": "c" * 64}],
        }
        errors = validate_teacher_class_order_evidence(evidence, factory_bundle_sha256="d" * 64)
        self.assertTrue(any("different teacher factory" in x for x in errors))

    def test_pair_init_cannot_silently_regenerate_after_seal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            assert_g1_creation_target_fresh(root)
            (root / "evidence").mkdir()
            (root / "evidence/PAIR_INIT_S1.json").write_text("{}")
            with self.assertRaises(Exception):
                assert_g1_creation_target_fresh(root)

    def _git_repo(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "ci@example.invalid"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "CI"], cwd=root, check=True)
        (root / "source.py").write_text("x=1\n")
        subprocess.run(["git", "add", "source.py"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        return td, root, sha

    def test_clean_source_exact_head_passes(self):
        td, root, sha = self._git_repo()
        try:
            outside = Path(td.name).parent / f"{Path(td.name).name}-out"
            result = verify_clean_source(root, authorized_source_sha=sha, output_roots=[outside])
            self.assertEqual(result["status"], "PASS")
        finally:
            td.cleanup()

    def test_dirty_git_working_tree_is_rejected(self):
        td, root, sha = self._git_repo()
        try:
            (root / "source.py").write_text("x=2\n")
            with self.assertRaises(SourceStateError):
                verify_clean_source(root, authorized_source_sha=sha)
        finally:
            td.cleanup()

    def test_incorrect_head_is_rejected(self):
        td, root, _sha = self._git_repo()
        try:
            with self.assertRaises(SourceStateError):
                verify_clean_source(root, authorized_source_sha="f" * 40)
        finally:
            td.cleanup()

    def test_mutable_output_inside_repo_is_rejected(self):
        td, root, sha = self._git_repo()
        try:
            with self.assertRaises(SourceStateError):
                verify_clean_source(root, authorized_source_sha=sha, output_roots=[root / "outputs"])
        finally:
            td.cleanup()

    def test_g2_rejects_different_global_g1_seals(self):
        seal = valid_seal()
        rows = [
            calibration("CAL-MNV4-DIRECT", seal),
            calibration("CAL-MNV4-TEACHER", seal, g1="e" * 64),
            calibration("CAL-CNXTT", seal),
        ]
        barrier = build_g2_barrier(rows)
        self.assertEqual(barrier["status"], "FAIL")
        self.assertTrue(any("G1 seal" in x for x in barrier["errors"]))

    def test_g2_rejects_pretrained_and_dependency_drift(self):
        seal = valid_seal()
        rows = [
            calibration("CAL-MNV4-DIRECT", seal),
            calibration("CAL-MNV4-TEACHER", seal, pretrained="e" * 64),
            calibration("CAL-CNXTT", seal, dep="f" * 64),
        ]
        barrier = build_g2_barrier(rows)
        self.assertEqual(barrier["status"], "FAIL")
        self.assertTrue(any("pretrained" in x.lower() for x in barrier["errors"]))
        self.assertTrue(any("dependency" in x.lower() for x in barrier["errors"]))

    def test_principal_launch_without_g1_or_g2_is_rejected(self):
        source = "a" * 40
        self.assertTrue(validate_principal_gate_bindings(None, None, source_git_sha=source))
        self.assertTrue(validate_principal_gate_bindings(valid_seal(source), None, source_git_sha=source))

    def test_principal_g1_g2_binding_passes_only_when_coherent(self):
        seal = valid_seal()
        barrier = build_g2_barrier([
            calibration("CAL-MNV4-DIRECT", seal),
            calibration("CAL-MNV4-TEACHER", seal),
            calibration("CAL-CNXTT", seal),
        ])
        self.assertEqual(barrier["status"], "PASS", barrier["errors"])
        self.assertEqual(validate_principal_gate_bindings(seal, barrier, source_git_sha=seal["source_git_sha"]), [])
        bad = dict(barrier)
        bad["g1_seal_sha256"] = "f" * 64
        from cropcop_je.g2 import barrier_hash
        bad["barrier_sha256"] = barrier_hash(bad)
        self.assertTrue(validate_principal_gate_bindings(seal, bad, source_git_sha=seal["source_git_sha"]))

    def test_resume_rejects_changed_g1_and_g2_seals(self):
        base = {
            "experiment_id": "R04", "authority_id": AUTHORITY_ID, "source_git_commit": "a" * 40,
            "config_sha256": "b" * 64, "ctc_v2_sha256": "c" * 64, "manifest_sha256": MANIFEST_SHA256,
            "class_map_sha256": CLASS_MAP_SHA256, "seed": 1, "student_init_sha256": "d" * 64,
            "pretrained_sha256": PRETRAINED, "teacher_sha256": None, "teacher_factory_sha256": None,
            "teacher_factory_bundle_sha256": None, "software_stack_sha256": STACK,
            "dependency_lock_sha256": DEP, "g1_seal_sha256": "e" * 64,
            "g2_barrier_sha256": "f" * 64, "lane_id": "K1",
        }
        assert_resume_identity(base, dict(base))
        for field in ("g1_seal_sha256", "g2_barrier_sha256"):
            changed = dict(base)
            changed[field] = "0" * 64
            with self.assertRaises(ValueError):
                assert_resume_identity(base, changed)

    def test_notebook_global_clock_is_shared_across_subprocess_semantics(self):
        env = {
            START_ENV: "100.0",
            HARD_LIMIT_ENV: "43200",
            FINALIZATION_MARGIN_ENV: "3600",
        }
        with mock.patch.dict(os.environ, env, clear=False), mock.patch("cropcop_je.session.time.monotonic", return_value=10000.0):
            first = SessionBudget.from_environment(require_global_clock=True)
            second = SessionBudget.from_environment(require_global_clock=True)
            self.assertEqual(first.started_monotonic, second.started_monotonic)
            self.assertEqual(first.elapsed, second.elapsed)
            self.assertEqual(first.remaining_safe_seconds, second.remaining_safe_seconds)

    def test_second_principal_does_not_receive_fresh_twelve_hours(self):
        env = {START_ENV: "0", HARD_LIMIT_ENV: "43200", FINALIZATION_MARGIN_ENV: "3600"}
        with mock.patch.dict(os.environ, env, clear=False), mock.patch("cropcop_je.session.time.monotonic", return_value=39400):
            budget = SessionBudget.from_environment(require_global_clock=True)
            self.assertFalse(budget.can_start_phase(1000, estimated_checkpoint_seconds=100, estimated_sync_seconds=200))
            self.assertTrue(budget.should_finalize(estimated_checkpoint_seconds=100, estimated_sync_seconds=200))

    def test_durable_locator_collision_protection(self):
        with self.assertRaises(ValueError):
            validate_durable_locator_template("kaggle-dataset", "owner/shared", ["R04", "R05"])
        resolved = validate_durable_locator_template(
            "kaggle-dataset", "owner/cropcop-{run_id_lower}", ["R04", "R05"]
        )
        self.assertEqual(len(set(resolved.values())), 2)

    def test_filesystem_store_restores_previous_generation_after_interrupted_rename(self):
        with tempfile.TemporaryDirectory() as td:
            store = FilesystemStore(Path(td) / "durable")
            previous = store._previous_dir("RUN")
            (previous / "objects").mkdir(parents=True)
            (previous / "durable_sync.json").write_text(json.dumps({
                "schema_version": "1.0", "run_id": "RUN", "segment_id": "S", "complete": True
            }))
            (previous / "checkpoint_index.json").write_text(json.dumps({"schema_version": "2.0"}))
            (previous / "objects/state.ckpt").write_bytes(b"state")
            dest = Path(td) / "restore"
            self.assertTrue(store.restore(dest, run_id="RUN"))
            self.assertEqual((dest / "objects/state.ckpt").read_bytes(), b"state")

    def test_terminal_selected_checkpoint_is_retained_by_filesystem_sync(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source"
            objects = source / "objects"
            objects.mkdir(parents=True)
            latest = objects / "latest.ckpt"
            selected = objects / "selected.ckpt"
            latest.write_bytes(b"latest")
            selected.write_bytes(b"selected")
            sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            (source / "checkpoint_index.json").write_text(json.dumps({
                "schema_version": "2.0", "generation": 2,
                "latest": {"relative_path": "objects/latest.ckpt", "sha256": sha(latest)},
                "previous": None,
                "selected": {"relative_path": "objects/selected.ckpt", "sha256": sha(selected)},
            }))
            store = FilesystemStore(Path(td) / "durable")
            store.sync(source, run_id="RUN", segment_id="S1")
            target = store._run_dir("RUN")
            self.assertEqual((target / "objects/selected.ckpt").read_bytes(), b"selected")
            self.assertEqual((target / "objects/latest.ckpt").read_bytes(), b"latest")

    def test_clean_session_runner_only_consumes_sealed_pair_initializations(self):
        runner = (ROOT / "journal_extension/kaggle/run_lane.py").read_text()
        self.assertNotIn("prepare_pair_init.py", runner)
        self.assertNotIn("save_pair_initialization", runner)
        self.assertIn('PAIR_INIT_S{idx}.pt', runner)
        self.assertIn("G1_MODEL_IDENTITY_SEAL.json", runner)

    def test_canonical_notebook_starts_clock_before_clone_and_supports_explicit_phases(self):
        nb = json.loads((ROOT / "journal_extension/kaggle/canonical_lane.ipynb").read_text())
        code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        self.assertLess(code.index("CROPCOP_NOTEBOOK_STARTED_MONOTONIC"), code.index('"clone"'))
        for phase in ("smoke-write", "smoke-restore", "g1", "calibration", "principal"):
            self.assertIn(phase, code)
        self.assertIn("PYTHONDONTWRITEBYTECODE", code)
        self.assertIn("GIT_ASKPASS", code)

    def test_planned_rollover_syncs_durable_state_before_exit_record(self):
        source = (ROOT / "journal_extension/scripts/run_training.py").read_text()
        self.assertLess(source.index("persistence = store.sync"), source.index('if result.get("planned_rollover")'))

    def test_one_principal_state_per_saved_version_is_default(self):
        source = (ROOT / "journal_extension/kaggle/run_lane.py").read_text()
        self.assertIn("select_principal(lane)", source)
        self.assertIn("no second principal state is started automatically", source)
        self.assertNotIn('for item in lane["principal"]:', source)

    def test_real_g1_requires_prior_smoke_evidence(self):
        source = (ROOT / "journal_extension/scripts/seal_g1.py").read_text()
        self.assertIn("--infra-smoke-evidence", source)
        self.assertIn("real G1 sealing requires a green real-Kaggle", source)


if __name__ == "__main__":
    unittest.main()
