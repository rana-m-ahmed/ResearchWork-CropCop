import hashlib
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "journal_extension" / "src"))

from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file, sha256_json
from cropcop_je.smoke_handoff import (
    RestoreSequence,
    SmokeHandoffError,
    build_smoke_a_manifest,
    locate_smoke_a_export,
    manifest_self_hash,
    recovery_file_records,
    restore_verified_recovery_bundle,
    snapshot_files,
    validate_smoke_a_manifest,
    verify_smoke_a_export,
)

SOURCE = "a" * 40
TREE = "b" * 40
DEP = "c" * 64
ENV = "d" * 64
QID = "INFRA-SMOKE-aaaaaaaaaaaa-123456789abc"
IDENTITY = {
    "experiment_id": "INFRA-SMOKE",
    "authority_id": "NON_SCIENTIFIC_INFRASTRUCTURE",
    "source_git_commit": SOURCE,
    "lane_id": "K1",
    "qualification_id": QID,
    "synthetic_only": True,
    "scientific": False,
}


def make_export(root: Path, *, source=SOURCE, dep=DEP, qid=QID) -> dict:
    objects = root / "objects"
    objects.mkdir(parents=True)
    checkpoint = objects / "latest.g00000001.fake.ckpt"
    checkpoint.write_bytes(b"synthetic-checkpoint-only")
    checkpoint_sha = sha256_file(checkpoint)
    identity = dict(IDENTITY, source_git_commit=source, qualification_id=qid)
    identity_sha = sha256_json(identity)
    index = {
        "schema_version": "2.0",
        "generation": 1,
        "latest": {
            "relative_path": checkpoint.relative_to(root).as_posix(),
            "sha256": checkpoint_sha,
            "bytes": checkpoint.stat().st_size,
            "generation": 1,
            "optimizer_step": 3,
            "epoch": 0,
            "batch_in_epoch": 3,
            "identity_sha256": identity_sha,
            "created_at_utc": "2026-09-05T00:00:00+00:00",
        },
        "previous": None,
        "selected": None,
    }
    atomic_write_json(root / "checkpoint_index.json", index)
    evidence = {
        "schema_version": "2.0",
        "status": "PASS",
        "qualification_id": qid,
        "mode": "WRITE",
        "scientific": False,
        "synthetic_unprotected_data_only": True,
        "source_git_sha": source,
        "dependency_lock_sha256": dep,
        "checkpoint_identity": identity,
        "checkpoint_sha256": checkpoint_sha,
        "optimizer_step": 3,
    }
    atomic_write_json(root / "SMOKE_A_EVIDENCE.json", evidence)
    manifest = build_smoke_a_manifest(
        qualification_id=qid,
        source_git_sha=source,
        source_tree_sha=TREE,
        dependency_lock_sha256=dep,
        environment_identity_sha256=ENV,
        checkpoint_relative_path=checkpoint.relative_to(root).as_posix(),
        checkpoint_sha256=checkpoint_sha,
        checkpoint_bytes=checkpoint.stat().st_size,
        checkpoint_identity_sha256=identity_sha,
        optimizer_step=3,
        recovery_files=recovery_file_records(root),
        smoke_a_evidence_sha256=sha256_file(root / "SMOKE_A_EVIDENCE.json"),
        created_at_utc="2026-09-05T00:00:00+00:00",
    )
    atomic_write_json(root / "SMOKE_A_MANIFEST.json", manifest)
    return manifest


class SmokeSRTests(unittest.TestCase):
    def setUp(self):
        self.smoke_source = (ROOT / "journal_extension/scripts/smoke_infrastructure.py").read_text()
        self.bootstrap_source = (ROOT / "journal_extension/kaggle/bootstrap_clean_session.py").read_text()
        self.notebook = json.loads((ROOT / "journal_extension/kaggle/canonical_lane.ipynb").read_text())
        self.code = "".join(
            "".join(cell.get("source", []))
            for cell in self.notebook["cells"]
            if cell.get("cell_type") == "code"
        )

    def _restore_source(self):
        start = self.smoke_source.index("def smoke_restore")
        end = self.smoke_source.index("\ndef main()", start)
        return self.smoke_source[start:end]

    def test_01_smoke_write_has_no_kaggle_api_credential_requirement(self):
        write = self.smoke_source[self.smoke_source.index("def smoke_write"):self.smoke_source.index("def smoke_restore")]
        self.assertNotIn("KAGGLE_USERNAME", write)
        self.assertNotIn("KAGGLE_KEY", write)
        self.assertNotIn("kaggle datasets", write.lower())

    def test_02_smoke_restore_has_no_kaggle_api_credential_requirement(self):
        restore = self._restore_source()
        self.assertNotIn("KAGGLE_USERNAME", restore)
        self.assertNotIn("KAGGLE_KEY", restore)
        self.assertNotIn("kaggle datasets", restore.lower())

    def test_03_smoke_bootstrap_only_requires_github_token(self):
        self.assertIn('return ("CROPCOP_GITHUB_TOKEN",)', self.bootstrap_source)
        smoke_branch = self.bootstrap_source[
            self.bootstrap_source.index("if phase in SMOKE_PHASES"):
            self.bootstrap_source.index("if phase in NON_SMOKE_PHASES")
        ]
        self.assertNotIn("KAGGLE_USERNAME", smoke_branch)
        self.assertNotIn("KAGGLE_KEY", smoke_branch)

    def test_04_notebook_supports_smoke_write(self):
        self.assertIn("smoke-write", self.code)

    def test_05_notebook_supports_smoke_restore(self):
        self.assertIn("smoke-restore", self.code)

    def test_06_old_ambiguous_smoke_phase_cannot_route_production_smoke(self):
        self.assertNotIn("EXECUTION_PHASE == 'smoke'", self.code)
        self.assertNotIn("phase == 'smoke'", self.code)
        self.assertNotIn("'smoke','g1'", self.code)

    def test_07_smoke_a_export_contains_complete_recovery_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_export(root)
            self.assertTrue((root / "SMOKE_A_MANIFEST.json").is_file())
            self.assertTrue((root / "SMOKE_A_EVIDENCE.json").is_file())
            self.assertTrue((root / "checkpoint_index.json").is_file())
            self.assertTrue(any((root / "objects").iterdir()))
            verify_smoke_a_export(root, expected_source_sha=SOURCE, expected_dependency_lock_sha256=DEP)

    def test_08_smoke_a_manifest_binds_checkpoint_sha(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); manifest = make_export(root)
            rel = manifest["checkpoint_relative_path"]
            self.assertEqual(manifest["checkpoint_sha256"], sha256_file(root / rel))

    def test_09_smoke_a_manifest_binds_source_sha(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = make_export(Path(td))
            self.assertEqual(manifest["source_git_sha"], SOURCE)

    def test_10_smoke_a_manifest_binds_dependency_lock_sha(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = make_export(Path(td))
            self.assertEqual(manifest["dependency_lock_sha256"], DEP)

    def test_11_smoke_restore_requires_explicit_input_root(self):
        restore = self._restore_source()
        self.assertIn("smoke-restore requires explicit --smoke-a-input-root", restore)

    def test_12_smoke_restore_refuses_missing_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SmokeHandoffError):
                locate_smoke_a_export(td)

    def test_13_smoke_restore_refuses_source_sha_drift(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); make_export(root)
            with self.assertRaises(SmokeHandoffError):
                verify_smoke_a_export(root, expected_source_sha="f"*40, expected_dependency_lock_sha256=DEP)

    def test_14_smoke_restore_refuses_dependency_lock_drift(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); make_export(root)
            with self.assertRaises(SmokeHandoffError):
                verify_smoke_a_export(root, expected_source_sha=SOURCE, expected_dependency_lock_sha256="f"*64)

    def test_15_smoke_restore_refuses_qualification_id_inconsistency(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); make_export(root)
            evidence_path = root / "SMOKE_A_EVIDENCE.json"
            evidence = json.loads(evidence_path.read_text())
            evidence["qualification_id"] = QID + "-DIFFERENT"
            atomic_write_json(evidence_path, evidence)
            manifest = json.loads((root / "SMOKE_A_MANIFEST.json").read_text())
            manifest["smoke_a_evidence_sha256"] = sha256_file(evidence_path)
            manifest["manifest_sha256"] = manifest_self_hash(manifest)
            atomic_write_json(root / "SMOKE_A_MANIFEST.json", manifest)
            with self.assertRaises(SmokeHandoffError):
                verify_smoke_a_export(root, expected_source_sha=SOURCE, expected_dependency_lock_sha256=DEP)

    def test_16_smoke_restore_refuses_checkpoint_sha_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); manifest = make_export(root)
            (root / manifest["checkpoint_relative_path"]).write_bytes(b"tampered")
            with self.assertRaises(SmokeHandoffError):
                verify_smoke_a_export(root, expected_source_sha=SOURCE, expected_dependency_lock_sha256=DEP)

    def test_17_no_training_before_verified_recovery(self):
        restore = self._restore_source()
        self.assertLess(restore.index("verify_smoke_a_export"), restore.index("_advance("))
        self.assertLess(restore.index("recover_latest"), restore.index("_advance("))

    def test_18_no_checkpoint_write_before_verified_recovery_and_resume(self):
        restore = self._restore_source()
        self.assertLess(restore.index("sequence.require_resume_before_checkpoint"), restore.index("save_torch_checkpoint"))

    def test_19_attached_smoke_a_input_is_not_modified_by_restore(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "a"; root.mkdir()
            make_export(root)
            before = snapshot_files(root)
            verified = verify_smoke_a_export(root, expected_source_sha=SOURCE, expected_dependency_lock_sha256=DEP)
            restore_verified_recovery_bundle(verified, Path(td) / "restore")
            self.assertEqual(before, snapshot_files(root))

    def test_20_restored_optimizer_step_must_equal_smoke_a_manifest(self):
        restore = self._restore_source()
        self.assertIn('restored_step != int(manifest["optimizer_step"])', restore)

    def test_21_resumed_optimizer_step_must_advance(self):
        restore = self._restore_source()
        self.assertIn("if resumed_step <= restored_step", restore)

    def test_22_smoke_b_output_binds_smoke_a_evidence(self):
        restore = self._restore_source()
        self.assertIn('"smoke_a_evidence_sha256": manifest["smoke_a_evidence_sha256"]', restore)
        self.assertIn('"smoke_a_manifest_sha256": manifest["manifest_sha256"]', restore)

    def test_23_non_scientific_and_synthetic_flags_are_mandatory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); manifest = make_export(root)
            manifest["scientific"] = True
            manifest["manifest_sha256"] = manifest_self_hash(manifest)
            errors = validate_smoke_a_manifest(manifest, expected_source_sha=SOURCE, expected_dependency_lock_sha256=DEP)
            self.assertTrue(any("non-scientific" in e for e in errors))

    def test_24_neither_smoke_mode_falls_through_to_g1(self):
        smoke_route = self.code[self.code.index("if EXECUTION_PHASE in {'smoke-write','smoke-restore'}"):self.code.index("elif EXECUTION_PHASE == 'g1'")]
        self.assertNotIn("run_g1.py", smoke_route)

    def test_25_notebook_clock_begins_before_clone_and_install(self):
        clock = self.code.index("CROPCOP_NOTEBOOK_STARTED_MONOTONIC")
        self.assertLess(clock, self.code.index("git','clone"))
        self.assertLess(clock, self.code.index("'pip','install"))

    def test_26_canonical_notebook_is_valid_nbformat_and_thin(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        code_cells = [c for c in self.notebook["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(code_cells), 1)
        self.assertEqual(len([c for c in self.notebook["cells"] if c["cell_type"] == "markdown"]), 1)

    def test_27_mutable_outputs_are_outside_git_checkout(self):
        for value in (
            "/kaggle/working/cropcop-je-output",
            "/kaggle/working/cropcop-smoke-input",
            "/kaggle/working/cropcop-smoke-a-export",
            "/kaggle/working/cropcop-smoke-b-export",
        ):
            self.assertIn(value, self.code)
        self.assertIn("mutable output must be outside Git checkout", self.code)

    def test_28_restore_sequence_rejects_checkpoint_before_resume(self):
        seq = RestoreSequence()
        for event in ("READ_A","VERIFY_A","RESTORE_A","RECOVER_A","LOAD_A"):
            seq.advance(event)
        with self.assertRaises(SmokeHandoffError):
            seq.advance("CHECKPOINT_B")

    def test_29_failed_verify_creates_no_restore_destination(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "a"; root.mkdir()
            manifest = make_export(root)
            (root / manifest["checkpoint_relative_path"]).write_bytes(b"corrupt")
            destination = Path(td) / "restore"
            with self.assertRaises(SmokeHandoffError):
                verified = verify_smoke_a_export(root, expected_source_sha=SOURCE, expected_dependency_lock_sha256=DEP)
                restore_verified_recovery_bundle(verified, destination)
            self.assertFalse(destination.exists())

    def test_30_manifest_self_hash_detects_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = make_export(Path(td))
            self.assertEqual(manifest["manifest_sha256"], manifest_self_hash(manifest))
            manifest["optimizer_step"] = 999
            self.assertNotEqual(manifest["manifest_sha256"], manifest_self_hash(manifest))

    def test_31_no_kaggle_dataset_cli_in_smoke_path(self):
        self.assertNotIn("kaggle datasets", self.smoke_source.lower())
        self.assertNotIn("KagglePrivateDatasetStore", self.smoke_source)

    def test_32_smoke_script_requires_only_github_secret(self):
        common_start = self.smoke_source.index("def _common_preflight")
        common_end = self.smoke_source.index("def smoke_write")
        common = self.smoke_source[common_start:common_end]
        self.assertIn("CROPCOP_GITHUB_TOKEN", common)
        self.assertNotIn("KAGGLE_USERNAME", common)
        self.assertNotIn("KAGGLE_KEY", common)


    def test_33_canonical_notebook_code_compiles_as_jupyter_source(self):
        code_cells = [c for c in self.notebook["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(code_cells), 1)
        raw = code_cells[0]["source"]
        source = "".join(raw) if isinstance(raw, list) else raw
        compile(source, "canonical_lane.ipynb", "exec")

    def test_34_canonical_notebook_has_real_multiline_python_structure(self):
        code_cells = [c for c in self.notebook["cells"] if c["cell_type"] == "code"]
        raw = code_cells[0]["source"]
        source = "".join(raw) if isinstance(raw, list) else raw
        lines = source.splitlines()
        self.assertGreater(len(lines), 100)
        self.assertEqual(lines[0], "import os, platform, shutil, stat, subprocess, sys, tempfile, time")
        self.assertEqual(lines[1], "from pathlib import Path")
        self.assertIn(
            'AUTHORIZED_SOURCE_SHA = "045fcf5c80366438b69a54288d9081e9b57ed973"',
            lines,
        )
        # Escaped newlines are legitimate inside the askpass Python string literal.
        self.assertIn('"#!/usr/bin/env python3\\n"', source)

    def test_35_notebook_generator_serializes_physical_source_lines(self):
        generator = (ROOT / "journal_extension/kaggle/generate_canonical_notebook.py").read_text()
        self.assertIn("splitlines(keepends=True)", generator)
        self.assertIn('compile(CODE, "canonical_lane.ipynb", "exec")', generator)
        self.assertIn("len(CODE.splitlines()) <= 40", generator)



if __name__ == "__main__":
    unittest.main()
