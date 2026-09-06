import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "journal_extension" / "src"))

from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file, sha256_json
from cropcop_je.publication import (
    EVIDENCE_AUTHOR_EMAIL,
    EVIDENCE_AUTHOR_NAME,
    publish_to_github_branch,
)
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

    def test_00_batch_gate_has_actionable_saved_version_instruction(self):
        from cropcop_je.smoke_handoff import require_qualifying_kaggle_batch
        with self.assertRaisesRegex(SmokeHandoffError, "Save Version -> Save & Run All"):
            require_qualifying_kaggle_batch(
                context="fixture",
                run_type="Interactive",
            )
        self.assertEqual(
            require_qualifying_kaggle_batch(context="fixture", run_type="Batch"),
            "Batch",
        )

    def test_00b_bootstrap_records_and_enforces_batch_run_type(self):
        self.assertIn("require_qualifying_kaggle_batch(", self.bootstrap_source)
        self.assertIn('"kaggle_run_type": run_type', self.bootstrap_source)
        self.assertIn('context=f"canonical bootstrap ({args.phase})"', self.bootstrap_source)

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
        smoke_route = self.code[
            self.code.index('if EXECUTION_PHASE in {"smoke-write", "smoke-restore"}:'):
            self.code.index('elif EXECUTION_PHASE == "g1":')
        ]
        self.assertNotIn("run_g1.py", smoke_route)

    def test_25_notebook_clock_begins_before_clone_and_install(self):
        clock = self.code.index("CROPCOP_NOTEBOOK_STARTED_MONOTONIC")
        self.assertLess(clock, self.code.index('"clone"'))
        self.assertLess(clock, self.code.index('"pip"'))

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
        self.assertEqual(lines[0], "import json")
        self.assertEqual(lines[1], "import os, platform, shutil, stat, subprocess, sys, tempfile, time")
        self.assertEqual(lines[2], "from pathlib import Path")
        self.assertEqual(lines[3], "from urllib.error import HTTPError, URLError")
        self.assertIn(
            'AUTHORIZED_SOURCE_SHA = "3c71331494b3e031bbbbc3f08d27cd2605c31097"',
            lines,
        )
        for token in (
            "CROPCOP_RFDV_ROOT",
            "CROPCOP_FINAL_V1_ROOT",
            "CROPCOP_G1_PRIVATE_DATASET_SLUG",
            "CROPCOP_G1_INPUT_ROOT",
        ):
            self.assertIn(token, source)
        # Escaped newlines are legitimate inside the askpass Python string literal.
        self.assertIn('"#!/usr/bin/env python3\\n"', source)

    def test_35_notebook_generator_serializes_physical_source_lines(self):
        generator = (ROOT / "journal_extension/kaggle/generate_canonical_notebook.py").read_text()
        self.assertIn("splitlines(keepends=True)", generator)
        self.assertIn('compile(CODE, "canonical_lane.ipynb", "exec")', generator)
        self.assertIn("len(CODE.splitlines()) <= 40", generator)

    def test_36_github_auth_preflight_never_prints_token_value(self):
        self.assertIn("token_value_not_printed=true", self.code)
        self.assertNotIn('print(os.environ["CROPCOP_GITHUB_TOKEN"])', self.code)

    def test_37_github_auth_uses_nonempty_username_and_pat_as_password(self):
        self.assertIn('os.environ["CROPCOP_GIT_USERNAME"] = _git_username', self.code)
        self.assertIn("if 'username' in p:", self.code)
        self.assertIn("print(os.environ['CROPCOP_GIT_USERNAME'])", self.code)
        self.assertIn("elif 'password' in p:", self.code)
        self.assertIn("print(os.environ['CROPCOP_GITHUB_TOKEN'])", self.code)

    def test_38_github_auth_proves_private_repo_before_clone(self):
        self.assertLess(self.code.index("GitHub API auth preflight: PASS"), self.code.index('"clone"'))
        self.assertIn("https://api.github.com/repos/", self.code)
        self.assertIn("_repo_payload.get(\"private\") is not True", self.code)

    def test_39_git_https_read_preflight_occurs_before_clone(self):
        self.assertLess(self.code.index('"ls-remote"'), self.code.index('"clone"'))
        self.assertIn("GitHub Git-over-HTTPS read preflight: PASS", self.code)

    def test_40_evidence_write_permission_is_dry_run_and_pre_smoke(self):
        self.assertIn('"--dry-run"', self.code)
        self.assertIn("run-evidence/auth-probe-", self.code)
        self.assertIn("no ref created", self.code)
        self.assertLess(self.code.index('"--dry-run"'), self.code.index("smoke_infrastructure.py"))

    def test_40a_wrapper_rejects_interactive_before_secrets_network_clone_or_install(self):
        gate = self.code.index("CROPCOP QUALIFICATION NOT STARTED")
        self.assertLess(gate, self.code.index("from kaggle_secrets import UserSecretsClient"))
        self.assertLess(gate, self.code.index("GitHub API auth preflight"))
        self.assertLess(gate, self.code.index('"clone"'))
        self.assertLess(gate, self.code.index('"pip"'))
        self.assertIn("Save Version -> Save & Run All", self.code)
        self.assertIn('if _kaggle_run_type != "Batch":', self.code)

    def test_40b_wrapper_batch_gate_runs_before_mutable_smoke_output(self):
        gate = self.code.index('if _kaggle_run_type != "Batch":')
        self.assertLess(gate, self.code.index("smoke_infrastructure.py"))
        self.assertIn("No source clone, dependency install, smoke checkpoint, or qualification evidence", self.code)

    def test_41_auth_errors_are_token_redacted(self):
        self.assertIn('.replace(_token, "<redacted>")', self.code)
        self.assertIn("CROPCOP_GITHUB_TOKEN contains whitespace/newlines", self.code)
        self.assertIn("appears to include surrounding quotes", self.code)




    def test_42_tracked_lf_text_blobs_are_canonical_in_git_index(self):
        output = subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-files", "--eol"],
            text=True,
        )
        offenders = []
        for line in output.splitlines():
            if "\t" not in line:
                continue
            meta, path = line.split("\t", 1)
            fields = meta.split()
            index_eol = fields[0] if fields else ""
            if "eol=lf" in meta and index_eol in {"i/crlf", "i/mixed"}:
                offenders.append((path, meta))
        self.assertEqual(offenders, [], f"non-canonical LF blobs: {offenders}")



    def test_43_fresh_linux_checkout_of_exact_head_is_clean(self):
        head = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        with tempfile.TemporaryDirectory() as td:
            clone = Path(td) / "fresh-checkout"
            env = dict(os.environ)
            env["GIT_CONFIG_NOSYSTEM"] = "1"
            env["GIT_CONFIG_GLOBAL"] = os.devnull
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--quiet",
                    "--no-hardlinks",
                    "--no-checkout",
                    str(ROOT),
                    str(clone),
                ],
                env=env,
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(clone), "checkout", "--quiet", "--detach", head],
                env=env,
                check=True,
            )
            actual = subprocess.check_output(
                ["git", "-C", str(clone), "rev-parse", "HEAD"],
                env=env,
                text=True,
            ).strip()
            status = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(clone),
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                ],
                env=env,
                text=True,
            )
            self.assertEqual(actual, head)
            self.assertEqual(status, "", f"fresh exact-head checkout is dirty: {status}")



    def test_44_publication_sets_isolated_git_identity(self):
        source = (ROOT / "journal_extension/src/cropcop_je/publication.py").read_text()
        self.assertIn('env["GIT_AUTHOR_NAME"] = EVIDENCE_AUTHOR_NAME', source)
        self.assertIn('env["GIT_AUTHOR_EMAIL"] = EVIDENCE_AUTHOR_EMAIL', source)
        self.assertIn('env["GIT_COMMITTER_NAME"] = EVIDENCE_AUTHOR_NAME', source)
        self.assertIn('env["GIT_COMMITTER_EMAIL"] = EVIDENCE_AUTHOR_EMAIL', source)

    def test_45_publication_twice_works_without_global_git_identity(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            remote = td / "remote.git"
            repo = td / "repo"
            empty_home = td / "empty-home"
            empty_home.mkdir()

            isolated = dict(os.environ)
            isolated["HOME"] = str(empty_home)
            isolated["GIT_CONFIG_NOSYSTEM"] = "1"
            isolated["GIT_CONFIG_GLOBAL"] = os.devnull

            subprocess.run(["git", "init", "--bare", "--quiet", str(remote)], env=isolated, check=True)
            subprocess.run(["git", "init", "--quiet", str(repo)], env=isolated, check=True)
            (repo / "README.md").write_text("source\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "README.md"], env=isolated, check=True)
            source_env = dict(isolated)
            source_env.update({
                "GIT_AUTHOR_NAME": "Source Fixture",
                "GIT_AUTHOR_EMAIL": "source@localhost.invalid",
                "GIT_COMMITTER_NAME": "Source Fixture",
                "GIT_COMMITTER_EMAIL": "source@localhost.invalid",
            })
            subprocess.run(
                ["git", "-C", str(repo), "commit", "--quiet", "-m", "source"],
                env=source_env,
                check=True,
            )
            source_sha = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                env=isolated,
                text=True,
            ).strip()
            subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", str(remote)], env=isolated, check=True)

            evidence = td / "metrics.json"
            evidence.write_text('{"status":"PENDING"}\n', encoding="utf-8")
            publish_env = dict(isolated)
            publish_env.update({
                "CROPCOP_GITHUB_TOKEN": "dummy-local-transport-token",
                "CROPCOP_GIT_USERNAME": "rana-m-ahmed",
            })
            with mock.patch.dict(os.environ, publish_env, clear=True):
                branch1 = publish_to_github_branch(
                    repo_dir=repo,
                    source_git_sha=source_sha,
                    run_id="SMOKE-LOCAL",
                    files=[evidence],
                )
                evidence.write_text('{"status":"PASS"}\n', encoding="utf-8")
                branch2 = publish_to_github_branch(
                    repo_dir=repo,
                    source_git_sha=source_sha,
                    run_id="SMOKE-LOCAL",
                    files=[evidence],
                )

            self.assertEqual(branch1, "run-evidence/SMOKE-LOCAL")
            self.assertEqual(branch2, branch1)
            remote_ref = "refs/heads/run-evidence/SMOKE-LOCAL"
            count = int(subprocess.check_output(
                ["git", "--git-dir", str(remote), "rev-list", "--count", remote_ref],
                env=isolated,
                text=True,
            ).strip())
            self.assertEqual(count, 3)
            author = subprocess.check_output(
                ["git", "--git-dir", str(remote), "log", "-1", "--format=%an <%ae>", remote_ref],
                env=isolated,
                text=True,
            ).strip()
            self.assertEqual(author, f"{EVIDENCE_AUTHOR_NAME} <{EVIDENCE_AUTHOR_EMAIL}>")
            published = subprocess.check_output(
                [
                    "git",
                    "--git-dir",
                    str(remote),
                    "show",
                    f"{remote_ref}:journal_extension/evidence/public/runs/SMOKE-LOCAL/metrics.json",
                ],
                env=isolated,
                text=True,
            )
            self.assertEqual(published, '{"status":"PASS"}\n')

    def test_46_publication_git_error_redacts_token(self):
        from cropcop_je.publication import _run_git
        secret = "github_pat_" + ("x" * 40)
        env = dict(os.environ)
        env["CROPCOP_GITHUB_TOKEN"] = secret
        with mock.patch(
            "cropcop_je.publication.subprocess.run",
            return_value=subprocess.CompletedProcess(
                ["git"], 128, "", f"fatal: rejected {secret}"
            ),
        ):
            with self.assertRaises(Exception) as ctx:
                _run_git(["git", "push"], env=env, context="synthetic publication")
        self.assertNotIn(secret, str(ctx.exception))
        self.assertIn("<redacted>", str(ctx.exception))



if __name__ == "__main__":
    unittest.main()
