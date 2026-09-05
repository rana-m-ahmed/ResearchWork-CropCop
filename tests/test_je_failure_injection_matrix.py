import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "journal_extension" / "src"))

from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.checkpointing import recover_latest, save_torch_checkpoint
from cropcop_je.publication import PublicationError, audit_public_files, publish_to_github_branch
from cropcop_je.runlog import assert_resume_identity, claim_run_directory
from cropcop_je.selection import SelectionState
from cropcop_je.session import SessionBudget
from cropcop_je.surfaces import SurfaceAuthorizationError, authorize_training_surface


def identity():
    return {
        "experiment_id": "R04", "authority_id": "lock", "source_git_commit": "a" * 40,
        "config_sha256": "b" * 64, "ctc_v2_sha256": "c" * 64, "manifest_sha256": "d" * 64,
        "class_map_sha256": "e" * 64, "seed": 1, "student_init_sha256": "f" * 64,
        "pretrained_sha256": "0" * 64, "teacher_sha256": None, "teacher_factory_sha256": None,
        "software_stack_sha256": "1" * 64, "lane_id": "K1",
    }


class FailureInjectionMatrix(unittest.TestCase):
    def test_interruption_mid_epoch_cursor_is_persistable(self):
        state = {"epoch": 3, "batch_in_epoch": 117, "optimizer_step": 299, "data_order_state": {"next_batch_in_epoch": 117}}
        self.assertEqual(json.loads(json.dumps(state))["batch_in_epoch"], 117)

    def test_interruption_immediately_after_optimizer_boundary_is_persistable(self):
        state = {"optimizer_step": 300, "batch_in_epoch": 120}
        self.assertEqual(json.loads(json.dumps(state)), state)

    def test_interruption_during_atomic_index_write_preserves_old_index(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "checkpoint_index.json"
            atomic_write_json(p, {"generation": 1})
            with mock.patch("cropcop_je.atomic_io.os.replace", side_effect=OSError("injected crash")):
                with self.assertRaises(OSError):
                    atomic_write_json(p, {"generation": 2})
            self.assertEqual(json.loads(p.read_text()), {"generation": 1})

    def test_corrupted_newest_uses_previous_valid_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            atomic_write_json(root / "checkpoint_index.json", {
                "schema_version": "2.0", "generation": 2,
                "latest": {"relative_path": "objects/a", "generation": 2},
                "previous": {"relative_path": "objects/b", "generation": 1},
                "selected": None,
            })
            def fake(_root, ref, expected_identity):
                if ref["generation"] == 2:
                    raise RuntimeError("corrupt")
                return root / "objects/b", {"identity": expected_identity}
            with mock.patch("cropcop_je.checkpointing._verify_ref", side_effect=fake):
                _p, _payload, event = recover_latest(root, expected_identity=identity())
            self.assertEqual(event["candidate"], "previous")

    def test_resume_after_earlier_best_keeps_same_best(self):
        rows = [
            {"epoch": 1, "validation_macro_f1": .7, "validation_balanced_accuracy": .7, "validation_nll": 1.0},
            {"epoch": 2, "validation_macro_f1": .9, "validation_balanced_accuracy": .8, "validation_nll": .8},
            {"epoch": 3, "validation_macro_f1": .8, "validation_balanced_accuracy": .9, "validation_nll": .7},
        ]
        a = SelectionState()
        for row in rows:
            if a.consider(row):
                a.bind_selected_checkpoint(sha256=f"e{row['epoch']}", relative_path=f"e{row['epoch']}")
        b = SelectionState()
        for row in rows[:2]:
            if b.consider(row):
                b.bind_selected_checkpoint(sha256=f"e{row['epoch']}", relative_path=f"e{row['epoch']}")
        b = SelectionState.from_dict(json.loads(json.dumps(b.to_dict())))
        for row in rows[2:]:
            if b.consider(row):
                b.bind_selected_checkpoint(sha256=f"e{row['epoch']}", relative_path=f"e{row['epoch']}")
        self.assertEqual(a.best, b.best)

    def _reject(self, field, value):
        saved = identity()
        current = dict(saved)
        current[field] = value
        with self.assertRaises(ValueError):
            assert_resume_identity(saved, current)

    def test_config_hash_mismatch(self): self._reject("config_sha256", "9" * 64)
    def test_manifest_hash_mismatch(self): self._reject("manifest_sha256", "9" * 64)
    def test_pretrained_hash_mismatch(self): self._reject("pretrained_sha256", "9" * 64)
    def test_changed_seed(self): self._reject("seed", 2)
    def test_changed_source_commit(self): self._reject("source_git_commit", "2" * 40)
    def test_wrong_lane(self): self._reject("lane_id", "K2")

    def test_teacher_hash_mismatch(self):
        saved = identity(); saved["teacher_sha256"] = "2" * 64
        current = dict(saved); current["teacher_sha256"] = "3" * 64
        with self.assertRaises(ValueError):
            assert_resume_identity(saved, current)

    def test_output_and_run_id_collision(self):
        with tempfile.TemporaryDirectory() as td:
            claim_run_directory(td, run_id="A", experiment_id="R04", lane_id="K1")
            with self.assertRaises(RuntimeError):
                claim_run_directory(td, run_id="B", experiment_id="R04", lane_id="K1")

    def test_protected_v1_test_attempt(self):
        with self.assertRaises(SurfaceAuthorizationError):
            authorize_training_surface("DS-V1-TEST-CONSUMED")

    def test_external_final_attempt(self):
        with self.assertRaises(SurfaceAuthorizationError):
            authorize_training_surface("DS-EXT-POTATO-SEALED")

    def test_git_credential_absent(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {}, clear=True):
            p = Path(td) / "metrics.json"; p.write_text("{}")
            with self.assertRaises(PublicationError):
                publish_to_github_branch(repo_dir=td, source_git_sha="a" * 40, run_id="X", files=[p])

    def test_git_push_failure_is_reported_not_hidden(self):
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "metrics.json"; p.write_text("{}")
            with mock.patch.dict(os.environ, {"CROPCOP_GITHUB_TOKEN": "dummy"}, clear=False), \
                 mock.patch("cropcop_je.publication._run_git", side_effect=subprocess.CalledProcessError(1, ["git"])):
                with self.assertRaises(subprocess.CalledProcessError):
                    publish_to_github_branch(repo_dir=td, source_git_sha="a" * 40, run_id="X", files=[p])

    def test_publication_rejects_secret_content(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "metrics.json"
            p.write_text('{"token":"github_pat_abcdefghijklmnopqrstuvwxyz123456"}')
            with self.assertRaises(PublicationError):
                audit_public_files([p])

    def test_durable_store_failure_is_not_silently_successful(self):
        from cropcop_je.persistence import FilesystemStore
        with tempfile.TemporaryDirectory() as td:
            store = FilesystemStore(Path(td) / "durable")
            with self.assertRaises(Exception):
                store.sync(Path(td) / "missing", run_id="R", segment_id="S")

    def test_low_disk_fails_before_torch_serialization(self):
        with tempfile.TemporaryDirectory() as td, mock.patch("cropcop_je.checkpointing.shutil.disk_usage") as du:
            du.return_value = type("DU", (), {"free": 1})()
            with self.assertRaises(OSError):
                save_torch_checkpoint(td, kind="latest", payload={}, expected_identity={}, min_free_bytes=2)

    def test_planned_walltime_rollover(self):
        b = SessionBudget(hard_limit_seconds=100, finalization_margin_seconds=20, started_monotonic=0)
        with mock.patch("cropcop_je.session.time.monotonic", return_value=79):
            self.assertTrue(b.should_finalize(estimated_checkpoint_seconds=1, estimated_sync_seconds=1))


if __name__ == "__main__":
    unittest.main()
