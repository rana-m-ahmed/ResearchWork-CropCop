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
from cropcop_je.checkpointing import recover_latest
from cropcop_je.g2 import build_g2_barrier
from cropcop_je.hashing import require_sha256
from cropcop_je.runlog import assert_resume_identity, claim_run_directory
from cropcop_je.selection import SelectionState
from cropcop_je.segments import append_segment_event
from cropcop_je.session import SessionBudget


class HardeningCoreTests(unittest.TestCase):
    def test_atomic_json_preserves_old_file_if_replace_fails(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            atomic_write_json(p, {"v": 1})
            with mock.patch("cropcop_je.atomic_io.os.replace", side_effect=OSError("injected")):
                with self.assertRaises(OSError):
                    atomic_write_json(p, {"v": 2})
            self.assertEqual(json.loads(p.read_text()), {"v": 1})

    def test_resume_selection_fixture_preserves_pre_interrupt_best(self):
        fixture = [
            {"epoch": 1, "validation_macro_f1": .70, "validation_balanced_accuracy": .70, "validation_nll": 1.0},
            {"epoch": 2, "validation_macro_f1": .81, "validation_balanced_accuracy": .79, "validation_nll": .8},
            {"epoch": 3, "validation_macro_f1": .79, "validation_balanced_accuracy": .80, "validation_nll": .7},
            {"epoch": 4, "validation_macro_f1": .78, "validation_balanced_accuracy": .81, "validation_nll": .6},
        ]
        uninterrupted = SelectionState()
        for row in fixture:
            if uninterrupted.consider(row):
                uninterrupted.bind_selected_checkpoint(sha256=f"epoch{row['epoch']}", relative_path=f"selected-{row['epoch']}")
        interrupted = SelectionState()
        for row in fixture[:2]:
            if interrupted.consider(row):
                interrupted.bind_selected_checkpoint(sha256=f"epoch{row['epoch']}", relative_path=f"selected-{row['epoch']}")
        resumed = SelectionState.from_dict(json.loads(json.dumps(interrupted.to_dict())))
        for row in fixture[2:]:
            if resumed.consider(row):
                resumed.bind_selected_checkpoint(sha256=f"epoch{row['epoch']}", relative_path=f"selected-{row['epoch']}")
        self.assertEqual(resumed.best, uninterrupted.best)
        self.assertEqual(resumed.history, uninterrupted.history)
        self.assertEqual(resumed.best["epoch"], 2)

    def test_session_budget_rolls_before_hard_limit(self):
        b = SessionBudget(hard_limit_seconds=100, finalization_margin_seconds=20, started_monotonic=0)
        with mock.patch("cropcop_je.session.time.monotonic", return_value=75):
            self.assertTrue(b.should_finalize(estimated_checkpoint_seconds=3, estimated_sync_seconds=3))

    def test_segment_ledger_is_append_only_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "segments.jsonl"
            append_segment_event(p, {"segment_id": "a", "state": "START"})
            append_segment_event(p, {"segment_id": "a", "state": "END"})
            rows = [json.loads(x) for x in p.read_text().splitlines()]
            self.assertEqual([r["state"] for r in rows], ["START", "END"])

    def test_corrupted_newest_falls_back_to_previous(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            atomic_write_json(root / "checkpoint_index.json", {
                "schema_version": "2.0", "generation": 2,
                "latest": {"relative_path": "objects/latest.ckpt", "sha256": "a", "bytes": 1, "generation": 2},
                "previous": {"relative_path": "objects/previous.ckpt", "sha256": "b", "bytes": 1, "generation": 1},
                "selected": None,
            })
            def fake_verify(_root, ref, expected_identity):
                if ref["generation"] == 2:
                    raise RuntimeError("injected corruption")
                return root / "objects/previous.ckpt", {"identity": expected_identity}
            with mock.patch("cropcop_je.checkpointing._verify_ref", side_effect=fake_verify):
                _p, _payload, event = recover_latest(root, expected_identity={"x": 1})
            self.assertEqual(event["candidate"], "previous")
            self.assertTrue(event["prior_candidate_errors"])

    def test_resume_rejects_changed_source_commit_and_lane(self):
        base = {
            "experiment_id": "R04", "authority_id": "lock", "source_git_commit": "a" * 40,
            "config_sha256": "b" * 64, "ctc_v2_sha256": "c" * 64, "manifest_sha256": "d" * 64,
            "class_map_sha256": "e" * 64, "seed": 1, "student_init_sha256": "f" * 64,
            "pretrained_sha256": "0" * 64, "teacher_sha256": None, "teacher_factory_sha256": None,
            "software_stack_sha256": "1" * 64, "lane_id": "K1",
        }
        assert_resume_identity(base, dict(base))
        changed = dict(base); changed["source_git_commit"] = "2" * 40
        with self.assertRaises(ValueError):
            assert_resume_identity(base, changed)
        changed = dict(base); changed["lane_id"] = "K2"
        with self.assertRaises(ValueError):
            assert_resume_identity(base, changed)

    def test_output_collision_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            claim_run_directory(td, run_id="A", experiment_id="R04", lane_id="K1")
            claim_run_directory(td, run_id="A", experiment_id="R04", lane_id="K1")
            with self.assertRaises(RuntimeError):
                claim_run_directory(td, run_id="B", experiment_id="R05", lane_id="K1")

    def test_hash_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "artifact.bin"
            p.write_bytes(b"abc")
            with self.assertRaises(ValueError):
                require_sha256(p, "0" * 64, "artifact")

    def test_g2_barrier_rejects_missing_lane_and_stack_drift(self):
        def summary(cid, stack):
            return {
                "calibration_id": cid, "status": "PASS", "resume_success": True,
                "calibration_weights_scientific": False, "source_git_commit": "a" * 40,
                "software_stack_sha256": stack,
                "measured": {
                    "sec_per_optimizer_step": 1.0, "examples_per_second": 10.0,
                    "dataloader_wait_seconds": 1.0, "dataloader_examples_per_wait_second": 100.0,
                    "peak_gpu_memory_bytes": 1, "checkpoint_save_seconds": 1.0,
                    "checkpoint_load_seconds": 1.0,
                    "validation_forward_benchmark": {"end_to_end_examples_per_second": 100.0},
                },
            }
        missing = build_g2_barrier([summary("CAL-MNV4-DIRECT", "x"), summary("CAL-MNV4-TEACHER", "x")])
        self.assertEqual(missing["status"], "FAIL")
        drift = build_g2_barrier([
            summary("CAL-MNV4-DIRECT", "x"), summary("CAL-MNV4-TEACHER", "y"), summary("CAL-CNXTT", "x")
        ])
        self.assertEqual(drift["status"], "FAIL")

    def test_persistent_worker_epoch_is_carried_in_sampler_key(self):
        train_source = (ROOT / "journal_extension/src/cropcop_je/train.py").read_text()
        data_source = (ROOT / "journal_extension/src/cropcop_je/data.py").read_text()
        self.assertIn("return iter((index, self.epoch) for index in self.order)", train_source)
        self.assertIn("if isinstance(index,tuple)", data_source)
        self.assertIn("row_augmentation_seed(self.training_seed,int(epoch),row.stable_row_id)", data_source)

    def test_teacher_projection_probe_restores_mode_and_isolates_rng(self):
        source = (ROOT / "journal_extension/src/cropcop_je/models.py").read_text()
        for token in (
            "student.eval()",
            "teacher.eval()",
            "student.train(student_was_training)",
            "teacher.train(teacher_was_training)",
            "with torch.no_grad():",
            "with torch.random.fork_rng(devices=[]):",
        ):
            self.assertIn(token, source)

    def test_one_canonical_notebook_and_three_small_lane_configs(self):
        kaggle = ROOT / "journal_extension" / "kaggle"
        notebooks = sorted(
            p for p in kaggle.glob("*.ipynb")
            if not p.name.startswith(("trackb_", "TrackB_"))
        )
        self.assertEqual([p.name for p in notebooks], ["canonical_lane.ipynb"])
        notebook = json.loads(notebooks[0].read_text())
        self.assertEqual(notebook.get("nbformat"), 4)
        code_cells = [c for c in notebook.get("cells", []) if c.get("cell_type") == "code"]
        self.assertEqual(len(code_cells), 1)
        code = "".join(code_cells[0].get("source", []))
        self.assertIn("journal_extension/kaggle/run_envelope.py", code)
        self.assertIn("journal_extension/scripts/smoke_dual_gpu.py", code)
        self.assertNotIn("journal_extension/kaggle/run_lane.py", code)
        expected = {"K1", "K2", "K3"}
        observed = set()
        experiment_ids = set()
        for p in sorted((kaggle / "lanes").glob("K*.json")):
            data = json.loads(p.read_text())
            observed.add(data["lane_id"])
            for item in data["principal"]:
                self.assertNotIn("run_id", item)
                self.assertNotIn(item["experiment_id"], experiment_ids)
                experiment_ids.add(item["experiment_id"])
        self.assertEqual(observed, expected)
        runner = (kaggle / "run_lane.py").read_text()
        self.assertIn("def resolve_run_id", runner)
        self.assertIn("source_sha[:12]", runner)


if __name__ == "__main__":
    unittest.main()
