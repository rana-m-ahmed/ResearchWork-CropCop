import importlib.util
import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "journal_extension" / "src"))

from cropcop_je.envelope import (
    AMENDMENT_ID,
    AMENDMENT_SHA256,
    RunningChild,
    continuation_publication_repair_set,
    continuation_skip_set,
    finalize_manifest,
    gracefully_finalize_process_groups,
    locate_prior_bundle,
    planned_finalization_grace_seconds,
    validate_prior_envelope_bundle,
)
from cropcop_je.session import SessionBudget
from cropcop_je.terminal_recovery import completed_scientific_checkpoint_result

SOURCE = "a" * 40
G1 = "1" * 64
G2 = "2" * 64


class FakeProcess:
    def __init__(self, pid, clock, exit_after=None):
        self.pid = pid
        self.clock = clock
        self.exit_after = exit_after
        self.term_at = None
        self.killed = False

    def poll(self):
        if self.killed:
            return -9
        if self.term_at is not None and self.exit_after is not None:
            if self.clock[0] - self.term_at >= self.exit_after:
                return 0
        return None

    def wait(self, timeout=None):
        return self.poll()


def fake_child(process, child_id="A"):
    return RunningChild(
        child_id=child_id,
        experiment_id="E",
        run_id="R",
        slot=0,
        process=process,
        log_handle=None,
        log_path=Path("/tmp/fake.log"),
        started_monotonic=0.0,
        started_utc_epoch=0.0,
    )


def write_prior_bundle(root: Path, *, publication_status="PASS", corrupt_manifest=False):
    env = root / "envelope"
    artifact = env / "children/A/prior_result/run_record.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    common = {
        "schema_version": "1.0",
        "envelope_id": "E",
        "amendment_id": AMENDMENT_ID,
        "amendment_sha256": AMENDMENT_SHA256,
        "source_git_sha": SOURCE,
        "g1_seal_sha256": G1,
        "g2_barrier_sha256": G2,
    }
    manifest = finalize_manifest({
        **common,
        "children": [{"child_id": "A", "run_id": "run-a"}],
    })
    if corrupt_manifest:
        manifest["source_git_sha"] = "9" * 40
    (env / "ENVELOPE_MANIFEST.json").write_text(json.dumps(manifest))
    complete = publication_status == "PASS"
    result = {
        "status": "PASS",
        "continuation_required": False,
        "publication_status": publication_status,
        "publication_branch": "run-evidence/run-a" if complete else None,
        "publication_error": None if complete else "fixture",
        "evidence_chain_complete": complete,
        "result_relative_path": "children/A/prior_result/run_record.json",
    }
    evidence = {
        **common,
        "status": "PASS" if complete else "FAIL_TECHNICAL",
        "manifest_sha256": manifest["manifest_sha256"],
        "child_results": {"A": result},
    }
    (env / "ENVELOPE_EVIDENCE.json").write_text(json.dumps(evidence))
    state = {
        **common,
        "state": "PASS" if complete else "FAIL_TECHNICAL",
        "children": [{
            "child_id": "A",
            "run_id": "run-a",
            "status": "PASS",
            "execution_status": "PASS",
            "publication_status": publication_status,
            "evidence_chain_complete": complete,
        }],
    }
    (env / "ENVELOPE_STATE.json").write_text(json.dumps(state))
    return env


class QA1WaveBTests(unittest.TestCase):
    def test_01_planned_grace_is_materially_longer_than_emergency_30_seconds(self):
        budget = SessionBudget(
            hard_limit_seconds=43200,
            finalization_margin_seconds=3600,
            started_monotonic=0.0,
        )
        with mock.patch("cropcop_je.session.time.monotonic", return_value=100.0):
            grace = planned_finalization_grace_seconds(budget)
        self.assertGreater(grace, 30.0)
        self.assertLess(grace, budget.finalization_margin_seconds)

    def test_02_child_exits_after_35_seconds_without_sigkill(self):
        clock = [0.0]
        proc = FakeProcess(11, clock, exit_after=35.0)
        child = fake_child(proc)
        sent = []

        def killpg(pid, sig):
            sent.append(sig)
            if sig == signal.SIGTERM:
                proc.term_at = clock[0]
            elif sig == signal.SIGKILL:
                proc.killed = True

        with mock.patch("cropcop_je.envelope.time.monotonic", side_effect=lambda: clock[0]),              mock.patch("cropcop_je.envelope.time.sleep", side_effect=lambda s: clock.__setitem__(0, clock[0] + s)),              mock.patch("cropcop_je.envelope.os.killpg", side_effect=killpg):
            outcome = gracefully_finalize_process_groups([child], grace_seconds=60, poll_seconds=5)
        self.assertIn(signal.SIGTERM, sent)
        self.assertNotIn(signal.SIGKILL, sent)
        self.assertEqual(outcome["A"]["termination_mode"], "graceful_sigterm")
        self.assertGreaterEqual(outcome["A"]["termination_duration_seconds"], 35)

    def test_03_hung_child_is_killed_after_bounded_finalization_grace(self):
        clock = [0.0]
        proc = FakeProcess(12, clock, exit_after=None)
        child = fake_child(proc)
        sent = []

        def killpg(pid, sig):
            sent.append(sig)
            if sig == signal.SIGTERM:
                proc.term_at = clock[0]
            elif sig == signal.SIGKILL:
                proc.killed = True

        with mock.patch("cropcop_je.envelope.time.monotonic", side_effect=lambda: clock[0]),              mock.patch("cropcop_je.envelope.time.sleep", side_effect=lambda s: clock.__setitem__(0, clock[0] + s)),              mock.patch("cropcop_je.envelope.os.killpg", side_effect=killpg):
            outcome = gracefully_finalize_process_groups([child], grace_seconds=10, poll_seconds=5)
        self.assertIn(signal.SIGKILL, sent)
        self.assertEqual(outcome["A"]["termination_mode"], "sigkill_after_finalization_grace")

    def test_04_publication_failure_is_repair_not_skip(self):
        state = {"children": [{
            "child_id": "A",
            "status": "PASS",
            "execution_status": "PASS",
            "publication_status": "FAIL",
            "evidence_chain_complete": False,
        }]}
        self.assertEqual(continuation_skip_set(state), set())
        self.assertEqual(continuation_publication_repair_set(state), {"A"})

    def test_05_publication_complete_pass_is_skipped(self):
        state = {"children": [{
            "child_id": "A",
            "status": "PASS",
            "execution_status": "PASS",
            "publication_status": "PASS",
            "evidence_chain_complete": True,
        }]}
        self.assertEqual(continuation_skip_set(state), {"A"})
        self.assertEqual(continuation_publication_repair_set(state), set())

    def test_06_coherent_prior_bundle_is_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            write_prior_bundle(Path(td))
            bundle = locate_prior_bundle(td)
            errors = validate_prior_envelope_bundle(
                bundle,
                envelope_id="E",
                source_sha=SOURCE,
                g1_seal_sha256=G1,
                g2_barrier_sha256=G2,
                expected_run_ids={"A": "run-a"},
            )
            self.assertEqual(errors, [])

    def test_07_corrupt_prior_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            write_prior_bundle(Path(td), corrupt_manifest=True)
            bundle = locate_prior_bundle(td)
            errors = validate_prior_envelope_bundle(
                bundle,
                envelope_id="E",
                source_sha=SOURCE,
                g1_seal_sha256=G1,
                g2_barrier_sha256=G2,
                expected_run_ids={"A": "run-a"},
            )
            self.assertTrue(errors)

    def test_08_prior_publication_failure_is_coherent_but_not_complete(self):
        with tempfile.TemporaryDirectory() as td:
            write_prior_bundle(Path(td), publication_status="FAIL")
            bundle = locate_prior_bundle(td)
            errors = validate_prior_envelope_bundle(
                bundle,
                envelope_id="E",
                source_sha=SOURCE,
                g1_seal_sha256=G1,
                g2_barrier_sha256=G2,
                expected_run_ids={"A": "run-a"},
            )
            self.assertEqual(errors, [])
            self.assertEqual(continuation_publication_repair_set(bundle.state), {"A"})

    def test_09_publication_repair_helper_declares_no_training_relaunch(self):
        path = ROOT / "journal_extension/kaggle/run_envelope.py"
        spec = importlib.util.spec_from_file_location("qa1_wave_b_run_envelope", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with mock.patch.object(
            module,
            "_copy_prior_terminal_result",
            return_value=({"status": "PASS", "publication_status": "FAIL", "run_id": "run-a"}, [Path("/tmp/a")]),
        ), mock.patch.object(
            module,
            "_try_publish",
            return_value={"publication_status": "PASS", "publication_branch": "run-evidence/R", "publication_error": None},
        ):
            result = module._repair_prior_child_publication(
                object(),
                child={"child_id": "A", "experiment_id": "R04", "lane": "K1"},
                prior_result={},
                phase="principal-dual",
                envelope_root=Path("/tmp"),
                central_g2=Path("/tmp/g2"),
                source_sha=SOURCE,
                g1_sha=G1,
                g2_sha=G2,
                repair_publication=True,
            )
        self.assertTrue(result["publication_repair_only"])
        self.assertFalse(result["training_relaunched"])
        self.assertTrue(result["evidence_chain_complete"])

    def test_10_completed_epoch_checkpoint_recovers_with_zero_optimizer_steps(self):
        payload = {
            "epoch": 30,
            "batch_in_epoch": 0,
            "optimizer_step": 4321,
            "examples_seen": 76376 * 30,
            "data_order_state": {"epoch": 30, "next_batch_in_epoch": 0},
            "selection_state": {
                "history": [{"epoch": i + 1} for i in range(30)],
                "best": {
                    "epoch": 27,
                    "metrics": {
                        "validation_macro_f1": 0.9,
                        "validation_balanced_accuracy": 0.9,
                        "validation_nll": 0.2,
                    },
                    "checkpoint_sha256": "d" * 64,
                },
            },
        }
        recovery = {"candidate": "latest", "recovered": {"sha256": "e" * 64}}
        with mock.patch(
            "cropcop_je.terminal_recovery.recover_latest",
            return_value=(Path("/tmp/latest"), payload, recovery),
        ), mock.patch("cropcop_je.terminal_recovery.verify_selected") as selected:
            result = completed_scientific_checkpoint_result(
                "/tmp/checkpoints",
                expected_identity={"x": 1},
                locked_epochs=30,
            )
        selected.assert_called_once()
        self.assertTrue(result["terminal_checkpoint_recovery"])
        self.assertEqual(result["optimizer_steps_segment"], 0)
        self.assertEqual(result["optimizer_step_total"], 4321)
        self.assertEqual(result["recovery_events"][-1]["optimizer_steps_advanced"], 0)

    def test_11_partial_final_epoch_is_not_promoted_to_terminal(self):
        payload = {
            "epoch": 30,
            "batch_in_epoch": 1,
            "optimizer_step": 4321,
            "data_order_state": {"epoch": 30, "next_batch_in_epoch": 1},
            "selection_state": {"history": [{"epoch": i + 1} for i in range(30)], "best": {}},
        }
        recovery = {"candidate": "latest", "recovered": {"sha256": "e" * 64}}
        with mock.patch(
            "cropcop_je.terminal_recovery.recover_latest",
            return_value=(Path("/tmp/latest"), payload, recovery),
        ):
            self.assertIsNone(completed_scientific_checkpoint_result(
                "/tmp/checkpoints",
                expected_identity={"x": 1},
                locked_epochs=30,
            ))

    def test_12_execution_wrapper_checks_terminal_recovery_before_training_loop(self):
        source = (ROOT / "journal_extension/scripts/run_training.py").read_text()
        self.assertLess(
            source.index("completed_scientific_checkpoint_result("),
            source.index("result = run_training("),
        )

    def test_13_global_stop_uses_graceful_finalization_and_then_result_collection(self):
        source = (ROOT / "journal_extension/kaggle/run_envelope.py").read_text()
        self.assertIn("gracefully_finalize_process_groups(", source)
        self.assertIn("_result_for_child(", source)
        self.assertNotIn('if global_stop["reason"]:\n            for obj in list(running.values()):\n                terminate_process_group(obj)\n            break', source)


if __name__ == "__main__":
    unittest.main()
