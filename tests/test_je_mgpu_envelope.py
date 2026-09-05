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

from cropcop_je.envelope import (
    AMENDMENT_ID,
    AMENDMENT_SHA256,
    child_environment,
    continuation_skip_set,
    finalize_manifest,
    manifest_self_hash,
    resolve_run_id,
    validate_continuation_state,
    validate_disjoint_mutable_roots,
    validate_envelope_config,
    validate_manifest,
    validate_t4x2_inventory,
)
from cropcop_je.publication import PublicationError, audit_public_files
from cropcop_je.persistence import validate_durable_access_plan
from cropcop_je.science_diff import validate_science_diff
from cropcop_je.smoke_handoff import (
    SmokeHandoffError,
    require_qualifying_kaggle_batch,
    validate_terminal_smoke_b_evidence,
)

SOURCE = "a" * 40
DEP = "b" * 64


def valid_smoke_b():
    return {
        "schema_version": "2.0",
        "status": "PASS",
        "mode": "RESTORE",
        "scientific": False,
        "synthetic_unprotected_data_only": True,
        "source_git_sha": SOURCE,
        "dependency_lock_sha256": DEP,
        "kaggle_run_type": "Batch",
        "restore_success": True,
        "recover_success": True,
        "resume_success": True,
        "sequence": [
            "READ_A", "VERIFY_A", "RESTORE_A", "RECOVER_A",
            "LOAD_A", "RESUME", "CHECKPOINT_B",
        ],
        "smoke_a_expected_checkpoint_sha256": "c" * 64,
        "smoke_a_observed_restored_checkpoint_sha256": "c" * 64,
        "restored_optimizer_step": 3,
        "resumed_optimizer_step": 4,
        "attached_smoke_a_input_unchanged": True,
        "g1_executed": False,
        "g2_executed": False,
        "r04_r05_executed": False,
        "restricted_cropcop_data_accessed": False,
        "git_publication_status": "PASS",
        "public_safe_evidence_branch": "run-evidence/SMOKE-B-fixture",
        "smoke_a_manifest_sha256": "d" * 64,
        "smoke_a_evidence_sha256": "e" * 64,
    }


class MGPUSmokeContractTests(unittest.TestCase):
    def test_01_correct_terminal_smoke_b_is_accepted_without_obsolete_secret_field(self):
        evidence = valid_smoke_b()
        self.assertNotIn("secret_retrieval_proved_without_value_disclosure", evidence)
        self.assertEqual(
            validate_terminal_smoke_b_evidence(
                evidence,
                expected_source_sha=SOURCE,
                expected_dependency_lock_sha256=DEP,
            ),
            [],
        )

    def test_02_interactive_terminal_smoke_is_rejected(self):
        evidence = valid_smoke_b()
        evidence["kaggle_run_type"] = "Interactive"
        errors = validate_terminal_smoke_b_evidence(
            evidence, expected_source_sha=SOURCE, expected_dependency_lock_sha256=DEP
        )
        self.assertTrue(any("Batch" in e for e in errors))

    def test_03_wrong_restore_sequence_is_rejected(self):
        evidence = valid_smoke_b()
        evidence["sequence"] = evidence["sequence"][:-1]
        self.assertTrue(
            validate_terminal_smoke_b_evidence(
                evidence, expected_source_sha=SOURCE, expected_dependency_lock_sha256=DEP
            )
        )

    def test_04_wrong_restored_checkpoint_sha_is_rejected(self):
        evidence = valid_smoke_b()
        evidence["smoke_a_observed_restored_checkpoint_sha256"] = "f" * 64
        errors = validate_terminal_smoke_b_evidence(
            evidence, expected_source_sha=SOURCE, expected_dependency_lock_sha256=DEP
        )
        self.assertTrue(any("checkpoint SHA differs" in e for e in errors))

    def test_05_missing_recover_is_rejected(self):
        evidence = valid_smoke_b()
        evidence["recover_success"] = False
        self.assertTrue(
            validate_terminal_smoke_b_evidence(
                evidence, expected_source_sha=SOURCE, expected_dependency_lock_sha256=DEP
            )
        )

    def test_06_nonadvancing_resume_is_rejected(self):
        evidence = valid_smoke_b()
        evidence["resumed_optimizer_step"] = 3
        errors = validate_terminal_smoke_b_evidence(
            evidence, expected_source_sha=SOURCE, expected_dependency_lock_sha256=DEP
        )
        self.assertTrue(any("did not advance" in e for e in errors))

    def test_07_attached_input_mutation_is_rejected(self):
        evidence = valid_smoke_b()
        evidence["attached_smoke_a_input_unchanged"] = False
        self.assertTrue(
            validate_terminal_smoke_b_evidence(
                evidence, expected_source_sha=SOURCE, expected_dependency_lock_sha256=DEP
            )
        )

    def test_08_dependency_lock_drift_is_rejected(self):
        errors = validate_terminal_smoke_b_evidence(
            valid_smoke_b(), expected_source_sha=SOURCE, expected_dependency_lock_sha256="9" * 64
        )
        self.assertTrue(any("dependency-lock" in e for e in errors))

    def test_09_source_drift_is_rejected(self):
        errors = validate_terminal_smoke_b_evidence(
            valid_smoke_b(), expected_source_sha="9" * 40, expected_dependency_lock_sha256=DEP
        )
        self.assertTrue(any("source Git SHA" in e for e in errors))

    def test_10_missing_publication_pass_is_rejected(self):
        evidence = valid_smoke_b()
        evidence["git_publication_status"] = "FAIL"
        self.assertTrue(
            validate_terminal_smoke_b_evidence(
                evidence, expected_source_sha=SOURCE, expected_dependency_lock_sha256=DEP
            )
        )

    def test_11_batch_helper_rejects_interactive(self):
        with self.assertRaises(SmokeHandoffError):
            require_qualifying_kaggle_batch(context="test", run_type="Interactive")

    def test_12_batch_helper_accepts_batch(self):
        self.assertEqual(require_qualifying_kaggle_batch(context="test", run_type="Batch"), "Batch")


class MGPUAuthorityAndScienceTests(unittest.TestCase):
    def test_13_science_diff_is_clean_on_refactor_tree(self):
        report = validate_science_diff(ROOT)
        self.assertEqual(report["status"], "PASS", report["errors"])
        self.assertTrue(report["protected_surfaces_unchanged"])

    def test_14_stage03r_and_stage04_base_remain_unchanged(self):
        report = validate_science_diff(ROOT)
        self.assertTrue(report["stage03r_authority_unchanged"])
        self.assertTrue(report["stage04_base_unchanged"])

    def test_15_amendment_is_bound_by_exact_sha256(self):
        report = validate_science_diff(ROOT)
        self.assertEqual(report["stage04_execution_amendment_sha256"], AMENDMENT_SHA256)

    def test_16_amendment_does_not_replace_scientific_authority(self):
        authority = json.loads((ROOT / "journal_extension/locks/scientific_authority.json").read_text())
        self.assertEqual(authority["authority_id"], "EAAI-JE-SDL-v2.1-QA")
        self.assertEqual(authority["stage04_execution_amendment_id"], AMENDMENT_ID)
        self.assertEqual(authority["stage04_execution_amendment_sha256"], AMENDMENT_SHA256)

    def test_17_train_data_models_and_configs_are_blob_frozen(self):
        sentinel = json.loads((ROOT / "01A_MGPU_01_SCIENCE_DIFF_SENTINEL.json").read_text())
        for path, expected in sentinel["protected_git_blobs"].items():
            actual = subprocess.check_output(["git", "hash-object", "--", path], cwd=ROOT, text=True).strip()
            self.assertEqual(actual, expected, path)

    def test_18_no_ddp_or_dataparallel_in_principal_child_path(self):
        train = (ROOT / "journal_extension/src/cropcop_je/train.py").read_text()
        runner = (ROOT / "journal_extension/kaggle/run_lane.py").read_text()
        for token in ("DistributedDataParallel", "nn.DataParallel", "torch.distributed.run", "SyncBatchNorm"):
            self.assertNotIn(token, train)
            self.assertNotIn(token, runner)

    def test_19_physical_gpu_slot_is_not_checkpoint_scientific_identity(self):
        train = (ROOT / "journal_extension/src/cropcop_je/train.py").read_text()
        identity = train[train.index("def _identity"):train.index("def _checkpoint_payload")]
        self.assertNotIn("physical_gpu", identity)
        self.assertNotIn("envelope_id", identity)


class MGPUEnvelopeConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env_dir = ROOT / "journal_extension/kaggle/envelopes"

    def _load(self, name):
        return json.loads((self.env_dir / name).read_text())

    def test_20_g2_envelope_is_predeclared_and_valid(self):
        cfg = self._load("G2_DUAL_T4.json")
        self.assertEqual(validate_envelope_config(cfg), [])
        self.assertEqual([x["experiment_id"] for x in cfg["children"]],
                         ["CAL-MNV4-DIRECT", "CAL-MNV4-TEACHER", "CAL-CNXTT"])

    def test_21_g2_first_two_start_on_distinct_fixed_slots(self):
        cfg = self._load("G2_DUAL_T4.json")
        self.assertEqual([x["slot"] for x in cfg["children"]], [0, 1, "first_free"])

    def test_22_p1_allows_only_s1_direct_teacher_pair(self):
        cfg = self._load("P1_S1_PAIR.json")
        self.assertEqual(validate_envelope_config(cfg), [])
        self.assertEqual({x["experiment_id"] for x in cfg["children"]},
                         {"R04-MNV4-DIRECT-S1", "R05-MNV4-TEACHER-S1"})

    def test_23_p2_allows_only_s2_direct_teacher_pair(self):
        cfg = self._load("P2_S2_PAIR.json")
        self.assertEqual(validate_envelope_config(cfg), [])
        self.assertEqual({x["experiment_id"] for x in cfg["children"]},
                         {"R04-MNV4-DIRECT-S2", "R05-MNV4-TEACHER-S2"})

    def test_24_p3_allows_only_s3_direct_teacher_pair(self):
        cfg = self._load("P3_S3_PAIR.json")
        self.assertEqual(validate_envelope_config(cfg), [])
        self.assertEqual({x["experiment_id"] for x in cfg["children"]},
                         {"R04-MNV4-DIRECT-S3", "R05-MNV4-TEACHER-S3"})

    def test_25_principal_pair_slot_mutation_is_rejected(self):
        cfg = self._load("P1_S1_PAIR.json")
        cfg["children"][1]["slot"] = 0
        self.assertTrue(validate_envelope_config(cfg))

    def test_26_one_gpu_host_is_rejected_for_t4x2(self):
        self.assertTrue(validate_t4x2_inventory([{"index": 0, "uuid": "a", "name": "Tesla T4"}]))

    def test_27_heterogeneous_host_is_rejected(self):
        inv = [
            {"index": 0, "uuid": "a", "name": "Tesla T4"},
            {"index": 1, "uuid": "b", "name": "A100"},
        ]
        self.assertTrue(validate_t4x2_inventory(inv))

    def test_28_distinct_t4_host_is_accepted(self):
        inv = [
            {"index": 0, "uuid": "a", "name": "Tesla T4"},
            {"index": 1, "uuid": "b", "name": "Tesla T4"},
        ]
        self.assertEqual(validate_t4x2_inventory(inv), [])


class MGPUIsolationTests(unittest.TestCase):
    def test_29_child_a_is_pinned_to_gpu0_and_sees_execution_metadata(self):
        env = child_environment(
            base_env={"CROPCOP_GITHUB_TOKEN": "secret", "GITHUB_TOKEN": "secret2"},
            envelope_id="E", physical_slot=0,
            output_root="/tmp/a/out", g2_summaries_root="/tmp/a/g2",
            terminal_root="/tmp/a/terminal", workers=2,
        )
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(env["CROPCOP_PHYSICAL_GPU_SLOT"], "0")
        self.assertEqual(env["CROPCOP_NUM_WORKERS_PER_CHILD"], "2")

    def test_30_child_b_is_pinned_to_gpu1(self):
        env = child_environment(
            base_env={}, envelope_id="E", physical_slot=1,
            output_root="/tmp/b/out", g2_summaries_root="/tmp/b/g2",
            terminal_root="/tmp/b/terminal", workers=2,
        )
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")

    def test_31_child_git_tokens_are_stripped(self):
        env = child_environment(
            base_env={"CROPCOP_GITHUB_TOKEN": "secret", "GITHUB_TOKEN": "secret2"},
            envelope_id="E", physical_slot=0,
            output_root="/tmp/a/out", g2_summaries_root="/tmp/a/g2",
            terminal_root="/tmp/a/terminal", workers=2,
        )
        self.assertNotIn("CROPCOP_GITHUB_TOKEN", env)
        self.assertNotIn("GITHUB_TOKEN", env)

    def test_32_thread_bounds_are_set(self):
        env = child_environment(
            base_env={}, envelope_id="E", physical_slot=0,
            output_root="/tmp/a/out", g2_summaries_root="/tmp/a/g2",
            terminal_root="/tmp/a/terminal", workers=2,
        )
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.assertEqual(env[key], "1")
        self.assertEqual(env["TOKENIZERS_PARALLELISM"], "false")

    def test_33_child_mutable_roots_must_be_disjoint(self):
        rows = [
            {"child_id": "A", "output_root": "/tmp/a/out", "g2_summaries_root": "/tmp/a/g2", "terminal_root": "/tmp/a/t"},
            {"child_id": "B", "output_root": "/tmp/b/out", "g2_summaries_root": "/tmp/b/g2", "terminal_root": "/tmp/b/t"},
        ]
        self.assertEqual(validate_disjoint_mutable_roots(rows), [])

    def test_34_cross_child_mutable_root_collision_is_rejected(self):
        rows = [
            {"child_id": "A", "output_root": "/tmp/shared", "g2_summaries_root": "/tmp/a/g2", "terminal_root": "/tmp/a/t"},
            {"child_id": "B", "output_root": "/tmp/shared/b", "g2_summaries_root": "/tmp/b/g2", "terminal_root": "/tmp/b/t"},
        ]
        self.assertTrue(validate_disjoint_mutable_roots(rows))

    def test_35_parent_core_has_no_torch_import(self):
        source = (ROOT / "journal_extension/src/cropcop_je/envelope.py").read_text()
        self.assertNotIn("import torch", source)

    def test_36_supervisor_uses_new_process_sessions(self):
        source = (ROOT / "journal_extension/src/cropcop_je/envelope.py").read_text()
        self.assertIn("start_new_session=True", source)

    def test_37_supervisor_has_sigterm_then_sigkill_fallback(self):
        source = (ROOT / "journal_extension/src/cropcop_je/envelope.py").read_text()
        self.assertIn("signal.SIGTERM", source)
        self.assertIn("signal.SIGKILL", source)

    def test_38_common_notebook_clock_is_not_reestablished_by_children(self):
        source = (ROOT / "journal_extension/src/cropcop_je/envelope.py").read_text()
        self.assertNotIn("establish_global_clock", source)
        self.assertIn("CROPCOP_NOTEBOOK_STARTED_MONOTONIC", (ROOT / "journal_extension/kaggle/run_envelope.py").read_text())

    def test_39_run_lane_enforces_exactly_one_visible_cuda_in_dual_mode(self):
        source = (ROOT / "journal_extension/kaggle/run_lane.py").read_text()
        self.assertIn("torch.cuda.device_count() != 1", source)

    def test_40_run_lane_plumbs_worker_count(self):
        source = (ROOT / "journal_extension/kaggle/run_lane.py").read_text()
        self.assertIn('CROPCOP_NUM_WORKERS_PER_CHILD', source)
        self.assertIn('"--num-workers", str(workers)', source)


class MGPUContinuationAndEvidenceTests(unittest.TestCase):
    def test_41_manifest_self_hash_round_trip(self):
        manifest = finalize_manifest({
            "schema_version": "1.0", "amendment_id": AMENDMENT_ID,
            "amendment_sha256": AMENDMENT_SHA256,
        })
        self.assertEqual(manifest["manifest_sha256"], manifest_self_hash(manifest))
        self.assertEqual(validate_manifest(manifest), [])

    def test_42_manifest_mutation_is_detected(self):
        manifest = finalize_manifest({
            "schema_version": "1.0", "amendment_id": AMENDMENT_ID,
            "amendment_sha256": AMENDMENT_SHA256,
        })
        manifest["schema_version"] = "2.0"
        self.assertTrue(validate_manifest(manifest))

    def test_43_terminal_prior_child_is_skipped(self):
        state = {"children": [
            {"child_id": "A", "status": "PASS", "execution_status": "PASS", "publication_status": "PASS", "evidence_chain_complete": True},
            {"child_id": "B", "status": "CONTINUATION_REQUIRED", "execution_status": "CONTINUATION_REQUIRED", "publication_status": "PASS", "evidence_chain_complete": False},
        ]}
        self.assertEqual(continuation_skip_set(state), {"A"})

    def test_44_continuation_source_drift_is_rejected(self):
        expected = {"A": "run-a", "B": "run-b"}
        state = {
            "envelope_id": "E", "amendment_id": AMENDMENT_ID,
            "amendment_sha256": AMENDMENT_SHA256, "source_git_sha": "9" * 40,
            "g1_seal_sha256": "1" * 64, "g2_barrier_sha256": "2" * 64,
            "children": [{"child_id": "A", "run_id": "run-a"}, {"child_id": "B", "run_id": "run-b"}],
        }
        errors = validate_continuation_state(
            state, envelope_id="E", source_sha=SOURCE,
            g1_seal_sha256="1" * 64, g2_barrier_sha256="2" * 64,
            expected_run_ids=expected,
        )
        self.assertTrue(any("source_git_sha" in e for e in errors))

    def test_45_continuation_run_id_drift_is_rejected(self):
        state = {
            "envelope_id": "E", "amendment_id": AMENDMENT_ID,
            "amendment_sha256": AMENDMENT_SHA256, "source_git_sha": SOURCE,
            "g1_seal_sha256": "1" * 64, "g2_barrier_sha256": "2" * 64,
            "children": [{"child_id": "A", "run_id": "wrong"}],
        }
        errors = validate_continuation_state(
            state, envelope_id="E", source_sha=SOURCE,
            g1_seal_sha256="1" * 64, g2_barrier_sha256="2" * 64,
            expected_run_ids={"A": "run-a"},
        )
        self.assertTrue(any("run-ID" in e for e in errors))

    def test_46_run_id_is_deterministic_and_source_bound(self):
        self.assertEqual(
            resolve_run_id("R04-MNV4-DIRECT-S1", SOURCE, {}),
            "JE-R04-MNV4-DIRECT-S1-aaaaaaaaaaaa-A01",
        )

    def test_47_forbidden_checkpoint_extension_cannot_be_public_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "model.pt"
            p.write_bytes(b"x")
            with self.assertRaises(PublicationError):
                audit_public_files([p])

    def test_48_envelope_publication_is_parent_only_by_static_contract(self):
        source = (ROOT / "journal_extension/kaggle/run_envelope.py").read_text()
        self.assertIn("publish_to_github_branch", source)
        self.assertIn("git_credentials_present_in_child", source)

    def test_49_dual_smoke_is_synthetic_and_has_no_cropcop_training_call(self):
        source = (ROOT / "journal_extension/scripts/smoke_dual_gpu.py").read_text()
        self.assertIn('"scientific": False', source)
        self.assertIn('"restricted_cropcop_data_accessed": False', source)
        self.assertNotIn("run_training(", source)

    def test_50_dual_smoke_requires_distinct_physical_gpu_uuids(self):
        source = (ROOT / "journal_extension/scripts/smoke_dual_gpu.py").read_text()
        self.assertIn('inventory[0]["uuid"] == inventory[1]["uuid"]', source)

    def test_51_g2_parent_validates_each_summary_before_central_copy(self):
        source = (ROOT / "journal_extension/kaggle/run_envelope.py").read_text()
        self.assertLess(source.index("validate_calibration_summary(summary)"), source.index("shutil.copy2(summary_path, tmp)"))

    def test_52_principal_preflight_rebuilds_common_g2_barrier(self):
        source = (ROOT / "journal_extension/kaggle/run_envelope.py").read_text()
        self.assertIn("_central_g2_barrier", source)
        self.assertIn("validate_g2_barrier_object", source)

    def test_53_child_failure_does_not_immediately_kill_sibling(self):
        source = (ROOT / "journal_extension/kaggle/run_envelope.py").read_text()
        terminal_block = source[source.index("for child_id, obj in list(running.items())"):source.index("if not any_terminal:")]
        self.assertNotIn("for sibling", terminal_block)
        self.assertNotIn("terminate_process_group(obj)", terminal_block.split("rc = obj.process.poll()")[1])

    def test_54_host_global_stop_gracefully_finalizes_running_children(self):
        source = (ROOT / "journal_extension/kaggle/run_envelope.py").read_text()
        self.assertIn('if global_stop["reason"] and not finalization_started:', source)
        self.assertIn("gracefully_finalize_process_groups(", source)

    def test_55_third_g2_child_is_first_free_not_metric_driven(self):
        cfg = json.loads((ROOT / "journal_extension/kaggle/envelopes/G2_DUAL_T4.json").read_text())
        self.assertEqual(cfg["children"][2]["slot"], "first_free")
        source = (ROOT / "journal_extension/kaggle/run_envelope.py").read_text()
        self.assertNotIn("validation_macro_f1", source)
        self.assertNotIn("selected_metrics", source)

    def test_56_bootstrap_has_explicit_new_phase_vocabulary(self):
        source = (ROOT / "journal_extension/kaggle/bootstrap_clean_session.py").read_text()
        for phase in ("dual-gpu-smoke", "calibration-dual", "principal-dual"):
            self.assertIn(phase, source)

    def test_57_g1_requires_batch_execution(self):
        source = (ROOT / "journal_extension/kaggle/run_g1.py").read_text()
        self.assertIn("require_qualifying_kaggle_batch", source)

    def test_58_smoke_producer_requires_batch_execution(self):
        source = (ROOT / "journal_extension/scripts/smoke_infrastructure.py").read_text()
        self.assertIn("require_qualifying_kaggle_batch", source)

    def test_59_obsolete_secret_retrieval_predicate_is_removed_from_lane(self):
        source = (ROOT / "journal_extension/kaggle/run_lane.py").read_text()
        self.assertNotIn("secret_retrieval_proved_without_value_disclosure", source)

    def test_60_no_real_science_evidence_is_committed_by_refactor(self):
        evidence_root = ROOT / "journal_extension/evidence/public/runs"
        if evidence_root.exists():
            for p in evidence_root.rglob("*"):
                if p.is_file():
                    self.assertNotIn("MGPU-P1", p.name)
                    self.assertNotIn("MGPU-P2", p.name)
                    self.assertNotIn("MGPU-P3", p.name)


    def test_61_filesystem_durable_preflight_exercises_write_read_delete(self):
        with tempfile.TemporaryDirectory() as td:
            resolved = {"run-a": str(Path(td) / "run-a"), "run-b": str(Path(td) / "run-b")}
            report = validate_durable_access_plan("filesystem", resolved, env={})
            self.assertEqual(report["status"], "PASS", report["errors"])
            self.assertEqual(report["resolved_locator_count"], 2)

    def test_62_kaggle_durable_owner_mismatch_fails_before_cli(self):
        resolved = {"run-a": "different-owner/run-a"}
        report = validate_durable_access_plan(
            "kaggle-dataset",
            resolved,
            env={"KAGGLE_USERNAME": "expected-owner", "KAGGLE_KEY": "fixture"},
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("owner mismatch" in e for e in report["errors"]))

    def test_63_gpu_telemetry_records_compute_process_mapping_contract(self):
        source = (ROOT / "journal_extension/src/cropcop_je/envelope.py").read_text()
        self.assertIn("--query-compute-apps=pid,gpu_uuid,process_name,used_memory", source)
        self.assertIn('"compute_processes"', source)

    def test_64_dual_child_writes_observed_preflight_evidence(self):
        source = (ROOT / "journal_extension/kaggle/run_lane.py").read_text()
        self.assertIn('"observed_visible_cuda_count": torch.cuda.device_count()', source)
        self.assertIn('"observed_visible_gpu_name": torch.cuda.get_device_name(0)', source)
        self.assertIn('"git_credentials_present": bool(', source)
        self.assertIn('"CHILD_PREFLIGHT.json"', source)

    def test_65_parent_validates_child_preflight_before_result_collection(self):
        source = (ROOT / "journal_extension/kaggle/run_envelope.py").read_text()
        terminal = source[source.index("if rc == 0:"):source.index("else:", source.index("if rc == 0:"))]
        self.assertLess(terminal.index("_validated_child_preflight"), terminal.index("_result_for_child"))

    def test_66_publication_failure_is_separate_from_execution_status(self):
        source = (ROOT / "journal_extension/kaggle/run_envelope.py").read_text()
        self.assertIn('"execution_status"', source)
        self.assertIn('"publication_status"', source)
        self.assertIn('"evidence_chain_complete"', source)
        self.assertIn("continuation_publication_repair_set", source)



if __name__ == "__main__":
    unittest.main()
