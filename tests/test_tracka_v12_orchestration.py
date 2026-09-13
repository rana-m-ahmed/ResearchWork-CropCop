from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
KAGGLE = ROOT / "journal_extension" / "kaggle"
for path in (SRC, KAGGLE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cropcop_je.tracka_v12 import EXPERIMENT_SPECS  # noqa: E402
from cropcop_je.tracka_v12_g1a import R13_PRETRAINED_SHA256, TEACHER_SHA256  # noqa: E402
from cropcop_je.tracka_v12_g2a import PROFILE_COVERAGE, REQUIRED_PROFILES, SLOT_ORDER  # noqa: E402
from cropcop_je.tracka_v12_g2a_v122 import build_g2a_v122_barrier, build_scheduler_freeze_v122  # noqa: E402
from cropcop_je.tracka_v12_orchestration import (  # noqa: E402
    ACCOUNT_SLOTS,
    account_queue_manifest,
    git_credentials_present,
    sanitized_child_environment,
    scientific_run_id,
    validate_account_queue,
    validate_durable_map,
)
import run_tracka_v12_account as parent  # noqa: E402


def calibration(calibration_id: str, steady: float) -> dict:
    row = {
        "schema_version": "1.2.2",
        "calibration_id": calibration_id,
        "status": "PASS",
        "calibration_weights_scientific": False,
        "validation_enabled": False,
        "scientific_metric_computed": False,
        "resume_success": True,
        "durable_roundtrip_success": True,
        "visible_cuda_device_count": 1,
        "visible_gpu_name": "NVIDIA T4",
        "git_credentials_present": False,
        "allowed_surfaces": ["DS-V1-TRAIN", "DS-V1-VAL"],
        "v1_test_accessed": False,
        "external_protected_surface_accessed": False,
        "coverage": sorted(PROFILE_COVERAGE[calibration_id]),
        "source_git_commit": "1" * 40,
        "software_stack_sha256": "2" * 64,
        "g1a_seal_sha256": "3" * 64,
        "dependency_lock_sha256": "4" * 64,
        "measured": {
            "sec_per_optimizer_step": steady + 1.0,
            "steady_sec_per_optimizer_step": steady,
            "examples_per_second": 100.0,
            "validation_end_to_end_examples_per_second": 100.0,
            "checkpoint_save_seconds_per_event": 0.5,
            "durable_sync_seconds_per_segment": 5.0,
            "peak_gpu_memory_bytes": 1,
            "checkpoint_save_seconds": 1.0,
            "checkpoint_load_seconds": 0.1,
        },
        "durability": {
            "sync_status": "PASS",
            "restore_status": "PASS",
            "sync_seconds": 1.0,
            "restore_seconds": 1.0,
        },
    }
    if calibration_id.startswith("CAL-MNV4-"):
        row["teacher_checkpoint_sha256"] = TEACHER_SHA256
    if calibration_id == "CAL-R13":
        row["r13_pretrained_sha256"] = R13_PRETRAINED_SHA256
    return row


def scheduler():
    costs = {
        "CAL-EFFB0": 1.0,
        "CAL-CNXTT": 2.5,
        "CAL-MNV4-LOGITS": 1.3,
        "CAL-MNV4-FEATURE": 1.8,
        "CAL-R13": 3.0,
    }
    summaries = [calibration(cid, costs[cid]) for cid in REQUIRED_PROFILES]
    probe = {
        "status": "PASS",
        "selected_checkpoint_verified": True,
        "identity_mismatch_rejected": True,
        "corrupt_checkpoint_rejected": True,
    }
    barrier = build_g2a_v122_barrier(summaries, checkpoint_contract_probe=probe)
    return build_scheduler_freeze_v122(barrier)


def durable_map():
    return {
        experiment_id: f"owner/cropcop-{index:02d}"
        for index, experiment_id in enumerate(sorted(EXPERIMENT_SPECS), start=1)
    }


class TrackAV12OrchestrationTests(unittest.TestCase):
    def test_account_slot_partition_is_exact_six_slots(self):
        self.assertEqual(set(ACCOUNT_SLOTS), {"K1", "K2", "K3"})
        flattened = [slot for account in ("K1", "K2", "K3") for slot in ACCOUNT_SLOTS[account]]
        self.assertEqual(tuple(flattened), SLOT_ORDER)
        self.assertEqual(len(flattened), 6)
        self.assertEqual(len(set(flattened)), 6)

    def test_three_account_manifests_partition_exact_eleven_states(self):
        freeze = scheduler()
        observed = []
        for account in ("K1", "K2", "K3"):
            self.assertEqual(validate_account_queue(freeze, account), [])
            manifest = account_queue_manifest(freeze, account)
            self.assertEqual(set(manifest["slots"]), set(ACCOUNT_SLOTS[account]))
            for slot in ACCOUNT_SLOTS[account]:
                observed.extend(manifest["slots"][slot]["queue"])
        self.assertEqual(len(observed), len(EXPERIMENT_SPECS))
        self.assertEqual(len(observed), len(set(observed)))
        self.assertEqual(set(observed), set(EXPERIMENT_SPECS))

    def test_child_environment_strips_git_credentials_and_isolates_gpu(self):
        parent_env = {
            "GITHUB_TOKEN": "secret",
            "GH_TOKEN": "secret2",
            "CROPCOP_GITHUB_TOKEN": "secret3",
            "GIT_ASKPASS": "/tmp/askpass",
            "GIT_ASKPASS_REQUIRE": "force",
            "SSH_AUTH_SOCK": "/tmp/sock",
            "KAGGLE_USERNAME": "kept",
            "KAGGLE_KEY": "kept-secret",
        }
        env0 = sanitized_child_environment(parent_env, slot_id="K2/GPU0")
        env1 = sanitized_child_environment(parent_env, slot_id="K2/GPU1")
        self.assertFalse(git_credentials_present(env0))
        self.assertFalse(git_credentials_present(env1))
        self.assertEqual(env0["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(env1["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env0["KAGGLE_USERNAME"], "kept")
        self.assertEqual(env0["KAGGLE_KEY"], "kept-secret")

    def test_durable_map_requires_exact_unique_owner_dataset_slugs(self):
        valid = durable_map()
        self.assertEqual(validate_durable_map(valid), [])
        missing = dict(valid)
        missing.pop(next(iter(missing)))
        self.assertTrue(validate_durable_map(missing))
        duplicate = dict(valid)
        keys = list(sorted(duplicate))
        duplicate[keys[1]] = duplicate[keys[0]]
        self.assertTrue(any("distinct" in error for error in validate_durable_map(duplicate)))
        malformed = dict(valid)
        malformed[keys[0]] = "not-a-slug"
        self.assertTrue(validate_durable_map(malformed))

    def test_scientific_run_id_is_deterministic_and_source_bound(self):
        experiment_id = "R13-VIT-DLITTLE-DIFF-CONTEXT-S3"
        source = "1" * 40
        first = scientific_run_id(experiment_id, source)
        second = scientific_run_id(experiment_id, source)
        self.assertEqual(first, second)
        self.assertEqual(first, f"JE-{experiment_id}-{'1' * 12}-A01")
        self.assertNotEqual(first, scientific_run_id(experiment_id, "2" * 40))

    def test_parent_child_command_uses_scientific_mode_and_durable_required(self):
        args = SimpleNamespace(
            repo_root=str(ROOT),
            manifest="/tmp/manifest.csv",
            class_map="/tmp/class.json",
            image_root="/tmp/images",
            g1a_bundle="/tmp/g1a",
            g2a_barrier="/tmp/g2.json",
            scheduler_freeze="/tmp/scheduler.json",
            science_authorization="/tmp/go.json",
            dependency_lock="journal_extension/locks/execution_dependency_lock.json",
            source_git_commit="1" * 40,
            num_workers=4,
            checkpoint_every_steps=250,
            session_hard_limit_seconds=43200,
            finalization_margin_seconds=3600,
            min_free_gb=5.0,
        )
        command = parent.child_command(
            args,
            experiment_id="R13-VIT-DLITTLE-DIFF-CONTEXT-S1",
            slot_id="K1/GPU0",
            run_id="RUN-1",
            output_dir=Path("/tmp/out"),
            durable_locator="owner/dataset",
        )
        joined = " ".join(command)
        self.assertIn("run_tracka_v12_training_v121.py", command[1])
        self.assertIn("--mode scientific", joined)
        self.assertIn("--durable-required", command)
        self.assertIn("--resume-mode auto", joined)
        self.assertIn("--slot-id K1/GPU0", joined)

    def test_parent_publication_candidates_do_not_include_console_or_private_checkpoints(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "run_record.json").write_text("{}\n", encoding="utf-8")
            (root / "metrics.json").write_text("{}\n", encoding="utf-8")
            (root / "segments.jsonl").write_text("{}\n", encoding="utf-8")
            (root / "console.log").write_text("secret path\n", encoding="utf-8")
            private = root / "private_checkpoints"
            private.mkdir()
            (private / "x.ckpt").write_bytes(b"x")
            names = {path.name for path in parent.public_evidence_candidates(root)}
            self.assertEqual(names, {"run_record.json", "metrics.json", "segments.jsonl"})


if __name__ == "__main__":
    unittest.main()
