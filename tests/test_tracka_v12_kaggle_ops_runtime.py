from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "journal_extension" / "kaggle" / "tracka_v12_ops"
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

import science_account_driver as science_driver
import tracka_v12_kaggle_operator_v2 as ops


EXPERIMENTS = [
    "R06-EFFB0-CONTEXT-S2",
    "R06-EFFB0-CONTEXT-S3",
    "R07-CNXTT-CONTEXT-S2",
    "R07-CNXTT-CONTEXT-S3",
    "R12-MNV4-LOGITS-S2",
    "R12-MNV4-LOGITS-S3",
    "R12-MNV4-FEATURE-S2",
    "R12-MNV4-FEATURE-S3",
    "R13-VIT-DLITTLE-DIFF-CONTEXT-S1",
    "R13-VIT-DLITTLE-DIFF-CONTEXT-S2",
    "R13-VIT-DLITTLE-DIFF-CONTEXT-S3",
]


class TrackAV12KaggleOperatorRuntimeTests(unittest.TestCase):
    def test_kaggle_slug_is_deterministic_safe_and_bounded(self):
        identity = "R13-VIT-DLITTLE-DIFF-CONTEXT-S1"
        first = ops.kaggle_safe_dataset_slug("cropcop", identity)
        second = ops.kaggle_safe_dataset_slug("cropcop", identity)
        other = ops.kaggle_safe_dataset_slug("cropcop", "R13-VIT-DLITTLE-DIFF-CONTEXT-S2")
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertGreaterEqual(len(first), 3)
        self.assertLessEqual(len(first), 50)
        self.assertTrue(all(ch.isalnum() or ch == "-" for ch in first))
        self.assertTrue(first.endswith(ops.SCIENCE_SHA[:8]))

    def test_all_frozen_experiment_slugs_are_unique_and_bounded(self):
        slugs = [ops.kaggle_safe_dataset_slug("cropcop", eid) for eid in EXPERIMENTS]
        self.assertEqual(len(slugs), 11)
        self.assertEqual(len(set(slugs)), 11)
        self.assertTrue(all(3 <= len(slug) <= 50 for slug in slugs))

    def test_scientific_durable_map_is_exact_unique_and_owner_bound(self):
        scheduler = {
            "static_slot_queues": {
                "K1/GPU0": [EXPERIMENTS[0], EXPERIMENTS[4]],
                "K1/GPU1": [EXPERIMENTS[2], EXPERIMENTS[8]],
                "K2/GPU0": [EXPERIMENTS[1], EXPERIMENTS[5]],
                "K2/GPU1": [EXPERIMENTS[3], EXPERIMENTS[9]],
                "K3/GPU0": [EXPERIMENTS[6], EXPERIMENTS[10]],
                "K3/GPU1": [EXPERIMENTS[7]],
            }
        }
        owners = {"K1": "alpha", "K2": "beta", "K3": "gamma"}
        mapping = ops.scientific_durable_map(scheduler, owners)
        self.assertEqual(set(mapping), set(EXPERIMENTS))
        self.assertEqual(len(set(mapping.values())), 11)
        for experiment_id, locator in mapping.items():
            owner, dataset = locator.split("/", 1)
            self.assertIn(owner, set(owners.values()))
            self.assertLessEqual(len(dataset), 50)
            self.assertGreaterEqual(len(dataset), 3)
            self.assertIn(ops.SCIENCE_SHA[:8], dataset)

    def test_scientific_durable_map_rejects_duplicate_experiment(self):
        scheduler = {
            "static_slot_queues": {
                "K1/GPU0": ["R06-EFFB0-CONTEXT-S2"],
                "K1/GPU1": ["R06-EFFB0-CONTEXT-S2"],
            }
        }
        with self.assertRaises(ops.OperatorError):
            ops.scientific_durable_map(scheduler, {"K1": "alpha"})

    def test_image_root_resolves_one_manifest_compatible_mount(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest.csv"
            rels = [f"class-{i % 2}/image-{i}.jpg" for i in range(8)]
            with manifest.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["portable_relpath"])
                writer.writeheader()
                for rel in rels:
                    writer.writerow({"portable_relpath": rel})
            mount = root / "mount-a"
            for rel in rels:
                path = mount / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            self.assertEqual(ops.resolve_image_root(manifest, input_root=root), mount.resolve())

    def test_image_root_rejects_duplicate_compatible_mounts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest.csv"
            rels = [f"class-{i % 2}/image-{i}.jpg" for i in range(8)]
            with manifest.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["portable_relpath"])
                writer.writeheader()
                for rel in rels:
                    writer.writerow({"portable_relpath": rel})
            for name in ("mount-a", "mount-b"):
                mount = root / name
                for rel in rels:
                    path = mount / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"x")
            with self.assertRaises(ops.OperatorError):
                ops.resolve_image_root(manifest, input_root=root)

    def test_principal_g1_bundle_resolves_by_seal_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bundle = root / "one" / "principal"
            (bundle / "private").mkdir(parents=True)
            (bundle / "evidence").mkdir()
            (bundle / "G1_MODEL_IDENTITY_SEAL.json").write_text(
                json.dumps({"g1_seal_sha256": ops.PRINCIPAL_G1_SEAL_SHA256}),
                encoding="utf-8",
            )
            self.assertEqual(ops.resolve_principal_g1_bundle(root), bundle.resolve())

            duplicate = root / "two" / "principal"
            (duplicate / "private").mkdir(parents=True)
            (duplicate / "evidence").mkdir()
            (duplicate / "G1_MODEL_IDENTITY_SEAL.json").write_text(
                json.dumps({"g1_seal_sha256": ops.PRINCIPAL_G1_SEAL_SHA256}),
                encoding="utf-8",
            )
            with self.assertRaises(ops.OperatorError):
                ops.resolve_principal_g1_bundle(root)

    def test_child_environment_strips_git_credentials_but_keeps_kaggle_credentials(self):
        env = {
            "GITHUB_TOKEN": "secret",
            "GH_TOKEN": "secret2",
            "CROPCOP_GITHUB_TOKEN": "secret3",
            "GIT_ASKPASS": "ask",
            "GIT_ASKPASS_REQUIRE": "force",
            "SSH_AUTH_SOCK": "/tmp/sock",
            "KAGGLE_USERNAME": "user",
            "KAGGLE_KEY": "key",
            "PATH": os.environ.get("PATH", ""),
        }
        with patch.dict(os.environ, env, clear=True):
            child = ops.sanitized_child_env(gpu_index=1, global_clock=123.5)
        for name in ops.GIT_CREDENTIAL_ENV_NAMES:
            self.assertNotIn(name, child)
        self.assertEqual(child["KAGGLE_USERNAME"], "user")
        self.assertEqual(child["KAGGLE_KEY"], "key")
        self.assertEqual(child["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(child["CROPCOP_NOTEBOOK_STARTED_MONOTONIC"], repr(123.5))
        self.assertEqual(child["OMP_NUM_THREADS"], "1")

    def test_rollover_rejects_empty_or_malformed_summary(self):
        self.assertFalse(science_driver.continuation_is_technical({}))
        self.assertFalse(science_driver.continuation_is_technical({"status": "PASS", "science_complete": False}))
        self.assertFalse(science_driver.continuation_is_technical({
            "status": "ATTENTION_REQUIRED",
            "science_complete": False,
            "worker_errors": {},
            "slot_results": {},
        }))

    def test_rollover_accepts_explicit_session_budget_continuation(self):
        summary = {
            "status": "ATTENTION_REQUIRED",
            "science_complete": False,
            "worker_errors": {},
            "slot_results": {
                "K1/GPU0": [{
                    "experiment_id": "R06-EFFB0-CONTEXT-S2",
                    "status": "NOT_STARTED_SESSION_BUDGET",
                    "return_code": None,
                    "run_status": None,
                    "continuation_required": True,
                }]
            },
        }
        self.assertTrue(science_driver.continuation_is_technical(summary))

    def test_rollover_accepts_explicit_checkpointed_rollover(self):
        summary = {
            "status": "ATTENTION_REQUIRED",
            "science_complete": False,
            "worker_errors": {},
            "slot_results": {
                "K2/GPU0": [{
                    "experiment_id": "R07-CNXTT-CONTEXT-S2",
                    "return_code": 0,
                    "run_status": "LAUNCHED",
                    "continuation_required": True,
                    "session_rollover_required": True,
                }]
            },
        }
        self.assertTrue(science_driver.continuation_is_technical(summary))

    def test_rollover_rejects_worker_failure_or_quarantine(self):
        worker_failure = {
            "status": "ATTENTION_REQUIRED",
            "science_complete": False,
            "worker_errors": {"K1/GPU0": {"reason": "boom"}},
            "slot_results": {"K1/GPU0": []},
        }
        self.assertFalse(science_driver.continuation_is_technical(worker_failure))

        quarantine = {
            "status": "ATTENTION_REQUIRED",
            "science_complete": False,
            "worker_errors": {},
            "slot_results": {
                "K1/GPU0": [{
                    "return_code": 1,
                    "run_status": "FAIL",
                    "slot_quarantined": True,
                    "continuation_required": False,
                }]
            },
        }
        self.assertFalse(science_driver.continuation_is_technical(quarantine))


if __name__ == "__main__":
    unittest.main()
