from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "journal_extension" / "scripts"
SRC = ROOT / "journal_extension" / "src"
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cropcop_je.hashing import sha256_file, sha256_json  # noqa: E402
import run_tracka_v12_posttraining_account as operator  # noqa: E402
import run_tracka_v12_xai_filesystem_wrapper as xai_wrapper  # noqa: E402
from cropcop_je.tracka_v12_posttraining_operator import required_stage_args, validate_state_operator_spec  # noqa: E402

ANALYSIS = "a" * 40
EXPERIMENT = "R13-VIT-DLITTLE-DIFF-CONTEXT-S1"
RUN = "JE-R13-VIT-DLITTLE-DIFF-CONTEXT-S1-56023042e577-A01"


def write_gate(root: Path, stage: str, payload: dict) -> Path:
    attempt = root / "attempt-001"
    gate = attempt / operator.STAGE_GATE[stage]
    gate.parent.mkdir(parents=True, exist_ok=True)
    gate.write_text(json.dumps(payload), encoding="utf-8")
    return attempt



def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_local_completion_chain(root: Path) -> tuple[dict, dict, dict]:
    cert_dir = root / "certificates"
    sync = {
        "schema_version": "1.0",
        "status": "PASS",
        "generation_kind": "final",
        "experiment_id": EXPERIMENT,
        "run_id": "TRACKA-POST-R13",
        "analysis_source_git_commit": ANALYSIS,
        "dataset_locator": "owner/private-evidence",
        "generation_roundtrip_verified": True,
        "private_dataset_verified": True,
    }
    sync["sync_certificate_sha256"] = sha256_json(sync)
    sync_path = cert_dir / "POSTTRAINING_PRIVATE_SYNC_CERTIFICATE.json"
    write_json(sync_path, sync)

    completion = {
        "schema_version": "1.0",
        "status": "PASS",
        "completion_kind": "track_a_posttraining_state",
        "experiment_id": EXPERIMENT,
        "run_id": RUN,
        "posttraining_public_run_id": "TRACKA-POST-R13",
        "analysis_source_git_commit": ANALYSIS,
        "role": "direct",
        "stage_gates": {},
        "private_sync_certificate_sha256": sha256_file(sync_path),
        "private_evidence_dataset_locator": "owner/private-evidence",
        "private_generation_roundtrip_verified": True,
        "publication_branch": "run-evidence/TRACKA-POST-R13",
        "training_or_adaptation_performed": False,
        "optimizer_state_advanced": False,
        "v1_test_accessed": False,
        "external_surface_accessed": False,
    }
    completion["completion_sha256"] = sha256_json(completion)
    write_json(cert_dir / "POSTTRAINING_STATE_COMPLETION.json", completion)

    publication = {
        "schema_version": "1.0",
        "status": "PASS",
        "publication_kind": "track_a_posttraining_state",
        "experiment_id": EXPERIMENT,
        "run_id": "TRACKA-POST-R13",
        "analysis_source_git_commit": ANALYSIS,
        "public_file_sha256": {},
        "public_file_count": 0,
        "private_material_published": False,
        "publication_manifest_sha256": "1" * 64,
        "publication_branch": "run-evidence/TRACKA-POST-R13",
        "publication_manifest_file_sha256": "2" * 64,
    }
    publication["publication_certificate_sha256"] = sha256_json(publication)
    write_json(cert_dir / "POSTTRAINING_PUBLICATION_CERTIFICATE.json", publication)
    return completion, publication, sync


class TrackAPosttrainingOperatorTests(unittest.TestCase):
    def base_gate(self, stage: str) -> dict:
        payload = {
            "status": "PASS",
            "experiment_id": EXPERIMENT,
            "run_id": RUN,
            "analysis_source_git_commit": ANALYSIS,
            "v1_test_accessed": False,
            "external_surface_accessed": False,
        }
        if stage in {"direct", "auxiliary"}:
            payload.update(
                {
                    "training_performed": False,
                    "optimizer_state_advanced": False,
                    "replay_gate": {"status": "PASS"},
                }
            )
        else:
            payload["training_or_adaptation_performed"] = False
        return payload

    def test_runtime_identity_binds_run_id_from_sealed_readiness_and_run_record(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_record = root / "run_record.json"
            payload = {
                "schema_version": "1.0",
                "status": "PASS",
                "experiment_id": EXPERIMENT,
                "run_id": RUN,
            }
            write_json(run_record, payload)
            spec = {
                "role": "direct",
                "run_record": str(run_record),
                "checkpoint_root": str(root / "checkpoint"),
                "evidence_dataset_locator": "owner/private-evidence",
                "executor_args": {},
            }
            readiness = {
                "run_id": RUN,
                "run_record_sha256": sha256_file(run_record),
            }
            bound = operator.bind_runtime_identity(
                experiment_id=EXPERIMENT,
                spec=spec,
                readiness_state=readiness,
                analysis_sha=ANALYSIS,
            )
            self.assertEqual(bound["run_id"], RUN)
            self.assertEqual(bound["experiment_id"], EXPERIMENT)
            self.assertEqual(
                bound["posttraining_public_run_id"],
                f"TRACKA-POST-{EXPERIMENT.lower()}-{ANALYSIS[:12]}",
            )

    def test_runtime_identity_rejects_readiness_run_id_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_record = root / "run_record.json"
            write_json(
                run_record,
                {
                    "schema_version": "1.0",
                    "status": "PASS",
                    "experiment_id": EXPERIMENT,
                    "run_id": RUN,
                },
            )
            spec = {
                "role": "direct",
                "run_record": str(run_record),
                "checkpoint_root": str(root / "checkpoint"),
                "evidence_dataset_locator": "owner/private-evidence",
                "executor_args": {},
            }
            readiness = {
                "run_id": "wrong-run",
                "run_record_sha256": sha256_file(run_record),
            }
            with self.assertRaises(RuntimeError):
                operator.bind_runtime_identity(
                    experiment_id=EXPERIMENT,
                    spec=spec,
                    readiness_state=readiness,
                    analysis_sha=ANALYSIS,
                )

    def test_xai_stage_entrypoint_override_is_runtime_only(self):
        repo = Path("/repo")
        with patch.dict("os.environ", {"CROPCOP_TRACKA_XAI_ENTRYPOINT": "/tmp/xai-wrapper.py"}):
            self.assertEqual(
                operator.stage_script_path(repo, "xai"),
                Path("/tmp/xai-wrapper.py"),
            )
            self.assertEqual(
                operator.stage_script_path(repo, "direct"),
                repo / operator.STAGE_SCRIPT["direct"],
            )


    def test_publication_overlay_is_scoped_to_publication_subprocess_env(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            package = root / "cropcop_je"
            package.mkdir()
            (package / "publication.py").write_text("# hotfix\n", encoding="utf-8")
            with patch.dict(
                "os.environ",
                {
                    "CROPCOP_TRACKA_PUBLICATION_PYTHONPATH": str(root),
                    "PYTHONPATH": "/frozen/science/src",
                },
                clear=False,
            ):
                env = operator.publication_subprocess_env()
                self.assertEqual(
                    env["PYTHONPATH"].split(os.pathsep),
                    [str(root.resolve()), "/frozen/science/src"],
                )
                self.assertEqual(os.environ["PYTHONPATH"], "/frozen/science/src")

    def test_publication_overlay_rejects_invalid_package_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with patch.dict(
                "os.environ",
                {"CROPCOP_TRACKA_PUBLICATION_PYTHONPATH": str(root)},
                clear=False,
            ):
                with self.assertRaises(RuntimeError):
                    operator.publication_subprocess_env()

    def test_xai_filesystem_wrapper_creates_nested_panel_parents_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "panel"
            seen = []

            def fake_save(_image, fp, *args, **kwargs):
                seen.append(Path(fp))
                return None

            wrapped = xai_wrapper.guarded_panel_save(fake_save, root)
            nested = root / "main::val" / "money_plant_healthy" / "sample.jpg.png"
            wrapped(object(), nested, format="PNG")
            self.assertTrue(nested.parent.is_dir())
            self.assertEqual(seen, [nested.resolve()])

            with self.assertRaises(RuntimeError):
                wrapped(object(), root / ".." / "escape.png", format="PNG")

    def test_terminal_direct_attempt_reused_only_when_all_gates_hold(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            expected = write_gate(root, "direct", self.base_gate("direct"))
            self.assertEqual(operator.terminal_stage(root, "direct", EXPERIMENT, RUN, ANALYSIS), expected)

    def test_wrong_analysis_sha_is_not_reused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gate = self.base_gate("direct")
            gate["analysis_source_git_commit"] = "b" * 40
            write_gate(root, "direct", gate)
            self.assertIsNone(operator.terminal_stage(root, "direct", EXPERIMENT, RUN, ANALYSIS))

    def test_open_v1_surface_is_not_reused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gate = self.base_gate("direct")
            gate["v1_test_accessed"] = True
            write_gate(root, "direct", gate)
            self.assertIsNone(operator.terminal_stage(root, "direct", EXPERIMENT, RUN, ANALYSIS))

    def test_training_marker_is_not_reused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gate = self.base_gate("auxiliary")
            gate["training_performed"] = True
            write_gate(root, "auxiliary", gate)
            self.assertIsNone(operator.terminal_stage(root, "auxiliary", EXPERIMENT, RUN, ANALYSIS))

    def test_xai_warning_is_terminal_but_failed_xai_is_not(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gate = self.base_gate("xai")
            gate["status"] = "WARNING_NONFINITE_MAPS"
            expected = write_gate(root, "xai", gate)
            self.assertEqual(operator.terminal_stage(root, "xai", EXPERIMENT, RUN, ANALYSIS), expected)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gate = self.base_gate("xai")
            gate["status"] = "FAIL"
            write_gate(root, "xai", gate)
            self.assertIsNone(operator.terminal_stage(root, "xai", EXPERIMENT, RUN, ANALYSIS))

    def test_executor_args_are_allowlisted(self):
        self.assertEqual(
            operator.cli_args({"manifest": "/m.csv", "batch_size": 16}, operator.DIRECT_ALLOWED),
            ["--batch-size", "16", "--manifest", "/m.csv"],
        )
        with self.assertRaises(RuntimeError):
            operator.cli_args({"v1_test": "/forbidden"}, operator.DIRECT_ALLOWED)


    def test_preinventory_validation_allows_missing_evidence_locator_only_when_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest.csv"
            class_map = root / "class.json"
            image_root = root / "dataset"
            run_record = root / "run_record.json"
            checkpoint_root = root / "checkpoint"
            principal_config = root / "config.json"
            principal_init = root / "init.json"
            principal_evidence = root / "evidence.json"
            image_root.mkdir()
            checkpoint_root.mkdir()
            for path in (manifest, class_map, run_record, principal_config, principal_init, principal_evidence):
                path.write_text("{}", encoding="utf-8")
            spec = {
                "role": "direct",
                "run_record": str(run_record),
                "checkpoint_root": str(checkpoint_root),
                "executor_args": {
                    "direct": {
                        "manifest": str(manifest),
                        "class_map": str(class_map),
                        "image_root": str(image_root),
                        "principal_config": str(principal_config),
                        "principal_pair_init": str(principal_init),
                        "principal_pair_evidence": str(principal_evidence),
                    },
                    "xai": {
                        "manifest": str(manifest),
                        "class_map": str(class_map),
                        "image_root": str(image_root),
                        "principal_config": str(principal_config),
                        "principal_pair_init": str(principal_init),
                        "principal_pair_evidence": str(principal_evidence),
                    },
                },
            }
            strict = validate_state_operator_spec("R04-MNV4-DIRECT-S1", spec, check_paths=True)
            self.assertIn("missing:evidence_dataset_locator", strict)
            preinventory = validate_state_operator_spec(
                "R04-MNV4-DIRECT-S1",
                spec,
                check_paths=True,
                require_evidence_dataset_locator=False,
            )
            self.assertEqual(preinventory, [])

    def test_historical_secondary_direct_does_not_require_secondary_g1_bundle(self):
        for experiment_id in ("R06-EFFB0-CONTEXT-S1", "R07-CNXTT-CONTEXT-S1"):
            required = required_stage_args(experiment_id, "direct")
            self.assertNotIn("secondary_g1_bundle", required)
            self.assertEqual(required, {"manifest", "class_map", "image_root"})

    def test_local_completion_reuse_requires_full_durability_publication_chain(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            completion, _publication, _sync = build_local_completion_chain(root)
            observed = operator.load_local_completed_state(root, EXPERIMENT, RUN, ANALYSIS)
            self.assertEqual(observed, completion)

    def test_local_completion_without_publication_certificate_is_not_reused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            build_local_completion_chain(root)
            (root / "certificates" / "POSTTRAINING_PUBLICATION_CERTIFICATE.json").unlink()
            self.assertIsNone(operator.load_local_completed_state(root, EXPERIMENT, RUN, ANALYSIS))

    def test_local_completion_with_tampered_sync_certificate_is_not_reused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            build_local_completion_chain(root)
            sync_path = root / "certificates" / "POSTTRAINING_PRIVATE_SYNC_CERTIFICATE.json"
            payload = json.loads(sync_path.read_text(encoding="utf-8"))
            payload["generation_roundtrip_verified"] = False
            write_json(sync_path, payload)
            self.assertIsNone(operator.load_local_completed_state(root, EXPERIMENT, RUN, ANALYSIS))

    def test_local_completion_with_wrong_publication_branch_is_not_reused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            build_local_completion_chain(root)
            publication_path = root / "certificates" / "POSTTRAINING_PUBLICATION_CERTIFICATE.json"
            payload = json.loads(publication_path.read_text(encoding="utf-8"))
            payload["publication_branch"] = "run-evidence/wrong"
            payload.pop("publication_certificate_sha256")
            payload["publication_certificate_sha256"] = sha256_json(payload)
            write_json(publication_path, payload)
            self.assertIsNone(operator.load_local_completed_state(root, EXPERIMENT, RUN, ANALYSIS))

    def test_attempt_numbers_are_append_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "attempt-001").mkdir(parents=True)
            (root / "attempt-004").mkdir()
            self.assertEqual(operator.next_attempt(root).name, "attempt-005")


if __name__ == "__main__":
    unittest.main()
