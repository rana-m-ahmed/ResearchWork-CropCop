from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "journal_extension" / "kaggle" / "tracka_v12_ops"
SRC = ROOT / "journal_extension" / "src"
for path in (OPS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import master_control
import master_g1a
import master_g2a
import master_publication_v4 as pub4
import master_science_v4 as science4
import tracka_v12_kaggle_operator_v3 as v3


class PublicationV4Tests(unittest.TestCase):
    def _file(self, root: Path, name: str = "evidence.json", body: str = "{}\n") -> Path:
        path = root / name
        path.write_text(body, encoding="utf-8")
        return path

    def test_existing_identical_remote_is_successful_noop(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            evidence = self._file(root)
            frozen = mock.Mock()
            with mock.patch.object(pub4, "_validated_inputs", return_value=(root, [evidence])), \
                 mock.patch.object(pub4.v3, "load_github_token", return_value="token"), \
                 mock.patch.object(pub4, "_publication_api", return_value=(mock.Mock(), frozen)), \
                 mock.patch.object(pub4, "remote_evidence_matches", return_value=True):
                branch = pub4.publish_public_files(root, "RUN-1", [evidence])
            self.assertEqual(branch, "run-evidence/RUN-1")
            frozen.assert_not_called()

    def test_differing_remote_uses_frozen_publisher_then_roundtrips(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            evidence = self._file(root)
            frozen = mock.Mock(return_value="run-evidence/RUN-2")
            with mock.patch.object(pub4, "_validated_inputs", return_value=(root, [evidence])), \
                 mock.patch.object(pub4.v3, "load_github_token", return_value="token"), \
                 mock.patch.object(pub4, "_publication_api", return_value=(mock.Mock(), frozen)), \
                 mock.patch.object(pub4, "remote_evidence_matches", side_effect=[False, False, True, True]):
                branch = pub4.publish_public_files(root, "RUN-2", [evidence])
            self.assertEqual(branch, "run-evidence/RUN-2")
            frozen.assert_called_once_with(
                repo_dir=root,
                source_git_sha=v3.SCIENCE_SHA,
                run_id="RUN-2",
                files=[str(evidence.resolve())],
            )

    def test_publish_exception_after_remote_update_is_recovered(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            evidence = self._file(root)
            frozen = mock.Mock(side_effect=RuntimeError("client lost response"))
            with mock.patch.object(pub4, "_validated_inputs", return_value=(root, [evidence])), \
                 mock.patch.object(pub4.v3, "load_github_token", return_value="token"), \
                 mock.patch.object(pub4, "_publication_api", return_value=(mock.Mock(), frozen)), \
                 mock.patch.object(pub4, "remote_evidence_matches", side_effect=[False, False, True, True]):
                branch = pub4.publish_public_files(root, "RUN-3", [evidence])
            self.assertEqual(branch, "run-evidence/RUN-3")
            frozen.assert_called_once()

    def test_partial_multi_file_bundle_only_publishes_missing_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            already = self._file(root, "already.json", '{"status":"old"}\n')
            missing = self._file(root, "missing.json", '{"status":"new"}\n')
            frozen = mock.Mock(return_value="run-evidence/RUN-PARTIAL")
            with mock.patch.object(pub4, "_validated_inputs", return_value=(root, [already, missing])), \
                 mock.patch.object(pub4.v3, "load_github_token", return_value="token"), \
                 mock.patch.object(pub4, "_publication_api", return_value=(mock.Mock(), frozen)), \
                 mock.patch.object(pub4, "remote_evidence_matches", side_effect=[False, True, False, True, True]):
                branch = pub4.publish_public_files(root, "RUN-PARTIAL", [already, missing])
            self.assertEqual(branch, "run-evidence/RUN-PARTIAL")
            frozen.assert_called_once_with(
                repo_dir=root,
                source_git_sha=v3.SCIENCE_SHA,
                run_id="RUN-PARTIAL",
                files=[str(missing.resolve())],
            )

    def test_multi_file_bundle_updates_only_changed_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            unchanged = self._file(root, "unchanged.json", '{"v":1}\n')
            changed = self._file(root, "changed.json", '{"v":2}\n')
            frozen = mock.Mock(return_value="run-evidence/RUN-CHANGED")
            with mock.patch.object(pub4, "_validated_inputs", return_value=(root, [unchanged, changed])), \
                 mock.patch.object(pub4.v3, "load_github_token", return_value="token"), \
                 mock.patch.object(pub4, "_publication_api", return_value=(mock.Mock(), frozen)), \
                 mock.patch.object(pub4, "remote_evidence_matches", side_effect=[False, True, False, True, True]):
                branch = pub4.publish_public_files(root, "RUN-CHANGED", [unchanged, changed])
            self.assertEqual(branch, "run-evidence/RUN-CHANGED")
            frozen.assert_called_once_with(
                repo_dir=root,
                source_git_sha=v3.SCIENCE_SHA,
                run_id="RUN-CHANGED",
                files=[str(changed.resolve())],
            )

    def test_wrong_remote_ancestry_fails_closed(self):
        fake = subprocess.CompletedProcess([], returncode=1, stdout=b"", stderr=b"")
        with mock.patch.object(pub4.subprocess, "run", return_value=fake):
            with self.assertRaises(v3.OperatorError):
                pub4._assert_source_ancestry(Path("/repo"), "refs/remotes/origin/run-evidence/x")

    def test_duplicate_basenames_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = root / "a" / "same.json"
            b = root / "b" / "same.json"
            a.parent.mkdir()
            b.parent.mkdir()
            a.write_text("{}\n", encoding="utf-8")
            b.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(v3.OperatorError):
                pub4._validated_inputs(root, "RUN", [a, b])

    def test_stage_hooks_replace_every_pre_science_publisher(self):
        old = (master_g1a.publish_public_files, master_g2a.publish_public_files, master_control.publish_public_files)
        try:
            pub4.install_stage_publication_hooks()
            self.assertIs(master_g1a.publish_public_files, pub4.publish_public_files)
            self.assertIs(master_g2a.publish_public_files, pub4.publish_public_files)
            self.assertIs(master_control.publish_public_files, pub4.publish_public_files)
        finally:
            master_g1a.publish_public_files, master_g2a.publish_public_files, master_control.publish_public_files = old

    def test_science_command_disables_frozen_git_publisher(self):
        command = science4.science_command(
            Path("/repo"),
            account_id="K1",
            manifest=Path("/manifest.csv"),
            class_map=Path("/class.json"),
            image_root=Path("/images"),
            g1a_bundle=Path("/g1a"),
            control_dir=Path("/control"),
            master_root=Path("/master"),
        )
        self.assertNotIn("--publish-evidence", command)
        self.assertEqual(command.count("--source-git-commit"), 1)
        self.assertEqual(command[command.index("--source-git-commit") + 1], v3.SCIENCE_SHA)

    def test_completed_science_evidence_is_parent_published(self):
        summary = {
            "slot_results": {
                "K1/GPU0": [{
                    "experiment_id": "R06-EFFB0-CONTEXT-S2",
                    "run_id": "RUN-SCIENCE",
                    "return_code": 0,
                    "run_status": "PASS",
                }]
            }
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "run"
            run_dir.mkdir()
            evidence = self._file(run_dir, "RUN_STATUS.json")
            with mock.patch.object(science4, "_orchestration_api", return_value=(
                lambda output_root, experiment_id: run_dir,
                lambda path: [evidence],
            )), mock.patch.object(science4, "publish_public_files", return_value="run-evidence/RUN-SCIENCE") as publish:
                result = science4.publish_completed_scientific_runs(root, "K1", summary, root)
            self.assertEqual(result["R06-EFFB0-CONTEXT-S2"]["status"], "PASS")
            publish.assert_called_once_with(root, "RUN-SCIENCE", [evidence])


if __name__ == "__main__":
    unittest.main()
