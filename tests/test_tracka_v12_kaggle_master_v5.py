from __future__ import annotations

import hashlib
import json
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

import master_account_driver_v5 as driver_v5
import master_g1a
import master_verified_pretrained_v5 as verified


class TrackAV12MasterV5Tests(unittest.TestCase):
    def _fake_pipeline(self, root: Path, *, omit_match: bool = False):
        calls: list[str] = []

        def fake_run(cmd, **kwargs):
            cmd = [str(x) for x in cmd]
            script = Path(cmd[1]).name
            calls.append(script)
            key = cmd[cmd.index("--model-key") + 1]
            if script == "prepare_torchvision_pretrained.py":
                out = Path(cmd[cmd.index("--output-dir") + 1])
                record = Path(cmd[cmd.index("--record") + 1])
                out.mkdir(parents=True, exist_ok=True)
                filename = "efficientnet_b0_rwightman-7f5810bc.pth" if key == "effb0" else "convnext_tiny-983f1562.pth"
                artifact = out / filename
                artifact.write_bytes((key + "-official-bytes").encode())
                digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
                record.write_text(json.dumps({
                    "status": "PASS", "model_key": key,
                    "official_filename": filename,
                    "artifact_sha256": digest, "artifact_bytes": artifact.stat().st_size,
                }), encoding="utf-8")
            elif script == "capture_torchvision_pretrained_provenance.py":
                artifact = Path(cmd[cmd.index("--artifact") + 1])
                output = Path(cmd[cmd.index("--output") + 1])
                digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
                payload = {
                    "status": "PASS", "model_key": key,
                    "artifact_sha256": digest, "artifact_bytes": artifact.stat().st_size,
                    "tensor_identity_algorithm": "sha256-canonical-tensor-v1",
                    "tensor_identity_sha256": "a" * 64,
                    "official_tensor_identity_algorithm": "sha256-canonical-tensor-v1",
                    "official_tensor_identity_sha256": "a" * 64,
                }
                if not omit_match:
                    payload["official_tensor_match"] = True
                output.write_text(json.dumps(payload), encoding="utf-8")
            else:
                raise AssertionError(f"unexpected script: {script}")
            return mock.Mock(returncode=0)

        return calls, fake_run

    def test_verified_pretrained_runs_prepare_then_capture_for_both_models(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            calls, fake_run = self._fake_pipeline(root)
            with mock.patch.object(verified, "run", side_effect=fake_run), \
                 mock.patch.object(verified, "_science_validator", return_value=lambda record, **kwargs: []):
                result = verified.prepare_verified_torchvision(root, root / "work")
            self.assertEqual(calls, [
                "prepare_torchvision_pretrained.py",
                "capture_torchvision_pretrained_provenance.py",
                "prepare_torchvision_pretrained.py",
                "capture_torchvision_pretrained_provenance.py",
            ])
            for key in ("effb0", "cnxtt"):
                self.assertTrue(result[key]["artifact"].is_file())
                self.assertTrue(result[key]["download_record"].is_file())
                provenance = json.loads(result[key]["provenance"].read_text())
                self.assertTrue(provenance["official_tensor_match"])
                self.assertEqual(provenance["tensor_identity_sha256"], provenance["official_tensor_identity_sha256"])

    def test_download_receipt_cannot_masquerade_as_tensor_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, fake_run = self._fake_pipeline(root, omit_match=True)
            with mock.patch.object(verified, "run", side_effect=fake_run), \
                 mock.patch.object(verified, "_science_validator", return_value=lambda record, **kwargs: []):
                with self.assertRaisesRegex(Exception, "official_tensor_match"):
                    verified.prepare_verified_torchvision(root, root / "work")

    def test_master_g1a_uses_verified_provenance_stage_only(self):
        source = (OPS / "master_g1a.py").read_text(encoding="utf-8")
        self.assertIn("prepare_verified_torchvision", source)
        self.assertNotIn("prepare_official_torchvision", source)
        self.assertIn('baselines["effb0"]["provenance"]', source)
        self.assertIn('baselines["cnxtt"]["provenance"]', source)

    def test_frozen_capture_script_contains_required_identity_contract(self):
        source = (ROOT / "journal_extension" / "scripts" / "capture_torchvision_pretrained_provenance.py").read_text(encoding="utf-8")
        for field in (
            "tensor_identity_algorithm",
            "tensor_identity_sha256",
            "official_tensor_identity_algorithm",
            "official_tensor_identity_sha256",
            "official_tensor_match",
            "validate_torchvision_provenance",
        ):
            self.assertIn(field, source)

    def test_v5_driver_orders_preflight_before_stack_and_science(self):
        source = (OPS / "master_account_driver_v5.py").read_text(encoding="utf-8")
        ordered = [
            'stage("RUNTIME_AND_HARDWARE_PREFLIGHT"',
            'stage("FROZEN_INPUT_RESOLUTION"',
            'stage("SCIENCE_SOURCE_AND_GITHUB_PREFLIGHT"',
            'stage("EXACT_EXECUTION_STACK"',
            'stage("CANONICAL_G1A"',
            'stage("ACCOUNT_G2A"',
            'stage("CONTROL_PLANE"',
            'stage("SCIENCE_DURABILITY_PREFLIGHT"',
            'stage("SCIENTIFIC_QUEUE"',
        ]
        positions = [source.index(token) for token in ordered]
        self.assertEqual(positions, sorted(positions))

    def test_optional_expected_account_gate_fails_closed(self):
        with mock.patch.dict("os.environ", {"CROPCOP_EXPECTED_KAGGLE_USERNAME": "expected-user"}, clear=False):
            with self.assertRaisesRegex(Exception, "Kaggle account mismatch"):
                driver_v5.assert_expected_account("K1", "other-user")
        with mock.patch.dict("os.environ", {"CROPCOP_EXPECTED_KAGGLE_USERNAME": "Expected-User"}, clear=False):
            driver_v5.assert_expected_account("K1", "expected-user")


if __name__ == "__main__":
    unittest.main()
