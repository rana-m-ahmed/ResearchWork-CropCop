import unittest
from unittest import mock

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "journal_extension" / "src"))

from cropcop_je.envelope import (
    all_reserved_run_ids,
    durable_plan,
    resolve_run_id,
)


class QA1WaveCTests(unittest.TestCase):
    def _env(self):
        return {
            "CROPCOP_DURABLE_STORE_KIND": "filesystem",
            "CROPCOP_DURABLE_LOCATOR_TEMPLATE": "/tmp/cropcop/{run_id}",
        }

    def _capture_access(self, current_ids):
        captured = {}
        def fake_access(kind, resolved, *, env=None):
            captured["kind"] = kind
            captured["resolved"] = dict(resolved)
            return {
                "schema_version": "1.0",
                "status": "PASS",
                "kind": kind,
                "resolved_locator_count": len(resolved),
                "checks": {},
                "errors": [],
            }
        with mock.patch("cropcop_je.envelope.validate_durable_access_plan", side_effect=fake_access):
            resolved, access = durable_plan(
                "a" * 40,
                self._env(),
                current_run_ids=current_ids,
            )
        return captured, resolved, access

    def test_01_p1_access_probe_contains_only_s1_pair(self):
        source = "a" * 40
        p1 = [
            resolve_run_id("R04-MNV4-DIRECT-S1", source, self._env()),
            resolve_run_id("R05-MNV4-TEACHER-S1", source, self._env()),
        ]
        captured, resolved, access = self._capture_access(p1)
        self.assertEqual(set(captured["resolved"]), set(p1))
        self.assertEqual(access["current_envelope_locator_count"], 2)
        self.assertEqual(access["global_reserved_locator_count"], 9)
        self.assertEqual(len(resolved), 9)
        self.assertFalse(any("S2" in rid or "S3" in rid for rid in captured["resolved"]))

    def test_02_p2_access_probe_does_not_require_p1_or_p3(self):
        source = "a" * 40
        p2 = [
            resolve_run_id("R04-MNV4-DIRECT-S2", source, self._env()),
            resolve_run_id("R05-MNV4-TEACHER-S2", source, self._env()),
        ]
        captured, _, _ = self._capture_access(p2)
        self.assertEqual(set(captured["resolved"]), set(p2))
        self.assertFalse(any("S1" in rid or "S3" in rid for rid in captured["resolved"]))

    def test_03_p3_access_probe_does_not_require_p1_or_p2(self):
        source = "a" * 40
        p3 = [
            resolve_run_id("R04-MNV4-DIRECT-S3", source, self._env()),
            resolve_run_id("R05-MNV4-TEACHER-S3", source, self._env()),
        ]
        captured, _, _ = self._capture_access(p3)
        self.assertEqual(set(captured["resolved"]), set(p3))
        self.assertFalse(any("S1" in rid or "S2" in rid for rid in captured["resolved"]))

    def test_04_g2_probes_exactly_three_calibration_locators(self):
        ids = ["CAL-MNV4-DIRECT", "CAL-MNV4-TEACHER", "CAL-CNXTT"]
        captured, resolved, access = self._capture_access(ids)
        self.assertEqual(list(captured["resolved"]), ids)
        self.assertEqual(access["current_envelope_locator_count"], 3)
        self.assertEqual(len(resolved), 9)

    def test_05_global_namespace_collision_still_fails_before_access_probe(self):
        env = {
            "CROPCOP_DURABLE_STORE_KIND": "filesystem",
            "CROPCOP_DURABLE_LOCATOR_TEMPLATE": "/tmp/cropcop/shared",
        }
        with mock.patch("cropcop_je.envelope.validate_durable_access_plan") as access:
            with self.assertRaises(ValueError):
                durable_plan("a" * 40, env, current_run_ids=["CAL-MNV4-DIRECT"])
        access.assert_not_called()

    def test_06_unknown_current_run_id_fails_closed(self):
        with mock.patch("cropcop_je.envelope.validate_durable_access_plan") as access:
            with self.assertRaisesRegex(Exception, "unknown durable run IDs"):
                durable_plan("a" * 40, self._env(), current_run_ids=["NOT-A-RESERVED-RUN"])
        access.assert_not_called()

    def test_07_global_reserved_namespace_remains_nine_unique_ids(self):
        ids = all_reserved_run_ids("a" * 40, self._env())
        self.assertEqual(len(ids), 9)
        self.assertEqual(len(set(ids)), 9)

    def test_08_access_preflight_wording_does_not_claim_fake_kaggle_write(self):
        source = (ROOT / "journal_extension/src/cropcop_je/persistence.py").read_text()
        self.assertIn("write_generation_mutated_by_preflight", source)
        self.assertIn("first real calibration/scientific sync", source)
        self.assertNotIn("throwaway dataset version", source.lower().replace("creating a ", ""))


if __name__ == "__main__":
    unittest.main()
