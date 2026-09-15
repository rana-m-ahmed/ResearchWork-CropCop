from __future__ import annotations

import builtins
import json
import symtable
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
SCRIPTS = ROOT / "journal_extension" / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cropcop_je import persistence_v8 as p8
from cropcop_je.persistence import FilesystemStore, PersistenceError


class TrackAV12MitigationV8Tests(unittest.TestCase):
    def test_kaggle_store_is_generation_aware(self):
        store = p8.build_store_v8("kaggle-dataset", "owner/cropcop-run")
        self.assertIsInstance(store, p8.GenerationAwareKagglePrivateDatasetStore)
        self.assertIsInstance(p8.build_store_v8("filesystem", "/tmp/cropcop"), FilesystemStore)

    def test_wait_requires_strictly_new_ready_generation(self):
        store = p8.GenerationAwareKagglePrivateDatasetStore("owner/cropcop-run")
        states = [
            PersistenceError("processing"),
            {"current_version_number": 4},
            {"current_version_number": 5},
        ]
        with mock.patch.object(p8, "_ready_state", side_effect=states), \
             mock.patch.object(p8.time, "monotonic", side_effect=[0.0, 1.0, 2.0, 3.0]), \
             mock.patch.object(p8.time, "sleep"):
            result = store._wait_for_new_generation(4, timeout_seconds=10.0, poll_interval_seconds=0.01)
        self.assertEqual(result["current_version_number"], 5)

    def test_generation_payload_binds_run_segment_nonce_and_index_hash(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            index = root / "checkpoint_index.json"
            index.write_text('{"schema_version":"2.0"}\n', encoding="utf-8")
            marker = {
                "schema_version": "2.0",
                "run_id": "RUN-1",
                "segment_id": "SEG-1",
                "sync_nonce": "nonce-1",
                "checkpoint_index_sha256": p8.sha256_file(index),
                "previous_version_number": 7,
                "complete": True,
            }
            (root / "durable_sync.json").write_text(json.dumps(marker), encoding="utf-8")
            loaded = p8.GenerationAwareKagglePrivateDatasetStore._validate_generation_payload(
                root,
                run_id="RUN-1",
                segment_id="SEG-1",
                expected_nonce="nonce-1",
            )
            self.assertEqual(loaded, marker)
            with self.assertRaises(PersistenceError):
                p8.GenerationAwareKagglePrivateDatasetStore._validate_generation_payload(
                    root,
                    run_id="RUN-1",
                    segment_id="SEG-1",
                    expected_nonce="stale-nonce",
                )

    def test_generation_payload_rejects_checkpoint_index_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            index = root / "checkpoint_index.json"
            index.write_text('{"schema_version":"2.0"}\n', encoding="utf-8")
            marker = {
                "schema_version": "2.0",
                "run_id": "RUN-1",
                "segment_id": "SEG-1",
                "sync_nonce": "nonce-1",
                "checkpoint_index_sha256": p8.sha256_file(index),
                "previous_version_number": 3,
                "complete": True,
            }
            (root / "durable_sync.json").write_text(json.dumps(marker), encoding="utf-8")
            index.write_text('{"schema_version":"2.0","tampered":true}\n', encoding="utf-8")
            with self.assertRaises(PersistenceError):
                p8.GenerationAwareKagglePrivateDatasetStore._validate_generation_payload(
                    root,
                    run_id="RUN-1",
                )

    def test_g1a_sealer_has_no_unresolved_runtime_globals(self):
        path = ROOT / "journal_extension" / "scripts" / "seal_tracka_v12_g1a.py"
        source = path.read_text(encoding="utf-8")
        table = symtable.symtable(source, str(path), "exec")
        main = next(child for child in table.get_children() if child.get_name() == "main")
        builtin_names = set(dir(builtins)) | {"__name__", "__file__"}
        unresolved = set()

        def walk(scope):
            for name in scope.get_identifiers():
                symbol = scope.lookup(name)
                if not symbol.is_global() or name in builtin_names:
                    continue
                module_symbol = table.lookup(name)
                if not (
                    module_symbol.is_imported()
                    or module_symbol.is_assigned()
                    or module_symbol.is_namespace()
                ):
                    unresolved.add(name)
            for child in scope.get_children():
                walk(child)

        walk(main)
        self.assertEqual(unresolved, set(), f"unresolved G1A runtime globals: {sorted(unresolved)}")

    def test_v121_runner_routes_durability_through_v8_store(self):
        text = (ROOT / "journal_extension" / "scripts" / "run_tracka_v12_training_v121.py").read_text(encoding="utf-8")
        self.assertIn("from cropcop_je.persistence_v8 import build_store_v8", text)
        self.assertIn("base.build_store = build_store_v8", text)

    def test_v122_qualifier_routes_destructive_restore_through_v8_store(self):
        text = (ROOT / "journal_extension" / "scripts" / "qualify_tracka_v12_profile_v122.py").read_text(encoding="utf-8")
        self.assertIn(
            "from cropcop_je.persistence_v8 import GenerationAwareKagglePrivateDatasetStore, build_store_v8",
            text,
        )
        self.assertIn("store = build_store_v8(args.durable_store_kind, args.durable_store_locator)", text)
        self.assertIn("isinstance(store, GenerationAwareKagglePrivateDatasetStore)", text)
        self.assertNotIn("from cropcop_je.persistence import build_store", text)
        self.assertNotIn("store = build_store(args.durable_store_kind, args.durable_store_locator)", text)


if __name__ == "__main__":
    unittest.main()
