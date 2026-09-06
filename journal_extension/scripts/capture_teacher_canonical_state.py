from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.g1 import (
    CLASS_MAP_SHA256,
    MANIFEST_SHA256,
    validate_teacher_factory_bundle,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--teacher-factory-root", required=True)
    ap.add_argument("--teacher-factory", required=True)
    ap.add_argument("--teacher-factory-manifest", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.teacher_factory_root).resolve()
    manifest = json.loads(Path(args.teacher_factory_manifest).read_text(encoding="utf-8"))
    errors = validate_teacher_factory_bundle(
        manifest,
        source_root=root,
        expected_entrypoint=args.teacher_factory,
    )
    if errors:
        raise SystemExit("teacher factory bundle invalid: " + "; ".join(errors))

    if ":" not in args.teacher_factory:
        raise SystemExit("teacher factory spec must be module:function")
    module_name, _fn_name = args.teacher_factory.split(":", 1)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module(module_name)
    inspect_fn = getattr(module, "inspect_checkpoint", None)
    if not callable(inspect_fn):
        raise SystemExit("teacher factory module must expose inspect_checkpoint()")

    facts = dict(inspect_fn(args.checkpoint))
    if facts.get("manifest_sha256") != MANIFEST_SHA256:
        raise SystemExit("canonical teacher evidence manifest SHA mismatch")
    if facts.get("class_map_sha256") != CLASS_MAP_SHA256:
        raise SystemExit("canonical teacher evidence class-map SHA mismatch")
    if facts.get("canonical_state") != "EMA":
        raise SystemExit("canonical teacher state is not EMA")
    if facts.get("ema_exact_complete_coverage") is not True:
        raise SystemExit("canonical teacher does not have complete EMA coverage")
    if facts.get("canonical_floating_tensors_equal_ema") is not True:
        raise SystemExit("canonical floating tensors do not exactly equal EMA")
    if facts.get("finite_state") is not True:
        raise SystemExit("canonical teacher state is not finite")
    if facts.get("output_order_transform") != "none":
        raise SystemExit("teacher output order transform is not none")

    record = {
        "schema_version": "1.0",
        "status": "PASS",
        **facts,
        "factory_bundle_sha256": manifest["bundle_sha256"],
        "transformers_version": importlib.metadata.version("transformers"),
        "scientific_metric_computed": False,
        "protected_data_accessed": False,
    }
    atomic_write_json(args.output, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
