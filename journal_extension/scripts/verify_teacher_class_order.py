from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.g1 import CLASS_MAP_SHA256, validate_teacher_factory_bundle
from cropcop_je.hashing import require_sha256, sha256_file
from cropcop_je.models import load_exact_teacher, prelogits_and_logits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--teacher-checkpoint", required=True)
    ap.add_argument("--teacher-factory", required=True)
    ap.add_argument("--teacher-factory-manifest", required=True)
    ap.add_argument("--teacher-factory-root", default="")
    ap.add_argument("--historical-lineage-manifest", required=True)
    ap.add_argument("--historical-evidence-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    require_sha256(args.class_map, CLASS_MAP_SHA256, "frozen 120-way class map")
    factory_manifest = json.loads(Path(args.teacher_factory_manifest).read_text(encoding="utf-8"))
    factory_errors = validate_teacher_factory_bundle(
        factory_manifest, source_root=(args.teacher_factory_root or args.repo_root), expected_entrypoint=args.teacher_factory
    )
    if factory_errors:
        raise SystemExit("teacher factory bundle invalid: " + "; ".join(factory_errors))

    lineage = json.loads(Path(args.historical_lineage_manifest).read_text(encoding="utf-8"))
    if lineage.get("class_map_sha256") != CLASS_MAP_SHA256:
        raise SystemExit("historical teacher lineage does not bind the frozen class-map SHA")
    if lineage.get("historical_index_semantics") != "class_map_index":
        raise SystemExit("historical teacher lineage does not establish class_map_index semantics")
    if lineage.get("not_inferred_from_shape_only") is not True:
        raise SystemExit("historical class-order proof may not rely on classifier shape alone")
    weight_key = lineage.get("classifier_weight_key")
    if not weight_key:
        raise SystemExit("historical lineage must identify the exact classifier weight key")
    evidence_root = Path(args.historical_evidence_root).resolve()
    sources = lineage.get("historical_evidence_sources", [])
    if not sources:
        raise SystemExit("historical class-order lineage has no source evidence")
    verified_sources = []
    for row in sources:
        rel = str(row.get("path", ""))
        p = (evidence_root / rel).resolve()
        if evidence_root not in p.parents and p != evidence_root:
            raise SystemExit(f"historical evidence path escapes evidence root: {rel}")
        expected = row.get("sha256")
        if not p.is_file() or not expected:
            raise SystemExit(f"historical evidence source unavailable: {rel}")
        require_sha256(p, expected, f"historical teacher evidence {rel}")
        verified_sources.append({"kind": row.get("kind"), "basename": p.name, "sha256": expected})

    teacher, factory_identity = load_exact_teacher(
        args.teacher_checkpoint,
        factory_spec=args.teacher_factory,
        factory_bundle_manifest=args.teacher_factory_manifest,
        repo_root=args.repo_root,
        factory_source_root=(args.teacher_factory_root or args.repo_root),
    )
    state = teacher.state_dict()
    if weight_key not in state:
        raise SystemExit(f"historical classifier weight key not present in loaded teacher: {weight_key}")
    classifier_out = int(state[weight_key].shape[0])
    if classifier_out != 120:
        raise SystemExit(f"historical classifier output shape is not 120: {classifier_out}")

    import torch
    with torch.no_grad():
        _features, logits = prelogits_and_logits(teacher, torch.zeros((1, 3, 256, 256), dtype=torch.float32))
    if logits.ndim != 2 or int(logits.shape[1]) != 120:
        raise SystemExit(f"teacher runtime output width is not 120: {tuple(logits.shape)}")

    record = {
        "schema_version": "1.0",
        "status": "PASS",
        "class_map_sha256": CLASS_MAP_SHA256,
        "teacher_output_width": int(logits.shape[1]),
        "classifier_out_features": classifier_out,
        "classifier_in_features": int(state[weight_key].shape[1]),
        "classifier_weight_key": weight_key,
        "classifier_weight_shape": list(state[weight_key].shape),
        "historical_index_semantics": "class_map_index",
        "factory_output_order_transform": "none",
        "teacher_factory_bundle_sha256": factory_identity["bundle_sha256"],
        "not_inferred_from_shape_only": True,
        "historical_evidence_sources": verified_sources,
        "historical_lineage_manifest_sha256": sha256_file(args.historical_lineage_manifest),
    }
    atomic_write_json(args.output, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
