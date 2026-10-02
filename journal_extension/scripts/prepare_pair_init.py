from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.hashing import sha256_file
from cropcop_je.models import create_student_from_pretrained, save_pair_initialization

SEEDS = {1: 21270083, 2: 606135704, 3: 1153870846}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-index", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--pretrained-evidence", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--evidence", required=True)
    args = ap.parse_args()

    pre_path = Path(args.pretrained)
    pre_sha = sha256_file(pre_path)
    pre_evidence_path = Path(args.pretrained_evidence)
    if pre_evidence_path.exists():
        bound = json.loads(pre_evidence_path.read_text(encoding="utf-8"))
        if bound.get("sha256") != pre_sha or bound.get("bytes") != pre_path.stat().st_size:
            raise SystemExit("MobileNetV4 pretrained bytes differ from existing G1 evidence")
    else:
        bound = {
            "schema_version": "1.0",
            "kind": "mnv4_pretrained",
            "sha256": pre_sha,
            "bytes": pre_path.stat().st_size,
            "artifact_basename": pre_path.name,
            "verification": "exact_byte_hash",
        }
        pre_evidence_path.parent.mkdir(parents=True, exist_ok=True)
        pre_evidence_path.write_text(json.dumps(bound, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    idx = args.seed_index
    seed = SEEDS[idx]
    pair_id = f"MNV4-PAIR-S{idx}"
    model = create_student_from_pretrained(pre_path, seed=seed, num_classes=120)
    init_sha = save_pair_initialization(model, args.output, pair_id=pair_id, seed=seed, pretrained_sha256=pre_sha)
    evidence = {
        "schema_version": "1.0",
        "pair_id": pair_id,
        "seed": seed,
        "model_name": "mobilenetv4_conv_medium.e500_r256_in1k",
        "num_classes": 120,
        "pretrained_sha256": pre_sha,
        "student_init_sha256": init_sha,
        "student_init_bytes": Path(args.output).stat().st_size,
        "student_init_basename": Path(args.output).name,
        "authorized_consumers": [f"R04-MNV4-DIRECT-S{idx}", f"R05-MNV4-TEACHER-S{idx}"],
    }
    out = Path(args.evidence)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
