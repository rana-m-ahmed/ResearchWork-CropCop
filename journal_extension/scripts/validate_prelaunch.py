from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.hashing import sha256_file
from cropcop_je.validate import validate_static


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--evidence-dir", required=True)
    ap.add_argument("--pair-init-dir", required=True)
    ap.add_argument("--require-teacher", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    report = validate_static(repo)
    errors = list(report["errors"])
    evidence = Path(args.evidence_dir)
    pair_dir = Path(args.pair_init_dir)

    pre = evidence / "MNV4_PRETRAINED.json"
    if not pre.exists():
        errors.append("G1 MNV4_PRETRAINED.json is missing")
    else:
        p = json.loads(pre.read_text(encoding="utf-8"))
        if len(p.get("sha256", "")) != 64:
            errors.append("G1 MobileNetV4 pretrained SHA is invalid")

    pair_shas = {}
    for i in (1, 2, 3):
        ev = evidence / f"PAIR_INIT_S{i}.json"
        if not ev.exists():
            errors.append(f"PAIR_INIT_S{i}.json missing")
            continue
        row = json.loads(ev.read_text(encoding="utf-8"))
        expected_consumers = {f"R04-MNV4-DIRECT-S{i}", f"R05-MNV4-TEACHER-S{i}"}
        if set(row.get("authorized_consumers", [])) != expected_consumers:
            errors.append(f"S{i} pair-init consumer set mismatch")
        binary = pair_dir / row.get("student_init_basename", "")
        if not binary.exists():
            errors.append(f"S{i} pair-init binary missing from supplied private directory")
        elif sha256_file(binary) != row.get("student_init_sha256"):
            errors.append(f"S{i} pair-init SHA mismatch")
        pair_shas[f"S{i}"] = row.get("student_init_sha256")

    if args.require_teacher:
        teacher = evidence / "DINO_TEACHER.json"
        if not teacher.exists():
            errors.append("DINO_TEACHER.json missing")
        else:
            t = json.loads(teacher.read_text(encoding="utf-8"))
            if t.get("sha256") != "74b4701b8931976c9227845ead50788ae47e3596f575f2817b7352a715f53b79":
                errors.append("historical DINO teacher SHA mismatch")
            if t.get("class_map_sha256") != "46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2":
                errors.append("teacher evidence class-map identity mismatch")

    report.update({
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "pair_init_sha256": pair_shas,
        "teacher_required": args.require_teacher,
    })
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
