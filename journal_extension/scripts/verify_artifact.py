from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.hashing import sha256_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--kind", required=True, choices=["mnv4_pretrained", "teacher"])
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--expected-sha256", default="")
    ap.add_argument("--class-map-sha256", default="")
    args = ap.parse_args()

    path = Path(args.artifact)
    digest = sha256_file(path)
    if args.expected_sha256 and digest.lower() != args.expected_sha256.lower():
        raise SystemExit(f"SHA-256 mismatch for {args.kind}: expected {args.expected_sha256}, got {digest}")
    record = {
        "schema_version": "1.0",
        "kind": args.kind,
        "sha256": digest,
        "bytes": path.stat().st_size,
        "artifact_basename": path.name,
        "verification": "exact_byte_hash",
        "class_map_sha256": args.class_map_sha256 or None,
    }
    if args.kind == "teacher" and not args.class_map_sha256:
        raise SystemExit("--class-map-sha256 is required for teacher evidence")
    out = Path(args.evidence)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        if old.get("sha256") != digest or old.get("bytes") != path.stat().st_size:
            raise SystemExit(f"existing evidence {out} binds different bytes")
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
