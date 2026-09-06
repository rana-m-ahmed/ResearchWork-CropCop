from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.g1_publication import (
    G1PublicationError,
    ensure_private_target,
    publish_and_roundtrip,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", required=True)
    ap.add_argument("--dataset-slug", required=True)
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--mode", choices=["publish", "repair"], default="publish")
    args = ap.parse_args()

    try:
        ensure_private_target(args.dataset_slug, env=os.environ)
        receipt_path = Path(args.receipt).resolve()
        receipt = publish_and_roundtrip(
            args.bundle_dir,
            args.dataset_slug,
            receipt_path,
            attempt_path=receipt_path.with_name("G1_PUBLICATION_ATTEMPT.json"),
            mode=args.mode,
            env=os.environ,
        )
    except Exception as exc:
        if args.mode == "repair":
            raise SystemExit(
                "BLOCKED — EXISTING G1 BUNDLE INVALID OR PUBLICATION REPAIR FAILED; "
                "MANUAL INVALIDATION REQUIRED ONLY IF LOCAL BUNDLE VALIDATION FAILED. "
                f"{type(exc).__name__}: {exc}"
            )
        raise

    print(json.dumps({
        "status": "PASS",
        "mode": args.mode,
        "dataset_slug": args.dataset_slug,
        "g1_seal_sha256": receipt["g1_seal_sha256"],
        "package_sha256": receipt["package_sha256"],
        "roundtrip_verified": receipt["roundtrip_verified"],
        "published_version_number": receipt["published_version_number"],
        "published_version_ref": receipt["published_version_ref"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
