from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.g1_publication import (
    G1PublicationError,
    create_private_target_if_missing,
    preflight_private_target,
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
        try:
            preflight_private_target(args.dataset_slug, env=os.environ)
        except G1PublicationError:
            if os.environ.get("CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET", "").strip() != "1":
                raise
            create_private_target_if_missing(args.dataset_slug, env=os.environ)
            preflight_private_target(args.dataset_slug, env=os.environ)

        receipt = publish_and_roundtrip(
            args.bundle_dir,
            args.dataset_slug,
            args.receipt,
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
        "current_version_number": receipt["current_version_number"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
