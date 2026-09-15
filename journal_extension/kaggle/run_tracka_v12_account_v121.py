from __future__ import annotations

"""Versioned Track-A account runner for the R13 parity v1.2.1 contract.

The historical account runner remains unchanged. This wrapper rebinds only its
G1A bundle preflight to the superseding v1.2.1 seal validator, while retaining
the frozen scheduler, authorization, queue, checkpoint and training behavior.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je import tracka_v12_g1a_v121 as g1a_v121  # noqa: E402
from cropcop_je import tracka_v12_runtime as runtime  # noqa: E402
import run_tracka_v12_account as base  # noqa: E402


def load_and_validate_g1a_bundle_v121(
    bundle_dir: str | Path,
    *,
    expected_source_sha: str | None = None,
):
    previous_validator = runtime.validate_g1a_seal_object
    try:
        runtime.validate_g1a_seal_object = g1a_v121.validate_g1a_seal_object
        return runtime.load_and_validate_g1a_bundle(
            bundle_dir,
            expected_source_sha=expected_source_sha,
        )
    finally:
        runtime.validate_g1a_seal_object = previous_validator


base.load_and_validate_g1a_bundle = load_and_validate_g1a_bundle_v121

if base.RUNNER.name != "run_tracka_v12_training_v121.py":
    raise RuntimeError("v1.2.1 account wrapper is not bound to the versioned training runner")


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
