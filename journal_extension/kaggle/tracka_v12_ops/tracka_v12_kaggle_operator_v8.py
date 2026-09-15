from __future__ import annotations

"""Track-A v1.2 active operator authority.

Historical v1/v2/v3 operator files remain untouched. This authority rebinding
pins the active Kaggle operator to the fully qualified science source and scopes
the superseding R13 v1.2.1 G1A validator to this release only.
"""

import sys
from pathlib import Path

import tracka_v12_kaggle_operator as _v1
import tracka_v12_kaggle_operator_v2 as _v2
import tracka_v12_kaggle_operator_v3 as _v3

SCIENCE_SHA_V8 = "56023042e57758591df9babb3438f191dbe10312"
OPERATOR_SCHEMA_VERSION_V8 = "4.6"
R13_PARITY_CONTRACT_ID_V8 = "TRACKA-A1-R13-PRETRAINED-IDENTITY-V1.2.1"
R13_PARITY_REQUIRED_MAX_ABS_V8 = 5e-5

_v1.SCIENCE_SHA = SCIENCE_SHA_V8
_v2.SCIENCE_SHA = SCIENCE_SHA_V8
_v3.SCIENCE_SHA = SCIENCE_SHA_V8

from tracka_v12_kaggle_operator_v3 import *  # noqa: F401,F403,E402

SCIENCE_SHA = SCIENCE_SHA_V8
OPERATOR_SCHEMA_VERSION = OPERATOR_SCHEMA_VERSION_V8


def validate_g1a_bundle_with_science(repo: str | Path, bundle: str | Path) -> dict:
    """Validate a canonical G1A bundle against the versioned v1.2.1 contract."""
    src = Path(repo).resolve() / "journal_extension" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from cropcop_je import tracka_v12_g1a_v121 as g1a_v121
    from cropcop_je import tracka_v12_runtime as runtime

    previous_validator = runtime.validate_g1a_seal_object
    try:
        runtime.validate_g1a_seal_object = g1a_v121.validate_g1a_seal_object
        payload, errors = runtime.load_and_validate_g1a_bundle(
            bundle,
            expected_source_sha=SCIENCE_SHA,
        )
    finally:
        runtime.validate_g1a_seal_object = previous_validator

    if errors:
        raise OperatorError("G1A v1.2.1 bundle validation failed: " + "; ".join(errors))
    if payload.get("status") != "PASS" or payload.get("science_authorized") is not False:
        raise OperatorError("G1A v1.2.1 bundle is not canonical PASS/non-authorizing")
    r13 = payload.get("r13") or {}
    if r13.get("parity_contract_id") != R13_PARITY_CONTRACT_ID_V8:
        raise OperatorError("G1A v1.2.1 parity contract identity mismatch")
    if float(r13.get("required_max_abs_difference", -1.0)) != R13_PARITY_REQUIRED_MAX_ABS_V8:
        raise OperatorError("G1A v1.2.1 parity tolerance mismatch")
    return payload


_v1.validate_g1a_bundle_with_science = validate_g1a_bundle_with_science
_v2.validate_g1a_bundle_with_science = validate_g1a_bundle_with_science
_v3.validate_g1a_bundle_with_science = validate_g1a_bundle_with_science

if _v1.SCIENCE_SHA != SCIENCE_SHA or _v2.SCIENCE_SHA != SCIENCE_SHA or _v3.SCIENCE_SHA != SCIENCE_SHA:
    raise RuntimeError("active process-local science authority rebinding failed")
if _v3.validate_g1a_bundle_with_science is not validate_g1a_bundle_with_science:
    raise RuntimeError("active G1A validator rebinding failed")
