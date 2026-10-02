from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import tracka_v12_g1a as _v12

R13_PARITY_CONTRACT_ID = "TRACKA-A1-R13-PRETRAINED-IDENTITY-V1.2.1"
R13_PARITY_TOLERANCE = 5e-5
R13_PARITY_HISTORICAL_V12_TOLERANCE = 1e-5

G1A_SCHEMA_VERSION = _v12.G1A_SCHEMA_VERSION
R13_PRETRAINED_SHA256 = _v12.R13_PRETRAINED_SHA256
R13_PRETRAINED_BYTES = _v12.R13_PRETRAINED_BYTES
R13_PRETRAINED_COMMIT = _v12.R13_PRETRAINED_COMMIT
R13_HF_REPOSITORY = _v12.R13_HF_REPOSITORY
R13_CTC_MEAN = _v12.R13_CTC_MEAN
R13_CTC_STD = _v12.R13_CTC_STD
R13_NATIVE_MEAN = _v12.R13_NATIVE_MEAN
R13_NATIVE_STD = _v12.R13_NATIVE_STD
R12_CONSUMERS = _v12.R12_CONSUMERS
BASELINE_CONSUMERS = _v12.BASELINE_CONSUMERS
R13_CONSUMERS = _v12.R13_CONSUMERS
TrackAV12G1AError = _v12.TrackAV12G1AError
TEACHER_SHA256 = _v12.TEACHER_SHA256
TIMM_VERSION = _v12.TIMM_VERSION

load_json = _v12.load_json
g1a_seal_hash = _v12.g1a_seal_hash
verify_principal_pair_for_reuse = _v12.verify_principal_pair_for_reuse
build_r12_reuse_evidence = _v12.build_r12_reuse_evidence
load_r13_verified_upstream = _v12.load_r13_verified_upstream
apply_r13_ctc_normalization_equivalence = _v12.apply_r13_ctc_normalization_equivalence
r13_patch_parity_max_abs = _v12.r13_patch_parity_max_abs
deterministic_r13_reset_classifier = _v12.deterministic_r13_reset_classifier


@contextmanager
def _superseding_tolerance():
    """Apply the v1.2.1 bound only for a versioned operation.

    Historical v1.2 module state is restored even if the wrapped operation fails,
    preventing cross-test/process contamination of the immutable 1e-5 contract.
    """
    previous = _v12.R13_PARITY_TOLERANCE
    _v12.R13_PARITY_TOLERANCE = R13_PARITY_TOLERANCE
    try:
        yield
    finally:
        _v12.R13_PARITY_TOLERANCE = previous


def validate_v12_g1a_seal_with_v121_tolerance(seal: dict[str, Any]) -> list[str]:
    """Run historical structural validation under the superseding numeric bound only."""
    with _superseding_tolerance():
        return _v12.validate_g1a_seal_object(seal)


def save_r13_initialization(
    model,
    path: str | Path,
    *,
    experiment_id: str,
    seed: int,
    parity_max_abs: float,
) -> str:
    with _superseding_tolerance():
        return _v12.save_r13_initialization(
            model,
            path,
            experiment_id=experiment_id,
            seed=seed,
            parity_max_abs=parity_max_abs,
        )


def load_r13_initialization(
    path: str | Path,
    *,
    expected_sha256: str,
    experiment_id: str,
    seed: int,
):
    # Contract identity is sealed and validated at the G1A bundle boundary.
    # The immutable initialization payload remains byte-compatible with v1.2
    # and carries the observed parity value itself.
    with _superseding_tolerance():
        return _v12.load_r13_initialization(
            path,
            expected_sha256=expected_sha256,
            experiment_id=experiment_id,
            seed=seed,
        )


def validate_g1a_seal_object(seal: dict[str, Any]) -> list[str]:
    errors = validate_v12_g1a_seal_with_v121_tolerance(seal)
    r13 = seal.get("r13", {})
    if r13.get("parity_contract_id") != R13_PARITY_CONTRACT_ID:
        errors.append("G1A R13 parity contract identity mismatch")
    if float(r13.get("required_max_abs_difference", -1.0)) != R13_PARITY_TOLERANCE:
        errors.append("G1A R13 required parity tolerance mismatch")
    if float(r13.get("historical_v1_2_required_max_abs_difference", -1.0)) != R13_PARITY_HISTORICAL_V12_TOLERANCE:
        errors.append("G1A R13 historical v1.2 tolerance provenance mismatch")
    return errors
