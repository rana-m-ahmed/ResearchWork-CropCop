from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.hashing import sha256_file
from cropcop_je.tracka_v12_g1a_v121 import (
    R13_PARITY_CONTRACT_ID,
    R13_PARITY_TOLERANCE,
    R13_PRETRAINED_BYTES,
    R13_PRETRAINED_SHA256,
    apply_r13_ctc_normalization_equivalence,
    load_r13_verified_upstream,
    r13_patch_parity_max_abs,
)

EXPECTED_CONTRACT_ID = "TRACKA-A1-R13-PRETRAINED-IDENTITY-V1.2.1"
EXPECTED_TOLERANCE = 5e-5
OLD_TOLERANCE = 1e-5


def synthetic_batch():
    import torch

    xs = []
    constants = [(0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0.5), (0.25, 0.5, 0.75), (0.9, 0.1, 0.6)]
    for rgb in constants:
        x = torch.empty((3, 256, 256), dtype=torch.float32)
        for channel, value in enumerate(rgb):
            x[channel].fill_(value)
        xs.append(x)

    axis = torch.linspace(0.0, 1.0, 256, dtype=torch.float32)
    gx = axis.view(1, 1, 256).expand(1, 256, 256)
    gy = axis.view(1, 256, 1).expand(1, 256, 256)
    gradients = [
        gx,
        gy,
        1 - gx,
        1 - gy,
        (gx + gy) / 2,
        gx * gy,
        torch.sqrt(torch.clamp((gx * gx + gy * gy) / 2, 0, 1)),
        torch.abs(gx - gy),
    ]
    for g in gradients:
        xs.append(torch.cat([g, torch.roll(g, 43, 2), torch.roll(g, 71, 1)], dim=0))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(120013)
    for _ in range(24):
        xs.append(torch.rand((3, 256, 256), generator=generator))
    return torch.stack(xs)


def validate(pretrained_path: Path) -> dict:
    errors: list[str] = []
    if R13_PARITY_CONTRACT_ID != EXPECTED_CONTRACT_ID:
        errors.append(f"source parity contract ID drift: {R13_PARITY_CONTRACT_ID}")
    if float(R13_PARITY_TOLERANCE) != EXPECTED_TOLERANCE:
        errors.append(f"source parity tolerance drift: {R13_PARITY_TOLERANCE}")
    if not pretrained_path.is_file():
        errors.append(f"exact pretrained artifact missing: {pretrained_path}")
        return {"schema_version": "1.0", "status": "FAIL", "errors": errors, "science_authorized": False}
    if pretrained_path.stat().st_size != R13_PRETRAINED_BYTES:
        errors.append(f"pretrained byte-count mismatch: {pretrained_path.stat().st_size}")
    if sha256_file(pretrained_path) != R13_PRETRAINED_SHA256:
        errors.append("pretrained SHA-256 mismatch")
    if errors:
        return {"schema_version": "1.0", "status": "FAIL", "errors": errors, "science_authorized": False}

    native = load_r13_verified_upstream(pretrained_path)
    adapted = load_r13_verified_upstream(pretrained_path)
    apply_r13_ctc_normalization_equivalence(adapted)
    batch = synthetic_batch()
    observed = r13_patch_parity_max_abs(native, adapted, batch)

    if observed > EXPECTED_TOLERANCE:
        errors.append(f"exact pretrained parity exceeds v1.2.1 tolerance: {observed} > {EXPECTED_TOLERANCE}")
    if observed <= OLD_TOLERANCE:
        errors.append(f"exact pretrained diagnostic no longer reproduces the v1.2 calibration defect: {observed} <= {OLD_TOLERANCE}")

    return {
        "schema_version": "1.0",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "contract_id": EXPECTED_CONTRACT_ID,
        "pretrained_sha256": R13_PRETRAINED_SHA256,
        "pretrained_bytes": R13_PRETRAINED_BYTES,
        "input_count": int(batch.shape[0]),
        "observed_max_abs_difference": observed,
        "historical_v1_2_tolerance": OLD_TOLERANCE,
        "required_max_abs_difference": EXPECTED_TOLERANCE,
        "historical_v1_2_gate_pass": observed <= OLD_TOLERANCE,
        "v1_2_1_gate_pass": observed <= EXPECTED_TOLERANCE,
        "scientific_metric_computed": False,
        "classifier_prediction_opened": False,
        "v1_test_accessed": False,
        "external_surface_accessed": False,
        "science_authorized": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    report = validate(Path(args.pretrained))
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
