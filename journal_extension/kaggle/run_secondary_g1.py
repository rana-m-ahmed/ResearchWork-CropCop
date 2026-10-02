from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "journal_extension/scripts"
SRC = ROOT / "journal_extension/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cropcop_je.envelope import require_parent_batch
from cropcop_je.secondary import validate_secondary_g1_bundle
from cropcop_je.source_state import verify_clean_source


def req(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required secondary G1 environment variable missing: {name}")
    return value


def execute(script: str, args: list[str]) -> None:
    print(f"+ execute {script} [arguments redacted]", flush=True)
    subprocess.run([sys.executable, "-u", str(SCRIPTS / script), *args], cwd=ROOT, check=True)


def main() -> int:
    require_parent_batch("secondary-g1")
    source_sha = req("CROPCOP_SOURCE_GIT_COMMIT")
    if len(source_sha) != 40:
        raise RuntimeError("CROPCOP_SOURCE_GIT_COMMIT must be exact 40-char SHA")
    output = Path(os.environ.get("CROPCOP_OUTPUT_ROOT", "/kaggle/working/cropcop-secondary-g1")).resolve()
    principal_g1 = Path(req("CROPCOP_PRINCIPAL_G1_INPUT_ROOT")).resolve()
    verify_clean_source(ROOT, authorized_source_sha=source_sha, output_roots=[output, principal_g1])
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("secondary G1 output root must be fresh")
    prep = output / "pretrained"
    bundle = output / "bundle"
    evidence = output / "preseal"
    prep.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)

    prepared = {}
    for key in ("effb0", "cnxtt"):
        prep_record = evidence / f"{key}_preparation.json"
        execute("prepare_torchvision_pretrained.py", [
            "--model-key", key, "--output-dir", str(prep / key), "--record", str(prep_record),
        ])
        record = json.loads(prep_record.read_text(encoding="utf-8"))
        artifact = Path(record["artifact_path"])
        provenance = evidence / f"{key}_provenance.json"
        execute("capture_torchvision_pretrained_provenance.py", [
            "--model-key", key, "--artifact", str(artifact), "--output", str(provenance),
        ])
        prepared[key] = (artifact, provenance)

    execute("seal_secondary_g1.py", [
        "--repo-root", str(ROOT),
        "--authorized-source-sha", source_sha,
        "--manifest", req("CROPCOP_MANIFEST"),
        "--class-map", req("CROPCOP_CLASS_MAP"),
        "--principal-g1-bundle", str(principal_g1),
        "--effb0-pretrained", str(prepared["effb0"][0]),
        "--effb0-provenance", str(prepared["effb0"][1]),
        "--cnxtt-pretrained", str(prepared["cnxtt"][0]),
        "--cnxtt-provenance", str(prepared["cnxtt"][1]),
        "--infra-smoke-evidence", req("CROPCOP_INFRA_SMOKE_EVIDENCE"),
        "--dual-gpu-smoke-evidence", req("CROPCOP_DUAL_GPU_SMOKE_EVIDENCE"),
        "--bundle-dir", str(bundle),
    ])
    seal, errors = validate_secondary_g1_bundle(bundle)
    if errors:
        raise RuntimeError("secondary G1 failed post-seal validation: " + "; ".join(errors))

    receipt = output / "SECONDARY_G1_PUBLICATION_RECEIPT.json"
    execute("publish_secondary_g1.py", [
        "--bundle-dir", str(bundle),
        "--dataset-slug", req("CROPCOP_SECONDARY_G1_PRIVATE_DATASET_SLUG"),
        "--receipt", str(receipt),
    ])
    terminal = {
        "schema_version": "1.0",
        "status": "PASS",
        "source_git_sha": source_sha,
        "secondary_g1_seal_sha256": seal["secondary_g1_seal_sha256"],
        "principal_science_source_sha": seal["principal_science_source_sha"],
        "principal_g1_seal_sha256": seal["principal_g1_seal_sha256"],
        "private_dataset_slug": req("CROPCOP_SECONDARY_G1_PRIVATE_DATASET_SLUG"),
        "roundtrip_verified": True,
        "scientific_result_produced": False,
        "training_performed": False,
        "v1_test_accessed": False,
    }
    (output / "SECONDARY_G1_TERMINAL_EVIDENCE.json").write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(terminal, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
