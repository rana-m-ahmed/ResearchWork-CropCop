from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "journal_extension" / "scripts"


def req(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required G1 environment variable missing: {name}")
    return value


def execute(script: str, args: list[str]) -> None:
    print(f"+ execute {script} [arguments redacted]")
    subprocess.run([sys.executable, str(SCRIPTS / script), *args], cwd=ROOT, check=True)


def main() -> int:
    source_sha = req("CROPCOP_SOURCE_GIT_COMMIT")
    smoke = req("CROPCOP_INFRA_SMOKE_EVIDENCE")
    bundle = Path(req("CROPCOP_G1_BUNDLE_DIR")).resolve()
    output_root = Path(req("CROPCOP_OUTPUT_ROOT")).resolve()
    preseal = output_root / "g1-preseal"
    preseal.mkdir(parents=True, exist_ok=True)

    factory_root = req("CROPCOP_TEACHER_FACTORY_ROOT")
    factory_spec = req("CROPCOP_TEACHER_FACTORY")
    factory_manifest = preseal / "TEACHER_FACTORY_BUNDLE.json"
    provenance = preseal / "MNV4_PRETRAINED_PROVENANCE.json"
    order_evidence = preseal / "TEACHER_CLASS_ORDER_EVIDENCE.json"

    execute("capture_mnv4_pretrained_provenance.py", [
        "--artifact", req("CROPCOP_MNV4_PRETRAINED"),
        "--output", str(provenance),
    ])

    factory_args = [
        "--repo-root", str(ROOT),
        "--source-root", factory_root,
        "--factory-spec", factory_spec,
        "--output", str(factory_manifest),
    ]
    raw_helpers = os.environ.get("CROPCOP_TEACHER_FACTORY_FILES", "").strip()
    if raw_helpers:
        for rel in [x for x in raw_helpers.split(os.pathsep) if x]:
            factory_args += ["--file", rel]
    execute("capture_teacher_factory_bundle.py", factory_args)

    execute("verify_teacher_class_order.py", [
        "--repo-root", str(ROOT),
        "--class-map", req("CROPCOP_CLASS_MAP"),
        "--teacher-checkpoint", req("CROPCOP_TEACHER_CHECKPOINT"),
        "--teacher-factory", factory_spec,
        "--teacher-factory-manifest", str(factory_manifest),
        "--teacher-factory-root", factory_root,
        "--historical-lineage-manifest", req("CROPCOP_TEACHER_LINEAGE_MANIFEST"),
        "--historical-evidence-root", req("CROPCOP_TEACHER_HISTORICAL_EVIDENCE_ROOT"),
        "--output", str(order_evidence),
    ])

    execute("seal_g1.py", [
        "--repo-root", str(ROOT),
        "--authorized-source-sha", source_sha,
        "--manifest", req("CROPCOP_MANIFEST"),
        "--class-map", req("CROPCOP_CLASS_MAP"),
        "--pretrained", req("CROPCOP_MNV4_PRETRAINED"),
        "--pretrained-provenance", str(provenance),
        "--teacher-checkpoint", req("CROPCOP_TEACHER_CHECKPOINT"),
        "--teacher-factory", factory_spec,
        "--teacher-factory-manifest", str(factory_manifest),
        "--teacher-factory-root", factory_root,
        "--teacher-class-order-evidence", str(order_evidence),
        "--infra-smoke-evidence", smoke,
        "--bundle-dir", str(bundle),
    ])

    execute("validate_g1_barrier.py", [
        "--repo-root", str(ROOT),
        "--authorized-source-sha", source_sha,
        "--g1-bundle-dir", str(bundle),
        "--manifest", req("CROPCOP_MANIFEST"),
        "--class-map", req("CROPCOP_CLASS_MAP"),
        "--pretrained", req("CROPCOP_MNV4_PRETRAINED"),
        "--teacher-checkpoint", req("CROPCOP_TEACHER_CHECKPOINT"),
        "--teacher-factory-root", factory_root,
        "--infra-smoke-evidence", smoke,
        "--output", str(output_root / "G1_BARRIER.json"),
    ])

    execute("publish_g1_bundle.py", [
        "--bundle-dir", str(bundle),
        "--dataset-slug", req("CROPCOP_G1_PRIVATE_DATASET_SLUG"),
    ])
    seal = json.loads((bundle / "G1_MODEL_IDENTITY_SEAL.json").read_text(encoding="utf-8"))
    print(json.dumps({
        "status": "PASS",
        "g1_seal_sha256": seal["g1_seal_sha256"],
        "source_git_sha": seal["source_git_sha"],
        "scientific_result_produced": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
