from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .atomic_io import atomic_write_json
from .g1 import validate_mounted_g1
from .source_state import verify_clean_source


@dataclass(frozen=True)
class G1BarrierInputs:
    repo_root: Path
    authorized_source_sha: str
    bundle_dir: Path
    manifest: Path
    class_map: Path
    infra_smoke_evidence: Path
    dual_gpu_smoke_evidence: Path
    dependency_lock: Path


def _required(env: Mapping[str, str], name: str) -> str:
    value = str(env.get(name, "")).strip()
    if not value:
        raise RuntimeError(f"required G1 barrier environment variable missing: {name}")
    return value


def from_environment(
    repo_root: str | Path,
    *,
    authorized_source_sha: str,
    env: Mapping[str, str],
) -> G1BarrierInputs:
    repo = Path(repo_root).resolve()
    return G1BarrierInputs(
        repo_root=repo,
        authorized_source_sha=authorized_source_sha,
        bundle_dir=Path(_required(env, "CROPCOP_G1_BUNDLE_DIR")).resolve(),
        manifest=Path(_required(env, "CROPCOP_MANIFEST")).resolve(),
        class_map=Path(_required(env, "CROPCOP_CLASS_MAP")).resolve(),
        infra_smoke_evidence=Path(_required(env, "CROPCOP_INFRA_SMOKE_EVIDENCE")).resolve(),
        dual_gpu_smoke_evidence=Path(_required(env, "CROPCOP_DUAL_GPU_SMOKE_EVIDENCE")).resolve(),
        dependency_lock=repo / "journal_extension/locks/execution_dependency_lock.json",
    )


def barrier_cli_args(inputs: G1BarrierInputs, *, output: str | Path = "") -> list[str]:
    args = [
        "--repo-root", str(inputs.repo_root),
        "--authorized-source-sha", inputs.authorized_source_sha,
        "--g1-bundle-dir", str(inputs.bundle_dir),
        "--manifest", str(inputs.manifest),
        "--class-map", str(inputs.class_map),
        "--infra-smoke-evidence", str(inputs.infra_smoke_evidence),
        "--dual-gpu-smoke-evidence", str(inputs.dual_gpu_smoke_evidence),
        "--dependency-lock", str(inputs.dependency_lock),
    ]
    if output:
        args += ["--output", str(output)]
    return args


def validate_g1_barrier(inputs: G1BarrierInputs) -> dict:
    bundle = inputs.bundle_dir
    seal_path = bundle / "G1_MODEL_IDENTITY_SEAL.json"
    if not seal_path.is_file():
        return {
            "schema_version": "2.0",
            "status": "FAIL",
            "g1_seal_sha256": None,
            "source_git_sha": None,
            "dependency_lock_sha256": None,
            "source_state": {"status": "FAIL", "error": "G1_MODEL_IDENTITY_SEAL.json is missing"},
            "errors": ["G1_MODEL_IDENTITY_SEAL.json is missing"],
        }
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    try:
        source = verify_clean_source(
            inputs.repo_root,
            authorized_source_sha=inputs.authorized_source_sha,
            output_roots=[bundle],
        )
    except Exception as exc:
        source = {"status": "FAIL", "error": str(exc)}
        errors.append(str(exc))

    pretrained = bundle / "private" / str(
        seal.get("student", {}).get("pretrained", {}).get("basename", "")
    )
    teacher = bundle / "private" / str(
        seal.get("teacher", {}).get("artifact_basename", "")
    )
    evidence = bundle / "evidence"
    errors.extend(
        validate_mounted_g1(
            seal,
            authorized_source_sha=inputs.authorized_source_sha,
            manifest_path=inputs.manifest,
            class_map_path=inputs.class_map,
            pretrained_path=pretrained,
            pretrained_provenance_path=evidence / "MNV4_PRETRAINED_PROVENANCE.json",
            pair_dir=bundle / "private",
            evidence_dir=evidence,
            teacher_checkpoint_path=teacher,
            teacher_factory_manifest_path=evidence / "TEACHER_FACTORY_BUNDLE.json",
            teacher_factory_source_root=inputs.repo_root / "journal_extension/teacher_factory",
            teacher_class_order_evidence_path=evidence / "TEACHER_CLASS_ORDER_EVIDENCE.json",
            teacher_canonical_state_evidence_path=evidence / "TEACHER_CANONICAL_STATE_EVIDENCE.json",
            dependency_lock_path=inputs.dependency_lock,
            infra_smoke_evidence_path=inputs.infra_smoke_evidence,
            dual_gpu_smoke_evidence_path=inputs.dual_gpu_smoke_evidence,
            repo_root=inputs.repo_root,
        )
    )
    return {
        "schema_version": "2.0",
        "status": "PASS" if not errors else "FAIL",
        "g1_seal_sha256": seal.get("g1_seal_sha256"),
        "source_git_sha": seal.get("source_git_sha"),
        "dependency_lock_sha256": seal.get("dependency_lock_sha256"),
        "source_state": source,
        "errors": errors,
    }


def validate_and_write(inputs: G1BarrierInputs, output: str | Path = "") -> dict:
    report = validate_g1_barrier(inputs)
    if output:
        atomic_write_json(output, report)
    return report
