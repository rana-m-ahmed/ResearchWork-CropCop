from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .g1 import CLASS_MAP_SHA256, MANIFEST_SHA256, TEACHER_SHA256
from .hashing import require_sha256, sha256_file

TEACHER_RELATIVE_PATH = Path(
    "cropcop_runs/stage1_dino_tiny_ce_256/checkpoints/best_macro_f1.pt"
)
FINAL_MANIFEST_RELATIVE_PATH = Path("audit/final_manifest.csv")
CLASS_MAP_RELATIVE_PATH = Path("audit/class_to_idx.json")
FACTORY_SPEC = "historical_dino_tiny:build_teacher"
LINEAGE_RELATIVE_PATH = Path(
    "journal_extension/evidence/historical/teacher_stage1/teacher_lineage_manifest.json"
)
HISTORICAL_EVIDENCE_RELATIVE_ROOT = Path(
    "journal_extension/evidence/historical/teacher_stage1"
)
FACTORY_RELATIVE_ROOT = Path("journal_extension/teacher_factory")
DEFAULT_G1_BUNDLE_DIR = Path("/kaggle/working/cropcop-g1")
DEFAULT_G1_PREP_DIR = Path("/kaggle/working/cropcop-g1-prep")
_SLUG = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class G1InputError(RuntimeError):
    pass


@dataclass(frozen=True)
class G1CreationInputs:
    rfdv_root: Path
    final_v1_root: Path
    teacher_checkpoint: Path
    manifest: Path
    class_map: Path
    teacher_factory_root: Path
    teacher_factory: str
    teacher_lineage_manifest: Path
    teacher_historical_evidence_root: Path
    g1_bundle_dir: Path
    g1_prep_dir: Path
    private_dataset_slug: str
    mnv4_pretrained: Path | None


def _env(env: Mapping[str, str], name: str, *, required: bool = False) -> str:
    value = str(env.get(name, "")).strip()
    if required and not value:
        raise G1InputError(f"required G1 input variable missing: {name}")
    return value


def _root(env: Mapping[str, str], name: str) -> Path:
    value = _env(env, name, required=True)
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise G1InputError(f"{name} is not a mounted directory: {root}")
    return root


def _verified_path(
    *,
    canonical: Path,
    expected_sha256: str,
    label: str,
    env: Mapping[str, str],
    legacy_name: str,
) -> Path:
    if not canonical.is_file():
        raise G1InputError(f"{label} missing at exact canonical path: {canonical}")
    require_sha256(canonical, expected_sha256, label)
    legacy = _env(env, legacy_name)
    if not legacy:
        return canonical
    alternate = Path(legacy).expanduser().resolve()
    if not alternate.is_file():
        raise G1InputError(f"{legacy_name} override is not a file: {alternate}")
    require_sha256(alternate, expected_sha256, f"{label} legacy override")
    return alternate


def _require_source_owned_path(
    env: Mapping[str, str],
    name: str,
    canonical: Path,
    *,
    is_dir: bool,
) -> Path:
    canonical = canonical.resolve()
    exists = canonical.is_dir() if is_dir else canonical.is_file()
    if not exists:
        raise G1InputError(f"source-owned G1 path missing: {canonical}")
    raw = _env(env, name)
    if not raw:
        return canonical
    supplied = Path(raw).expanduser().resolve()
    if supplied != canonical:
        raise G1InputError(
            f"{name} conflicts with frozen source-owned path: {supplied} != {canonical}"
        )
    return canonical


def validate_private_slug(slug: str) -> str:
    slug = slug.strip()
    if not _SLUG.fullmatch(slug):
        raise G1InputError(
            "CROPCOP_G1_PRIVATE_DATASET_SLUG must be exactly owner/dataset-slug"
        )
    return slug


def resolve_creation_inputs(
    repo_root: str | Path,
    env: Mapping[str, str] | None = None,
) -> G1CreationInputs:
    env = os.environ if env is None else env
    repo = Path(repo_root).resolve()
    rfdv_root = _root(env, "CROPCOP_RFDV_ROOT")
    final_v1_root = _root(env, "CROPCOP_FINAL_V1_ROOT")

    teacher = _verified_path(
        canonical=rfdv_root / TEACHER_RELATIVE_PATH,
        expected_sha256=TEACHER_SHA256,
        label="historical DINO teacher",
        env=env,
        legacy_name="CROPCOP_TEACHER_CHECKPOINT",
    )
    manifest = _verified_path(
        canonical=final_v1_root / FINAL_MANIFEST_RELATIVE_PATH,
        expected_sha256=MANIFEST_SHA256,
        label="frozen V1 manifest",
        env=env,
        legacy_name="CROPCOP_MANIFEST",
    )
    class_map = _verified_path(
        canonical=final_v1_root / CLASS_MAP_RELATIVE_PATH,
        expected_sha256=CLASS_MAP_SHA256,
        label="frozen 120-way class map",
        env=env,
        legacy_name="CROPCOP_CLASS_MAP",
    )

    factory_root = _require_source_owned_path(
        env,
        "CROPCOP_TEACHER_FACTORY_ROOT",
        repo / FACTORY_RELATIVE_ROOT,
        is_dir=True,
    )
    supplied_factory = _env(env, "CROPCOP_TEACHER_FACTORY")
    if supplied_factory and supplied_factory != FACTORY_SPEC:
        raise G1InputError(
            f"CROPCOP_TEACHER_FACTORY conflicts with frozen factory: "
            f"{supplied_factory!r} != {FACTORY_SPEC!r}"
        )
    lineage = _require_source_owned_path(
        env,
        "CROPCOP_TEACHER_LINEAGE_MANIFEST",
        repo / LINEAGE_RELATIVE_PATH,
        is_dir=False,
    )
    historical_root = _require_source_owned_path(
        env,
        "CROPCOP_TEACHER_HISTORICAL_EVIDENCE_ROOT",
        repo / HISTORICAL_EVIDENCE_RELATIVE_ROOT,
        is_dir=True,
    )

    bundle_raw = _env(env, "CROPCOP_G1_BUNDLE_DIR")
    bundle = (
        Path(bundle_raw).expanduser().resolve()
        if bundle_raw
        else DEFAULT_G1_BUNDLE_DIR.resolve()
    )
    prep_raw = _env(env, "CROPCOP_G1_PREP_DIR")
    prep = (
        Path(prep_raw).expanduser().resolve()
        if prep_raw
        else DEFAULT_G1_PREP_DIR.resolve()
    )
    for label, path in (("G1 bundle", bundle), ("G1 prep", prep)):
        try:
            path.relative_to(repo)
        except ValueError:
            pass
        else:
            raise G1InputError(f"{label} directory must remain outside the Git repository")

    slug = validate_private_slug(_env(env, "CROPCOP_G1_PRIVATE_DATASET_SLUG", required=True))
    mnv4_raw = _env(env, "CROPCOP_MNV4_PRETRAINED")
    mnv4 = Path(mnv4_raw).expanduser().resolve() if mnv4_raw else None
    if mnv4 is not None and not mnv4.is_file():
        raise G1InputError(f"CROPCOP_MNV4_PRETRAINED is not a file: {mnv4}")

    return G1CreationInputs(
        rfdv_root=rfdv_root,
        final_v1_root=final_v1_root,
        teacher_checkpoint=teacher,
        manifest=manifest,
        class_map=class_map,
        teacher_factory_root=factory_root,
        teacher_factory=FACTORY_SPEC,
        teacher_lineage_manifest=lineage,
        teacher_historical_evidence_root=historical_root,
        g1_bundle_dir=bundle,
        g1_prep_dir=prep,
        private_dataset_slug=slug,
        mnv4_pretrained=mnv4,
    )


def input_identity_summary(inputs: G1CreationInputs) -> dict:
    return {
        "teacher_checkpoint_sha256": sha256_file(inputs.teacher_checkpoint),
        "manifest_sha256": sha256_file(inputs.manifest),
        "class_map_sha256": sha256_file(inputs.class_map),
        "teacher_factory": inputs.teacher_factory,
        "teacher_factory_root": str(inputs.teacher_factory_root),
        "teacher_lineage_manifest": str(inputs.teacher_lineage_manifest),
        "private_dataset_slug": inputs.private_dataset_slug,
        "mnv4_pretrained_supplied": inputs.mnv4_pretrained is not None,
    }
