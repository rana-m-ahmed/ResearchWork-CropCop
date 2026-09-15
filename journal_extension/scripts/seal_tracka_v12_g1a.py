from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.data import deterministic_pad_resize
from cropcop_je.frozen_v1_manifest import load_frozen_v1_rows
from cropcop_je.g1 import (
    validate_dependency_environment,
    validate_dependency_lock_object,
    validate_g1_seal_object,
)
from cropcop_je.hashing import require_sha256, sha256_file, sha256_json
from cropcop_je.secondary import (
    BASELINE_SPECS,
    TORCHVISION_VERSION,
    create_baseline_from_pretrained,
    save_baseline_initialization,
    validate_torchvision_provenance,
)
from cropcop_je.source_state import verify_clean_source
from cropcop_je.tracka_v12 import (
    AUTHORITY_ID,
    CLASS_MAP_SHA256,
    MANIFEST_SHA256,
    SEEDS,
    TEACHER_SHA256,
    validate_materialized_configs,
)
from cropcop_je.tracka_v12_g1a import (
    BASELINE_CONSUMERS,
    R12_CONSUMERS,
    R13_CONSUMERS,
    R13_HF_REPOSITORY,
    R13_MODEL_ID,
    R13_PARITY_TOLERANCE,
    R13_PRETRAINED_BYTES,
    R13_PRETRAINED_COMMIT,
    R13_PRETRAINED_SHA256,
    TIMM_VERSION,
    apply_r13_ctc_normalization_equivalence,
    build_r12_reuse_evidence,
    deterministic_r13_reset_classifier,
    g1a_seal_hash,
    load_r13_verified_upstream,
    r13_patch_parity_max_abs,
    save_r13_initialization,
    validate_g1a_seal_object,
    verify_principal_pair_for_reuse,
)

VAL_COUNT = 16368


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def copy_exact(source: str | Path, target: Path, *, expected_sha256: str, label: str) -> None:
    source = Path(source)
    require_sha256(source, expected_sha256, label)
    if target.exists():
        raise SystemExit(f"refuse to overwrite existing sealed {label}: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    require_sha256(target, expected_sha256, f"sealed {label}")
    if target.stat().st_size != source.stat().st_size:
        raise SystemExit(f"sealed {label} byte count changed during copy")


def verify_principal_g1(bundle: Path) -> dict:
    seal_path = bundle / "G1_MODEL_IDENTITY_SEAL.json"
    if not seal_path.is_file():
        raise SystemExit("principal G1 seal missing")
    seal = load_json(seal_path)
    errors = validate_g1_seal_object(seal)
    if errors:
        raise SystemExit("principal G1 seal invalid: " + "; ".join(errors))
    for label in ("S2", "S3"):
        verify_principal_pair_for_reuse(bundle, seed_label=label)
    teacher = seal.get("teacher", {})
    teacher_path = bundle / "private" / str(teacher.get("artifact_basename", ""))
    require_sha256(teacher_path, TEACHER_SHA256, "principal historical teacher")
    return seal


def deterministic_parity_rows(manifest: str | Path, class_map: str | Path):
    rows = load_frozen_v1_rows(
        manifest,
        class_map,
        expected_manifest_sha256=MANIFEST_SHA256,
        expected_class_map_sha256=CLASS_MAP_SHA256,
        surface="DS-V1-VAL",
        expected_count=VAL_COUNT,
    )
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"TRACKA-A1-R13-G1A-PARITY|{row.stable_row_id}".encode("utf-8")
        ).hexdigest(),
    )[:8]


def build_r13_parity_batch(manifest: str | Path, class_map: str | Path, image_root: str | Path):
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.transforms import functional as TF

    tensors = []
    constant_values = [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.5, 0.5),
        (0.25, 0.5, 0.75),
        (0.9, 0.1, 0.6),
    ]
    for rgb in constant_values:
        x = torch.empty((3, 256, 256), dtype=torch.float32)
        for c, value in enumerate(rgb):
            x[c].fill_(float(value))
        tensors.append(x)

    axis = torch.linspace(0.0, 1.0, 256, dtype=torch.float32)
    gx = axis.view(1, 1, 256).expand(1, 256, 256)
    gy = axis.view(1, 256, 1).expand(1, 256, 256)
    gradients = [
        gx, gy, 1.0 - gx, 1.0 - gy,
        (gx + gy) / 2.0,
        (gx * gy),
        torch.sqrt(torch.clamp((gx * gx + gy * gy) / 2.0, 0.0, 1.0)),
        torch.abs(gx - gy),
    ]
    for g in gradients:
        tensors.append(torch.cat([g, torch.roll(g, shifts=43, dims=2), torch.roll(g, shifts=71, dims=1)], dim=0))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(120013)
    for _ in range(16):
        tensors.append(torch.rand((3, 256, 256), generator=generator, dtype=torch.float32))

    selected_rows = deterministic_parity_rows(manifest, class_map)
    root = Path(image_root).resolve()
    row_ids = []
    row_byte_sha256 = []
    for row in selected_rows:
        path = (root / row.relative_path).resolve()
        if root not in path.parents and path != root:
            raise SystemExit(f"R13 parity path escapes image root: {row.relative_path}")
        if not path.is_file():
            raise SystemExit(f"R13 parity image missing: {row.relative_path}")
        row_ids.append(row.stable_row_id)
        row_byte_sha256.append(sha256_file(path))
        with Image.open(path) as image:
            resized = deterministic_pad_resize(image, 256)
            tensors.append(TF.pil_to_tensor(resized).to(torch.float32).div_(255.0))

    batch = torch.stack(tensors, dim=0)
    if batch.shape != (40, 3, 256, 256):
        raise SystemExit(f"R13 parity batch shape drift: {tuple(batch.shape)}")
    return batch, row_ids, row_byte_sha256


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--authorized-source-sha", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--principal-g1-bundle", required=True)
    ap.add_argument("--effb0-pretrained", required=True)
    ap.add_argument("--effb0-provenance", required=True)
    ap.add_argument("--cnxtt-pretrained", required=True)
    ap.add_argument("--cnxtt-provenance", required=True)
    ap.add_argument("--r13-pretrained", required=True)
    ap.add_argument("--dependency-lock", default="journal_extension/locks/execution_dependency_lock.json")
    ap.add_argument("--bundle-dir", required=True)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    bundle = Path(args.bundle_dir).resolve()
    if (bundle / "TRACKA_V12_G1A_SEAL.json").exists():
        raise SystemExit("Track-A v1.2 G1A target already sealed; immutable bundle cannot be overwritten")
    verify_clean_source(repo, authorized_source_sha=args.authorized_source_sha, output_roots=[bundle])

    config_errors = validate_materialized_configs(repo)
    if config_errors:
        raise SystemExit("Track-A v1.2 config contract failed: " + "; ".join(config_errors))

    dep_path = Path(args.dependency_lock)
    if not dep_path.is_absolute():
        dep_path = repo / dep_path
    dep = load_json(dep_path)
    dep_errors = validate_dependency_lock_object(dep) + validate_dependency_environment(dep)
    if dep_errors:
        raise SystemExit("dependency lock validation failed: " + "; ".join(dep_errors))

    require_sha256(args.manifest, MANIFEST_SHA256, "frozen V1 manifest")
    require_sha256(args.class_map, CLASS_MAP_SHA256, "frozen class map")
    principal_root = Path(args.principal_g1_bundle).resolve()
    principal_seal = verify_principal_g1(principal_root)

    baseline_inputs = {
        "effb0": (Path(args.effb0_pretrained), Path(args.effb0_provenance)),
        "cnxtt": (Path(args.cnxtt_pretrained), Path(args.cnxtt_provenance)),
    }
    baseline_provenance = {}
    for key, (artifact, provenance_path) in baseline_inputs.items():
        provenance = load_json(provenance_path)
        errors = validate_torchvision_provenance(provenance, model_key=key, artifact_path=artifact)
        if errors:
            raise SystemExit(f"{key} pretrained provenance invalid: " + "; ".join(errors))
        baseline_provenance[key] = provenance

    r13_upstream_path = Path(args.r13_pretrained)
    require_sha256(r13_upstream_path, R13_PRETRAINED_SHA256, "R13 upstream safetensors")
    if r13_upstream_path.stat().st_size != R13_PRETRAINED_BYTES:
        raise SystemExit("R13 upstream safetensors byte-count mismatch")

    private = bundle / "private"
    evidence = bundle / "evidence"
    private.mkdir(parents=True, exist_ok=False)
    evidence.mkdir(parents=True, exist_ok=False)

    r12_rows = {}
    for label in ("S2", "S3"):
        pair, pair_evidence, source_binary = verify_principal_pair_for_reuse(principal_root, seed_label=label)
        target = private / f"PAIR_INIT_{label}.pt"
        copy_exact(source_binary, target, expected_sha256=pair["sha256"], label=f"MNV4 {label} pair initialization")
        reuse_evidence = build_r12_reuse_evidence(
            seed_label=label,
            pair=pair,
            principal_pair_evidence=pair_evidence,
            copied_binary=target,
            source_git_sha=args.authorized_source_sha,
            principal_g1_seal_sha256=principal_seal["g1_seal_sha256"],
        )
        atomic_write_json(evidence / f"R12_REUSED_PAIR_{label}.json", reuse_evidence)
        r12_rows[label] = {
            "pair_id": reuse_evidence["pair_id"],
            "seed": reuse_evidence["seed"],
            "student_init_sha256": reuse_evidence["student_init_sha256"],
            "student_init_bytes": reuse_evidence["student_init_bytes"],
            "student_init_basename": target.name,
            "reuse_evidence_sha256": sha256_json(reuse_evidence),
            "authorized_consumers": R12_CONSUMERS[label],
            "initialization_bytes_reused_exactly": True,
        }

    principal_teacher = principal_seal["teacher"]
    teacher_target = private / "DINO_TEACHER.pt"
    copy_exact(
        principal_root / "private" / principal_teacher["artifact_basename"],
        teacher_target,
        expected_sha256=TEACHER_SHA256,
        label="historical DINO teacher",
    )
    teacher_evidence_hashes = {}
    for basename in (
        "DINO_TEACHER.json",
        "TEACHER_FACTORY_BUNDLE.json",
        "TEACHER_CLASS_ORDER_EVIDENCE.json",
        "TEACHER_CANONICAL_STATE_EVIDENCE.json",
        "TEACHER_ADAPTER_PARITY_EVIDENCE.json",
        "MNV4_PRETRAINED_PROVENANCE.json",
    ):
        source = principal_root / "evidence" / basename
        if not source.is_file():
            raise SystemExit(f"principal G1 evidence missing: {basename}")
        shutil.copyfile(source, evidence / basename)
        teacher_evidence_hashes[basename] = sha256_json(load_json(evidence / basename))

    baseline_rows = {"effb0": {}, "cnxtt": {}}
    for key, spec in BASELINE_SPECS.items():
        source_artifact, _ = baseline_inputs[key]
        provenance = baseline_provenance[key]
        pretrained_target = private / spec["pretrained_basename"]
        copy_exact(
            source_artifact,
            pretrained_target,
            expected_sha256=provenance["artifact_sha256"],
            label=f"{key} official pretrained",
        )
        sealed_provenance = dict(provenance)
        sealed_provenance["sealed_basename"] = pretrained_target.name
        sealed_provenance["sealed_sha256"] = sha256_file(pretrained_target)
        sealed_provenance["sealed_bytes"] = pretrained_target.stat().st_size
        atomic_write_json(evidence / spec["pretrained_evidence_basename"], sealed_provenance)

        for label in ("S2", "S3"):
            experiment_id = BASELINE_CONSUMERS[key][label]
            seed = SEEDS[label]
            model = create_baseline_from_pretrained(pretrained_target, model_key=key, seed=seed, num_classes=120)
            init_target = private / f"{key.upper()}_INIT_{label}.pt"
            init_sha = save_baseline_initialization(
                model,
                init_target,
                model_key=key,
                seed=seed,
                pretrained_sha256=provenance["artifact_sha256"],
                authorized_consumers=[experiment_id],
            )
            init_evidence = {
                "schema_version": "1.0",
                "model_key": key,
                "model_name": spec["model_name"],
                "torchvision_version": TORCHVISION_VERSION,
                "weight_enum": spec["weight_enum"],
                "seed_label": label,
                "seed": seed,
                "num_classes": 120,
                "pretrained_sha256": provenance["artifact_sha256"],
                "init_sha256": init_sha,
                "init_bytes": init_target.stat().st_size,
                "init_basename": init_target.name,
                "authorized_consumers": [experiment_id],
                "generated_under_source_sha": args.authorized_source_sha,
            }
            atomic_write_json(evidence / f"{key.upper()}_INIT_{label}.json", init_evidence)
            baseline_rows[key][label] = {
                "seed": seed,
                "pretrained_sha256": provenance["artifact_sha256"],
                "pretrained_bytes": pretrained_target.stat().st_size,
                "pretrained_basename": pretrained_target.name,
                "pretrained_provenance_sha256": sha256_json(sealed_provenance),
                "init_sha256": init_sha,
                "init_bytes": init_target.stat().st_size,
                "init_basename": init_target.name,
                "init_evidence_sha256": sha256_json(init_evidence),
                "authorized_consumers": [experiment_id],
            }

    r13_target = private / "R13_PRETRAINED.safetensors"
    copy_exact(r13_upstream_path, r13_target, expected_sha256=R13_PRETRAINED_SHA256, label="R13 upstream safetensors")
    native_r13 = load_r13_verified_upstream(r13_target)
    adapted_r13 = load_r13_verified_upstream(r13_target)
    apply_r13_ctc_normalization_equivalence(adapted_r13)
    parity_batch, parity_row_ids, parity_row_hashes = build_r13_parity_batch(args.manifest, args.class_map, args.image_root)
    parity_max_abs = r13_patch_parity_max_abs(native_r13, adapted_r13, parity_batch)
    if parity_max_abs > R13_PARITY_TOLERANCE:
        raise SystemExit(
            f"R13 normalization-equivalence parity failed: {parity_max_abs} > {R13_PARITY_TOLERANCE}"
        )
    parity_evidence = {
        "schema_version": "1.0",
        "status": "PASS",
        "model_id": R13_MODEL_ID,
        "upstream_sha256": R13_PRETRAINED_SHA256,
        "upstream_bytes": R13_PRETRAINED_BYTES,
        "synthetic_input_count": 32,
        "prediction_blind_validation_input_count": 8,
        "total_input_count": 40,
        "validation_selection_rule": "ascending SHA256('TRACKA-A1-R13-G1A-PARITY|' + stable_row_id), first 8 DS-V1-VAL rows",
        "validation_row_ids": parity_row_ids,
        "validation_image_byte_sha256": parity_row_hashes,
        "max_abs_difference": parity_max_abs,
        "required_max_abs_difference": R13_PARITY_TOLERANCE,
        "scientific_metric_computed": False,
        "classifier_prediction_opened": False,
        "v1_test_accessed": False,
        "external_surface_accessed": False,
        "generated_under_source_sha": args.authorized_source_sha,
    }
    atomic_write_json(evidence / "R13_NORMALIZATION_PARITY.json", parity_evidence)

    r13_rows = {}
    adapted_base_state = {key: value.detach().clone() for key, value in adapted_r13.state_dict().items()}
    for label in ("S1", "S2", "S3"):
        experiment_id = R13_CONSUMERS[label]
        seed = SEEDS[label]
        model = load_r13_verified_upstream(r13_target)
        apply_r13_ctc_normalization_equivalence(model)
        for key, value in adapted_base_state.items():
            if key in model.state_dict():
                model.state_dict()[key].copy_(value)
        deterministic_r13_reset_classifier(model, seed=seed, num_classes=120)
        init_target = private / f"R13_VIT_DLITTLE_INIT_{label}.pt"
        init_sha = save_r13_initialization(
            model,
            init_target,
            experiment_id=experiment_id,
            seed=seed,
            parity_max_abs=parity_max_abs,
        )
        init_evidence = {
            "schema_version": "1.0",
            "model_family": "vit_dlittle_diff",
            "model_id": R13_MODEL_ID,
            "timm_version": TIMM_VERSION,
            "seed_label": label,
            "seed": seed,
            "num_classes": 120,
            "pretrained_sha256": R13_PRETRAINED_SHA256,
            "normalization_parity_evidence_sha256": sha256_json(parity_evidence),
            "normalization_parity_max_abs": parity_max_abs,
            "init_sha256": init_sha,
            "init_bytes": init_target.stat().st_size,
            "init_basename": init_target.name,
            "authorized_consumers": [experiment_id],
            "generated_under_source_sha": args.authorized_source_sha,
        }
        atomic_write_json(evidence / f"R13_VIT_DLITTLE_INIT_{label}.json", init_evidence)
        r13_rows[label] = {
            "seed": seed,
            "init_sha256": init_sha,
            "init_bytes": init_target.stat().st_size,
            "init_basename": init_target.name,
            "init_evidence_sha256": sha256_json(init_evidence),
            "authorized_consumers": [experiment_id],
        }

    seal = {
        "schema_version": "1.0",
        "status": "PASS",
        "science_authorized": False,
        "authority": {
            "id": AUTHORITY_ID,
            "comprehensive_execution_authority": "journal_extension/locks/track_a_comprehensive_execution_authority_v1_2_1.json",
        },
        "source_git_sha": args.authorized_source_sha,
        "principal_g1_seal_sha256": principal_seal["g1_seal_sha256"],
        "dataset": {
            "manifest_sha256": MANIFEST_SHA256,
            "class_map_sha256": CLASS_MAP_SHA256,
            "v1_test_accessed": False,
            "external_surface_accessed": False,
        },
        "r12_reused_pairs": r12_rows,
        "teacher": {
            "checkpoint_sha256": TEACHER_SHA256,
            "checkpoint_bytes": teacher_target.stat().st_size,
            "artifact_basename": teacher_target.name,
            "reused_exactly_from_principal_g1": True,
            "evidence_sha256": teacher_evidence_hashes,
        },
        "baselines": baseline_rows,
        "r13": {
            "model_id": R13_MODEL_ID,
            "timm_version": TIMM_VERSION,
            "hf_repository": R13_HF_REPOSITORY,
            "hf_model_add_commit": R13_PRETRAINED_COMMIT,
            "pretrained_sha256": R13_PRETRAINED_SHA256,
            "pretrained_bytes": R13_PRETRAINED_BYTES,
            "pretrained_basename": r13_target.name,
            "parity_max_abs": parity_max_abs,
            "parity_evidence_sha256": sha256_json(parity_evidence),
            "states": r13_rows,
        },
        "dependency_lock_sha256": dep["dependency_lock_sha256"],
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    seal["g1a_seal_sha256"] = g1a_seal_hash(seal)
    errors = validate_g1a_seal_object(seal)
    if errors:
        raise SystemExit("Track-A v1.2 G1A self-validation failed: " + "; ".join(errors))
    atomic_write_json(bundle / "TRACKA_V12_G1A_SEAL.json", seal)
    print(json.dumps(seal, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())