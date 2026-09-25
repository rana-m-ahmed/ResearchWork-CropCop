from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from cropcop_je.trackb_eaai_audit import (
    build_external_manifest,
    read_historical_sha_set,
    write_manifest_csv,
)
from cropcop_je.trackb_eaai_common import (
    CHECKPOINTS,
    CLASS_MAP_SHA256,
    TrackBEAAIError,
    atomic_json,
    discover_bundles,
    load_class_map,
    load_json,
    load_protocol,
    resolve_dir,
    resolve_file,
    validate_materialization_pair,
    sha256_file,
    sha256_json,
)
from cropcop_je.trackb_eaai_eval import (
    aggregate_three_seed,
    environment_record,
    load_model,
    metrics,
    paired_stratified_bootstrap,
    run_three_seed_inference,
    write_confusion_csv,
    write_predictions,
)


DATASET_ROLE = {
    "gvlid_grape": "gvlid_v5",
    "irish_potato": "irish_potato",
}
SEEDS = ("S1", "S2", "S3")


def _validate_source_bundle(bundle, spec: dict[str, Any]) -> dict[str, Any]:
    manifest = bundle.manifest
    if str(manifest.get("doi", "")) != str(spec["doi"]):
        raise TrackBEAAIError(
            f"{bundle.role}: DOI mismatch: {manifest.get('doi')!r}"
        )
    if str(manifest.get("version", "")) != str(spec["version"]):
        raise TrackBEAAIError(
            f"{bundle.role}: version mismatch: {manifest.get('version')!r}"
        )
    if manifest.get("source_checksum_verified") is not True:
        raise TrackBEAAIError(
            f"{bundle.role}: materialized source checksum status is not PASS"
        )
    if int(manifest.get("preflight_original_count", -1)) != int(
        spec["expected_original_images"]
    ):
        raise TrackBEAAIError(
            f"{bundle.role}: materialized original-image count drift"
        )

    support = {
        str(key): int(value)
        for key, value in (manifest.get("preflight_label_support") or {}).items()
    }
    if set(support) != set(spec["mapping"]):
        raise TrackBEAAIError(
            f"{bundle.role}: materialized labels differ from protocol: "
            f"{sorted(support)}"
        )
    expected_support = spec.get("expected_source_support")
    if expected_support is not None:
        expected_support = {
            str(key): int(value)
            for key, value in expected_support.items()
        }
        if support != expected_support:
            raise TrackBEAAIError(
                f"{bundle.role}: materialized class support drift"
            )

    metadata_path = resolve_file(bundle, "source_metadata_record")
    metadata = load_json(metadata_path)
    if (
        str(metadata.get("doi", "")) != str(spec["doi"])
        or str(metadata.get("version", "")) != str(spec["version"])
    ):
        raise TrackBEAAIError(
            f"{bundle.role}: source metadata DOI/version mismatch"
        )
    for field in ("retrieved_at", "license_or_access_text", "source_url"):
        if not str(metadata.get(field, "")).strip():
            raise TrackBEAAIError(
                f"{bundle.role}: source metadata missing {field}"
            )
    return {
        "role": bundle.role,
        "doi": spec["doi"],
        "version": str(spec["version"]),
        "expected_original_images": int(spec["expected_original_images"]),
        "observed_label_support": dict(sorted(support.items())),
        "input_manifest_sha256": sha256_file(bundle.manifest_path),
        "source_metadata_sha256": sha256_file(metadata_path),
    }


def _rows_for_cohort(
    rows: list[dict[str, Any]],
    cohort: str,
) -> list[dict[str, Any]]:
    if cohort == "full_published":
        return list(rows)
    if cohort == "leakage_clean_primary":
        return [
            row
            for row in rows
            if not row["exact_v1_train_val_overlap"]
        ]
    if cohort == "exact_deduplicated_sensitivity":
        return [
            row
            for row in rows
            if (
                not row["exact_v1_train_val_overlap"]
                and row["exact_duplicate_representative"]
            )
        ]
    raise TrackBEAAIError(f"unknown evaluation cohort: {cohort}")


def _filter_prediction_rows(
    prediction_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    keep = {row["row_id"] for row in source_rows}
    selected = [
        row
        for row in prediction_rows
        if row["row_id"] in keep
    ]
    if len(selected) != len(keep):
        raise TrackBEAAIError(
            "cohort prediction filtering lost row identities"
        )
    selected.sort(key=lambda row: row["row_id"])
    return selected


def _write_paper_table(
    path: Path,
    dataset_results: dict[str, Any],
) -> None:
    fields = [
        "dataset",
        "scope",
        "cohort",
        "n",
        "seed",
        "macro_f1",
        "accuracy",
        "balanced_accuracy",
        "out_of_mapped_scope_prediction_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for dataset_id, result in dataset_results.items():
            primary = result["cohorts"]["leakage_clean_primary"]
            for seed in SEEDS:
                metric = primary["per_seed"][seed]
                writer.writerow(
                    {
                        "dataset": dataset_id,
                        "scope": result["scope"],
                        "cohort": "leakage_clean_primary",
                        "n": metric["row_count"],
                        "seed": seed,
                        "macro_f1": metric["macro_f1"],
                        "accuracy": metric["accuracy"],
                        "balanced_accuracy": metric["balanced_accuracy"],
                        "out_of_mapped_scope_prediction_rate": metric[
                            "out_of_mapped_scope_prediction_rate"
                        ],
                    }
                )
            aggregate = primary["aggregate"]
            writer.writerow(
                {
                    "dataset": dataset_id,
                    "scope": result["scope"],
                    "cohort": "leakage_clean_primary",
                    "n": primary["row_count"],
                    "seed": "MEAN",
                    "macro_f1": aggregate["macro_f1"]["mean"],
                    "accuracy": aggregate["accuracy"]["mean"],
                    "balanced_accuracy": aggregate["balanced_accuracy"]["mean"],
                    "out_of_mapped_scope_prediction_rate": aggregate[
                        "out_of_mapped_scope_prediction_rate"
                    ]["mean"],
                }
            )


def _final_manifest(
    output_root: Path,
    *,
    protocol: dict[str, Any],
    audit: dict[str, Any],
    dataset_results: dict[str, Any] | None,
    external_prediction_count: int,
) -> dict[str, Any]:
    files = []
    for path in sorted(
        p
        for p in output_root.rglob("*")
        if p.is_file() and p.name != "TRACKB_FINAL_MANIFEST.json"
    ):
        files.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": "1.0",
        "status": (
            "PASS_TRACKB_EAAI_EXTERNAL_VALIDATION"
            if dataset_results is not None
            else "PASS_TRACKB_EAAI_AUDIT"
        ),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_json(protocol),
        "materialization_id": protocol["materialization"][
            "materialization_id"
        ],
        "v1_test_accessed": False,
        "new_training_performed": False,
        "external_prediction_count": int(external_prediction_count),
        "audit_summary": audit,
        "dataset_results_sha256": (
            sha256_json(dataset_results)
            if dataset_results is not None
            else None
        ),
        "files": files,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Paper-first CropCop Track-B cross-source external validation."
        )
    )
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument(
        "--mode",
        choices=("audit", "all"),
        default="all",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--audit-workers", type=int, default=8)
    parser.add_argument("--loader-workers", type=int, default=4)
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    bundles = discover_bundles(args.input_root)

    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        if any(output_root.iterdir()):
            raise TrackBEAAIError(
                f"refusing to overwrite non-empty output directory: {output_root}"
            )
    else:
        output_root.mkdir(parents=True)

    shutil.copy2(args.protocol, output_root / "protocol.json")

    materialization = validate_materialization_pair(
        args.input_root,
        bundles,
        protocol["materialization"]["materialization_id"],
    )
    atomic_json(output_root / "materialization_receipt.json", materialization)

    core = bundles["core"]
    class_map_path = resolve_file(core, "class_map")
    if sha256_file(class_map_path) != CLASS_MAP_SHA256:
        raise TrackBEAAIError("frozen 120-way class-map SHA-256 mismatch")
    class_map = load_class_map(class_map_path)
    class_names_by_index = {value: key for key, value in class_map.items()}

    historical = bundles["historical_compare"]
    historical_manifest = resolve_file(historical, "historical_manifest")
    historical_sha, historical_count = read_historical_sha_set(
        historical_manifest
    )

    source_authority: dict[str, Any] = {}
    external_rows: dict[str, list[dict[str, Any]]] = {}
    leakage_audit: dict[str, Any] = {
        "schema_version": "1.0",
        "historical_surface": {
            "coverage_scope": "V1_TRAIN_VAL_ONLY",
            "row_count": historical_count,
            "manifest_sha256": sha256_file(historical_manifest),
            "v1_test_accessed": False,
        },
        "datasets": {},
    }

    for dataset_id in ("gvlid_grape", "irish_potato"):
        spec = protocol["datasets"][dataset_id]
        bundle = bundles[DATASET_ROLE[dataset_id]]
        source_authority[dataset_id] = _validate_source_bundle(bundle, spec)

        for target_name in spec["mapping"].values():
            if target_name not in class_map:
                raise TrackBEAAIError(
                    f"{dataset_id}: mapped CropCop class missing: {target_name}"
                )
        if len(set(spec["mapping"].values())) != len(spec["mapping"]):
            raise TrackBEAAIError(
                f"{dataset_id}: mapping is not one-to-one"
            )

        rows, dataset_audit = build_external_manifest(
            dataset_id=dataset_id,
            data_root=resolve_dir(bundle, "data_root"),
            mapping=spec["mapping"],
            expected_count=int(spec["expected_original_images"]),
            expected_support=spec.get("expected_source_support"),
            historical_sha=historical_sha,
            workers=args.audit_workers,
        )
        external_rows[dataset_id] = rows
        leakage_audit["datasets"][dataset_id] = dataset_audit
        write_manifest_csv(
            output_root / dataset_id / "source_manifest.csv",
            rows,
        )

    atomic_json(output_root / "source_authority.json", source_authority)
    atomic_json(output_root / "leakage_audit.json", leakage_audit)

    if args.mode == "audit":
        audit_receipt = {
            "schema_version": "1.0",
            "status": "PASS_TRACKB_EAAI_AUDIT",
            "protocol_id": protocol["protocol_id"],
            "protected_external_prediction_count": 0,
            "v1_test_accessed": False,
            "leakage_audit_sha256": sha256_file(
                output_root / "leakage_audit.json"
            ),
        }
        atomic_json(output_root / "TRACKB_AUDIT.json", audit_receipt)
        final = _final_manifest(
            output_root,
            protocol=protocol,
            audit=leakage_audit,
            dataset_results=None,
            external_prediction_count=0,
        )
        atomic_json(output_root / "TRACKB_FINAL_MANIFEST.json", final)
        print(json.dumps(final, indent=2, sort_keys=True))
        return 0

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise TrackBEAAIError(
            "CUDA device requested but unavailable; enable a Kaggle GPU"
        )

    atomic_json(
        output_root / "environment.json",
        environment_record(args.device),
    )

    checkpoint_paths = {
        seed: resolve_file(core, f"r07_{seed.lower()}")
        for seed in SEEDS
    }
    for seed, checkpoint in checkpoint_paths.items():
        if sha256_file(checkpoint) != CHECKPOINTS[seed]:
            raise TrackBEAAIError(f"{seed} checkpoint binding mismatch")

    dataset_results: dict[str, Any] = {}
    total_predictions = 0

    for dataset_id in ("gvlid_grape", "irish_potato"):
        print(f"\n=== Track B EAAI :: {dataset_id} ===", flush=True)
        spec = protocol["datasets"][dataset_id]
        bundle = bundles[DATASET_ROLE[dataset_id]]
        rows = external_rows[dataset_id]
        data_root = resolve_dir(bundle, "data_root")
        mapped_indices = {
            int(class_map[target_name])
            for target_name in spec["mapping"].values()
        }

        cohort_rows = {
            cohort: _rows_for_cohort(rows, cohort)
            for cohort in (
                "full_published",
                "leakage_clean_primary",
                "exact_deduplicated_sensitivity",
            )
        }
        for cohort, values in cohort_rows.items():
            if not values:
                raise TrackBEAAIError(
                    f"{dataset_id}: cohort {cohort} is empty"
                )

        print(
            f"--- loading frozen S1/S2/S3 models for {dataset_id}",
            flush=True,
        )
        models = {
            seed: load_model(
                checkpoint_paths[seed],
                seed,
                args.device,
            )
            for seed in SEEDS
        }
        predictions = run_three_seed_inference(
            models,
            rows,
            data_root,
            class_map,
            device=args.device,
            batch_size=args.batch_size,
            workers=args.loader_workers,
        )
        for seed in SEEDS:
            prediction_rows = predictions[seed]
            total_predictions += len(prediction_rows)
            write_predictions(
                output_root
                / dataset_id
                / f"{seed}_predictions.csv",
                prediction_rows,
                mapped_indices,
            )
        del models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cohort_results: dict[str, Any] = {}
        for cohort, cohort_source_rows in cohort_rows.items():
            per_seed: dict[str, Any] = {}
            filtered_by_seed: dict[str, list[dict[str, Any]]] = {}
            for seed in SEEDS:
                filtered = _filter_prediction_rows(
                    predictions[seed],
                    cohort_source_rows,
                )
                filtered_by_seed[seed] = filtered
                per_seed[seed] = metrics(filtered, mapped_indices)

            result = {
                "row_count": len(cohort_source_rows),
                "per_seed": per_seed,
                "aggregate": aggregate_three_seed(per_seed),
            }
            if cohort == "leakage_clean_primary":
                result["bootstrap_macro_f1"] = paired_stratified_bootstrap(
                    filtered_by_seed,
                    labels=mapped_indices,
                    replicates=int(protocol["bootstrap"]["replicates"]),
                    rng_seed=int(protocol["bootstrap"]["seed"]),
                )
                for seed in SEEDS:
                    write_confusion_csv(
                        output_root
                        / dataset_id
                        / f"{seed}_primary_confusion_matrix.csv",
                        per_seed[seed],
                        class_names_by_index,
                    )
            cohort_results[cohort] = result
            atomic_json(
                output_root
                / dataset_id
                / f"metrics_{cohort}.json",
                result,
            )

        dataset_results[dataset_id] = {
            "scope": spec["scope"],
            "mapping": spec["mapping"],
            "mapped_class_indices": sorted(mapped_indices),
            "cohorts": cohort_results,
        }

    _write_paper_table(output_root / "paper_table.csv", dataset_results)
    atomic_json(output_root / "summary.json", dataset_results)

    final = _final_manifest(
        output_root,
        protocol=protocol,
        audit=leakage_audit,
        dataset_results=dataset_results,
        external_prediction_count=total_predictions,
    )
    atomic_json(output_root / "TRACKB_FINAL_MANIFEST.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
