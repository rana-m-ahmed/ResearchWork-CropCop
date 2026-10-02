from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.checkpointing import verify_selected
from cropcop_je.frozen_v1_manifest import load_frozen_v1_rows
from cropcop_je.hashing import sha256_file, sha256_json
from cropcop_je.models import load_pair_initialization
from cropcop_je.persistence import build_store
from cropcop_je.source_state import verify_clean_source
from cropcop_je.surfaces import validate_training_config
from cropcop_je.tracka_v12 import (
    CLASS_MAP_SHA256,
    EXPERIMENT_SPECS,
    MANIFEST_SHA256,
    experiment_config_path,
)
from cropcop_je.tracka_v12_analysis import ROBUSTNESS_CORRUPTIONS, ROBUSTNESS_SEVERITIES
from cropcop_je.tracka_v12_evidence import validate_selected_checkpoint_replay
from cropcop_je.tracka_v12_historical import (
    HISTORICAL_SECONDARY_DIRECT_SPECS,
    HISTORICAL_TRACKA_CLOSURE_SPECS,
    load_historical_secondary_direct_model,
    validate_historical_closure_identity,
)
from cropcop_je.tracka_v12_posttraining import (
    clean_replay,
    efficiency_evidence,
    robustness_sweep,
    write_jsonl,
)
from cropcop_je.tracka_v12_runtime import load_student_and_teacher
from cropcop_je.train import _identity as checkpoint_identity

VAL_COUNT = 16368


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def restore_checkpoint_root(args) -> Path:
    root = Path(args.checkpoint_root).resolve()
    if (root / "checkpoint_index.json").is_file():
        return root
    if not args.durable_store_kind or not args.durable_store_locator:
        raise RuntimeError("checkpoint index absent and no durable store was supplied")
    store = build_store(args.durable_store_kind, args.durable_store_locator)
    restored = store.restore(root, run_id=args.run_id)
    if restored is not True or not (root / "checkpoint_index.json").is_file():
        raise RuntimeError("durable selected-checkpoint restore failed")
    return root


def load_model(args, run_record: dict):
    repo = Path(args.repo_root).resolve()
    experiment_id = str(run_record.get("experiment_id", ""))
    if experiment_id in EXPERIMENT_SPECS:
        config_path = repo / experiment_config_path(experiment_id)
        config = load_json(config_path)
        loaded = load_student_and_teacher(
            repo_root=repo,
            config=config,
            g1a_bundle=args.g1a_bundle,
            expected_source_sha=str(run_record["source_git_commit"]),
        )
        model = loaded["student"]
    elif experiment_id in HISTORICAL_SECONDARY_DIRECT_SPECS:
        model, config = load_historical_secondary_direct_model(
            repo_root=repo,
            run_record=run_record,
            secondary_g1_bundle=getattr(args, "secondary_g1_bundle", "") or None,
        )
    elif experiment_id.startswith("R04-MNV4-DIRECT-"):
        if not args.principal_config or not args.principal_pair_init or not args.principal_pair_evidence:
            raise RuntimeError("historical R04 evidence requires principal config, pair init and pair-init evidence")
        config = load_json(args.principal_config)
        validate_training_config(config)
        if config.get("experiment_id") != experiment_id or config.get("condition") != "direct":
            raise RuntimeError("historical R04 config/run-record identity mismatch")
        if sha256_json(config) != run_record.get("config_sha256"):
            raise RuntimeError("historical R04 config SHA differs from scientific run record")
        pair_evidence = load_json(args.principal_pair_evidence)
        init_sha = str(pair_evidence.get("student_init_sha256", ""))
        if init_sha != run_record.get("student_init_sha256"):
            raise RuntimeError("historical R04 pair-init identity differs from run record")
        model, payload = load_pair_initialization(
            args.principal_pair_init,
            expected_sha256=init_sha,
            pair_id=str(config["pair_id"]),
            seed=int(config["seed"]),
        )
        if payload.get("pretrained_sha256") != run_record.get("pretrained_sha256"):
            raise RuntimeError("historical R04 pretrained identity differs from run record")
    else:
        raise RuntimeError(f"run is not one of the 12 direct Track-A candidate states: {experiment_id}")

    checkpoint_root = restore_checkpoint_root(args)
    expected_selected = run_record.get("artifact_locators", {}).get("selected_checkpoint", {}).get("sha256")
    if not expected_selected:
        expected_selected = run_record.get("result_summary", {}).get("selected_checkpoint_sha256")
    if not expected_selected:
        raise RuntimeError("scientific run record does not bind a selected checkpoint SHA")
    checkpoint_path, payload = verify_selected(
        checkpoint_root,
        expected_identity=checkpoint_identity(run_record),
        expected_sha256=str(expected_selected),
    )
    model.load_state_dict(payload["student"], strict=True)
    selection = payload.get("selection_state", {}).get("best") or {}
    if selection.get("checkpoint_sha256") not in {None, expected_selected}:
        raise RuntimeError("selected checkpoint payload/selection-state hash mismatch")
    return model, config, checkpoint_path, payload, str(expected_selected)


def validation_rows(args):
    return load_frozen_v1_rows(
        args.manifest,
        args.class_map,
        expected_manifest_sha256=MANIFEST_SHA256,
        expected_class_map_sha256=CLASS_MAP_SHA256,
        surface="DS-V1-VAL",
        row_id_column=args.row_id_column,
        path_column=args.path_column,
        split_column=args.split_column,
        label_column=args.label_column,
        class_index_column=args.class_index_column,
        train_split_value=args.train_split_value,
        val_split_value=args.val_split_value,
        expected_count=VAL_COUNT,
    )


def state_robustness_summary(clean_macro_f1: float, cell_summaries: dict) -> dict:
    deltas = []
    monotonic = {}
    for corruption in ROBUSTNESS_CORRUPTIONS:
        corruption_deltas = []
        for severity in ROBUSTNESS_SEVERITIES:
            corrupted = float(cell_summaries[corruption][severity]["validation_macro_f1"])
            delta = (clean_macro_f1 - corrupted) * 100.0
            deltas.append(delta)
            corruption_deltas.append(delta)
        violations = sum(later < prior for prior, later in zip(corruption_deltas, corruption_deltas[1:]))
        monotonic[corruption] = violations
    return {
        "mean_corruption_degradation_pp": sum(deltas) / len(deltas),
        "worst_case_macro_f1_delta_pp": max(deltas),
        "severity_monotonicity_violations_count": sum(monotonic.values()),
        "severity_monotonicity_violations_by_corruption": monotonic,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--analysis-source-git-commit", required=True)
    ap.add_argument("--run-record", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--checkpoint-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--g1a-bundle", default="")
    ap.add_argument("--secondary-g1-bundle", default="")
    ap.add_argument("--principal-config", default="")
    ap.add_argument("--principal-pair-init", default="")
    ap.add_argument("--principal-pair-evidence", default="")
    ap.add_argument("--durable-store-kind", choices=["", "filesystem", "kaggle-dataset"], default="")
    ap.add_argument("--durable-store-locator", default="")
    ap.add_argument("--row-id-column", default="record_key")
    ap.add_argument("--path-column", default="portable_relpath")
    ap.add_argument("--split-column", default="split")
    ap.add_argument("--label-column", default="label")
    ap.add_argument("--class-index-column", default="")
    ap.add_argument("--train-split-value", default="train")
    ap.add_argument("--val-split-value", default="val")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = ap.parse_args()

    import torch

    repo = Path(args.repo_root).resolve()
    output = Path(args.output_dir).resolve()
    analysis_source = args.analysis_source_git_commit.strip()
    if len(analysis_source) != 40:
        raise SystemExit("analysis source Git commit must be a full 40-character SHA")
    verify_clean_source(repo, authorized_source_sha=analysis_source, output_roots=[output])

    run_record = load_json(args.run_record)
    if run_record.get("run_id") != args.run_id or run_record.get("status") != "PASS":
        raise SystemExit("run record is not the requested terminal PASS scientific run")
    experiment_id = str(run_record.get("experiment_id", ""))
    if experiment_id in HISTORICAL_TRACKA_CLOSURE_SPECS:
        historical_errors = validate_historical_closure_identity(run_record)
        if historical_errors:
            raise SystemExit("historical run record is not the canonical sealed Track-A state: " + "; ".join(historical_errors))
    if run_record.get("v1_test_accessed") not in {None, False}:
        raise SystemExit("run record indicates V1-test access")
    if run_record.get("protected_external_surface_accessed") not in {None, False}:
        raise SystemExit("run record indicates protected external-surface access")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA analysis requested but CUDA is unavailable")
    device = torch.device(args.device)

    model, config, checkpoint_path, checkpoint_payload, selected_sha = load_model(args, run_record)
    rows = validation_rows(args)
    private = output / "private_evidence"
    public = output / "public_evidence"
    private.mkdir(parents=True, exist_ok=False)
    public.mkdir(parents=True, exist_ok=False)

    clean_rows, clean_summary = clean_replay(model, rows, args.image_root, device, batch_size=args.batch_size, num_workers=args.num_workers)
    clean_path = private / "selected_checkpoint_validation_predictions.jsonl"
    write_jsonl(clean_path, clean_rows)
    expected_metrics = run_record.get("result_summary", {}).get("selected_metrics") or {}
    replay_gate = validate_selected_checkpoint_replay(clean_summary, expected_metrics)
    if replay_gate["status"] != "PASS":
        raise SystemExit("selected-checkpoint validation replay failed frozen metric tolerance")

    scientific_source = str(run_record["source_git_commit"])
    efficiency = efficiency_evidence(model)
    atomic_write_json(public / "efficiency.json", {
        "schema_version": "1.2",
        "experiment_id": run_record["experiment_id"],
        "source_git_commit": scientific_source,
        "scientific_source_git_commit": scientific_source,
        "analysis_source_git_commit": analysis_source,
        "selected_checkpoint_sha256": selected_sha,
        "training_performed": False,
        "v1_test_accessed": False,
        "external_surface_accessed": False,
        **efficiency,
    })

    cell_rows, cell_summaries = robustness_sweep(model, rows, args.image_root, device, batch_size=args.batch_size, num_workers=args.num_workers)
    private_hashes = {"clean": sha256_file(clean_path)}
    for (corruption, severity), predictions in cell_rows.items():
        path = private / f"robustness__{corruption}__s{severity}.jsonl"
        write_jsonl(path, predictions)
        private_hashes[f"{corruption}/s{severity}"] = sha256_file(path)

    robustness = state_robustness_summary(float(clean_summary["validation_macro_f1"]), cell_summaries)
    robustness_public = {
        "schema_version": "1.2",
        "experiment_id": run_record["experiment_id"],
        "source_git_commit": scientific_source,
        "scientific_source_git_commit": scientific_source,
        "analysis_source_git_commit": analysis_source,
        "selected_checkpoint_sha256": selected_sha,
        "surface": "DS-V1-VAL",
        "training_or_adaptation_performed": False,
        "v1_test_accessed": False,
        "external_surface_accessed": False,
        "clean_summary": clean_summary,
        "cells": cell_summaries,
        "derived": robustness,
        "private_row_evidence_sha256": private_hashes,
    }
    atomic_write_json(public / "robustness_and_replay.json", robustness_public)

    final = {
        "schema_version": "1.2",
        "status": "PASS",
        "experiment_id": run_record["experiment_id"],
        "run_id": run_record["run_id"],
        "source_git_commit": scientific_source,
        "scientific_source_git_commit": scientific_source,
        "analysis_source_git_commit": analysis_source,
        "selected_checkpoint_sha256": selected_sha,
        "selected_checkpoint_file_sha256": sha256_file(checkpoint_path),
        "selected_checkpoint_epoch": int(checkpoint_payload["epoch"]),
        "replay_gate": replay_gate,
        "classwise_pass": len(clean_summary["classwise"]) == 120,
        "robustness_pass": len(private_hashes) == 16,
        "efficiency_pass": all(int(value) > 0 for value in efficiency.values()),
        "training_performed": False,
        "optimizer_state_advanced": False,
        "v1_test_accessed": False,
        "external_surface_accessed": False,
        "public_evidence_sha256": {
            "efficiency.json": sha256_file(public / "efficiency.json"),
            "robustness_and_replay.json": sha256_file(public / "robustness_and_replay.json"),
        },
    }
    atomic_write_json(public / "DIRECT_STATE_EVIDENCE_GATE.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
