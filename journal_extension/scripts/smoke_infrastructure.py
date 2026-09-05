from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.checkpointing import recover_latest, save_torch_checkpoint
from cropcop_je.hashing import sha256_json
from cropcop_je.persistence import build_store
from cropcop_je.publication import publish_to_github_branch
from cropcop_je.session import SessionBudget
from cropcop_je.source_state import verify_clean_source


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--authorized-source-sha", required=True)
    ap.add_argument("--synthetic-bundle-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--durable-store-kind", choices=["kaggle-dataset", "filesystem"], required=True)
    ap.add_argument("--durable-store-locator", required=True)
    ap.add_argument("--allow-filesystem-dev", action="store_true")
    args = ap.parse_args()

    if args.durable_store_kind == "filesystem" and not args.allow_filesystem_dev:
        raise SystemExit("production smoke requires the chosen Kaggle private-dataset backend")
    if not os.environ.get("CROPCOP_GITHUB_TOKEN"):
        raise SystemExit("infrastructure smoke requires private Git/evidence publication credential")
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        raise SystemExit("infrastructure smoke requires Kaggle API credentials")

    repo = Path(args.repo_root).resolve()
    output = Path(args.output_dir).resolve()
    synthetic_bundle = Path(args.synthetic_bundle_dir).resolve()
    if not synthetic_bundle.is_dir() or not any(synthetic_bundle.iterdir()):
        raise SystemExit("synthetic/unprotected smoke bundle is missing or empty")
    source = verify_clean_source(
        repo,
        authorized_source_sha=args.authorized_source_sha,
        output_roots=[output, synthetic_bundle],
    )
    budget = SessionBudget.from_environment(require_global_clock=True)
    if not budget.can_start_phase(900, estimated_checkpoint_seconds=60, estimated_sync_seconds=180):
        raise SystemExit("not enough notebook-global time remains for infrastructure smoke + finalization")

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("real Kaggle infrastructure smoke requires CUDA")
    random.seed(99173)
    np.random.seed(99173)
    torch.manual_seed(99173)
    torch.cuda.manual_seed_all(99173)

    device = torch.device("cuda")
    model = torch.nn.Linear(16, 4).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    run_id = f"INFRA-SMOKE-{args.authorized_source_sha[:12]}"
    identity = {
        "experiment_id": "INFRA-SMOKE",
        "authority_id": "NON_SCIENTIFIC_INFRASTRUCTURE",
        "source_git_commit": args.authorized_source_sha,
        "lane_id": os.environ.get("CROPCOP_LANE"),
        "synthetic_only": True,
    }
    root = output / run_id / "private_checkpoints"
    root.mkdir(parents=True, exist_ok=True)

    step = 0
    for _ in range(3):
        x = torch.randn(16, 16, device=device)
        y = torch.randint(0, 4, (16,), device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.cross_entropy(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1

    payload = {
        "schema_version": "smoke-1.0",
        "identity": identity,
        "identity_sha256": sha256_json(identity),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "student": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": {"smoke_seed": 99173},
        "epoch": 0,
        "batch_in_epoch": step,
        "optimizer_step": step,
        "examples_seen": 48,
        "data_order_state": {"synthetic": True, "next_batch_in_epoch": step},
        "selection_state": {"history": [], "best": None},
    }
    ref, save_seconds = save_torch_checkpoint(root, kind="latest", payload=payload, expected_identity=identity)

    store = build_store(args.durable_store_kind, args.durable_store_locator)
    sync = store.sync(root, run_id=run_id, segment_id="smoke-segment-1")
    shutil.rmtree(root)
    restored = store.restore(root, run_id=run_id)
    if not restored:
        raise SystemExit("durable smoke restore did not recover the checkpoint")
    _path, recovered, event = recover_latest(root, expected_identity=identity)
    model.load_state_dict(recovered["student"], strict=True)
    optimizer.load_state_dict(recovered["optimizer"])

    x = torch.randn(16, 16, device=device)
    y = torch.randint(0, 4, (16,), device=device)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
    resumed_step = int(recovered["optimizer_step"]) + 1
    if resumed_step != 4:
        raise SystemExit("smoke resume did not advance the expected optimizer step")

    evidence = {
        "schema_version": "1.0",
        "status": "PASS",
        "run_id": run_id,
        "scientific": False,
        "synthetic_unprotected_data_only": True,
        "source_git_sha": args.authorized_source_sha,
        "source_state": source,
        "notebook_session": budget.snapshot(),
        "cuda_device": torch.cuda.get_device_name(0),
        "checkpoint_sha256": ref.sha256,
        "checkpoint_save_seconds": save_seconds,
        "durable_backend": args.durable_store_kind,
        "durable_sync_status": sync.status,
        "restore_success": True,
        "resume_success": True,
        "resumed_optimizer_step": resumed_step,
        "recovery_candidate": event.get("candidate"),
        "secret_retrieval_proved_without_value_disclosure": True,
    }
    ev_path = output / f"{run_id}.json"
    atomic_write_json(ev_path, evidence)
    branch = publish_to_github_branch(
        repo_dir=repo,
        source_git_sha=args.authorized_source_sha,
        run_id=run_id,
        files=[ev_path],
    )
    evidence["public_safe_evidence_branch"] = branch
    atomic_write_json(ev_path, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
