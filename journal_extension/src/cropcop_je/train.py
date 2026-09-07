from __future__ import annotations

import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from .checkpointing import recover_latest, save_torch_checkpoint, verify_selected
from .data import epoch_order_seed
from .evaluate import benchmark_forward, evaluate_classifier
from .hashing import sha256_json
from .models import prelogits_and_logits
from .selection import SelectionState
from .session import SessionBudget


class EpochPermutationSampler(Sampler[Any]):
    def __init__(self, dataset, *, training_seed: int, epoch: int, start_sample: int = 0):
        self.dataset = dataset
        self.epoch = int(epoch)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(epoch_order_seed(training_seed, epoch))
        full_order = torch.randperm(len(dataset), generator=self.generator).tolist()
        if not 0 <= start_sample <= len(full_order):
            raise ValueError(f"invalid resume sample cursor {start_sample} for {len(full_order)} rows")
        self.full_order_sha256 = sha256_json(full_order)
        self.order = full_order[start_sample:]
        self.start_sample = int(start_sample)
        self.generator_state = self.generator.get_state()

    def __iter__(self):
        return iter((index, self.epoch) for index in self.order)

    def __len__(self):
        return len(self.order)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Locked determinism controls for the scientific training path.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _loader(dataset, *, sampler=None, batch_size: int, num_workers: int, shuffle: bool = False):
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": False,
        "shuffle": shuffle if sampler is None else False,
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
    if num_workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(**kwargs)


def build_optimizer(student, projection, ctc):
    new_ids = {id(p) for p in student.get_classifier().parameters()}
    if projection is not None:
        new_ids.update(id(p) for p in projection.parameters())
    groups = {}
    parameters = list(student.named_parameters()) + (list(projection.named_parameters()) if projection is not None else [])
    for name, p in parameters:
        if not p.requires_grad:
            continue
        scope = "new" if id(p) in new_ids else "base"
        decay = not (name.endswith(".bias") or p.ndim <= 1)
        groups.setdefault((scope, decay), []).append(p)
    o = ctc["optimizer"]
    pg = []
    for (scope, decay), ps in groups.items():
        pg.append(
            {
                "params": ps,
                "lr": o["new_parameter_lr"] if scope == "new" else o["backbone_lr"],
                "weight_decay": o["weight_decay"] if decay else 0.0,
            }
        )
    return torch.optim.AdamW(pg, betas=tuple(o["betas"]), eps=o["eps"])


def build_scheduler(opt, total_steps, warmup_fraction, min_fraction):
    warm = max(1, round(total_steps * warmup_fraction))

    def f(step):
        if step < warm:
            return max((step + 1) / warm, 1e-12)
        p = min(1.0, (step - warm + 1) / max(total_steps - warm, 1))
        return min_fraction + 0.5 * (1 - min_fraction) * (1 + math.cos(math.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def kd_loss(s, t, T):
    return (T * T) * F.kl_div(F.log_softmax(s / T, dim=1), F.softmax(t / T, dim=1), reduction="batchmean")


def feature_loss(s, t):
    return (1 - F.cosine_similarity(F.normalize(s, p=2, dim=1), F.normalize(t, p=2, dim=1), dim=1)).mean()


def _identity(r):
    keys = (
        "experiment_id",
        "authority_id",
        "source_git_commit",
        "config_sha256",
        "ctc_v2_sha256",
        "manifest_sha256",
        "class_map_sha256",
        "seed",
        "student_init_sha256",
        "pretrained_sha256",
        "teacher_sha256",
        "teacher_factory_sha256",
        "teacher_factory_bundle_sha256",
        "software_stack_sha256",
        "dependency_lock_sha256",
        "g1_seal_sha256",
        "g2_barrier_sha256",
        "lane_id",
    )
    return {k: r.get(k) for k in keys}


def _checkpoint_payload(*, student, projection, optimizer, scheduler, scaler, epoch, batch_in_epoch, optimizer_step, run_identity, selection_state, examples_seen, data_order_state):
    identity = _identity(run_identity)
    return {
        "schema_version": "2.0",
        "identity": identity,
        "identity_sha256": sha256_json(identity),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "student": student.state_dict(),
        "projection": projection.state_dict() if projection is not None else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "optimizer_step": int(optimizer_step),
        "examples_seen": int(examples_seen),
        "data_order_state": data_order_state,
        "selection_state": selection_state.to_dict(),
    }


def _restore_payload(payload, *, student, projection, optimizer, scheduler, scaler, run_identity):
    if payload["identity"] != _identity(run_identity):
        raise ValueError("resume identity mismatch")
    student.load_state_dict(payload["student"], strict=True)
    if projection is not None:
        if payload["projection"] is None:
            raise ValueError("resume checkpoint lacks teacher projection")
        projection.load_state_dict(payload["projection"], strict=True)
    elif payload["projection"] is not None:
        raise ValueError("direct run cannot resume teacher checkpoint")
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    random.setstate(payload["rng"]["python"])
    np.random.set_state(payload["rng"]["numpy"])
    torch.set_rng_state(payload["rng"]["torch"])
    if torch.cuda.is_available() and payload["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state_all(payload["rng"]["cuda"])


def run_training(
    *,
    student,
    teacher,
    projection,
    train_dataset,
    val_dataset,
    ctc,
    objective,
    run_identity,
    output_dir,
    num_workers=2,
    resume=False,
    max_optimizer_steps=None,
    validation_enabled=True,
    checkpoint_every_steps=250,
    session_budget: SessionBudget | None = None,
    min_free_bytes: int = 5 * 1024**3,
):
    _seed(int(run_identity["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("locked Stage-03R path requires qualified CUDA/FP16 host")
    student.to(device)
    if teacher is not None:
        teacher.to(device).eval()
    if projection is not None:
        projection.to(device)
    micro = ctc["training"]["micro_batch_size"]
    accum = ctc["training"]["gradient_accumulation"]
    validation_batch = int(ctc["training"].get("validation_batch_size", 64))
    epochs = ctc["schedule"]["epochs"]
    batches_per_epoch = math.ceil(len(train_dataset) / micro)
    steps_per_epoch = math.ceil(batches_per_epoch / accum)
    total_steps = epochs * steps_per_epoch
    opt = build_optimizer(student, projection, ctc)
    sched = build_scheduler(opt, total_steps, ctc["schedule"]["warmup_fraction_optimizer_steps"], ctc["schedule"]["minimum_lr_fraction"])
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    start_batch = 0
    step = 0
    examples_seen_total = 0
    selection = SelectionState()
    recovery_events = []
    if resume:
        load_started = time.perf_counter()
        _path, payload, fallback = recover_latest(root, expected_identity=_identity(run_identity))
        checkpoint_load_seconds = time.perf_counter() - load_started
        _restore_payload(payload, student=student, projection=projection, optimizer=opt, scheduler=sched, scaler=scaler, run_identity=run_identity)
        start_epoch = int(payload["epoch"])
        start_batch = int(payload["batch_in_epoch"])
        step = int(payload["optimizer_step"])
        examples_seen_total = int(payload.get("examples_seen", 0))
        selection = SelectionState.from_dict(payload.get("selection_state"))
        if fallback:
            recovery_events.append(fallback)
        if selection.best is not None:
            if not selection.best.get("checkpoint_sha256") and fallback and fallback.get("candidate") == "selected":
                ref = fallback["recovered"]
                selection.bind_selected_checkpoint(sha256=ref["sha256"], relative_path=ref["relative_path"])
            verify_selected(root, expected_identity=_identity(run_identity), expected_sha256=selection.best.get("checkpoint_sha256"))

    session_budget = session_budget or SessionBudget()
    session_budget.start()
    session_budget.install_signal_handlers()
    segment_start_step = step
    segment_examples = 0
    dataloader_wait = 0.0
    checkpoint_save_seconds = 0.0
    checkpoint_load_seconds = locals().get("checkpoint_load_seconds", 0.0)
    estimated_checkpoint_seconds = 0.0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    opt.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, epochs):
        train_dataset.set_epoch(epoch)
        absolute_start_batch = start_batch if epoch == start_epoch else 0
        sampler = EpochPermutationSampler(
            train_dataset,
            training_seed=run_identity["seed"],
            epoch=epoch,
            start_sample=min(absolute_start_batch * micro, len(train_dataset)),
        )
        loader = _loader(train_dataset, sampler=sampler, batch_size=micro, num_workers=num_workers)
        student.train()
        it = iter(loader)
        local_bi = 0
        while True:
            wait_start = time.perf_counter()
            try:
                x, y, _ids = next(it)
            except StopIteration:
                break
            dataloader_wait += time.perf_counter() - wait_start
            absolute_bi = absolute_start_batch + local_bi
            local_bi += 1
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            n = int(y.numel())
            segment_examples += n
            examples_seen_total += n
            bucket_start_batch = (absolute_bi // accum) * accum
            bucket_end_batch = min(bucket_start_batch + accum, batches_per_epoch)
            bucket_start_sample = bucket_start_batch * micro
            bucket_end_sample = min(bucket_end_batch * micro, len(train_dataset))
            bucket_samples = bucket_end_sample - bucket_start_sample
            if bucket_samples <= 0:
                raise RuntimeError("invalid gradient-accumulation bucket sample count")

            with torch.amp.autocast("cuda", dtype=torch.float16):
                sf, sl = prelogits_and_logits(student, x)
                ce = F.cross_entropy(sl, y, label_smoothing=ctc["training"]["label_smoothing"])
                kd = feat = torch.zeros((), device=device, dtype=torch.float32)
                if teacher is not None:
                    # The frozen design requires teacher targets/features to be
                    # computed in FP32 on the exact same augmented tensor.
                    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                        tf, tl = prelogits_and_logits(teacher, x.float())
                    if objective.get("kd", 0) > 0:
                        kd = kd_loss(sl.float(), tl.float(), ctc["teacher"]["temperature"])
                    if objective.get("feature", 0) > 0:
                        feat = feature_loss(projection(sf).float(), tf.float())
                mean_loss = (
                    objective["ce"] * ce
                    + objective.get("kd", 0) * kd
                    + objective.get("feature", 0) * feat
                )
                # Convert the per-microbatch mean into the exact contribution to
                # the current accumulation bucket. This preserves full-bucket
                # behavior and correctly normalizes the final partial bucket.
                loss = mean_loss * (float(n) / float(bucket_samples))
            scaler.scale(loss).backward()
            update_boundary = ((absolute_bi + 1) % accum == 0) or (absolute_bi + 1 == batches_per_epoch)
            if not update_boundary:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + (list(projection.parameters()) if projection is not None else []),
                ctc["training"]["gradient_clip_norm"],
            )
            old_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            optimizer_applied = scaler.get_scale() >= old_scale
            if optimizer_applied:
                sched.step()
                step += 1
            opt.zero_grad(set_to_none=True)
            data_order_state = {
                "epoch": epoch,
                "next_batch_in_epoch": absolute_bi + 1,
                "epoch_order_seed": epoch_order_seed(run_identity["seed"], epoch),
                "full_order_sha256": sampler.full_order_sha256,
            }
            should_periodic = checkpoint_every_steps and step and step % checkpoint_every_steps == 0
            should_stop_for_calibration = max_optimizer_steps is not None and step - segment_start_step >= max_optimizer_steps
            should_rollover = session_budget.should_finalize(estimated_checkpoint_seconds=estimated_checkpoint_seconds, estimated_sync_seconds=max(60.0, estimated_checkpoint_seconds))
            if should_periodic or should_stop_for_calibration or should_rollover:
                payload = _checkpoint_payload(
                    student=student, projection=projection, optimizer=opt, scheduler=sched, scaler=scaler,
                    epoch=epoch, batch_in_epoch=absolute_bi + 1, optimizer_step=step, run_identity=run_identity,
                    selection_state=selection, examples_seen=examples_seen_total, data_order_state=data_order_state,
                )
                ref, save_elapsed = save_torch_checkpoint(root, kind="latest", payload=payload, expected_identity=_identity(run_identity), min_free_bytes=min_free_bytes)
                checkpoint_save_seconds += save_elapsed
                estimated_checkpoint_seconds = max(estimated_checkpoint_seconds, save_elapsed)
            if should_stop_for_calibration or should_rollover:
                elapsed = time.perf_counter() - started
                forward_benchmark = None
                if val_dataset is not None:
                    forward_benchmark = benchmark_forward(
                        student,
                        _loader(val_dataset, batch_size=validation_batch, num_workers=num_workers, shuffle=False),
                        device,
                        max_batches=20,
                    )
                return {
                    "mode": "calibration" if should_stop_for_calibration else "planned_rollover",
                    "planned_rollover": bool(should_rollover),
                    "rollover_reason": session_budget.signal_reason or ("SESSION_WALL_CLOCK_BUDGET" if should_rollover else None),
                    "optimizer_steps_segment": step - segment_start_step,
                    "optimizer_step_total": step,
                    "examples_segment": segment_examples,
                    "examples_total": examples_seen_total,
                    "wall_seconds_segment": elapsed,
                    "sec_per_optimizer_step": elapsed / max(step - segment_start_step, 1),
                    "examples_per_second": segment_examples / max(elapsed, 1e-12),
                    "dataloader_wait_seconds": dataloader_wait,
                    "dataloader_examples_per_wait_second": segment_examples / max(dataloader_wait, 1e-12),
                    "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                    "checkpoint_save_seconds": checkpoint_save_seconds,
                    "checkpoint_load_seconds": checkpoint_load_seconds,
                    "latest_checkpoint_sha256": ref.sha256,
                    "validation_forward_benchmark": forward_benchmark,
                    "recovery_events": recovery_events,
                }
        start_batch = 0
        summary = {"epoch": epoch + 1}
        if validation_enabled:
            val_loader = _loader(val_dataset, batch_size=validation_batch, num_workers=num_workers, shuffle=False)
            metrics = evaluate_classifier(student, val_loader, device)
            summary.update(
                {
                    "validation_macro_f1": metrics.macro_f1,
                    "validation_balanced_accuracy": metrics.balanced_accuracy,
                    "validation_nll": metrics.nll,
                    "validation_accuracy": metrics.accuracy,
                    "validation_examples_per_second": metrics.examples_per_second,
                }
            )
            if selection.consider(summary):
                selected_payload = _checkpoint_payload(
                    student=student, projection=projection, optimizer=opt, scheduler=sched, scaler=scaler,
                    epoch=epoch + 1, batch_in_epoch=0, optimizer_step=step, run_identity=run_identity,
                    selection_state=selection, examples_seen=examples_seen_total,
                    data_order_state={"epoch": epoch + 1, "next_batch_in_epoch": 0},
                )
                selected_ref, selected_save = save_torch_checkpoint(root, kind="selected", payload=selected_payload, expected_identity=_identity(run_identity), min_free_bytes=min_free_bytes)
                checkpoint_save_seconds += selected_save
                estimated_checkpoint_seconds = max(estimated_checkpoint_seconds, selected_save)
                selection.bind_selected_checkpoint(sha256=selected_ref.sha256, relative_path=selected_ref.relative_path)
        latest_payload = _checkpoint_payload(
            student=student, projection=projection, optimizer=opt, scheduler=sched, scaler=scaler,
            epoch=epoch + 1, batch_in_epoch=0, optimizer_step=step, run_identity=run_identity,
            selection_state=selection, examples_seen=examples_seen_total,
            data_order_state={"epoch": epoch + 1, "next_batch_in_epoch": 0},
        )
        latest_ref, save_elapsed = save_torch_checkpoint(root, kind="latest", payload=latest_payload, expected_identity=_identity(run_identity), min_free_bytes=min_free_bytes)
        checkpoint_save_seconds += save_elapsed
        estimated_checkpoint_seconds = max(estimated_checkpoint_seconds, save_elapsed)

    if validation_enabled and selection.best is None:
        raise RuntimeError("30-epoch run completed without validation-selected checkpoint")
    elapsed = time.perf_counter() - started
    forward_benchmark = benchmark_forward(student, _loader(val_dataset, batch_size=validation_batch, num_workers=num_workers, shuffle=False), device, max_batches=20) if val_dataset is not None else None
    return {
        "mode": "scientific_training",
        "planned_rollover": False,
        "optimizer_steps_segment": step - segment_start_step,
        "optimizer_step_total": step,
        "examples_segment": segment_examples,
        "examples_total": examples_seen_total,
        "wall_seconds_segment": elapsed,
        "sec_per_optimizer_step": elapsed / max(step - segment_start_step, 1),
        "examples_per_second": segment_examples / max(elapsed, 1e-12),
        "dataloader_wait_seconds": dataloader_wait,
        "dataloader_examples_per_wait_second": segment_examples / max(dataloader_wait, 1e-12),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint_save_seconds": checkpoint_save_seconds,
        "checkpoint_load_seconds": checkpoint_load_seconds,
        "history": selection.history,
        "selected_epoch": selection.best["epoch"],
        "selected_metrics": selection.best["metrics"],
        "selected_checkpoint_sha256": selection.best["checkpoint_sha256"],
        "latest_checkpoint_sha256": latest_ref.sha256,
        "validation_forward_benchmark": forward_benchmark,
        "recovery_events": recovery_events,
    }
