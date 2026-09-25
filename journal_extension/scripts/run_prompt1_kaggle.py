#!/usr/bin/env python3
"""QA-locked Kaggle executor for CropCop Prompt 1 (WP-02 through WP-07).

The script is intentionally stage-oriented and fail-closed.  It never opens the
consumed V1 test split, never uses Track-B protected results, and writes all
validation-derived data only below the restricted bundle.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator


PLAN_SHA256 = "c2a354cb0f9e5033742deb354b1fc42b49181e8884f5b6dfde14d2d5018a59c0"
CHECKPOINT_SHA256 = "dc7fea2e8db91bf1fc023cb5e792b23b67659edec22e10a7c1d46b4010db3974"
CHECKPOINT_BYTES = 335_202_763
RUN_RECORD_SHA256 = "0f403138ee43b1f0e464f092b51cf4f80a13233c9bc8ed4b17e86cf7446c5516"
MANIFEST_SHA256 = "bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2"
CLASS_MAP_SHA256 = "46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2"
CTC_FILE_SHA256 = "53937a6d8e87d18b7de086ecd1c000700d946770c523e50bb85cf124048764c4"
CTC_CANONICAL_SHA256 = "ed2331e63455e35ade00b548217fe987a9504f26a41e6c5d8a794e38741550bb"
S1_CONFIG_FILE_SHA256 = "15b2e8d2c288f05e8f5fc915d7c7de50cca5554b76544a4c773bc8d60f814793"
S1_CONFIG_CANONICAL_SHA256 = "a70cc70b11074fae07e5a6116bbb80e850a8011cce76bc52fc4406758abcb8d5"
SOURCE_GIT_COMMIT = "8904b100d223e4319776199c87ab397db23600ce"
AUTHORITY_ID = "EAAI-JE-TRACKBC-R07-DOWNSTREAM-v3"
MODEL_ID = "R07-CNXTT-CONTEXT-S1"
BASELINE_AUDIT_SHA256 = "d7acc4e615b5edd00fbd66b70a940452935c8cde94bb0d8860802be1fdc07ce5"
AUTHORITY_BINDING_SHA256 = "9f0e75fa0147430872b5e3bdee53814a84942d540d21720b657b2a3bc31da81d"
RECONSTRUCTION_CONTRACT_SHA256 = "48dab3895aac9740ce16642787b50068debf4272db76578176ccfeb57cd57daf"
NUM_CLASSES = 120
TRAIN_COUNT = 76_376
VAL_COUNT = 16_368
REPLAY_TOLERANCE = 1e-6
TORCH_VERSION = "2.12.1"
TORCHVISION_VERSION = "0.27.1"
EXECUTORCH_VERSION = "1.3.1"
TORCHAO_VERSION = "0.17.0"
SKLEARN_VERSION = "1.7.1"
PILLOW_VERSION = "12.3.0"
NUMPY_VERSION = "2.5.2"
AAR_COORDINATE = "org.pytorch:executorch-android:1.3.1"
AAR_SHA256 = "a9acc1e85a7f8b45c06ecb0fea398ebc6a5ad474d53278c2058c060ba4f37d17"
AAR_URL = (
    "https://repo.maven.apache.org/maven2/org/pytorch/executorch-android/1.3.1/"
    "executorch-android-1.3.1.aar"
)

EXPECTED_SOURCE_METRICS = {
    "validation_accuracy": 0.9853983382209188,
    "validation_balanced_accuracy": 0.966641504228933,
    "validation_macro_f1": 0.9681638484931451,
    "validation_nll": 0.09627299043618485,
}

# Frozen before the notebook opens full parity results.
FP32_PARITY_POLICY = {
    "raw_logit_endpoint": "pre-calibration float32 logits[120]",
    "top1_agreement_min": 1.0,
    "top5_set_agreement_min": 1.0,
    "max_absolute_logit_error_max": 0.002,
    "mean_absolute_logit_error_max": 0.00002,
    "p99_absolute_logit_error_max": 0.0002,
    "nonfinite_allowed": 0,
    "shape_mismatch_allowed": 0,
    "interpretation": "All conditions must pass; thresholds cannot be edited after result opening.",
}

# One and only one authorized PTQ route.
PTQ_POLICY = {
    "route_id": "R07-XNNPACK-PT2E-STATIC-INT8-PC-v1",
    "backend": "XNNPACK",
    "api": "PT2E",
    "activation": "static asymmetric int8",
    "weights": "symmetric int8 per-channel where supported",
    "unsupported_operator_policy": "remain floating and report actual coverage",
    "calibration_surface": "DS-V1-TRAIN only",
    "calibration_seed": 285554146,
    "calibration_rule": "8 hash-ranked rows per class plus 64 globally hash-ranked remaining rows",
    "calibration_count": 1024,
    "validation_surface": "DS-V1-VAL only",
    "acceptance": {
        "accuracy_drop_max": 0.003,
        "balanced_accuracy_drop_max": 0.005,
        "macro_f1_drop_max": 0.005,
        "nll_increase_max": 0.03,
        "top1_agreement_with_fp32_source_min": 0.99,
        "finite_logits_required": True,
        "output_width": 120,
        "runtime_load_and_forward_required": True,
    },
    "selection_if_pass": "INT8 becomes production default; FP32 remains mandatory scientific reference.",
    "selection_if_fail": "FP32 becomes production default; preserve terminal INT8 failure certificate.",
    "prohibitions": [
        "no QAT",
        "no retraining",
        "no observer/backend/bit-width sweep",
        "no V1 test",
        "no external Track-B data",
        "no phone-performance-driven selection",
    ],
}

DEVICE_POLICY = {
    "policy_id": "TRACKC-DEVICE-SELECTION-POLICY-v1",
    "status": "FROZEN_BEFORE_PERFORMANCE_OBSERVATION",
    "primary_rule": (
        "Freeze every compatible physical ARM64 handset present at the first Track-C readiness "
        "inventory before any performance-bearing execution. If more than one exists, select the "
        "primary by ascending SHA-256 of manufacturer|model|ADB-serial; this non-performance rule "
        "is deterministic and must be applied before latency, memory, or accuracy is observed."
    ),
    "compatibility_criteria": [
        "physical Android handset",
        "arm64-v8a",
        "Android API level >= 24",
        "sufficient free storage for the already-selected artifact and evidence capture",
        "ADB identity and build fingerprint obtainable",
    ],
    "secondary_rule": (
        "A secondary device is included only if it is present in the pre-performance readiness "
        "inventory; never add or remove a device in response to observed results."
    ),
    "performance_based_selection_forbidden": True,
}


class GateError(RuntimeError):
    """A fail-closed Prompt 1 gate failure."""


@dataclass(frozen=True)
class Inputs:
    repo_root: str
    checkpoint: str
    manifest: str
    class_map: str
    image_root: str
    plan: str
    run_record: str
    ctc_config: str
    s1_config: str
    baseline_audit: str
    authority_binding: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def seal_json(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("canonical_self_sha256", None)
    result["canonical_self_sha256"] = canonical_sha256(result)
    return result


def validate_sealed(payload: dict[str, Any]) -> None:
    expected = str(payload.get("canonical_self_sha256", ""))
    clean = dict(payload)
    clean.pop("canonical_self_sha256", None)
    if len(expected) != 64 or canonical_sha256(clean) != expected:
        raise GateError("canonical JSON self-hash mismatch")


def write_json(path: str | Path, payload: Any, *, seal: bool = False) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if seal:
        if not isinstance(payload, dict):
            raise TypeError("only JSON objects can be self-sealed")
        payload = seal_json(payload)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_file_hash(path: str | Path, expected: str, label: str) -> Path:
    value = Path(path).resolve()
    if not value.is_file():
        raise GateError(f"{label} missing: {value}")
    observed = sha256_file(value)
    if observed != expected:
        raise GateError(f"{label} SHA-256 mismatch: expected={expected}, observed={observed}, path={value}")
    return value


def normalized_version(value: str) -> str:
    return str(value).split("+", 1)[0]


def assert_versions() -> dict[str, Any]:
    import importlib.metadata
    import executorch
    import torch
    import torchvision
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    _ = (executorch, XNNPACKQuantizer, convert_pt2e, prepare_pt2e)

    observed = {
        "python": platform.python_version(),
        "torch": normalized_version(torch.__version__),
        "torchvision": normalized_version(torchvision.__version__),
        "executorch": normalized_version(importlib.metadata.version("executorch")),
        "torchao": normalized_version(importlib.metadata.version("torchao")),
        "scikit-learn": normalized_version(importlib.metadata.version("scikit-learn")),
        "pillow": normalized_version(importlib.metadata.version("pillow")),
        "numpy": normalized_version(importlib.metadata.version("numpy")),
        "flatc": subprocess.check_output(["flatc", "--version"], text=True).strip() if shutil.which("flatc") else "MISSING",
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_runtime": str(torch.version.cuda),
        "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }
    required = {
        "torch": TORCH_VERSION,
        "torchvision": TORCHVISION_VERSION,
        "executorch": EXECUTORCH_VERSION,
        "torchao": TORCHAO_VERSION,
        "scikit-learn": SKLEARN_VERSION,
        "pillow": PILLOW_VERSION,
        "numpy": NUMPY_VERSION,
    }
    drift = {name: (required[name], observed[name]) for name in required if observed[name] != required[name]}
    if drift:
        raise GateError(f"version-matched export/runtime environment required; drift={drift}")
    if observed["flatc"] == "MISSING":
        raise GateError("flatc compiler is required for ExecuTorch serialization")
    if sys.version_info[:2] != (3, 12):
        raise GateError(f"frozen S1 replay requires Python 3.12.x; observed={platform.python_version()}")
    if not torch.cuda.is_available() or not any("T4" in name for name in observed["cuda_devices"]):
        raise GateError(f"frozen S1 replay requires a Kaggle Tesla T4 GPU; observed={observed['cuda_devices']}")
    if str(torch.version.cuda) != "13.0":
        raise GateError(f"frozen S1 replay requires PyTorch CUDA runtime 13.0; observed={torch.version.cuda}")
    return observed


def executorch_toolchain_probe(work: Path) -> dict[str, Any]:
    """Prove export, flatc serialization, XNNPACK delegation, load, and execute."""
    import torch

    class Probe(torch.nn.Module):
        def forward(self, value):
            return torch.relu(value + 1.0)

    sample = torch.tensor([[-2.0, 0.0, 3.0]], dtype=torch.float32)
    destination = work / "runtime" / "toolchain_probe.pte"
    graph = export_program(Probe().eval(), sample, destination, generate_etrecord=False)
    output, _program, _method = runtime_forward(destination, sample)
    expected = Probe()(sample)
    max_abs = float((output.detach().cpu() - expected).abs().max())
    if graph["delegate_call_count"] <= 0 or max_abs > 1e-6:
        raise GateError(
            f"ExecuTorch Kaggle toolchain probe failed: delegates={graph['delegate_call_count']}, max_abs={max_abs}"
        )
    return {
        "status": "PASS",
        "artifact_sha256": sha256_file(destination),
        "artifact_bytes": destination.stat().st_size,
        "delegate_call_count": graph["delegate_call_count"],
        "max_absolute_error": max_abs,
    }


def executorch_ptq_toolchain_probe(work: Path) -> dict[str, Any]:
    """Fail early if the pinned PT2E/XNNPACK conversion path is unusable."""
    import torch
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    from executorch.exir import to_edge_transform_and_lower
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    class Probe(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)

        def forward(self, value):
            return torch.relu(self.conv(value))

    torch.manual_seed(21270083)
    sample = torch.randn(1, 3, 16, 16, dtype=torch.float32)
    captured = torch.export.export(Probe().eval(), (sample,), strict=True).module()
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    prepared = prepare_pt2e(captured, quantizer)
    with torch.inference_mode():
        prepared(sample)
    # Exported PT2E GraphModules reject train()/eval(); conversion preserves
    # the evaluation semantics captured above.
    quantized = convert_pt2e(prepared)
    quantized_graph = str(quantized.graph).lower()
    quantize_markers = quantized_graph.count("quantized_decomposed.quantize_per_")
    dequantize_markers = quantized_graph.count("quantized_decomposed.dequantize_per_")
    if quantize_markers <= 0 or dequantize_markers <= 0:
        raise GateError("PTQ toolchain probe produced no Q/DQ coverage markers")

    exported = torch.export.export(quantized, (sample,), strict=True)
    lowered = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    lowered_graph = str(lowered.exported_program().graph_module.graph)
    delegate_calls = lowered_graph.count("executorch_call_delegate")
    if delegate_calls <= 0:
        raise GateError("PTQ toolchain probe produced no XNNPACK delegated subgraph")
    destination = work / "runtime" / "toolchain_ptq_probe.pte"
    destination.parent.mkdir(parents=True, exist_ok=True)
    lowered.to_executorch().save(str(destination))
    output, _program, _method = runtime_forward(destination, sample)
    if tuple(output.shape) != (1, 4, 16, 16) or not bool(torch.isfinite(output).all()):
        raise GateError(f"PTQ toolchain runtime probe returned invalid output: shape={tuple(output.shape)}")
    return {
        "status": "PASS",
        "artifact_sha256": sha256_file(destination),
        "artifact_bytes": destination.stat().st_size,
        "delegate_call_count": delegate_calls,
        "quantized_graph_markers": {
            "quantize": quantize_markers,
            "dequantize": dequantize_markers,
        },
    }


def git_identity(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        # Kaggle ZIP extraction does not preserve Unix executable bits. Ignore only
        # that transport artifact; content changes remain visible and fail closed.
        return subprocess.check_output(
            ["git", "-c", "core.fileMode=false", "-C", str(repo), *args], text=True
        ).strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--porcelain=v1")),
        }
    except Exception as exc:
        return {"commit": None, "branch": None, "dirty": None, "error": f"{type(exc).__name__}: {exc}"}


def iter_metadata_candidates(root: Path, basenames: set[str]) -> Iterator[Path]:
    skip_dirs = {"train", "val", "test", ".git", "__pycache__", ".ipynb_checkpoints"}
    visited: set[Path] = set()
    for current, dirs, files in os.walk(root, followlinks=True):
        resolved_current = Path(current).resolve()
        if resolved_current in visited:
            dirs[:] = []
            continue
        visited.add(resolved_current)
        dirs[:] = [name for name in dirs if name not in skip_dirs]
        for name in files:
            if name in basenames:
                yield Path(current) / name


def unique_hash_match(root: Path, basenames: set[str], expected_sha: str, label: str) -> Path:
    matches = [candidate.resolve() for candidate in iter_metadata_candidates(root, basenames) if sha256_file(candidate) == expected_sha]
    unique = list(dict.fromkeys(matches))
    if len(unique) != 1:
        raise GateError(f"expected one hash-valid {label}; found={list(map(str, unique))}")
    return unique[0]


def find_repo(root: Path) -> Path:
    candidates: list[Path] = []
    marker = Path("journal_extension/src/cropcop_je/trackb_r07.py")
    if (root / marker).is_file():
        candidates.append(root.resolve())
    for path in iter_metadata_candidates(root, {"trackb_r07.py"}):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if relative.as_posix().endswith(marker.as_posix()):
            candidates.append(path.parents[3].resolve())
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise GateError(f"expected one research repository; found={list(map(str, candidates))}")
    return candidates[0]


def resolve_image_root(manifest: Path) -> Path:
    candidates: list[Path] = []
    for parent in (manifest.parent, manifest.parent.parent, manifest.parent.parent.parent):
        candidate = (parent / "dataset").resolve()
        if (candidate / "train").is_dir() and (candidate / "val").is_dir():
            candidates.append(candidate)
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise GateError(f"expected one Final-V1 image root paired with manifest; found={list(map(str, candidates))}")
    return candidates[0]


def find_final_v1_authority(root: Path) -> tuple[Path, Path, Path]:
    manifests = [
        path.resolve()
        for path in iter_metadata_candidates(root, {"final_manifest.csv"})
        if sha256_file(path) == MANIFEST_SHA256
        and "CropCop_Final_v1_CERTIFICATION_REPORTS" not in path.resolve().parts
    ]
    class_maps = [
        path.resolve()
        for path in iter_metadata_candidates(root, {"class_to_idx.json"})
        if sha256_file(path) == CLASS_MAP_SHA256
        and "CropCop_Final_v1_CERTIFICATION_REPORTS" not in path.resolve().parts
    ]
    candidates: list[tuple[Path, Path, Path]] = []
    for manifest in manifests:
        for class_map in class_maps:
            if manifest.parent != class_map.parent:
                continue
            try:
                image_root = resolve_image_root(manifest)
            except GateError:
                continue
            candidates.append((manifest, class_map, image_root))
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise GateError(
            "expected exactly one hash-valid, image-backed Final-V1 authority pair; "
            f"candidates={[(str(a), str(b), str(c)) for a, b, c in candidates]}"
        )
    return candidates[0]


def ensure_repo_import(repo: Path) -> None:
    source = str((repo / "journal_extension" / "src").resolve())
    if source not in sys.path:
        sys.path.insert(0, source)


def find_inputs(input_root: Path) -> Inputs:
    repo = find_repo(input_root)
    checkpoint = unique_hash_match(
        input_root,
        {"selected.g00000170.dc7fea2e8db91bf1.ckpt"},
        CHECKPOINT_SHA256,
        "R07-S1 checkpoint",
    )
    if checkpoint.stat().st_size != CHECKPOINT_BYTES:
        raise GateError("checkpoint byte count mismatch")
    manifest, class_map, image_root = find_final_v1_authority(input_root)
    plan = require_file_hash(input_root / "plan.md", PLAN_SHA256, "QA-locked plan") if (input_root / "plan.md").is_file() else unique_hash_match(input_root, {"plan.md"}, PLAN_SHA256, "QA-locked plan")
    run_record = require_file_hash(
        repo / "journal_extension/track_b_r07/replay_authority/R07_S1_ORIGINAL_RUN_RECORD.json",
        RUN_RECORD_SHA256,
        "original R07-S1 run record",
    )
    ctc = require_file_hash(repo / "journal_extension/configs/common/ctc_v2.json", CTC_FILE_SHA256, "CTC-v2 config")
    s1 = require_file_hash(repo / "journal_extension/configs/r07_cnxtt/s1.json", S1_CONFIG_FILE_SHA256, "R07-S1 config")
    baseline = unique_hash_match(
        input_root,
        {"R07_PRODUCTION_BASELINE_AUDIT.md"},
        BASELINE_AUDIT_SHA256,
        "P0 mobile baseline audit",
    )
    authority = require_file_hash(
        repo / "journal_extension/track_c_r07/shared_foundation/TRACKBC_R07_AUTHORITY_BINDING_v1.json",
        AUTHORITY_BINDING_SHA256,
        "F0 downstream authority binding",
    )
    if canonical_sha256(load_json(ctc)) != CTC_CANONICAL_SHA256:
        raise GateError("CTC-v2 canonical JSON identity mismatch")
    if canonical_sha256(load_json(s1)) != S1_CONFIG_CANONICAL_SHA256:
        raise GateError("S1 config canonical JSON identity mismatch")
    return Inputs(
        repo_root=str(repo),
        checkpoint=str(checkpoint),
        manifest=str(manifest),
        class_map=str(class_map),
        image_root=str(image_root),
        plan=str(plan),
        run_record=str(run_record),
        ctc_config=str(ctc),
        s1_config=str(s1),
        baseline_audit=str(baseline),
        authority_binding=str(authority),
    )


def load_rows(inputs: Inputs, surface: str):
    ensure_repo_import(Path(inputs.repo_root))
    from cropcop_je.frozen_v1_manifest import load_frozen_v1_rows

    expected = TRAIN_COUNT if surface == "DS-V1-TRAIN" else VAL_COUNT
    rows = load_frozen_v1_rows(
        inputs.manifest,
        inputs.class_map,
        expected_manifest_sha256=MANIFEST_SHA256,
        expected_class_map_sha256=CLASS_MAP_SHA256,
        surface=surface,
        expected_count=expected,
    )
    forbidden = [row for row in rows if row.split == "test" or row.relative_path.startswith("test/")]
    if forbidden:
        raise GateError("forbidden V1-test row entered an authorized surface")
    return rows


def verify_image_paths(rows: Iterable[Any], image_root: Path) -> dict[str, Any]:
    root = image_root.resolve()
    count = 0
    total = 0
    inventory = hashlib.sha256()
    for row in rows:
        relative = PurePosixPath(row.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise GateError(f"unsafe image path: {relative}")
        path = (root / Path(*relative.parts)).resolve()
        if root not in path.parents:
            raise GateError(f"image path escapes Final-V1 root: {relative}")
        if not path.is_file():
            raise GateError(f"authorized image missing: {path}")
        size = path.stat().st_size
        inventory.update(f"{relative.as_posix()}\0{size}\n".encode("utf-8"))
        count += 1
        total += size
    return {"rows": count, "bytes": total, "path_size_inventory_sha256": inventory.hexdigest()}


def stage_preflight(args: argparse.Namespace) -> Inputs:
    output = Path(args.work_root).resolve()
    evidence = output / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    inputs = find_inputs(Path(args.input_root).resolve())
    ensure_repo_import(Path(inputs.repo_root))
    versions = assert_versions()
    toolchain_probe = executorch_toolchain_probe(output)
    ptq_toolchain_probe = (
        {"status": "SKIPPED_PROSPECTIVELY_DISABLED"}
        if args.skip_ptq
        else executorch_ptq_toolchain_probe(output)
    )
    train_rows = load_rows(inputs, "DS-V1-TRAIN")
    val_rows = load_rows(inputs, "DS-V1-VAL")
    image_inventory = {
        "train": verify_image_paths(train_rows, Path(inputs.image_root)),
        "validation": verify_image_paths(val_rows, Path(inputs.image_root)),
    }
    repo_identity = git_identity(Path(inputs.repo_root))
    if repo_identity.get("commit") is None or repo_identity.get("dirty") is not False:
        raise GateError(f"research repository must include Git identity and be clean: {repo_identity}")
    payload = {
        "schema_version": "1.0",
        "stage": "PREFLIGHT",
        "status": "PASS",
        "created_at_utc": utc_now(),
        "inputs": asdict(inputs),
        "identities": {
            "plan_sha256": PLAN_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_bytes": CHECKPOINT_BYTES,
            "run_record_sha256": RUN_RECORD_SHA256,
            "manifest_sha256": MANIFEST_SHA256,
            "class_map_sha256": CLASS_MAP_SHA256,
            "ctc_file_sha256": CTC_FILE_SHA256,
            "ctc_canonical_sha256": CTC_CANONICAL_SHA256,
            "s1_config_file_sha256": S1_CONFIG_FILE_SHA256,
            "s1_config_canonical_sha256": S1_CONFIG_CANONICAL_SHA256,
            "baseline_audit_sha256": BASELINE_AUDIT_SHA256,
            "authority_binding_sha256": AUTHORITY_BINDING_SHA256,
        },
        "versions": versions,
        "kaggle_toolchain_probe": toolchain_probe,
        "kaggle_ptq_toolchain_probe": ptq_toolchain_probe,
        "repository": repo_identity,
        "executor_sha256": sha256_file(__file__),
        "image_inventory": image_inventory,
        "forbidden_surface_attestation": {
            "v1_test_opened": False,
            "track_b_external_results_opened": False,
            "s2_or_s3_substituted": False,
        },
    }
    write_json(evidence / "PROMPT1_PREFLIGHT_v1.json", payload, seal=True)
    write_json(output / "resolved_inputs.json", asdict(inputs))
    copy_verified(Path(inputs.baseline_audit), evidence / "P0_R07_PRODUCTION_BASELINE_AUDIT.md")
    copy_verified(Path(inputs.authority_binding), evidence / "F0_TRACKBC_R07_AUTHORITY_BINDING_v1.json")
    copy_verified(Path(inputs.plan), evidence / "PROMPT1_QA_LOCKED_PLAN.md")
    return inputs


def resolved_inputs(work_root: Path) -> Inputs:
    path = work_root / "resolved_inputs.json"
    if not path.is_file():
        raise GateError("preflight has not completed")
    return Inputs(**load_json(path))


def verify_input_bindings(inputs: Inputs) -> dict[str, Any]:
    expected = {
        "plan": PLAN_SHA256,
        "checkpoint": CHECKPOINT_SHA256,
        "run_record": RUN_RECORD_SHA256,
        "manifest": MANIFEST_SHA256,
        "class_map": CLASS_MAP_SHA256,
        "ctc_config": CTC_FILE_SHA256,
        "s1_config": S1_CONFIG_FILE_SHA256,
        "baseline_audit": BASELINE_AUDIT_SHA256,
        "authority_binding": AUTHORITY_BINDING_SHA256,
    }
    observed: dict[str, str] = {}
    for field, digest in expected.items():
        path = require_file_hash(getattr(inputs, field), digest, f"resolved {field}")
        observed[field] = sha256_file(path)
    if Path(inputs.checkpoint).stat().st_size != CHECKPOINT_BYTES:
        raise GateError("resolved checkpoint byte count drift")
    if canonical_sha256(load_json(inputs.ctc_config)) != CTC_CANONICAL_SHA256:
        raise GateError("resolved CTC canonical identity drift")
    if canonical_sha256(load_json(inputs.s1_config)) != S1_CONFIG_CANONICAL_SHA256:
        raise GateError("resolved S1 config canonical identity drift")
    identity = git_identity(Path(inputs.repo_root))
    if identity.get("commit") is None or identity.get("dirty") is not False:
        raise GateError(f"resolved repository is not clean and Git-identifiable: {identity}")
    return {"file_hashes": observed, "repository": identity}


def load_sealed(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise GateError(f"sealed evidence missing: {path}")
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise GateError(f"sealed evidence is not a JSON object: {path}")
    validate_sealed(payload)
    return payload


def verify_resume_marker(work: Path, relative: str) -> None:
    marker = (work / relative).resolve()
    if work not in marker.parents:
        raise GateError("resume marker escapes work root")
    payload = load_sealed(marker)
    if not str(payload.get("status", "")).startswith("PASS"):
        raise GateError(f"resume marker is not terminal PASS: {payload.get('status')}")
    inputs = resolved_inputs(work)
    binding = verify_input_bindings(inputs)
    preflight = load_sealed(work / "evidence" / "PROMPT1_PREFLIGHT_v1.json")
    if preflight.get("status") != "PASS" or preflight.get("executor_sha256") != sha256_file(__file__):
        raise GateError("preflight belongs to a different executor or is not PASS")
    if preflight.get("repository", {}).get("commit") != binding["repository"].get("commit"):
        raise GateError("repository commit drift since preflight")
    if payload.get("executor_sha256") != sha256_file(__file__):
        raise GateError("resume marker executor drift")

    name = marker.name
    dependencies = {
        "TRACKC_R07_RUNTIME_FEASIBILITY_REPORT_v1.json": "evidence/R07_S1_VALIDATION_REPLAY_REPORT_v1.json",
        "R07_FP32_SOURCE_ARTIFACT_FIDELITY_v1.json": "evidence/TRACKC_R07_RUNTIME_FEASIBILITY_REPORT_v1.json",
        "R07_INT8_RESULT_v1.json": "evidence/R07_FP32_SOURCE_ARTIFACT_FIDELITY_v1.json",
        "R07_INT8_FAILURE_CERTIFICATE_v1.json": "evidence/R07_FP32_SOURCE_ARTIFACT_FIDELITY_v1.json",
        "PROMPT1_FOUNDATION_HANDOFF.json": "evidence/R07_FP32_SOURCE_ARTIFACT_FIDELITY_v1.json",
    }
    if name in dependencies:
        verify_resume_marker(work, dependencies[name])
    if name == "PROMPT1_PREFLIGHT_v1.json":
        probe = payload.get("kaggle_toolchain_probe", {})
        require_file_hash(work / "runtime" / "toolchain_probe.pte", probe.get("artifact_sha256", ""), "preflight toolchain probe")
        if probe.get("status") != "PASS":
            raise GateError("preflight toolchain probe is not PASS")
        ptq_probe = payload.get("kaggle_ptq_toolchain_probe", {})
        if ptq_probe.get("status") == "PASS":
            require_file_hash(
                work / "runtime" / "toolchain_ptq_probe.pte",
                ptq_probe.get("artifact_sha256", ""),
                "preflight PTQ toolchain probe",
            )
        elif ptq_probe.get("status") != "SKIPPED_PROSPECTIVELY_DISABLED":
            raise GateError("preflight PTQ toolchain probe is neither PASS nor prospectively disabled")
    elif name == "R07_S1_VALIDATION_REPLAY_REPORT_v1.json":
        restricted = work / "R07_RESTRICTED_VERIFICATION_BUNDLE" / "source_replay"
        for filename, key in (("source_logits.npy", "source_logits_sha256"), ("targets.npy", "targets_sha256"), ("row_ids.json", "row_ids_sha256")):
            require_file_hash(restricted / filename, payload[key], f"F1 {filename}")
        if payload.get("validation_rows") != VAL_COUNT or payload.get("checkpoint_sha256") != CHECKPOINT_SHA256:
            raise GateError("F1 marker identity drift")
    elif name == "TRACKC_R07_RUNTIME_FEASIBILITY_REPORT_v1.json":
        artifact = payload.get("artifact", {})
        require_file_hash(artifact.get("path", ""), artifact.get("sha256", ""), "F2 FP32 artifact")
        aar = payload.get("aar", {})
        require_file_hash(aar.get("path", ""), aar.get("sha256", ""), "F2 Android AAR")
    elif name == "R07_FP32_SOURCE_ARTIFACT_FIDELITY_v1.json":
        require_file_hash(work / "runtime" / "r07_s1_xnnpack_fp32.pte", payload.get("artifact_sha256", ""), "F3 artifact")
        restricted = work / "R07_RESTRICTED_VERIFICATION_BUNDLE" / "fp32_parity"
        for filename, digest in payload.get("restricted_outputs", {}).items():
            require_file_hash(restricted / filename, digest, f"F3 {filename}")
        lock = load_sealed(work / "evidence" / "R07_FP32_SOURCE_ARTIFACT_PARITY_LOCK_v1.json")
        if sha256_file(work / "evidence" / "R07_FP32_SOURCE_ARTIFACT_PARITY_LOCK_v1.json") != payload.get("lock_sha256"):
            raise GateError("F3 frozen lock drift")
        if lock.get("executor_sha256") != sha256_file(__file__):
            raise GateError("F3 lock executor drift")
    elif name == "R07_INT8_RESULT_v1.json":
        artifact = payload.get("artifact", {})
        require_file_hash(artifact.get("path", ""), artifact.get("sha256", ""), "F4 INT8 artifact")
        if sha256_file(work / "evidence" / "R07_PTQ_RECIPE_LOCK_v1.json") != payload.get("recipe_lock_sha256"):
            raise GateError("F4 recipe lock drift")
    elif name == "PROMPT1_FOUNDATION_HANDOFF.json":
        final = load_sealed(work / "FINAL_DOWNLOAD.json")
        zip_path = require_file_hash(final.get("path", ""), final.get("sha256", ""), "final Kaggle ZIP")
        round_trip = verify_final_zip(zip_path, work / "PROMPT1_FINAL_EVIDENCE")
        if round_trip != final.get("round_trip_verification"):
            raise GateError("final ZIP round-trip verification record drift")


def load_source_model(inputs: Inputs):
    ensure_repo_import(Path(inputs.repo_root))
    from cropcop_je.trackb_r07 import load_r07_checkpoint

    adapter, payload = load_r07_checkpoint(inputs.checkpoint, seed_label="S1")
    identity = payload.get("identity") or {}
    required = {
        "source_git_commit": SOURCE_GIT_COMMIT,
        "manifest_sha256": MANIFEST_SHA256,
        "class_map_sha256": CLASS_MAP_SHA256,
        "ctc_v2_sha256": CTC_CANONICAL_SHA256,
        "config_sha256": S1_CONFIG_CANONICAL_SHA256,
    }
    mismatches = {key: (value, identity.get(key)) for key, value in required.items() if identity.get(key) != value}
    if mismatches:
        raise GateError(f"checkpoint scientific identity mismatch: {mismatches}")
    if int(payload.get("epoch", -1)) != 27:
        raise GateError("selected checkpoint epoch mismatch")
    return adapter, payload


def dataloader(inputs: Inputs, rows: list[Any], *, batch_size: int, workers: int):
    import torch
    ensure_repo_import(Path(inputs.repo_root))
    from cropcop_je.data import CropCopManifestDataset

    dataset = CropCopManifestDataset(rows, inputs.image_root, training_seed=21270083, train=False)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def metrics_from_logits(logits, targets) -> dict[str, float]:
    import numpy as np

    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if logits.shape != (targets.shape[0], NUM_CLASSES):
        raise GateError(f"metric input shape mismatch: logits={logits.shape}, targets={targets.shape}")
    if not np.isfinite(logits).all():
        raise GateError("non-finite logits")
    predictions = logits.argmax(axis=1)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    tp = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    maxima = logits.max(axis=1, keepdims=True)
    logsumexp = maxima[:, 0] + np.log(np.exp(logits - maxima).sum(axis=1))
    nll = float(np.mean(logsumexp - logits[np.arange(targets.shape[0]), targets]))
    return {
        "validation_accuracy": float(np.mean(predictions == targets)),
        "validation_balanced_accuracy": float(np.mean(recall)),
        "validation_macro_f1": float(np.mean(f1)),
        "validation_nll": nll,
    }


def metrics_from_streaming_counts(confusion, *, correct: int, count: int, nll_sum: float) -> dict[str, float]:
    """Match the frozen evaluator's float32, batch-summed NLL semantics exactly."""
    import numpy as np

    confusion = np.asarray(confusion, dtype=np.int64)
    if confusion.shape != (NUM_CLASSES, NUM_CLASSES) or int(confusion.sum()) != count or count <= 0:
        raise GateError("streaming metric accumulator is inconsistent")
    tp = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return {
        "validation_accuracy": float(correct / count),
        "validation_balanced_accuracy": float(np.mean(recall)),
        "validation_macro_f1": float(np.mean(f1)),
        "validation_nll": float(nll_sum / count),
    }


def stage_source(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    work = Path(args.work_root).resolve()
    inputs = resolved_inputs(work)
    verify_input_bindings(inputs)
    rows = load_rows(inputs, "DS-V1-VAL")
    adapter, payload = load_source_model(inputs)
    torch.manual_seed(21270083)
    controlled = torch.randn(2, 3, 256, 256, dtype=torch.float32)
    with torch.inference_mode():
        adapter_logits = adapter(controlled)
        underlying_logits = adapter.model(controlled)
    adapter_exact = bool(torch.equal(adapter_logits, underlying_logits))
    adapter_max_abs = float((adapter_logits - underlying_logits).abs().max())
    if not adapter_exact or adapter_max_abs != 0.0:
        raise GateError("scientific adapter and underlying nn.Module differ on controlled inputs")
    controlled_logits_sha = hashlib.sha256(adapter_logits.numpy().tobytes()).hexdigest()
    model = adapter.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    loader = dataloader(inputs, rows, batch_size=args.source_batch_size, workers=args.workers)
    logits_parts: list[Any] = []
    targets_parts: list[Any] = []
    row_ids: list[str] = []
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    correct = 0
    count = 0
    nll_sum = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (images, targets, ids) in enumerate(loader):
            device_targets = targets.to(device, non_blocking=True)
            device_outputs = model(images.to(device, non_blocking=True))
            predictions = device_outputs.argmax(dim=1)
            correct += int((predictions == device_targets).sum().item())
            count += int(device_targets.numel())
            nll_sum += float(
                -torch.log_softmax(device_outputs, dim=1)
                .gather(1, device_targets.view(-1, 1))
                .sum()
                .item()
            )
            flat = (device_targets * NUM_CLASSES + predictions).detach().cpu()
            confusion += torch.bincount(flat, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
            outputs = device_outputs.detach().cpu().to(torch.float32).numpy()
            logits_parts.append(outputs)
            targets_parts.append(targets.numpy())
            row_ids.extend(map(str, ids))
            if (batch_index + 1) % 50 == 0:
                print(f"source replay: {len(row_ids)}/{VAL_COUNT}", flush=True)
    logits = np.concatenate(logits_parts, axis=0)
    targets = np.concatenate(targets_parts, axis=0)
    metrics = metrics_from_streaming_counts(
        confusion.numpy(), correct=correct, count=count, nll_sum=nll_sum
    )
    deltas = {name: abs(metrics[name] - expected) for name, expected in EXPECTED_SOURCE_METRICS.items()}
    passed = all(value <= REPLAY_TOLERANCE for value in deltas.values())
    restricted = work / "R07_RESTRICTED_VERIFICATION_BUNDLE" / "source_replay"
    restricted.mkdir(parents=True, exist_ok=True)
    np.save(restricted / "source_logits.npy", logits.astype(np.float32), allow_pickle=False)
    np.save(restricted / "targets.npy", targets.astype(np.int64), allow_pickle=False)
    write_json(restricted / "row_ids.json", row_ids)
    result = {
        "schema_version": "1.0",
        "gate": "F1",
        "status": "PASS" if passed else "FAIL",
        "model_id": MODEL_ID,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_git_commit": SOURCE_GIT_COMMIT,
        "selected_epoch": int(payload["epoch"]),
        "validation_rows": len(row_ids),
        "expected_metrics": EXPECTED_SOURCE_METRICS,
        "observed_metrics": metrics,
        "absolute_deltas": deltas,
        "tolerance": REPLAY_TOLERANCE,
        "elapsed_seconds": time.perf_counter() - started,
        "source_device": str(device),
        "source_batch_size": args.source_batch_size,
        "source_logits_sha256": sha256_file(restricted / "source_logits.npy"),
        "targets_sha256": sha256_file(restricted / "targets.npy"),
        "row_ids_sha256": sha256_file(restricted / "row_ids.json"),
        "forbidden_surface_attestation": {"v1_test_opened": False, "track_b_external_results_opened": False},
        "executor_sha256": sha256_file(__file__),
    }
    write_json(work / "evidence" / "R07_S1_VALIDATION_REPLAY_REPORT_v1.json", result, seal=True)
    if not passed:
        raise GateError(f"F1 validation replay differs from frozen metrics: {deltas}")
    contract_source = require_file_hash(
        Path(inputs.repo_root) / "journal_extension/track_c_r07/shared_foundation/R07_DEPLOYMENT_RECONSTRUCTION_CONTRACT_v1.json",
        RECONSTRUCTION_CONTRACT_SHA256,
        "frozen deployment reconstruction contract",
    )
    copy_verified(contract_source, work / "evidence" / contract_source.name)
    certificate = {
        "schema_version": "1.0",
        "status": "PASS",
        "gate": "F1",
        "checkpoint": {
            "bytes": Path(inputs.checkpoint).stat().st_size,
            "sha256": sha256_file(inputs.checkpoint),
            "selected_epoch": int(payload["epoch"]),
            "identity_sha256": payload.get("identity_sha256"),
        },
        "strict_reconstruction": {
            "construction": "torchvision.models.convnext_tiny(weights=None, num_classes=120)",
            "missing_keys": [],
            "unexpected_keys": [],
            "adapter_underlying_exact_equal": adapter_exact,
            "adapter_underlying_max_abs_error": adapter_max_abs,
            "controlled_input_seed": 21270083,
            "controlled_input_shape": [2, 3, 256, 256],
            "controlled_logits_sha256": controlled_logits_sha,
        },
        "run_record_sha256": RUN_RECORD_SHA256,
        "source_git_commit": SOURCE_GIT_COMMIT,
        "manifest_sha256": MANIFEST_SHA256,
        "class_map_sha256": CLASS_MAP_SHA256,
        "ctc_canonical_sha256": CTC_CANONICAL_SHA256,
    }
    certificate_path = write_json(
        work / "evidence" / "R07_S1_CHECKPOINT_VERIFICATION_CERTIFICATE_v1.json",
        certificate,
        seal=True,
    )
    model_handoff = {
        "schema_version": "1.0",
        "status": "PASS",
        "gate": "F1",
        "model_id": MODEL_ID,
        "source_git_commit": SOURCE_GIT_COMMIT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_bytes": CHECKPOINT_BYTES,
        "selected_epoch": 27,
        "num_classes": NUM_CLASSES,
        "run_record_sha256": RUN_RECORD_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "class_map_sha256": CLASS_MAP_SHA256,
        "ctc_file_sha256": CTC_FILE_SHA256,
        "ctc_canonical_sha256": CTC_CANONICAL_SHA256,
        "reconstruction_contract_sha256": sha256_file(contract_source),
        "checkpoint_certificate_sha256": sha256_file(certificate_path),
        "validation_replay_report_sha256": sha256_file(work / "evidence" / "R07_S1_VALIDATION_REPLAY_REPORT_v1.json"),
        "next_gate_authorized": True,
        "forbidden_surface_attestation": {"v1_test_accessed": False, "track_b_protected_results_used": False},
    }
    write_json(work / "evidence" / "TRACKC_R07_MODEL_HANDOFF_v1.json", model_handoff, seal=True)
    (work / "evidence" / "R07_S1_SOURCE_RECONSTRUCTION_REPORT.md").write_text(
        "# R07-S1 source reconstruction\n\n"
        f"F1 PASS. The exact `{MODEL_ID}` checkpoint was reconstructed as ConvNeXt-Tiny, "
        f"strictly loaded at epoch 27, and replayed over all {VAL_COUNT:,} frozen validation rows. "
        f"All four frozen metrics matched within {REPLAY_TOLERANCE:g}. No V1-test image or "
        "protected Track-B result was opened.\n",
        encoding="utf-8",
    )


def direct_model(inputs: Inputs):
    adapter, _ = load_source_model(inputs)
    return adapter.model.eval()


def runtime_forward(pte: Path, tensor):
    from executorch.runtime import Runtime

    program = Runtime.get().load_program(str(pte))
    method = program.load_method("forward")
    output = method.execute([tensor])[0]
    return output, program, method


def inspect_aar(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = sorted(archive.namelist())
        libraries = [name for name in names if name.startswith("jni/") and name.endswith(".so")]
        arm64 = [name for name in libraries if "/arm64-v8a/" in f"/{name}"]
        if "classes.jar" not in names or not arm64:
            raise GateError(f"Android AAR lacks classes.jar or ARM64 native libraries: {path}")
        identities = {
            name: {
                "bytes": archive.getinfo(name).file_size,
                "sha256": hashlib.sha256(archive.read(name)).hexdigest(),
            }
            for name in libraries
        }
    return {"native_libraries": identities, "entry_count": len(names)}


def acquire_aar(work: Path, input_root: Path) -> Path:
    destination = work / "runtime" / "executorch-android-1.3.1.aar"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        attached = sorted(iter_metadata_candidates(input_root, {"executorch-android-1.3.1.aar"}))
        if attached:
            hashes = {sha256_file(path): path for path in attached}
            if len(hashes) != 1:
                raise GateError(f"multiple non-identical attached Android AARs: {attached}")
            copy_verified(next(iter(hashes.values())), destination)
        else:
            print(f"downloading version-matched Android AAR: {AAR_URL}", flush=True)
            urllib.request.urlretrieve(AAR_URL, destination)
    if destination.stat().st_size <= 0:
        raise GateError("downloaded Android AAR is empty")
    if sha256_file(destination) != AAR_SHA256:
        raise GateError(
            f"Android AAR identity mismatch: expected={AAR_SHA256}, observed={sha256_file(destination)}"
        )
    inspect_aar(destination)
    return destination


def export_program(model, sample, destination: Path, *, generate_etrecord: bool = True) -> dict[str, Any]:
    import torch
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    exported = torch.export.export(model.eval(), (sample,), strict=True)
    manager = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
        generate_etrecord=generate_etrecord,
    )
    graph_text = str(manager.exported_program().graph_module.graph)
    runtime_manager = manager.to_executorch()
    destination.parent.mkdir(parents=True, exist_ok=True)
    runtime_manager.save(str(destination))
    etrecord_path = None
    if generate_etrecord:
        try:
            etrecord = runtime_manager.get_etrecord()
            etrecord.update_representative_inputs((sample,))
            etrecord_path = destination.with_suffix(".etrecord.bin")
            etrecord.save(str(etrecord_path))
        except Exception as exc:
            print(f"ETRecord unavailable (non-gating): {type(exc).__name__}: {exc}", flush=True)
    return {
        "exported_graph": str(exported.graph_module.graph),
        "lowered_graph": graph_text,
        "delegate_call_count": graph_text.count("executorch_call_delegate"),
        "etrecord_path": str(etrecord_path) if etrecord_path else None,
    }


def stage_fp32(args: argparse.Namespace) -> None:
    import torch

    work = Path(args.work_root).resolve()
    inputs = resolved_inputs(work)
    verify_input_bindings(inputs)
    assert_versions()
    model = direct_model(inputs).cpu().eval()
    torch.manual_seed(21270083)
    sample = torch.randn(1, 3, 256, 256, dtype=torch.float32)
    artifact = work / "runtime" / "r07_s1_xnnpack_fp32.pte"
    started = time.perf_counter()
    graph = export_program(model, sample, artifact)
    exported_graph_path = work / "evidence" / "R07_FP32_EXPORTED_GRAPH.txt"
    lowered_graph_path = work / "evidence" / "R07_FP32_LOWERED_GRAPH.txt"
    exported_graph_path.write_text(graph["exported_graph"] + "\n", encoding="utf-8")
    lowered_graph_path.write_text(graph["lowered_graph"] + "\n", encoding="utf-8")
    if graph["delegate_call_count"] <= 0:
        raise GateError("XNNPACK lowering produced no visible delegated subgraph")
    portable_lines = [
        line.strip()
        for line in graph["lowered_graph"].splitlines()
        if "call_function" in line and "executorch_call_delegate" not in line
    ]
    operator_report_path = write_json(
        work / "evidence" / "R07_FP32_OPERATOR_DELEGATION_REPORT_v1.json",
        {
            "schema_version": "1.0",
            "gate": "F2",
            "status": "PASS",
            "backend": "XNNPACK CPU",
            "delegated_subgraph_calls": graph["delegate_call_count"],
            "portable_or_bookkeeping_call_function_lines": portable_lines,
            "portable_or_bookkeeping_line_count": len(portable_lines),
            "exported_graph_sha256": sha256_file(exported_graph_path),
            "lowered_graph_sha256": sha256_file(lowered_graph_path),
            "interpretation": (
                "The lowered graph contains at least one XNNPACK delegate call. Any listed residual "
                "call_function lines are reported verbatim and are not silently described as delegated."
            ),
            "executor_sha256": sha256_file(__file__),
        },
        seal=True,
    )
    with torch.inference_mode():
        source = model(sample)
    runtime, _program, _method = runtime_forward(artifact, sample)
    runtime = runtime.detach().cpu()
    if tuple(runtime.shape) != (1, NUM_CLASSES):
        raise GateError(f"FP32 runtime output shape invalid: {tuple(runtime.shape)}")
    smoke_error = float((source - runtime).abs().max())
    if not math.isfinite(smoke_error) or smoke_error > FP32_PARITY_POLICY["max_absolute_logit_error_max"]:
        raise GateError(f"FP32 runtime smoke parity failed: max_abs={smoke_error}")
    aar = acquire_aar(work, Path(args.input_root).resolve())
    report = {
        "schema_version": "1.0",
        "gate": "F2",
        "status": "PASS",
        "model_id": MODEL_ID,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "software": assert_versions(),
        "backend": "XNNPACK CPU",
        "runtime_coordinate": AAR_COORDINATE,
        "aar": {
            "path": str(aar),
            "bytes": aar.stat().st_size,
            "sha256": sha256_file(aar),
            **inspect_aar(aar),
        },
        "artifact": {"path": str(artifact), "bytes": artifact.stat().st_size, "sha256": sha256_file(artifact)},
        "input_schema": {"dtype": "float32", "shape": [1, 3, 256, 256], "layout": "NCHW"},
        "output_schema": {"dtype": "float32", "shape": [1, 120], "semantics": "raw pre-calibration logits"},
        "smoke_max_absolute_error": smoke_error,
        "delegate_call_count": graph["delegate_call_count"],
        "exported_graph_sha256": sha256_file(exported_graph_path),
        "lowered_graph_sha256": sha256_file(lowered_graph_path),
        "etrecord": graph["etrecord_path"],
        "operator_delegation_report_sha256": sha256_file(operator_report_path),
        "export_implementation": {
            "file": str(Path(__file__).resolve()),
            "sha256": sha256_file(__file__),
            "function": "export_program",
        },
        "source_git_commit": SOURCE_GIT_COMMIT,
        "class_map_sha256": CLASS_MAP_SHA256,
        "preprocessing_sha256": CTC_FILE_SHA256,
        "fallback_report": {
            "alternate_runtime_used": False,
            "alternate_runtime_authorized": False,
            "portable_or_bookkeeping_lines_reported": len(portable_lines),
        },
        "failed_attempts": [],
        "completion_semantics": "Python Runtime method.execute returned synchronously before output inspection.",
        "elapsed_seconds": time.perf_counter() - started,
        "documentation_basis": [
            "https://docs.pytorch.org/executorch/stable/pathway-quickstart.html",
            "https://docs.pytorch.org/executorch/stable/using-executorch-android.html",
            "https://docs.pytorch.org/executorch/stable/backends/xnnpack/xnnpack-partitioner.html"
        ],
        "executor_sha256": sha256_file(__file__),
    }
    write_json(work / "evidence" / "TRACKC_R07_RUNTIME_FEASIBILITY_REPORT_v1.json", report, seal=True)
    (work / "evidence" / "TRACKC_R07_RUNTIME_FEASIBILITY_REPORT.md").write_text(
        "# Track-C R07 runtime feasibility\n\n"
        f"F2 PASS. Exact R07-S1 exported with ExecuTorch {EXECUTORCH_VERSION} to XNNPACK, "
        f"loaded by the Python runtime, and returned raw logits `[1,120]`. Artifact SHA-256: "
        f"`{report['artifact']['sha256']}`. Android AAR SHA-256: `{report['aar']['sha256']}`.\n",
        encoding="utf-8",
    )


def topk_sets(logits, k: int):
    import numpy as np
    return np.sort(np.argpartition(logits, -k, axis=1)[:, -k:], axis=1)


def stage_parity(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    work = Path(args.work_root).resolve()
    inputs = resolved_inputs(work)
    verify_input_bindings(inputs)
    artifact = work / "runtime" / "r07_s1_xnnpack_fp32.pte"
    if not artifact.is_file():
        raise GateError("F2 artifact missing")
    f1_report = load_sealed(work / "evidence" / "R07_S1_VALIDATION_REPLAY_REPORT_v1.json")
    if f1_report.get("status") != "PASS" or f1_report.get("executor_sha256") != sha256_file(__file__):
        raise GateError("F3 requires the current F1 source replay PASS")
    source_replay = work / "R07_RESTRICTED_VERIFICATION_BUNDLE" / "source_replay"
    source_logits_path = require_file_hash(
        source_replay / "source_logits.npy", f1_report["source_logits_sha256"], "F3 source logits"
    )
    targets_path = require_file_hash(
        source_replay / "targets.npy", f1_report["targets_sha256"], "F3 source targets"
    )
    source_ids_path = require_file_hash(
        source_replay / "row_ids.json", f1_report["row_ids_sha256"], "F3 source row IDs"
    )
    lock_path = work / "evidence" / "R07_FP32_SOURCE_ARTIFACT_PARITY_LOCK_v1.json"
    if not lock_path.exists():
        lock = {
            "schema_version": "1.0",
            "status": "FROZEN_BEFORE_FULL_RESULT_OPENING",
            "created_at_utc": utc_now(),
            "model_id": MODEL_ID,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "artifact_sha256": sha256_file(artifact),
            "artifact_bytes": artifact.stat().st_size,
            "source_runner": (
                f"frozen F1 torchvision 0.27.1 eager nn.Module replay, eval, float32, "
                f"device={f1_report['source_device']}, batch={f1_report['source_batch_size']}"
            ),
            "source_logits_sha256": f1_report["source_logits_sha256"],
            "runtime_runner": "ExecuTorch 1.3.1 Python Runtime, XNNPACK artifact, batch=1",
            "surface": "DS-V1-VAL",
            "manifest_sha256": MANIFEST_SHA256,
            "row_count": VAL_COUNT,
            "policy": FP32_PARITY_POLICY,
            "failure_interpretation": "F3 BOUNDED; do not authorize production migration or Prompt 2.",
            "executor_sha256": sha256_file(__file__),
        }
        write_json(lock_path, lock, seal=True)
    else:
        existing_lock = load_sealed(lock_path)
        if (
            existing_lock.get("executor_sha256") != sha256_file(__file__)
            or existing_lock.get("artifact_sha256") != sha256_file(artifact)
            or existing_lock.get("policy") != FP32_PARITY_POLICY
            or existing_lock.get("checkpoint_sha256") != CHECKPOINT_SHA256
            or existing_lock.get("source_logits_sha256") != f1_report["source_logits_sha256"]
        ):
            raise GateError("F3 frozen parity lock identity drift")
    rows = load_rows(inputs, "DS-V1-VAL")
    loader = dataloader(inputs, rows, batch_size=1, workers=args.workers)
    from executorch.runtime import Runtime
    program = Runtime.get().load_program(str(artifact))
    method = program.load_method("forward")
    source_logits = np.load(source_logits_path, allow_pickle=False)
    targets = np.load(targets_path, allow_pickle=False)
    row_ids = load_json(source_ids_path)
    if source_logits.shape != (VAL_COUNT, NUM_CLASSES) or targets.shape != (VAL_COUNT,) or len(row_ids) != VAL_COUNT:
        raise GateError("F1 source replay arrays have invalid F3 shapes")
    runtime_logits = np.empty((VAL_COUNT, NUM_CLASSES), dtype=np.float32)
    started = time.perf_counter()
    with torch.inference_mode():
        for index, (images, target, ids) in enumerate(loader):
            runtime = method.execute([images])[0].detach().cpu().to(torch.float32).numpy()[0]
            if runtime.shape != (NUM_CLASSES,):
                raise GateError(f"runtime output shape invalid at row {index}: {runtime.shape}")
            runtime_logits[index] = runtime
            if targets[index] != int(target.item()) or row_ids[index] != str(ids[0]):
                raise GateError(f"F1/F3 validation row alignment drift at index {index}")
            if (index + 1) % 250 == 0:
                print(f"FP32 parity: {index + 1}/{VAL_COUNT}", flush=True)
    absolute = np.abs(source_logits.astype(np.float64) - runtime_logits.astype(np.float64))
    top1 = float(np.mean(source_logits.argmax(1) == runtime_logits.argmax(1)))
    source_top5 = topk_sets(source_logits, 5)
    runtime_top5 = topk_sets(runtime_logits, 5)
    top5_rows_equal = np.all(source_top5 == runtime_top5, axis=1)
    top5 = float(np.mean(top5_rows_equal))
    summary = {
        "nonfinite": int((~np.isfinite(runtime_logits)).sum()),
        "top1_agreement": top1,
        "top5_set_agreement": top5,
        "max_absolute_logit_error": float(absolute.max()),
        "mean_absolute_logit_error": float(absolute.mean()),
        "p99_absolute_logit_error": float(np.quantile(absolute, 0.99)),
    }
    policy = FP32_PARITY_POLICY
    passed = (
        summary["nonfinite"] == policy["nonfinite_allowed"]
        and top1 >= policy["top1_agreement_min"]
        and top5 >= policy["top5_set_agreement_min"]
        and summary["max_absolute_logit_error"] <= policy["max_absolute_logit_error_max"]
        and summary["mean_absolute_logit_error"] <= policy["mean_absolute_logit_error_max"]
        and summary["p99_absolute_logit_error"] <= policy["p99_absolute_logit_error_max"]
    )
    restricted = work / "R07_RESTRICTED_VERIFICATION_BUNDLE" / "fp32_parity"
    restricted.mkdir(parents=True, exist_ok=True)
    np.save(restricted / "source_logits.npy", source_logits, allow_pickle=False)
    np.save(restricted / "runtime_logits.npy", runtime_logits, allow_pickle=False)
    np.save(restricted / "targets.npy", targets, allow_pickle=False)
    write_json(restricted / "row_ids.json", row_ids)
    row_max_errors = absolute.max(axis=1)
    mismatch_indices = np.flatnonzero(
        (source_logits.argmax(1) != runtime_logits.argmax(1))
        | (~top5_rows_equal)
        | (row_max_errors > FP32_PARITY_POLICY["max_absolute_logit_error_max"])
    )
    mismatch_ledger = {
        "schema_version": "1.0",
        "surface": "DS-V1-VAL",
        "rows_checked": VAL_COUNT,
        "mismatch_definition": (
            "source/runtime top-1 differs, top-5 set differs, or row max absolute logit error "
            "exceeds the frozen global maximum threshold"
        ),
        "mismatch_count": int(mismatch_indices.size),
        "entries": [
            {
                "row_id": row_ids[int(index)],
                "source_top1": int(source_logits[index].argmax()),
                "runtime_top1": int(runtime_logits[index].argmax()),
                "top5_set_equal": bool(top5_rows_equal[index]),
                "row_max_absolute_logit_error": float(row_max_errors[index]),
            }
            for index in mismatch_indices
        ],
    }
    write_json(restricted / "mismatch_ledger.json", mismatch_ledger)
    result = {
        "schema_version": "1.0",
        "gate": "F3",
        "status": "PASS" if passed else "BOUNDED",
        "lock_sha256": sha256_file(lock_path),
        "artifact_sha256": sha256_file(artifact),
        "artifact_bytes": artifact.stat().st_size,
        "export_commit": git_identity(Path(inputs.repo_root)).get("commit"),
        "runtime_version": EXECUTORCH_VERSION,
        "class_map_sha256": CLASS_MAP_SHA256,
        "preprocessing_sha256": CTC_FILE_SHA256,
        "rows": VAL_COUNT,
        "summary": summary,
        "policy": policy,
        "source_metrics": metrics_from_logits(source_logits, targets),
        "runtime_metrics": metrics_from_logits(runtime_logits, targets),
        "elapsed_seconds": time.perf_counter() - started,
        "restricted_outputs": {
            path.name: sha256_file(path) for path in sorted(restricted.iterdir()) if path.is_file()
        },
        "executor_sha256": sha256_file(__file__),
    }
    result_path = write_json(work / "evidence" / "R07_FP32_SOURCE_ARTIFACT_FIDELITY_v1.json", result, seal=True)
    if not passed:
        raise GateError(f"F3 bounded/inconclusive under frozen policy: {summary}")
    write_json(
        work / "evidence" / "R07_FP32_RUNTIME_ARTIFACT_LOCK_v1.json",
        {
            "schema_version": "1.0",
            "gate": "F3",
            "status": "PASS_LOCKED",
            "artifact_sha256": sha256_file(artifact),
            "artifact_bytes": artifact.stat().st_size,
            "fidelity_report_sha256": sha256_file(result_path),
            "executor_sha256": sha256_file(__file__),
        },
        seal=True,
    )


def calibration_rows(rows: list[Any]) -> list[Any]:
    grouped: dict[int, list[Any]] = {index: [] for index in range(NUM_CLASSES)}
    for row in rows:
        grouped[int(row.class_index)].append(row)
    selected: list[Any] = []
    selected_ids: set[str] = set()
    prefix = str(PTQ_POLICY["calibration_seed"])
    key = lambda row: (hashlib.sha256(f"{prefix}|{row.stable_row_id}".encode()).hexdigest(), row.stable_row_id)
    for class_index in range(NUM_CLASSES):
        candidates = sorted(grouped[class_index], key=key)
        if len(candidates) < 8:
            raise GateError(f"class {class_index} has fewer than 8 training rows")
        for row in candidates[:8]:
            selected.append(row)
            selected_ids.add(row.stable_row_id)
    remainder = sorted((row for row in rows if row.stable_row_id not in selected_ids), key=key)
    selected.extend(remainder[:64])
    if len(selected) != 1024 or len({row.stable_row_id for row in selected}) != 1024:
        raise GateError("PTQ calibration manifest cardinality/uniqueness failure")
    return selected


def stage_ptq(args: argparse.Namespace) -> None:
    import numpy as np
    import torch

    work = Path(args.work_root).resolve()
    inputs = resolved_inputs(work)
    verify_input_bindings(inputs)
    preflight = load_sealed(work / "evidence" / "PROMPT1_PREFLIGHT_v1.json")
    if preflight.get("kaggle_ptq_toolchain_probe", {}).get("status") != "PASS":
        raise GateError("PTQ execution requires a PASS preflight PTQ toolchain probe")
    lock_path = work / "evidence" / "R07_PTQ_RECIPE_LOCK_v1.json"
    rows = load_rows(inputs, "DS-V1-TRAIN")
    selected = calibration_rows(rows)
    calibration_manifest = [
        {"row_id": row.stable_row_id, "relative_path": row.relative_path, "class_index": row.class_index}
        for row in selected
    ]
    restricted = work / "R07_RESTRICTED_VERIFICATION_BUNDLE" / "ptq"
    restricted.mkdir(parents=True, exist_ok=True)
    manifest_path = restricted / "calibration_manifest.json"
    write_json(manifest_path, calibration_manifest)
    if not lock_path.exists():
        write_json(
            lock_path,
            {
                "schema_version": "1.0",
                "status": "FROZEN_BEFORE_CALIBRATION_OUTPUT_OPENING",
                "created_at_utc": utc_now(),
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "versions": assert_versions(),
                "recipe": PTQ_POLICY,
                "calibration_manifest_sha256": sha256_file(manifest_path),
                "executor_sha256": sha256_file(__file__),
            },
            seal=True,
        )
    else:
        existing_lock = load_sealed(lock_path)
        if existing_lock["calibration_manifest_sha256"] != sha256_file(manifest_path):
            raise GateError("PTQ calibration manifest drift")
        if (
            existing_lock.get("executor_sha256") != sha256_file(__file__)
            or existing_lock.get("checkpoint_sha256") != CHECKPOINT_SHA256
            or existing_lock.get("recipe") != PTQ_POLICY
        ):
            raise GateError("PTQ frozen lock identity drift")
    fp32_report = load_sealed(work / "evidence" / "R07_FP32_SOURCE_ARTIFACT_FIDELITY_v1.json")
    if fp32_report.get("status") != "PASS" or fp32_report.get("executor_sha256") != sha256_file(__file__):
        raise GateError("PTQ requires the current F3 PASS evidence")
    fp32_restricted = work / "R07_RESTRICTED_VERIFICATION_BUNDLE" / "fp32_parity"
    for filename in ("source_logits.npy", "targets.npy", "row_ids.json"):
        require_file_hash(
            fp32_restricted / filename,
            fp32_report.get("restricted_outputs", {}).get(filename, ""),
            f"PTQ F3 source {filename}",
        )
    try:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )
        from executorch.exir import to_edge_transform_and_lower
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

        model = direct_model(inputs).cpu().eval()
        sample = torch.zeros(1, 3, 256, 256, dtype=torch.float32)
        captured = torch.export.export(model, (sample,), strict=True).module()
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
        prepared = prepare_pt2e(captured, quantizer)
        loader = dataloader(inputs, selected, batch_size=1, workers=args.workers)
        with torch.inference_mode():
            for index, (images, _target, _ids) in enumerate(loader):
                prepared(images)
                if (index + 1) % 128 == 0:
                    print(f"PTQ calibration: {index + 1}/1024", flush=True)
        # PT2E returns an exported GraphModule whose train()/eval() methods are
        # deliberately disabled.  The source model was captured in eval mode,
        # so conversion must not re-apply nn.Module.eval() here.
        quantized = convert_pt2e(prepared)
        quantized_graph_text = str(quantized.graph)
        quantize_markers = quantized_graph_text.lower().count("quantized_decomposed.quantize_per_")
        dequantize_markers = quantized_graph_text.lower().count("quantized_decomposed.dequantize_per_")
        if quantize_markers <= 0 or dequantize_markers <= 0:
            raise GateError("PTQ conversion produced no Q/DQ coverage markers")
        quantized_graph_path = work / "evidence" / "R07_INT8_QUANTIZED_GRAPH.txt"
        quantized_graph_path.write_text(quantized_graph_text + "\n", encoding="utf-8")
        call_function_lines = [
            line.strip() for line in quantized_graph_text.splitlines() if "call_function" in line
        ]
        artifact = work / "runtime" / "r07_s1_xnnpack_int8.pte"
        exported = torch.export.export(quantized, (sample,), strict=True)
        lowered = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
        graph_text = str(lowered.exported_program().graph_module.graph)
        if graph_text.count("executorch_call_delegate") <= 0:
            raise GateError("INT8 route produced no visible XNNPACK delegated subgraph")
        manager = lowered.to_executorch()
        manager.save(str(artifact))
        smoke, _program, _method = runtime_forward(artifact, sample)
        if tuple(smoke.shape) != (1, NUM_CLASSES):
            raise GateError("INT8 runtime smoke output width mismatch")
        source_logits = np.load(fp32_restricted / "source_logits.npy", allow_pickle=False)
        targets = np.load(fp32_restricted / "targets.npy", allow_pickle=False)
        val_rows = load_rows(inputs, "DS-V1-VAL")
        val_loader = dataloader(inputs, val_rows, batch_size=1, workers=args.workers)
        from executorch.runtime import Runtime
        program = Runtime.get().load_program(str(artifact))
        method = program.load_method("forward")
        int8_logits = np.empty((VAL_COUNT, NUM_CLASSES), dtype=np.float32)
        with torch.inference_mode():
            for index, (images, _target, _ids) in enumerate(val_loader):
                int8_logits[index] = method.execute([images])[0].detach().cpu().numpy()[0]
                if (index + 1) % 250 == 0:
                    print(f"INT8 validation: {index + 1}/{VAL_COUNT}", flush=True)
        np.save(restricted / "int8_runtime_logits.npy", int8_logits, allow_pickle=False)
        source_metrics = metrics_from_logits(source_logits, targets)
        int8_metrics = metrics_from_logits(int8_logits, targets)
        acceptance = PTQ_POLICY["acceptance"]
        deltas = {
            "accuracy_drop": source_metrics["validation_accuracy"] - int8_metrics["validation_accuracy"],
            "balanced_accuracy_drop": source_metrics["validation_balanced_accuracy"] - int8_metrics["validation_balanced_accuracy"],
            "macro_f1_drop": source_metrics["validation_macro_f1"] - int8_metrics["validation_macro_f1"],
            "nll_increase": int8_metrics["validation_nll"] - source_metrics["validation_nll"],
            "top1_agreement": float(np.mean(source_logits.argmax(1) == int8_logits.argmax(1))),
        }
        passed = (
            np.isfinite(int8_logits).all()
            and deltas["accuracy_drop"] <= acceptance["accuracy_drop_max"]
            and deltas["balanced_accuracy_drop"] <= acceptance["balanced_accuracy_drop_max"]
            and deltas["macro_f1_drop"] <= acceptance["macro_f1_drop_max"]
            and deltas["nll_increase"] <= acceptance["nll_increase_max"]
            and deltas["top1_agreement"] >= acceptance["top1_agreement_with_fp32_source_min"]
        )
        result = {
            "schema_version": "1.0",
            "gate": "F4",
            "status": "PASS_INT8_LOCKED" if passed else "PASS_INT8_FAILED_TERMINALLY",
            "recipe_lock_sha256": sha256_file(lock_path),
            "artifact": {"path": str(artifact), "bytes": artifact.stat().st_size, "sha256": sha256_file(artifact)},
            "source_metrics": source_metrics,
            "int8_metrics": int8_metrics,
            "deltas": deltas,
            "acceptance": acceptance,
            "delegate_call_count": graph_text.count("executorch_call_delegate"),
            "quantized_graph_markers": {
                "quantize": quantize_markers,
                "dequantize": dequantize_markers,
            },
            "coverage_evidence": {
                "pre_lowering_graph_sha256": sha256_file(quantized_graph_path),
                "call_function_lines": call_function_lines,
                "delegated_subgraph_calls": graph_text.count("executorch_call_delegate"),
                "interpretation": (
                    "Q/DQ-marked regions are quantized; unmarked residual operations remain floating. "
                    "No full-INT8 claim is made."
                ),
            },
            "claim": "hybrid PT2E INT8/floating XNNPACK artifact; actual Q/DQ and residual graph evidence reported",
            "executor_sha256": sha256_file(__file__),
        }
        write_json(work / "evidence" / "R07_INT8_RESULT_v1.json", result, seal=True)
        if not passed:
            write_json(
                work / "evidence" / "R07_INT8_FAILURE_CERTIFICATE_v1.json",
                {"schema_version": "1.0", "status": "TERMINAL_FAILURE", "result": result},
                seal=True,
            )
    except Exception as exc:
        failure = {
            "schema_version": "1.0",
            "gate": "F4",
            "status": "PASS_INT8_FAILED_TERMINALLY",
            "route_id": PTQ_POLICY["route_id"],
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "recipe_lock_sha256": sha256_file(lock_path),
            "no_alternate_route_attempted": True,
            "executor_sha256": sha256_file(__file__),
        }
        write_json(work / "evidence" / "R07_INT8_FAILURE_CERTIFICATE_v1.json", failure, seal=True)
        print(f"Optional PTQ route failed terminally; continuing with FP32: {failure['error']}", flush=True)


def copy_verified(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise GateError(f"copy verification failed: {source} -> {destination}")


def directory_manifest(root: Path) -> dict[str, dict[str, Any]]:
    symlinks = [path for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise GateError(f"bundle contains forbidden symbolic links: {symlinks}")
    return {
        path.relative_to(root).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def publication_scan(production: Path, *, require_single_classifier: bool = False) -> dict[str, Any]:
    forbidden_names = {
        "source_logits.npy", "runtime_logits.npy", "int8_runtime_logits.npy", "targets.npy",
        "row_ids.json", "calibration_manifest.json", "final_manifest.csv", "kaggle.json",
    }
    files = [path for path in production.rglob("*") if path.is_file()]
    offenders = [
        str(path.relative_to(production))
        for path in files
        if path.name in forbidden_names or path.suffix.lower() == ".ckpt"
    ]
    ptes = [str(path.relative_to(production)) for path in files if path.suffix.lower() == ".pte"]
    test_markers: list[str] = []
    secret_markers: list[str] = []
    for path in files:
        if path.suffix.lower() in {".json", ".md", ".txt", ".csv"}:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            if "ds-v1-test-consumed" in text or '"split":"test"' in text.replace(" ", ""):
                test_markers.append(str(path.relative_to(production)))
            if any(marker in text for marker in ('"api_key"', '"password"', 'kaggle.json')):
                secret_markers.append(str(path.relative_to(production)))
    classifier_ok = len(ptes) == 1 if require_single_classifier else len(ptes) <= 1
    passed = not offenders and not test_markers and not secret_markers and classifier_ok
    return {
        "status": "PASS" if passed else "FAIL",
        "restricted_file_offenders": offenders,
        "test_markers": test_markers,
        "secret_markers": secret_markers,
        "classifier_pte_files": ptes,
    }


def verify_final_zip(zip_path: Path, final_root: Path) -> dict[str, Any]:
    expected = load_json(final_root / "FINAL_FILE_MANIFEST.json")
    if not isinstance(expected, dict):
        raise GateError("final file manifest is not an object")
    prefix = final_root.name + "/"
    with zipfile.ZipFile(zip_path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise GateError("final ZIP contains duplicate names")
        for name in names:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or not name.startswith(prefix):
                raise GateError(f"unsafe or unexpected final ZIP path: {name}")
        relative_names = {name[len(prefix):] for name in names}
        required_names = set(expected) | {"FINAL_FILE_MANIFEST.json"}
        if relative_names != required_names:
            raise GateError(
                f"final ZIP membership mismatch: missing={sorted(required_names-relative_names)}, "
                f"extra={sorted(relative_names-required_names)}"
            )
        for relative, identity in expected.items():
            data = archive.read(prefix + relative)
            if len(data) != identity["bytes"] or hashlib.sha256(data).hexdigest() != identity["sha256"]:
                raise GateError(f"final ZIP member identity mismatch: {relative}")
        bad_member = archive.testzip()
        if bad_member:
            raise GateError(f"final ZIP CRC failure: {bad_member}")
    return {"status": "PASS", "files_verified": len(expected), "zip_sha256": sha256_file(zip_path)}


def stage_seal(args: argparse.Namespace) -> Path:
    import numpy as np
    import torch

    work = Path(args.work_root).resolve()
    inputs = resolved_inputs(work)
    binding = verify_input_bindings(inputs)
    preflight = load_sealed(work / "evidence" / "PROMPT1_PREFLIGHT_v1.json")
    if preflight.get("status") != "PASS" or preflight.get("executor_sha256") != sha256_file(__file__):
        raise GateError("current preflight evidence is invalid")
    if preflight.get("repository", {}).get("commit") != binding["repository"].get("commit"):
        raise GateError("repository commit changed after preflight")
    if args.skip_ptq and not (work / "evidence" / "R07_INT8_RESULT_v1.json").exists() and not (work / "evidence" / "R07_INT8_FAILURE_CERTIFICATE_v1.json").exists():
        write_json(
            work / "evidence" / "R07_INT8_FAILURE_CERTIFICATE_v1.json",
            {
                "schema_version": "1.0",
                "gate": "F4",
                "status": "PASS_INT8_ABSENT_BY_PROSPECTIVE_OPERATOR_CHOICE",
                "reason": "--skip-ptq was fixed before result opening",
                "no_alternate_route_attempted": True,
                "executor_sha256": sha256_file(__file__),
            },
            seal=True,
        )
    f1 = load_sealed(work / "evidence" / "R07_S1_VALIDATION_REPLAY_REPORT_v1.json")
    f2 = load_sealed(work / "evidence" / "TRACKC_R07_RUNTIME_FEASIBILITY_REPORT_v1.json")
    f3 = load_sealed(work / "evidence" / "R07_FP32_SOURCE_ARTIFACT_FIDELITY_v1.json")
    if f1["status"] != "PASS" or f2["status"] != "PASS" or f3["status"] != "PASS":
        raise GateError("F1/F2/F3 must pass before production artifact selection")
    if any(report.get("executor_sha256") != sha256_file(__file__) for report in (f1, f2, f3)):
        raise GateError("F1/F2/F3 evidence executor drift")
    require_file_hash(f2["artifact"]["path"], f2["artifact"]["sha256"], "F2 artifact")
    require_file_hash(f2["aar"]["path"], f2["aar"]["sha256"], "F2 Android AAR")
    if f3.get("artifact_sha256") != f2["artifact"]["sha256"]:
        raise GateError("F2/F3 FP32 artifact identity mismatch")
    int8_result_path = work / "evidence" / "R07_INT8_RESULT_v1.json"
    int8_result = load_sealed(int8_result_path) if int8_result_path.is_file() else None
    if int8_result and int8_result.get("status") not in {"PASS_INT8_LOCKED", "PASS_INT8_FAILED_TERMINALLY"}:
        raise GateError(f"unexpected F4 terminal status: {int8_result.get('status')}")
    int8_passed = bool(int8_result and int8_result["status"] == "PASS_INT8_LOCKED")
    if int8_result and int8_result.get("executor_sha256") != sha256_file(__file__):
        raise GateError("F4 result executor drift")
    if int8_result and sha256_file(work / "evidence" / "R07_PTQ_RECIPE_LOCK_v1.json") != int8_result.get("recipe_lock_sha256"):
        raise GateError("F4 result recipe-lock drift")
    if not int8_result:
        failure = load_sealed(work / "evidence" / "R07_INT8_FAILURE_CERTIFICATE_v1.json")
        if not str(failure.get("status", "")).startswith("PASS") or failure.get("executor_sha256") != sha256_file(__file__):
            raise GateError("F4 terminal branch certificate is invalid")
    selected = work / "runtime" / ("r07_s1_xnnpack_int8.pte" if int8_passed else "r07_s1_xnnpack_fp32.pte")
    reference = work / "runtime" / "r07_s1_xnnpack_fp32.pte"
    require_file_hash(reference, f2["artifact"]["sha256"], "sealed FP32 reference")
    if int8_passed:
        require_file_hash(selected, int8_result["artifact"]["sha256"], "sealed INT8 selection")
    selection = {
        "schema_version": "1.0",
        "gate": "F5",
        "status": "PASS",
        "frozen_before_phone_performance": True,
        "fp32_reference": {"sha256": sha256_file(reference), "bytes": reference.stat().st_size},
        "int8_eligible": int8_passed,
        "production_default": "INT8" if int8_passed else "FP32",
        "selected_artifact_sha256": sha256_file(selected),
        "selected_artifact_bytes": selected.stat().st_size,
        "selection_basis": PTQ_POLICY["selection_if_pass"] if int8_passed else PTQ_POLICY["selection_if_fail"],
        "phone_performance_used": False,
        "silent_fallback_allowed": False,
        "load_failure_behavior": "fail explicitly with structured integrity/runtime error",
    }
    selection_path = write_json(work / "evidence" / "R07_PRODUCT_RUNTIME_SELECTION_v1.json", selection, seal=True)
    device_policy_path = write_json(
        work / "evidence" / "TRACKC_DEVICE_SELECTION_POLICY_v1.json",
        {"schema_version": "1.0", "gate": "F7", **DEVICE_POLICY},
        seal=True,
    )
    production = work / "R07_PRODUCTION_BUNDLE"
    if production.exists():
        shutil.rmtree(production)
    copy_verified(selected, production / "model" / selected.name)
    copy_verified(Path(f2["aar"]["path"]), production / "runtime" / "executorch-android-1.3.1.aar")
    copy_verified(Path(inputs.class_map), production / "contracts" / "class_to_idx.json")
    copy_verified(Path(inputs.ctc_config), production / "contracts" / "preprocessing_contract.json")
    copy_verified(selection_path, production / "contracts" / "product_runtime_selection.json")
    copy_verified(device_policy_path, production / "contracts" / "trackc_device_selection_policy.json")
    runtime_manifest = {
        "schema_version": "1.0",
        "model_id": MODEL_ID,
        "runtime": f"ExecuTorch {EXECUTORCH_VERSION}",
        "backend": "XNNPACK CPU",
        "android_coordinate": AAR_COORDINATE,
        "android_aar_sha256": f2["aar"]["sha256"],
        "artifact_sha256": sha256_file(selected),
        "input": {"dtype": "float32", "shape": [1, 3, 256, 256], "layout": "NCHW"},
        "output": {"dtype": "float32", "shape": [1, 120], "semantics": "raw logits"},
    }
    write_json(production / "contracts" / "runtime_manifest.json", runtime_manifest, seal=True)
    model_manifest = {
        "schema_version": "1.0",
        "model_id": MODEL_ID,
        "source_checkpoint_sha256": CHECKPOINT_SHA256,
        "source_git_commit": SOURCE_GIT_COMMIT,
        "class_map_sha256": CLASS_MAP_SHA256,
        "preprocessing_file_sha256": CTC_FILE_SHA256,
        "selected_artifact_sha256": sha256_file(selected),
        "precision": selection["production_default"],
    }
    write_json(production / "contracts" / "model_manifest.json", model_manifest, seal=True)
    torch.manual_seed(7654321)
    fixture = torch.randn(1, 3, 256, 256, dtype=torch.float32)
    output, _program, _method = runtime_forward(selected, fixture)
    fixture_dir = production / "synthetic_smoke_fixtures_only"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    np.save(fixture_dir / "input.npy", fixture.numpy(), allow_pickle=False)
    np.save(fixture_dir / "expected_raw_logits.npy", output.detach().cpu().numpy(), allow_pickle=False)
    write_json(production / "artifact_manifest.json", directory_manifest(production))
    scan = publication_scan(production, require_single_classifier=True)
    if scan["status"] != "PASS":
        raise GateError(f"production publication-boundary scan failed: {scan}")
    write_json(work / "evidence" / "R07_PUBLICATION_BOUNDARY_SCAN_v1.json", scan, seal=True)
    restricted = work / "R07_RESTRICTED_VERIFICATION_BUNDLE"
    restricted.mkdir(parents=True, exist_ok=True)
    if int8_passed:
        copy_verified(reference, restricted / "reference_model" / reference.name)
    attempted_int8 = work / "runtime" / "r07_s1_xnnpack_int8.pte"
    if attempted_int8.is_file() and not int8_passed:
        copy_verified(attempted_int8, restricted / "attempted_models" / attempted_int8.name)
    etrecord = work / "runtime" / "r07_s1_xnnpack_fp32.etrecord.bin"
    if etrecord.is_file():
        copy_verified(etrecord, restricted / "runtime_debug" / etrecord.name)
    for path in (work / "evidence").glob("*.json"):
        if "PARITY" in path.name or "VALIDATION" in path.name or "PTQ" in path.name or "INT8" in path.name:
            copy_verified(path, restricted / "evidence" / path.name)
    write_json(restricted / "artifact_manifest.json", directory_manifest(restricted))
    bundle_seal_path = write_json(
        work / "evidence" / "R07_DEPLOYMENT_BUNDLE_SEAL_v1.json",
        {
            "schema_version": "1.0",
            "gate": "F6",
            "status": "PASS",
            "production_manifest_sha256": sha256_file(production / "artifact_manifest.json"),
            "restricted_manifest_sha256": sha256_file(restricted / "artifact_manifest.json"),
            "publication_scan_sha256": sha256_file(work / "evidence" / "R07_PUBLICATION_BOUNDARY_SCAN_v1.json"),
            "selected_artifact_sha256": sha256_file(selected),
            "executor_sha256": sha256_file(__file__),
        },
        seal=True,
    )
    handoff = {
        "schema_version": "2.0",
        "prompt_id": "PROMPT_1",
        "status": "PASS",
        "created_at_utc": utc_now(),
        "plan_sha256": PLAN_SHA256,
        "repository": git_identity(Path(inputs.repo_root)),
        "executor_sha256": sha256_file(__file__),
        "gates": {
            "P0": "PASS_BOUND_BY_PRIOR_MOBILE_AUDIT",
            "F0": "PASS_BOUND_BY_CURRENT_DOWNSTREAM_AUTHORITY",
            "F1": "PASS",
            "F2": "PASS",
            "F3": "PASS",
            "F4": "PASS_INT8_LOCKED" if int8_passed else "PASS_INT8_ABSENT_OR_FAILED_TERMINALLY",
            "F5": "PASS",
            "F6": "PASS",
            "F7": "PASS",
        },
        "identities": {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "run_record_sha256": RUN_RECORD_SHA256,
            "manifest_sha256": MANIFEST_SHA256,
            "class_map_sha256": CLASS_MAP_SHA256,
            "ctc_file_sha256": CTC_FILE_SHA256,
            "ctc_canonical_sha256": CTC_CANONICAL_SHA256,
            "fp32_artifact_sha256": sha256_file(reference),
            "selected_production_artifact_sha256": sha256_file(selected),
            "production_bundle_manifest_sha256": sha256_file(production / "artifact_manifest.json"),
            "restricted_bundle_manifest_sha256": sha256_file(restricted / "artifact_manifest.json"),
            "runtime_selection_sha256": sha256_file(selection_path),
            "device_policy_sha256": sha256_file(device_policy_path),
            "deployment_bundle_seal_sha256": sha256_file(bundle_seal_path),
        },
        "forbidden_surface_attestation": {
            "v1_test_accessed": False,
            "track_b_protected_results_used": False,
            "phone_performance_used_for_selection": False,
            "s2_or_s3_substituted": False,
        },
        "clean_tree_at_input": preflight.get("repository", {}).get("dirty") is False,
        "clean_tree_at_seal": binding["repository"].get("dirty") is False,
        "next_prompt_authorized": True,
        "next_prompt": "PROMPT_2",
    }
    handoff_path = write_json(work / "PROMPT1_FOUNDATION_HANDOFF.json", handoff, seal=True)
    final_root = work / "PROMPT1_FINAL_EVIDENCE"
    if final_root.exists():
        shutil.rmtree(final_root)
    final_root.mkdir(parents=True)
    copy_verified(handoff_path, final_root / handoff_path.name)
    shutil.copytree(work / "evidence", final_root / "evidence")
    shutil.copytree(production, final_root / production.name)
    shutil.copytree(restricted, final_root / restricted.name)
    failures = work / "failures"
    if failures.is_dir():
        shutil.copytree(failures, final_root / "failed_attempts")
    execution_source = final_root / "execution_source"
    copy_verified(Path(__file__), execution_source / Path(__file__).name)
    for relative in (
        "journal_extension/kaggle/CropCop_Prompt1_R07_Foundation_QA_Locked.ipynb",
        "journal_extension/kaggle/PROMPT1_KAGGLE_RUNBOOK.md",
    ):
        source = Path(inputs.repo_root) / relative
        if source.is_file():
            copy_verified(source, execution_source / source.name)
    write_json(final_root / "FINAL_FILE_MANIFEST.json", directory_manifest(final_root))
    zip_path = work / "PROMPT1_KAGGLE_EXECUTION_BUNDLE.zip"
    temporary = zip_path.with_suffix(".zip.tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(final_root.rglob("*")):
            if path.is_file():
                archive.write(path, Path(final_root.name) / path.relative_to(final_root))
    os.replace(temporary, zip_path)
    zip_verification = verify_final_zip(zip_path, final_root)
    write_json(
        work / "FINAL_DOWNLOAD.json",
        {
            "status": "PASS",
            "path": str(zip_path),
            "bytes": zip_path.stat().st_size,
            "sha256": sha256_file(zip_path),
            "round_trip_verification": zip_verification,
            "executor_sha256": sha256_file(__file__),
        },
        seal=True,
    )
    return zip_path


def stage_all(args: argparse.Namespace) -> Path:
    stage_preflight(args)
    stage_source(args)
    stage_fp32(args)
    stage_parity(args)
    if not args.skip_ptq:
        stage_ptq(args)
    else:
        write_json(
            Path(args.work_root) / "evidence" / "R07_INT8_FAILURE_CERTIFICATE_v1.json",
            {
                "schema_version": "1.0",
                "gate": "F4",
                "status": "PASS_INT8_ABSENT_BY_PROSPECTIVE_OPERATOR_CHOICE",
                "reason": "--skip-ptq was fixed before result opening",
                "no_alternate_route_attempted": True,
                "executor_sha256": sha256_file(__file__),
            },
            seal=True,
        )
    return stage_seal(args)


def stage_verify(args: argparse.Namespace) -> None:
    if not args.marker:
        raise GateError("verify requires --marker relative to the work root")
    verify_resume_marker(Path(args.work_root).resolve(), args.marker)
    print(f"VERIFIED {args.marker}", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("stage", choices=("preflight", "source", "fp32", "parity", "ptq", "seal", "all", "verify"))
    result.add_argument("--input-root", required=True)
    result.add_argument("--work-root", required=True)
    result.add_argument("--workers", type=int, default=2)
    result.add_argument("--source-batch-size", type=int, default=16)
    result.add_argument("--skip-ptq", action="store_true")
    result.add_argument("--marker")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    Path(args.work_root).mkdir(parents=True, exist_ok=True)
    dispatch = {
        "preflight": stage_preflight,
        "source": stage_source,
        "fp32": stage_fp32,
        "parity": stage_parity,
        "ptq": stage_ptq,
        "seal": stage_seal,
        "all": stage_all,
        "verify": stage_verify,
    }
    try:
        output = dispatch[args.stage](args)
        if output is not None:
            print(output)
        return 0
    except Exception as exc:
        failure_dir = Path(args.work_root) / "failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "stage": args.stage,
            "status": "FAIL_CLOSED",
            "created_at_utc": utc_now(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        write_json(failure_dir / f"{args.stage}_failure.json", failure, seal=True)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
