from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hashing import require_sha256, sha256_file, sha256_json
from .tracka_v12 import (
    AUTHORITY_ID,
    CLASS_MAP_SHA256,
    EXPERIMENT_SPECS,
    MANIFEST_SHA256,
    MNV4_MODEL_NAME,
    PAIR_IDS,
    R13_MODEL_ID,
    SEEDS,
    TEACHER_SHA256,
    TIMM_VERSION,
    TORCHVISION_VERSION,
)

G1A_SCHEMA_VERSION = "1.0"
R13_PRETRAINED_SHA256 = "92ec2d996329be8c9a449d4e38f847e79c34fadc32b6491739bacdaf425ab0ed"
R13_PRETRAINED_BYTES = 90095744
R13_PRETRAINED_COMMIT = "10143029df1b2df4c63623de6b740ecccb2b8ea2"
R13_HF_REPOSITORY = "timm/vit_dlittle_patch16_reg1_gap_256.sbb_nadamuon_in1k"
R13_CTC_MEAN = (0.485, 0.456, 0.406)
R13_CTC_STD = (0.229, 0.224, 0.225)
R13_NATIVE_MEAN = (0.5, 0.5, 0.5)
R13_NATIVE_STD = (0.5, 0.5, 0.5)
R13_PARITY_TOLERANCE = 1e-5

R12_CONSUMERS = {
    "S2": ["R12-MNV4-LOGITS-S2", "R12-MNV4-FEATURE-S2"],
    "S3": ["R12-MNV4-LOGITS-S3", "R12-MNV4-FEATURE-S3"],
}
BASELINE_CONSUMERS = {
    "effb0": {
        "S2": "R06-EFFB0-CONTEXT-S2",
        "S3": "R06-EFFB0-CONTEXT-S3",
    },
    "cnxtt": {
        "S2": "R07-CNXTT-CONTEXT-S2",
        "S3": "R07-CNXTT-CONTEXT-S3",
    },
}
R13_CONSUMERS = {
    "S1": "R13-VIT-DLITTLE-DIFF-CONTEXT-S1",
    "S2": "R13-VIT-DLITTLE-DIFF-CONTEXT-S2",
    "S3": "R13-VIT-DLITTLE-DIFF-CONTEXT-S3",
}


class TrackAV12G1AError(RuntimeError):
    pass


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def g1a_seal_hash(seal: dict[str, Any]) -> str:
    clean = dict(seal)
    clean.pop("g1a_seal_sha256", None)
    return sha256_json(clean)


def verify_principal_pair_for_reuse(
    principal_bundle: str | Path,
    *,
    seed_label: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    if seed_label not in {"S2", "S3"}:
        raise TrackAV12G1AError("G1A R12 reuse is authorized only for frozen S2/S3 pair identities")
    root = Path(principal_bundle)
    seal_path = root / "G1_MODEL_IDENTITY_SEAL.json"
    if not seal_path.is_file():
        raise TrackAV12G1AError("principal G1 seal missing")
    seal = load_json(seal_path)
    pair = seal.get("pair_initializations", {}).get(seed_label, {})
    expected_seed = SEEDS[seed_label]
    expected_pair_id = PAIR_IDS[seed_label]
    if pair.get("pair_id") != expected_pair_id or int(pair.get("seed", -1)) != expected_seed:
        raise TrackAV12G1AError(f"principal {seed_label} pair identity mismatch")
    init_sha = str(pair.get("sha256", ""))
    if len(init_sha) != 64:
        raise TrackAV12G1AError(f"principal {seed_label} pair SHA invalid")
    binary = root / "private" / str(pair.get("basename", ""))
    require_sha256(binary, init_sha, f"principal {seed_label} pair initialization")
    evidence_path = root / "evidence" / f"PAIR_INIT_{seed_label}.json"
    if not evidence_path.is_file():
        raise TrackAV12G1AError(f"principal {seed_label} pair evidence missing")
    evidence = load_json(evidence_path)
    if evidence.get("pair_id") != expected_pair_id or int(evidence.get("seed", -1)) != expected_seed:
        raise TrackAV12G1AError(f"principal {seed_label} pair evidence identity mismatch")
    if evidence.get("student_init_sha256") != init_sha:
        raise TrackAV12G1AError(f"principal {seed_label} pair evidence/binary mismatch")
    if pair.get("evidence_sha256") and sha256_json(evidence) != pair.get("evidence_sha256"):
        raise TrackAV12G1AError(f"principal {seed_label} pair evidence hash mismatch")
    return pair, evidence, binary


def build_r12_reuse_evidence(
    *,
    seed_label: str,
    pair: dict[str, Any],
    principal_pair_evidence: dict[str, Any],
    copied_binary: str | Path,
    source_git_sha: str,
    principal_g1_seal_sha256: str,
) -> dict[str, Any]:
    copied_binary = Path(copied_binary)
    expected_sha = str(pair["sha256"])
    require_sha256(copied_binary, expected_sha, f"G1A reused {seed_label} pair")
    return {
        "schema_version": "1.0",
        "reuse_kind": "exact_principal_pair_bytes",
        "seed_label": seed_label,
        "seed": SEEDS[seed_label],
        "pair_id": PAIR_IDS[seed_label],
        "model_name": MNV4_MODEL_NAME,
        "num_classes": 120,
        "pretrained_sha256": pair.get("pretrained_sha256") or principal_pair_evidence.get("pretrained_sha256"),
        "student_init_sha256": expected_sha,
        "student_init_bytes": copied_binary.stat().st_size,
        "student_init_basename": copied_binary.name,
        "authorized_consumers": R12_CONSUMERS[seed_label],
        "principal_pair_evidence_sha256": sha256_json(principal_pair_evidence),
        "principal_g1_seal_sha256": principal_g1_seal_sha256,
        "generated_under_source_sha": source_git_sha,
        "initialization_bytes_reused_exactly": True,
        "scientific_state_mutated": False,
    }


def load_r13_verified_upstream(pretrained_path: str | Path):
    import timm
    from safetensors.torch import load_file

    path = Path(pretrained_path)
    require_sha256(path, R13_PRETRAINED_SHA256, "R13 upstream safetensors")
    if path.stat().st_size != R13_PRETRAINED_BYTES:
        raise TrackAV12G1AError(
            f"R13 upstream byte count mismatch: expected {R13_PRETRAINED_BYTES}, got {path.stat().st_size}"
        )
    if timm.__version__ != TIMM_VERSION:
        raise TrackAV12G1AError(f"R13 timm drift: expected {TIMM_VERSION}, got {timm.__version__}")
    model = timm.create_model(R13_MODEL_ID, pretrained=False, num_classes=1000)
    state = load_file(str(path), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise TrackAV12G1AError(f"R13 strict state identity mismatch: missing={missing}, unexpected={unexpected}")
    return model


def apply_r13_ctc_normalization_equivalence(model) -> None:
    import torch

    conv = model.patch_embed.proj
    if getattr(conv, "bias", None) is None:
        raise TrackAV12G1AError("R13 patch_embed.proj has no bias; normalization-equivalence contract fails closed")
    with torch.no_grad():
        native_mean = torch.tensor(R13_NATIVE_MEAN, dtype=conv.weight.dtype, device=conv.weight.device)
        native_std = torch.tensor(R13_NATIVE_STD, dtype=conv.weight.dtype, device=conv.weight.device)
        ctc_mean = torch.tensor(R13_CTC_MEAN, dtype=conv.weight.dtype, device=conv.weight.device)
        ctc_std = torch.tensor(R13_CTC_STD, dtype=conv.weight.dtype, device=conv.weight.device)
        w_native = conv.weight.detach().clone()
        b_native = conv.bias.detach().clone()
        conv.weight.copy_(w_native * (ctc_std / native_std).view(1, 3, 1, 1))
        offset = ((ctc_mean - native_mean) / native_std).view(1, 3, 1, 1)
        conv.bias.copy_(b_native + (w_native * offset).sum(dim=(1, 2, 3)))


def r13_patch_parity_max_abs(native_model, ctc_model, rgb_batch) -> float:
    import torch

    native_conv = native_model.patch_embed.proj
    ctc_conv = ctc_model.patch_embed.proj
    dtype = native_conv.weight.dtype
    device = native_conv.weight.device
    rgb_batch = rgb_batch.to(device=device, dtype=dtype)
    native_mean = torch.tensor(R13_NATIVE_MEAN, dtype=dtype, device=device).view(1, 3, 1, 1)
    native_std = torch.tensor(R13_NATIVE_STD, dtype=dtype, device=device).view(1, 3, 1, 1)
    ctc_mean = torch.tensor(R13_CTC_MEAN, dtype=dtype, device=device).view(1, 3, 1, 1)
    ctc_std = torch.tensor(R13_CTC_STD, dtype=dtype, device=device).view(1, 3, 1, 1)
    with torch.no_grad():
        native = native_conv((rgb_batch - native_mean) / native_std)
        adapted = ctc_conv((rgb_batch - ctc_mean) / ctc_std)
    return float((native - adapted).abs().max().item())


def deterministic_r13_reset_classifier(model, *, seed: int, num_classes: int = 120) -> None:
    import torch

    if int(seed) not in set(SEEDS.values()):
        raise TrackAV12G1AError(f"R13 classifier reset refused for unauthorized seed {seed}")
    classifier = model.get_classifier()
    in_features = int(classifier.in_features)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model.head = torch.nn.Linear(in_features, int(num_classes))


def save_r13_initialization(
    model,
    path: str | Path,
    *,
    experiment_id: str,
    seed: int,
    parity_max_abs: float,
) -> str:
    import torch

    spec = EXPERIMENT_SPECS.get(experiment_id)
    if spec is None or spec.get("model_family") != "vit_dlittle_diff":
        raise TrackAV12G1AError(f"unauthorized R13 consumer {experiment_id}")
    if int(spec["seed"]) != int(seed):
        raise TrackAV12G1AError("R13 initialization seed/experiment mismatch")
    if float(parity_max_abs) > R13_PARITY_TOLERANCE:
        raise TrackAV12G1AError(f"R13 normalization parity failed: {parity_max_abs}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "1.0",
            "model_family": "vit_dlittle_diff",
            "model_id": R13_MODEL_ID,
            "timm_version": TIMM_VERSION,
            "seed": int(seed),
            "num_classes": 120,
            "pretrained_sha256": R13_PRETRAINED_SHA256,
            "normalization_equivalence_applied": True,
            "normalization_parity_max_abs": float(parity_max_abs),
            "authorized_consumers": [experiment_id],
            "model_state": model.state_dict(),
        },
        path,
    )
    return sha256_file(path)


def load_r13_initialization(
    path: str | Path,
    *,
    expected_sha256: str,
    experiment_id: str,
    seed: int,
):
    import timm
    import torch

    require_sha256(path, expected_sha256, "R13 initialization")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = EXPERIMENT_SPECS.get(experiment_id)
    if expected is None or expected.get("model_family") != "vit_dlittle_diff":
        raise TrackAV12G1AError("experiment is not an authorized R13 consumer")
    if payload.get("model_id") != R13_MODEL_ID or payload.get("timm_version") != TIMM_VERSION:
        raise TrackAV12G1AError("R13 initialization model identity mismatch")
    if int(payload.get("seed", -1)) != int(seed) or int(expected["seed"]) != int(seed):
        raise TrackAV12G1AError("R13 initialization seed mismatch")
    if payload.get("pretrained_sha256") != R13_PRETRAINED_SHA256:
        raise TrackAV12G1AError("R13 initialization pretrained identity mismatch")
    if payload.get("normalization_equivalence_applied") is not True:
        raise TrackAV12G1AError("R13 normalization-equivalence marker missing")
    if float(payload.get("normalization_parity_max_abs", 1.0)) > R13_PARITY_TOLERANCE:
        raise TrackAV12G1AError("R13 initialization parity evidence exceeds tolerance")
    if payload.get("authorized_consumers") != [experiment_id]:
        raise TrackAV12G1AError("R13 initialization consumer mismatch")
    if timm.__version__ != TIMM_VERSION:
        raise TrackAV12G1AError(f"R13 timm drift: {timm.__version__}")
    model = timm.create_model(R13_MODEL_ID, pretrained=False, num_classes=120)
    model.load_state_dict(payload["model_state"], strict=True)
    return model, payload


def validate_g1a_seal_object(seal: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if seal.get("schema_version") != G1A_SCHEMA_VERSION:
        errors.append("G1A schema version mismatch")
    if seal.get("status") != "PASS":
        errors.append("G1A seal status is not PASS")
    if seal.get("science_authorized") is not False:
        errors.append("G1A seal must not independently authorize science")
    authority = seal.get("authority", {})
    if authority.get("id") != AUTHORITY_ID:
        errors.append("G1A scientific authority mismatch")
    dataset = seal.get("dataset", {})
    if dataset.get("manifest_sha256") != MANIFEST_SHA256 or dataset.get("class_map_sha256") != CLASS_MAP_SHA256:
        errors.append("G1A dataset identity mismatch")
    if dataset.get("v1_test_accessed") is not False or dataset.get("external_surface_accessed") is not False:
        errors.append("G1A protected-surface marker invalid")

    r12 = seal.get("r12_reused_pairs", {})
    for label in ("S2", "S3"):
        row = r12.get(label, {})
        if row.get("pair_id") != PAIR_IDS[label] or int(row.get("seed", -1)) != SEEDS[label]:
            errors.append(f"G1A R12 {label} pair identity mismatch")
        if row.get("authorized_consumers") != R12_CONSUMERS[label]:
            errors.append(f"G1A R12 {label} consumer mismatch")
        if row.get("initialization_bytes_reused_exactly") is not True:
            errors.append(f"G1A R12 {label} was not exact byte reuse")
        if len(str(row.get("student_init_sha256", ""))) != 64:
            errors.append(f"G1A R12 {label} init SHA invalid")

    baselines = seal.get("baselines", {})
    for key in ("effb0", "cnxtt"):
        for label in ("S2", "S3"):
            row = baselines.get(key, {}).get(label, {})
            if int(row.get("seed", -1)) != SEEDS[label]:
                errors.append(f"G1A {key} {label} seed mismatch")
            if row.get("authorized_consumers") != [BASELINE_CONSUMERS[key][label]]:
                errors.append(f"G1A {key} {label} consumer mismatch")
            for field in ("pretrained_sha256", "init_sha256", "init_evidence_sha256"):
                if len(str(row.get(field, ""))) != 64:
                    errors.append(f"G1A {key} {label} {field} invalid")

    r13 = seal.get("r13", {})
    if r13.get("model_id") != R13_MODEL_ID or r13.get("timm_version") != TIMM_VERSION:
        errors.append("G1A R13 model identity mismatch")
    if r13.get("pretrained_sha256") != R13_PRETRAINED_SHA256 or int(r13.get("pretrained_bytes", -1)) != R13_PRETRAINED_BYTES:
        errors.append("G1A R13 pretrained identity mismatch")
    if float(r13.get("parity_max_abs", 1.0)) > R13_PARITY_TOLERANCE:
        errors.append("G1A R13 parity gate failed")
    states = r13.get("states", {})
    for label in ("S1", "S2", "S3"):
        row = states.get(label, {})
        if int(row.get("seed", -1)) != SEEDS[label]:
            errors.append(f"G1A R13 {label} seed mismatch")
        if row.get("authorized_consumers") != [R13_CONSUMERS[label]]:
            errors.append(f"G1A R13 {label} consumer mismatch")
        for field in ("init_sha256", "init_evidence_sha256"):
            if len(str(row.get(field, ""))) != 64:
                errors.append(f"G1A R13 {label} {field} invalid")

    if seal.get("g1a_seal_sha256") != g1a_seal_hash(seal):
        errors.append("G1A seal self-hash mismatch")
    return errors
