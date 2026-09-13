from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from typing import Any

from .data import deterministic_pad_resize
from .tracka_v12_evidence import deterministic_subset, xai_random_control_seed, xai_sample

XAI_FRACTIONS = (0.1, 0.2, 0.3)
R13_TARGET_PATH = "blocks.13.norm1"
R13_PREFIX_TOKENS = 1
R13_GRID = (16, 16)
R13_CHANNELS = 320


def module_by_path(model, path: str):
    current = model
    for part in path.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


def resolve_cnn_target_layer(model, device) -> str:
    import torch

    candidates: list[str] = []
    hooks = []
    modules = dict(model.named_modules())
    for path, module in modules.items():
        if isinstance(module, torch.nn.Conv2d) and not any(True for _ in module.children()):
            def hook(_module, _inputs, output, path=path):
                if hasattr(output, "shape") and len(output.shape) == 4 and int(output.shape[-2]) > 1 and int(output.shape[-1]) > 1:
                    candidates.append(path)
            hooks.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model.eval()
            _ = model(torch.zeros((1, 3, 256, 256), dtype=torch.float32, device=device))
    finally:
        for hook in hooks:
            hook.remove()
    if not candidates:
        raise RuntimeError("no qualifying spatial Conv2d target found for frozen CNN Grad-CAM++ rule")
    return candidates[-1]


def validate_r13_target(model) -> Any:
    module = module_by_path(model, R13_TARGET_PATH)
    prefix = int(getattr(model, "num_prefix_tokens", -1))
    grid = tuple(int(value) for value in getattr(model.patch_embed, "grid_size", ()))
    if prefix != R13_PREFIX_TOKENS or grid != R13_GRID:
        raise RuntimeError(f"R13 token contract drift: prefix={prefix}, grid={grid}")
    return module


def _r13_reshape(tensor):
    import torch

    if tensor.ndim != 3 or tuple(tensor.shape[1:]) != (257, R13_CHANNELS):
        raise RuntimeError(f"R13 target tensor contract drift: {tuple(tensor.shape)}")
    patches = tensor[:, R13_PREFIX_TOKENS:, :]
    return patches.reshape(tensor.shape[0], 16, 16, R13_CHANNELS).permute(0, 3, 1, 2).contiguous()


def _normalize_cam(cam):
    import torch

    if not torch.isfinite(cam).all():
        return None, False, True
    minimum = cam.min()
    maximum = cam.max()
    if float((maximum - minimum).abs().item()) <= 1e-12:
        return torch.zeros_like(cam), True, False
    return (cam - minimum) / (maximum - minimum), False, False


def gradcampp(model, x, target_module, *, target_class: int | None = None, reshape_transform=None):
    import torch
    import torch.nn.functional as F

    activations = []
    gradients = []

    def forward_hook(_module, _inputs, output):
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    handle = target_module.register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise RuntimeError("Grad-CAM++ executor requires one image per attribution")
        predicted = int(logits.argmax(dim=1).item())
        target = predicted if target_class is None else int(target_class)
        score = logits[0, target]
        score.backward(retain_graph=False)
        if len(activations) != 1 or len(gradients) != 1:
            raise RuntimeError("target layer hook did not capture exactly one activation/gradient tensor")
        activation = activations[0]
        gradient = gradients[0]
        if reshape_transform is not None:
            activation = reshape_transform(activation)
            gradient = reshape_transform(gradient)
        if activation.ndim != 4 or gradient.shape != activation.shape:
            raise RuntimeError("Grad-CAM++ target representation must be matching BCHW activation/gradient")
        grad2 = gradient.pow(2)
        grad3 = gradient.pow(3)
        spatial_sum = activation.sum(dim=(2, 3), keepdim=True)
        denominator = 2.0 * grad2 + spatial_sum * grad3
        denominator = torch.where(denominator.abs() > 1e-12, denominator, torch.ones_like(denominator))
        alpha = grad2 / denominator
        positive_grad = torch.relu(gradient)
        weights = (alpha * positive_grad).sum(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(256, 256), mode="bilinear", align_corners=False)[0, 0]
        normalized, degenerate, nonfinite = _normalize_cam(cam)
        probability = float(torch.softmax(logits.detach(), dim=1)[0, target].item())
        return {
            "predicted_class_index": predicted,
            "target_class_index": target,
            "target_probability": probability,
            "heatmap": normalized.detach() if normalized is not None else None,
            "degenerate": bool(degenerate),
            "nonfinite": bool(nonfinite),
        }
    finally:
        handle.remove()


def pil_working_image(image):
    return deterministic_pad_resize(image, 256)


def image_to_normalized_tensor(image):
    import torch
    from torchvision.transforms import functional as TF

    x = TF.pil_to_tensor(image).to(torch.float32).div_(255.0)
    return TF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def _channel_mean_uint8(image):
    import numpy as np

    array = np.asarray(image, dtype=np.uint8)
    means = np.rint(array.reshape(-1, 3).mean(axis=0)).clip(0, 255).astype(np.uint8)
    return array, means


def _saliency_indices(heatmap, fraction: float):
    import numpy as np

    if fraction not in XAI_FRACTIONS:
        raise ValueError("deletion fraction outside frozen XAI protocol")
    flat = heatmap.detach().cpu().numpy().reshape(-1)
    count = int(math.ceil(fraction * flat.size))
    indices = np.arange(flat.size, dtype=np.int64)
    order = np.lexsort((indices, -flat))
    return order[:count]


def _mask_indices(image, indices):
    from PIL import Image
    import numpy as np

    array, means = _channel_mean_uint8(image)
    result = array.copy().reshape(-1, 3)
    result[indices] = means
    return Image.fromarray(result.reshape(array.shape), mode="RGB")


def _probability(model, tensor, target_class: int, device) -> float:
    import torch

    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device))
        return float(torch.softmax(logits, dim=1)[0, int(target_class)].item())


def deletion_faithfulness(model, working_image, heatmap, *, stable_row_id: str, model_run_id: str, target_class: int, device):
    import numpy as np

    original = _probability(model, image_to_normalized_tensor(working_image), target_class, device)
    outputs = []
    total_pixels = 256 * 256
    for fraction in XAI_FRACTIONS:
        indices = _saliency_indices(heatmap, fraction)
        saliency_image = _mask_indices(working_image, indices)
        saliency_after = _probability(model, image_to_normalized_tensor(saliency_image), target_class, device)
        random_after = []
        for control_index in range(10):
            seed = xai_random_control_seed(stable_row_id, model_run_id, fraction, control_index)
            rng = np.random.Generator(np.random.PCG64(seed))
            random_indices = rng.choice(total_pixels, size=len(indices), replace=False)
            random_image = _mask_indices(working_image, random_indices)
            random_after.append(_probability(model, image_to_normalized_tensor(random_image), target_class, device))
        random_mean = sum(random_after) / len(random_after)
        outputs.append(
            {
                "fraction": fraction,
                "deleted_pixel_count": len(indices),
                "predicted_class_probability_before_masking": original,
                "probability_after_saliency_deletion": saliency_after,
                "mean_probability_after_random_deletion": random_mean,
                "saliency_confidence_drop": original - saliency_after,
                "random_confidence_drop": original - random_mean,
                "saliency_minus_random_confidence_drop_advantage": random_mean - saliency_after,
            }
        )
    return outputs


def spearman_flat(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size or a.size == 0:
        raise ValueError("Spearman inputs must be non-empty and equal length")

    def ranks(values):
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        result = np.empty(values.size, dtype=np.float64)
        start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and sorted_values[end] == sorted_values[start]:
                end += 1
            rank = (start + end - 1) / 2.0 + 1.0
            result[order[start:end]] = rank
            start = end
        return result

    ra = ranks(a)
    rb = ranks(b)
    if float(ra.std()) == 0.0 or float(rb.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def classifier_module(model):
    if hasattr(model, "get_classifier"):
        module = model.get_classifier()
        if module is not None:
            return module
    if hasattr(model, "classifier"):
        return model.classifier
    if hasattr(model, "head"):
        return model.head
    raise RuntimeError("unable to resolve classifier head for frozen randomization sanity check")


def deterministic_randomize_classifier(model, *, seed: int):
    import torch

    head = classifier_module(model)
    original = copy.deepcopy(head.state_dict())
    devices = []
    if torch.cuda.is_available():
        devices = [torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        for module in head.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
    return head, original


def restore_classifier(head, original_state) -> None:
    head.load_state_dict(original_state, strict=True)


def xai_sample_plan(manifest_rows):
    rows = [{"stable_row_id": row.stable_row_id, "class_index": int(row.class_index)} for row in manifest_rows]
    selected = xai_sample(rows)
    return {
        "sample_240": selected,
        "randomization_30": deterministic_subset(selected, count=30, salt="TRACKA-A1-XAI-RANDOMIZE-V1"),
        "flip_30": deterministic_subset(selected, count=30, salt="TRACKA-A1-XAI-FLIP-V1"),
        "panel_12": deterministic_subset(selected, count=12, salt="TRACKA-A1-XAI-PANEL-V1"),
    }


def target_for_family(model, family: str, device):
    if family == "R13":
        return R13_TARGET_PATH, validate_r13_target(model), _r13_reshape
    if family in {"R04", "R06", "R07"}:
        path = resolve_cnn_target_layer(model, device)
        return path, module_by_path(model, path), None
    raise ValueError(f"unsupported XAI family: {family}")
