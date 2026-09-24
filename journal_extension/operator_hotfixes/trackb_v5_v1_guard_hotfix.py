from __future__ import annotations

import os
import selectors
import subprocess
import time
from collections import deque
from typing import Any

HOTFIX_ID = "TRACKB_V5_V1_GUARD_FALSE_POSITIVE_FIX_v1"
ACTIVE_ENV = "TRACKB_V1_GUARD_HOTFIX_ACTIVE"


class GuardHotfixError(RuntimeError):
    pass


def _path_is_forbidden(value: str) -> bool:
    normalized = "/" + str(value).replace("\\", "/").strip().strip("/").lower() + "/"
    if "/ds-v1-test-consumed/" in normalized or "/ds-v1-test/" in normalized:
        return True
    if "/v1_test/" in normalized or "/v1-test/" in normalized:
        return True
    if "/test_consumed/" in normalized:
        return True
    return False


def _iter_path_values(obj: Any):
    if isinstance(obj, dict):
        for child_key, value in obj.items():
            child_lower = str(child_key).lower()
            if isinstance(value, str) and (
                child_lower == "path"
                or child_lower in {
                    "data_root",
                    "repository_root",
                    "v1_validation_root",
                    "validation_root",
                    "image_root",
                    "source_root",
                }
                or child_lower.endswith(("_path", "_root", "_dir"))
            ):
                yield str(child_key), value
            else:
                yield from _iter_path_values(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_path_values(value)


def _iter_string_values(obj: Any):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_string_values(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_string_values(value)
    elif isinstance(obj, str):
        yield obj


def _iter_access_flags(obj: Any):
    if isinstance(obj, dict):
        for child_key, value in obj.items():
            child_lower = str(child_key).lower()
            if child_lower in {
                "v1_test_accessed",
                "v1_test_image_bytes_accessed",
                "consumed_v1_test_accessed",
            }:
                yield str(child_key), value
            yield from _iter_access_flags(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_access_flags(value)


def validate_manifest_safety(role: str, manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise GuardHotfixError(f"Track-B input manifest is not an object: role={role!r}")

    for value in _iter_string_values(manifest):
        if value.strip().upper() == "DS-V1-TEST-CONSUMED":
            raise GuardHotfixError(
                f"forbidden consumed V1-test surface referenced by input role {role!r}"
            )

    for key, value in _iter_access_flags(manifest):
        if value is not False:
            raise GuardHotfixError(
                f"consumed V1-test access flag is not explicitly false: role={role!r}, "
                f"field={key!r}, value={value!r}"
            )

    for key, value in _iter_path_values(manifest):
        if _path_is_forbidden(value):
            raise GuardHotfixError(
                f"forbidden consumed V1-test path referenced by input role {role!r}: "
                f"field={key!r}, value={value!r}"
            )

    if role == "historical_compare":
        if manifest.get("coverage_scope") != "V1_TRAIN_VAL_ONLY":
            raise GuardHotfixError(
                "historical_compare must remain on the V1_TRAIN_VAL_ONLY safe surface"
            )
        if manifest.get("v1_test_image_bytes_accessed") is not False:
            raise GuardHotfixError(
                "historical_compare must explicitly assert v1_test_image_bytes_accessed=false"
            )
        if manifest.get("ext_i_eligible") is not False:
            raise GuardHotfixError(
                "historical_compare must remain ineligible for EXT-I under the safe route"
            )
        if manifest.get("maximum_evidence_grade") != "EXT-S":
            raise GuardHotfixError(
                "historical_compare safe route must remain capped at EXT-S"
            )


def _optimized_topk_cosine_neighbors(
    query_features,
    reference_features,
    *,
    k: int = 50,
    device="cuda",
    block_rows: int = 256,
):
    """Exact-equivalent deterministic cosine top-k with vectorized tie detection."""
    import numpy as np
    import torch
    from cropcop_je.trackb_r07 import TrackBError

    q = np.asarray(query_features, dtype=np.float32)
    r = np.asarray(reference_features, dtype=np.float32)
    if q.ndim != 2 or r.ndim != 2 or q.shape[1] != r.shape[1]:
        raise TrackBError("feature matrix dimensionality mismatch")
    if len(r) < k:
        raise TrackBError(f"reference feature surface has fewer than top-k={k} rows")

    ref = torch.from_numpy(r).to(device)
    out_idx = np.empty((len(q), k), dtype=np.int64)
    out_score = np.empty((len(q), k), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(q), int(block_rows)):
            block = torch.from_numpy(q[start:start + int(block_rows)]).to(device)
            score = block @ ref.T
            values, indices = torch.topk(score, k=k, dim=1, largest=True, sorted=True)

            threshold = values[:, -1:].clone()
            strict_counts = (score > threshold).sum(dim=1)
            equal_counts = (score == threshold).sum(dim=1)
            ambiguous = (strict_counts + equal_counts) > int(k)

            index_order = torch.argsort(indices, dim=1, stable=True)
            canonical_idx = torch.gather(indices, 1, index_order)
            canonical_values = torch.gather(values, 1, index_order)
            score_order = torch.argsort(
                canonical_values, dim=1, descending=True, stable=True
            )
            indices = torch.gather(canonical_idx, 1, score_order)
            values = torch.gather(canonical_values, 1, score_order)

            for row_index in torch.nonzero(
                ambiguous, as_tuple=False
            ).flatten().tolist():
                row_threshold = threshold[row_index, 0]
                strict_idx = torch.nonzero(
                    score[row_index] > row_threshold, as_tuple=False
                ).flatten()
                tie_idx = torch.nonzero(
                    score[row_index] == row_threshold, as_tuple=False
                ).flatten()
                slots = int(k) - int(strict_idx.numel())
                if slots < 0:
                    raise TrackBError("top-k cutoff accounting became inconsistent")
                chosen_ties = torch.sort(tie_idx).values[:slots]
                chosen = torch.cat((strict_idx, chosen_ties), dim=0)
                if int(chosen.numel()) != int(k):
                    raise TrackBError(
                        "deterministic top-k tie resolution did not produce k neighbors"
                    )
                chosen_scores = score[row_index, chosen]
                order = torch.argsort(
                    chosen_scores, descending=True, stable=True
                )
                indices[row_index] = chosen[order]
                values[row_index] = chosen_scores[order]

            out_idx[start:start + len(block)] = indices.cpu().numpy()
            out_score[start:start + len(block)] = values.cpu().numpy()

            completed = min(start + len(block), len(q))
            if completed == len(q) or completed % max(int(block_rows) * 16, 1) == 0:
                print(
                    f"DINO top-k {completed:,}/{len(q):,} queries against "
                    f"{len(r):,} references",
                    flush=True,
                )
    return out_idx, out_score


def selftest_dino_topk_equivalence() -> dict[str, object]:
    """Prove optimized top-k equals the original reference under the active Torch runtime."""
    import numpy as np
    import torch
    from cropcop_je.trackb_r07 import TrackBError

    def reference(q, r, *, k):
        q = np.asarray(q, dtype=np.float32)
        r = np.asarray(r, dtype=np.float32)
        ref = torch.from_numpy(r)
        out_idx = np.empty((len(q), k), dtype=np.int64)
        out_score = np.empty((len(q), k), dtype=np.float32)
        with torch.no_grad():
            score = torch.from_numpy(q) @ ref.T
            values, indices = torch.topk(score, k=k, dim=1, largest=True, sorted=True)
            for row_index in range(len(q)):
                threshold = values[row_index, -1]
                strict_idx = torch.nonzero(
                    score[row_index] > threshold, as_tuple=False
                ).flatten()
                tie_idx = torch.nonzero(
                    score[row_index] == threshold, as_tuple=False
                ).flatten()
                slots = int(k) - int(strict_idx.numel())
                if slots < 0:
                    raise TrackBError("reference top-k cutoff accounting became inconsistent")
                chosen_ties = torch.sort(tie_idx).values[:slots]
                chosen = torch.cat((strict_idx, chosen_ties), dim=0)
                chosen_scores = score[row_index, chosen]
                order = torch.argsort(chosen_scores, descending=True, stable=True)
                out_idx[row_index] = chosen[order].cpu().numpy()
                out_score[row_index] = chosen_scores[order].cpu().numpy()
        return out_idx, out_score

    rng = np.random.default_rng(1701)
    q = rng.normal(size=(37, 23)).astype(np.float32)
    r = rng.normal(size=(113, 23)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    r /= np.linalg.norm(r, axis=1, keepdims=True)

    cases = [
        ("random", q, r, 11),
        (
            "cutoff_ties",
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            np.asarray([
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.5, 0.5],
                [0.5, 0.5],
                [0.0, 1.0],
                [0.0, 1.0],
            ], dtype=np.float32),
            2,
        ),
    ]

    for name, case_q, case_r, k in cases:
        expected_idx, expected_score = reference(case_q, case_r, k=k)
        observed_idx, observed_score = _optimized_topk_cosine_neighbors(
            case_q, case_r, k=k, device="cpu", block_rows=8
        )
        if not np.array_equal(expected_idx, observed_idx):
            raise GuardHotfixError(f"DINO top-k equivalence index mismatch: {name}")
        if not np.array_equal(expected_score, observed_score):
            raise GuardHotfixError(f"DINO top-k equivalence score mismatch: {name}")

    return {
        "status": "PASS_DINO_TOPK_EXACT_EQUIVALENCE",
        "torch_version": str(torch.__version__),
        "cases": [name for name, *_rest in cases],
    }


def _observable_encode_audit_features(
    model,
    image_paths,
    transform,
    device,
    *,
    batch_size: int = 64,
):
    """Exact-equivalent DINO encoding with bounded progress logging."""
    import numpy as np
    import torch
    from cropcop_je.trackb_r07_audit import _canonical_rgb

    model = model.to(device).eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(image_paths), int(batch_size)):
            batch_paths = image_paths[start:start + int(batch_size)]
            tensors = [transform(_canonical_rgb(path)) for path in batch_paths]
            x = torch.stack(tensors, dim=0).to(device, non_blocking=True)
            features = model.forward_features(x)
            features = model.forward_head(features, pre_logits=True)
            features = torch.nn.functional.normalize(features.float(), p=2, dim=1)
            outputs.append(features.cpu().numpy().astype(np.float32, copy=False))
            completed = min(start + len(batch_paths), len(image_paths))
            if completed == len(image_paths) or completed % 2048 < len(batch_paths):
                print(
                    f"DINO feature encoding {completed:,}/{len(image_paths):,}",
                    flush=True,
                )
    if not outputs:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def _stream_science_command(original_run_checked, ops_module):
    def run_checked(args, *, cwd=None, timeout=3600):
        is_science_runner = any(
            str(part).endswith("/run_trackb_r07.py")
            or str(part).endswith("\\run_trackb_r07.py")
            for part in args
        )
        if not is_science_runner:
            return original_run_checked(args, cwd=cwd, timeout=timeout)

        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        started = time.monotonic()
        tail = deque(maxlen=200)
        proc = subprocess.Popen(
            args,
            cwd=None if cwd is None else str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        selector = selectors.DefaultSelector()
        try:
            assert proc.stdout is not None
            selector.register(proc.stdout, selectors.EVENT_READ)
            while True:
                elapsed = time.monotonic() - started
                remaining = float(timeout) - elapsed
                if remaining <= 0:
                    proc.terminate()
                    try:
                        proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=20)
                    raise subprocess.TimeoutExpired(args, timeout)

                events = selector.select(timeout=min(1.0, remaining))
                if events:
                    line = proc.stdout.readline()
                    if line:
                        safe = ops_module.redact(line.rstrip("\n"))
                        tail.append(safe)
                        print(safe, flush=True)

                if proc.poll() is not None:
                    # Drain any complete lines already buffered at process exit.
                    for line in proc.stdout:
                        safe = ops_module.redact(line.rstrip("\n"))
                        tail.append(safe)
                        print(safe, flush=True)
                    break
            rc = proc.wait()
        finally:
            selector.close()
            if proc.stdout is not None:
                proc.stdout.close()

        if rc != 0:
            detail = "\n".join(tail)
            if len(detail) > 12000:
                detail = detail[-12000:]
            raise ops_module.TrackBOpsError(
                f"command failed rc={rc}: {' '.join(map(str, args))}\n{detail}"
            )
        return subprocess.CompletedProcess(args, rc, stdout="", stderr="")

    return run_checked


def install(expected_source_sha256: str) -> dict[str, str]:
    expected_source_sha256 = str(expected_source_sha256).strip().lower()
    if len(expected_source_sha256) != 64 or any(
        ch not in "0123456789abcdef" for ch in expected_source_sha256
    ):
        raise GuardHotfixError("operator hotfix source SHA-256 is missing or malformed")

    from cropcop_je import trackb_r07
    from cropcop_je import trackb_r07_audit
    from cropcop_je import trackb_r07_ops

    def _strict_no_v1_test_surface(inputs):
        for bundle in inputs.values():
            validate_manifest_safety(str(bundle.role), bundle.manifest)

    trackb_r07.assert_no_v1_test_surface = _strict_no_v1_test_surface
    trackb_r07_audit.topk_cosine_neighbors = _optimized_topk_cosine_neighbors
    trackb_r07_audit.encode_audit_features = _observable_encode_audit_features

    original_run_checked = trackb_r07_ops.run_checked
    trackb_r07_ops.run_checked = _stream_science_command(
        original_run_checked, trackb_r07_ops
    )

    os.environ[ACTIVE_ENV] = expected_source_sha256
    return {
        "status": "PASS_TRACKB_OPERATOR_HOTFIX_INSTALLED",
        "dino_topk_optimization": "EXACT_EQUIVALENT_VECTORIZED_CUTOFF_TIE_DETECTION",
        "dino_feature_progress_logging": True,
        "science_subprocess_live_streaming": True,
        "hotfix_id": HOTFIX_ID,
        "source_sha256": expected_source_sha256,
    }
