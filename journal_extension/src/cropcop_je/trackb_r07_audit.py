from __future__ import annotations

import hashlib
import json
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .trackb_r07 import TrackBAuditPolicy, TrackBError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_CV2_RANSAC_LOCK = threading.Lock()


@dataclass(frozen=True)
class ImageAuditRecord:
    row_id: str
    relative_path: str
    source_label: str
    bytes: int
    sha256: str
    phash64: int
    dhash64: int
    width: int
    height: int


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {str(x): str(x) for x in items}
        self.rank = {str(x): 0 for x in items}

    def find(self, x: str) -> str:
        parent = self.parent[x]
        if parent != x:
            self.parent[x] = self.find(parent)
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def components(self) -> list[list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for item in self.parent:
            groups[self.find(item)].append(item)
        return [sorted(values) for values in groups.values()]


class BKTree64:
    """Compact Hamming-distance BK-tree for 64-bit hashes."""

    def __init__(self):
        self.root: tuple[int, dict[int, Any]] | None = None

    @staticmethod
    def distance(a: int, b: int) -> int:
        return int(a ^ b).bit_count()

    def add(self, value: int) -> None:
        value = int(value)
        if self.root is None:
            self.root = (value, {})
            return
        node = self.root
        while True:
            current, children = node
            dist = self.distance(value, current)
            child = children.get(dist)
            if child is None:
                children[dist] = (value, {})
                return
            node = child

    def query(self, value: int, radius: int) -> list[int]:
        if self.root is None:
            return []
        out: list[int] = []
        queue = deque([self.root])
        while queue:
            current, children = queue.popleft()
            dist = self.distance(int(value), current)
            if dist <= radius:
                out.append(current)
            lo, hi = dist - radius, dist + radius
            for edge, child in children.items():
                if lo <= edge <= hi:
                    queue.append(child)
        return out


def _pil_dependencies():
    from PIL import Image, ImageOps
    return Image, ImageOps


def _canonical_rgb(path: str | Path):
    Image, ImageOps = _pil_dependencies()
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB").copy()


def phash64(image) -> int:
    import cv2
    import numpy as np
    from PIL import Image

    gray = image.convert("L").resize((32, 32), resample=Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32)
    dct = cv2.dct(arr)
    low = dct[:8, :8]
    median = float(np.median(low))
    bits = (low > median).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value


def dhash64(image) -> int:
    import numpy as np
    from PIL import Image

    gray = image.convert("L").resize((9, 8), resample=Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.uint8)
    bits = (arr[:, 1:] > arr[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _record_from_canonical_image(path: Path, root: Path, image, *, source_label: str) -> ImageAuditRecord:
    raw_sha = sha256_file(path)
    rel = path.relative_to(root).as_posix()
    row_id = hashlib.sha256(f"{rel}|{raw_sha}".encode("utf-8")).hexdigest()
    return ImageAuditRecord(
        row_id=row_id,
        relative_path=rel,
        source_label=str(source_label),
        bytes=path.stat().st_size,
        sha256=raw_sha,
        phash64=phash64(image),
        dhash64=dhash64(image),
        width=int(image.width),
        height=int(image.height),
    )


def make_image_record(path: str | Path, root: str | Path, *, source_label: str) -> ImageAuditRecord:
    path = Path(path).resolve()
    root = Path(root).resolve()
    if root not in path.parents and path != root:
        raise TrackBError(f"image path escapes candidate root: {path}")
    image = _canonical_rgb(path)
    return _record_from_canonical_image(path, root, image, source_label=source_label)


def make_image_record_and_orb(
    path: str | Path,
    root: str | Path,
    *,
    source_label: str = "",
    max_side: int = 800,
    nfeatures: int = 1200,
) -> tuple[ImageAuditRecord, dict[str, Any]]:
    """Decode once to produce the immutable hash record and frozen ORB representation."""
    path = Path(path).resolve()
    root = Path(root).resolve()
    if root not in path.parents and path != root:
        raise TrackBError(f"image path escapes candidate root: {path}")
    image = _canonical_rgb(path)
    record = _record_from_canonical_image(path, root, image, source_label=source_label)
    orb = orb_features_from_image(image, max_side=max_side, nfeatures=nfeatures)
    return record, orb


def discover_candidate_images(
    root: str | Path,
    *,
    eligible_subtree: str | None = None,
    allowed_labels: set[str] | None = None,
) -> list[tuple[Path, str]]:
    root = Path(root).resolve()
    if eligible_subtree:
        matches = [p for p in root.rglob(eligible_subtree) if p.is_dir()]
        if len(matches) != 1:
            raise TrackBError(f"expected exactly one {eligible_subtree!r} directory under {root}, found {len(matches)}")
        scan_root = matches[0]
    else:
        scan_root = root
    rows: list[tuple[Path, str]] = []
    for path in sorted(p for p in scan_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
        label = path.parent.name
        if allowed_labels is not None and label not in allowed_labels:
            continue
        rows.append((path, label))
    return rows


def exact_duplicate_pairs(records: list[ImageAuditRecord]) -> set[tuple[str, str]]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for row in records:
        by_hash[row.sha256].append(row.row_id)
    edges: set[tuple[str, str]] = set()
    for ids in by_hash.values():
        ids = sorted(ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                edges.add((ids[i], ids[j]))
    return edges


def near_hash_pairs(records: list[ImageAuditRecord], *, field: str, radius: int) -> set[tuple[str, str]]:
    tree = BKTree64()
    value_to_ids: dict[int, list[str]] = defaultdict(list)
    edges: set[tuple[str, str]] = set()
    for row in records:
        value = int(getattr(row, field))
        for neighbor_value in tree.query(value, radius):
            for other_id in value_to_ids[neighbor_value]:
                a, b = sorted((row.row_id, other_id))
                if a != b:
                    edges.add((a, b))
        if not value_to_ids[value]:
            tree.add(value)
        value_to_ids[value].append(row.row_id)
    return edges


def cross_hash_pairs(
    external_records: list[ImageAuditRecord],
    historical_rows: list[dict[str, Any]],
    *,
    field: str,
    radius: int,
) -> set[tuple[str, str]]:
    tree = BKTree64()
    value_to_hist: dict[int, list[str]] = defaultdict(list)
    for row in historical_rows:
        value = int(row[field])
        if not value_to_hist[value]:
            tree.add(value)
        value_to_hist[value].append(str(row["hist_id"]))
    out: set[tuple[str, str]] = set()
    for row in external_records:
        value = int(getattr(row, field))
        for neighbor_value in tree.query(value, radius):
            for hist_id in value_to_hist[neighbor_value]:
                out.add((row.row_id, hist_id))
    return out


def encode_audit_features(model, image_paths: list[Path], transform, device, *, batch_size: int = 64):
    import numpy as np
    import torch

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
    if not outputs:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def topk_cosine_neighbors(query_features, reference_features, *, k: int = 50, device="cuda", block_rows: int = 256):
    import numpy as np
    import torch

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

            # torch.topk does not promise stable indices for equal values.  The
            # candidate set is scientific evidence, so resolve only cutoff ties
            # deterministically by ascending reference index without perturbing
            # any non-tied similarity ordering.
            for row_index in range(len(block)):
                threshold = values[row_index, -1]
                strict_idx = torch.nonzero(
                    score[row_index] > threshold, as_tuple=False
                ).flatten()
                tie_idx = torch.nonzero(
                    score[row_index] == threshold, as_tuple=False
                ).flatten()
                slots = int(k) - int(strict_idx.numel())
                if slots < 0:
                    raise TrackBError("top-k cutoff accounting became inconsistent")
                chosen_ties = torch.sort(tie_idx).values[:slots]
                chosen = torch.cat((strict_idx, chosen_ties), dim=0)
                if int(chosen.numel()) != int(k):
                    raise TrackBError("deterministic top-k tie resolution did not produce k neighbors")
                chosen_scores = score[row_index, chosen]
                order = torch.argsort(chosen_scores, descending=True, stable=True)
                chosen = chosen[order]
                chosen_scores = chosen_scores[order]
                indices[row_index] = chosen
                values[row_index] = chosen_scores

            out_idx[start:start + len(block)] = indices.cpu().numpy()
            out_score[start:start + len(block)] = values.cpu().numpy()
    return out_idx, out_score


def _resize_for_orb(image, max_side: int = 800):
    import cv2
    import numpy as np

    arr = np.asarray(image.convert("L"), dtype=np.uint8)
    h, w = arr.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale < 1.0:
        arr = cv2.resize(arr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return arr


def orb_features_from_image(image, *, max_side: int = 800, nfeatures: int = 1200) -> dict[str, Any]:
    import cv2
    import numpy as np

    gray = _resize_for_orb(image, max_side=max_side)
    orb = cv2.ORB_create(nfeatures=int(nfeatures))
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    xy = np.asarray([kp.pt for kp in keypoints], dtype=np.float32) if keypoints else np.empty((0, 2), dtype=np.float32)
    if descriptors is None:
        descriptors = np.empty((0, 32), dtype=np.uint8)
    return {"xy": xy, "desc": descriptors, "shape": tuple(int(x) for x in gray.shape[:2])}


def orb_features_from_path(path: str | Path, *, max_side: int = 800, nfeatures: int = 1200) -> dict[str, Any]:
    return orb_features_from_image(_canonical_rgb(path), max_side=max_side, nfeatures=nfeatures)


def _coverage(points, shape) -> float:
    import cv2
    import numpy as np

    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32))
    area = float(cv2.contourArea(hull))
    h, w = shape
    return area / max(float(h * w), 1.0)


def verify_orb_pair(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    policy: TrackBAuditPolicy,
    rng_seed: int = 0,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    desc_a, desc_b = a["desc"], b["desc"]
    kp_a, kp_b = a["xy"], b["xy"]
    result = {
        "accepted": False,
        "decision_stage": "UNRESOLVED",
        "good_matches": 0,
        "normalized_good_match_ratio": 0.0,
        "homography_inliers": 0,
        "inlier_ratio": 0.0,
        "coverage_a": 0.0,
        "coverage_b": 0.0,
        "median_symmetric_reprojection_px": None,
    }
    if (
        len(desc_a) < policy.minimum_good_matches
        or len(desc_b) < policy.minimum_good_matches
        or len(kp_a) < policy.minimum_good_matches
        or len(kp_b) < policy.minimum_good_matches
    ):
        result["decision_stage"] = "INSUFFICIENT_DESCRIPTORS"
        return result
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = matcher.knnMatch(desc_a, desc_b, k=2)
    good = [m for m, n in pairs if m.distance < policy.lowe_ratio * n.distance]
    result["good_matches"] = len(good)
    ratio = len(good) / max(min(len(kp_a), len(kp_b)), 1)
    result["normalized_good_match_ratio"] = ratio
    if len(good) < policy.minimum_good_matches or ratio < policy.minimum_normalized_good_match_ratio:
        result["decision_stage"] = "GOOD_MATCH_GATE"
        return result
    pts_a = np.float32([kp_a[m.queryIdx] for m in good])
    pts_b = np.float32([kp_b[m.trainIdx] for m in good])
    # OpenCV's RNG is process-global.  Seed + serialize only the RANSAC
    # section so threaded pair verification remains deterministic.
    with _CV2_RANSAC_LOCK:
        cv2.setRNGSeed(int(rng_seed) & 0x7FFFFFFF)
        H, mask = cv2.findHomography(
            pts_a, pts_b, cv2.RANSAC, policy.homography_ransac_reprojection_px
        )
    if H is None or mask is None:
        result["decision_stage"] = "HOMOGRAPHY_FAIL"
        return result
    mask = mask.reshape(-1).astype(bool)
    inliers = int(mask.sum())
    inlier_ratio = inliers / len(good)
    result["homography_inliers"] = inliers
    result["inlier_ratio"] = inlier_ratio
    if inliers < policy.minimum_homography_inliers or inlier_ratio < policy.minimum_inlier_ratio:
        result["decision_stage"] = "INLIER_GATE"
        return result
    in_a, in_b = pts_a[mask], pts_b[mask]
    cov_a, cov_b = _coverage(in_a, a["shape"]), _coverage(in_b, b["shape"])
    result["coverage_a"], result["coverage_b"] = cov_a, cov_b
    if (
        cov_a < policy.minimum_convex_hull_coverage_each_image
        or cov_b < policy.minimum_convex_hull_coverage_each_image
    ):
        result["decision_stage"] = "COVERAGE_GATE"
        return result
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        result["decision_stage"] = "HOMOGRAPHY_INVERSE_FAIL"
        return result
    fwd = cv2.perspectiveTransform(in_a.reshape(-1, 1, 2), H).reshape(-1, 2)
    rev = cv2.perspectiveTransform(in_b.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
    symmetric = 0.5 * (np.linalg.norm(fwd - in_b, axis=1) + np.linalg.norm(rev - in_a, axis=1))
    median = float(np.median(symmetric))
    result["median_symmetric_reprojection_px"] = median
    result["accepted"] = bool(median <= policy.maximum_median_symmetric_reprojection_px)
    result["decision_stage"] = "ACCEPT" if result["accepted"] else "REPROJECTION_GATE"
    return result


def build_family_components(record_ids: Iterable[str], accepted_edges: Iterable[tuple[str, str]]) -> list[list[str]]:
    ids = [str(x) for x in record_ids]
    uf = UnionFind(ids)
    for a, b in accepted_edges:
        if a not in uf.parent or b not in uf.parent:
            raise TrackBError("family edge references an unknown candidate record")
        uf.union(a, b)
    return sorted(uf.components(), key=lambda component: component[0])


def representative_manifest(
    records: list[ImageAuditRecord],
    components: list[list[str]],
    *,
    contaminated_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    by_id = {row.row_id: row for row in records}
    contaminated_ids = contaminated_ids or set()
    out = []
    for family_index, component in enumerate(components):
        labels = sorted({by_id[row_id].source_label for row_id in component})
        representative = min(component, key=lambda row_id: (by_id[row_id].sha256, row_id))
        family_hash = hashlib.sha256("\n".join(sorted(component)).encode("utf-8")).hexdigest()
        out.append({
            "family_id": f"F{family_index:08d}-{family_hash[:12]}",
            "representative_row_id": representative,
            "representative_raw_sha256": by_id[representative].sha256,
            "member_count": len(component),
            "source_labels": labels,
            "label_conflict": len(labels) > 1,
            "historically_contaminated": any(row_id in contaminated_ids for row_id in component),
        })
    return out


def deterministic_representative_order(
    rows: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Return a deterministic seeded ordering independent of filesystem traversal."""
    import numpy as np

    canonical = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            str(row.get("representative_raw_sha256", "")),
            str(row.get("family_id", "")),
            str(row.get("representative_row_id", "")),
        ),
    )
    if not canonical:
        return []
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    permutation = rng.permutation(len(canonical)).tolist()
    return [canonical[index] for index in permutation]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
