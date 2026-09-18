from __future__ import annotations

import csv
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .hashing import sha256_file, sha256_json

MANIFEST_SHA256 = "bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2"
CLASS_MAP_SHA256 = "46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2"
EXPECTED_TRAIN = 76376
EXPECTED_VAL = 16368
EXPECTED_TEST = 16363
EXPECTED_CLASSES = 120

SCIENCE_SOURCE_SHA = "56023042e57758591df9babb3438f191dbe10312"
_DISCOVERY_KEYWORDS = (
    "tracka",
    "track-a",
    "science",
    "r12",
    "r13",
    "secondary",
    "principal",
)
_SCIENCE_CODE_TOKENS = (
    SCIENCE_SOURCE_SHA,
    "tracka_v12",
    "run_tracka_v12",
    "TRACKA_V12",
)
_IDENTITY_NAME_HINTS = (
    "manifest",
    "class",
    "index",
    "label",
    "map",
    "final",
    "cropcop",
)


class FinalV1ResolutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class FinalV1Resolution:
    source_kind: str
    source_ref: str | None
    root: Path
    manifest: Path
    class_map: Path
    image_root: Path
    train_rows: int
    val_rows: int
    test_rows: int

    def certificate(self) -> dict[str, Any]:
        payload = {
            "schema_version": "1.0",
            "status": "PASS",
            "resolution_kind": "cropcop_final_v1_exact_identity",
            "source_kind": self.source_kind,
            "source_ref": self.source_ref,
            "root": str(self.root),
            "manifest": str(self.manifest),
            "class_map": str(self.class_map),
            "image_root": str(self.image_root),
            "manifest_sha256": sha256_file(self.manifest),
            "class_map_sha256": sha256_file(self.class_map),
            "split_counts": {
                "train": self.train_rows,
                "val": self.val_rows,
                "test": self.test_rows,
            },
            "num_classes": EXPECTED_CLASSES,
            "v1_test_image_bytes_opened": False,
            "selection_metrics_opened": False,
        }
        payload["certificate_sha256"] = sha256_json(payload)
        return payload


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FinalV1ResolutionError(f"JSON object required: {path}")
    return payload


def _candidate_identity_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.casefold()
        if suffix not in {".csv", ".json"}:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > 256 * 1024 * 1024:
            continue
        yield path


def _exact_identity_pair(root: Path) -> tuple[Path, Path] | None:
    manifests: list[Path] = []
    class_maps: list[Path] = []
    for path in _candidate_identity_files(root):
        digest = sha256_file(path)
        if digest == MANIFEST_SHA256:
            manifests.append(path.resolve())
        if digest == CLASS_MAP_SHA256:
            class_maps.append(path.resolve())
    if len(manifests) != 1 or len(class_maps) != 1:
        return None
    return manifests[0], class_maps[0]


def _manifest_rows(manifest: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    rows: list[dict[str, str]] = []
    counts = {"train": 0, "val": 0, "test": 0}
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"record_key", "portable_relpath", "split", "label"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise FinalV1ResolutionError(
                f"Final-V1 manifest missing required columns: {sorted(missing)}"
            )
        for row in reader:
            split = str(row.get("split", "")).strip()
            if split in counts:
                counts[split] += 1
            rows.append({k: str(v or "") for k, v in row.items()})
    expected = {"train": EXPECTED_TRAIN, "val": EXPECTED_VAL, "test": EXPECTED_TEST}
    if counts != expected:
        raise FinalV1ResolutionError(
            f"Final-V1 split count mismatch: observed={counts}, expected={expected}"
        )
    return rows, counts


def _class_map_contract(path: Path) -> None:
    payload = _json_object(path)
    num_classes = payload.get("num_classes")
    if num_classes is not None and int(num_classes) != EXPECTED_CLASSES:
        raise FinalV1ResolutionError(
            f"Final-V1 class-map class count mismatch: {num_classes}"
        )
    c2i = payload.get("class_to_idx")
    i2c = payload.get("idx_to_class")
    observed = None
    if isinstance(c2i, dict):
        observed = len(c2i)
    elif isinstance(i2c, (dict, list)):
        observed = len(i2c)
    if observed is not None and observed != EXPECTED_CLASSES:
        raise FinalV1ResolutionError(
            f"Final-V1 class-map cardinality mismatch: {observed}"
        )


def _candidate_image_roots(bundle_root: Path, manifest: Path) -> list[Path]:
    roots: list[Path] = []
    for value in (
        bundle_root.resolve(),
        manifest.parent.resolve(),
        *[p.resolve() for p in bundle_root.iterdir() if p.is_dir()],
    ):
        if value not in roots:
            roots.append(value)
    for child in list(roots):
        try:
            for p in child.iterdir():
                if p.is_dir():
                    rp = p.resolve()
                    if rp not in roots:
                        roots.append(rp)
        except OSError:
            pass
    return roots


def _resolve_image_root(bundle_root: Path, manifest: Path, rows: list[dict[str, str]]) -> Path:
    protected = [row for row in rows if row["split"] in {"train", "val"}]
    sample = protected[:64] + protected[-64:]
    viable: list[Path] = []
    for root in _candidate_image_roots(bundle_root, manifest):
        if all((root / row["portable_relpath"]).is_file() for row in sample):
            viable.append(root)
    exact: list[Path] = []
    for root in viable:
        missing = 0
        for row in protected:
            rel = row["portable_relpath"]
            if not rel or not (root / rel).is_file():
                missing += 1
                break
        if missing == 0:
            exact.append(root)
    if not exact:
        raise FinalV1ResolutionError(
            "Final-V1 identity files found but no image root resolves all frozen Train/Val paths"
        )
    exact.sort(key=lambda p: (len(p.parts), str(p)))
    return exact[0]


def validate_final_v1_root(
    root: str | Path,
    *,
    source_kind: str,
    source_ref: str | None = None,
) -> FinalV1Resolution:
    root = Path(root).resolve()
    pair = _exact_identity_pair(root)
    if pair is None:
        raise FinalV1ResolutionError(
            f"root does not contain one exact Final-V1 manifest/class-map pair: {root}"
        )
    manifest, class_map = pair
    rows, counts = _manifest_rows(manifest)
    _class_map_contract(class_map)
    image_root = _resolve_image_root(root, manifest, rows)
    return FinalV1Resolution(
        source_kind=source_kind,
        source_ref=source_ref,
        root=root,
        manifest=manifest,
        class_map=class_map,
        image_root=image_root,
        train_rows=counts["train"],
        val_rows=counts["val"],
        test_rows=counts["test"],
    )


def scan_attached_inputs(input_root: str | Path = "/kaggle/input") -> list[FinalV1Resolution]:
    root = Path(input_root)
    if not root.is_dir():
        return []
    results: list[FinalV1Resolution] = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            results.append(
                validate_final_v1_root(
                    child,
                    source_kind="attached_kaggle_input",
                    source_ref=child.name,
                )
            )
        except FinalV1ResolutionError:
            continue
    return results


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _ref(value: object) -> str:
    return str(getattr(value, "ref", "") or "").strip()


def _kernel_pull_text(root: Path) -> str:
    chunks: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name == "kernel-metadata.json":
            continue
        if path.suffix.casefold() not in {".py", ".ipynb", ".md", ".txt"}:
            continue
        try:
            if path.stat().st_size > 16 * 1024 * 1024:
                continue
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(chunks)


def recent_kernel_dataset_sources(
    api,
    *,
    max_pages: int = 6,
    page_size: int = 100,
) -> tuple[list[str], list[str]]:
    kernels_seen: list[str] = []
    preferred: list[object] = []
    fallback: list[object] = []
    for page in range(1, max_pages + 1):
        rows = api.kernels_list(
            mine=True,
            page=page,
            page_size=page_size,
            sort_by="dateRun",
        ) or []
        if not rows:
            break
        for row in rows:
            ref = _ref(row)
            if not ref:
                continue
            kernels_seen.append(ref)
            title = str(getattr(row, "title", "") or "")
            haystack = f"{ref} {title}".casefold()
            if any(token in haystack for token in _DISCOVERY_KEYWORDS):
                preferred.append(row)
            elif len(fallback) < 60:
                fallback.append(row)

    science_sources: list[str] = []
    preferred_sources: list[str] = []
    fallback_sources: list[str] = []
    seen_science: set[str] = set()
    seen_preferred: set[str] = set()
    seen_fallback: set[str] = set()

    to_pull = preferred[:140] + fallback[:60]
    for row in to_pull:
        ref = _ref(row)
        if not ref:
            continue
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            try:
                api.kernels_pull(ref, path=td, metadata=True, quiet=True)
                meta_path = root / "kernel-metadata.json"
                meta = _json_object(meta_path)
            except Exception:
                continue

            code_text = _kernel_pull_text(root)
            code_lower = code_text.casefold()
            is_science = (
                SCIENCE_SOURCE_SHA.casefold() in code_lower
                or "tracka_v12" in code_lower
                or "run_tracka_v12" in code_lower
            )
            title = str(getattr(row, "title", "") or "")
            title_haystack = f"{ref} {title}".casefold()
            is_preferred = any(token in title_haystack for token in _DISCOVERY_KEYWORDS)

            for source in meta.get("dataset_sources") or []:
                value = str(source or "").strip()
                if "/" not in value:
                    continue
                key = value.casefold()
                if is_science:
                    if key not in seen_science:
                        seen_science.add(key)
                        science_sources.append(value)
                elif is_preferred:
                    if key not in seen_preferred:
                        seen_preferred.add(key)
                        preferred_sources.append(value)
                else:
                    if key not in seen_fallback:
                        seen_fallback.add(key)
                        fallback_sources.append(value)

    ordered: list[str] = []
    seen: set[str] = set()
    for collection in (science_sources, preferred_sources, fallback_sources):
        for value in collection:
            key = value.casefold()
            if key not in seen:
                seen.add(key)
                ordered.append(value)
    return ordered, kernels_seen


def mine_dataset_refs(
    api,
    *,
    max_pages: int = 12,
    page_size: int = 100,
) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for page in range(1, max_pages + 1):
        response = api.dataset_list_with_response(
            mine=True,
            page=page,
            page_size=page_size,
            sort_by="updated",
        )
        rows = getattr(response, "datasets", None) or []
        if not rows:
            break
        for row in rows:
            value = _ref(row)
            if value and value.casefold() not in seen:
                seen.add(value.casefold())
                refs.append(value)
        if len(rows) < page_size:
            break
    return refs


def _dataset_files(api, dataset_ref: str) -> list[object]:
    files: list[object] = []
    token = None
    while True:
        response = api.dataset_list_files(
            dataset_ref,
            page_token=token,
            page_size=100,
        )
        rows = getattr(response, "files", None) or []
        files.extend(rows)
        token = getattr(response, "next_page_token", None)
        if not token:
            break
    return files


def _candidate_remote_identity_names(api, dataset_ref: str) -> list[str]:
    names: list[str] = []
    scored: list[tuple[int, str]] = []
    for row in _dataset_files(api, dataset_ref):
        name = str(getattr(row, "name", "") or "").strip()
        if not name:
            continue
        suffix = Path(name).suffix.casefold()
        if suffix not in {".csv", ".json"}:
            continue
        size = int(getattr(row, "total_bytes", 0) or 0)
        if size and size > 256 * 1024 * 1024:
            continue
        lname = name.casefold()
        score = sum(1 for token in _IDENTITY_NAME_HINTS if token in lname)
        scored.append((-score, name))
    for _, name in sorted(scored):
        if name not in names:
            names.append(name)
    return names[:40]


def _download_remote_file(api, dataset_ref: str, file_name: str, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    before = {p.resolve() for p in destination.rglob("*") if p.is_file()}
    api.dataset_download_file(
        dataset_ref,
        file_name,
        path=str(destination),
        force=True,
        quiet=True,
    )
    after = [p.resolve() for p in destination.rglob("*") if p.is_file()]
    return [p for p in after if p not in before] or after


def remote_dataset_matches_identity(api, dataset_ref: str) -> bool:
    found_manifest = False
    found_class_map = False
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for name in _candidate_remote_identity_names(api, dataset_ref):
            probe_dir = root / f"probe_{len(list(root.iterdir())):03d}"
            try:
                downloaded = _download_remote_file(api, dataset_ref, name, probe_dir)
            except Exception:
                continue
            for path in downloaded:
                if not path.is_file() or path.stat().st_size > 256 * 1024 * 1024:
                    continue
                digest = sha256_file(path)
                found_manifest = found_manifest or digest == MANIFEST_SHA256
                found_class_map = found_class_map or digest == CLASS_MAP_SHA256
            if found_manifest and found_class_map:
                return True
    return False


def hydrate_remote_dataset(api, dataset_ref: str, destination: str | Path) -> FinalV1Resolution:
    destination = Path(destination).resolve()
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=False)
    api.dataset_download_files(
        dataset_ref,
        path=str(destination),
        force=True,
        quiet=True,
        unzip=True,
    )
    return validate_final_v1_root(
        destination,
        source_kind="historical_kaggle_dataset",
        source_ref=dataset_ref,
    )


def _final_v1_candidate_score(dataset_ref: str, *, kernel_rank: int | None) -> tuple[int, int, str]:
    name = dataset_ref.casefold()
    score = 0
    if "finalized" in name:
        score += 600
    if "final-v1" in name or "final_v1" in name or "finalv1" in name:
        score += 550
    if "120-class" in name or "120_class" in name or "120class" in name:
        score += 450
    if "cropcop" in name:
        score += 120
    if "dataset" in name:
        score += 40

    for token, penalty in (
        ("checkpoint", 700),
        ("model", 500),
        ("readiness", 450),
        ("qa", 400),
        ("audit", 350),
        ("cicps", 600),
        ("agri-", 250),
        ("g1a", 300),
        ("g1-", 300),
        ("g2", 300),
    ):
        if token in name:
            score -= penalty

    # Kernel provenance is stronger than an owned-dataset inventory fallback.
    provenance_bonus = 300 if kernel_rank is not None else 0
    rank_penalty = kernel_rank if kernel_rank is not None else 10_000
    return (-(score + provenance_bonus), rank_penalty, name)


def _ordered_remote_candidates(kernel_refs: list[str], mine_refs: list[str]) -> list[str]:
    kernel_rank = {value.casefold(): index for index, value in enumerate(kernel_refs)}
    merged: list[str] = []
    seen: set[str] = set()
    for value in [*kernel_refs, *mine_refs]:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            merged.append(value)
    return sorted(
        merged,
        key=lambda value: _final_v1_candidate_score(
            value,
            kernel_rank=kernel_rank.get(value.casefold()),
        ),
    )


def resolve_final_v1(
    *,
    output_root: str | Path,
    attached_input_root: str | Path = "/kaggle/input",
    api_factory=_api,
) -> tuple[FinalV1Resolution, dict[str, Any]]:
    attached = scan_attached_inputs(attached_input_root)
    if len(attached) == 1:
        return attached[0], {
            "attached_matches": 1,
            "kernel_dataset_candidates": [],
            "mine_dataset_candidates": [],
            "remote_candidates_checked": [],
        }
    if len(attached) > 1:
        refs = [row.source_ref for row in attached]
        raise FinalV1ResolutionError(
            f"multiple attached inputs independently match exact Final-V1 identity: {refs}"
        )

    api = api_factory()
    kernel_refs, kernels_seen = recent_kernel_dataset_sources(api)
    mine_refs = mine_dataset_refs(api)

    ordered = _ordered_remote_candidates(kernel_refs, mine_refs)

    checked: list[dict[str, Any]] = []
    output_root = Path(output_root).resolve()
    output_parent = output_root.parent
    output_parent.mkdir(parents=True, exist_ok=True)

    # Final-V1 is required in full for post-training evidence anyway. Instead of
    # probing dozens of loose files, hydrate provenance-ranked candidates one by
    # one and validate the complete frozen identity contract. This also handles
    # datasets whose manifest/class-map live inside the dataset archive/layout.
    for index, ref in enumerate(ordered[:12]):
        staging = output_parent / f".final_v1_candidate_{index:02d}"
        if staging.exists():
            shutil.rmtree(staging)
        row: dict[str, Any] = {
            "dataset_ref": ref,
            "candidate_rank": index,
            "status": "FAILED",
        }
        try:
            resolution = hydrate_remote_dataset(api, ref, staging)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            checked.append(row)
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            continue

        row["status"] = "PASS"
        row["manifest_sha256"] = sha256_file(resolution.manifest)
        row["class_map_sha256"] = sha256_file(resolution.class_map)
        checked.append(row)

        if output_root.exists():
            shutil.rmtree(output_root)
        staging.rename(output_root)
        resolution = validate_final_v1_root(
            output_root,
            source_kind="historical_kaggle_dataset",
            source_ref=ref,
        )
        diagnostics = {
            "attached_matches": 0,
            "kernel_dataset_candidates": kernel_refs,
            "mine_dataset_candidates": mine_refs,
            "kernels_seen_count": len(kernels_seen),
            "remote_candidates_checked": checked,
            "matched_dataset_ref": ref,
            "candidate_order": ordered[:20],
        }
        return resolution, diagnostics

    raise FinalV1ResolutionError(
        "Final-V1 provenance-ranked hydration did not find an exact accessible dataset; "
        f"kernel_sources={kernel_refs[:40]}, mine_dataset_count={len(mine_refs)}, "
        f"kernels_seen_count={len(kernels_seen)}, candidate_order={ordered[:20]}, "
        f"checked={checked}"
    )
