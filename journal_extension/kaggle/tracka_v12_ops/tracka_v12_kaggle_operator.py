from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

SCIENCE_SHA = "e21a505792ddd91df55712e248391c79c55cf235"
REPOSITORY_URL = "https://github.com/rana-m-ahmed/ResearchWork-CropCop.git"
AUTHORITY_ID = "EAAI-JE-SDL-v2.1-QA"
MANIFEST_SHA256 = "bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2"
CLASS_MAP_SHA256 = "46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2"
PRINCIPAL_G1_SEAL_SHA256 = "442d9e7708749efedcb82ef9f4fd131211549770eecad985177eaaf4117052cd"
R13_SHA256 = "92ec2d996329be8c9a449d4e38f847e79c34fadc32b6491739bacdaf425ab0ed"
R13_BYTES = 90095744
R13_HF_REPOSITORY = "timm/vit_dlittle_patch16_reg1_gap_256.sbb_nadamuon_in1k"
R13_HF_COMMIT = "10143029df1b2df4c63623de6b740ecccb2b8ea2"
LOCKED_PYTHON = "3.12.13"
LOCKED_PACKAGES = {
    "torch": "2.12.1",
    "torchvision": "0.27.1",
    "timm": "1.0.26",
    "numpy": "2.5.2",
    "Pillow": "12.3.0",
    "safetensors": "0.8.0",
    "kaggle": "2.2.4",
    "huggingface-hub": "1.30.0",
    "transformers": "5.0.0",
}
GIT_CREDENTIAL_ENV_NAMES = (
    "GITHUB_TOKEN", "GH_TOKEN", "CROPCOP_GITHUB_TOKEN",
    "GIT_ASKPASS", "GIT_ASKPASS_REQUIRE", "SSH_AUTH_SOCK",
)
G2A_ACCOUNT_PROFILES = {
    "K1": (
        ("CAL-EFFB0", "R06-EFFB0-CONTEXT-S2", "K1/GPU0", 0),
        ("CAL-CNXTT", "R07-CNXTT-CONTEXT-S2", "K1/GPU1", 1),
    ),
    "K2": (
        ("CAL-MNV4-LOGITS", "R12-MNV4-LOGITS-S2", "K2/GPU0", 0),
        ("CAL-MNV4-FEATURE", "R12-MNV4-FEATURE-S2", "K2/GPU1", 1),
    ),
    "K3": (
        ("CAL-R13", "R13-VIT-DLITTLE-DIFF-CONTEXT-S1", "K3/GPU0", 0),
    ),
}

class OperatorError(RuntimeError):
    pass

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def run(cmd: list[str], *, cwd: str | Path | None = None, env: dict[str, str] | None = None,
        capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    cp = subprocess.run(
        [str(x) for x in cmd],
        cwd=None if cwd is None else str(cwd),
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if check and cp.returncode != 0:
        detail = (cp.stdout or "")[-4000:] if capture else ""
        raise OperatorError(f"command failed ({cp.returncode}): {' '.join(map(str, cmd))}\n{detail}")
    return cp

def assert_kaggle_paths() -> None:
    if not Path("/kaggle/working").is_dir() or not Path("/kaggle/input").is_dir():
        raise OperatorError("This operator notebook must run inside a Kaggle notebook session.")

def assert_python_version() -> None:
    observed = ".".join(map(str, sys.version_info[:3]))
    if observed != LOCKED_PYTHON:
        raise OperatorError(
            f"Python drift: frozen execution requires {LOCKED_PYTHON}, Kaggle kernel is {observed}. "
            "Do not continue qualification/science on a different interpreter."
        )

def ensure_science_checkout(target: str | Path = "/kaggle/working/cropcop-science") -> Path:
    target = Path(target)
    if target.exists():
        shutil.rmtree(target)
    run(["git", "clone", "--quiet", "--no-tags", REPOSITORY_URL, str(target)])
    run(["git", "checkout", "--detach", SCIENCE_SHA], cwd=target)
    head = run(["git", "rev-parse", "HEAD"], cwd=target, capture=True).stdout.strip()
    if head != SCIENCE_SHA:
        raise OperatorError(f"science checkout mismatch: {head}")
    dirty = run(["git", "status", "--porcelain"], cwd=target, capture=True).stdout.strip()
    if dirty:
        raise OperatorError("science checkout is dirty immediately after checkout")
    return target

def assert_clean_science_checkout(repo: str | Path) -> None:
    repo = Path(repo)
    head = run(["git", "rev-parse", "HEAD"], cwd=repo, capture=True).stdout.strip()
    dirty = run(["git", "status", "--porcelain"], cwd=repo, capture=True).stdout.strip()
    if head != SCIENCE_SHA:
        raise OperatorError(f"science source changed: expected {SCIENCE_SHA}, got {head}")
    if dirty:
        raise OperatorError(f"science checkout is dirty:\n{dirty}")

def install_locked_stack(repo: str | Path) -> None:
    assert_python_version()
    req = Path(repo) / "journal_extension" / "requirements-training.lock.txt"
    run([
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
        "--no-input", "--upgrade", "--no-cache-dir", "-r", str(req)
    ])

def verify_locked_stack(repo: str | Path) -> dict:
    assert_python_version()
    observed = {}
    errors = []
    for dist, expected in LOCKED_PACKAGES.items():
        try:
            value = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            value = None
        observed[dist] = value
        if value != expected:
            errors.append(f"{dist}: expected {expected}, got {value}")
    if errors:
        raise OperatorError("locked package drift: " + "; ".join(errors))
    dep = load_json(Path(repo) / "journal_extension" / "locks" / "execution_dependency_lock.json")
    if dep.get("python") != LOCKED_PYTHON or dep.get("packages") != LOCKED_PACKAGES:
        raise OperatorError("repository dependency lock differs from operator freeze")
    return {"python": LOCKED_PYTHON, "packages": observed, "dependency_lock_sha256": dep["dependency_lock_sha256"]}

def find_exact_hash(root: str | Path, expected_sha: str, *, suffixes: Iterable[str] | None = None) -> Path:
    root = Path(root)
    allowed = None if suffixes is None else {x.lower() for x in suffixes}
    matches = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if allowed is not None and path.suffix.lower() not in allowed:
            continue
        try:
            if sha256_file(path) == expected_sha:
                matches.append(path.resolve())
        except (OSError, PermissionError):
            continue
    if len(matches) != 1:
        raise OperatorError(f"expected exactly one file with sha256={expected_sha}, found {len(matches)}: {matches[:10]}")
    return matches[0]

def resolve_frozen_dataset(input_root: str | Path = "/kaggle/input",
                           manifest_override: str = "", class_map_override: str = "") -> tuple[Path, Path]:
    manifest = Path(manifest_override).resolve() if manifest_override else find_exact_hash(
        input_root, MANIFEST_SHA256, suffixes={".csv", ".jsonl", ".json"}
    )
    class_map = Path(class_map_override).resolve() if class_map_override else find_exact_hash(
        input_root, CLASS_MAP_SHA256, suffixes={".json"}
    )
    if sha256_file(manifest) != MANIFEST_SHA256:
        raise OperatorError("manual manifest override has wrong SHA-256")
    if sha256_file(class_map) != CLASS_MAP_SHA256:
        raise OperatorError("manual class-map override has wrong SHA-256")
    return manifest, class_map

def _manifest_sample_paths(manifest: Path, limit: int = 16) -> list[str]:
    with manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "portable_relpath" not in (reader.fieldnames or []):
            raise OperatorError("frozen manifest is missing portable_relpath")
        rows = []
        for row in reader:
            rel = str(row.get("portable_relpath", "")).strip()
            if rel:
                rows.append(rel)
            if len(rows) >= limit:
                break
    if len(rows) < min(8, limit):
        raise OperatorError("manifest did not provide enough image paths for root qualification")
    return rows

def resolve_image_root(manifest: str | Path, *, input_root: str | Path = "/kaggle/input",
                       override: str = "", max_depth: int = 3) -> Path:
    manifest = Path(manifest)
    sample = _manifest_sample_paths(manifest)
    if override:
        candidate = Path(override).resolve()
        if not all((candidate / rel).is_file() for rel in sample):
            raise OperatorError("manual IMAGE_ROOT does not resolve the frozen manifest sample")
        return candidate
    base = Path(input_root).resolve()
    candidates = [base]
    frontier = [base]
    for _ in range(max_depth):
        nxt = []
        for parent in frontier:
            try:
                children = [p for p in parent.iterdir() if p.is_dir()]
            except OSError:
                children = []
            candidates.extend(children)
            nxt.extend(children)
        frontier = nxt
    matches = []
    for candidate in candidates:
        if all((candidate / rel).is_file() for rel in sample):
            matches.append(candidate)
    unique = sorted({p.resolve() for p in matches}, key=lambda p: (len(p.parts), str(p)))
    if len(unique) != 1:
        raise OperatorError(
            "IMAGE_ROOT auto-resolution must be unique. "
            f"Found {len(unique)} candidates: {[str(x) for x in unique[:12]]}. "
            "Set IMAGE_ROOT_OVERRIDE explicitly if the same dataset is mounted more than once."
        )
    return unique[0]

def resolve_principal_g1_bundle(input_root: str | Path = "/kaggle/input", override: str = "") -> Path:
    if override:
        roots = [Path(override).resolve()]
    else:
        roots = [p.parent.resolve() for p in Path(input_root).rglob("G1_MODEL_IDENTITY_SEAL.json")]
    matches = []
    for root in roots:
        seal_path = root / "G1_MODEL_IDENTITY_SEAL.json"
        if not seal_path.is_file():
            continue
        try:
            seal = load_json(seal_path)
        except Exception:
            continue
        if seal.get("g1_seal_sha256") != PRINCIPAL_G1_SEAL_SHA256:
            continue
        if not (root / "private").is_dir() or not (root / "evidence").is_dir():
            continue
        matches.append(root)
    matches = sorted({p.resolve() for p in matches})
    if len(matches) != 1:
        raise OperatorError(
            "principal historical G1 bundle auto-resolution must be unique; "
            f"found {len(matches)}: {[str(x) for x in matches]}"
        )
    return matches[0]

def prepare_official_torchvision(repo: str | Path, work_root: str | Path) -> dict[str, dict[str, Path]]:
    repo = Path(repo)
    work_root = Path(work_root)
    result = {}
    for key in ("effb0", "cnxtt"):
        out = work_root / key
        record = work_root / f"{key}_provenance.json"
        if out.exists():
            shutil.rmtree(out)
        record.unlink(missing_ok=True)
        run([
            sys.executable,
            str(repo / "journal_extension" / "scripts" / "prepare_torchvision_pretrained.py"),
            "--model-key", key,
            "--output-dir", str(out),
            "--record", str(record),
        ], cwd=repo)
        payload = load_json(record)
        artifact = out / payload["official_filename"]
        if sha256_file(artifact) != payload["artifact_sha256"]:
            raise OperatorError(f"{key} downloaded artifact/provenance mismatch")
        result[key] = {"artifact": artifact, "provenance": record}
    return result

def prepare_r13(work_root: str | Path) -> Path:
    from huggingface_hub import hf_hub_download
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    cached = Path(hf_hub_download(
        repo_id=R13_HF_REPOSITORY,
        filename="model.safetensors",
        revision=R13_HF_COMMIT,
    ))
    target = work_root / "R13_PRETRAINED.safetensors"
    shutil.copyfile(cached, target)
    if target.stat().st_size != R13_BYTES or sha256_file(target) != R13_SHA256:
        raise OperatorError("R13 pinned model.safetensors failed exact byte identity")
    return target

def load_kaggle_credentials() -> tuple[str, str]:
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    if username and key:
        return username, key
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        username = client.get_secret("KAGGLE_USERNAME").strip()
        key = client.get_secret("KAGGLE_KEY").strip()
    except Exception as exc:
        raise OperatorError(
            "Kaggle API credentials not found. Add Kaggle notebook secrets "
            "KAGGLE_USERNAME and KAGGLE_KEY."
        ) from exc
    if not username or not key:
        raise OperatorError("Kaggle username/key secret is empty")
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    return username, key

def sanitized_child_env(*, gpu_index: int | None = None, global_clock: float | None = None) -> dict[str, str]:
    env = dict(os.environ)
    for name in GIT_CREDENTIAL_ENV_NAMES:
        env.pop(name, None)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    if global_clock is not None:
        env["CROPCOP_NOTEBOOK_STARTED_MONOTONIC"] = repr(global_clock)
        env["CROPCOP_NOTEBOOK_HARD_LIMIT_SECONDS"] = repr(12 * 3600.0)
        env["CROPCOP_NOTEBOOK_FINALIZATION_MARGIN_SECONDS"] = repr(3600.0)
    return env

def kaggle_dataset_exists(locator: str, *, env: dict[str, str] | None = None) -> bool:
    cp = run(["kaggle", "datasets", "files", "-d", locator], env=env, capture=True, check=False)
    return cp.returncode == 0

def ensure_private_dataset(locator: str, *, title: str | None = None, env: dict[str, str] | None = None) -> None:
    env = dict(os.environ if env is None else env)
    owner, sep, slug = locator.partition("/")
    if not sep or not owner or not slug:
        raise OperatorError(f"invalid Kaggle locator: {locator}")
    if kaggle_dataset_exists(locator, env=env):
        return
    with tempfile.TemporaryDirectory(dir="/kaggle/working") as td:
        root = Path(td)
        write_json(root / "dataset-metadata.json", {
            "title": title or slug,
            "id": locator,
            "licenses": [{"name": "other"}],
            "isPrivate": True,
        })
        (root / "README.txt").write_text(
            "Private CropCop operational durability/handoff dataset. Do not make public.\n",
            encoding="utf-8",
        )
        run(["kaggle", "datasets", "create", "-p", str(root), "-q"], env=env)
    if not kaggle_dataset_exists(locator, env=env):
        raise OperatorError(f"private Kaggle dataset creation did not become readable: {locator}")

def version_private_dataset(locator: str, source_dir: str | Path, *, message: str,
                            env: dict[str, str] | None = None, include_names: set[str] | None = None) -> None:
    env = dict(os.environ if env is None else env)
    source_dir = Path(source_dir)
    _, _, slug = locator.partition("/")
    ensure_private_dataset(locator, env=env)
    with tempfile.TemporaryDirectory(dir="/kaggle/working") as td:
        staging = Path(td)
        for src in source_dir.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(source_dir)
            if include_names is not None and rel.as_posix() not in include_names:
                continue
            dst = staging / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        write_json(staging / "dataset-metadata.json", {
            "title": slug,
            "id": locator,
            "licenses": [{"name": "other"}],
            "isPrivate": True,
        })
        run(["kaggle", "datasets", "version", "-p", str(staging), "-m", message, "-q", "-r", "zip"], env=env)

def download_private_dataset(locator: str, destination: str | Path, *, env: dict[str, str] | None = None) -> Path:
    env = dict(os.environ if env is None else env)
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=False)
    run(["kaggle", "datasets", "download", "-d", locator, "-p", str(destination), "--unzip", "-q"], env=env)
    return destination

def validate_private_locators_with_science(repo: str | Path, mapping: dict[str, str],
                                           *, env: dict[str, str] | None = None) -> dict:
    env = dict(os.environ if env is None else env)
    code = (
        "import json,sys; "
        f"sys.path.insert(0,{str(Path(repo)/'journal_extension'/'src')!r}); "
        "from cropcop_je.persistence import validate_durable_access_plan; "
        f"m=json.loads({json.dumps(json.dumps(mapping))!r}); "
        "r=validate_durable_access_plan('kaggle-dataset',m); "
        "print(json.dumps(r,sort_keys=True)); "
        "raise SystemExit(0 if r.get('status')=='PASS' else 2)"
    )
    cp = run([sys.executable, "-c", code], env=env, capture=True, check=False)
    if cp.returncode != 0:
        raise OperatorError(f"private durable preflight failed:\n{cp.stdout[-6000:]}")
    return json.loads(cp.stdout.strip().splitlines()[-1])

def discover_g1a_bundle(input_root: str | Path = "/kaggle/input", override: str = "") -> Path:
    roots = [Path(override).resolve()] if override else [
        p.parent.resolve() for p in Path(input_root).rglob("TRACKA_V12_G1A_SEAL.json")
    ]
    matches = []
    for root in roots:
        seal = root / "TRACKA_V12_G1A_SEAL.json"
        if not seal.is_file() or not (root / "private").is_dir() or not (root / "evidence").is_dir():
            continue
        try:
            payload = load_json(seal)
        except Exception:
            continue
        if payload.get("status") == "PASS" and payload.get("source_git_commit") == SCIENCE_SHA:
            matches.append(root)
    matches = sorted({p.resolve() for p in matches})
    if len(matches) != 1:
        raise OperatorError(f"G1A bundle resolution must be unique, found {len(matches)}: {matches}")
    return matches[0]

def discover_json_by_key(input_root: str | Path, *, key: str, values: set[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in Path(input_root).rglob("*.json"):
        try:
            payload = load_json(path)
        except Exception:
            continue
        value = str(payload.get(key, ""))
        if value in values:
            if value in found:
                raise OperatorError(f"duplicate {key}={value} JSONs: {found[value]} and {path}")
            found[value] = path.resolve()
    missing = values - set(found)
    if missing:
        raise OperatorError(f"missing JSON payloads for {key}: {sorted(missing)}")
    return found

def slugify(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")

def scientific_durable_map(scheduler: dict, account_owners: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    queues = scheduler.get("static_slot_queues") or {}
    for slot_id, queue in queues.items():
        account_id = slot_id.split("/", 1)[0]
        owner = account_owners.get(account_id, "")
        if not owner:
            raise OperatorError(f"missing Kaggle owner for {account_id}")
        for experiment_id in queue:
            if experiment_id in result:
                raise OperatorError(f"experiment appears more than once in scheduler: {experiment_id}")
            result[experiment_id] = f"{owner}/cropcop-{slugify(experiment_id)}-{SCIENCE_SHA[:12]}"
    return result
