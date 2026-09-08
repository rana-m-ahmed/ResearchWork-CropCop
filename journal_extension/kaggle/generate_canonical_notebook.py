from __future__ import annotations

import json
from pathlib import Path

MGPU_EXECUTION_SOURCE_SHA = "f171309fc7e9dc22241ecc137ebbb8e4bcdc5433"
AUTHORIZED_SOURCE_SHA = "f171309fc7e9dc22241ecc137ebbb8e4bcdc5433"

MARKDOWN = """# CropCop EAAI — Canonical Stage-01A-MGPU Kaggle Wrapper

Thin orchestration only, hard-bound to the frozen Stage-01A-G1P-v2.2 execution source.

Operator phases: `smoke-write`, `smoke-restore`, `dual-gpu-smoke`, `g1`, `calibration-dual`, `principal-dual`.

For `principal-dual`, set non-secret `CROPCOP_PRINCIPAL_ENVELOPE` to `P1`, `P2`, or `P3`. The notebook delegates the fixed experiment mapping to repository envelope configs.

Smoke A/B remain cross-Saved-Version and API-free. `smoke-restore` requires the exact Smoke-A Notebook Output attached read-only.
"""

CODE = r'''import base64, json
import os, platform, shutil, stat, subprocess, sys, tempfile, time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

# ============================================================
# CROPCOP EAAI — KAGGLE OPERATOR CONFIGURATION
# ============================================================
# Frozen Stage-01A-G1P-v2.2 execution source. Wrapper commits are not execution sources.
AUTHORIZED_SOURCE_SHA = "f171309fc7e9dc22241ecc137ebbb8e4bcdc5433"
LANE = os.environ.get("CROPCOP_LANE", "K1")
EXECUTION_PHASE = os.environ.get("CROPCOP_EXECUTION_PHASE", "").strip()
PRINCIPAL_ENVELOPE = os.environ.get("CROPCOP_PRINCIPAL_ENVELOPE", "").strip().upper()
# Exact operator phases: smoke-write | smoke-restore | dual-gpu-smoke | g1 | calibration-dual | principal-dual
REPOSITORY_URL = os.environ.get(
    "CROPCOP_REPOSITORY_URL",
    "https://github.com/rana-m-ahmed/ResearchWork-CropCop.git",
)
REPO_WORKDIR = os.environ.get("CROPCOP_REPO_WORKDIR", "/kaggle/working/cropcop-je")
OUTPUT_ROOT = os.environ.get("CROPCOP_OUTPUT_ROOT", "/kaggle/working/cropcop-je-output")
SYNTHETIC_SMOKE_ROOT = os.environ.get(
    "CROPCOP_SYNTHETIC_SMOKE_ROOT",
    "/kaggle/working/cropcop-smoke-input",
)
SMOKE_A_EXPORT_ROOT = os.environ.get(
    "CROPCOP_SMOKE_A_EXPORT_ROOT",
    "/kaggle/working/cropcop-smoke-a-export",
)
SMOKE_B_EXPORT_ROOT = os.environ.get(
    "CROPCOP_SMOKE_B_EXPORT_ROOT",
    "/kaggle/working/cropcop-smoke-b-export",
)
SMOKE_A_INPUT_ROOT = os.environ.get(
    "CROPCOP_SMOKE_A_INPUT_ROOT",
    "<SET_AFTER_ATTACHING_SMOKE_A_OUTPUT>",
)
SMOKE_B_INPUT_ROOT = os.environ.get(
    "CROPCOP_SMOKE_B_INPUT_ROOT",
    "<SET_AFTER_ATTACHING_SMOKE_B_OUTPUT>",
)
DUAL_GPU_SMOKE_INPUT_ROOT = os.environ.get(
    "CROPCOP_DUAL_GPU_SMOKE_INPUT_ROOT",
    "<SET_AFTER_ATTACHING_DUAL_GPU_SMOKE_OUTPUT>",
)

RFDV_ROOT = os.environ.get(
    "CROPCOP_RFDV_ROOT",
    "<SET_G1_RFDV_INPUT_ROOT>",
)
FINAL_V1_ROOT = os.environ.get(
    "CROPCOP_FINAL_V1_ROOT",
    "<SET_FROZEN_FINAL_V1_ROOT>",
)
G1_PRIVATE_DATASET_SLUG = os.environ.get(
    "CROPCOP_G1_PRIVATE_DATASET_SLUG",
    "<SET_OWNER/PRIVATE_G1_DATASET>",
)
G1_ALLOW_CREATE_PRIVATE_DATASET = os.environ.get(
    "CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET",
    "<SET_0_OR_1_FOR_G1>",
).strip()
G1_INPUT_ROOT = os.environ.get(
    "CROPCOP_G1_INPUT_ROOT",
    "<SET_AFTER_ATTACHING_SEALED_G1_DATASET>",
)

# Wrapper-only downstream preflight. This does not alter the frozen scientific source.
if EXECUTION_PHASE in {"calibration-dual", "principal-dual"}:
    _required_downstream_env = [
        "CROPCOP_MANIFEST",
        "CROPCOP_CLASS_MAP",
        "CROPCOP_IMAGE_ROOT",
        "CROPCOP_G2_SUMMARIES_DIR",
        "CROPCOP_DURABLE_STORE_KIND",
        "CROPCOP_DURABLE_LOCATOR_TEMPLATE",
    ]
    _missing_downstream = [
        _name for _name in _required_downstream_env
        if not str(os.environ.get(_name, "")).strip()
    ]
    if _missing_downstream:
        raise RuntimeError(
            "Missing required downstream environment variables before source clone: "
            + ", ".join(_missing_downstream)
        )

    _cnxtt_required = EXECUTION_PHASE == "calibration-dual" or (
        EXECUTION_PHASE == "principal-dual" and PRINCIPAL_ENVELOPE == "P3"
    )
    if _cnxtt_required:
        _cnxtt_raw = str(os.environ.get("CROPCOP_CNXTT_PRETRAINED", "")).strip()
        if not _cnxtt_raw:
            raise RuntimeError(
                "CROPCOP_CNXTT_PRETRAINED is required for G2 calibration and P3 principal."
            )
        _cnxtt_path = Path(_cnxtt_raw).resolve()
        if not _cnxtt_path.is_file():
            raise RuntimeError(
                f"CROPCOP_CNXTT_PRETRAINED is not a file: {_cnxtt_path}"
            )
        if _cnxtt_path.name != "convnext_tiny-983f1562.pth":
            raise RuntimeError(
                "CROPCOP_CNXTT_PRETRAINED must point to convnext_tiny-983f1562.pth"
            )
        print(f"ConvNeXt operator preflight: PASS ({_cnxtt_path})")

_FROZEN_V1_COLUMN_ENV = {
    "CROPCOP_ROW_ID_COLUMN": "record_key",
    "CROPCOP_PATH_COLUMN": "portable_relpath",
    "CROPCOP_SPLIT_COLUMN": "split",
    "CROPCOP_LABEL_COLUMN": "label",
}
if EXECUTION_PHASE in {"calibration-dual", "principal-dual"}:
    for _name, _expected in _FROZEN_V1_COLUMN_ENV.items():
        _observed = str(os.environ.get(_name, _expected)).strip()
        if _observed != _expected:
            raise RuntimeError(
                f"{_name} is frozen to {_expected!r}; observed {_observed!r}"
            )
        os.environ[_name] = _expected

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CROPCOP_NOTEBOOK_STARTED_MONOTONIC"] = repr(time.monotonic())
os.environ.setdefault("CROPCOP_NOTEBOOK_HARD_LIMIT_SECONDS", str(12 * 3600))
os.environ.setdefault("CROPCOP_NOTEBOOK_FINALIZATION_MARGIN_SECONDS", str(3600))
os.environ["CROPCOP_SOURCE_GIT_COMMIT"] = AUTHORIZED_SOURCE_SHA
os.environ["CROPCOP_LANE"] = LANE
os.environ["CROPCOP_EXECUTION_PHASE"] = EXECUTION_PHASE
os.environ["CROPCOP_OUTPUT_ROOT"] = OUTPUT_ROOT
os.environ["CROPCOP_SYNTHETIC_SMOKE_ROOT"] = SYNTHETIC_SMOKE_ROOT
os.environ["CROPCOP_SMOKE_A_EXPORT_ROOT"] = SMOKE_A_EXPORT_ROOT
os.environ["CROPCOP_SMOKE_B_EXPORT_ROOT"] = SMOKE_B_EXPORT_ROOT

assert platform.python_version() == "3.12.13", (
    f"Python re-lock required: {platform.python_version()}"
)
assert len(AUTHORIZED_SOURCE_SHA) == 40
assert all(c in "0123456789abcdef" for c in AUTHORIZED_SOURCE_SHA.lower())
assert LANE in {"K1", "K2", "K3"}
_ALLOWED_EXECUTION_PHASES = {
    "smoke-write",
    "smoke-restore",
    "dual-gpu-smoke",
    "g1",
    "calibration-dual",
    "principal-dual",
}
if EXECUTION_PHASE not in _ALLOWED_EXECUTION_PHASES:
    raise RuntimeError(
        "Set CROPCOP_EXECUTION_PHASE explicitly to exactly one of: "
        + ", ".join(sorted(_ALLOWED_EXECUTION_PHASES))
    )
if EXECUTION_PHASE == "principal-dual":
    if PRINCIPAL_ENVELOPE not in {"P1", "P2", "P3"}:
        raise RuntimeError(
            "Set CROPCOP_PRINCIPAL_ENVELOPE explicitly to exactly P1, P2, or P3 for principal-dual"
        )
    os.environ["CROPCOP_ENVELOPE_ID"] = PRINCIPAL_ENVELOPE


def _visible_nvidia_gpu_names() -> list[str]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    probe = subprocess.run(
        [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if probe.returncode != 0:
        return []
    return [line.strip() for line in probe.stdout.splitlines() if line.strip()]


def _validate_phase_hardware(phase: str) -> list[str]:
    names = _visible_nvidia_gpu_names()
    if phase in {"smoke-write", "smoke-restore"} and not names:
        raise RuntimeError(
            f"{phase} requires a CUDA-capable Kaggle GPU; observed no NVIDIA GPU. "
            "Choose the required GPU accelerator before Save Version -> Save & Run All."
        )
    if phase in {"dual-gpu-smoke", "calibration-dual", "principal-dual"}:
        if len(names) != 2 or any(name not in {"Tesla T4", "NVIDIA T4"} for name in names):
            raise RuntimeError(
                f"{phase} requires exactly Kaggle T4 x2; observed GPU inventory: {names or ['<none>']}"
            )
    if phase == "g1":
        print(
            "G1 hardware preflight: PASS "
            f"(CPU-defined; visible_nvidia_gpu_count={len(names)}; GPUs are not required)"
        )
    else:
        print(f"{phase} hardware preflight: PASS (visible_nvidia_gpus={names})")
    return names


_PHASE_GPU_NAMES = _validate_phase_hardware(EXECUTION_PHASE)

# Qualification runs must be Kaggle Saved-Version/Batch jobs.
# Fail before secrets, network access, clone, package installation, or any smoke output.
_kaggle_run_type = str(os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "")).strip()
if _kaggle_run_type != "Batch":
    raise RuntimeError(
        "CROPCOP QUALIFICATION NOT STARTED: this notebook is running in the Kaggle editor "
        f"with KAGGLE_KERNEL_RUN_TYPE={_kaggle_run_type or '<missing>'}. "
        "Do not qualify by pressing Run/Run All in the editor. "
        "Use Save Version -> Save & Run All. Select no accelerator for CPU G1, and the "
        "required T4x2 accelerator for dual-GPU phases. Interactive execution is diagnostic only. "
        "No source clone, dependency install, smoke checkpoint, or qualification evidence "
        "has been accepted from this session."
    )
print(f"Kaggle Saved-Version/Batch preflight: PASS (run_type={_kaggle_run_type})")

def _locate_exact_attached_evidence(root_value: str, filename: str, label: str) -> str:
    if not root_value or root_value.startswith("<"):
        raise RuntimeError(f"Set {label} to the exact attached Notebook Output root")
    root = Path(root_value).resolve()
    if not root.is_dir():
        raise RuntimeError(f"{label} does not exist or is not a directory: {root}")
    matches = sorted(path.resolve() for path in root.rglob(filename) if path.is_file())
    matches = [path for path in matches if path == root or root in path.parents]
    if len(matches) != 1:
        raise RuntimeError(
            f"{label} must contain exactly one {filename}; found {len(matches)} below explicit root {root}"
        )
    return str(matches[0])


def _resolve_g1_final_v1_root(root_value: str) -> str:
    if not root_value or root_value.startswith("<"):
        raise RuntimeError("Set CROPCOP_FINAL_V1_ROOT explicitly for CPU G1")
    root = Path(root_value).resolve()
    candidates = [root, (root / "CropCop_Final_v1").resolve()]
    valid = []
    for candidate in candidates:
        manifest = candidate / "audit" / "final_manifest.csv"
        class_map = candidate / "audit" / "class_to_idx.json"
        if manifest.is_file() and class_map.is_file():
            valid.append(candidate)
    checked = " | ".join(str(candidate) for candidate in candidates)
    print(f"G1 Final-V1 candidates checked: {checked}")
    if not valid:
        raise RuntimeError(
            "CROPCOP_FINAL_V1_ROOT has no valid canonical layout; "
            f"checked exactly: {checked}"
        )
    if len(valid) != 1:
        raise RuntimeError(
            "CROPCOP_FINAL_V1_ROOT is ambiguous; more than one allowed candidate is valid: "
            + " | ".join(str(candidate) for candidate in valid)
        )
    return str(valid[0])


def _validate_g1_rfdv_root(root_value: str) -> str:
    if not root_value or root_value.startswith("<"):
        raise RuntimeError("Set CROPCOP_RFDV_ROOT explicitly for CPU G1")
    root = Path(root_value).resolve()
    checkpoint = (
        root
        / "cropcop_runs"
        / "stage1_dino_tiny_ce_256"
        / "checkpoints"
        / "best_macro_f1.pt"
    )
    if not checkpoint.is_file():
        raise RuntimeError(
            "CROPCOP_RFDV_ROOT is missing the exact historical teacher checkpoint: "
            f"{checkpoint}"
        )
    print(f"G1 RFDV exact checkpoint preflight: PASS ({checkpoint})")
    return str(root)


def _validate_g1_create_policy(value: str) -> str:
    policy = str(value).strip()
    if policy not in {"0", "1"}:
        raise RuntimeError(
            "CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET must be explicitly set to exactly 0 or 1 for G1"
        )
    return policy


_G1_TARGET_READY_STATUSES = {"ready", "complete", "completed"}
_G1_TARGET_FAILED_STATUSES = {"error", "failed", "failure"}


def _decode_kaggle_status(raw) -> dict:
    if isinstance(raw, dict):
        return dict(raw)
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"status": text}
    return payload if isinstance(payload, dict) else {"status": str(payload)}


def _g1_target_metadata(api, slug: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        path = Path(api.dataset_metadata(slug, td))
        payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("info") or payload


def _g1_target_mine_refs(api, dataset_slug: str) -> set[str]:
    response = api.dataset_list_with_response(
        mine=True,
        search=dataset_slug,
        page_size=100,
    )
    datasets = getattr(response, "datasets", None) or []
    return {
        str(getattr(row, "ref", "") or "")
        for row in datasets
        if getattr(row, "ref", None)
    }


def _validate_g1_private_metadata(metadata: dict, slug: str) -> None:
    if metadata.get("isPrivate") is not True:
        raise RuntimeError(
            "G1 target exists but Kaggle metadata does not authoritatively report it private"
        )
    meta_id = str(metadata.get("id") or metadata.get("ref") or "")
    if meta_id and meta_id.casefold() != slug.casefold():
        raise RuntimeError(
            f"G1 target metadata identity mismatch: {meta_id!r} != {slug!r}"
        )


def _g1_target_probe(api, slug: str) -> dict:
    _, dataset_slug = slug.split("/", 1)
    state = {
        "metadata": None,
        "metadata_error": None,
        "mine_refs": set(),
        "mine_error": None,
        "status": None,
        "status_error": None,
    }
    try:
        state["metadata"] = _g1_target_metadata(api, slug)
        _validate_g1_private_metadata(state["metadata"], slug)
    except RuntimeError:
        raise
    except Exception as exc:
        state["metadata_error"] = f"{type(exc).__name__}: {exc}"
    try:
        state["mine_refs"] = _g1_target_mine_refs(api, dataset_slug)
    except Exception as exc:
        state["mine_error"] = f"{type(exc).__name__}: {exc}"
    try:
        state["status"] = _decode_kaggle_status(api.dataset_status(slug, format="json"))
    except Exception as exc:
        state["status_error"] = f"{type(exc).__name__}: {exc}"
    return state


def _g1_target_state_ready(state: dict, slug: str) -> bool:
    metadata = state.get("metadata")
    if metadata is None:
        return False
    _validate_g1_private_metadata(metadata, slug)
    refs = {str(ref).casefold() for ref in state.get("mine_refs", set())}
    if slug.casefold() not in refs:
        return False
    status = str((state.get("status") or {}).get("status", "")).strip().lower()
    if status in _G1_TARGET_FAILED_STATUSES:
        raise RuntimeError(f"Kaggle G1 private-target processing failed with status={status!r}")
    return status in _G1_TARGET_READY_STATUSES


def _g1_target_state_summary(state: dict, slug: str) -> dict:
    refs = {str(ref).casefold() for ref in state.get("mine_refs", set())}
    status_obj = state.get("status") or {}
    return {
        "metadata_visible": state.get("metadata") is not None,
        "metadata_error": state.get("metadata_error"),
        "mine_membership": slug.casefold() in refs,
        "mine_error": state.get("mine_error"),
        "dataset_status": status_obj.get("status"),
        "current_version_number": status_obj.get("current_version_number"),
        "status_error": state.get("status_error"),
    }


def _wait_for_g1_private_target(api, slug: str, *, timeout_seconds: float = 600.0) -> dict:
    deadline = time.monotonic() + timeout_seconds
    attempt = 0
    last_state = None
    while time.monotonic() < deadline:
        attempt += 1
        last_state = _g1_target_probe(api, slug)
        if _g1_target_state_ready(last_state, slug):
            summary = _g1_target_state_summary(last_state, slug)
            summary["attempts"] = attempt
            return summary
        if attempt == 1 or attempt % 6 == 0:
            print(
                "G1 private-target settle probe: "
                + json.dumps(_g1_target_state_summary(last_state, slug), sort_keys=True)
            )
        time.sleep(5)
    raise RuntimeError(
        "G1 private target did not become metadata-visible, mine-listed, and ready "
        f"within {timeout_seconds:.0f}s; last="
        + json.dumps(_g1_target_state_summary(last_state or {}, slug), sort_keys=True)
    )


def _redact_operator_secrets(text: str) -> str:
    safe = str(text or "")
    for name in ("KAGGLE_KEY", "CROPCOP_GITHUB_TOKEN", "GITHUB_TOKEN"):
        secret = str(os.environ.get(name, "") or "")
        if secret:
            safe = safe.replace(secret, "<redacted>")
    return safe


def _create_g1_private_target(slug: str) -> subprocess.CompletedProcess:
    _, dataset_slug = slug.split("/", 1)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "README.txt").write_text(
            "Private CropCop G1 transport target. Real sealed package versions are uploaded separately.\n",
            encoding="utf-8",
        )
        (root / "dataset-metadata.json").write_text(
            json.dumps(
                {
                    "title": dataset_slug,
                    "id": slug,
                    "licenses": [{"name": "other"}],
                    "isPrivate": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.run(
            ["kaggle", "datasets", "create", "-p", str(root), "-q", "-r", "skip"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1800,
        )


def _ensure_g1_private_target_settled(slug: str, policy: str, *, settle_timeout_seconds: float = 600.0) -> dict:
    username = str(os.environ.get("KAGGLE_USERNAME", "")).strip()
    if not username:
        raise RuntimeError("KAGGLE_USERNAME is required before G1 private-target preflight")
    owner, _ = slug.split("/", 1)
    if owner.casefold() != username.casefold():
        raise RuntimeError(
            f"G1 private-target owner mismatch: slug owner {owner!r} != KAGGLE_USERNAME {username!r}"
        )
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    initial = _g1_target_probe(api, slug)
    if _g1_target_state_ready(initial, slug):
        result = _g1_target_state_summary(initial, slug)
        result.update({"schema_version":"1.0","status":"PASS","dataset_slug":slug,"authenticated_username":username,"created_this_run":False,"creation_command_nonzero_but_target_settled":False})
        return result
    refs = {str(ref).casefold() for ref in initial.get("mine_refs", set())}
    existence_signal = initial.get("metadata") is not None or initial.get("status") is not None or slug.casefold() in refs
    if existence_signal:
        settled = _wait_for_g1_private_target(api, slug, timeout_seconds=settle_timeout_seconds)
        settled.update({"schema_version":"1.0","status":"PASS","dataset_slug":slug,"authenticated_username":username,"created_this_run":False,"creation_command_nonzero_but_target_settled":False})
        return settled
    if initial.get("mine_error"):
        raise RuntimeError("Kaggle authenticated 'mine' listing is unavailable; refusing to create a G1 target because ownership cannot be proven. " + str(initial["mine_error"]))
    if policy != "1":
        raise RuntimeError("G1 private target is absent/inaccessible and CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET=0")
    create = _create_g1_private_target(slug)
    if create.returncode != 0:
        try:
            settled = _wait_for_g1_private_target(api, slug, timeout_seconds=60.0)
        except RuntimeError as settle_exc:
            detail = _redact_operator_secrets((create.stderr or "").strip() or (create.stdout or "").strip())
            raise RuntimeError(f"Kaggle private-target create command failed and the target did not settle. rc={create.returncode}; detail={detail[-1600:] or '<no diagnostic>'}") from settle_exc
        settled.update({"schema_version":"1.0","status":"PASS","dataset_slug":slug,"authenticated_username":username,"created_this_run":False,"creation_command_nonzero_but_target_settled":True})
        return settled
    settled = _wait_for_g1_private_target(api, slug, timeout_seconds=settle_timeout_seconds)
    settled.update({"schema_version":"1.0","status":"PASS","dataset_slug":slug,"authenticated_username":username,"created_this_run":True,"creation_command_nonzero_but_target_settled":False})
    return settled



if EXECUTION_PHASE == "dual-gpu-smoke":
    os.environ["CROPCOP_INFRA_SMOKE_EVIDENCE"] = _locate_exact_attached_evidence(
        SMOKE_B_INPUT_ROOT,
        "SMOKE_B_EVIDENCE.json",
        "CROPCOP_SMOKE_B_INPUT_ROOT",
    )
elif EXECUTION_PHASE in {"g1", "calibration-dual", "principal-dual"}:
    os.environ["CROPCOP_INFRA_SMOKE_EVIDENCE"] = _locate_exact_attached_evidence(
        SMOKE_B_INPUT_ROOT,
        "SMOKE_B_EVIDENCE.json",
        "CROPCOP_SMOKE_B_INPUT_ROOT",
    )
    os.environ["CROPCOP_DUAL_GPU_SMOKE_EVIDENCE"] = _locate_exact_attached_evidence(
        DUAL_GPU_SMOKE_INPUT_ROOT,
        "DUAL_GPU_SMOKE_EVIDENCE.json",
        "CROPCOP_DUAL_GPU_SMOKE_INPUT_ROOT",
    )

if EXECUTION_PHASE == "g1":
    if not G1_PRIVATE_DATASET_SLUG or G1_PRIVATE_DATASET_SLUG.startswith("<"):
        raise RuntimeError("Set CROPCOP_G1_PRIVATE_DATASET_SLUG explicitly for CPU G1")
    _resolved_rfdv_root = _validate_g1_rfdv_root(RFDV_ROOT)
    _resolved_final_v1_root = _resolve_g1_final_v1_root(FINAL_V1_ROOT)
    _g1_create_policy = _validate_g1_create_policy(G1_ALLOW_CREATE_PRIVATE_DATASET)
    os.environ["CROPCOP_RFDV_ROOT"] = _resolved_rfdv_root
    os.environ["CROPCOP_FINAL_V1_ROOT"] = _resolved_final_v1_root
    os.environ["CROPCOP_G1_PRIVATE_DATASET_SLUG"] = G1_PRIVATE_DATASET_SLUG
    os.environ["CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET"] = _g1_create_policy
    print(
        "G1 private-target policy: "
        + ("explicit create-if-missing" if _g1_create_policy == "1" else "must already exist")
    )
elif EXECUTION_PHASE in {"calibration-dual", "principal-dual"}:
    if not G1_INPUT_ROOT or G1_INPUT_ROOT.startswith("<"):
        raise RuntimeError(
            "Set CROPCOP_G1_INPUT_ROOT to the exact attached sealed G1 Kaggle Dataset root"
        )
    _g1_input = Path(G1_INPUT_ROOT).resolve()
    if not _g1_input.is_dir():
        raise RuntimeError(f"CROPCOP_G1_INPUT_ROOT is not a directory: {_g1_input}")
    os.environ["CROPCOP_G1_INPUT_ROOT"] = str(_g1_input)


repo_workdir = Path(REPO_WORKDIR).resolve()
for _mutable in (
    OUTPUT_ROOT,
    SYNTHETIC_SMOKE_ROOT,
    SMOKE_A_EXPORT_ROOT,
    SMOKE_B_EXPORT_ROOT,
):
    _m = Path(_mutable).resolve()
    assert _m != repo_workdir and repo_workdir not in _m.parents, (
        f"mutable output must be outside Git checkout: {_m}"
    )

from kaggle_secrets import UserSecretsClient

_secrets = UserSecretsClient()
_required_secrets = ["CROPCOP_GITHUB_TOKEN"]
if EXECUTION_PHASE in {"g1", "calibration-dual", "principal-dual"}:
    _required_secrets += ["KAGGLE_USERNAME", "KAGGLE_KEY"]

for _key in _required_secrets:
    if not os.environ.get(_key):
        try:
            os.environ[_key] = _secrets.get_secret(_key)
        except Exception as _exc:
            raise RuntimeError(f"Required Kaggle Secret missing: {_key}") from _exc

_token = str(os.environ.get("CROPCOP_GITHUB_TOKEN", "")).strip()
if not _token:
    raise RuntimeError("CROPCOP_GITHUB_TOKEN is empty after Kaggle Secrets retrieval")
if any(ch.isspace() for ch in _token):
    raise RuntimeError(
        "CROPCOP_GITHUB_TOKEN contains whitespace/newlines; store the raw token only"
    )
if (_token.startswith("'") and _token.endswith("'")) or (
    _token.startswith('"') and _token.endswith('"')
):
    raise RuntimeError(
        "CROPCOP_GITHUB_TOKEN appears to include surrounding quotes; store the raw token only"
    )
os.environ["CROPCOP_GITHUB_TOKEN"] = _token

_repo_parts = urlparse(REPOSITORY_URL)
if _repo_parts.scheme != "https" or _repo_parts.netloc.lower() != "github.com":
    raise RuntimeError("Stage 01A-MGPU requires an HTTPS github.com repository URL")
_repo_segments = [part for part in _repo_parts.path.strip("/").split("/") if part]
if len(_repo_segments) != 2:
    raise RuntimeError(f"Unexpected GitHub repository URL path: {_repo_parts.path}")
_repo_owner = _repo_segments[0]
_repo_name = _repo_segments[1][:-4] if _repo_segments[1].endswith(".git") else _repo_segments[1]
_git_username = _repo_owner
os.environ["CROPCOP_GIT_USERNAME"] = _git_username

_api_url = f"https://api.github.com/repos/{_repo_owner}/{_repo_name}"
_api_request = Request(
    _api_url,
    headers={
        "Authorization": f"Bearer {_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "cropcop-kaggle-auth-preflight",
    },
)
try:
    with urlopen(_api_request, timeout=30) as _response:
        _repo_payload = json.loads(_response.read().decode("utf-8"))
except HTTPError as _exc:
    if _exc.code == 401:
        raise RuntimeError(
            "GitHub rejected CROPCOP_GITHUB_TOKEN (HTTP 401). "
            "The token is invalid, expired, revoked, or was copied incorrectly."
        ) from _exc
    if _exc.code == 404:
        raise RuntimeError(
            "GitHub could not expose the private repository to this token (HTTP 404). "
            "For a fine-grained PAT, select resource owner 'rana-m-ahmed', include "
            "repository 'ResearchWork-CropCop', and grant Contents: Read and write."
        ) from _exc
    if _exc.code == 403:
        raise RuntimeError(
            "GitHub denied the token by policy/permission (HTTP 403). "
            "Check token repository access, organization/SSO policy if applicable, and expiry."
        ) from _exc
    raise RuntimeError(f"GitHub repository authorization preflight failed with HTTP {_exc.code}") from _exc
except URLError as _exc:
    raise RuntimeError(f"GitHub authorization preflight network failure: {_exc.reason}") from _exc

if _repo_payload.get("full_name") != f"{_repo_owner}/{_repo_name}":
    raise RuntimeError("GitHub authorization preflight returned an unexpected repository identity")
if _repo_payload.get("private") is not True:
    raise RuntimeError("Expected CropCop repository to be private during smoke qualification")

print(
    "GitHub API auth preflight: PASS "
    f"(repo={_repo_owner}/{_repo_name}, token_value_not_printed=true)"
)

if repo_workdir.exists():
    shutil.rmtree(repo_workdir)

with tempfile.TemporaryDirectory() as td:
    askpass = Path(td) / "askpass.py"
    askpass.write_text(
        "#!/usr/bin/env python3\n"
        "import os,sys\n"
        "p=(sys.argv[1] if len(sys.argv)>1 else '').lower()\n"
        "if 'username' in p:\n"
        "    print(os.environ['CROPCOP_GIT_USERNAME'])\n"
        "elif 'password' in p:\n"
        "    print(os.environ['CROPCOP_GITHUB_TOKEN'])\n"
        "else:\n"
        "    raise SystemExit(2)\n"
    )
    askpass.chmod(askpass.stat().st_mode | stat.S_IXUSR)
    env = dict(os.environ)
    env["GIT_ASKPASS"] = str(askpass)
    env["GIT_ASKPASS_REQUIRE"] = "force"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_CONFIG_COUNT"] = "1"
    env["GIT_CONFIG_KEY_0"] = "credential.helper"
    env["GIT_CONFIG_VALUE_0"] = ""

    _ls_remote = subprocess.run(
        ["git", "ls-remote", "--exit-code", REPOSITORY_URL, "HEAD"],
        env=env,
        capture_output=True,
        text=True,
    )
    if _ls_remote.returncode != 0:
        _safe_stderr = (_ls_remote.stderr or "").replace(_token, "<redacted>")
        raise RuntimeError(
            "GitHub API token validation passed, but Git-over-HTTPS read authentication failed. "
            "The token may lack repository Contents read permission. "
            f"git ls-remote stderr: {_safe_stderr[-1200:]}"
        )
    print("GitHub Git-over-HTTPS read preflight: PASS")

    subprocess.run(
        [
            "git",
            "clone",
            "--no-checkout",
            "--filter=blob:none",
            REPOSITORY_URL,
            str(repo_workdir),
        ],
        env=env,
        check=True,
    )

    subprocess.run(
        ["git", "-C", str(repo_workdir), "checkout", "--detach", AUTHORIZED_SOURCE_SHA],
        env=env,
        check=True,
    )

    _probe_ref = f"refs/heads/run-evidence/auth-probe-{AUTHORIZED_SOURCE_SHA[:12]}"
    _push_probe = subprocess.run(
        [
            "git",
            "-C",
            str(repo_workdir),
            "push",
            "--dry-run",
            "origin",
            f"HEAD:{_probe_ref}",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if _push_probe.returncode != 0:
        _safe_stderr = (_push_probe.stderr or "").replace(_token, "<redacted>")
        raise RuntimeError(
            "Private clone/read succeeded, but GitHub evidence-branch write preflight failed. "
            "For a fine-grained PAT, grant Contents: Read and write on ResearchWork-CropCop. "
            f"git push --dry-run stderr: {_safe_stderr[-1200:]}"
        )
    print("GitHub evidence-branch write preflight: PASS (dry-run only; no ref created)")

actual = subprocess.check_output(
    ["git", "-C", str(repo_workdir), "rev-parse", "HEAD"],
    text=True,
).strip()
assert actual == AUTHORIZED_SOURCE_SHA, (
    f"HEAD mismatch: expected {AUTHORIZED_SOURCE_SHA}, got {actual}"
)
status = subprocess.check_output(
    ["git", "-C", str(repo_workdir), "status", "--porcelain=v1", "--untracked-files=all"],
    text=True,
)
assert not status.strip(), f"Git checkout is not clean: {status[:1000]}"

lockfile = repo_workdir / "journal_extension/requirements-training.lock.txt"
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "-r",
        str(lockfile),
    ],
    check=True,
)

bootstrap_out = Path(OUTPUT_ROOT) / "bootstrap" / "clean_session.json"
bootstrap_out.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [
        sys.executable,
        str(repo_workdir / "journal_extension/kaggle/bootstrap_clean_session.py"),
        "--repo-root",
        str(repo_workdir),
        "--authorized-source-sha",
        AUTHORIZED_SOURCE_SHA,
        "--phase",
        EXECUTION_PHASE,
        "--output",
        str(bootstrap_out),
    ],
    cwd=repo_workdir,
    check=True,
)

# Wrapper-only exact ConvNeXt transport/model-load qualification. This mirrors the
# frozen CAL-CNXTT identity checks without changing the execution source.
if EXECUTION_PHASE == "calibration-dual" or (
    EXECUTION_PHASE == "principal-dual" and PRINCIPAL_ENVELOPE == "P3"
):
    import hashlib
    _cnxtt_path = Path(os.environ["CROPCOP_CNXTT_PRETRAINED"]).resolve()
    _cnxtt_sha = hashlib.sha256(_cnxtt_path.read_bytes()).hexdigest()
    if not _cnxtt_sha.startswith("983f1562"):
        raise RuntimeError(
            "ConvNeXt operator preflight SHA mismatch: expected official "
            f"983f1562 prefix, observed {_cnxtt_sha}"
        )
    _cnxtt_probe_code = (
        "import sys;"
        "from pathlib import Path;"
        "repo=Path(sys.argv[1]).resolve();"
        "sys.path.insert(0,str(repo/'journal_extension'/'src'));"
        "from cropcop_je.models import create_convnext_tiny_from_pretrained;"
        "m=create_convnext_tiny_from_pretrained(sys.argv[2],seed=21270083,num_classes=120);"
        "print('CNXTT_MODEL_LOAD_PASS')"
    )
    _cnxtt_probe = subprocess.run(
        [sys.executable, "-c", _cnxtt_probe_code, str(repo_workdir), str(_cnxtt_path)],
        cwd=repo_workdir,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if _cnxtt_probe.returncode != 0:
        _detail = _redact_operator_secrets(
            (_cnxtt_probe.stderr or "") + "\n" + (_cnxtt_probe.stdout or "")
        )
        raise RuntimeError(
            "ConvNeXt operator model-load preflight failed before envelope launch: "
            + _detail[-4000:]
        )
    print(
        "ConvNeXt exact transport/model-load preflight: PASS "
        f"(sha256={_cnxtt_sha}, filename={_cnxtt_path.name})"
    )

# Principal sessions are fresh Kaggle Saved Versions. Rehydrate the exact
# public-safe G2 summaries from authenticated GitHub evidence, while pinning
# the already-qualified G2 barrier and summary identities. This wrapper-only
# transport must not alter or reinterpret the frozen scientific source.
if EXECUTION_PHASE == "principal-dual":
    _shared_g2 = Path(os.environ["CROPCOP_G2_SUMMARIES_DIR"]).resolve()
    if repo_workdir == _shared_g2 or repo_workdir in _shared_g2.parents:
        raise RuntimeError(
            "CROPCOP_G2_SUMMARIES_DIR must be outside the Git checkout"
        )
    _shared_g2.mkdir(parents=True, exist_ok=True)

    _required_g2 = ("CAL-MNV4-DIRECT", "CAL-MNV4-TEACHER", "CAL-CNXTT")
    _expected_g2_barrier_sha256 = "3be3ef666b1e9b479e0173cd74909c43023093e320bd0b3f2731f11e97d21a0a"
    _expected_g2_summary_sha256 = {
        "CAL-MNV4-DIRECT": "c89293bc4d0390737160b9aa39cd8aa973d5a54da3b7b83d8c21056e1942106d",
        "CAL-MNV4-TEACHER": "8439e003934d58d0d278145376fc452a7afbb93c599a34146f5e545cd6a65260",
        "CAL-CNXTT": "00d3518b3229ab16786d5d97d19ce0e16c966930740b2bf432ee81bc059a06e0",
    }

    def _github_evidence_json(branch: str, repo_path: str, label: str) -> dict:
        _encoded_path = quote(repo_path, safe="/")
        _encoded_ref = quote(branch, safe="")
        _url = (
            f"https://api.github.com/repos/{_repo_owner}/{_repo_name}/contents/"
            f"{_encoded_path}?ref={_encoded_ref}"
        )
        _request = Request(
            _url,
            headers={
                "Authorization": f"Bearer {_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "cropcop-principal-g2-handoff",
            },
        )
        try:
            with urlopen(_request, timeout=30) as _response:
                _payload = json.loads(_response.read().decode("utf-8"))
        except HTTPError as _exc:
            raise RuntimeError(
                f"{label} GitHub evidence fetch failed with HTTP {_exc.code} "
                f"(branch={branch}, path={repo_path})"
            ) from _exc
        except URLError as _exc:
            raise RuntimeError(
                f"{label} GitHub evidence fetch network failure "
                f"(branch={branch}, path={repo_path}): {_exc.reason}"
            ) from _exc
        if not isinstance(_payload, dict) or _payload.get("type") != "file":
            raise RuntimeError(
                f"{label} GitHub evidence response is not a file "
                f"(branch={branch}, path={repo_path})"
            )
        if _payload.get("path") != repo_path or _payload.get("encoding") != "base64":
            raise RuntimeError(
                f"{label} GitHub evidence response identity/encoding mismatch "
                f"(branch={branch}, path={repo_path})"
            )
        _encoded = str(_payload.get("content") or "")
        try:
            _raw = base64.b64decode(_encoded, validate=False)
        except Exception as _exc:
            raise RuntimeError(f"{label} GitHub evidence base64 decode failed") from _exc
        if not _raw or len(_raw) > 1_000_000:
            raise RuntimeError(
                f"{label} GitHub evidence payload size is invalid: {len(_raw)} bytes"
            )
        try:
            _obj = json.loads(_raw.decode("utf-8"))
        except Exception as _exc:
            raise RuntimeError(f"{label} GitHub evidence is not valid UTF-8 JSON") from _exc
        if not isinstance(_obj, dict):
            raise RuntimeError(f"{label} GitHub evidence JSON must be an object")
        return _obj

    _src = repo_workdir / "journal_extension" / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    from cropcop_je.g2 import (
        validate_calibration_summary as _validate_calibration_summary,
        validate_g2_barrier_object as _validate_g2_barrier_object,
    )
    from cropcop_je.hashing import sha256_json as _sha256_json

    _g2_barrier = _github_evidence_json(
        "run-evidence/G2-CALIBRATION-BARRIER",
        "journal_extension/evidence/public/runs/G2-CALIBRATION-BARRIER/G2_CALIBRATION_BARRIER.json",
        "G2 barrier",
    )
    _g2_barrier_errors = _validate_g2_barrier_object(
        _g2_barrier,
        expected_source_sha=AUTHORIZED_SOURCE_SHA,
    )
    if _g2_barrier_errors:
        raise RuntimeError(
            "Qualified G2 barrier validation failed during principal handoff: "
            + "; ".join(_g2_barrier_errors)
        )
    if _g2_barrier.get("barrier_sha256") != _expected_g2_barrier_sha256:
        raise RuntimeError(
            "Qualified G2 barrier identity changed: expected "
            f"{_expected_g2_barrier_sha256}, observed {_g2_barrier.get('barrier_sha256')}"
        )
    if tuple(_g2_barrier.get("required_calibrations") or ()) != _required_g2:
        raise RuntimeError("Qualified G2 barrier required-calibration set changed")
    if _g2_barrier.get("input_summary_sha256") != _expected_g2_summary_sha256:
        raise RuntimeError("Qualified G2 barrier summary-hash map changed")

    _hydrated_hashes = {}
    for _cid in _required_g2:
        _branch = f"run-evidence/{_cid}"
        _repo_path = f"journal_extension/evidence/public/runs/{_cid}/{_cid}.json"
        _summary = _github_evidence_json(_branch, _repo_path, _cid)
        _summary_errors = _validate_calibration_summary(_summary)
        if _summary_errors:
            raise RuntimeError(
                f"{_cid} calibration summary validation failed during principal handoff: "
                + "; ".join(_summary_errors)
            )
        if _summary.get("calibration_id") != _cid:
            raise RuntimeError(f"{_cid} calibration summary identity mismatch")
        if _summary.get("source_git_commit") != AUTHORIZED_SOURCE_SHA:
            raise RuntimeError(f"{_cid} calibration summary source SHA mismatch")
        _observed_hash = _sha256_json(_summary)
        _expected_hash = _expected_g2_summary_sha256[_cid]
        if _observed_hash != _expected_hash:
            raise RuntimeError(
                f"{_cid} calibration summary hash mismatch: "
                f"expected {_expected_hash}, observed {_observed_hash}"
            )
        if _g2_barrier["input_summary_sha256"].get(_cid) != _observed_hash:
            raise RuntimeError(f"{_cid} calibration summary no longer matches qualified G2 barrier")
        _dest = _shared_g2 / f"{_cid}.json"
        _tmp = _dest.with_suffix(".json.tmp")
        _tmp.write_text(
            json.dumps(_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(_tmp, _dest)
        _hydrated_hashes[_cid] = _observed_hash

    _hydration_audit = {
        "schema_version": "1.0",
        "status": "PASS",
        "transport": "authenticated-github-contents-api",
        "source_git_commit": AUTHORIZED_SOURCE_SHA,
        "g2_barrier_sha256": _expected_g2_barrier_sha256,
        "input_summary_sha256": _hydrated_hashes,
        "scientific_source_modified": False,
    }
    _hydration_audit_path = _shared_g2 / "G2_HYDRATION_AUDIT.json"
    _hydration_audit_tmp = _hydration_audit_path.with_suffix(".json.tmp")
    _hydration_audit_tmp.write_text(
        json.dumps(_hydration_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(_hydration_audit_tmp, _hydration_audit_path)

    _missing_g2 = [
        _cid for _cid in _required_g2
        if not (_shared_g2 / f"{_cid}.json").is_file()
    ]
    if _missing_g2:
        raise RuntimeError(
            "Principal G2 handoff is incomplete after authenticated hydration: "
            + ", ".join(_missing_g2)
        )
    print(
        "Principal G2 authenticated evidence hydration: PASS "
        f"(barrier_sha256={_expected_g2_barrier_sha256}, "
        "CAL-MNV4-DIRECT, CAL-MNV4-TEACHER, CAL-CNXTT)"
    )

# G1 external target orchestration happens only after clean-source/dependency/bootstrap
# validation and before the frozen G1 entrypoint independently revalidates the target.
if EXECUTION_PHASE == "g1":
    _g1_target_preflight = _ensure_g1_private_target_settled(
        G1_PRIVATE_DATASET_SLUG,
        G1_ALLOW_CREATE_PRIVATE_DATASET,
    )
    _g1_target_record = bootstrap_out.parent / "g1_private_target_preflight.json"
    _g1_target_record.write_text(
        json.dumps(_g1_target_preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "G1 private target wrapper preflight: PASS "
        f"(slug={G1_PRIVATE_DATASET_SLUG}, "
        f"created_this_run={_g1_target_preflight['created_this_run']}, "
        f"status={_g1_target_preflight.get('dataset_status')})"
    )

if EXECUTION_PHASE in {"smoke-write", "smoke-restore"}:
    cmd = [
        sys.executable,
        str(repo_workdir / "journal_extension/scripts/smoke_infrastructure.py"),
        "--mode",
        "write" if EXECUTION_PHASE == "smoke-write" else "restore",
        "--repo-root",
        str(repo_workdir),
        "--authorized-source-sha",
        AUTHORIZED_SOURCE_SHA,
        "--lane",
        LANE,
        "--runtime-output-root",
        OUTPUT_ROOT,
        "--synthetic-root",
        SYNTHETIC_SMOKE_ROOT,
        "--smoke-a-export-root",
        SMOKE_A_EXPORT_ROOT,
        "--smoke-b-export-root",
        SMOKE_B_EXPORT_ROOT,
    ]
    if EXECUTION_PHASE == "smoke-restore":
        if SMOKE_A_INPUT_ROOT.startswith("<"):
            raise RuntimeError(
                "Set SMOKE_A_INPUT_ROOT to the attached Smoke-A /kaggle/input/... path"
            )
        cmd += ["--smoke-a-input-root", SMOKE_A_INPUT_ROOT]
    subprocess.run(cmd, cwd=repo_workdir, check=True)
elif EXECUTION_PHASE == "dual-gpu-smoke":
    subprocess.run(
        [sys.executable, str(repo_workdir / "journal_extension/scripts/smoke_dual_gpu.py")],
        cwd=repo_workdir,
        check=True,
    )
elif EXECUTION_PHASE == "g1":
    subprocess.run(
        [sys.executable, str(repo_workdir / "journal_extension/kaggle/run_g1.py")],
        cwd=repo_workdir,
        check=True,
    )
elif EXECUTION_PHASE in {"calibration-dual", "principal-dual"}:
    # The frozen source performs a serialized second envelope publication after
    # recording its branch in ENVELOPE_EVIDENCE.json. In the known idempotent
    # case only ENVELOPE_EVIDENCE.json changes, while the frozen publication
    # helper expects every allowlisted file to be staged. Keep the scientific
    # source immutable and repair only that parent-publication tail here.
    _envelope_cp = subprocess.run(
        [sys.executable, str(repo_workdir / "journal_extension/kaggle/run_envelope.py")],
        cwd=repo_workdir,
        check=False,
    )
    if _envelope_cp.returncode != 0:
        _envelope_cfg_name = (
            "G2_DUAL_T4.json"
            if EXECUTION_PHASE == "calibration-dual"
            else {"P1": "P1_S1_PAIR.json", "P2": "P2_S2_PAIR.json", "P3": "P3_S3_PAIR.json"}[PRINCIPAL_ENVELOPE]
        )
        _envelope_cfg = json.loads(
            (repo_workdir / "journal_extension/kaggle/envelopes" / _envelope_cfg_name).read_text(
                encoding="utf-8"
            )
        )
        _envelope_id = _envelope_cfg["envelope_id"]
        _envelope_root = Path(
            os.environ.get(
                "CROPCOP_ENVELOPE_OUTPUT_ROOT",
                str(Path(OUTPUT_ROOT) / "envelopes" / _envelope_id),
            )
        ).resolve()
        _state_path = _envelope_root / "ENVELOPE_STATE.json"
        _evidence_path = _envelope_root / "ENVELOPE_EVIDENCE.json"
        _repair_ok = False
        if _state_path.is_file() and _evidence_path.is_file():
            _state = json.loads(_state_path.read_text(encoding="utf-8"))
            _evidence = json.loads(_evidence_path.read_text(encoding="utf-8"))
            _branch = str(_evidence.get("envelope_publication_branch") or "").strip()
            if _state.get("state") in {"PASS", "CONTINUATION_REQUIRED"} and _branch:
                _src = repo_workdir / "journal_extension" / "src"
                if str(_src) not in sys.path:
                    sys.path.insert(0, str(_src))
                from cropcop_je.publication import publish_to_github_branch
                _repaired_branch = publish_to_github_branch(
                    repo_dir=repo_workdir,
                    source_git_sha=AUTHORIZED_SOURCE_SHA,
                    run_id=_envelope_id,
                    files=[_evidence_path],
                )
                if _repaired_branch != _branch:
                    raise RuntimeError(
                        "wrapper publication repair changed the envelope evidence branch"
                    )
                _repair_ok = True
                print(
                    "Envelope terminal publication repair: PASS "
                    f"(branch={_repaired_branch}, scientific_execution_relaunched=false)"
                )
        if not _repair_ok:
            # Surface child-local diagnostics into the Saved Version log. Child
            # processes have Git credentials stripped; redact remaining operator
            # secrets defensively before printing.
            _log_root = Path(
                os.environ.get(
                    "CROPCOP_ENVELOPE_OUTPUT_ROOT",
                    str(Path(OUTPUT_ROOT) / "envelopes"),
                )
            ).resolve()
            _logs = sorted(_log_root.rglob("console.log")) if _log_root.exists() else []
            for _log in _logs:
                try:
                    _tail = _log.read_text(encoding="utf-8", errors="replace")[-8000:]
                except Exception as _log_exc:
                    _tail = f"<unable to read console log: {type(_log_exc).__name__}: {_log_exc}>"
                print(
                    "\n===== ENVELOPE CHILD CONSOLE TAIL: "
                    + str(_log)
                    + " =====\n"
                    + _redact_operator_secrets(_tail)
                )
            raise subprocess.CalledProcessError(
                _envelope_cp.returncode,
                _envelope_cp.args,
            )
else:
    raise RuntimeError(f"unsupported canonical execution phase: {EXECUTION_PHASE}")
'''


def _source_lines(source: str) -> list[str]:
    # Jupyter-compatible source serialization: physical Python lines with LF endings.
    return source.splitlines(keepends=True)


def build_notebook() -> dict:
    compile(CODE, "canonical_lane.ipynb", "exec")
    if len(CODE.splitlines()) <= 40:
        raise RuntimeError("canonical notebook source unexpectedly collapsed")
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": _source_lines(MARKDOWN),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": _source_lines(CODE),
            },
        ],
    }


def main() -> int:
    target = Path(__file__).with_name("canonical_lane.ipynb")
    notebook = build_notebook()
    target.write_text(
        json.dumps(notebook, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
