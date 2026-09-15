from __future__ import annotations

import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from tracka_v12_kaggle_operator_v8 import (
    SCIENCE_SHA,
    R13_PARITY_CONTRACT_ID_V8,
    R13_PARITY_REQUIRED_MAX_ABS_V8,
    OperatorError,
    load_json,
    sha256_file,
)

RELEASE_MANIFEST_FILE_SHA256 = "a89ad7368eaf6f685087cffe4e428dfb5295d0909dc60d448b871946f3e2b61f"
CODE_ATTESTATION_MEMBER_SHA256 = "5bfb757219996aa7df48e21fd0531124c60a86f6a0c8ad17840a2a3eb88cda2e"
LOCK_RUNTIME_ATTESTATION_MEMBER_SHA256 = "49efe43b76423c82c6849d43d3f1bea2ff24503f65564e722d205cdac2a21fea"
REPOSITORY = "rana-m-ahmed/ResearchWork-CropCop"


def verified_release_attestation_manifest() -> tuple[Path, dict]:
    path = Path(__file__).resolve().parent / "attestations_v8r2" / "RELEASE_ATTESTATION_MANIFEST.json"
    if not path.is_file():
        raise OperatorError("v8r2 release attestation manifest is missing")
    if sha256_file(path) != RELEASE_MANIFEST_FILE_SHA256:
        raise OperatorError("v8r2 release attestation manifest bytes changed")

    payload = load_json(path)
    if payload.get("schema_version") != "1.0" or payload.get("status") != "LOCKED_PRE_SCIENCE":
        raise OperatorError("v8r2 release attestation manifest status/schema mismatch")
    if payload.get("science_source_sha") != SCIENCE_SHA:
        raise OperatorError("v8r2 release attestation manifest science source mismatch")
    if payload.get("science_authorized") is not False:
        raise OperatorError("v8r2 release attestation manifest unexpectedly authorizes science")
    if payload.get("protected_test_accessed") is not False or payload.get("external_surface_accessed") is not False:
        raise OperatorError("v8r2 release attestation manifest reports protected-surface access")
    if int(payload.get("remaining_scientific_states", -1)) != 11:
        raise OperatorError("v8r2 release attestation manifest does not bind exactly 11 remaining states")

    parity = payload.get("r13_parity_contract") or {}
    if parity.get("contract_id") != R13_PARITY_CONTRACT_ID_V8:
        raise OperatorError("v8r2 release attestation manifest parity-contract mismatch")
    if float(parity.get("required_max_abs_difference", -1.0)) != R13_PARITY_REQUIRED_MAX_ABS_V8:
        raise OperatorError("v8r2 release attestation manifest parity tolerance mismatch")
    if float(parity.get("historical_v1_2_required_max_abs_difference", -1.0)) != 1e-5:
        raise OperatorError("v8r2 release attestation manifest lost historical parity provenance")
    if parity.get("exact_pretrained_gate_pass") is not True:
        raise OperatorError("v8r2 release attestation manifest lacks exact-pretrained parity PASS")

    code = payload.get("code_attestation") or {}
    lock = payload.get("lock_runtime_attestation") or {}
    if code.get("status") != "PASS" or code.get("member_sha256") != CODE_ATTESTATION_MEMBER_SHA256:
        raise OperatorError("v8r2 code-attestation provenance mismatch")
    if lock.get("status") != "PASS" or lock.get("member_sha256") != LOCK_RUNTIME_ATTESTATION_MEMBER_SHA256:
        raise OperatorError("v8r2 lock/runtime-attestation provenance mismatch")

    required = set(payload.get("required_static_pre_science_gates") or [])
    expected = {
        "exact_head_ci",
        "kaggle_generation_durability_contract",
        "g1a_runtime_global_resolution_contract",
        "r13_parity_amendment_contract",
        "principal_science_diff",
        "secondary_science_diff",
        "checkpoint_recovery_contract",
        "cross_slot_recovery_contract",
        "six_gpu_parent_orchestration",
        "training_runner_contract",
    }
    if required != expected:
        raise OperatorError("v8r2 release attestation manifest required-gate set drifted")
    return path, payload


def _download_artifact_zip(artifact_id: int, token: str, *, attempts: int = 3) -> bytes:
    url = f"https://api.github.com/repos/{REPOSITORY}/actions/artifacts/{artifact_id}/zip"
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "CropCop-TrackA-v8r2",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            if not data:
                raise OperatorError(f"GitHub artifact {artifact_id} download returned empty bytes")
            return data
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, OperatorError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep((5, 15, 30)[min(attempt - 1, 2)])
    raise OperatorError(f"unable to download exact-head GitHub artifact {artifact_id}: {last_error}")


def _member_bytes(zip_bytes: bytes, member_name: str) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as archive:
            matches = [name for name in archive.namelist() if Path(name).name == member_name and not name.endswith("/")]
            if len(matches) != 1:
                raise OperatorError(f"artifact member {member_name!r} must occur exactly once; found {matches}")
            return archive.read(matches[0])
    except zipfile.BadZipFile as exc:
        raise OperatorError("GitHub Actions artifact is not a valid ZIP archive") from exc


def _validate_attestation_payload(payload: dict, *, kind: str, required_gates: set[str]) -> None:
    if payload.get("status") != "PASS" or payload.get("github_actions") is not True:
        raise OperatorError(f"materialized {kind} attestation is not a GitHub Actions PASS")
    if payload.get("source_git_commit") != SCIENCE_SHA or payload.get("pull_request_head_sha") != SCIENCE_SHA:
        raise OperatorError(f"materialized {kind} attestation source mismatch")
    if payload.get("r13_parity_contract_id") != R13_PARITY_CONTRACT_ID_V8:
        raise OperatorError(f"materialized {kind} attestation parity-contract mismatch")
    if float(payload.get("r13_parity_required_max_abs_difference", -1.0)) != R13_PARITY_REQUIRED_MAX_ABS_V8:
        raise OperatorError(f"materialized {kind} attestation parity tolerance mismatch")
    if kind == "code":
        gates = payload.get("static_pre_science_gates") or {}
        missing = [name for name in sorted(required_gates) if gates.get(name) != "PASS"]
        if missing:
            raise OperatorError("materialized code attestation lacks required PASS gates: " + ", ".join(missing))
        implementation = payload.get("implementation_sha256") or {}
        for required_path in (
            "journal_extension/kaggle/run_tracka_v12_account_v121.py",
            "journal_extension/scripts/seal_tracka_v12_science_go_v124.py",
            "journal_extension/scripts/seal_tracka_v12_g1a_v121.py",
            "journal_extension/scripts/run_tracka_v12_training_v121.py",
        ):
            value = str(implementation.get(required_path, ""))
            if len(value) != 64:
                raise OperatorError(f"materialized code attestation does not bind {required_path}")
    else:
        if payload.get("science_authorized") is not False:
            raise OperatorError("materialized lock/runtime attestation unexpectedly authorizes science")
        if payload.get("r13_exact_pretrained_parity") != "PASS" or payload.get("r13_v121_parity_amendment") != "PASS":
            raise OperatorError("materialized lock/runtime attestation lacks parity qualification PASS")
        exact = payload.get("exact_pretrained_parity_report") or {}
        if exact.get("v1_2_1_gate_pass") is not True or exact.get("historical_v1_2_gate_pass") is not False:
            raise OperatorError("materialized lock/runtime attestation does not prove superseding parity behavior")
        if exact.get("scientific_metric_computed") is not False or exact.get("v1_test_accessed") is not False:
            raise OperatorError("materialized lock/runtime attestation touched a protected/scientific surface")


def materialize_verified_attestations(destination: str | Path) -> tuple[Path, Path]:
    _manifest_path, manifest = verified_release_attestation_manifest()
    token = str(os.environ.get("CROPCOP_GITHUB_TOKEN", "") or "").strip()
    if not token:
        raise OperatorError("exact-head attestation materialization requires CROPCOP_GITHUB_TOKEN")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    required_gates = set(manifest["required_static_pre_science_gates"])
    outputs: dict[str, Path] = {}

    for kind, key in (("code", "code_attestation"), ("lock", "lock_runtime_attestation")):
        spec = manifest[key]
        member = str(spec["member"])
        expected_sha = str(spec["member_sha256"])
        target = destination / member
        if target.is_file() and sha256_file(target) == expected_sha:
            payload = load_json(target)
            _validate_attestation_payload(payload, kind=kind, required_gates=required_gates)
            outputs[kind] = target
            continue

        zip_bytes = _download_artifact_zip(int(spec["artifact_id"]), token)
        data = _member_bytes(zip_bytes, member)
        observed_sha = hashlib.sha256(data).hexdigest()
        if observed_sha != expected_sha:
            raise OperatorError(
                f"materialized {kind} attestation byte hash mismatch: expected {expected_sha}, got {observed_sha}"
            )
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OperatorError(f"materialized {kind} attestation is not valid UTF-8 JSON") from exc
        _validate_attestation_payload(payload, kind=kind, required_gates=required_gates)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)
        outputs[kind] = target

    return outputs["code"], outputs["lock"]


# Historical compatibility only; active v8r2 control uses materialize_verified_attestations().
def verified_attestation_paths() -> tuple[Path, Path]:
    path, _ = verified_release_attestation_manifest()
    return path, path
