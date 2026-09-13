from __future__ import annotations

import shutil
import sys
from pathlib import Path

from tracka_v12_kaggle_operator_v3 import OperatorError, load_json, run, sha256_file


def _science_validator(repo: Path):
    src = repo / "journal_extension" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from cropcop_je.secondary import validate_torchvision_provenance
    return validate_torchvision_provenance


def prepare_verified_torchvision(repo: str | Path, work_root: str | Path) -> dict[str, dict[str, Path]]:
    """Prepare official TorchVision bytes, then capture exact tensor-identity provenance."""
    repo = Path(repo).resolve()
    work_root = Path(work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    validator = _science_validator(repo)
    result: dict[str, dict[str, Path]] = {}

    for key in ("effb0", "cnxtt"):
        model_root = work_root / key
        download_record = work_root / f"{key}_download_record.json"
        provenance = work_root / f"{key}_pretrained_provenance.json"
        shutil.rmtree(model_root, ignore_errors=True)
        download_record.unlink(missing_ok=True)
        provenance.unlink(missing_ok=True)

        run([
            sys.executable,
            str(repo / "journal_extension" / "scripts" / "prepare_torchvision_pretrained.py"),
            "--model-key", key,
            "--output-dir", str(model_root),
            "--record", str(download_record),
        ], cwd=repo)

        prepared = load_json(download_record)
        if prepared.get("status") != "PASS" or prepared.get("model_key") != key:
            raise OperatorError(f"{key} TorchVision preparation receipt is not PASS for the requested model")
        artifact = model_root / str(prepared.get("official_filename", ""))
        if not artifact.is_file():
            raise OperatorError(f"{key} official TorchVision artifact missing after preparation: {artifact}")
        if sha256_file(artifact) != prepared.get("artifact_sha256"):
            raise OperatorError(f"{key} official TorchVision artifact differs from preparation receipt")
        if artifact.stat().st_size != int(prepared.get("artifact_bytes", -1)):
            raise OperatorError(f"{key} official TorchVision artifact byte count differs from preparation receipt")

        run([
            sys.executable,
            str(repo / "journal_extension" / "scripts" / "capture_torchvision_pretrained_provenance.py"),
            "--model-key", key,
            "--artifact", str(artifact),
            "--output", str(provenance),
        ], cwd=repo)

        captured = load_json(provenance)
        for field, expected in {
            "status": "PASS",
            "model_key": key,
            "official_tensor_match": True,
        }.items():
            if captured.get(field) != expected:
                raise OperatorError(f"{key} captured TorchVision provenance {field} mismatch")
        if captured.get("artifact_sha256") != prepared.get("artifact_sha256"):
            raise OperatorError(f"{key} capture/preparation artifact SHA mismatch")
        if captured.get("artifact_bytes") != prepared.get("artifact_bytes"):
            raise OperatorError(f"{key} capture/preparation artifact byte-count mismatch")
        if not captured.get("tensor_identity_algorithm"):
            raise OperatorError(f"{key} captured TorchVision provenance lacks tensor identity algorithm")
        if captured.get("tensor_identity_algorithm") != captured.get("official_tensor_identity_algorithm"):
            raise OperatorError(f"{key} candidate/official tensor identity algorithm mismatch")
        if captured.get("tensor_identity_sha256") != captured.get("official_tensor_identity_sha256"):
            raise OperatorError(f"{key} candidate/official tensor identity SHA mismatch")

        errors = validator(captured, model_key=key, artifact_path=artifact)
        if errors:
            raise OperatorError(f"{key} frozen TorchVision provenance validator rejected capture: " + "; ".join(errors))

        result[key] = {
            "artifact": artifact,
            "download_record": download_record,
            "provenance": provenance,
        }

    return result
