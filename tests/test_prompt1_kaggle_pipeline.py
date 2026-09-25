from __future__ import annotations

import importlib.util
import base64
import gzip
import inspect
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "journal_extension" / "scripts" / "run_prompt1_kaggle.py"
SPEC = importlib.util.spec_from_file_location("prompt1_kaggle", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@dataclass(frozen=True)
class Row:
    stable_row_id: str
    relative_path: str
    split: str
    class_index: int


def test_json_self_seal_round_trip() -> None:
    payload = MODULE.seal_json({"schema_version": "1.0", "value": [3, 1, 2]})
    MODULE.validate_sealed(payload)
    assert len(payload["canonical_self_sha256"]) == 64


def test_calibration_manifest_is_deterministic_balanced_and_unique() -> None:
    rows = [
        Row(f"row-{class_index:03d}-{item:03d}", f"train/{class_index}/{item}.jpg", "train", class_index)
        for class_index in range(120)
        for item in range(12)
    ]
    first = MODULE.calibration_rows(rows)
    second = MODULE.calibration_rows(list(reversed(rows)))
    assert [row.stable_row_id for row in first] == [row.stable_row_id for row in second]
    assert len(first) == len({row.stable_row_id for row in first}) == 1024
    counts = {index: 0 for index in range(120)}
    for row in first:
        counts[row.class_index] += 1
    assert min(counts.values()) >= 8


def test_metrics_from_logits_perfect_predictions() -> None:
    targets = np.arange(120, dtype=np.int64)
    logits = np.full((120, 120), -5.0, dtype=np.float32)
    logits[np.arange(120), targets] = 5.0
    result = MODULE.metrics_from_logits(logits, targets)
    assert result["validation_accuracy"] == 1.0
    assert result["validation_balanced_accuracy"] == 1.0
    assert result["validation_macro_f1"] == 1.0
    assert result["validation_nll"] > 0.0


def test_streaming_metrics_reject_inconsistent_accumulator() -> None:
    confusion = np.zeros((120, 120), dtype=np.int64)
    confusion[0, 0] = 1
    result = MODULE.metrics_from_streaming_counts(confusion, correct=1, count=1, nll_sum=0.25)
    assert result["validation_accuracy"] == 1.0
    assert result["validation_nll"] == 0.25
    try:
        MODULE.metrics_from_streaming_counts(confusion, correct=1, count=2, nll_sum=0.25)
    except MODULE.GateError:
        pass
    else:
        raise AssertionError("inconsistent count must fail closed")


def test_publication_scan_rejects_validation_derived_files(tmp_path: Path) -> None:
    safe = tmp_path / "safe"
    safe.mkdir()
    (safe / "manifest.json").write_text("{}", encoding="utf-8")
    assert MODULE.publication_scan(safe)["status"] == "PASS"
    (safe / "source_logits.npy").write_bytes(b"restricted")
    result = MODULE.publication_scan(safe)
    assert result["status"] == "FAIL"
    assert result["restricted_file_offenders"] == ["source_logits.npy"]


def test_publication_scan_requires_exactly_one_classifier_when_requested(tmp_path: Path) -> None:
    production = tmp_path / "production"
    production.mkdir()
    assert MODULE.publication_scan(production, require_single_classifier=True)["status"] == "FAIL"
    (production / "model.pte").write_bytes(b"pte")
    assert MODULE.publication_scan(production, require_single_classifier=True)["status"] == "PASS"
    (production / "other.pte").write_bytes(b"pte")
    assert MODULE.publication_scan(production, require_single_classifier=True)["status"] == "FAIL"


def test_final_zip_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    root = tmp_path / "PROMPT1_FINAL_EVIDENCE"
    root.mkdir()
    (root / "payload.txt").write_text("sealed", encoding="utf-8")
    MODULE.write_json(root / "FINAL_FILE_MANIFEST.json", MODULE.directory_manifest(root))
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                handle.write(path, Path(root.name) / path.relative_to(root))
    assert MODULE.verify_final_zip(archive, root)["status"] == "PASS"
    with zipfile.ZipFile(archive, "a") as handle:
        handle.writestr(f"{root.name}/unexpected.txt", b"tamper")
    try:
        MODULE.verify_final_zip(archive, root)
    except MODULE.GateError:
        pass
    else:
        raise AssertionError("unexpected ZIP member must fail closed")


def test_frozen_runtime_and_ptq_policies_are_single_route() -> None:
    assert MODULE.EXECUTORCH_VERSION == "1.3.1"
    assert MODULE.PTQ_POLICY["route_id"] == "R07-XNNPACK-PT2E-STATIC-INT8-PC-v1"
    assert MODULE.PTQ_POLICY["calibration_count"] == 1024
    assert MODULE.FP32_PARITY_POLICY["top1_agreement_min"] == 1.0
    assert MODULE.DEVICE_POLICY["performance_based_selection_forbidden"] is True
    assert MODULE.NUMPY_VERSION == "2.5.2"
    args = MODULE.parser().parse_args(["preflight", "--input-root", "in", "--work-root", "out"])
    assert args.source_batch_size == 16


def test_ptq_conversion_does_not_call_eval_on_exported_graph() -> None:
    for function in (MODULE.executorch_ptq_toolchain_probe, MODULE.stage_ptq):
        source = inspect.getsource(function)
        assert "quantized = convert_pt2e(prepared)" in source
        assert "convert_pt2e(prepared).eval()" not in source
        assert "quantized.eval()" not in source


def test_repo_discovery_accepts_pinned_checkout_layout(tmp_path: Path) -> None:
    marker = tmp_path / "journal_extension" / "src" / "cropcop_je" / "trackb_r07.py"
    marker.parent.mkdir(parents=True)
    marker.write_text("# marker\n", encoding="utf-8")
    assert MODULE.find_repo(tmp_path) == tmp_path.resolve()


def test_final_v1_resolver_excludes_certification_report_copy(tmp_path: Path, monkeypatch) -> None:
    real = tmp_path / "CropCop_Final_v1"
    reports = tmp_path / "CropCop_Final_v1_CERTIFICATION_REPORTS"
    for base in (real, reports):
        (base / "audit").mkdir(parents=True)
        (base / "audit" / "final_manifest.csv").write_text("manifest\n", encoding="utf-8")
        (base / "audit" / "class_to_idx.json").write_text("{}\n", encoding="utf-8")
        (base / "dataset" / "train").mkdir(parents=True)
        (base / "dataset" / "val").mkdir(parents=True)

    def fake_sha(path: Path) -> str:
        if Path(path).name == "final_manifest.csv":
            return MODULE.MANIFEST_SHA256
        if Path(path).name == "class_to_idx.json":
            return MODULE.CLASS_MAP_SHA256
        return "0" * 64

    monkeypatch.setattr(MODULE, "sha256_file", fake_sha)
    manifest, class_map, image_root = MODULE.find_final_v1_authority(tmp_path)
    assert manifest == (real / "audit" / "final_manifest.csv").resolve()
    assert class_map == (real / "audit" / "class_to_idx.json").resolve()
    assert image_root == (real / "dataset").resolve()


def test_notebook_pins_complete_overlay_and_isolated_runtime() -> None:
    notebook_path = SCRIPT.parents[1] / "kaggle" / "CropCop_Prompt1_R07_Foundation_QA_Locked.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell.get("source", [])), f"notebook-cell-{index}", "exec")
    code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert 'RESEARCH_BASE_COMMIT = "a4f9d6e3d417b024ecbfbf4247cd847881935236"' in code
    assert 'RESEARCH_REF = "main"' not in code
    assert 'RUNTIME_ROOT = Path("/kaggle/working/cropcop_prompt1_venv")' in code
    assert "RUNTIME_PYTHON" in code
    assert '"venv", "--without-pip"' in code
    assert '"--target", str(RUNTIME_SITE), "pip"' in code
    assert '[sys.executable, "-m", "pip", "install", "--no-cache-dir"' not in code
    assert '"wrapt==1.17.3"' in code
    assert 'subprocess.run([str(RUNTIME_PYTHON), "-c", probe], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)' in code
    assert "preflight_extra = () if ENABLE_SINGLE_ROUTE_PTQ else ('--skip-ptq',)" in code
    assert "pip\", \"check" in code
    assert "status\",\"--porcelain=v1" in code
    for name, source in {
        "_EXECUTOR_B64": SCRIPT,
        "_AUTHORITY_BINDING_B64": SCRIPT.parents[1] / "track_c_r07" / "shared_foundation" / "TRACKBC_R07_AUTHORITY_BINDING_v1.json",
        "_RECONSTRUCTION_CONTRACT_B64": SCRIPT.parents[1] / "track_c_r07" / "shared_foundation" / "R07_DEPLOYMENT_RECONSTRUCTION_CONTRACT_v1.json",
        "_RUNBOOK_B64": SCRIPT.parents[1] / "kaggle" / "PROMPT1_KAGGLE_RUNBOOK.md",
    }.items():
        match = re.search(rf'{name} = "([A-Za-z0-9+/=]+)"', code)
        assert match, name
        assert gzip.decompress(base64.b64decode(match.group(1))) == source.read_bytes()
