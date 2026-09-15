from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCIENCE_SHA = "05ac7084a6be2fecd9c370477340ee0c8c4769bc"
RUNTIME_SHA = "2f127cbb61d752eec22c3bfd3ecd527a3a81c283"
RUNTIME_BRANCH = "ops-tracka-kaggle-master-runtime-v8-2f127cb"
DISTRIBUTION_BRANCH = "ops-tracka-kaggle-master-v8-distribution-20260915"
AUTHORITY = "EAAI-JE-SDL-v2.1-QA"
CODE_ATTESTATION_SHA256 = "cd8a23ca6a5ebdba18e8466d8578d2646723c349b9428b8d8890aac7c62ab187"
LOCK_RUNTIME_ATTESTATION_SHA256 = "046814249b119f6d520df58095bf951b13e35fcc09f995599c36202f8b0bed7f"
RUNTIME_QA_RUN_ID = 34936176259
NOTEBOOK_NAMES = {
    "K1": "TRACKA_V12_MASTER_K1.ipynb",
    "K2": "TRACKA_V12_MASTER_K2.ipynb",
    "K3": "TRACKA_V12_MASTER_K3.ipynb",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def notebook(account_id: str) -> dict:
    is_k1 = account_id == "K1"
    intro = [
        f"# Track-A v1.2 — MASTER {account_id} (v8)\n",
        "\n",
        f"Exact qualified science: `{SCIENCE_SHA}`  \n",
        f"Immutable operator runtime: `{RUNTIME_SHA}`  \n",
        f"Runtime branch: `{RUNTIME_BRANCH}`\n",
        "\n",
        "Use Kaggle **T4 x2**, **Internet ON**, and **Save Version → Save & Run All / Batch**. Do not edit scientific settings.\n",
        "Required Kaggle secrets: `KAGGLE_USERNAME`, `KAGGLE_KEY`, `CROPCOP_GITHUB_TOKEN`, and `CROPCOP_EXPECTED_KAGGLE_USERNAME`. The expected-username value must name this account and is an independent lane-binding check.\n",
    ]
    if is_k1:
        intro.append("K1 additionally requires the historical principal G1 bundle mounted read-only. Start K1 first; launch K2/K3 only after K1 publishes canonical G1A READY and grants their accounts Can view access to the private G1A dataset.\n")
    else:
        intro.append("Launch this worker only after K1 has produced canonical G1A READY and this Kaggle account has Can view access to K1's private G1A dataset.\n")

    setup = [
        "from pathlib import Path\n",
        "import os\n",
        "\n",
        "V1 = Path('/kaggle/input/datasets/ranamuhammadahmed6/cropcop-finalized-v8-11-2026-1/CropCop_Final_v1')\n",
        "if V1.is_dir():\n",
        "    preferred = {\n",
        "        'CROPCOP_MANIFEST': V1 / 'audit' / 'final_manifest.csv',\n",
        "        'CROPCOP_CLASS_MAP': V1 / 'audit' / 'class_to_idx.json',\n",
        "        'CROPCOP_IMAGE_ROOT': V1 / 'dataset',\n",
        "    }\n",
        "    for name, path in preferred.items():\n",
        "        if path.exists():\n",
        "            os.environ.setdefault(name, str(path))\n",
    ]
    if is_k1:
        setup += [
            "\n",
            "PRINCIPAL_G1 = Path('/kaggle/input/datasets/ranamuhammadahmed6/cropcop-g1-sealed/G1_PACKAGE')\n",
            "if PRINCIPAL_G1.is_dir():\n",
            "    os.environ.setdefault('CROPCOP_PRINCIPAL_G1', str(PRINCIPAL_G1))\n",
        ]
    setup += [
        "\n",
        "expected = os.environ.get('CROPCOP_EXPECTED_KAGGLE_USERNAME', '').strip()\n",
        "if not expected:\n",
        "    try:\n",
        "        from kaggle_secrets import UserSecretsClient\n",
        "        expected = UserSecretsClient().get_secret('CROPCOP_EXPECTED_KAGGLE_USERNAME').strip()\n",
        "    except Exception as exc:\n",
        "        raise RuntimeError('Add Kaggle secret CROPCOP_EXPECTED_KAGGLE_USERNAME with this account username before running v8.') from exc\n",
        "if not expected:\n",
        "    raise RuntimeError('CROPCOP_EXPECTED_KAGGLE_USERNAME is empty.')\n",
        "os.environ['CROPCOP_EXPECTED_KAGGLE_USERNAME'] = expected\n",
        f"print('Expected account binding loaded for {account_id}.')\n",
    ]

    checkout = [
        "from pathlib import Path\n",
        "import shutil, subprocess, sys\n",
        "\n",
        f"OPS_RUNTIME_SHA = '{RUNTIME_SHA}'\n",
        f"OPS_RUNTIME_BRANCH = '{RUNTIME_BRANCH}'\n",
        "OPS_ROOT = Path('/kaggle/working/cropcop-tracka-master-runtime-v8')\n",
        "if OPS_ROOT.exists():\n",
        "    shutil.rmtree(OPS_ROOT)\n",
        "subprocess.run([\n",
        "    'git', 'clone', '--quiet', '--depth', '1', '--branch', OPS_RUNTIME_BRANCH,\n",
        "    'https://github.com/rana-m-ahmed/ResearchWork-CropCop.git', str(OPS_ROOT)\n",
        "], check=True)\n",
        "head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=OPS_ROOT, text=True).strip()\n",
        "dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=OPS_ROOT, text=True).strip()\n",
        "if head != OPS_RUNTIME_SHA:\n",
        "    raise RuntimeError(f'operator runtime SHA mismatch: expected {OPS_RUNTIME_SHA}, got {head}')\n",
        "if dirty:\n",
        "    raise RuntimeError(f'operator runtime checkout is dirty: {dirty}')\n",
        "print('Pinned Track-A v8 runtime:', head)\n",
    ]

    launch = [
        "guard = OPS_ROOT / 'journal_extension/kaggle/tracka_v12_ops/master_launch_guard_v8.py'\n",
        f"cp = subprocess.run([sys.executable, '-u', str(guard), '{account_id}'], cwd=guard.parent)\n",
        "if cp.returncode == 0:\n",
        f"    print('{account_id} v8 master: TERMINAL PASS for this account queue.')\n",
        "elif cp.returncode == 2:\n",
        f"    print('{account_id} v8 master: controlled dependency/session/publication continuation. Save & Run All again in a fresh Batch session.')\n",
        "else:\n",
        f"    raise RuntimeError('{account_id} v8 master requires investigation; rc=' + str(cp.returncode))\n",
    ]

    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": intro},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": setup},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": checkout},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": launch},
            {"cell_type": "markdown", "metadata": {}, "source": ["Recovery rule: use this same notebook in a fresh Batch session only for an explicitly controlled continuation. Validation or scientific failures remain fail-closed.\n"]},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.13"},
            "cropcop_operator": {
                "schema_version": "4.0",
                "account_id": account_id,
                "science_sha": SCIENCE_SHA,
                "operator_runtime_sha": RUNTIME_SHA,
                "operator_runtime_branch": RUNTIME_BRANCH,
                "launcher": "master_launch_guard_v8.py",
                "driver": "master_account_driver_v8.py",
                "authority": AUTHORITY,
                "expected_account_binding_required": True,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    notebook_hashes = {}
    for account_id, name in NOTEBOOK_NAMES.items():
        data = (json.dumps(notebook(account_id), indent=2, sort_keys=True) + "\n").encode("utf-8")
        (ROOT / name).write_bytes(data)
        notebook_hashes[name] = sha256_bytes(data)

    distribution = {
        "schema_version": "4.0",
        "status": "LOCKED_DISTRIBUTION",
        "authority_id": AUTHORITY,
        "science_source_sha": SCIENCE_SHA,
        "operator_runtime_sha": RUNTIME_SHA,
        "operator_runtime_branch": RUNTIME_BRANCH,
        "distribution_branch": DISTRIBUTION_BRANCH,
        "launcher": "master_launch_guard_v8.py",
        "driver": "master_account_driver_v8.py",
        "normal_operator_notebooks": list(NOTEBOOK_NAMES.values()),
        "normal_operator_notebook_count": 3,
        "notebook_sha256": notebook_hashes,
        "account_binding": {
            "method": "account-local CROPCOP_EXPECTED_KAGGLE_USERNAME value independently compared to authenticated KAGGLE_USERNAME",
            "required": True,
            "k1_known_proven_username": "ranamuhammadahmed6",
            "k2_k3_must_be_configured_on_their_respective_accounts": True,
        },
        "release_integrity_before_expensive_work": True,
        "exact_head_code_attestation_sha256": CODE_ATTESTATION_SHA256,
        "exact_head_lock_runtime_attestation_sha256": LOCK_RUNTIME_ATTESTATION_SHA256,
        "generation_aware_kaggle_durability": True,
        "session_aware_dependency_deadline": True,
        "g1a_failed_handoff_fail_fast": True,
        "g2a_account_failed_status_fail_fast": True,
        "per_account_singleton_guard": True,
        "duplicate_master_invocation_runs_stages": False,
        "scientific_runner_git_publication_enabled": False,
        "scientific_evidence_parent_published": True,
        "protected_test_open_before_track_a_closure": False,
        "external_prediction_open_before_track_a_closure": False,
        "batch_required": True,
        "internet_required": True,
        "accelerator": "T4 x2",
    }
    write_json(ROOT / "MASTER_NOTEBOOK_DISTRIBUTION.json", distribution)

    freeze = {
        "schema_version": "2.0",
        "status": "PASS",
        "science_source_sha": SCIENCE_SHA,
        "runtime_candidate_sha": RUNTIME_SHA,
        "runtime_branch": RUNTIME_BRANCH,
        "qa_check": "v8-operator-qa",
        "qa_conclusion": "success",
        "qa_workflow_run_id": RUNTIME_QA_RUN_ID,
        "code_attestation_file_sha256": CODE_ATTESTATION_SHA256,
        "lock_runtime_attestation_file_sha256": LOCK_RUNTIME_ATTESTATION_SHA256,
        "note": "Operator implementation/CI qualification only. Real Kaggle G1A/G2A and scientific-result PASS require actual account execution evidence.",
    }
    write_json(ROOT / "MASTER_RUNTIME_FREEZE.json", freeze)

    contract = {
        "schema_version": "4.0",
        "status": "LOCKED_MASTER_OPERATOR_CONTRACT",
        "authority_id": AUTHORITY,
        "scientific_source_sha": SCIENCE_SHA,
        "operator_runtime_sha": RUNTIME_SHA,
        "operator_runtime_branch": RUNTIME_BRANCH,
        "launcher": "master_launch_guard_v8.py",
        "driver": "master_account_driver_v8.py",
        "notebooks": list(NOTEBOOK_NAMES.values()),
        "kaggle": {
            "canonical_g1a_private_dataset_count": 1,
            "g2a_distinct_private_dataset_count": 5,
            "scientific_distinct_private_dataset_count": 11,
            "required_accelerator": "T4 x2",
            "required_account_identity_binding": True,
            "g1a_worker_access_required_before_k2_k3_launch": True,
        },
        "durability": {
            "kaggle_generation_must_advance_before_sync_success": True,
            "generation_marker_roundtrip_required": True,
            "checkpoint_index_hash_bound": True,
            "checkpoint_scientific_identity_validation_preserved": True,
        },
        "coordination": {
            "fixed_30_minute_cross_account_wait": False,
            "session_deadline_shared_across_dependency_waits": True,
            "failure_statuses_are_source_runtime_g1a_bound": True,
        },
        "publication": {
            "v8_source_ancestry_required": True,
            "idempotent_roundtrip_required": True,
            "scientific_runner_git_publication_enabled": False,
            "scientific_evidence_published_by_master_parent": True,
        },
        "rules": {
            "scientific_source_may_follow_operator_head": False,
            "notebooks_may_follow_distribution_head": False,
            "protected_test_open_before_track_a_closure": False,
            "external_prediction_open_before_track_a_closure": False,
            "science_requires_durability_bound_go": True,
            "ddp_allowed": False,
            "dataparallel_allowed": False,
            "fsdp_allowed": False,
        },
    }
    write_json(ROOT / "OPERATOR_CONTRACT.json", contract)

    (ROOT / ".runtime-freeze-marker").write_text(
        f"runtime_sha={RUNTIME_SHA}\nruntime_branch={RUNTIME_BRANCH}\nscience_sha={SCIENCE_SHA}\n",
        encoding="utf-8",
    )
    (ROOT / "RUNTIME_BRANCH_NOTE.md").write_text(
        "# Track-A v1.2 v8 Runtime Freeze\n\n"
        f"Immutable runtime target: `{RUNTIME_BRANCH}` at `{RUNTIME_SHA}`.\n\n"
        f"Qualified science source: `{SCIENCE_SHA}`.\n\n"
        "The three distributed notebooks must clone that branch and verify the exact SHA before invoking `master_launch_guard_v8.py`. The runtime packages exact-head source attestations, performs release-integrity validation before expensive work, uses generation-aware Kaggle durability, source/runtime/G1A-bound failure signaling, a shared session dependency deadline, and parent-only Git evidence publication. Protected Track-A test/external surfaces remain closed.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
