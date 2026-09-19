from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file, sha256_json
from cropcop_je.persistence_v8 import build_store_v8
from cropcop_je.tracka_v12_recovery import (
    TerminalRecoveryError,
    account_result,
    build_recovered_terminal_record,
    load_selected_checkpoint_evidence,
    validate_recovered_terminal_record,
)


def load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TerminalRecoveryError(f"JSON object required: {path}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--account-report", required=True)
    ap.add_argument("--checkpoint-root", required=True)
    ap.add_argument("--durable-store-locator", default="")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    account_report_path = Path(args.account_report).resolve()
    report = load_json(account_report_path)
    row = account_result(report, args.experiment_id)
    if row["run_id"] != args.run_id:
        raise SystemExit("requested run_id differs from terminal public account report")

    restored = False
    if not (checkpoint_root / "checkpoint_index.json").is_file():
        if not args.durable_store_locator:
            raise SystemExit("checkpoint index absent and no v8 durable locator supplied")
        store = build_store_v8("kaggle-dataset", args.durable_store_locator)
        if store.restore(checkpoint_root, run_id=args.run_id) is not True:
            raise SystemExit("v8 durable selected-checkpoint restore failed")
        restored = True

    checkpoint = load_selected_checkpoint_evidence(
        repo_root=repo,
        checkpoint_root=checkpoint_root,
        experiment_id=args.experiment_id,
        run_id=args.run_id,
        expected_selected_sha256=row["selected_checkpoint_sha256"],
    )
    record = build_recovered_terminal_record(
        experiment_id=args.experiment_id,
        account_report=report,
        checkpoint_evidence=checkpoint,
        account_report_sha256=sha256_file(account_report_path),
        durable_locator=args.durable_store_locator or None,
    )
    errors = validate_recovered_terminal_record(record)
    if errors:
        raise SystemExit("recovered terminal record failed validation: " + "; ".join(errors))

    record_path = output / "RECOVERED_TERMINAL_RUN_RECORD.json"
    atomic_write_json(record_path, record)
    certificate = {
        "schema_version": "1.0",
        "status": "PASS",
        "recovery_kind": "track_a_v12_terminal_metadata_recovery",
        "experiment_id": args.experiment_id,
        "run_id": args.run_id,
        "durable_restore_performed": restored,
        "durable_store_locator": args.durable_store_locator or None,
        "public_account_report_sha256": sha256_file(account_report_path),
        "checkpoint_index_sha256": checkpoint["checkpoint_index_sha256"],
        "selected_checkpoint_sha256": checkpoint["selected_checkpoint_sha256"],
        "checkpoint_identity_sha256": checkpoint["identity_sha256"],
        "recovered_run_record_sha256": sha256_file(record_path),
        "recovered_run_record_content_sha256": sha256_json(record),
        "scientific_training_reperformed": False,
        "protected_surface_opened": False,
        "note": (
            "Recovery uses only the already-terminal public account report and exact selected-checkpoint "
            "identity/selection state. It does not retrain, adapt, retune, or access V1-test/external/Track-C results."
        ),
    }
    certificate["certificate_sha256"] = sha256_json(certificate)
    certificate_path = output / "TERMINAL_RECOVERY_CERTIFICATE.json"
    atomic_write_json(certificate_path, certificate)
    print(json.dumps(certificate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
