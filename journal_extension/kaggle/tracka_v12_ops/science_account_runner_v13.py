from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PRISTINE_ENV = "CROPCOP_V13_PRISTINE_EXPERIMENTS"
SCIENCE_REPO_ENV = "CROPCOP_V13_SCIENCE_REPO"


def _load_pristine_experiments() -> set[str]:
    raw = str(os.environ.get(PRISTINE_ENV, "") or "").strip()
    if not raw:
        return set()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{PRISTINE_ENV} is not valid JSON") from exc
    if not isinstance(payload, list) or any(not isinstance(value, str) or not value for value in payload):
        raise RuntimeError(f"{PRISTINE_ENV} must be a JSON list of non-empty experiment IDs")
    if len(payload) != len(set(payload)):
        raise RuntimeError(f"{PRISTINE_ENV} contains duplicate experiment IDs")
    return set(payload)


def main() -> int:
    repo_raw = str(os.environ.get(SCIENCE_REPO_ENV, "") or "").strip()
    if not repo_raw:
        raise RuntimeError(f"missing required {SCIENCE_REPO_ENV}")
    repo = Path(repo_raw).resolve()
    for path in (
        repo / "journal_extension" / "src",
        repo / "journal_extension" / "scripts",
        repo / "journal_extension" / "kaggle",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import run_tracka_v12_account as account_base
    import run_tracka_v12_account_v121 as qualified_account
    from cropcop_je.tracka_v12 import EXPERIMENT_SPECS

    pristine = _load_pristine_experiments()
    unknown = pristine.difference(EXPERIMENT_SPECS)
    if unknown:
        raise RuntimeError(f"v13 pristine bootstrap contains unauthorized experiments: {sorted(unknown)}")

    historical_child_command = account_base.child_command

    def child_command_v13(args, *, experiment_id: str, slot_id: str, run_id: str, output_dir: Path, durable_locator: str):
        command = historical_child_command(
            args,
            experiment_id=experiment_id,
            slot_id=slot_id,
            run_id=run_id,
            output_dir=output_dir,
            durable_locator=durable_locator,
        )
        try:
            resume_index = command.index("--resume-mode") + 1
        except ValueError as exc:
            raise RuntimeError("qualified science child command lost --resume-mode") from exc
        if command[resume_index] != "auto":
            raise RuntimeError(f"qualified science child resume mode drifted: {command[resume_index]!r}")
        if experiment_id in pristine:
            command[resume_index] = "never"
            print(
                f"V13_PRISTINE_DURABILITY_BOOTSTRAP experiment={experiment_id} slot={slot_id} "
                f"resume_mode=never durable={durable_locator}",
                flush=True,
            )
        else:
            print(
                f"V13_DURABLE_RESUME experiment={experiment_id} slot={slot_id} "
                f"resume_mode=auto durable={durable_locator}",
                flush=True,
            )
        return command

    account_base.child_command = child_command_v13
    return qualified_account.main()


if __name__ == "__main__":
    raise SystemExit(main())
