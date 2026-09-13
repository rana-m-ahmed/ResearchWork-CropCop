from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from tracka_v12_kaggle_operator import *  # noqa: F401,F403

OPERATOR_SCHEMA_VERSION = "2.0"


def validate_private_locators_with_science(
    repo: str | Path,
    mapping: dict[str, str],
    *,
    env: dict[str, str] | None = None,
) -> dict:
    """Run the frozen repository's Kaggle private-durability preflight.

    v2 deliberately serializes the locator mapping exactly once before embedding it
    into the isolated subprocess. This replaces the superseded v1 helper implementation
    that double-encoded the mapping and could turn it into a JSON string.
    """
    env = dict(os.environ if env is None else env)
    payload = json.dumps(mapping, sort_keys=True)
    code = (
        "import json,sys; "
        f"sys.path.insert(0,{str(Path(repo)/'journal_extension'/'src')!r}); "
        "from cropcop_je.persistence import validate_durable_access_plan; "
        f"m=json.loads({payload!r}); "
        "r=validate_durable_access_plan('kaggle-dataset',m); "
        "print(json.dumps(r,sort_keys=True)); "
        "raise SystemExit(0 if r.get('status')=='PASS' else 2)"
    )
    cp = run([sys.executable, "-c", code], env=env, capture=True, check=False)
    if cp.returncode != 0:
        raise OperatorError(f"private durable preflight failed:\n{(cp.stdout or '')[-6000:]}")
    lines = [line for line in (cp.stdout or "").splitlines() if line.strip()]
    if not lines:
        raise OperatorError("private durable preflight produced no JSON result")
    result = json.loads(lines[-1])
    if not isinstance(result, dict) or result.get("status") != "PASS":
        raise OperatorError(f"private durable preflight returned invalid result: {result!r}")
    return result
