from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from tracka_v12_kaggle_operator import *  # noqa: F401,F403

OPERATOR_SCHEMA_VERSION = "2.1"


def discover_g1a_bundle(input_root: str | Path = "/kaggle/input", override: str = "") -> Path:
    """Resolve one complete Track-A G1A bundle using the frozen seal schema."""
    roots = [Path(override).resolve()] if override else [
        p.parent.resolve() for p in Path(input_root).rglob("TRACKA_V12_G1A_SEAL.json")
    ]
    matches = []
    for root in roots:
        seal_path = root / "TRACKA_V12_G1A_SEAL.json"
        if not seal_path.is_file() or not (root / "private").is_dir() or not (root / "evidence").is_dir():
            continue
        try:
            payload = load_json(seal_path)
        except Exception:
            continue
        if (
            payload.get("status") == "PASS"
            and payload.get("science_authorized") is False
            and payload.get("source_git_sha") == SCIENCE_SHA
            and len(str(payload.get("g1a_seal_sha256", ""))) == 64
        ):
            matches.append(root)
    matches = sorted({p.resolve() for p in matches})
    if len(matches) != 1:
        raise OperatorError(f"G1A bundle resolution must be unique, found {len(matches)}: {matches}")
    return matches[0]


def validate_private_locators_with_science(
    repo: str | Path,
    mapping: dict[str, str],
    *,
    env: dict[str, str] | None = None,
) -> dict:
    """Run the frozen repository's Kaggle private-durability preflight."""
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
