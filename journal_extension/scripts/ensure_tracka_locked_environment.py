from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file


class EnvironmentLockError(RuntimeError):
    pass


def load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EnvironmentLockError(f"JSON object required: {path}")
    return payload


def installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def observed_versions(packages: dict[str, str]) -> dict[str, str | None]:
    return {name: installed_version(name) for name in packages}


def mismatches(expected: dict[str, str], observed: dict[str, str | None]) -> dict[str, dict[str, str | None]]:
    return {
        name: {"expected": expected[name], "observed": observed.get(name)}
        for name in expected
        if observed.get(name) != expected[name]
    }


def fresh_verify(lock_path: Path) -> dict:
    code = r"""
import importlib.metadata, json, platform, sys
lock=json.loads(open(sys.argv[1], encoding='utf-8').read())
observed={name: importlib.metadata.version(name) if _has(name) else None for name in lock['packages']}
result={
    'python_expected': lock['python'],
    'python_observed': platform.python_version(),
    'packages_expected': lock['packages'],
    'packages_observed': observed,
}
result['python_ok'] = result['python_observed'] == result['python_expected']
result['package_mismatches'] = {
    name: {'expected': lock['packages'][name], 'observed': observed.get(name)}
    for name in lock['packages'] if observed.get(name) != lock['packages'][name]
}
print(json.dumps(result, sort_keys=True))
raise SystemExit(0 if result['python_ok'] and not result['package_mismatches'] else 1)
"""
    # Avoid a conditional expression around metadata.version so absent packages
    # are handled deterministically in the child interpreter.
    helper = "def _has(name):\n    try:\n        importlib.metadata.version(name); return True\n    except importlib.metadata.PackageNotFoundError:\n        return False\n"
    code = code.replace("lock=json.loads", helper + "lock=json.loads")
    cp = subprocess.run(
        [sys.executable, "-c", code, str(lock_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    try:
        result = json.loads((cp.stdout or "").strip().splitlines()[-1])
    except Exception as exc:
        raise EnvironmentLockError(
            "fresh environment verifier did not emit valid JSON: "
            + (cp.stdout or cp.stderr or "<no diagnostic>")[-3000:]
        ) from exc
    result["return_code"] = cp.returncode
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument(
        "--lock",
        default="journal_extension/locks/execution_dependency_lock.json",
    )
    ap.add_argument("--repair", action="store_true")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    lock_path = Path(args.lock)
    if not lock_path.is_absolute():
        lock_path = repo / lock_path
    if not lock_path.is_file():
        raise SystemExit(f"execution dependency lock missing: {lock_path}")

    lock = load_json(lock_path)
    expected_python = str(lock.get("python", ""))
    expected_packages = dict(lock.get("packages") or {})
    if not expected_python or not expected_packages:
        raise SystemExit("execution dependency lock is incomplete")

    python_observed = platform.python_version()
    if python_observed != expected_python:
        raise SystemExit(
            f"Python runtime drift cannot be repaired in-process: "
            f"expected={expected_python}, observed={python_observed}"
        )

    before = observed_versions(expected_packages)
    drift_before = mismatches(expected_packages, before)
    repaired = []
    if drift_before:
        if not args.repair:
            raise SystemExit("frozen dependency environment drift: " + json.dumps(drift_before, sort_keys=True))
        exact = [f"{name}=={expected_packages[name]}" for name in sorted(drift_before)]
        cp = subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "--disable-pip-version-check", "--no-input", "--upgrade",
                *exact,
            ],
            cwd=repo,
            check=False,
            timeout=3600,
        )
        if cp.returncode != 0:
            raise SystemExit(f"frozen dependency repair failed rc={cp.returncode}")
        repaired = sorted(drift_before)

    verified = fresh_verify(lock_path)
    if verified.get("return_code") != 0:
        raise SystemExit(
            "fresh-interpreter frozen dependency verification failed: "
            + json.dumps(verified, sort_keys=True)
        )

    result = {
        "schema_version": "1.0",
        "status": "PASS",
        "gate_kind": "track_a_frozen_execution_environment",
        "python": expected_python,
        "packages": expected_packages,
        "dependency_lock_file_sha256": sha256_file(lock_path),
        "dependency_lock_content_sha256": lock.get("dependency_lock_sha256"),
        "drift_before_repair": drift_before,
        "repaired_packages": repaired,
        "fresh_interpreter_verified": True,
    }
    if args.output:
        atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
