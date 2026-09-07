from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.environment import capture_environment, software_stack_identity, validate_locked_core
from cropcop_je.hashing import sha256_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    env = capture_environment()
    env["locked_core_drift"] = validate_locked_core(env)
    env["software_stack_sha256"] = sha256_json(software_stack_identity(env))
    atomic_write_json(args.output, env)
    print(json.dumps(env, indent=2, sort_keys=True))
    return 0 if not env["locked_core_drift"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
