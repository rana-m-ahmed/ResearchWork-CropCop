"""Fail-closed validation for the public POCO M3 blocker certificate."""
from __future__ import annotations

import json
from pathlib import Path


def require(value: bool, message: str) -> None:
    if not value:
        raise SystemExit(f"TRACKC_SECONDARY_EVIDENCE_INVALID: {message}")


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "results" / "public" / "device2_poco_m3"
    preflight = json.loads((root / "DEVICE_PREFLIGHT.json").read_text(encoding="utf-8"))
    blocker = json.loads((root / "RUNTIME_BLOCKER.json").read_text(encoding="utf-8"))
    expected = "2e0c54a1b5bb7c0018d0159a642e49e4f4bd79fcc1f940e215a40734a2695bb8"
    require(preflight["status"] == "PASS_PRE_FORWARD_PREFLIGHT", "preflight status")
    require("arm64-v8a" in preflight["device"]["abis"], "ARM64 compatibility")
    require(preflight["artifact"]["sha256"] == expected, "artifact identity")
    require(preflight["inputs"]["tensor_count"] == 256 and preflight["inputs"]["raw_count"] == 256, "input counts")
    require(blocker["status"] == "BLOCKED_TRACKC_SECONDARY_RUNTIME_SIGILL", "blocker status")
    require(blocker["attempted_artifact_sha256"] == expected, "attempted artifact identity")
    require(blocker["failure"]["signal"] == "SIGILL", "native failure signal")
    require(blocker["failure"]["produced_tensor_fidelity_rows"] == 0, "no false fidelity result")
    print("PASS_TRACKC_SECONDARY_DEVICE_BLOCKER_VALIDATION")


if __name__ == "__main__":
    main()
