from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import run_tracka_v12_xai as frozen_xai

FROZEN_XAI_GIT_BLOB_SHA1 = "9f5edd2efdf3608ddbdfa78978417a5e9cd1f9f5"


def cli_value(flag: str) -> str:
    try:
        index = sys.argv.index(flag)
    except ValueError as exc:
        raise RuntimeError(f"missing required wrapper argument: {flag}") from exc
    if index + 1 >= len(sys.argv):
        raise RuntimeError(f"missing value for wrapper argument: {flag}")
    return sys.argv[index + 1]


def verify_frozen_xai_executor(repo: Path) -> None:
    script = repo / "journal_extension/scripts/run_tracka_v12_xai.py"
    if not script.is_file():
        raise RuntimeError(f"frozen XAI executor missing: {script}")
    observed = subprocess.check_output(
        ["git", "-C", str(repo), "hash-object", str(script)],
        text=True,
    ).strip()
    if observed != FROZEN_XAI_GIT_BLOB_SHA1:
        raise RuntimeError(
            "frozen XAI executor blob mismatch: "
            f"expected={FROZEN_XAI_GIT_BLOB_SHA1}, observed={observed}"
        )


def guarded_panel_save(original_save, panel_root: Path):
    root = panel_root.resolve()

    def _save(image, fp, *args, **kwargs):
        if isinstance(fp, (str, os.PathLike)):
            path = Path(fp)
            requested_png = str(kwargs.get("format", "")).upper() == "PNG" or path.suffix.lower() == ".png"
            if requested_png:
                resolved = path.resolve()
                try:
                    resolved.relative_to(root)
                except ValueError as exc:
                    raise RuntimeError(
                        f"XAI PNG save escaped canonical qualitative-panel root: {resolved}"
                    ) from exc
                resolved.parent.mkdir(parents=True, exist_ok=True)
                fp = resolved
        return original_save(image, fp, *args, **kwargs)

    return _save


def main() -> int:
    repo = Path(cli_value("--repo-root")).resolve()
    output = Path(cli_value("--output-dir")).resolve()
    verify_frozen_xai_executor(repo)

    from PIL import Image

    panel_root = output / "private_xai" / "qualitative_panel"
    original_save = Image.Image.save
    Image.Image.save = guarded_panel_save(original_save, panel_root)
    try:
        return int(frozen_xai.main())
    finally:
        Image.Image.save = original_save


if __name__ == "__main__":
    raise SystemExit(main())
