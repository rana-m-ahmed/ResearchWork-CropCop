from __future__ import annotations

import ast
import os
import runpy
import subprocess
import sys
from pathlib import Path

from tracka_v12_kaggle_operator_v3 import OperatorError

SEALER_RELATIVE_PATH = Path("journal_extension/scripts/seal_tracka_v12_g1a.py")
EXPECTED_FROZEN_SEALER_GIT_BLOB = "90a918fcdf14130b44b9c4ec24b0b60b1706bb2c"
EXPECTED_TORCHVISION_VERSION = "0.27.1"
MISSING_GLOBAL = "TORCHVISION_VERSION"


def _bound_module_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for part in ast.walk(target):
                    if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store):
                        names.add(part.id)
    return names


def validate_frozen_sealer_contract(repo: Path) -> dict[str, str]:
    repo = Path(repo).resolve()
    script = repo / SEALER_RELATIVE_PATH
    if not script.is_file():
        raise OperatorError(f"frozen G1A sealer missing: {script}")

    blob = subprocess.check_output(["git", "hash-object", str(script)], cwd=repo, text=True).strip()
    if blob != EXPECTED_FROZEN_SEALER_GIT_BLOB:
        raise OperatorError(
            "G1A compatibility shim refuses unexpected sealer bytes: "
            f"expected git blob {EXPECTED_FROZEN_SEALER_GIT_BLOB}, got {blob}"
        )

    source = script.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script))
    bound = _bound_module_names(tree)
    loads = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    if MISSING_GLOBAL not in loads:
        raise OperatorError("frozen G1A sealer no longer references TORCHVISION_VERSION; shim must be retired")
    if MISSING_GLOBAL in bound:
        raise OperatorError("frozen G1A sealer now binds TORCHVISION_VERSION itself; shim must be retired")

    src_root = repo / "journal_extension" / "src"
    sys.path.insert(0, str(src_root))
    try:
        from cropcop_je.secondary import TORCHVISION_VERSION
    finally:
        try:
            sys.path.remove(str(src_root))
        except ValueError:
            pass
    if TORCHVISION_VERSION != EXPECTED_TORCHVISION_VERSION:
        raise OperatorError(
            "frozen secondary TorchVision version drift: "
            f"expected {EXPECTED_TORCHVISION_VERSION}, got {TORCHVISION_VERSION}"
        )

    return {
        "sealer_git_blob": blob,
        "injected_global": MISSING_GLOBAL,
        "injected_value": TORCHVISION_VERSION,
    }


def run_frozen_g1a_sealer(repo: Path, sealer_args: list[str]) -> int:
    repo = Path(repo).resolve()
    contract = validate_frozen_sealer_contract(repo)
    script = repo / SEALER_RELATIVE_PATH
    scripts_root = str(script.parent)
    src_root = str(repo / "journal_extension" / "src")

    sys.path.insert(0, scripts_root)
    sys.path.insert(0, src_root)
    old_argv = list(sys.argv)
    try:
        from cropcop_je.secondary import TORCHVISION_VERSION

        print(
            "G1A_SEALER_COMPAT_V6 "
            f"blob={contract['sealer_git_blob']} "
            f"inject={contract['injected_global']}={contract['injected_value']}",
            flush=True,
        )
        sys.argv = [str(script), *sealer_args]
        try:
            runpy.run_path(
                str(script),
                run_name="__main__",
                init_globals={MISSING_GLOBAL: TORCHVISION_VERSION},
            )
        except SystemExit as exc:
            code = exc.code
            if code is None:
                return 0
            if isinstance(code, int):
                return int(code)
            print(str(code), file=sys.stderr)
            return 1
        return 0
    finally:
        sys.argv = old_argv
        for value in (src_root, scripts_root):
            try:
                sys.path.remove(value)
            except ValueError:
                pass


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit("usage: master_g1a_sealer_compat_v6.py REPO_ROOT -- <sealer args...>")
    repo = Path(sys.argv[1]).resolve()
    rest = sys.argv[2:]
    if not rest or rest[0] != "--":
        raise SystemExit("compatibility runner requires '--' before sealer arguments")
    return run_frozen_g1a_sealer(repo, rest[1:])


if __name__ == "__main__":
    raise SystemExit(main())
