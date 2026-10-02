from __future__ import annotations
import os
import sys
from pathlib import Path

repo = Path(os.environ["CROPCOP_VALIDATION_REPO"]).resolve()
src = repo / "journal_extension" / "src"
if not src.is_dir():
    raise RuntimeError(f"CropCop validation source package missing: {src}")
if str(src) not in sys.path:
    sys.path.insert(0, str(src))
