from __future__ import annotations

import _bootstrap  # noqa: F401
import qualify_tracka_v12_profile as base
from run_tracka_v12_training_v121 import execute, parser

base.execute = execute
base.training_parser = parser
main = base.main


if __name__ == "__main__":
    raise SystemExit(main())
