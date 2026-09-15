from __future__ import annotations

import master_account_driver_v11 as base
from master_control_v12 import build_control_k1

base.build_control_k1 = build_control_k1


def main() -> int:
    print("TRACKA_V12_RUNTIME_V12 control-schema compatibility bridge active.", flush=True)
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
