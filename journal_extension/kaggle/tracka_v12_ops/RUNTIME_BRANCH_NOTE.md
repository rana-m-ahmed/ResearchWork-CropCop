# Track-A v1.2 v8.1 Runtime Freeze

Immutable runtime target: `ops-tracka-kaggle-master-runtime-v8r1-f8aea6c-20260915` at `31d6224bd4b87c02ac3f4985d11d0a5f7a2b615d`.

Qualified science source: `f8aea6c2b481f8436653d4c6504f406948a98082`.

The three distributed notebooks must clone that branch and verify the exact SHA before invoking `master_launch_guard_v8.py`. The runtime packages exact-head source attestations, performs release-integrity validation before expensive work, uses generation-aware Kaggle durability, source/runtime/G1A-bound failure signaling, a shared session dependency deadline, and parent-only Git evidence publication. Protected Track-A test/external surfaces remain closed.
