# Track-A v1.2 v8 Runtime Freeze

Immutable runtime target: `ops-tracka-kaggle-master-runtime-v8-2f127cb` at `2f127cbb61d752eec22c3bfd3ecd527a3a81c283`.

Qualified science source: `05ac7084a6be2fecd9c370477340ee0c8c4769bc`.

The three distributed notebooks must clone that branch and verify the exact SHA before invoking `master_launch_guard_v8.py`. The runtime packages exact-head source attestations, performs release-integrity validation before expensive work, uses generation-aware Kaggle durability, source/runtime/G1A-bound failure signaling, a shared session dependency deadline, and parent-only Git evidence publication. Protected Track-A test/external surfaces remain closed.
