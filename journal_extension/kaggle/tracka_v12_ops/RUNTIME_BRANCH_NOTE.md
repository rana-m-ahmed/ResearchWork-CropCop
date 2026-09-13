Immutable runtime target: `ops-tracka-kaggle-master-runtime-v3-fix1-5654f35` at `5654f35fa52c9b6ae6c28f062969c9ebd65af3fa`.

Distribution notebooks must clone that branch and verify the exact SHA before invoking `master_account_driver.py`.

This runtime supersedes `f606311...` for new execution because it moves exact V1/principal-G1 input validation ahead of dependency installation and reports mounted candidate hashes on mismatch. Frozen science remains unchanged at `9a72e9466a9a3e7429e0e36a028edac662f83146`.
