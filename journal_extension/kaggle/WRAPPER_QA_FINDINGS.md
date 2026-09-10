# Wrapper QA Findings

Pre-CI adversarial review found and remediated:

1. durable locator placeholder accidentally evaluated as a notebook variable instead of remaining the runner's `{run_id_lower}` template;
2. Smoke phases were initially over-provisioned with Kaggle API credentials;
3. attached Smoke/G1/G2 evidence discovery was initially filename-oriented and was hardened to exact source/status/seal identity;
4. repository notebook copies differed from the downloadable pack by one trailing newline; the downloadable pack and byte manifest were normalized to the repository bytes.

No finding was waived. Exact-head CI is required after all remediations.
