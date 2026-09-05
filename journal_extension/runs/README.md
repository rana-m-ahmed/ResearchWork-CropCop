# Runs

Scientific jobs write one small JSON evidence record per run. Model checkpoints, raw predictions,
restricted manifests, and other large/private artifacts remain outside public Git. The JSON record
stores hashes and durable locators instead.

No run is terminal until its small evidence bundle survives outside the ephemeral compute session.
