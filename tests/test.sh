#!/bin/bash

# Verifier entrypoint. This runs in the verifier container, not the agent's — it sees
# only the paths declared in `artifacts`. All tooling is baked into tests/Dockerfile;
# nothing is installed or fetched here.

set -uo pipefail

mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt

# The graded predictions must come out of the agent's own crate, so discard anything on
# disk and rebuild from the collected sources. The dependency tree is already compiled
# into this image, so only the agent's own sources recompile. A build or run failure is
# not fatal here — the tests report the missing file.
rm -f /app/predictions.csv
(cd /app/predictor && cargo run --release --offline)
echo "predictor exit status: $?"

python -m pytest -o cache_dir=/tmp/pytest_cache \
  --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA
exit_code=$?
if [ "$exit_code" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi

# Harbor grades from reward.txt, never from this script's exit status.
exit 0
