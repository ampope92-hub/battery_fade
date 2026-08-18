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
#
# That rebuild executes code the agent wrote, so it runs as the unprivileged `runner`. As
# root it could read the answer key and echo it back, or rewrite test_outputs.py before
# pytest reads it below. Artifacts land owned by root, so hand the crate over first and pin
# the shipped export read-only. Nothing here touches /tests or /logs — Harbor mounts both at
# runtime and modifying them breaks the verifier; the graded labels are read from the
# root-only copy baked into this image instead (see tests/Dockerfile).
chown -R runner:runner /app/predictor
chmod -R a-w /app/data

# The shipped tests/ directory also lands here at runtime, so a plaintext copy of the answer
# key sits at /tests/golden alongside the graded one. Close it to `runner` — best effort,
# because Harbor owns this mount and may present it read-only. Grading never depends on this
# succeeding (the labels are read from /opt/golden), so a failure here must not fail the run;
# it is logged instead, because it is the difference between a closed and an open verifier.
chmod 600 /tests/golden/test_labels.csv 2>/dev/null || true
if runuser -u runner -- test -r /tests/golden/test_labels.csv 2>/dev/null; then
  echo "WARNING: /tests/golden/test_labels.csv is still readable by the unprivileged user"
else
  echo "answer key closed to the unprivileged user"
fi

rm -f /app/predictions.csv
runuser -u runner -- bash -c 'cd /app/predictor && cargo run --release --offline'
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
