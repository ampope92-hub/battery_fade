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
# the shipped export read-only. /logs is left alone — Harbor owns it and the reward is
# written there last, after every agent process is dead. /tests is closed off below; the
# graded labels are read from the root-only copy baked into this image (see tests/Dockerfile).
chown -R runner:runner /app/predictor
chmod -R a-w /app/data

# Everything below that touches agent-authored code runs through this helper: the build user,
# with supplementary groups cleared and no-new-privileges set, so an inherited group or a
# setuid binary cannot widen what that code can reach.
as_runner() {
  setpriv --clear-groups --no-new-privs --reuid runner --regid runner "$@"
}

# The shipped tests/ directory also lands here at runtime, so the verifier's own code and a
# plaintext copy of the answer key sit under /tests just as agent-authored code is about to
# run. Close the whole directory to the build user before anything untrusted starts: root
# still reads it, `runner` cannot even traverse it. If it is still reachable the run is
# abandoned with reward 0 rather than graded — a reachable /tests is one the agent can read
# the answers out of, or whose test file it can rewrite before pytest reads it. The graded
# labels are read from the root-only copy at /opt/golden either way.
chmod 700 /tests 2>/dev/null || true
if as_runner bash -c 'test -r /tests/test_outputs.py || test -r /tests/golden/test_labels.csv' \
     2>/dev/null; then
  echo "FATAL: /tests is still reachable by the unprivileged build user; the verifier's code"
  echo "and answer key are exposed to agent-authored code, so this run cannot be graded."
  echo 0 > /logs/verifier/reward.txt
  exit 0
fi
echo "verifier files closed to the unprivileged build user"

rm -f /app/predictions.csv
# Harbor uploads the agent's sources with whatever mtime they carried in its container,
# which can predate this image's own layers. Cargo decides freshness by mtime, so an older
# source silently counts as "already built" — stamp the collected sources to now so the
# rebuild below is unconditional. Belt and braces with the `cargo clean -p` in tests/Dockerfile.
find /app/predictor/src -type f -exec touch {} +

# Run the crate in its own session, so every process it spawns shares one process group that
# can be torn down as a unit. A bare `runuser` leaves a double-forked child reparented to init
# and still alive during grading, where it could rewrite reward.txt after the reward is
# computed. The timeout bounds the build well inside the verifier's own timeout_sec.
setsid timeout -k 30 1200 \
  setpriv --clear-groups --no-new-privs --reuid runner --regid runner \
  bash -c 'cd /app/predictor && cargo run --release --offline' &
BUILD_PID=$!
# /proc rather than ps: procps is not in this image. Fields after "comm)" are state, ppid, pgrp.
BUILD_PGID="$(sed 's/.*) //' /proc/$BUILD_PID/stat 2>/dev/null | awk '{print $3}')"
wait "$BUILD_PID"
echo "predictor exit status: $?"

# Nothing the agent started may be alive while pytest runs or while the reward is written.
# Kill the build's whole process group, then sweep anything the build user still owns —
# `kill -9 -1` as `runner` reaches exactly that user's processes and nothing of root's.
if [ -n "${BUILD_PGID:-}" ]; then
  kill -TERM "-$BUILD_PGID" 2>/dev/null || true
  kill -KILL "-$BUILD_PGID" 2>/dev/null || true
fi
as_runner bash -c 'kill -9 -1' 2>/dev/null || true

# -P keeps the working directory (/app, per the Dockerfile's WORKDIR) off sys.path. The
# crate runs as `runner` and writes into /app, so without this it can drop an /app/pytest.py
# holding sys.exit(0); `python -m pytest` would import that no-op, exit 0, and bypass the
# whole suite for reward 1. The test file and cache are addressed absolutely either way.
python -P -m pytest -o cache_dir=/tmp/pytest_cache \
  --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA
exit_code=$?
if [ "$exit_code" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi

# Harbor grades from reward.txt, never from this script's exit status.
exit 0
