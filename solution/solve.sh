#!/usr/bin/env bash
set -euo pipefail

# Reference solution for battery_fade.
#
# Two separate things have to be right here.
#
# The cycle ranges don't overlap: 50-600 in the teardown cohort, 850-1700 out in
# the field. Trees, forests and KNN all partition the feature space, so past the
# edge of the training range they just repeat the nearest capacity they saw and
# go flat. Only a model that extrapolates is any use.
#
# And the fade is diffusion-limited, so capacity goes with sqrt(cycles) rather
# than cycles. Fit on raw cycles and you pick up the average slope over 50-600,
# much steeper than the real slope out at 1200, so the old packs keep coming out
# too low. Feed it sqrt(cycles) instead.
#
# The stress drivers scale the fade rather than shifting it (a hot pack on deep
# discharge loses more capacity per cycle), so they go in multiplied by
# sqrt(cycles) too. Without those interactions this lands about 0.75 and misses.

# The model itself is solution/main.rs, next to this script — Rust is not a shell
# language and 100 lines of it inside a heredoc is neither reviewable nor compilable
# on its own. Drop it in and let cargo build it.
install -m 644 "$(dirname "$0")/main.rs" /app/predictor/src/main.rs

cd /app/predictor
cargo run --release --offline
