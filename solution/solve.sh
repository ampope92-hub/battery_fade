#!/usr/bin/env bash
set -euo pipefail

# Reference solution for battery_fade.
#
# The capacity column is the easier half. Teardown packs have 150-600 cycles on them
# and the field packs 850-1700, so anything that partitions the feature space — trees,
# forests, KNN — runs out of training data at the edge and flattens. Fade is
# diffusion-limited too, so capacity tracks sqrt(cycles), and the duty drivers multiply
# the fade rate rather than shifting it. Put those together and log(rate) is linear in
# the drivers, which is what main.rs actually fits.
#
# replace_cycle is the half that needs a second model. Two packs with identical spec
# sheets do not fade identically: cell imbalance and hot spots make a hard-charged pack
# scatter around its expected rate far more than a gently used one does. So the margin a
# pack needs before its 80% crossing is not one number — it depends on the same duty
# columns the rate does. main.rs fits the SPREAD as its own function of duty by maximum
# likelihood and schedules each pack off the pessimistic end of its own distribution.
# A single fleet-wide safety factor can be tuned until the fleet-wide count looks right;
# it just puts the margin in the wrong places, generous on the gentle packs and thin on
# the ones that need it.

# The model itself is solution/main.rs, next to this script — Rust is not a shell
# language and 200 lines of it inside a heredoc is neither reviewable nor compilable
# on its own. Drop it in and let cargo build it.
install -m 644 "$(dirname "$0")/main.rs" /app/predictor/src/main.rs

cd /app/predictor
cargo run --release --offline
