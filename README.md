# Battery Fade: Capacity and Replacement Scheduling in Rust

A battery-leasing company wants a capacity estimate and a replacement cycle for every pack still in its delivery fleet. The agent gets a small teardown cohort of bench-tested packs (`train.csv`, with measured `capacity_pct`), the fleet still running (`test_features.csv`, no capacity), and a data dictionary. It has to write a Rust program in a pre-made Cargo project (`smartcore` and `csv` available, no network) that writes `/app/predictions.csv` with `pack_id`, `capacity_pct` (percent of nameplate, 0-100) and `replace_cycle` (the cycle count, counted from new, at which the pack should come off the van before it drops below 80% of nameplate).

## What makes it hard

The task has three separate traps, and getting one right does not fix the others.

- **Contaminated columns that validation cannot catch.** Two columns share a name across cohorts but come from different instruments. On teardown rows `soh_pct` has been overwritten with the bench result, and `dc_resistance_mohm` is a 25 C bench reading. On field rows both are in-service BMS estimates that were never temperature-corrected. Every training row is contaminated in the same way, so any honest holdout built from `train.csv` ranks the leaky routes *above* the clean one: 0.78 R2 with resistance in the model, 0.97 for shipping `soh_pct` as the answer, against 0.67 for the clean model. Scored on the field cohort, those routes get 0.50 and -1.22, and their schedules run 33.5% and 54.6% late. You can only see the trap by reading the dictionary: the field packs are more degraded but their resistances read *lower*.
- **Extrapolation.** Training packs have 150-600 cycles and field packs have 850-1700. Tree and neighbour models flatten outside the training range (gradient boosting -0.08, random forest -0.19, KNN -0.64 on the shipped instance). Fade is diffusion-limited (capacity tracks sqrt(cycles)) and the duty drivers scale the fade rate, so a log-rate linear model reaches 0.66, close to the 0.67 ceiling set by pack-to-pack noise.
- **Heteroscedastic replacement margin.** How much packs vary from each other depends on duty (heat and fast-charging), so the safety margin before the 80% crossing has to be a function of the same columns. A single pooled margin gets the fleet-wide late rate in band (7.6%) but lands at 2.2% in the gentlest third. The obvious spread estimator is also biased at low cycle counts.

The reference scores R2 0.664 with a fleet-wide late rate of 9.43% and roughly 8.6 / 9.7 / 10.1% across the three duty thirds. The thresholds were calibrated over 14 generator seeds with `solution/calibrate.py`. Two quite different correct routes (a parametric maximum-likelihood spread fit and an empirical quantile per duty bin) pass on every seed. Four wrong routes fail on every seed, including two tuned on the withheld truth.

## How it's graded

The verifier runs in a separate, offline container. It collects only the crate sources and `/app/data`, drops them onto a pre-compiled skeleton, and runs `cargo run --release --offline` itself, so a hand-written `predictions.csv` never reaches the grader. `tests/test_outputs.py` runs nine checks:

- the shipped CSVs are unaltered (sha256)
- exact schema, and one row per field pack
- capacities numeric and within 0-100; replacement cycles finite and positive
- neither column is a constant
- R2 >= 0.55 against withheld bench results
- the share of packs already below 80% when their `replace_cycle` comes up is within 7-13.5% fleet-wide, and within 6-14.5% in each third of the fleet by fast-charge share

Anti-cheat: the agent's crate is built and run as an unprivileged user (`setpriv`, no-new-privs, groups cleared). The answer key is root-only at `/opt/golden`. `/tests` is closed to that user, and if it is still readable the run scores 0. The build's process group and all build-user processes are killed before pytest runs. pytest runs with `-P` so a planted `/app/pytest.py` cannot shadow it.

## Repository layout

- `instruction.md` - the task prompt given to the agent
- `task.toml` - task metadata, timeouts, resource limits, and the difficulty, solution and verification write-ups
- `environment/` - agent container: `Dockerfile`, `data/` (train/test CSVs and data dictionary), `predictor/` (Cargo project with a stub `main.rs`)
- `solution/` - `solve.sh`, reference `main.rs`, `generate_data.py` (regenerates the CSVs byte for byte), `calibrate.py` (multi-seed threshold calibration)
- `tests/` - verifier `Dockerfile`, `test.sh`, `test_outputs.py`, pre-built crate `skeleton/`, and `golden/` labels

## Running it

Requires Docker and Harbor (`uv tool install harbor` or `pip install harbor`). From the repo root:

```
harbor run -p . -a oracle
```

runs the reference solution through the verifier (should score reward 1). To try an agent:

```
harbor run -p . -a <agent> -m <model>
```

for example `-a claude-code`.

---

Task design, data generators, reference solution and verifier by Audrey Pope.
