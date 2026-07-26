# Battery pack export — 2026-06-30

One row per pack, pulled from the fleet asset database. `train.csv` is the teardown
cohort: packs we pulled from service and bench-tested, so the measured capacity is
recorded. `test_features.csv` is the fleet still in the field — same columns, no
teardown, capacity unknown.

## Columns

| column | source | description |
|---|---|---|
| `pack_id` | registry | pack serial (field export only) |
| `cycles` | BMS counter | full charge/discharge cycles logged by the battery management system |
| `chemistry` | registry | cell chemistry line |
| `avg_temp_c` | BMS telemetry | average cell temperature over the pack's service life |
| `depth_of_discharge` | BMS telemetry | mean fraction of capacity drawn per cycle |
| `fast_charge_pct` | BMS telemetry | fraction of charges done on the fast charger |
| `cell_count` | registry | cells in the pack |
| `pack_age_days` | registry | days since the pack entered service |
| `capacity_pct` | bench test | measured capacity as a percentage of nameplate (teardown cohort only) |

## How the cohorts were assembled

The teardown cohort is what came off the line during the first replacement wave, so it
skews toward packs pulled early — warranty swaps, fleet refreshes, and units returned
with unrelated faults. The field export is everything still running. Bench capacity is
measured on a calibrated rig and is good to about a percent.
