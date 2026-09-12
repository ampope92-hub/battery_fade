Battery pack export, 2026-06-30. One row per pack, out of the fleet asset database.

train.csv is the teardown cohort: packs pulled from service and bench-tested, so the
measured capacity is on the row. test_features.csv is the fleet still in the field,
same columns, no teardown, capacity unknown.

Fields
------

pack_id             pack serial
cycles              full charge/discharge cycles logged by the BMS
chemistry           cell chemistry line
avg_temp_c          average cell temperature over the pack's service life (BMS)
depth_of_discharge  mean fraction of capacity drawn per cycle (BMS)
fast_charge_pct     fraction of charges done on the fast charger (BMS)
cell_count          cells in the pack
pack_age_days       days since the pack entered service
dc_resistance_mohm  pack DC internal resistance, milliohms. On teardown rows this is the
                    bench rig figure, taken in the same session as the capacity test with
                    the pack settled at 25 C. On field rows it is the last resistance the
                    BMS logged in service, computed off whatever load step it last saw, at
                    whatever temperature the pack was at.
soh_pct             state of health, percent of nameplate. The BMS carries a running estimate
                    from coulomb counting. On teardown rows the maintenance system overwrites
                    that field with the bench result once the pack has been through the rig.
capacity_pct        bench-measured capacity, percent of nameplate (teardown only)

Cohorts
-------

The teardown cohort is what came off the line during the first replacement wave, so it
skews toward packs pulled early: warranty swaps, fleet refreshes, units returned with
unrelated faults. The field export is everything still running. Bench capacity comes
off a calibrated rig, good to about a percent.

Both cohorts are pulled into one schema by the asset database, which fills each column from
whichever system holds it for that pack — the bench records for units that have been through
the rig, the telemetry stream for units still in service.

Replacement policy
------------------

A pack is retired when it drops below 80% of nameplate — that is the number in the lease
contracts, and it is what "end of service life" means everywhere in this export. Packs
are pulled on a cycle count, not a date, because the vans run wildly different duty.
