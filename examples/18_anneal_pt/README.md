# 18 -- parallel tempering (ANNEAL_PT)

Replica-exchange simulated annealing, TADAH built-in.

What this shows:
- `LIB TADAH, ALGO ANNEAL_PT` -- K replicas run the ANNEAL stepper at a
  fixed temperature ladder `T_COLD .. T_HOT` (slot 1 coldest); every
  `SWAP_EVERY` epochs alternating adjacent pairs attempt a Metropolis state
  swap. A lower-loss state found on the hot side moves cold-ward with
  probability one; reverse exchanges remain possible for detailed balance.
- `THREADS 2` -- the total shared-pool width for replica evaluations,
  PC_LAMMPS fan-out, and MLIP train/predict work. It does not multiply
  `OMP_NUM_THREADS`. Because this example fixes `REPLICAS`, uses a
  deterministic objective, and has no timing-dependent stop, the same
  `SEED` gives the same evaluated set for any `THREADS` value.
- `PARAM CORANA ON` -- the adaptive per-dim stepper; its step vector V
  belongs to the temperature slot (swaps move states, not V). Because the
  adaptation is per dimension, so is the reporting: the journal carries a
  per-slot acceptance census and the full per-`(slot, dimension)` table
  lands in `hpo_corana_acceptance.tsv`, with each slot's adapted sigma next
  to the acceptance it produced. On this 2-dimension example that is 8 rows;
  it is the diagnostic that matters on a search space with hundreds.
- Ladder-scaled initial jitter (`PARAM INIT_JITTER_LADDER`, `AUTO` by
  default from 1.5.0). Slot 1 starts exactly at the seed and the others are
  displaced by `sigma * (T_k/T_HOT)^0.5`, so on this ladder slot 2 draws
  21.5% of the base sigma and slot 4 the full amount -- visible in the
  journal's `initial-jitter factor per slot` table. `OFF` restores the
  pre-1.5.0 placement, where every non-seed slot was displaced equally.
- Temperature alternatives are strict: choose `T_COLD` or `LADDER_SPAN`,
  not both. A static ladder (`RT 1`) does not use `T_MIN`; omit it or set
  `RT < 1` for a cooling PT-to-SA hybrid.
- Exchange decisions resolve equal- and zero-temperature endpoints before
  inverse-temperature arithmetic. At a zero cold endpoint, a lower/equal
  hot-side loss is accepted cold-ward and a higher loss is rejected;
  ordinary finite exchanges are compared in the log domain.
- The journal (`hpo_run.log`) carries one line per epoch plus a final
  cumulative and `SWAP_WINDOW`-epoch per-pair acceptance tables. The often
  cited 20-40% range is a starting heuristic, not a workload-independent
  target; use measured ladder-flow counts as the outcome diagnostic.

Run:

    bash ../common/run.sh

Artifacts worth reading afterwards: `hpo_run.log` (the narrative),
`hpo_corana_acceptance.tsv` (per-dimension acceptance and step) and
`hpo_temperature_ladder.tsv` (ladder history, static here).
