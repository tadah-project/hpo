# 18 -- parallel tempering (ANNEAL_PT)

Replica-exchange simulated annealing, TADAH built-in.

What this shows:
- `LIB TADAH, ALGO ANNEAL_PT` -- K replicas run the ANNEAL stepper at a
  fixed temperature ladder `T_COLD .. T_HOT` (slot 0 coldest); every
  `SWAP_EVERY` epochs adjacent pairs attempt a Metropolis state swap,
  so good states migrate cold-ward while hot replicas keep exploring.
- `THREADS 2` -- replicas evaluate concurrently on a persistent worker
  pool over the one thread-safe host. The result is IDENTICAL for any
  THREADS value (per-replica RNG streams; deterministic swap coins).
- `PARAM CORANA 1` -- the adaptive per-dim stepper; its step vector V
  belongs to the temperature slot (swaps move states, not V).
- The journal (`hpo_run.log`) carries one line per epoch plus a final
  per-pair swap-acceptance table -- tune `LADDER_SPAN`/`REPLICAS` for
  20-40% per pair on real workloads.

Run:

    bash ../common/run.sh
