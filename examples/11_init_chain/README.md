# 11 -- chained INIT

Multiple init strategies in one chain.

What this shows:
- Several `STRATEGY` lines inside `INIT { ... }` stack into a single
  chain of seeds.
- The chain runs in the exact order it is written -- no entry is
  reordered. List `WARM` first when you want the warm point at iter 0
  (and its warmstart cache primed before the other seeds).
- The chain is consumed by the outer optimiser; multi-seed consumers
  like `DLIB:GFS` use every entry, single-seed consumers like
  `DLIB:BFGS` use just the front.

The local `run.sh` bootstraps `seed_pot.tadah` the same way as 10.
