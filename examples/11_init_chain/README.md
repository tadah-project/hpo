# 11 -- chained INIT

Multiple init strategies in one chain.

What this shows:
- Several `STRATEGY` lines inside `INIT { ... }` stack into a single
  chain of seeds.
- `WARM` entries automatically move to the front of the chain so
  warmstart-cache priming happens before random seeds are evaluated.
- The chain is consumed by the outer optimiser; multi-seed consumers
  like `DLIB:GFS` use every entry, single-seed consumers like
  `DLIB:BFGS` use just the front.

The local `run.sh` bootstraps `seed_pot.tadah` the same way as 10.
