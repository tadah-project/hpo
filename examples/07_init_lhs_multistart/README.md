# 07 -- LHS init + MULTISTART strategy

K starts, each refined with its own local run.

What this shows:
- `STRATEGY LHS K` -- Latin-hypercube sampling; K seeds spread one
  per stratum per dimension.
- `STRATEGY { TYPE MULTISTART, N_STARTS, BUDGET_PER_START }` -- run
  the local optimiser `N_STARTS` times, each with its own
  `BUDGET_PER_START` eval budget. Outer `MAXEVAL` is a hard cap on
  the total.

Use when the loss surface is multimodal and a single local run risks
getting stuck. Cost: ~ `N_STARTS x BUDGET_PER_START`.
