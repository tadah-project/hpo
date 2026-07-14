# 20 -- Corana adaptive SA (ANNEAL + CORANA 1)

The serial annealer's adaptive variant (Corana et al., ACM TOMS 13:262,
1987; Brommer 2003), companion to example 15's classic variant.

What this shows:
- `PARAM CORANA 1` -- one-dim-at-a-time proposals with a per-dim step
  vector `V[i]` that grows/shrinks with the per-dim accept ratio
  (0.4-0.6 deadband, gain `C`). A rung costs `NT * NS * D` evals.
- `RESET_TO_BEST` defaults ON under CORANA: after every cooling step
  the chain snaps back to the best-so-far point (basin memory against
  FAILSCORE-cliff wander).
- Auto-T0: probes mirror the proposal kernel (one-dim moves around the
  seed), so the measured dE scale matches what the chain will actually
  see; FAILSCORE probes are excluded from the statistics.
- `REANNEAL 1`: when the schedule cools out with budget left, the run
  warm-restarts from the best point at a colder start temperature.

Run:

    bash ../common/run.sh
