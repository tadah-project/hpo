# 20 -- Corana adaptive SA (ANNEAL + CORANA ON)

The serial annealer's adaptive variant (Corana et al., ACM TOMS 13:262,
1987; also used by Brommer 2003), companion to example 15's classic
variant.

What this shows:
- `PARAM CORANA ON` -- one-dim-at-a-time proposals with a per-dim step
  vector `V[i]` that grows/shrinks with the per-dim accept ratio
  (0.4-0.6 deadband, gain `C`). A rung costs `NT * NS * D` evals.
- `V[i]` is always clamped to `[V_FLOOR_FRAC * width, width]`.
  `V_FLOOR_FRAC` accepts `[0, 1]`; zero disables the lower floor. The
  journal reports how many dimensions are at each clamp.
- `RESET_TO_BEST` defaults ON under CORANA: after every cooling step
  the chain snaps back to the best-so-far point (step 7 of Corana et
  al.'s original algorithm; basin memory against FAILSCORE-cliff
  wander).
- `BOUNDARY_MODE 0` keeps TADAH's exact-reflection default;
  `BOUNDARY_MODE 1` selects the paper-compatible redraw rule. A 4096-draw
  guard turns a malformed proposal kernel into a hard error instead of
  looping forever.
- Auto-T0: probes mirror the proposal kernel (one-dim uniform moves around
  the seed). The calibration uses the median uphill finite delta, or the
  median absolute finite delta when none is uphill; FAILSCORE probes are
  excluded.
- `REANNEAL 1`: when the schedule cools out with budget left, the run
  warm-restarts from the best point at a colder start temperature only while the
  next scaled start can execute two rungs before falling below `T_MIN`.

Run:

    bash ../common/run.sh
