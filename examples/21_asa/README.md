# 21 -- Adaptive Simulated Annealing (ASA)

Lester Ingber's ASA-30.53, vendored under `HPO/external/asa`
(BSD-3-Clause; see `ASA-LICENSE` and `TADAH-PATCHES.md` there).

What this shows:

- `LIB ASA, ALGO ASA` -- the vendored adaptive simulated annealer.
  Unlike the fixed-schedule `TADAH:ANNEAL`, ASA anneals a SEPARATE
  temperature per dimension, T_i(k) = T0_i exp(-c_i k^(1/D)), generates
  proposals from a fat-tailed distribution (occasional long jumps at
  low temperature), and periodically REANNEALS: parameter temperatures
  are rescaled by finite-difference sensitivities (`Delta_X` tangents)
  so stiff directions stay hot longer.
- `PARAM` keys keep the ORIGINAL ASA option spellings (case-sensitive):
  - `Limit_Generated` / `Limit_Acceptances` -- ASA's own budget stops.
  - `Temperature_Ratio_Scale` -- THE primary annealing-rate knob
    (`m = -log(TRS)`); raise it toward 1 to slow annealing.
  - `Temperature_Anneal_Scale`, `Cost_Parameter_Scale_Ratio` -- the
    other two schedule constants (`c = -log(TRS) exp(-log(TAS)/D)`).
  - `Acceptance_Frequency_Modulus` -- cadence of the periodic tests and
    of the `ASA gen= acc= best=` journal lines.
  - `Curvature_0 1` -- skip the exit-time curvature array (upstream
    default `0` costs O(dim^2) evaluations at exit; documented TADAH
    default deviation).
  - Per-dimension options accept an indexed form, e.g.
    `PARAM User_Quench_Param_Scale[1] 0.9`. The index is 1-based and is
    the dimension number the run prints (the `dim` column of
    `hpo_search_space.tsv`).
- Stops honoured: `MAXEVAL` (hard cap incl. the `Number_Cost_Samples`
  initial cost-temperature probes), `MAXTIME`, `STOPVAL`, plus ASA's own
  `Limit_*`, `Accepted_To_Generated_Ratio`, `Cost_Precision` x
  `Maximum_Cost_Repeat` (plateau) and temperature-underflow exits.
  `FTOL_REL`/`FTOL_ABS` are rejected with a pointer to `Cost_Precision`.
- Parallelism: `PARAM Gener_Block N` (+ `Gener_Block_Max`) evaluates
  each generated block of N states concurrently on the THREADS-wide
  pool; results are THREADS-invariant, and `Gener_Block 1` (default)
  reproduces canonical serial ASA exactly. Block mode changes the
  sampled sequence vs serial ASA (upstream-documented caveat).
- ASA's own journal goes to `asa_out` next to `hpo_run.log`.
- Single-instance semantics: one `asa()` run per process at a time
  (upstream statics). The RESTART decorator re-runs it sequentially and
  works unchanged; `STRATEGY TYPE MULTISTART` (and a PRESTAGE refinement
  over `TOPK > 1` seeds) rejects ASA, a single-seed global optimiser, at
  startup, before any data is loaded.

Run it:

```bash
TADAH=/path/to/tadah bash ../common/run.sh
```
