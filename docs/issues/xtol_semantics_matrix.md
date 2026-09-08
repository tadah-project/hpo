# XTOL_REL / XTOL_ABS mean different things per backend

Verified against nlopt 2.10 source (phase-0 investigation):

| backend | XTOL_REL | XTOL_ABS |
|---|---|---|
| NELDERMEAD / SBPLX / MMA / CCSAQ / CRS / ISRES / Luksan LD_* / SLSQP / PRAXIS(atanh space) | whole-vector L1 relative to ‖x‖ (`util/stop.c:98-108`); SBPLX adds a per-dim guard `|xstep_j|·0.25 ≤ max(xtol_abs_j, xtol_rel·|x_j|)` | per-dim absolute |
| BOBYQA / COBYLA / NEWUOA* | `rhoend = xtol_rel · rhobeg` — relative to the initial STEP, not |x| | `rhoend = max(…, xtol_abs_j/|s_j|)` |
| DIRECT / DIRECT_L | relative to the box width (`cdirect.c:382-390`) | per-dim / width |
| ESCH | ignored | ignored |
| Luksan LD_* | + absolute gradient test `tolg = 1e-8` (`PARAM tolg`) in internal units | |
| CERES | `parameter_tolerance` relative to ‖x‖₂ of the whole vector | n/a |
| DLIB:BFGS | n/a (objective_delta / gradient norm) | n/a |
| ANNEAL family | rejected | rejected |

Under `OPTIM_SCALE AUTO` (unit box) XTOL_ABS is a fraction of every width and the |x|-relative tests are
comparable across dims but vacuous next to a lower bound (phase 3 HINT). Under NONE they mix units.

**Proposal:** document this table in `hpo.rst` (optimizer.rst "Scaling, steps and tolerances by
backend"), and consider deriving a per-dim `xtol_abs` vector from `NOISE_CALIBRATION` (see
`noise_aware_step_floor.md`).
