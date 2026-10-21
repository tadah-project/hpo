# NLopt LN_NEWUOA_BOUND stalls for 10-60 s per step once its model is not finite (refused)

**Status:** refused since 1.6.0. `tadah hpo` stops before it loads any data, naming the phase and
`NLOPT:LN_BOBYQA` (until 2026-10-12 also `NLOPT:LN_NEWUOA` where the search may leave the box; that one is
refused now too, `nlopt_newuoa_ignores_bounds.md`): at `ENDOPTIMIZER` for the
`STRATEGY`-level optimiser (`PLAIN`, `MULTISTART` or the `PRESTAGE` refinement, whether the `STRATEGY` block,
a `REFINE` block or the `OPTIMIZER` block itself names it), at `ENDEXPLORE` for the `PRESTAGE` exploration
and at `ENDINNER` for an `INNER` block. The refusal is
the entry of the refusal table in `src/optimizer_caps.cpp` (`backend_refusal()`, `optimizer_caps.h`). It is
tied to **NLopt v2.10.0** (`CMakeLists.txt` fetches `GIT_TAG v2.10.0`; offline builds use `deps/nlopt`):
before an NLopt upgrade lifts it, run the repro below on the new version. The version is recorded once, as
`tadah::hpo::kNloptTestedVersion` (`include/tadah/hpo/nlopt_tested_version.h`), the version of every NLOPT
entry of the refusal table and of the NLopt behaviour HPO pins: a build against another NLopt
(`FETCHCONTENT_SOURCE_DIR_NLOPT`, a new `GIT_TAG`) fails the test case "the NLopt HPO is built against is the
version its NLopt refusals were measured on" (`tests/test_nlopt_version.cpp`), which names this document.
Change the constant only after the repros.

## What happened

NLopt's `LN_NEWUOA_BOUND` is NLopt's own bound-constrained variant of Powell's NEWUOA, not Powell's code. To
keep each step inside the box it plans every step with a separate NLopt run of `LD_MMA`. That nested run has
no time limit and does not see a stop request, and once a non-finite value (inf or NaN) is in NEWUOA's
quadratic model, each nested run goes on to its evaluation caps: about 10^8 evaluations of the model,
10-60 s, with no evaluation of the loss. HPO only gets control back when the loss is evaluated, so
Ctrl-C, SIGTERM, `MAXTIME` and `RESTART` wait that long, and a run that hits it repeatedly looks hung.

A non-finite value reaches the model in three ways:

1. **Any non-finite objective value -- every stop HPO makes through the objective from a run's (2n+1)-th
   evaluation on.** When the objective throws, NLopt's own C++ wrapper (`nlopt.hpp`, `myfunc`) and HPO's
   (`nlopt_adapter.h`) call `force_stop()` and return `HUGE_VAL`: that is NLopt's convention for a forced
   stop, and HPO keeps it. HPO's objective throws for a signal (SIGINT, SIGTERM, SIGUSR2:
   `core::ShutdownRequested`), for the `RESTART` stagnation stop (`StagnationStop`) and for an error that
   ends the run (e.g. the best potential cannot be saved). The first 2n+1 evaluations of a run (n
   dimensions) are NEWUOA's starting points: after each of the first 2n it goes straight to the next point,
   where it tests the stop flag, so a stop there returns at once. After the (2n+1)-th it folds the values,
   the `HUGE_VAL` included, into its model and plans its first step before it tests the flag again, and so
   after every later evaluation: every such stop from the (2n+1)-th evaluation on costs one stalled
   subproblem, on any loss, smooth or not, with or without `XTOL`. A failed evaluation scored `FAILSCORE` is
   finite and does not do this.
2. **A flat loss without an `XTOL` floor.** With no `XTOL_REL`/`XTOL_ABS` NEWUOA's final trust radius
   `rhoend` is 0, so on a loss that does not change over the trust region (a flat direction, or a box where
   every evaluation fails and scores `FAILSCORE`) the radius shrinks by 10x per step without limit until
   NEWUOA's `ZMAT` overflows (about rho = 1e-143).
3. **Near convergence.** MMA's step control is not scale-invariant (its `rho` is dimensionful), so once the
   model changes little over the trust region the nested MMA returns steps of about 1e-8 of the radius,
   NEWUOA stalls and its `BMAT` overflows. This happens with `XTOL_REL 1e-12`, `XTOL_ABS 1e-14` or `1e-20`
   too, so no `XTOL` threshold avoids it.

`LN_BOBYQA` (Powell's own bound-constrained successor of NEWUOA) and `LN_NEWUOA` use no MMA: in the same
cases they return at once after a forced stop and never stalled. `LN_COBYLA`, `LN_SBPLX`, `LN_NELDERMEAD`
and `LN_PRAXIS` also return at once after the same forced stop (case F below).

## Mechanism (NLopt 2.10.0 sources, `deps/nlopt/src`)

- `api/optimize.c:857` calls `newuoa(n, 2n+1, x, lb, ub, ...)` for `LN_NEWUOA_BOUND` (`:849` passes no
  bounds for `LN_NEWUOA`).
- `algs/newuoa/newuoa.c:158-211` (`trsapp_`, the trust-region subproblem) and `:1222-1262` (`biglag_`) each
  create an `NLOPT_LD_MMA` run (`:185`, `:1252`) with `maxeval 1000` and an x tolerance of `1e-7*delta`
  (`1e-5*delta` for `biglag_`). Each MMA iteration solves its dual with another MMA run of up to 100000
  evaluations (`api/optimize.c:817`, `dual_maxeval`).
- The nested runs have their own stop structures: no `MAXTIME`, no shared forced-stop flag. `newuob_` tests
  the forced stop, `MAXEVAL` and `MAXTIME` only at `L310` (`newuoa.c:2119-2124`), immediately before an
  objective call. Its starting points (`L50`-`L70`, `:1731-1832`) go from each evaluation straight to the
  next `L310` (`:1777`) while `nf < npt` (`:1829`), npt = 2n+1; after the npt-th the first step is planned
  (`L100`, `trsapp_`, `:1856`).
- The model update: `diff = f - fopt - vquad` (`newuoa.c:2163`), then `gq[i] += diff*bmat[...]` (`:2313`).
  An `f` of `HUGE_VAL`, or an inf in `BMAT`, makes the model gradient `GQ` inf or NaN. `newuob_` never tests
  the model for finiteness (only `s`, `hcol` and the `tau`/`alpha` products, `:677`, `:972-973`, `:1148`).
- MMA on such a model: `inner_done = dd.gval >= fcur` (`algs/mma/mma.c:302`) and `if (fcur > dd.gval)`
  (`:363`) are false for NaN, so `rho` never changes and the inner loop ends only at its cap; every dual
  solve runs its 100000 evaluations. About 10^8 model evaluations per subproblem; `trsapp_` then returns a
  zero step, NEWUOA treats it as a short step and calls the subproblem again.
- Route 2: `rhoend = xtol_rel*rhobeg`, floored by `xtol_abs` (`newuoa.c:1627-1632`); HPO sets neither
  unless the block writes `XTOL_*` (`nlopt_adapter.h`), and the rho loop `L490` (`:2452-2464`) divides rho
  by 10 with no floor.
- Route 3: `MMA_RHOMIN 1e-5` (`mma.c:41`) and `rho_init` 1.0 (`optimize.c:800`) are in absolute units; when
  |g|*delta is much smaller than rho, MMA's first move is below its own x tolerance and it returns
  `XTOL_REACHED` after 2 evaluations.

NLopt's documentation (`doc/docs/NLopt_Algorithms.md`) describes the MMA-based variant and calls it
"largely superseded by BOBYQA". The vendored `NEWS.md`/`ChangeLog` mention no fix; the upstream tracker was
not consulted (offline workspace).

## Measurements

NLopt level, the repro below, built against this build's `libnlopt.a` (NLopt 2.10.0), 2026-10-07:

| case | `LN_NEWUOA_BOUND` | `LN_BOBYQA` | `LN_NEWUOA` |
|---|---|---|---|
| A: constant objective, n = 2, no `XTOL` | stall: no objective call for > 5 s after 300 calls | returns after 159 calls | returns after 313 calls |
| B: 4-D sphere, `XTOL_REL 1e-12` | stall after 348 calls | f = 0 after 47 calls | f = 0 after 48 calls |
| F: forced stop at call 30, `HUGE_VAL` returned, n = 2 | 10.84 s from the stop to the return | 0.00 s | 0.00 s |
| F with the finite loss returned instead (`FINITE=1`) | 0.00 s | | |
| F, n = 2 (2n+1 = 5), stop at call 1, 2, 3 or 4 (`STOPAT`) | 0.00 s | | |
| F, n = 2, stop at call 5 or 6 | 19.13 s, 9.48 s | 0.00 s at call 5 | 0.00 s at call 5 |
| F, n = 3 (2n+1 = 7, `N=3`), stop at call 6 or 7 | 0.00 s, 21.43 s | | |

`LN_COBYLA`, `LN_SBPLX`, `LN_NELDERMEAD` and `LN_PRAXIS` in case F: 0.00 s each.

`tadah hpo` built before the refusal, on the Ta test data (`tests/data/ta/common`, its `config.tadah`
as the `-c`), `OPTIM RCUT2B (1) 4.0 6.0`, `OPTIM SGRIDMB (1) 0.1 0.4`, `MAXEVAL 300`, a `RESTART` block with
`AFTER_STAGNATION 20` and `MAX 2`: each of the three `StagnationStop fired` journal lines was followed by
20-21 s without an evaluation before the next `RESTART` (or the end), and the run took 60.8 s for 99
evaluations. With `LN_BOBYQA` the same configuration took 0.57 s for 169 evaluations (it converged,
roundoff-limited, before a stagnation stop). With the refusal the same configuration stops in 0.15 s:
`HPOEngine error: Error on line 17: the STRATEGY-level optimiser: NLOPT:LN_NEWUOA_BOUND is refused: ...`.

The investigation of 2026-10-07 (scratch runs with a `tadah` built before the refusal, some alongside other
jobs, so the seconds are indicative):

- forced stop at NLopt level: 9.5-24.5 s for n = 2, 4 and 8; one subproblem on a NaN model alone takes
  15 s (n = 2), 21 s (n = 8) and 59 s (n = 32);
- `tadah hpo`, the two OPTIM dimensions above, a smooth loss: Ctrl-C at about evaluation 70 of a run that
  otherwise takes 0.9 s was honoured 10.2 s later (`LN_BOBYQA`: 0.007 s);
- `tadah hpo` on a flat loss (an energy-only fit searching `FWEIGHT` and `SWEIGHT`, two parameters it did not
  depend on; both are retired keys of Tadah! 1.6.0, which refuses such a search) with `MAXEVAL 400` and no
  `XTOL`: 300 evaluations in 5 s, then 20 evaluations in 590 s; `MAXTIME 20` ended after 31 s, and
  `MAXTIME 15` after 40.5 s;
- a survey of 720 bounded cases without `XTOL` (12 objectives, n = 2-6): 57 stalled with
  `LN_NEWUOA_BOUND`, none with `LN_BOBYQA` or `LN_NEWUOA`. With `XTOL_REL 1e-8` none stalled, but some runs
  reported SUCCESS at stalled points (a 4-D scaled Rosenbrock at f = 3.70, where `LN_BOBYQA` reaches 5e-18).

## Why the algorithm is refused

- Requiring an `XTOL` is not enough: route 1 does not depend on it, and route 3 stalls with `XTOL_REL 1e-12`
  and `XTOL_ABS 1e-20`.
- HPO's `HUGE_VAL` after a forced stop is NLopt's own convention (its C++ wrapper does the same). A finite
  value instead would hand NLopt a loss that was never computed, which could become the minimum NLopt
  reports, and would not help routes 2 and 3.
- A fix inside NLopt would need a scale-aware subproblem solver, a finiteness guard on the model, and
  `MAXTIME` and the forced stop passed into the nested runs, applied as a patch to both the fetched and the
  vendored source -- research work on a variant NLopt itself calls superseded.
- `LN_BOBYQA` is the replacement: the same model family, bound-constrained by design, and it neither stalled
  nor delayed a stop in any case above.

## Moving a study to LN_BOBYQA

The trajectory changes, so an old study is not reproduced bit for bit: `STEP` becomes a per-dimension
coordinate rescaling instead of one radius equal to the smallest step; BOBYQA moves a start closer to a
bound than its initial step onto the bound or one step away from it (when `STEP` is set, HPO's step plan
shrinks the step of such a dimension, `initial_step.h`); `STEP 0.5` is capped just below
(`width >= 2*step`). `LN_NEWUOA` keeps one radius but ignores the box, and is refused since 2026-10-12
(`nlopt_newuoa_ignores_bounds.md`).

## Not changed here

- `LN_NEWUOA` stayed registered here. It never stalled, but it ignores the box, needs n >= 2 (NLopt returns
  `INVALID_ARGS` otherwise; HPO did not check it at startup), and in 10 of the 720 survey cases (a noisy
  sphere, n = 2) it passed NaN coordinates to the objective and ended with a NaN minimum. It is refused
  since 2026-10-12: see `nlopt_newuoa_ignores_bounds.md`.
- A cooperative stop had no escalation: a second Ctrl-C did nothing more (`CORE/src/signal_handler.cpp`
  only set a flag), so any long computation inside a third-party optimiser between objective calls could
  only be ended with SIGKILL. Since 2026-10-12 a second SIGINT, or a second SIGTERM, ends the process at
  once (`CORE/src/signal_handler.cpp`, `SignalHandler`).

## Repro (re-run before an NLopt upgrade lifts the refusal)

Build against the NLopt to be tested (here the one HPO builds):

```bash
g++ -O2 -std=c++17 -pthread -I deps/nlopt/src/api newuoa_bound_stall.cpp \
    build_tadah/_deps/nlopt-build/libnlopt.a -o newuoa_bound_stall
for c in A B F; do ./newuoa_bound_stall $c; echo "exit $?"; done   # LN_NEWUOA_BOUND
for c in A B F; do ./newuoa_bound_stall $c LN_BOBYQA; done          # the replacement
for k in 4 5; do STOPAT=$k ./newuoa_bound_stall F; done              # n = 2: from call 2n+1 = 5
```

With NLopt 2.10.0 cases A and B print `STALL: no objective call for > 5 s after 300 calls` (348 for B) and
case F prints about 10 s from the forced stop to the return; each exits 3. `STOPAT=4` returns at once and
`STOPAT=5` takes 10-25 s: a stop stalls from the (2n+1)-th call on. The refusal can be lifted only when A, B
and F exit 0 for `LN_NEWUOA_BOUND` on the new version, F with `STOPAT=5` too (then also re-check the
survey's other objectives), and the entry in `kRefused` (`src/optimizer_caps.cpp`) and its tests
(`tests/test_optimizer_caps_step.cpp`, `tests/test_startup_validation_strategy.cpp`) are removed together.

`newuoa_bound_stall.cpp`:

```cpp
// Re-test for HPO's refusal of NLOPT:LN_NEWUOA_BOUND (docs/issues/
// nlopt_newuoa_bound_stall.md). Exit 3: NLopt made no objective call for
// more than 5 s (A, B), or took more than 5 s to return after a forced stop
// (F). Exit 0: the algorithm returned without such a stall.
//
//   A  constant objective, n = 2, box [0,1]^2, start on the lower corner,
//      NLopt's default step, MAXEVAL 400, no XTOL (rhoend = 0)
//   B  sphere sum (x_i - 0.3)^2, n = 4, start 0.1, STEP 0.25,
//      XTOL_REL 1e-12, MAXEVAL 400
//   F  weighted sphere, n = 2, start 0.1, MAXEVAL 400; at call 30 the
//      objective calls nlopt_force_stop and returns HUGE_VAL, as NLopt's own
//      C++ wrapper (nlopt.hpp) and HPO's (nlopt_adapter.h) do when the
//      objective throws; prints the seconds from the stop to the return.
//      In the environment: FINITE=1 returns the finite loss instead,
//      STOPAT=k stops at call k, N=n searches n dimensions (NEWUOA's
//      first 2n+1 calls are its starting points).
//
// Usage: newuoa_bound_stall A|B|F [ALGO]   (ALGO default LN_NEWUOA_BOUND)
#include <nlopt.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

static std::atomic<long> calls{0};
static std::atomic<double> last{0};
static nlopt_opt O;
static long stop_at = 30;
static double t_stop = 0;
static double now() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double constant(unsigned, const double*, double*, void*) {
    ++calls; last = now(); return 1.0;
}
static double sphere(unsigned n, const double* x, double*, void*) {
    ++calls; last = now();
    double f = 0; for (unsigned i = 0; i < n; ++i) f += (x[i] - 0.3) * (x[i] - 0.3);
    return f;
}
static double forced(unsigned n, const double* x, double*, void*) {
    double f = 0; for (unsigned i = 0; i < n; ++i) f += (1 + i) * (x[i] - 0.3) * (x[i] - 0.3);
    if (++calls < stop_at) return f;
    if (calls == stop_at) t_stop = now();
    nlopt_force_stop(O);
    return std::getenv("FINITE") ? f : HUGE_VAL;
}

int main(int argc, char** argv) {
    const char mode = argc > 1 ? argv[1][0] : 'A';
    const char* algo = argc > 2 ? argv[2] : "LN_NEWUOA_BOUND";
    if (const char* k = std::getenv("STOPAT")) stop_at = std::atol(k);
    const unsigned n = mode == 'B' ? 4
                     : mode == 'F' && std::getenv("N") ? std::atoi(std::getenv("N")) : 2;
    O = nlopt_create(nlopt_algorithm_from_string(algo), n);
    std::vector<double> lb(n, 0.0), ub(n, 1.0), x(n, mode == 'A' ? 0.0 : 0.1), dx(n, 0.25);
    nlopt_set_lower_bounds(O, lb.data());
    nlopt_set_upper_bounds(O, ub.data());
    nlopt_set_min_objective(O, mode == 'A' ? constant : mode == 'B' ? sphere : forced, nullptr);
    nlopt_set_maxeval(O, 400);
    if (mode == 'B') { nlopt_set_xtol_rel(O, 1e-12); nlopt_set_initial_step(O, dx.data()); }
    last = now();
    if (mode != 'F') std::thread([] {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (now() - last.load() > 5.0) {
                std::printf("STALL: no objective call for > 5 s after %ld calls\n", calls.load());
                std::fflush(stdout); std::_Exit(3);
            }
        }
    }).detach();
    double f = 0;
    const nlopt_result r = nlopt_optimize(O, x.data(), &f);
    const double after = mode == 'F' ? now() - t_stop : 0.0;
    std::printf("%s %c: result %d after %ld calls, f = %g", algo, mode, (int)r, calls.load(), f);
    if (mode == 'F') std::printf(", %.2f s from the forced stop to the return", after);
    std::printf("\n");
    return after > 5.0 ? 3 : 0;
}
```
