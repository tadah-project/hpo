# TADAH patches and build notes for vendored ASA-30.53

Upstream: Adaptive Simulated Annealing (ASA) 30.53, Lester Ingber,
https://www.ingber.com/ , BSD-3-Clause (see `ASA-LICENSE`, which must ship
with this code). Vendored from `SIMULATED_ANNEALING/ASA-30.53/ASA` on
2026-08-19. Upstream development has ended; no re-vendoring is expected.

## Vendored verbatim

`asa.h`, `asa_usr_asa.h`, `ASA-LICENSE`, `ASA-README.txt`, `ASA-NOTES`.
The user-module files (`asa_usr.c`, `asa_usr.h`, `asa_usr_cst.c`,
`asa_opt`, `ASA-Makefile`) are NOT vendored: TADAH provides its own
driver, RNG, and option handling (`src/algorithms_asa.cpp`).

## TADAH files (not upstream)

- `tadah_asa_config.h` — compile-time gate set of the featured build;
  the single source of truth (asa.c is compiled only through wrapper TUs).
- `asa_build.c` — featured-build wrapper TU (config + `#include "asa.c"`).
- `tadah_asa_hook.h` / `tadah_asa_hook.c` — parallel-evaluation hook
  declaration and NULL definition (contract documented in the header).
- `tadah_asa_pristine_config.h` / `asa_pristine.c` — all-algorithmic-
  gates-off reference build with `tadah_asaref_`-prefixed symbols, used
  only by the golden-neutrality unit test.
- `CMakeLists.txt` — `tadah_asa` (featured) and `tadah_asa_pristine`
  (BUILD_TESTING only) static libraries; compiled as C11, `-w`, not
  linked against `tadah::compile_options`.

## Patches to asa.c

All hunks are fenced with `/* TADAH-PATCH begin */ ... /* TADAH-PATCH end */`
(grep for `TADAH-PATCH`). The annealing algorithm itself is untouched.
`asa.c`, like the other upstream files, keeps upstream's CRLF line endings:
edit it without converting them, so that the diff of a patch shows only its
hunks and `git blame` keeps crediting the rest to the vendoring.

1. **Hook include + layout check** (top of file, after `#include "asa.h"`,
   under `#if ASA_PARALLEL`): includes `tadah_asa_hook.h` and
   `_Static_assert`s that `tadah_asa_state_view` mirrors ASA's `STATE`.

2. **Parallel-spawn region** (inside the `ASA_PARALLEL` main loop, between
   Ingber's markers `*** ENTER/EXIT CODE TO SPAWN OFF PARALLEL GENERATED
   STATES ***`, which upstream designates as the user's parallel-spawn
   insertion point): when `tadah_asa_parallel_hook` is non-NULL, cached
   ASA_QUEUE slots are settled serially, the hook evaluates the fresh
   slots of the generated block (on the TADAH thread pool), and
   `cost_function_test` bookkeeping runs serially afterwards; when the
   hook is NULL, the original upstream loop (including its OpenMP pragma)
   runs verbatim in the `else` branch. The hook path deliberately never
   writes `OPTIONS->parallel_id` (upstream writes it from inside the
   OpenMP loop, a data race on the shared `USER_DEFINES`). The hook
   branch also mirrors the live `*current_cost_temperature` into
   `tadah_asa_cost_temp` before each hook call, so the driver can stamp
   the acceptance temperature on journal tags (`Tcost=`) -- a plain
   store with no trajectory effect.

3. **Invalid-state retry-loop bound (upstream bug fix)**: the
   `ASA_PARALLEL` invalid-state retry `do { ... } while
   (valid_state_generated_flag_par_test >= Gener_Block)` loop has no
   in-loop `Limit_Invalid_Generated_States` bound (the serial branch
   exits mid-loop), so a block that stays all-invalid -- e.g. a host
   that fails persistently, or the TADAH stop path flagging states
   invalid after MAXEVAL/MAXTIME -- spins forever. The patched
   condition also stops when the limit is exceeded (the existing check
   right after the loop then raises `TOO_MANY_INVALID_STATES`) or when
   `Immediate_Exit` is pending (so a stop request cannot be starved by
   `Limit_Invalid_Generated_States = -1` in the main loop; patch 6 does
   the same for the sampling loops outside it).

4. **P_TEMP_TOO_SMALL dimension mirror**: upstream records the index of
   the dimension whose temperature underflowed only in an internal
   variable (`*index_exit_v`) and its own journal. The patch (guarded
   `#if ASA_PARALLEL`, i.e. featured build only) mirrors it into
   `tadah_asa_exit_index` (declared in `tadah_asa_hook.h`) at the exit
   site so the driver can name the dimension in the stop-reason line.
   A plain store; no trajectory effect (golden-gated).

5. **Adaptive block size floor (upstream bug fix)**: the `ASA_PARALLEL`
   periodic test (the `Acceptance_Frequency_Modulus` /
   `Generated_Frequency_Modulus` / `Accepted_To_Generated_Ratio`
   section) resizes `OPTIONS->Gener_Block` to the moving average, over
   the last `Gener_Mov_Avr` tests, of `*recent_number_generated`,
   truncated to an integer and capped at `Gener_Block_Max`.
   `*recent_number_generated` is reset to 0 whenever a new best is
   accepted, and at a periodic test whose accepted-to-generated ratio is
   below `Accepted_To_Generated_Ratio` (after the resize), so a test that
   follows a new best adds a 0 and the average
   can truncate to 0 -- `{1, 1, 0} / 3` at the first test of a run at the
   default options (6 of 30 seeds on the HPO termination-test fixture;
   the HPO examples' data with `OPTIM RCUT2B (1)` and `SGRIDMB (1)`,
   `SEED 1`). With `Gener_Block` 0 the block-generation loop generates
   and evaluates nothing, and its exit conditions cannot change without
   an evaluation: serial `asa()` spun for ever without calling the cost
   function, so no `MAXEVAL`, `MAXTIME` or signal stop could end it; with
   the TADAH hook installed the hook was called with empty blocks, and
   only its `MAXTIME` check could. The patch keeps the resized
   block at 1 or more (upstream's own `asa_test_asa_options` rejects
   `Gener_Block < 1`). Only runs that would have spun change; a run at
   `Gener_Block = Gener_Block_Max = 1` now stays canonical serial ASA
   throughout (golden-gated on a cost whose every state is a new best).
   The patch does not change what the resizing averages. The README
   (section `ASA_PARALLEL`) describes the average as one over "the
   previous numbers of generated states required to find a new best
   accepted state", but the code samples the running count at each
   periodic test, which restarts at 0 at every new best: a test soon
   after a new best feeds a small count, so the block leans towards 1
   and fewer than `min(THREADS, Gener_Block_Max)` workers may be busy
   (in a standalone sweep, 63 of 300 `Gener_Block = Gener_Block_Max = 4`
   runs ended with blocks of 1 to 3 states, most of them on an objective
   that kept improving or with a periodic test every 1 or 7
   acceptances). Recording the count at each new best instead would
   change the trajectory of every block run; it is left as upstream
   wrote it.

6. **A pending stop ends the sampling loops (upstream limitation)**:
   outside its main loop `asa()` samples states in three retry loops
   that draw until a state is valid, bounded only by
   `Limit_Invalid_Generated_States`: the `Number_Cost_Samples` states of
   the initial cost temperature, a random valid start with
   `User_Initial_Parameters = FALSE`, and the `|Reanneal_Cost|` states
   of the cost-temperature resampling at a periodic test
   (`|Reanneal_Cost| > 1`). The README says `Immediate_Exit` is ignored
   while the cost function is called to calculate derivatives or the
   cost temperature, and the resampling restores the value
   `Immediate_Exit` had before it. Once the TADAH driver has flagged a
   stop (`MAXEVAL`, `MAXTIME`, `STOPVAL`, a signal, an evaluation's
   exception), every callback returns an invalid state without
   evaluating, so the loop ran on for the whole limit (1000 no-op
   callbacks by default, then `TOO_MANY_INVALID_STATES`), and with
   `Limit_Invalid_Generated_States = -1` for ever, at full CPU and deaf
   to further signals. After upstream's own limit check in each of the
   three loops, the patch exits with `IMMEDIATE_EXIT` when
   `Immediate_Exit` is set, skipping the resampling's restore. Guarded
   `#if ASA_PARALLEL` (featured build only, as patch 4). Only a run with
   a stop already pending changes, and it evaluates nothing more either
   way: it ends at the stop with `IMMEDIATE_EXIT`, as in the main loop.
   The bounded calls of `cost_derivatives()` (reanneal tangents, exit
   curvature) need no guard: they end after a fixed number of calls.

7. **`Gener_Mov_Avr 0` is accepted (upstream inconsistency)**: the README
   documents `Gener_Mov_Avr 0` as "`Gener_Block` is not changed
   thereafter" (`OPTIONS->Gener_Mov_Avr`, and the `ASA_PARALLEL`
   section), and the periodic-test resizing (patch 5's section) already
   runs only when `Gener_Mov_Avr > 0`, but `asa_test_asa_options` rejected
   every value below 1. The patched check rejects only negative values,
   and the moving-average buffer `parallel_gen_ratio_block` is allocated
   with at least one element (`calloc(0, ...)` may return NULL, which
   would read as `CALLOC_FAILED`); the buffer is not read at 0. With
   `PARAM Gener_Mov_Avr 0` every generated block holds `Gener_Block`
   states. Opt-in: the default window stays 3, and a run that does not set
   it is unchanged (golden-gated). Both hunks are inside `#if
   ASA_PARALLEL` (featured build only).

## Known upstream quirks the embedding must respect

- `ASA_PRINT` must stay `1`: `asa_exit()` calls
  `fflush(ptr_asa_out); fclose(ptr_asa_out);` unconditionally, and the
  stream is only opened under `ASA_PRINT`. For the same reason the
  output filename must never be the literal `"STDOUT"` (the process's
  stdout would be fclosed at every `asa()` exit).
- `asa()` is not re-entrant: function-local statics
  (`asa_open`/`number_asa_open`/`recursive_asa_open`; note
  `recursive_asa_open` is never decremented), and file-scope globals
  (`exit_msg`, `gener_block_state_qsort`, `multi_cost_qsort`). The TADAH
  driver serializes all calls behind a process-wide mutex.
- `USER_ACCEPTANCE_TEST=TRUE` does not compile in 30.53
  (`asa.c` references `OPTIONS->Acceptance_Fit`; the field is
  `Acceptance_Test`). Left disabled rather than patched.
- `USER_INITIAL_COST_TEMP=TRUE` is not trajectory-neutral: it compiles
  out the `Number_Cost_Samples` initial-cost-temperature sampling loop
  (changes the RNG stream and eval count) — excluded.
- `ASA_RESOLUTION=TRUE` grid-snaps every generated state
  (`Coarse_Resolution`) and has no disable value — excluded;
  `ASA_QUEUE`'s `Queue_Resolution` covers the dedup use case.
- `ASA_SAVE*` uses `popen("/bin/cp ...")` (under `SYSTEM_CALL`),
  `#include "asa_opt"` as C source, and persists the statics above —
  excluded; HPO runs are not resumable.
- `Gener_Block_Max` caps the block resizing from above only: with
  `Gener_Block_Max == Gener_Block` the moving average can still shrink the
  block below `Gener_Block` (down to 1, patch 5). `Gener_Mov_Avr 0`
  switches the resizing off (patch 7).

## Gate-set change procedure

A compile gate in `tadah_asa_config.h` may be enabled only if the
golden-neutrality test (`tests/test_asa_golden.cpp`) still proves
bitwise-identical trajectories against `tadah_asa_pristine` at default
option values. If a gate fails that test: set it to 0 here, remove its
PARAM keys from `src/asa_options.cpp`, and record the exclusion above.
