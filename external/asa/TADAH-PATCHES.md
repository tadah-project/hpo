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
   OpenMP loop, a data race on the shared `USER_DEFINES`).

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
   `Limit_Invalid_Generated_States = -1`).

4. **P_TEMP_TOO_SMALL dimension mirror**: upstream records the index of
   the dimension whose temperature underflowed only in an internal
   variable (`*index_exit_v`) and its own journal. The patch (guarded
   `#if ASA_PARALLEL`, i.e. featured build only) mirrors it into
   `tadah_asa_exit_index` (declared in `tadah_asa_hook.h`) at the exit
   site so the driver can name the dimension in the stop-reason line.
   A plain store; no trajectory effect (golden-gated).

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
  excluded; HPO has its own checkpointing.
- `Gener_Mov_Avr < 1` is rejected by `asa_test_asa_options`; a frozen
  block size is expressed as `Gener_Block_Max == Gener_Block`.

## Gate-set change procedure

A compile gate in `tadah_asa_config.h` may be enabled only if the
golden-neutrality test (`tests/test_asa_golden.cpp`) still proves
bitwise-identical trajectories against `tadah_asa_pristine` at default
option values. If a gate fails that test: set it to 0 here, remove its
PARAM keys from `src/asa_options.cpp`, and record the exclusion above.
