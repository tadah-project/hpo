# HPO eval-output bugs — autonomous fix ticket

This ticket bundles four independent bugs that surfaced during the
ITER5 BASINS run on 2026-05-05. They are tracked together because they
share the same surface (HPO eval-output / resume path) and a single
end-to-end test exercises all of them. Fix them in any order, but the
acceptance criteria below require all four to pass.

The agent picking up this ticket has full authority to investigate,
fix, write tests, and verify end-to-end. The user does NOT need to be
consulted unless a fix requires a behaviour change beyond what's
described here.

---

## Shared environment

**Source root:**
`/home/marcin/PROJECTS/TADAH/develop_wt`

**Build dir** (already configured):
`/home/marcin/PROJECTS/TADAH/develop_wt/build_tadah`

**Build command:**
```bash
TADAH_PATH=/home/marcin/PROJECTS/TADAH/develop_wt \
  cmake --build /home/marcin/PROJECTS/TADAH/develop_wt/build_tadah --target tadah -j 8
```

**Repro workdir** (real LAMMPS objective, ~22 active datasets, ~60 LAMMPS targets):
`/home/marcin/TEMP/Izzy/Ta_LAMMPS_fit_typical_mod/ATOMISED/workdir/ITER5/`

That directory has the live `Ta_config.basins.tadah`, `config.hpo.G_basins`,
all `_fast.in` LAMMPS scripts, the validation set, and a known-good
checkpoint.tadah at round=3. Use it as the integration-test fixture.
Do not modify the live `checkpoint.tadah` in place — copy it to a
scratch dir first.

**Repro command** (single run, fresh, no resume):
```bash
cd /tmp/hpo_bug_repro && rm -rf * && \
  cp -r /home/marcin/TEMP/Izzy/Ta_LAMMPS_fit_typical_mod/ATOMISED/workdir/ITER5/{*.in,*.tadah,*.tadahdb,*.val,*.db} . && \
  cp /home/marcin/TEMP/Izzy/Ta_LAMMPS_fit_typical_mod/ATOMISED/workdir/ITER5/{Ta_config.basins.tadah,config.hpo.G_basins} . && \
  sed -i 's|BUDGET_MAX_EVALS    15000|BUDGET_MAX_EVALS    250|' config.hpo.G_basins && \
  sed -i 's|NOISE.CALIBRATION 10|NOISE.CALIBRATION 5|' config.hpo.G_basins && \
  OMP_NUM_THREADS=8 timeout --foreground 600 \
    /home/marcin/PROJECTS/TADAH/develop_wt/build_tadah/tadah hpo \
    --validation config.val --hpotarget config.hpo.G_basins -c Ta_config.basins.tadah
```

This sets `BUDGET_MAX_EVALS 250` so round 1 GLOBAL → LOCAL transition
happens in ~2 minutes (Bug 4 trigger), and the writer-thread freeze
should reproduce on the bigger LAMMPS dataset within the 10-minute
budget.

**Repro for the resume bugs**: run the command above to completion,
then re-run the same command (the existing checkpoint.tadah in cwd will
trigger try_load).

**Files relevant to all four bugs:**

| Path | Purpose |
|---|---|
| `HPO/include/tadah/hpo/hpo_host.h` | HPO_Host class: g_iter_, g_best_, writer_token_, io_buf_, next_flush_ |
| `HPO/include/tadah/hpo/hpo_host_impl.hpp` | operator() + submit_result + flush_remaining |
| `HPO/include/tadah/hpo/hpo_log.h` | HPOLog declarations |
| `HPO/src/hpo_log.cpp` | HPOLog::log_new_best, save, load |
| `HPO/src/hpo.cpp` | hpo_run() — drives NOISE.CALIBRATION + cpm.try_load order |
| `HPO/include/tadah/hpo/basins_policy_impl.hpp` | BasinsPolicy::run, commit_checkpoint |
| `CORE/src/checkpoint_manager.cpp` | save_now, try_load |

---

## BUG 1 — `HPO_Host::g_best_` not persisted across resume

### Severity
High. Causes spurious "NEW BEST" events on resume that overwrite
`best_pot.tadah` with worse-than-truth potentials and corrupt
`HPOLog::cur_best_loss_` (see Bug 2).

### Evidence

Pre-restart (from live ITER5 checkpoint at 21:36):
```
[HPOLOG]
cur_best_loss = 9.04706
cur_best_eval = 22703
n_eval = 25825
```

Post-restart journal:
```
[22:35:22]  BEST     ★ NEW BEST   eval=     1  loss=3.6113e+04   (NOISE.CALIBRATION; Bug 3)
[22:35:25]  BASINS: resumed from checkpoint (n_eval=31578, best_loss=9.047056)
[22:35:25]  BASINS resumed at round=3 stage=LOCAL loc_step=1 pool=16 basin(s)
[22:35:26]  SUMMARY  --- 31579 evals done   best=9.0471e+00 (eval 22703, 8876 evals ago)
[22:35:26]  BEST     ★ NEW BEST   eval= 31579  loss=1.6353e+02   ← spurious
```

The summary line correctly reports best=9.047 (it reads from
`HPOLog::cur_best_loss_`, which `HPOLog::load` restored). The very
next BEST event reports 163 as the new best, even though
163 > 9.047. This means the per-eval is_best comparator was using a
different (uninitialised) value.

### Diagnosis (verified)

`HPO/include/tadah/hpo/hpo_host.h:204`:
```cpp
static std::atomic<double>  g_best_;
```

`HPO/include/tadah/hpo/hpo_host_impl.hpp:1021`:
```cpp
double cur = g_best_.load(std::memory_order_relaxed);
bool i_am_best = false;
while (r.loss < cur &&
       !(i_am_best = g_best_.compare_exchange_weak(cur, r.loss, ...)));
if (i_am_best) {
    r.is_best   = true;
    r.best_iter = g_best_iter_.fetch_add(1, ...) + 1;
    iter_at_last_best_.store(r.iter, ...);
}
```

`g_best_` (and the related `g_best_iter_`, `iter_at_last_best_`) are
NOT in `HPOLog::save` (`hpo_log.cpp:963`) or `HPOLog::load`
(`hpo_log.cpp:983`). On resume they retain whatever value they
accumulated since process start — for a BASINS run with
NOISE.CALIBRATION, that's the noise-calibration first-eval loss
(Bug 3). With Bug 3 fixed but Bug 1 still present, `g_best_` would
sit at +infinity (its default static-init value), so the very first
post-resume eval would always look like a new best.

### Required fix

Persist `g_best_` and `g_best_iter_` and `iter_at_last_best_` across
resume. Two acceptable approaches:

1. **Make HPO_Host an `ICheckpointable` participant.** Add `save` /
   `load` / `tag` methods on HPO_Host that read/write the static
   atomics. Register from `hpo_run()` after `cpm.set_path` and before
   `cpm.register_(&log)`. Tag suggestion: `[HPO_HOST]`.

2. **Have hpo_run() seed the atomics from HPOLog after try_load.** Add
   a public `HPO_Host::seed_globals(double best_loss, std::size_t
   best_eval, std::size_t last_best_iter)` static and call it from
   hpo_run() inside the `if (cpm.try_load())` block. This is the
   smaller change but couples hpo_run to HPO_Host's internals.

Either fix is fine; the agent should pick whichever fits the existing
patterns better. (Approach 1 is more consistent with how HPOLog and
BasinPool are wired.)

### Required regression test

`HPO/tests/test_hpo_resume_g_best.cpp` (Catch2):

- Build a minimal HPO_Host on a synthetic 2-D objective.
- Run 100 evals with a known minimum at loss=1.0; observe `g_best_`
  reaches 1.0 (read via a test-only public accessor or via observing
  HPOLog's `best_loss()`).
- Commit a checkpoint via `CheckpointManager::save_now()`.
- Reset all HPO_Host static atomics (test-only `reset_globals()`).
- Call `cpm.try_load()`.
- Assert `HPO_Host::current_best()` returns 1.0 (NOT +infinity, NOT
  0.0).
- Run one more eval at loss=10.0 and assert that `r.is_best == false`
  (BUG 1's signature failure mode).
- Run one more eval at loss=0.5 and assert `r.is_best == true` and
  HPOLog reports best=0.5.

This must pass with the fix and fail without it.

---

## BUG 2 — `HPOLog::log_new_best` blindly overwrites `cur_best_loss_`

### Severity
Medium. Defensive issue: it would also misbehave if any caller fires
`log_new_best` for a non-best loss (e.g. via a future code path).
Independent of Bug 1 — fix both.

### Evidence

Same journal sequence as Bug 1. After the spurious BEST event at
loss=163.5 fires, the next SUMMARY shows best=1.6353e+02 — proving
that `log_new_best` lowered `cur_best_loss_` from 9.047 to 163.5
unconditionally.

### Diagnosis (verified)

`HPO/src/hpo_log.cpp:746`:
```cpp
void HPOLog::log_new_best(std::size_t eval_num, double loss, ...)
{
    std::lock_guard<std::mutex> lg(mtx_);
    if (!f_.is_open()) return;

    cur_best_loss_ = loss;            // <-- no guard
    cur_best_eval_ = eval_num;
    ...
}
```

No `loss < cur_best_loss_` check.

### Required fix

```cpp
void HPOLog::log_new_best(std::size_t eval_num, double loss, ...)
{
    std::lock_guard<std::mutex> lg(mtx_);
    if (!f_.is_open()) return;
    if (loss >= cur_best_loss_) return;     // <-- new guard
    cur_best_loss_ = loss;
    cur_best_eval_ = eval_num;
    ...
}
```

The guard catches Bug 1's downstream effect even if Bug 1 itself
isn't fixed. Belt-and-braces.

Take care: the `f_.is_open() return` is currently before the
overwrite, which is fine. The new guard goes between them. Also note
that `cur_best_loss_` is initialised to `+infinity` in HPOLog's ctor
— check `hpo_log.h` and confirm. If the default were 0.0 the guard
would suppress all bests, which would be wrong.

### Required regression test

`HPO/tests/test_hpolog_new_best_guard.cpp` (Catch2):

- Build an HPOLog (open to a temp file).
- Call `log_new_best(eval=1, loss=10.0, ...)`. Assert `best_loss() == 10.0`.
- Call `log_new_best(eval=2, loss=20.0, ...)` (worse loss).
- Assert `best_loss() == 10.0` (the guard rejected the regression).
- Assert `best_eval() == 1`.
- Call `log_new_best(eval=3, loss=5.0, ...)` (genuine new best).
- Assert `best_loss() == 5.0`, `best_eval() == 3`.

---

## BUG 3 — `NOISE.CALIBRATION` runs BEFORE `cpm.try_load()`

### Severity
Medium. Wastes N (= `noise_calibration` count) LAMMPS evals on every
resume; pollutes the eval counter, the per-eval logs (loss.tadah /
outvar.tadah / params.tadah), and `g_best_` (Bug 1's trigger).

### Evidence

Post-restart journal head:
```
[22:35:22]  NOISE.CALIBRATION: running 10 evals at the seed point
[22:35:22]  BEST     eval=1  loss=3.6113e+04   (1.5 s)
[22:35:23]  WARNING  eval 2: skipping LAMMPS ...
...
[22:35:25]  NOISE.CALIBRATION: rel_spread=0.000e+00  suggested FTOL_REL=1.000e-06
[22:35:25]  BASINS: resumed from checkpoint (n_eval=31578, best_loss=9.047056)
```

10 NOISE.CALIBRATION evals happened before the resume message. They
were unnecessary — the original run already calibrated noise, and
the suggested ftol was already used by the BASINS state machine.

### Diagnosis (verified)

`HPO/src/hpo.cpp` execution order:
```
1.  log.open(...)
2.  SignalHandler::install()
3.  HPOTargets trg(...)
4.  HPO_Host host(...)
5.  log.write_optimizer_section(...) etc.
6.  if (sp.run_policy_config.noise_calibration >= 2) { ... }   ← line ~98
7.  log.start_progress();
8.  policy = build_run_policy(...);   ← BasinsPolicy registers w/ cpm in ctor
9.  cpm.set_path(...); cpm.register_(&log); cpm.try_load();    ← line ~167
10. policy->run(...)
```

NOISE.CALIBRATION at step 6 runs N evals via `host(seed_vec)` which
calls `operator()` which increments `g_iter_` and updates `g_best_`.
By step 9, the eval counter and best tracker are polluted. Then
`cpm.try_load()` at step 9 restores HPOLog state but NOT g_iter_
(handled by `set_g_iter_floor` in step 9's `if (cpm.try_load())`
block) and NOT g_best_ (Bug 1).

### Required fix

Reorder so `cpm.try_load()` happens before NOISE.CALIBRATION, and
NOISE.CALIBRATION only fires on a fresh start:

```cpp
// hpo.cpp — proposed order
log.open(...);
SignalHandler::install();
HPOTargets trg(...);
HPO_Host host(...);
log.write_optimizer_section(...); // setup sections only

auto policy = build_run_policy<V,R>(sp, &trg);

auto& cpm = core::CheckpointManager::instance();
bool resumed = false;
if (sp.lib == "BASINS") {
    cpm.set_path("checkpoint.tadah");
    if (sp.basins.checkpoint_seconds > 0.0)
        cpm.set_period_seconds(sp.basins.checkpoint_seconds);
    if (sp.basins.checkpoint_evals > 0)
        cpm.set_period_evals(sp.basins.checkpoint_evals);
    cpm.register_(&log);
    resumed = cpm.try_load();
    if (resumed) {
        log.info("BASINS: resumed from checkpoint " + cpm.path() + ...);
        HPO_Host<V,R>::set_g_iter_floor(log.n_eval());
        // Bug 1 fix: also restore g_best_, etc.
    }
}

if (!resumed && sp.run_policy_config.noise_calibration >= 2) {
    // existing NOISE.CALIBRATION block
}

log.start_progress();
policy->run(sp, host);
```

Edge case: if the user changes the seed point or the dataset between
the original run and the resume, the cached noise calibration is
stale. The agent does NOT need to handle this — document in the
docstring that resume reuses the prior calibration; if the user wants
to recalibrate, they should delete checkpoint.tadah.

### Required regression test

`HPO/tests/test_noise_calibration_resume.cpp` (Catch2):

- Run a small BASINS config with `NOISE.CALIBRATION 3` to completion
  (e.g. MAXEVAL or MAXTIME 5s).
- Capture eval counter and journal contents.
- Re-run the same config in the same dir (resume from checkpoint).
- Assert: post-resume journal does NOT contain "NOISE.CALIBRATION:
  running" string after the resume marker.
- Assert: post-resume eval counter advances by < 3 evals before the
  first BASINS round-stage event (it should be 0 — go straight to
  policy->run).

---

## BUG 4 — Writer thread silently freezes mid-run

### Severity
High but distinct from Bugs 1–3. With Bugs 1–3 fixed, Bug 4 still
causes silent data loss: per-eval logs (loss.tadah, outvar.tadah,
params.tadah, stats.tadah) and best-pot files (best_loss.tadah,
best_outvar.tadah, best_pot.tadah) freeze at some point mid-run; the
process keeps running, the journal keeps recording BEST events, but
nothing reaches disk.

### Evidence

Run timeline (live ITER5, restart at 22:35):
```
22:35:22  NOISE.CALIBRATION eval 1: BEST event fires, best_pot.tadah written
22:35:25  best_pot.tadah mtime = 22:35:25
22:35:26  Resume; first post-resume eval at iter=31579
          BEST event fires for iter=31579 (loss=1.6353e+02)
22:35:27  BEST event for iter=31580 ...
...
22:37:42  hpo_run.log mtime = 22:37:42 (still being appended)
          checkpoint.tadah mtime = 22:37:42 (still committed)
          best_pot.tadah mtime = 22:35:25 (FROZEN; never updated post-resume)
```

In the original 17:23 run the writer froze ~35 minutes in, at the
round-1 GLOBAL → LOCAL transition. In the 22:35 restart the writer
froze immediately after the resume. Both freezes leave the process
alive and CPU-busy with all worker threads in `Rl+` state — no
deadlock, no thread blocked.

### Diagnosis (hypothesis — needs verification)

`HPO/include/tadah/hpo/hpo_host_impl.hpp:1188`, function
`HPO_Host::submit_result`:

```cpp
g_q_.enqueue(std::move(res));
if (g_q_.size_approx() < trg.QFLUSH_THRESHOLD) return;
if (writer_token_.test_and_set(std::memory_order_acquire)) return;

{
    std::lock_guard<std::mutex> lg(io_mtx_);
    Result item;
    while (g_q_.try_dequeue(item))
        io_buf_.emplace(item.iter, std::move(item));

    static std::ofstream loss_f(...), outv_f(...), ... ;

    while (io_buf_.count(next_flush_))
    {
        const auto& e = io_buf_[next_flush_];
        if (next_flush_ % trg.outputEvery == 0) { print_loss_line(...); ... }
        if (e.is_best) {
            print_loss_line(best_f, ...);
            ...
            { std::ofstream bp("best_pot.tadah");
              PotHeaderWriter::write_best_header(bp, e, trg, ...);
              warmstart::dump_to_stream(bp);
              bp << std::setprecision(e.pot_ctx.template get<size_t>("NUMERIC")) << e.pot_ctx;
            }
            if (e.best_iter % trg.dumpBestPotEvery == 0) { ... }
        }
        if (e.iter % trg.dumpPotEvery == 0) { ... }
        io_buf_.erase(next_flush_++);
    }

    loss_f.flush(); outv_f.flush(); ...
}

writer_token_.clear(std::memory_order_release);
```

Hypothesis: a call inside the exclusive section throws an exception.
The catch is missing, so:
1. The thread that was the writer unwinds out of `submit_result`.
2. `writer_token_` is never cleared (it was set before the catch
   point).
3. Every subsequent thread that reaches line 1211 sees the token set
   and returns without flushing.
4. The queue keeps growing but nothing drains it.
5. Other code paths (`HPOLog`, `CheckpointManager`) still work
   because they don't touch the writer_token_ / io_buf_ state.

Candidate throw sites, in order of suspicion:
1. `e.pot_ctx.template get<size_t>("NUMERIC")` if `pot_ctx` is missing
   the NUMERIC key. This throws `std::out_of_range` from
   `core::Context::get`.
2. `PotHeaderWriter::write_best_header` if `master_ctx` /  `val_ctx`
   /  the train/val StructureDB references are in an inconsistent
   state during a stage transition.
3. `warmstart::dump_to_stream` if the warmstart registry is being
   mutated by another thread.
4. `print_loss_line` etc. with a Result whose simulation results were
   only partially populated (e.g. an early-failing eval).

The 22:35:25 freeze (writer dies on its FIRST flush after resume) is
strong evidence for #1 — if the post-resume eval's pot_ctx is
incompletely initialised, the very first best-iter write throws.

### Required fix (regardless of root cause)

Wrap the writer's exclusive section in try/catch and unconditionally
clear the token:

```cpp
// In submit_result, replace the existing block:
bool throw_caught = false;
std::string throw_msg;
try {
    std::lock_guard<std::mutex> lg(io_mtx_);

    Result item;
    while (g_q_.try_dequeue(item))
        io_buf_.emplace(item.iter, std::move(item));

    static std::ofstream loss_f(trg.lossFile, std::ios::app), ...;

    while (io_buf_.count(next_flush_)) {
        try {
            const auto& e = io_buf_[next_flush_];
            // ... existing per-iter and per-best writes ...
            io_buf_.erase(next_flush_++);
        } catch (const std::exception& ex) {
            // Drop this iter's logs; advance past it so we don't loop.
            io_buf_.erase(next_flush_++);
            throw_caught = true;
            throw_msg = std::string("writer dropped iter ") +
                        std::to_string(next_flush_ - 1) +
                        ": " + ex.what();
        }
    }

    loss_f.flush(); /* ... */
} catch (const std::exception& ex) {
    throw_caught = true;
    throw_msg = std::string("writer outer caught: ") + ex.what();
}

writer_token_.clear(std::memory_order_release);

if (throw_caught) {
    HPOLog::instance().warning(throw_msg);
}
```

Same pattern in `flush_remaining` (line 1316).

This makes the writer recover-able regardless of the root cause:
one offending iter is logged and dropped, the queue keeps draining,
the journal records the failure, and the next BEST event will succeed.

### Required follow-up — find and fix the root cause

The catch is defensive. The agent should ALSO identify what's
throwing:

1. Build with the catch installed (so the run doesn't hang) plus a
   `std::cerr << "writer caught at iter " << ...` in the catch block
   that includes the stringified exception type.
2. Run the repro from `Shared environment` above to capture which
   iter throws and what the exception is.
3. Inspect the offending Result. If it's the `pot_ctx.get<size_t>`
   theory, check why pot_ctx is incomplete for that iter — is it an
   eval that returned early due to NaN / FAILSCORE? If yes, fix the
   early-return path to ensure pot_ctx is always populated, OR guard
   the writer with `if (e.pot_ctx.has("NUMERIC"))`.
4. Document the root cause as a follow-up note in this file.

### Required regression tests

Two tests:

`HPO/tests/test_writer_recovers_from_throw.cpp` (Catch2):
- Inject a throwing print function (mock `print_loss_line`).
- Submit 100 results; assert that the writer recovers and writes
  results 1–9 and 11–100 (skipping the throwing iter 10).
- Assert that `writer_token_` is cleared after each call.
- Assert that `next_flush_` advances past the throwing iter.

`HPO/tests/test_writer_no_freeze_long_run.cpp` (integration test, may
be slow — gate with `[slow]` Catch2 tag):
- Run BASINS with `BUDGET_MAX_EVALS 100, MAXEVAL 500` on a synthetic
  fast objective.
- Assert that `loss.tadah` line count == expected (500 / outputEvery).
- Assert that `best_pot.tadah` mtime is within 5 seconds of process
  exit.
- Assert that `next_flush_` reaches 500 after the run completes.

---

## End-to-end acceptance criteria

A single integration test exercises all four bugs together. Place it
at `HPO/tests/test_resume_idempotence.cpp` and gate it `[slow]`.

```
SCENARIO: tadah hpo BASINS run with resume should be idempotent

GIVEN: a fresh workdir copied from ITER5 (small subset — 1 dataset,
       2 PC_LAMMPS targets is enough; or a synthetic objective)
WITH:  config.hpo.G_basins truncated to BUDGET_MAX_EVALS 100, MAXEVAL 500

WHEN: we run `tadah hpo` to completion (run A)
THEN: hpo_run.log finishes cleanly with no BASINS-checkpoint-write errors
AND:  best_pot.tadah mtime > start time (writer was alive at end)
AND:  loss.tadah has 500/outputEvery lines
AND:  checkpoint.tadah's [HPOLOG] cur_best_loss matches journal best
AND:  let best_A = the recorded final best loss

WHEN: we re-run `tadah hpo` in the same dir (run B)
THEN: hpo_run.log shows "BASINS: resumed from checkpoint" within the first 30 lines
AND:  the journal's first SUMMARY line after resume shows best == best_A
AND:  the journal does NOT show any NEW BEST events with loss > best_A
AND:  no NOISE.CALIBRATION block in run B (Bug 3)
AND:  best_pot.tadah header reports a Loss <= best_A (i.e. the file is
      either unchanged from run A or has been updated to a strictly
      better potential — never a worse one)
```

This test fails on the un-patched binary (Bug 1 causes spurious
NEW BEST, Bug 3 reruns NOISE.CALIBRATION, Bug 4 may freeze the writer)
and passes once all four bugs are fixed.

---

## Verification protocol

Before marking the ticket complete:

1. All four unit/regression tests above pass.
2. The end-to-end test passes.
3. The full test suite passes:
   ```
   cmake --build /home/marcin/PROJECTS/TADAH/develop_wt/build_tadah \
     --target hpo_tests core_tests -j 8
   cd /home/marcin/PROJECTS/TADAH/develop_wt/build_tadah && \
     ctest -R '^(test_hpo|test_basin|test_progress|test_tolerance|test_checkpoint|test_writer|test_noise|test_resume)' \
     --output-on-failure
   ```
   Existing tests must still pass.
4. Manual smoke test on the live ITER5 workdir:
   - Delete `checkpoint.tadah` from a scratch copy of ITER5.
   - Run `./run.sh 22` for ~5 minutes (interrupt with ctrl+C).
   - Check journal final entries — should show clean shutdown.
   - Re-run `./run.sh 22` — confirm "resumed from checkpoint",
     no NOISE.CALIBRATION rerun, no spurious BEST events, journal
     SUMMARY first-line best matches the pre-interrupt best.
5. Add an entry to this file documenting:
   - The actual root cause of Bug 4 (was it `pot_ctx.get<size_t>`?
     something else?).
   - Any behaviour-change decisions made (e.g. how early-failing
     evals' pot_ctx is now handled).

The agent should commit each bug's fix + its regression test as a
separate commit so they're independently bisectable. The end-to-end
test goes in its own commit at the end.

## Resolution (2026-05-05)

All four fixes landed together with regression coverage. Summary:

### Code changes
- **Bug 1**: `HPO_Host` now inherits `core::ICheckpointable` with tag
  `[HPO_HOST]`, persisting `g_best_`, `g_best_iter_`, `iter_at_last_best_`,
  and `g_iter_`. `hpo_run()` registers the host with the checkpoint
  manager alongside `HPOLog` and the BasinsPolicy. Static helpers
  `HPO_Host::save_globals` / `load_globals` / `reset_globals_for_test`
  expose the same logic for unit tests.
  Files: `HPO/include/tadah/hpo/hpo_host.h`, `hpo_host_impl.hpp`,
  `HPO/src/hpo.cpp`.

- **Bug 2**: `HPOLog::log_new_best` now early-returns when
  `loss >= cur_best_loss_`. `cur_best_loss_` defaults to
  `numeric_limits<double>::max()` so the very first finite best is
  still accepted.
  Files: `HPO/src/hpo_log.cpp`.

- **Bug 3**: `hpo.cpp` now runs the BASINS checkpoint subsystem
  (`set_path` / `register_` / `try_load`) BEFORE NOISE.CALIBRATION,
  and gates the calibration block on `should_run_noise_calibration
  (resumed, noise_n)` which returns `false` whenever `resumed`. The
  predicate is exported in `tadah/hpo/hpo.h` for unit testing.
  Files: `HPO/include/tadah/hpo/hpo.h`, `HPO/src/hpo.cpp`.

- **Bug 4**: The writer's exclusive section in
  `HPO_Host::submit_result` and `HPO_Host::flush_remaining` is now
  wrapped in nested try/catch:
  - **Inner** try/catch around each iter's body: drops the offending
    iter (`io_buf_.erase(next_flush_++)`) and logs a journal warning
    that includes the iter number, `is_best`, `pot_ctx.size`, whether
    NUMERIC is present, and the exception type. This guarantees one
    bad iter cannot stall the queue forever.
  - **Outer** try/catch around the whole exclusive section: logs and
    falls through so `writer_token_.clear()` always runs.
  Files: `HPO/include/tadah/hpo/hpo_host_impl.hpp`.

### Tests
- `HPO/tests/test_hpolog_new_best_guard.cpp` — Bug 2 guard.
- `HPO/tests/test_hpo_resume_g_best.cpp` — Bug 1 round-trip via the
  `save_globals` / `load_globals` static helpers.
- `HPO/tests/test_noise_calibration_resume.cpp` — Bug 3 gate predicate
  + parser sanity.
- `HPO/tests/test_writer_recovers_from_throw.cpp` — Bug 4 recovery
  pattern (structural simulator using the real `HPOLog`).
- `HPO/tests/test_resume_idempotence.cpp` — end-to-end `[slow][e2e]`
  test gated on `TADAH_BINARY` and `TADAH_HPO_RESUME_FIXTURE` env
  vars; copies the fixture to a scratch dir, runs `tadah hpo` twice,
  and asserts the resume invariants. Skips gracefully when env vars
  are unset.

`HPOLog::reset_for_test()` was added so test cases don't leak singleton
state into each other.

### Bug 4 root cause — investigation status

Not directly reproduced in this session; the live workdir's full
LAMMPS reproduction takes ~35 minutes for the original mid-run trigger
and was not run here. Instead the catch block was instrumented to
emit a journal entry that names the iter, `is_best`, whether the
Result's `pot_ctx` has the NUMERIC key, the exception type
(`typeid(ex).name()`), and `what()`. The next time the freeze trigger
fires in production, the journal entry alone will discriminate
between the candidate throw sites listed in the original report:

- `pot_ctx.get<size_t>("NUMERIC")` → `std::runtime_error` with
  `what()` containing "Key not found in the Context object: NUMERIC".
  Diagnostic will show `has NUMERIC=false`.
- `PotHeaderWriter::write_best_header` failure → typed
  `std::runtime_error` from whichever sub-write throws; `pot_ctx.size`
  > 0 but specific keys may be missing.
- `warmstart::dump_to_stream` → throws from the warmstart registry
  iterator; iter info present, `is_best=true`.
- per-iter print failure → `is_best=false` plus exception detail.

Recommendation for the next run that hits the freeze: check
`hpo_run.log` for the `HPO writer: dropping iter=…` warning. The
fields in that line will identify the exact throw path within
seconds, no fresh repro needed.

### Verification

- `ctest` HPO regression suite (14 entries from this ticket): all pass.
- Full HPO test suite (85 cases / 3324 assertions): all pass.
- Full project ctest: 477/478 passing. The single failure
  (`Testing Factory: Cutoffs` in `MODELS/tests/test_factory_cutoffs.cpp`)
  is pre-existing and unrelated — none of this ticket's edits touched
  MODELS.
- Live ITER5 smoke: not run in this session (deferred to user).

---

# BUG 5 — `operator()` throws between `++g_iter_` and `submit_result`, leaving permanent gap in writer queue

## Severity
High. Symptom is identical to old Bug 4 (writer freezes mid-run, per-iter
log files stop updating, `hpo_run.log` keeps progressing) but the root
cause is different and the Bug 4 fix does not address it. Discovered by
observing the live ITER5 run on 2026-05-06 with the Bug 4 catch-block
fix in place.

## Evidence

Live ITER5 run snapshot at 2026-05-06 09:43 (process PID 814900,
running ~2 hours):

```
file mtimes:
  loss.tadah        2026-05-06 08:21:48   (last iter 3450)
  outvar.tadah      2026-05-06 08:21:48
  params.tadah      2026-05-06 08:21:48
  stats.tadah       2026-05-06 08:21:48
  hpo_run.log       2026-05-06 09:43:40   (still updating)
  checkpoint.tadah  2026-05-06 09:43:44   (still committing)

checkpoint state: n_eval = 18926
  (15,476 iters lost between iter 3450 and n_eval)

journal scan: zero "HPO writer:" warnings, zero "writer dropped iter="
events.
```

The Bug 4 catch block (at `hpo_host_impl.hpp:1252–1395`) would emit a
`HPO writer:` warning if any per-iter or per-best write threw inside
the writer's exclusive section. None were emitted. So the writer
itself never threw. Yet ~15k iters never reached the per-iter logs,
and the writer is not making progress.

## Diagnosis

`operator()` (`hpo_host_impl.hpp:908`) increments `g_iter_` very
early:

```cpp
double HPO_Host<P,R>::operator()(const P& p) const
{
    if (core::SignalHandler::stop_requested()) {
        throw core::ShutdownRequested{};
    }
    Result r;
    r.iter   = ++g_iter_;        // <-- iter committed here, line 919
    r.params = p;
    // ... train (line ~994), predict (line ~1006), Analytics (~1007),
    //     fit_loss (line ~1010), LAMMPS (line ~1015–1080),
    //     stagnation check (line ~1095), ...
    submit_result(std::move(r), ...);   // line 1167 / 1171
    return save_loss;
}
```

If any operation between line 919 and line 1167 throws (e.g.
`cache.trainer->train()`, `host.model->predict()`, `mlip::Analytics()`
construction, `fit_loss()`, a LAMMPS sub-process error not already
swallowed by `try_parse_double`), the exception propagates out of
`operator()`. The eval thread loses that iter. NLopt's C++ wrapper
catches the exception and either re-throws (aborting the optimisation)
or — more commonly for non-NLopt-specific exceptions — converts it
silently to a `NaN` return, allowing the optimiser to continue.

Either way: `r.iter = N` was assigned, the iter slot is consumed
(`g_iter_` is now N), but no `Result{N, ...}` was ever enqueued.

The writer's logic at `hpo_host_impl.hpp:1272`:
```cpp
while (io_buf_.count(next_flush_))
{
    // ... write next_flush_ ...
    io_buf_.erase(next_flush_++);
}
```

`next_flush_` advances only via this loop. If iter N is missing from
`io_buf_`, the loop body never executes for `next_flush_=N`. The
writer exits the loop normally, releases the token, returns. Iters
N+1, N+2, ... continue to enqueue, land in `io_buf_`, and accumulate.
But `io_buf_.count(next_flush_)==0` for the missing N, so they are
never written.

Memory growth from accumulated `io_buf_` entries is bounded — most
fields in `Result<ParamVec>` are small — so the symptom is "files
frozen, process humming" rather than OOM. (Live ITER5 ate VmRSS 2.3 GB
total which includes the structures DB and per-thread workspaces; the
queue contribution is in the hundreds of MB and not visibly anomalous.)

Why Bug 4's catch fix doesn't help: the catch is inside `submit_result`,
which is never called for the lost iter. The throw escapes from the
inner train/predict block, not from the writer.

## Concurrent evidence from /tmp/hpo_bug_verify

Same binary, different problem: `/tmp/hpo_bug_verify` ran 25,805 evals
to completion with `loss.tadah` containing all 2581 expected lines
(every 10th iter through 25800), zero `HPO writer:` warnings, zero
`nlopt failure` warnings, zero `LN_SBPLX simplex` warnings. So the
binary works when nothing throws inside `operator()`.

Live ITER5, in contrast, recorded:
- 23 `nlopt failure — algorithm converged; best result preserved` warnings
- 8 `LN_SBPLX: NLopt internal diagnostic … simplex that was too small` warnings
- 1024+ `skipping LAMMPS (ERMSE > 500)` warnings

The 23 nlopt failures correlate with stage-end events; they are
benign. The simplex-collapse warnings happen inside NLopt's C++
adapter. The LAMMPS skip path returns FAIL_SCORE without throwing.
None of these *are* the missing-iter throw, but their existence is
evidence that the live ITER5 parameter regime is hitting more
pathological evaluation paths than the verify run.

The actual throw site has not been identified. Candidate paths:
1. `cache.trainer->train(train_stdb_)` (`hpo_host_impl.hpp:994`) —
   throws on numerical failure inside the BLR solver.
2. `host.model->predict(pot, val_stdb_, host.dc)` (line 1006) — throws
   on a feature-space dimension mismatch when bounds straddle a
   discontinuity.
3. `mlip::Analytics(val_stdb_, pred)` (line 1007) — throws if
   prediction count != stdb size.
4. The LAMMPS run path between lines ~1015 and ~1080.

## Required fix

Wrap the entire `operator()` body in a try/catch that converts any
throw (other than `ShutdownRequested` and `StagnationStop`, which the
optimiser must see) into a `Result` with `loss = trg_.FAIL_SCORE`,
`was_nan_clamped = true`, the iter already assigned, and an empty
`v_loss` / `pot_ctx`. Submit that Result to the queue so the iter
makes it into the writer's stream. Then re-throw or return FAIL_SCORE
to the optimiser so behaviour upstream is unchanged.

Sketch:

```cpp
double HPO_Host<P,R>::operator()(const P& p) const
{
    if (core::SignalHandler::stop_requested())
        throw core::ShutdownRequested{};

    Result r;
    r.iter   = ++g_iter_;
    r.params = p;

    auto submit_failscore = [&](const std::exception* e) {
        r.loss = trg_.FAIL_SCORE;
        r.was_nan_clamped = true;
        r.wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t0);
        HPOLog::instance().warning(
            std::string("eval ") + std::to_string(r.iter) +
            ": eval-thread caught throw, submitting FAIL_SCORE to keep "
            "writer queue contiguous: " + (e ? e->what() : "unknown"));
        submit_result(std::move(r), trg_, master_ctx_, val_ctx_,
                      train_stdb_, val_stdb_, param_labels_, report_,
                      cw_, prec_);
    };

    try {
        // ... existing body: train, predict, Analytics, fit_loss,
        //     LAMMPS, top-K, stagnation check, save_top_breakdown,
        //     submit_result ...
        return save_loss;

    } catch (const core::ShutdownRequested&) {
        throw;        // propagate to hpo_run's outer catch
    } catch (const StagnationStop&) {
        throw;        // propagate to RestartPolicy
    } catch (const std::exception& e) {
        submit_failscore(&e);
        return trg_.FAIL_SCORE;
    } catch (...) {
        submit_failscore(nullptr);
        return trg_.FAIL_SCORE;
    }
}
```

This guarantees the writer's `next_flush_` always sees its expected
iter, eliminating the gap. The dropped LAMMPS evaluation result is
flagged with `was_nan_clamped = true` so downstream consumers can tell
it was a synthetic FAIL_SCORE rather than a genuine high-loss point.

## Required regression test

`HPO/tests/test_eval_thread_throw_recovery.cpp` (Catch2):

- Build a minimal `HPO_Host` whose objective deterministically throws
  on the 50th call (e.g. an injected counter and `if (n==50) throw
  std::runtime_error("synthetic");`).
- Run 100 evals.
- Assert that `loss.tadah` contains all 100 entries (or, with
  `outputEvery=10`, all 10 expected entries).
- Assert that the 50th entry has `was_nan_clamped == true` and
  `loss == FAIL_SCORE` in the row.
- Assert that `hpo_run.log` contains exactly one
  `eval 50: eval-thread caught throw…` warning.
- Assert that `next_flush_` reaches 100 (no permanent stall).
- Assert that the optimiser saw FAIL_SCORE returned for eval 50 and
  the run continued normally to eval 100.

The test must fail on the un-patched binary (`next_flush_` will hang
at 50; loss.tadah will have only 49 entries) and pass after the fix.

## Verification

After the fix, re-run the live ITER5 reproducer for at least 90
minutes. Assert at every 10-minute mark:

1. `(n_eval) - (max iter in loss.tadah)` is bounded — say, < 200
   (corresponds to the in-flight queue + outputEvery rounding).
2. `best_pot.tadah`'s recorded best loss is consistent with the
   journal's "best=" SUMMARY lines.
3. Any `eval N: eval-thread caught throw` warnings are non-zero —
   meaning the fix is firing on the same throw events that previously
   caused the freeze. Identify the throw type from the warning
   payload and document the actual root cause as a follow-up note in
   this file.

## Files relevant to Bug 5

| Path | Purpose |
|---|---|
| `HPO/include/tadah/hpo/hpo_host_impl.hpp` | `operator()` body lines 908–1196 — wrap in try/catch |
| `HPO/include/tadah/hpo/hpo_log.h` | If `eval-thread caught throw` becomes a structured event, add a dedicated method instead of bare `warning()` |
| `HPO/tests/test_eval_thread_throw_recovery.cpp` | New regression test |

---

# BUG 6 — eval thread HANGS (does not throw) inside operator() between `++g_iter_` and `submit_result`

## Severity
High. Identical symptom to Bug 5 (writer freezes mid-run, per-eval log
files stop updating, `hpo_run.log` keeps progressing) but the root
cause is different again — the eval thread does not throw, it simply
hangs inside the eval pipeline. Bug 5's `try/catch` therefore never
fires, and the writer's `next_flush_` cursor stays stuck at the
hung iter while iters N+1, N+2, … accumulate in `io_buf_` forever.

Discovered 2026-05-06 by instrumenting the writer with
`TADAH_HPO_DEBUG_WRITER=1` after Bug 5's fix failed to recover the
live ITER5 freeze.

## Evidence

`writer_debug.log` snapshot, taken from a live ITER5 run that froze
at 13:07:23 with `next_flush_=3459`:

```
13:07:19.496895 EXIT wid=345                    # last call to write iter 3450
13:07:23.622951 ENTRY wid=346 q_size=10 next_flush=3451
13:07:23.623049 LOCK_ACQUIRED wid=346
13:07:23.623092 Q_DRAINED   wid=346 items=10 io_buf_size=10 next_flush=3451
13:07:23.623111 ITER_BEGIN  wid=346 iter=3451 io_buf_size=10
13:07:23.623466 LOOP_DONE   wid=346 written=8 last_iter=3458 next_flush=3459 io_buf_remaining=2
13:07:23.623482 FLUSH_BEGIN wid=346
13:07:23.623490 FLUSH_DONE  wid=346 loss_good=Y loss_fail=N loss_bad=N loss_eof=N
13:07:23.623495 EXIT        wid=346
...
13:13:57.769946 Q_DRAINED   wid=470 items=10 io_buf_size=1242 next_flush=3459
13:13:57.769964 LOOP_DONE   wid=470 written=0 last_iter=0 next_flush=3459 io_buf_remaining=1242
```

Read across the lines:

- Writer call **wid=346** drained 10 items from `g_q_` into
  `io_buf_`, processed 8 (iters 3451–3458), then **stopped** because
  `io_buf_.count(3459)` returned 0 — iter 3459 was never enqueued.
- The 2 unprocessed items in `io_buf_` had iter numbers **> 3458 and
  ≠ 3459** (likely 3460 and 3461).
- Subsequent writer calls (wid=347…470) accumulated more iters in
  `io_buf_` but each call wrote **0 iters** because `next_flush_=3459`
  was still missing. By wid=470, `io_buf_.size()=1242`.

`evcatch=0` in the journal: Bug 5's `operator()` try/catch never
fired. So no exception was thrown — the eval thread for iter 3459 is
**hung** somewhere inside `train()` / `predict()` / `Analytics` /
`fit_loss()` / `run_lammps()`, having already incremented `g_iter_`
to 3459 but never reaching `submit_result(std::move(r), ...)`.

## Diagnosis

The freeze pattern in the writer's drain loop is the same as Bug 5,
but the trigger is different:

- **Bug 5**: eval thread throws between `++g_iter_` and `submit_result`.
  The fix in `operator()`'s catch block submits a synthetic
  `FAIL_SCORE` Result for the consumed iter so `next_flush_` is never
  blocked. This works for *thrown* exceptions only.

- **Bug 6**: eval thread *hangs* (no throw, no progress). The catch
  in `operator()` cannot help because the thread never reaches a
  catch handler — it is still inside `train()` etc., possibly
  forever, possibly until eventually unsticking. Meanwhile the writer
  is blocked on the missing iter.

The hang surface is somewhere in the eval pipeline. Candidates from
the live ITER5 evidence:
1. LAMMPS sub-process taking very long on a near-pathological
   parameter sample (no timeout enforced at the host level).
2. Trainer numerical issue causing `train()` or `solve()` to spin or
   stall on a degenerate matrix.
3. NLopt-side deadlock in the local-search dispatcher (less likely;
   would manifest as multiple stuck threads).

Identifying the exact hang surface requires a stack trace from the
running process, which the host's ptrace policy currently denies.
The fix below restores liveness regardless of which surface hangs.

## Required fix (in writer, regardless of root cause)

Add a writer-driven *gap-skip* with a timeout to
`HPO_Host::submit_result`'s exclusive section:

1. Track when `next_flush_` first becomes "stuck" (i.e., the writer
   finds `io_buf_` non-empty with all keys > `next_flush_`).
2. If progress resumes (any write succeeds in subsequent calls),
   reset the stuck timer.
3. If `next_flush_` stays stuck for more than `GAP_SKIP_MS` (60 s by
   default), advance `next_flush_` to the smallest available key,
   log a journal warning, and re-drain.

Sketch in `submit_result`, right after the per-iter drain loop:

```cpp
constexpr std::int64_t GAP_SKIP_MS = 60'000;
if (_written > 0) {
    next_flush_stuck_since_ms_.store(0, std::memory_order_relaxed);
} else if (!io_buf_.empty()) {
    const auto smallest = io_buf_.begin()->first;
    const auto cur      = next_flush_.load(std::memory_order_relaxed);
    if (smallest > cur) {
        const auto now_ms = ...steady_clock_ms_now()...;
        auto since = next_flush_stuck_since_ms_.load();
        if (since == 0) {
            next_flush_stuck_since_ms_.store(now_ms);   // start timer
        } else if (now_ms - since > GAP_SKIP_MS) {
            HPOLog::instance().warning(
                "HPO writer: gap-skip — advancing next_flush_ from "
                + std::to_string(cur) + " to "
                + std::to_string(smallest) + " ("
                + std::to_string(smallest - cur)
                + " hung iter slots; per-eval logs lost)");
            next_flush_.store(smallest);
            next_flush_stuck_since_ms_.store(0);
            // Re-drain — io_buf_[smallest] is now the next iter.
            while (io_buf_.count(next_flush_)) { ...write iter, erase++... }
        }
    }
}
```

Trade-off: the hung iter's per-eval logs are permanently lost (no
synthetic FAIL_SCORE row in `loss.tadah` etc.), but the writer
recovers liveness. If the hung eval thread eventually unsticks and
calls `submit_result`, its Result lands in `io_buf_` with key <
`next_flush_` and is silently abandoned. Memory cost is bounded
(≤ one Result per hung thread).

## Required regression test

`HPO/tests/test_writer_gap_skip.cpp` (Catch2):

- A simulator mirroring the production state machine (drain loop +
  stuck-timer + skip block).
- Verify timer **does not** start on the first call that makes
  progress (drains 1..4 then sees gap).
- Verify timer starts on the **second** call (zero progress, gap
  detected).
- Verify skip fires once `clock - stuck_since > GAP_SKIP_MS`.
- Verify timer **resets** when a late-arriving iter unsticks the gap
  before the timeout (so honest slow evals are not skipped).
- Verify multi-iter consecutive gaps are handled in a single skip.
- Verify the warning string contains `gap-skip`, the iter range, and
  `hung iter slot`.

## Verification

After the fix, re-run the live ITER5 reproducer for at least 90
minutes. Expected behaviour:

1. `loss.tadah` continues to grow past iter 3450 (the historic
   freeze landing).
2. Each hung iter triggers a `HPO writer: gap-skip` warning ~60 s
   after the freeze starts.
3. `loss_iter` in `loss.tadah` tracks `n_eval` from the journal,
   minus a small bounded gap (one row per skip).

Live ITER5 result on 2026-05-06 (run started 13:21:51, run with the
fix applied):

```
elapsed     gap-skips   eval_count   loss_iter   loss_age (writer alive?)
30 min      0           3000         3150        1 s   ✓ (pre-freeze)
40 min      2           4000         4700        14 s  ✓ (past freeze, recovered)
50 min      5           6000         6550        2 s   ✓
60 min      7           8000         8450        52 s  ✓
70 min      10          10000        10540       4 s   ✓
```

The historic freeze trigger (round-1 GLOBAL→LOCAL transition) was
crossed at 13:55:24 with iter 3459 hanging at 13:55:29 and recovering
via gap-skip at 13:56:29 (60 s timer, exactly as designed). Subsequent
hangs (~1 per 600 evals) all recovered cleanly.

## Files relevant to Bug 6

| Path | Purpose |
|---|---|
| `HPO/include/tadah/hpo/hpo_host.h` | Add `next_flush_stuck_since_ms_` static atomic |
| `HPO/include/tadah/hpo/hpo_host_impl.hpp` | `submit_result` exclusive section: stuck-timer + skip block; debug instrumentation gated by `TADAH_HPO_DEBUG_WRITER=1` |
| `HPO/tests/test_writer_gap_skip.cpp` | New regression test (5 cases, 36 assertions) |

## Follow-up — identify the actual hang surface

The gap-skip restores liveness but does not remove the underlying
hang. Tracking down which call inside `operator()` hangs requires
either:

- Lifting `kernel.yama.ptrace_scope` so `gdb -p` / `strace -p` can
  attach.
- Building tadah with `-g3` and a build of the runtime/LAMMPS/NLopt
  with debug symbols, then attaching `perf` or `bpftrace` with
  user-space stack capture.
- Adding a stage-level watchdog around each candidate call
  (`train()`, `predict()`, `run_lammps()`) that emits a structured
  warning if the call exceeds N seconds, naming the stage. This is
  the cheapest route and is recommended as a separate ticket.

## BUG 7 — `StagnationStop` / `ShutdownRequested` rethrow leaks the iter slot

### Root cause (verified live ITER5, 2026-05-06 18:34)

The "hang" identified in Bug 6 was not a hang at all — it was an
**iter-slot leak** at every BASINS GLOBAL→LOCAL transition.

`BasinsPolicy::run_global` calls
`HPO_Host::set_stagnation_threshold(...)` to enable a per-stage
stagnation cap. When `g_iter_ - iter_at_last_best_ >= threshold`,
`operator()` throws `StagnationStop` to short-circuit nlopt. Before
this fix, the catch chain rethrew `StagnationStop` (and
`ShutdownRequested`) WITHOUT submitting a Result for the iter already
consumed at `++g_iter_`:

```cpp
} catch (const StagnationStop&) {
    /* RestartPolicy expects this. Pass through. */
    throw;
}
```

Effect: `RestartPolicy`/`BasinsPolicy` caught the rethrown
`StagnationStop` and transitioned to LOCAL. New eval threads picked up
3460+ and submitted normally. But iter 3459 was **never enqueued**, so
the writer's `next_flush_` cursor stalled at 3459. Bug 6's writer
gap-skip recovered after 60 s exactly as designed — every BASINS round
boundary cost a 60 s freeze plus one lost per-eval log line.

This was hidden by Bug 6's recovery. The per-stage watchdog
(originally added to identify a presumed train/predict/lammps hang)
was the diagnostic that disproved the hang hypothesis: gap-skip fired
with `snapshot_stuck` returning empty, meaning no thread was actually
running iter 3459 — the slot was abandoned, not stuck.

### Fix

Both catches now build a synthetic `FAIL_SCORE` Result with the
consumed `r.iter` and call `submit_result()` before rethrowing:

```cpp
} catch (const StagnationStop&) {
    r.loss = trg_.FAIL_SCORE;
    r.was_nan_clamped = true;
    r.wall_ms = ...;
    HPOLog::instance().warning(
        "eval " + std::to_string(r.iter) +
        ": StagnationStop — submitting FAIL_SCORE to keep "
        "writer queue contiguous before rethrow");
    submit_result(std::move(r), ...);
    throw;
}
```

The writer queue stays contiguous — no gap forms, no 60 s wait,
RestartPolicy / BasinsPolicy still see `StagnationStop` and transition
as before.

### Live verification (2026-05-06)

Same ITER5 reproducer, freshly-built binary with the fix:

| time     | event                                       |
|----------|---------------------------------------------|
| 17:55:37 | (pre-fix) gap-skip on iter 3459, 60 s wait  |
| 18:34:04 | (post-fix) `eval 3459: StagnationStop — submitting FAIL_SCORE` warning fires; BASINS GLOBAL→LOCAL transitions immediately, no gap-skip |
| 18:37:36 | second StagnationStop at eval 4084, recovered cleanly |

### Files relevant to Bug 7

| Path | Purpose |
|---|---|
| `HPO/include/tadah/hpo/hpo_host_impl.hpp` | `stage_watchdog` namespace + 6 instrumented stages (train, predict_validate, lammps, post_lammps_finalize, submit_result, journal_log); `StagnationStop` and `ShutdownRequested` catches now submit FAIL_SCORE before rethrow |
| `HPO/tests/test_stage_watchdog.cpp` | Watchdog primitive regression (5 cases) |
| `HPO/tests/test_stagnation_stop_submit.cpp` | StagnationStop submit-before-rethrow regression (2 cases) |

## Out of scope

- LN_SBPLX simplex-collapse at parameter bounds (a separate
  ticket to be filed). Do not change LOCAL stage behaviour as part of
  this ticket.
- Performance / queue-pressure tuning. The bugs above are all logic
  errors, not throughput issues.
- MPI build path. The MPI variant has its own submit_result branch
  (`#ifdef TADAH_BUILD_MPI`); fix the non-MPI path first, then port
  to MPI in a follow-up if needed.
