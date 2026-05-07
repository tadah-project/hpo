#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/run_policy.h>
#include <tadah/hpo/hpo.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/core/checkpoint.h>
#include <tadah/core/checkpoint_manager.h>
#include <tadah/core/context.h>
#include <tadah/core/blueprint_parser.h>
#include <tadah/core/signal_handler.h>
#include <tadah/mlip/structure_db.h>

#include <filesystem>
#include <set>
#include <unistd.h>   // getpid()
// #include <dlib/global_optimization/find_max_global.h>
#include <dlib/optimization.h>
#include <dlib/global_optimization.h>

#include <nlopt.hpp>

#include <random>
#include <variant>

namespace tadah::hpo {
using R = LammpsRunner;
using V = dlib::matrix<double,0,1>;

/* NOISE_CALIBRATION is gated by two predicates: a non-trivial knob
 * (>=2 evals) and a fresh start (no checkpoint to resume from). When
 * resuming, the prior run already calibrated noise — repeating it
 * would charge N more LAMMPS evals against the resumed counter and
 * pollute the per-eval logs. Pulled out of hpo_run() into a small free
 * function so the gating is unit-testable without invoking the full
 * optimiser; see HPO/tests/test_noise_calibration_resume.cpp. */
bool should_run_noise_calibration(bool resumed, std::size_t noise_n)
{
    return !resumed && noise_n >= 2;
}

int hpo_run(core::Context& ctx)
{
    /* ── Open run journal first ─────────────────────────────────────
     * This must precede HPOTargets construction so that diagnostics
     * emitted while parsing OPTIMIZER, OPTIM, and INIT_STRATEGY blocks
     * (NOISE_AUTO, "InitStrategy X produced N points", WARM-front
     * promotion, implicit-CONFIG suppression, …) reach both stdout
     * and the journal. The OPTIMISER section is written later, after
     * trg.spec has been finalised. */
    auto& log = HPOLog::instance();
    {
        std::string version = "unknown";
        try { version = core::BlueprintParser::getBlueprintTadah().version; }
        catch (...) {}
        std::string cwd = std::filesystem::current_path().string();
        log.open("hpo_run.log", version, cwd, static_cast<int>(getpid()));
    }

    /* Cooperative shutdown: install SIGINT/SIGTERM/SIGUSR2 handlers so
     * the BasinsPolicy and HPO_Host can drop a final checkpoint and
     * exit cleanly under ctrl+C or SLURM cancellation. */
    core::SignalHandler::install();

    HPOTargets trg(ctx, ctx.get<std::string>("HPOTARGET"));
    std::string val = ctx.get<std::string>("VALIDATION");

    /* ── Build param labels for the search space section ────────────── */
    std::vector<std::string> param_labels;
    for (const auto& kv : trg.optimap) {
        for (std::size_t i = 0; i < kv.second.opt.size(); ++i) {
            if (!kv.second.opt[i]) continue;
            std::string lbl = kv.first;
            if (kv.second.opt.size() > 1) lbl += "[" + std::to_string(i) + "]";
            param_labels.push_back(lbl);
        }
    }

    /* ── Build element string ────────────────────────────────────────── */
    std::string elem_str;
    try {
        auto elems = mlip::StructureDB::find_unique_elements(ctx);
        for (const auto& e : elems) {
            if (!elem_str.empty()) elem_str += ", ";
            elem_str += e.symbol;
        }
    } catch (...) { elem_str = "(could not determine)"; }

    /* ── Write setup sections ───────────────────────────────────────── */
    log.write_files_section(ctx.get<std::string>("HPOTARGET"), val, trg);

    log.write_optimizer_section(trg.spec, param_labels);

    /* Build HPO_Host (loads train + val structures, applies load transforms) */
    HPO_Host<V,R> host(ctx, trg, val);
    log.write_potential_section(ctx, host.train_size(), host.val_size(), elem_str);
    {
        const auto& tr = host.load_transform_report();
        const auto& vr = host.val_load_transform_report();
        const std::size_t n_train_loaded =
            host.train_size() + tr.n_dropped_energy + tr.n_dropped_force;
        const std::size_t n_val_loaded =
            host.val_size() + vr.n_dropped_energy + vr.n_dropped_force;
        log.write_load_transforms_section(tr, n_train_loaded,
                                          vr, n_val_loaded);
    }
    log.write_loss_section(trg);

    const OptimizerSpec& sp = trg.spec;

    /* ── Run optimiser via the configured RunPolicy ─────────────────── */

    /* Build the decorator stack from sp's MULTISTART / PRESTAGE block /
     * RESTART knobs. With no knobs set, this collapses to BasePolicy.
     * Targets pointer flows through to RestartPolicy for fresh-init
     * generation on stagnation.
     *
     * BasinsPolicy registers itself + its BasinPool with
     * CheckpointManager from its constructor, so build_run_policy()
     * must run *before* try_load(). */
    auto policy = build_run_policy<V,R>(sp, &trg);

    /* ── Checkpoint subsystem wiring (LIB BASINS only for now) ──────────
     *
     * Order is important: try_load() must run BEFORE NOISE_CALIBRATION
     * so that a resumed run does not re-run the calibration evals (each
     * one is a real LAMMPS evaluation that pollutes the eval counter,
     * per-eval logs, and the best-so-far comparator). The host is
     * registered alongside HPOLog so its load() restores g_best_ et al.
     * before any further evals happen. */
    auto& cpm = core::CheckpointManager::instance();
    bool resumed = false;
    if (sp.strategy_type == StrategyType::BASINS) {
        cpm.set_path("checkpoint.tadah");
        if (sp.basins.checkpoint_seconds > 0.0)
            cpm.set_period_seconds(sp.basins.checkpoint_seconds);
        if (sp.basins.checkpoint_evals > 0)
            cpm.set_period_evals(sp.basins.checkpoint_evals);
        /* HPOLog persists journal counters so eval numbering and
         * best-so-far stay coherent across the resume boundary. */
        cpm.register_(&log);
        /* HPO_Host persists the static best-so-far comparator
         * (g_best_, g_best_iter_, iter_at_last_best_, g_iter_) so the
         * first post-resume eval is judged against the prior best, not
         * +infinity. Without this, every resume produces a spurious
         * "NEW BEST" event that overwrites best_pot.tadah. */
        cpm.register_(&host);
        resumed = cpm.try_load();
        if (resumed) {
            log.info("BASINS: resumed from checkpoint " + cpm.path() +
                     " (n_eval=" + std::to_string(log.n_eval()) +
                     ", best_loss=" +
                     std::to_string(log.best_loss()) + ")");
            /* Belt-and-suspenders: HPO_Host::load() already restores
             * g_iter_; the floor call below is idempotent and keeps
             * the eval counter ≥ HPOLog's n_eval even if the
             * checkpoint predates the HPO_HOST section. */
            HPO_Host<V,R>::set_g_iter_floor(log.n_eval());
        }
    }

    /* ── Noise calibration (NOISE_CALIBRATION N): repeat the seed point
     * N times before the optimiser starts, surface the loss spread, and
     * suggest a realistic FTOL_REL. Each repeat is a real evaluation —
     * it counts toward the global eval counter, the best-so-far tracker,
     * and the per-eval log files (loss.tadah / params.tadah / ...).
     * The overhead is N - 1 extra wall-clock evals; the user opted in
     * by setting NOISE_CALIBRATION, so we do not also charge it against
     * MAXEVAL (the optimiser still gets its full budget).
     *
     * Skipped on resume: the prior run already calibrated noise, the
     * suggested FTOL_REL is preserved by the BASINS state machine, and
     * re-running would charge N more LAMMPS evals against the resumed
     * counter. If the user changes the seed point or dataset between
     * runs, they should delete checkpoint.tadah to recalibrate. */
    if (should_run_noise_calibration(resumed, sp.noise.calibration_n)) {
        const std::size_t N = sp.noise.calibration_n;
        if (sp.optim_init.empty()) {
            log.warning("NOISE_CALIBRATION requested but no starting "
                        "point is available; skipping.");
        } else {
            std::cout << "[HPO] NOISE_CALIBRATION: running " << N
                      << " evals at the seed point to measure objective noise\n";
            log.info("NOISE_CALIBRATION: running " + std::to_string(N) +
                     " evals at the seed point");

            const auto seed_real = sp.optim_init.front();
            V seed_vec;
            seed_vec.set_size(static_cast<long>(seed_real.size()));
            for (long i = 0; i < seed_vec.size(); ++i)
                seed_vec(i) = seed_real[static_cast<std::size_t>(i)];

            std::vector<double> losses;
            losses.reserve(N);
            for (std::size_t i = 0; i < N; ++i)
                losses.push_back(host(seed_vec));

            /* Suggest FTOL_REL = 10 * relative_spread, floored at 1e-6
             * (no point recommending tolerances below double precision
             * times a sensible fudge factor). NLopt's FTOL_REL test
             * fires when |Δf| < ftol_rel * |f|, so 10x the observed
             * relative spread gives the local search room to terminate
             * once it crosses the noise floor instead of chasing it. */
            double mn = losses.front(), mx = losses.front(), sum = 0.0;
            for (double v : losses) {
                if (v < mn) mn = v;
                if (v > mx) mx = v;
                sum += v;
            }
            const double mean = sum / static_cast<double>(losses.size());
            const double rel_spread =
                (mean != 0.0) ? (mx - mn) / std::fabs(mean) : 0.0;
            const double suggested_ftol = std::max(10.0 * rel_spread, 1.0e-6);

            log.write_noise_calibration_section(losses, suggested_ftol);
            log.set_noise_suggested_ftol(suggested_ftol);

            std::ostringstream oss;
            oss << std::scientific << std::setprecision(3);
            oss << "NOISE_CALIBRATION: rel_spread=" << rel_spread
                << "  suggested FTOL_REL=" << suggested_ftol
                << "  (mean=" << mean << ", min=" << mn << ", max=" << mx << ")";
            std::cout << "[HPO] " << oss.str() << "\n";
            log.info(oss.str());

            /* C3: plumb the noise floor into BasinsPolicy so the (opt-in)
             * ladder clamp can drop sub-noise rungs. dynamic_cast is local
             * and safe — BasinsPolicy is the only RunPolicy not wrapped
             * by the decorator stack (run_policy.h:434-436). Using the
             * cast keeps RunPolicy's interface free of noise-specific
             * methods, per the ticket's "Do not change the RunPolicy
             * decorator pattern" rule. */
            if (auto* bp = dynamic_cast<BasinsPolicy<V,R>*>(policy.get())) {
                bp->set_noise_floor(suggested_ftol);
            }
        }
    } else if (resumed) {
        /* C3 resume path: NOISE_CALIBRATION did not re-run (gated on
         * !resumed), but the prior run's suggested_ftol was persisted in
         * the [HPOLOG] section. Re-plumb it into BasinsPolicy so the
         * (opt-in) clamp keeps applying across resumes. */
        if (auto suggested = log.noise_suggested_ftol()) {
            if (auto* bp = dynamic_cast<BasinsPolicy<V,R>*>(policy.get())) {
                bp->set_noise_floor(*suggested);
            }
        }
    }

    log.start_progress();

    /* quick info */
    std::cout<<"[HPO] "<<sp.lib<<":"<<sp.algo<<"   dim="<<sp.low.size()<<"\n";

    try {
        policy->run(sp, host);

        /* Drain any in-flight evals, then write the journal trailer once.
         * Passing an empty status_str makes HPOLog::finish() fall back to
         * the most recent HPOLog::set_status() recorded by the inner
         * driver (e.g. NLopt's "MAXEVAL_REACHED"), or "COMPLETED" if none
         * was set. This is what makes MULTISTART + NLopt log correctly:
         * each per-start run sets its status, the last one wins, and the
         * journal stays open until we get here. */
        host.flush_remaining();
        log.finish();
    } catch (const core::ShutdownRequested& e) {
        /* SIGINT/SIGTERM/SIGUSR2 — the policy already attempted to
         * commit a checkpoint on the way out. Drain the I/O queue,
         * close the journal cleanly with the signal reason. */
        log.warning(std::string("Shutdown requested: ") +
                    core::SignalHandler::reason());
        host.flush_remaining();
        log.finish(std::string("SHUTDOWN (") + core::SignalHandler::reason() + ")");
        /* Final flush of the checkpoint after the queue drain so the
         * persisted n_eval matches what the eval files contain. */
        if (sp.strategy_type == StrategyType::BASINS) {
            try { cpm.save_now(); } catch (...) { /* best effort */ }
        }
    } catch (const std::exception& e) {
        const std::string err = std::string("STOPPED with error: ") + e.what();
        log.warning(std::string("Optimiser threw: ") + e.what());
        host.flush_remaining();
        log.finish(err);
        throw;
    }

    return 0;
}
} // namespace tadah::hpo
