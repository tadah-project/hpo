#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/run_policy.h>
#include <tadah/hpo/hpo.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/script_executor.h>
#include <tadah/hpo/pc_worker_pool.h>   // PC_ISOLATION worker-process pool
#include <tadah/hpo/iter_health.h>
#include <tadah/hpo/rho_gates.h>
#include <tadah/hpo/startup_validation.h>
#include <tadah/core/checkpoint.h>
#include <tadah/core/checkpoint_manager.h>
#include <tadah/core/context.h>
#include <tadah/core/blueprint_parser.h>
#include <tadah/core/signal_handler.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/preflight_audit.h>
#include <tadah/mlip/parallel.h>   // set_parallel_backend (Stage 2.3)
#include <tadah/models/descriptors/dm/embf0.h>

#include <algorithm>  // std::max
#include <cstdlib>    // getenv()
#include <sstream>    // std::ostringstream (PC_SCRIPT_TIMEOUT diagnostic)
#include <functional> // std::function
#include <filesystem>
#include <set>
#include <unistd.h>   // getpid()
// #include <dlib/global_optimization/find_max_global.h>
#include <dlib/optimization.h>
#include <dlib/global_optimization.h>

#include <nlopt.hpp>

#include <optional>
#include <random>
#include <variant>

namespace tadah::hpo {
using R = LammpsRunner;
using V = dlib::matrix<double,0,1>;

/* NOISE_CALIBRATION is gated by two predicates: a non-trivial knob
 * (>=2 evals) and a fresh start (no checkpoint to resume from). When
 * resuming, the prior run already calibrated noise -- repeating it
 * would charge N more LAMMPS evals against the resumed counter and
 * pollute the per-eval logs. Pulled out of hpo_run() into a small free
 * function so the gating is unit-testable without invoking the full
 * optimiser; see HPO/tests/test_noise_calibration_resume.cpp. */
bool should_run_noise_calibration(bool resumed, std::size_t noise_n)
{
    return !resumed && noise_n >= 2;
}

bool hpo_mpi_dispatch_supported(int world_size,
                                const OptimizerSpec& spec) noexcept
{
    if (world_size <= 1) return true;
    return spec.strategy_type == StrategyType::PLAIN &&
           spec.lib == "TADAH" &&
           (spec.algo == "ANNEAL_PT" || spec.algo == "ANNEAL_PA");
}

int hpo_run(core::Context& ctx)
{
    /* -- Open run journal first -------------------------------------
     * This must precede HPOTargets construction so that diagnostics
     * emitted while parsing OPTIMIZER, OPTIM, and INIT_STRATEGY blocks
     * (NOISE_AUTO, "InitStrategy X produced N points", WARM-front
     * promotion, implicit-CONFIG suppression, ...) reach both stdout
     * and the journal. The OPTIMISER section is written later, after
     * trg.spec has been finalised. */
    auto& log = HPOLog::instance();
    {
        std::string version = "unknown";
        try { version = core::BlueprintParser::getBlueprintTadah().version; }
        catch (...) {}
        std::string cwd = std::filesystem::current_path().string();
        /* Rank-suffixed on MPI ranks > 0 (no-op otherwise) so multi-rank
         * runs sharing one working directory keep separate journals. */
        log.open("hpo_run.log" + hpo_rank_suffix(), version, cwd,
                 static_cast<int>(getpid()));
    }

#ifdef TADAH_ENABLE_OPENMP
    {
        /* Under the shared-pool model (Option A) OpenMP no longer drives the
         * pc_lammps fan-out -- THREADS does (see the ScriptExecutor log below).
         * OMP_NUM_THREADS now controls only the residual OpenMP kernels
         * (neighbour-list build, data-stats); journal its width for the record. */
        const char* env_omp = std::getenv("OMP_NUM_THREADS");
        log.info("OpenMP: max threads = " +
                 std::to_string(omp_get_max_threads()) +
                 "  (OMP_NUM_THREADS=" +
                 (env_omp && env_omp[0] ? std::string(env_omp) : "unset") +
                 "); drives only the residual OpenMP kernels (NN lists, "
                 "data-stats), NOT the pc_lammps fan-out");
        /* Journal the wait policy. The shared pool's idle workers block on a
         * futex (0 CPU), so OMP_WAIT_POLICY no longer governs fan-out idle time
         * -- it now only affects those residual OMP kernels, which are short
         * setup steps. 'passive' stays the safe default on a shared node (idle
         * OMP threads sleep between kernels); the pool is unaffected either way. */
        const char* env_wait = std::getenv("OMP_WAIT_POLICY");
        log.info(std::string("OpenMP: OMP_WAIT_POLICY=") +
                 (env_wait && env_wait[0] ? env_wait : "unset (runtime default)") +
                 "; affects only the residual OMP kernels now (pool idles on a "
                 "futex); prefer 'passive' on shared nodes");
    }
#endif

    /* Cooperative shutdown: install SIGINT/SIGTERM/SIGUSR2 handlers so
     * the BasinsPolicy and HPO_Host can drop a final checkpoint and
     * exit cleanly under ctrl+C or SLURM cancellation. */
    core::SignalHandler::install();

    HPOTargets trg(ctx, ctx.get<std::string>("HPOTARGET"));

    /* This is more than a dispatch convenience: HPO_Host's live best/HOT
     * state is intentionally rank-local while best_pot.tadah is the global
     * cross-rank artifact. PT/PA do not register checkpoint state, so a rank
     * can safely keep its local incumbent when another rank wins canonical
     * publication. Allowing a checkpointed/decorated strategy here would mix
     * those ownership domains on resume. Keep size-one MPI launches identical
     * to serial builds. */
    const int hpo_world = hpo_size();
    if (!hpo_mpi_dispatch_supported(hpo_world, trg.spec)) {
        throw std::runtime_error(
            "multi-rank HPO supports only STRATEGY TYPE PLAIN with "
            "LIB TADAH and ALGO ANNEAL_PT or ANNEAL_PA; got " +
            trg.spec.lib + ":" + trg.spec.algo +
            " on " + std::to_string(hpo_world) + " ranks");
    }

    /* Parallelisation model (Option A): THREADS is the TOTAL width of the shared
     * worker pool -- the number of threads that actually run pc_lammps scripts
     * AND MLIP train/predict, i.e. the per-rank CPU-core budget. ONE pool serves
     * serial ANNEAL (SA) and every ANNEAL_PT/PA replica; idle workers block on a
     * futex (0 CPU). THREADS no longer multiplies OMP_NUM_THREADS -- the pool IS
     * the parallelism, so `exec_width = THREADS`. OMP_NUM_THREADS independently
     * controls the residual OpenMP kernels (neighbour-list build, data-stats) via
     * the OpenMP runtime; it does NOT size this pool. (PT/PA additionally cap the
     * number of *concurrent* replicas at min(THREADS, REPLICAS) -- the ReplicaPool
     * in algorithms_p{t,a}.h -- which is enough to keep the THREADS-wide pool fed
     * since one replica's eval alone submits many more tasks than THREADS.) */
    {
        const int exec_width =
            std::max<int>(1, static_cast<int>(trg.spec.num_threads.value_or(1)));
        init_script_executor(exec_width);
        /* Give the journal the true pool width -- the honest denominator
         * for the periodic pc_lammps pool-utilisation figure. */
        log.set_fanout_width(exec_width);
        /* Stage 2.3: route the MLIP train/predict kernels (design-matrix build,
         * descriptor calc, predict) onto the SAME pool. This drops the nested
         * per-replica OpenMP teams those kernels used to spawn and lets one
         * replica's train/predict overlap another's fan-out. The kernels write
         * disjoint output slots, so results are unchanged (verified by the
         * determinism gate). Installed once here, before any optimiser eval. */
        tadah::mlip::set_parallel_backend(
            [](std::size_t n, const std::function<void(std::size_t)>& body) {
                script_executor().run_batch(
                    n, [&](std::size_t i, int /*worker_id*/) { body(i); });
            });
#ifdef TADAH_ENABLE_OPENMP
        const int omp_w = std::max(1, omp_get_max_threads());
#else
        int omp_w = 1;
        if (const char* e = std::getenv("OMP_NUM_THREADS")) {
            const int v = std::atoi(e); if (v > 0) omp_w = v;
        }
#endif
        log.info("ScriptExecutor: " + std::to_string(exec_width) +
                 " workers = THREADS (total pool: fan-out + train/predict); "
                 "OMP_NUM_THREADS=" + std::to_string(omp_w) +
                 " drives only the residual OpenMP kernels (NN lists, data-stats)");
        /* Migration nudge: pre-Option-A, OMP_NUM_THREADS widened the fan-out, so
         * a config with no THREADS line still got OMP-wide parallelism. Now the
         * pool is THREADS alone, so that same config runs the fan-out on ONE
         * worker. Warn once so users know to set THREADS for concurrency. */
        if (!trg.spec.num_threads && omp_w > 1)
            log.warning(
                "THREADS not set: the pc_lammps fan-out and MLIP train/predict "
                "run on a SINGLE worker. Under the shared-pool model "
                "OMP_NUM_THREADS (=" + std::to_string(omp_w) + ") no longer "
                "widens the fan-out -- set THREADS <n> in the STRATEGY block to "
                "run scripts concurrently.");

        /* ---- PC_ISOLATION: pc_lammps execution mode --------------------
         * PROCESS = persistent worker-process pool (one child per fan-out
         * slot; LAMMPS globals unshared -- the free(): invalid pointer race
         * class is structurally gone). THREAD = in-process (pre-existing
         * behaviour; kept one production cycle as a deprecated fallback).
         * Env override TADAH_PC_ISOLATION lets the A/B and determinism
         * harnesses switch modes without touching configs. Design:
         * INVESTIGATIONS/PC_ISOLATION_process_pool.md */
        std::string pc_mode = trg.PC_ISOLATION;
        bool pc_mode_explicit = trg.PC_ISOLATION_set;
        if (const char* e = std::getenv("TADAH_PC_ISOLATION")) {
            if (*e) { pc_mode = e; pc_mode_explicit = true; }
        }
        double pc_wtimeout = trg.PC_WORKER_TIMEOUT;
        if (const char* e = std::getenv("TADAH_PC_WORKER_TIMEOUT")) {
            if (*e) pc_wtimeout = std::atof(e);
        }
        if (pc_mode != "THREAD" && pc_mode != "PROCESS")
            throw std::runtime_error(
                "PC_ISOLATION must be THREAD or PROCESS, got '" + pc_mode + "'");
#if defined(TADAH_HPO_MPI) || defined(TADAH_BUILD_MPI)
        if (pc_mode == "PROCESS") {
            /* A fork+exec'd child of an MPI parent must never touch MPI
             * (LAMMPS instances need an initialized MPI in these builds).
             * EXPLICIT requests hard-error; the serial-build DEFAULT of
             * PROCESS quietly falls back to THREAD here. */
            if (pc_mode_explicit)
                throw std::runtime_error(
                    "PC_ISOLATION PROCESS is not supported in MPI builds "
                    "(TADAH_HPO_MPI/TADAH_BUILD_MPI): the worker processes "
                    "cannot initialize MPI. Use the serial build, or "
                    "PC_ISOLATION THREAD.");
            pc_mode = "THREAD";
            log.info("PC_ISOLATION: THREAD (MPI build -- the PROCESS worker "
                     "pool is serial-build only)");
        }
#endif
        (void)pc_mode_explicit;
        if (pc_mode == "PROCESS") {
#if defined(TADAH_HPO_MPI) || defined(TADAH_BUILD_MPI)
            /* unreachable: handled above */
#else
            if (!trg.LS.empty()) {
                /* Per-slot ScriptSpec tables: slots differ only in the
                 * .t<slot> -log/-screen suffix (make_worker_invar -- the
                 * same helper the THREAD path applies per call). */
                std::vector<std::vector<PcScriptSpec>> per_slot(exec_width);
                for (int s = 0; s < exec_width; ++s) {
                    per_slot[s].reserve(trg.LS.size());
                    for (const auto& lsim : trg.LS) {
                        PcScriptSpec sp;
                        sp.script       = lsim.script;
                        sp.script_lines = lsim.script_lines;
                        sp.invar_final  = make_worker_invar(
                            lsim.invar, hpo_rank_suffix(), s);
                        sp.outvar_names = lsim.n_outvar;
                        for (const auto& hv : lsim.hotvars)
                            sp.hotvar_names.push_back(hv.name);
                        for (const auto& fv : lsim.failvars)
                            sp.failvar_names.push_back(fv.name);
                        per_slot[s].push_back(std::move(sp));
                    }
                }
                pc_worker_pool_init(exec_width, std::move(per_slot),
                                    pc_wtimeout);
                log.info("PC_ISOLATION: PROCESS -- pc_lammps tasks execute in "
                         + std::to_string(exec_width)
                         + " persistent worker processes (LAMMPS globals "
                           "unshared); MLIP train/predict stays in-process");
            } else {
                log.info("PC_ISOLATION: PROCESS requested but no PC_LAMMPS "
                         "sims are configured -- nothing to isolate");
            }
#endif
        } else {
#if !defined(TADAH_HPO_MPI) && !defined(TADAH_BUILD_MPI)
            /* Serial build running THREAD => it was explicitly requested
             * (the default is PROCESS): always flag the deprecated mode. */
            log.warning("PC_ISOLATION: THREAD (deprecated fallback) -- "
                        "concurrent in-process LAMMPS instances share process "
                        "globals (the free(): invalid pointer race class); "
                        "PROCESS is the supported mode. THREAD will be "
                        "removed after one production cycle.");
#endif
        }

        /* ---- PC_SCRIPT_TIMEOUT (legacy alias: MAXTIME): per-pc_script soft
         * wall-clock limit, injected as LAMMPS `timer timeout` after each
         * clear. Distinct from PC_WORKER_TIMEOUT (the outer per-worker-task
         * hard deadline) and from the OPTIMIZER-block MAXTIME (whole-run
         * stop). No startup diagnostic existed for it before -- print the
         * effective value, which keyword supplied it, and the hard deadline. */
        {
            const std::string src = trg.MAX_TIME_key.empty()
                ? "default" : trg.MAX_TIME_key;
            const std::string soft = (trg.MAX_TIME == "off")
                ? "off (no per-script limit)" : (trg.MAX_TIME + " s");
            std::string hard;
            if (pc_mode == "PROCESS") {
                if (pc_wtimeout > 0.0) {
                    std::ostringstream w; w << pc_wtimeout;
                    hard = w.str() + " s (PC_WORKER_TIMEOUT)";
                } else if (trg.MAX_TIME == "off") {
                    hard = "none";
                } else {
                    hard = "derived max(3x, x+120) s from the per-script limit";
                }
            } else {
                hard = "n/a (THREAD mode)";
            }
            log.info("PC_SCRIPT_TIMEOUT: " + soft + " (from " + src +
                     ") -- soft LAMMPS `timer timeout`, re-armed each "
                     "run-section; hard per-worker-task deadline: " + hard);
            if (trg.MAX_TIME_seen_new && trg.MAX_TIME_seen_legacy)
                log.warning("both PC_SCRIPT_TIMEOUT and legacy MAXTIME were "
                            "given; the later line wins (effective: " +
                            trg.MAX_TIME + " s from " + src + ")");
            else if (trg.MAX_TIME_seen_legacy)
                log.info("MAXTIME is a legacy alias for PC_SCRIPT_TIMEOUT "
                         "(per-pc_script timeout, seconds); prefer "
                         "PC_SCRIPT_TIMEOUT in new configs");
        }

        /* ---- FAILSCORE_GRADING: graded ("smooth") failure losses --------
         * ON: every failed eval returns a banded, monotone value inside
         * [0.999*FAILSCORE, FAILSCORE) instead of the constant plateau.
         * Env override TADAH_FAILSCORE_GRADING=ON|OFF (harness A/B).
         * Design: INVESTIGATIONS/SMOOTH_FAILSCORE_analysis.md */
        if (const char* e = std::getenv("TADAH_FAILSCORE_GRADING")) {
            if (*e) {
                const std::string v(e);
                if (v == "ON")       trg.FAILSCORE_GRADING = true;
                else if (v == "OFF") trg.FAILSCORE_GRADING = false;
                else throw std::runtime_error(
                        "TADAH_FAILSCORE_GRADING must be ON or OFF");
            }
        }
        if (trg.FAILSCORE_GRADING) {
            /* The band arithmetic needs a finite FAILSCORE: with the legacy
             * DBL_MAX default, 0.999*FS and FS are not meaningfully ordered
             * in loss space and L2 of the substituted value overflows. */
            if (!std::isfinite(trg.FAIL_SCORE) ||
                trg.FAIL_SCORE >= 1.0e300 || trg.FAIL_SCORE <= 0.0)
                throw std::runtime_error(
                    "FAILSCORE_GRADING ON requires a finite positive "
                    "FAILSCORE (e.g. `FAILSCORE 1e9`); the default DBL_MAX "
                    "sentinel cannot be banded.");
            if (!graded_failscore_is_ordered(trg.FAIL_SCORE))
                throw std::runtime_error(
                    "FAILSCORE_GRADING ON requires a FAILSCORE large enough "
                    "to represent four strictly ordered grading bands.");
            std::string probe_note = "no PC_FAILPROBE (bands C/D constant)";
            if (trg.failprobe_idx >= 0) {
                probe_note = "PC_FAILPROBE '" +
                    trg.LS[static_cast<std::size_t>(trg.failprobe_idx)].script +
                    "' S=" + trg.failprobe_svar;
                if (std::isfinite(trg.failprobe_gate))
                    probe_note += " gate=" + std::to_string(trg.failprobe_gate);
                else
                    probe_note += " gate=off";
            }
            log.info("FAILSCORE_GRADING: ON -- failed evals graded in "
                     "[0.999*FS, FS), bands A-D (see docs); " + probe_note);
        } else if (trg.failprobe_idx >= 0) {
            log.warning("PC_FAILPROBE is defined but FAILSCORE_GRADING is "
                        "OFF -- the probe runs as a plain diagnostic sim; "
                        "no grading, no gate.");
        }

        /* ---- BUMPGATE: bump-support enforcement mode --------------------
         * CANDIDATE (default): full OPTIM boxes; infeasible (c, eta)
         * combinations are rejected per evaluation as graded band-D
         * failures. STRICT: legacy startup box-corner hard-error.
         * Env override TADAH_BUMPGATE=CANDIDATE|STRICT (harness A/B). */
        if (const char* e = std::getenv("TADAH_BUMPGATE")) {
            if (*e) {
                const std::string v(e);
                if (v == "STRICT")         trg.BUMPGATE_STRICT = true;
                else if (v == "CANDIDATE") trg.BUMPGATE_STRICT = false;
                else throw std::runtime_error(
                        "TADAH_BUMPGATE must be CANDIDATE or STRICT");
            }
        }
        if (trg.BUMPGATE_STRICT)
            log.info("BUMPGATE: STRICT -- OPTIM boxes on hard-cutoff bump "
                     "channels must be feasible at the worst corner "
                     "(legacy startup gate); no per-candidate rejection.");
    }

    /* --validation is optional: PC_OBSERVABLE / PC_LAMMPS can carry the
     * loss on their own; validate_hpo_setup below rejects setups where
     * validation-set metrics are requested without a validation file. */
    std::string val = ctx.contains("VALIDATION")
        ? ctx.get<std::string>("VALIDATION") : "";

    /* Fail fast on inconsistent weight-pinning / loss-source setups
     * before any data is loaded or the optimiser starts. */
    const bool has_train_db =
        ctx.contains("DBFILE") && ctx.size("DBFILE") > 0;
    validate_hpo_setup(ctx, trg, has_train_db, !val.empty());

    /* -- Param labels for the search space section: built by the
     * HPOTargets constructor (canonical flattened OPTIM order). ------- */
    const std::vector<std::string>& param_labels = trg.spec.parameter_labels;

    /* -- Build element string ------------------------------------------
     * From the training data when present, otherwise from the explicit
     * ATOM key (mandatory in no-training-DBFILE runs). Guarded because
     * find_unique_elements calls ctx("DBFILE"), and Context::operator()
     * logs an ERROR line before throwing. */
    std::string elem_str;
    try {
        if (has_train_db) {
            auto elems = mlip::StructureDB::find_unique_elements(ctx);
            for (const auto& e : elems) {
                if (!elem_str.empty()) elem_str += ", ";
                elem_str += e.symbol;
            }
        } else if (ctx.contains("ATOM")) {
            for (const auto& s : ctx("ATOM")) {
                if (!elem_str.empty()) elem_str += ", ";
                elem_str += s;
            }
        }
    } catch (...) { elem_str.clear(); }
    if (elem_str.empty()) elem_str = "(could not determine)";

    /* -- Write setup sections ----------------------------------------- */
    log.write_files_section(ctx.get<std::string>("HPOTARGET"), val, trg);

    log.write_optimizer_section(trg.spec, param_labels);

    /* Machine-readable search-space sidecars next to the journal:
     * hpo_search_space.tsv (one row per dim) always; hpo_optim_resolved.hpo
     * (a paste-ready explicit OPTIM block) when any box was derived by an
     * AUTO rule, so a later run can pin exactly what this one searched. */
    trg.write_search_space_files(".");

    /* Build HPO_Host (loads train + val structures, applies load transforms) */
    HPO_Host<V,R> host(ctx, trg, val);
    /* EMBF0 (DM_REAM only): the -c potential's raw F(0) for POTENTIAL SETUP,
     * and the RESULT line's start/best values. Reads stored values only. */
    const bool has_ream = models::context_has_ream(ctx);
    const bool embf0_zero =
        has_ream &&
        models::stored_embf0(ctx).value_or(models::EmbF0Mode::Zero) ==
            models::EmbF0Mode::Zero;
    std::optional<RawEmbeddingF0> embf0_seed;
    if (has_ream) embf0_seed = derive_raw_embedding_f0(ctx);
    log.write_potential_section(ctx, host.train_size(), host.val_size(), elem_str,
                                embf0_seed ? &*embf0_seed : nullptr);
    if (has_ream) log.arm_embf0(embf0_zero, host.embf0_diagnostic());
    /* DM_REAM density report (a bare DM_REAM many-body term): the -c
     * potential's and each WARM seed's rho_raw(r) -- and rhobar(a) under
     * RHOBARGATE -- in POTENTIAL SETUP; RESULT repeats it for the best.
     * Stored values only; never throws. */
    if (detail::context_is_bare_ream(ctx))
        log.write_density_report(detail::startup_density_report(ctx, trg),
                                 trg.RHOBARGATE.has_value());
    {
        const auto& tr = host.load_transform_report();
        const auto& vr = host.val_load_transform_report();
        const std::size_t n_train_loaded =
            host.train_size() + tr.n_dropped_energy + tr.n_dropped_force;
        const std::size_t n_val_loaded =
            host.val_size() + vr.n_dropped_energy + vr.n_dropped_force;
        log.write_load_transforms_section(tr, n_train_loaded,
                                          vr, n_val_loaded);

        /* EMBF0 ZERO fixes the DM_REAM isolated-atom energy at 0: fitting
         * absolute reference energies then needs another per-atom constant.
         * n_intercept_columns > 0, from stored values (no BIAS default
         * inserted). */
        const auto bias = ctx.stored_value("BIAS");
        const bool intercepts =
            bias && *bias == "true" && ctx.stored_value("ATOM").has_value();
        // The stored keys as well as the reports: on the MPI host the
        // training report is not filled.
        const bool eshift_keys =
            ctx.has_stored("ESHIFT") ||
            ctx.stored_value("ESHIFT_ATOM") == std::optional<std::string>("true") ||
            ctx.stored_value("ESHIFT_DBATOM") == std::optional<std::string>("true");
        if ((has_train_db || !val.empty()) && has_ream && embf0_zero &&
            !intercepts && !eshift_keys && tr.eshift.empty() &&
            vr.eshift.empty())
            log.hint("EMBF0 ZERO fixes the DM_REAM isolated-atom energy at 0 "
                     "and the model has no other per-atom constant (BIAS "
                     "false, no ESHIFT). If reference energies are absolute "
                     "(e.g. DFT totals), set ESHIFT_ATOM, ESHIFT_DBATOM or "
                     "ESHIFT (compatible with EAMFAST). BIAS true also adds "
                     "a per-atom constant, but its intercept column precedes "
                     "the descriptor weights (FIXINDEX/FIXWEIGHT indices "
                     "shift) and it disables EAMFAST.");
    }
    log.write_loss_section(trg);

    /* -- IterHealthMonitor mode (Phase 6) ----------------------------- */
    {
        HealthLogMode hlm = health_log_mode_from_context(ctx);
        global_iter_health_monitor().set_mode(hlm);
        global_iter_health_monitor().clear_state();
    }

    /* -- Pre-flight audit (Phase 5) -----------------------------------
     * Walk every configured descriptor's meta() and run the per-param /
     * cross-param HealthChecks on the training StructureDB. This is the
     * BASINS-style guard: a fixed-and-off-shell `CGRIDMB` is flagged
     * here with a rho-collapse explanation BEFORE the optimiser runs. The
     * critical-fixed-not-optimised audit additionally flags Critical
     * params that lack any OPTIM block. Honours the AUDIT context key:
     *   off    -> skipped
     *   warn   -> findings logged via Logger; HPO continues
     *   error  -> any FAIL throws and aborts before NOISE_CALIBRATION
     */
    if (!has_train_db) {
        core::Logger::instance().log(
            tadah::core::LogLevel::INFO,
            "No training DBFILE: pre-flight data audit skipped.");
    } else {
        mlip::AuditMode mode = mlip::audit_mode_from_context(ctx);
        if (mode != mlip::AuditMode::Off) {
            std::vector<std::string> optim_keys;
            optim_keys.reserve(trg.optimap.size());
            for (const auto& kv : trg.optimap) optim_keys.push_back(kv.first);
            auto rep_soft  = mlip::run_preflight_from_db(
                host.train_stdb_mut(), ctx, mode);
            auto rep_optim = mlip::run_critical_optim_audit(ctx, optim_keys);
            rep_soft.log();
            rep_optim.log();
            mlip::enforce_audit_mode(rep_soft,  mode);
            mlip::enforce_audit_mode(rep_optim, mode);
        }
    }

    const OptimizerSpec& sp = trg.spec;

    /* -- Run optimiser via the configured RunPolicy ------------------- */

    /* Build the decorator stack from sp's MULTISTART / PRESTAGE block /
     * RESTART knobs. With no knobs set, this collapses to BasePolicy.
     * Targets pointer flows through to RestartPolicy for fresh-init
     * generation on stagnation.
     *
     * BasinsPolicy registers itself + its BasinPool with
     * CheckpointManager from its constructor, so build_run_policy()
     * must run *before* try_load(). */
    auto policy = build_run_policy<V,R>(sp, &trg);

    /* The stagnation threshold and top-K tracking are process-wide statics
     * that each policy arms inside run() (RESTART, PRESTAGE, BASINS). A
     * policy that ended by an exception in an earlier run in this process
     * can leave them armed -- a stale window would stop a PLAIN run as
     * "FAILED -- HPO stagnation threshold reached". Every run, fresh or
     * resumed, starts disarmed. */
    HPO_Host<V,R>::set_stagnation_threshold(0);
    HPO_Host<V,R>::set_track_topk(0);

    /* -- Checkpoint subsystem wiring (LIB BASINS only for now) ----------
     *
     * Order is important: try_load() must run BEFORE NOISE_CALIBRATION
     * so that a resumed run does not re-run the calibration evals (each
     * one is a real LAMMPS evaluation that pollutes the eval counter,
     * per-eval logs, and the best-so-far comparator). The host is
     * registered alongside HPOLog so its load() restores g_best_ et al.
     * before any further evals happen. */
    auto& cpm = core::CheckpointManager::instance();
    bool resumed = false;
    if (sp.strategy_type == StrategyType::BASINS ||
        sp.strategy_type == StrategyType::BASINSV2) {
        cpm.set_path("checkpoint.tadah");
        const double cp_secs = (sp.strategy_type == StrategyType::BASINS)
            ? sp.basins.checkpoint_seconds
            : sp.basins_v2.checkpoint_seconds;
        const std::size_t cp_evs = (sp.strategy_type == StrategyType::BASINS)
            ? sp.basins.checkpoint_evals
            : sp.basins_v2.checkpoint_evals;
        if (cp_secs > 0.0)
            cpm.set_period_seconds(cp_secs);
        if (cp_evs > 0)
            cpm.set_period_evals(cp_evs);
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
            const std::string strat_name = (sp.strategy_type == StrategyType::BASINSV2)
                ? "BASINSv2" : "BASINS";
            log.info(strat_name + ": resumed from checkpoint " + cpm.path() +
                     " (n_eval=" + std::to_string(log.n_eval()) +
                     ", best_loss=" +
                     std::to_string(log.best_loss()) + ")");
            /* Belt-and-suspenders: HPO_Host::load() already restores
             * g_iter_; the floor call below is idempotent and keeps
             * the eval counter >= HPOLog's n_eval even if the
             * checkpoint predates the HPO_HOST section. */
            HPO_Host<V,R>::set_g_iter_floor(log.n_eval());
        }
    }

    /* The best potential could not be saved even after retrying transient
     * errors (see durable_best.h): the run cannot promise its result, so it
     * fails loudly instead of optimising on. */
    auto fail_persistence = [&](const std::string& what) {
        log.warning("Best-potential persistence failed: " + what);
        log.record_best_persistence_failure();
        /* The storage has just failed for up to the whole durable budget:
         * drain what is queued without backoff (a permanent errno still
         * gets its one immediate re-attempt). */
        host.flush_remaining(/*retry_io=*/false);
        log.finish("FAILED -- best-potential persistence: " + what);
    };

    /* Everything from here on can evaluate -- and therefore publish the
     * best potential: the fresh-run reset below, NOISE_CALIBRATION, the
     * optimiser. One handler set closes the journal with the right status
     * for all of it and lets the error reach the caller (non-zero exit):
     *   ShutdownRequested    -> "SHUTDOWN (<signal>)", clean return (0),
     *                           unless a new best could not be saved;
     *   durable_best::Error  -> "FAILED -- best-potential persistence: ...";
     *   anything else        -> "FAILED -- <what>". */
    try {
        /* Bind best-potential publication to this run only after checkpoint
         * detection. A confirmed fresh run atomically discards stale artifacts
         * from an unrelated prior objective; an explicit resume retains and
         * validates its incumbent. MPI ranks synchronize before any evaluation. */
        host.prepare_best_publication(resumed);

        /* -- Noise calibration (NOISE_CALIBRATION N): repeat the seed point
         * N times before the optimiser starts, surface the loss spread, and
         * suggest a realistic FTOL_REL. Each repeat is a real evaluation --
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
                 * fires when |Deltaf| < ftol_rel * |f|, so 10x the observed
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
                 * and safe -- BasinsPolicy is the only RunPolicy not wrapped
                 * by the decorator stack (run_policy.h:434-436). Using the
                 * cast keeps RunPolicy's interface free of noise-specific
                 * methods, per the ticket's "Do not change the RunPolicy
                 * decorator pattern" rule. */
                if (auto* bp = dynamic_cast<BasinsPolicy<V,R>*>(policy.get())) {
                    bp->set_noise_floor(suggested_ftol);
                }
                if (auto* bp2 = dynamic_cast<BasinsV2Policy<V,R>*>(policy.get())) {
                    bp2->set_noise_floor(suggested_ftol);
                }
            }
        } else if (resumed) {
            /* C3 resume path: NOISE_CALIBRATION did not re-run (gated on
             * !resumed), but the prior run's suggested_ftol was persisted in
             * the [HPOLOG] section. Re-plumb it into BasinsPolicy / BasinsV2Policy
             * so the (opt-in) clamp keeps applying across resumes. */
            if (auto suggested = log.noise_suggested_ftol()) {
                if (auto* bp = dynamic_cast<BasinsPolicy<V,R>*>(policy.get())) {
                    bp->set_noise_floor(*suggested);
                }
                if (auto* bp2 = dynamic_cast<BasinsV2Policy<V,R>*>(policy.get())) {
                    bp2->set_noise_floor(*suggested);
                }
            }
        }

        log.start_progress();

        /* quick info */
        std::cout<<"[HPO] "<<sp.lib<<":"<<sp.algo<<"   dim="<<sp.low.size()<<"\n";

        policy->run(sp, host);
        /* A new best that could not be saved fails the run even when the
         * durable_best::Error was swallowed on its way up. */
        detail::rethrow_if_best_unsaved();

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
        /* A new best that could not be saved outranks the shutdown: the
         * run did not keep its result (see the durable_best::Error
         * handler below). */
        if (const auto ub = log.unsaved_best()) {
            fail_persistence(ub->what);
            throw durable_best::Error(ub->what);
        }
        /* SIGINT/SIGTERM/SIGUSR2 -- the policy already attempted to
         * commit a checkpoint on the way out. Drain the I/O queue,
         * close the journal cleanly with the signal reason. (After the
         * stop request this drain's retry budget is the 2 s stop grace for
         * all files together, not 10 s: a blip still keeps its rows, and
         * a dead mount cannot hold the exit past a scheduler's kill
         * timeout.) */
        log.warning(std::string("Shutdown requested: ") +
                    core::SignalHandler::reason());
        host.flush_remaining();
        log.finish(std::string("SHUTDOWN (") + core::SignalHandler::reason() + ")");
        /* Final flush of the checkpoint after the queue drain so the
         * persisted n_eval matches what the eval files contain. */
        if (sp.strategy_type == StrategyType::BASINS) {
            try { cpm.save_now(); } catch (...) { /* best effort */ }
        }
    } catch (const durable_best::Error& e) {
        /* Of several failed candidates (parallel chains) the one that
         * failed first is the cause, even when a later one's error reaches
         * this handler first (detail::durable_failure_cause). */
        const std::string cause = detail::durable_failure_cause(e);
        fail_persistence(cause);
        if (cause != e.what()) throw durable_best::Error(cause);
        throw;
    } catch (const std::exception& e) {
        const std::string err = std::string("FAILED -- ") + e.what();
        log.warning(std::string("Optimiser threw: ") + e.what());
        host.flush_remaining();
        log.finish(err);
        throw;
    } catch (...) {
        log.warning("Optimiser threw a non-standard exception");
        host.flush_remaining();
        log.finish("FAILED -- non-standard exception");
        throw;
    }

    /* PC_ISOLATION: deterministic worker-pool teardown on the clean paths
     * (SHUTDOWN frames + reap). On the throwing path above, the singleton's
     * static destructor performs the same shutdown at process exit. No-op in
     * THREAD mode (pool never constructed). */
    pc_worker_pool_shutdown();

    return 0;
}
} // namespace tadah::hpo
