#include <tadah/hpo/check_summary.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/run_policy.h>
#include <tadah/hpo/run_driver.h>
#include <tadah/mlip/key_provenance.h>
#include <tadah/core/retired_keys.h>
#include <tadah/mlip/training_record.h>  // ignore_configuration_potential_lines
#include <tadah/hpo/hpo.h>
#include <tadah/hpo/phase_specs.h>   // with_block_line
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_mpi.h>     // hpo_rank(), hpo_size(), hpo_rank_suffix()
#include <tadah/hpo/script_executor.h>
#include <tadah/hpo/pc_worker_pool.h>   // PC_ISOLATION worker-process pool
#include <tadah/hpo/iter_health.h>
#include <tadah/hpo/rho_gates.h>
#include <tadah/hpo/startup_validation.h>
#include <tadah/hpo/warmstart_cache.h>
#include <tadah/core/checkpoint.h>   // core::ShutdownRequested
#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/core/blueprint_parser.h>
#include <tadah/core/signal_handler.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/preflight_audit.h>
#include <tadah/mlip/parallel.h>   // set_parallel_backend (Stage 2.3)
#include <tadah/models/descriptors/dm/embf0.h>
#include <tadah/models/functions/kernels/kern_base.h>   // KernelSettingsLineScope

#include <algorithm>  // std::max
#include <cstdlib>    // getenv()
#include <sstream>    // std::ostringstream (PC_SCRIPT_TIMEOUT diagnostic)
#include <functional> // std::function
#include <filesystem>
#include <iostream>   // std::cout, std::cerr
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

/* NOISE_CALIBRATION runs only for a non-trivial knob (>= 2 evals: one
 * evaluation has no spread to measure). Pulled out of hpo_run() into a
 * small free function so the gating is unit-testable without invoking the
 * full optimiser; see HPO/tests/test_noise_calibration.cpp. */
bool should_run_noise_calibration(std::size_t noise_n)
{
    return noise_n >= 2;
}

std::string hpo_mpi_dispatch_refusal(int world_size, const OptimizerSpec& spec)
{
    if (world_size <= 1) return {};
    const std::string ranks = std::to_string(world_size) + " ranks";
    /* At the line the OPTIMIZER block closes at, as the startup refusals
     * of its settings (with_block_line); a spec built outside the parser
     * has none. */
    if (!(spec.strategy_type == StrategyType::PLAIN &&
          spec.lib == "TADAH" &&
          (spec.algo == "ANNEAL_PT" || spec.algo == "ANNEAL_PA")))
        return with_block_line(spec,
               "multi-rank HPO supports only STRATEGY TYPE PLAIN with "
               "LIB TADAH and ALGO ANNEAL_PT or ANNEAL_PA; got " +
               spec.lib + ":" + spec.algo + " on " + ranks);
    /* Each rank tracks stagnation over its own evaluations: the rank that
     * stagnated would restart and block in the driver's next collective
     * while its peers fail with "worker failed on another MPI rank". */
    if (spec.restart.active())
        return with_block_line(spec,
               "multi-rank HPO does not support RESTART (AFTER_STAGNATION " +
               std::to_string(spec.restart.after_stagnation) + ", MAX " +
               std::to_string(spec.restart.max) + " on " + ranks +
               "): stagnation is tracked per rank; run on one rank or "
               "remove RESTART");
    return {};
}

bool hpo_mpi_dispatch_supported(int world_size, const OptimizerSpec& spec)
{
    return hpo_mpi_dispatch_refusal(world_size, spec).empty();
}

namespace {

/* NOISE_CALIBRATION N: repeat the seed point N times before the optimiser
 * starts, surface the loss spread, and suggest a realistic FTOL_REL. Each
 * repeat is a real evaluation -- it counts toward the global eval counter,
 * the best-so-far tracker, and the per-eval log files (loss.tadah /
 * params.tadah / ...). The overhead is N - 1 extra wall-clock evals; the user
 * opted in by setting NOISE_CALIBRATION, so we do not also charge it against
 * MAXEVAL (the optimiser still gets its full budget). Runs under drive_run()
 * in hpo_run(), like the optimiser. */
void run_noise_calibration(const OptimizerSpec& sp, const HPO_Host<V,R>& host,
                           HPOLog& log)
{
    if (!should_run_noise_calibration(sp.noise.calibration_n)) return;
    const std::size_t N = sp.noise.calibration_n;
    if (sp.optim_init.empty()) {
        log.warning("NOISE_CALIBRATION requested but no starting "
                    "point is available; skipping.");
        return;
    }
    std::cout << "[HPO] NOISE_CALIBRATION: running " << N
              << " evals at the seed point to measure objective noise\n";
    log.info("NOISE_CALIBRATION: running " + std::to_string(N) +
             " evals at the seed point");

    const auto seed_real = sp.optim_init.front();
    V seed_vec;
    seed_vec.set_size(static_cast<long>(seed_real.size()));
    for (long i = 0; i < seed_vec.size(); ++i)
        seed_vec(i) = seed_real[static_cast<std::size_t>(i)];

    /* The losses of the evaluations made: N is up to INT_MAX, and room
     * for all of them (16 GiB) is not taken before the first. */
    std::vector<double> losses;
    for (std::size_t i = 0; i < N; ++i)
        losses.push_back(host(seed_vec));

    /* Suggest FTOL_REL = 10 * relative_spread, floored at 1e-6 (no point
     * recommending tolerances below double precision times a sensible fudge
     * factor). NLopt's FTOL_REL test fires when |Deltaf| < ftol_rel * |f|,
     * so 10x the observed relative spread gives the local search room to
     * terminate once it crosses the noise floor instead of chasing it. */
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

    std::ostringstream oss;
    oss << std::scientific << std::setprecision(3);
    oss << "NOISE_CALIBRATION: rel_spread=" << rel_spread
        << "  suggested FTOL_REL=" << suggested_ftol
        << "  (mean=" << mean << ", min=" << mn << ", max=" << mx << ")";
    std::cout << "[HPO] " << oss.str() << "\n";
    log.info(oss.str());
}

}  // namespace

namespace {

/* Where FORCE and STRESS came from, in words for the journal. Read when
 * hpo_run() starts, before any code can insert a default into ctx. */
struct FitSwitchSources {
    std::string force;
    std::string stress;
};

FitSwitchSources fit_switch_sources(const core::Context& ctx,
                                    const mlip::KeyProvenance* provenance)
{
    auto source = [&](const char* key) -> std::string {
        if (!ctx.stored_value(key)) return "default";
        if (provenance) return provenance->origin(key).describe(key);
        return "source not recorded";
    };
    return {source("FORCE"), source("STRESS")};
}

/* What hpo_run() does with its context: a run, or the startup checks of a
 * run (CHECK true: `tadah hpo --check`). */
enum class RunMode { Run, Check };

int hpo_run_from_startup(core::Context& ctx, HPOLog& log,
                         const FitSwitchSources& fit_sources,
                         const mlip::KeyProvenance* provenance, RunMode mode);

/* How the run executes its PC_LAMMPS simulations: PC_ISOLATION of the
 * target file or its environment override TADAH_PC_ISOLATION, and the
 * worker deadline PC_WORKER_TIMEOUT or TADAH_PC_WORKER_TIMEOUT. */
struct PcIsolation {
    std::string mode;            // "THREAD" | "PROCESS"
    bool        explicit_mode;   // the target file or the environment named it
    double      worker_timeout;  // seconds; 0: derived from each script's limit
};

/* The run's PC_ISOLATION settings, with every check of them: an override
 * that is not a number (TADAH_PC_WORKER_TIMEOUT), a mode that is neither
 * THREAD nor PROCESS, and, in the MPI build, a PROCESS that was asked for
 * (the default PROCESS is THREAD there) throw. */
PcIsolation resolve_pc_isolation(const HPOTargets& trg, HPOLog& log)
{
    PcIsolation pc{trg.PC_ISOLATION, trg.PC_ISOLATION_set,
                   trg.PC_WORKER_TIMEOUT};
    if (const char* e = std::getenv("TADAH_PC_ISOLATION")) {
        if (*e) { pc.mode = e; pc.explicit_mode = true; }
    }
    if (const char* e = std::getenv("TADAH_PC_WORKER_TIMEOUT")) {
        if (*e) pc.worker_timeout = HPOTargets::parse_pc_worker_timeout(
                    e, "TADAH_PC_WORKER_TIMEOUT");
    }
    if (pc.mode != "THREAD" && pc.mode != "PROCESS")
        throw std::runtime_error(
            "PC_ISOLATION must be THREAD or PROCESS, got '" + pc.mode + "'");
#ifdef TADAH_HPO_MPI
    if (pc.mode == "PROCESS") {
        /* A fork+exec'd child of an MPI parent must never touch MPI
         * (LAMMPS instances need an initialized MPI in this build).
         * EXPLICIT requests hard-error; the serial-build DEFAULT of
         * PROCESS quietly falls back to THREAD here. */
        if (pc.explicit_mode)
            throw std::runtime_error(
                "PC_ISOLATION PROCESS is not supported in the MPI build "
                "(TADAH_HPO_MPI): the worker processes cannot initialize "
                "MPI. Use the serial build, or PC_ISOLATION THREAD.");
        pc.mode = "THREAD";
        log.info("PC_ISOLATION: THREAD (MPI build -- the PROCESS worker "
                 "pool is serial-build only)");
    }
#else
    (void)log;
#endif
    return pc;
}

/* The worker pools of one run: the script executor -- THREADS workers that
 * run the PC_LAMMPS fan-out and, through MLIP's parallel backend, the
 * train/predict kernels -- and, under PC_ISOLATION PROCESS, the worker
 * processes (pc_worker_pool_init, which the run calls once it knows the
 * mode). Built when the run sizes them; on every way out of hpo_run() -- a
 * return, a signal stop, an error, also one at startup once they are built
 * -- the destructor removes MLIP's backend (a later task of the process
 * runs its kernels on OpenMP again), stops the worker processes and joins
 * the executor's workers, so the next run in the process builds its own.
 * A run that threw used to leave its worker processes up (the next run's
 * pc_worker_pool_init refused: "pool already initialized"), and every run
 * ran on the executor of the first run's THREADS. */
class RunPools {
public:
    explicit RunPools(int width)
    {
        init_script_executor(width);
        try {
            /* Stage 2.3: route the MLIP train/predict kernels (design-matrix
             * build, descriptor calc, predict) onto the SAME pool. This drops
             * the nested per-replica OpenMP teams those kernels used to spawn
             * and lets one replica's train/predict overlap another's fan-out.
             * The kernels write disjoint output slots, so results are
             * unchanged (verified by the determinism gate). Installed for the
             * run, before any optimiser eval. */
            tadah::mlip::set_parallel_backend(
                [](std::size_t n,
                   const std::function<void(std::size_t)>& body) {
                    script_executor().run_batch(
                        n, [&](std::size_t i, int /*worker_id*/) { body(i); });
                });
        } catch (...) {
            shutdown_script_executor();
            throw;
        }
    }
    ~RunPools()
    {
        tadah::mlip::set_parallel_backend({});
        pc_worker_pool_shutdown();   /* no-op unless PROCESS built one */
        shutdown_script_executor();
    }
    RunPools(const RunPools&) = delete;
    RunPools& operator=(const RunPools&) = delete;
};
}  // namespace

namespace {

/* HPOTargets::create_archive_directories() on every rank of the run (each
 * creates them; one that exists is kept): an error on any rank is every
 * rank's (run_on_every_rank, one collective, as in
 * HPO_Host::prepare_best_publication), so no rank waits for a peer that
 * stopped. */
void create_archive_directories_on_every_rank(const HPOTargets& trg)
{
    run_on_every_rank("the creation of the DUMP and BDUMP directories",
                      [&] { trg.create_archive_directories(); });
}

/* HPOTargets::check_run_outputs() on every rank of the run: each checks
 * its own rank-suffixed files, and an error on any rank is every rank's
 * (run_on_every_rank). */
void check_run_outputs_on_every_rank(const std::string& journal)
{
    run_on_every_rank("the check of the output files",
                      [&] { HPOTargets::check_run_outputs(journal); });
}

}  // namespace

int hpo_run(core::Context& ctx)
{
    return hpo_run(ctx, nullptr);
}

int hpo_run(core::Context& ctx, const mlip::KeyProvenance* provenance)
{
    /* CHECK (`tadah hpo --check`, CHECK true in a TASK hpo section or in a
     * training configuration that is not a potential) tells this function
     * what to do; it is a setting of the run, no key of the model. It is
     * taken out of the context before anything copies it (a task's context
     * holds CHECK false by default): the run's training record does not
     * list it, and no potential has a CHECK key line (training_record_keys)
     * or sets it (run_setting_keys). */
    const bool check = ctx.has_stored("CHECK") && ctx.get<bool>("CHECK");
    if (ctx.has_stored("CHECK")) ctx.remove("CHECK");

    /* The run builds the model of every evaluation, and the potential each
     * PC_LAMMPS script runs, from one configuration: at VERBOSE > 0 a
     * kernel prints its settings line ("Kern_RBF | sigma: 1 | gamma: 1")
     * at the first build of the run only, the -c model's at startup
     * (validate_hpo_setup). A run that starts after this one has ended
     * prints its own. */
    models::KernelSettingsLineScope kernel_settings_line;
    const FitSwitchSources fit_sources = fit_switch_sources(ctx, provenance);

    auto& log = HPOLog::instance();
    /* Rank-suffixed on MPI ranks > 0 (no-op otherwise) so multi-rank runs
     * sharing one working directory keep separate journals. */
    const std::string journal = "hpo_run.log" + hpo_rank_suffix();
    if (check) {
        /* A check writes no file, so it keeps no journal: the journal
         * lines of the startup go nowhere but its WARNING and HINT lines,
         * which the check prints (in its summary, or before its error). A
         * journal an earlier run in this process left open (it stopped at
         * startup) is closed first, so that none of them goes into it. */
        log.begin_without_journal();
        try {
            /* The first check of the run, made before it opens its
             * journal. */
            check_run_outputs_on_every_rank(journal);
            return hpo_run_from_startup(ctx, log, fit_sources, provenance,
                                        RunMode::Check);
        } catch (...) {
            const std::vector<JournalNote> notes = log.notes_without_journal();
            if (!notes.empty()) {
                std::cerr << "tadah hpo --check: the startup checks stopped "
                             "at the error below.\n";
                write_check_notes(std::cerr, notes);
                std::cerr.flush();
            }
            throw;
        }
    }

    /* The run writes its journal, result files and best_pot.tadah into the
     * working directory: a working directory, or one of those files, it
     * cannot write stops it here, before it opens (and truncates) its
     * journal, as it stops a check. */
    check_run_outputs_on_every_rank(journal);

    /* -- Open run journal first -------------------------------------
     * This must precede HPOTargets construction so that diagnostics
     * emitted while parsing OPTIMIZER, OPTIM, and INIT_STRATEGY blocks
     * (NOISE_AUTO, "InitStrategy X produced N points", WARM-front
     * promotion, implicit-CONFIG suppression, ...) reach both stdout
     * and the journal. The OPTIMISER section is written later, after
     * trg.spec has been finalised. */
    {
        std::string version = "unknown";
        try { version = core::BlueprintParser::getBlueprintTadah().version; }
        catch (...) {}
        std::string cwd = std::filesystem::current_path().string();
        log.open(journal, version, cwd, static_cast<int>(getpid()));
        log.set_rank_scope(hpo_rank(), hpo_size());
    }

    /* A startup error -- the parser, validate_hpo_setup(), building the
     * host -- is recorded in the journal before it propagates; the journal
     * used to end without a word and the reason went to stderr only. An
     * error of the run itself arrives after drive_run() has written the
     * trailer and closed the journal, and then nothing more is written. */
    try {
        return hpo_run_from_startup(ctx, log, fit_sources, provenance,
                                    RunMode::Run);
    } catch (const std::exception& e) {
        log.event("ERROR", std::string("startup: ") + e.what());
        throw;
    }
}

namespace {

/* hpo_run() once the journal is open (RunMode::Run), or without a journal
 * (RunMode::Check). @p provenance: where the keys of ctx came from (for the
 * training record of the potential files).
 *
 * A check makes every startup check of the run, in the run's order, and
 * gives the WARNING and HINT lines the run writes to its journal at
 * startup; it skips what writes or starts something: the signal handlers,
 * the worker pools (no PC_ISOLATION PROCESS worker), the journal sections,
 * the search-space sidecars, the result files (HPO_Host's
 * HostPurpose::Check) and the DUMP and BDUMP directories, and everything
 * that evaluates. Once the STRATEGY TYPE and RESTART settings are built
 * into the run's policy it writes the summary (write_check_summary) to
 * standard output and returns 0. An error is thrown as the run throws
 * it. */
int hpo_run_from_startup(core::Context& ctx, HPOLog& log,
                         const FitSwitchSources& fit_sources,
                         const mlip::KeyProvenance* provenance, RunMode mode)
{
    const bool check = mode == RunMode::Check;
#ifdef TADAH_ENABLE_OPENMP
    {
        /* Under the shared-pool model OpenMP no longer drives the
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
     * HPO_Host stops evaluating (it throws core::ShutdownRequested at the
     * start of the next evaluation) under ctrl+C or SLURM cancellation. The
     * exception unwinds every optimiser and strategy to drive_run() below,
     * which flushes the result files, writes the SHUTDOWN trailer and makes
     * hpo_run() return 128 + the signal number. A check evaluates nothing:
     * a signal ends it at once. */
    if (!check) core::SignalHandler::install();

    /* The hot-variable seeds of an earlier run in this process are not this
     * run's: cleared before HPOTargets, whose WARM init strategy primes
     * this run's own. */
    warmstart::begin_run();

    /* A potential's lines of the keys of its training record but DBFILE (a
     * -c seed with WEIGHTS, or a potential a configuration or TASK hpo
     * section includes with CONFIG; a potential of an earlier version has
     * them as key lines: a 1.5.x best_pot.tadah has HPOTARGET, VERBOSE and
     * NUMERIC, and VALIDATION when its run had a validation file) are not
     * read for this run: TaskFactory's parse skipped them and the WARNING names
     * those this run reads (no UNCERTAINTY, no OUTFILE); a ctx without a
     * provenance is taken as one file and the dataset keys and UNCERTAINTY
     * are removed here (no parse stores a potential's run settings), before
     * the record is taken, which then lists the run's own settings. */
    {
        const std::string ignored =
            mlip::ignore_configuration_potential_lines(ctx, provenance, "hpo");
        if (!ignored.empty()) {
            core::Logger::instance().log(core::LogLevel::WARNING, ignored);
            log.warning(ignored);
        }
    }
    /* A configuration with a line of a retired key (its own, or of a file
     * it includes with CONFIG: the parse recorded them) is refused with the
     * conversion of each, before the target file is read: every fit of the
     * run would refuse it (core::require_no_retired_keys, which
     * validate_hpo_setup applies again through mlip::require_fit_settings
     * for a caller that skips this). */
    core::require_no_retired_keys(ctx, "tadah hpo");
    /* The hpotarget file is the run's own: --hpotarget (tadah hpo -c
     * requires it) or an HPOTARGET line of the task file. A TASK hpo
     * section whose only HPOTARGET line was that of the potential it
     * includes has none: the refusal names where the run's own goes, by
     * where its keys came from (a run of a task file takes no
     * command-line option), and the potential's line that was not read. */
    if (!ctx.has_stored("HPOTARGET"))
        throw std::runtime_error(mlip::missing_run_setting(
            "hpo", "HPOTARGET", "the hpotarget file", provenance));

    /* Its training_record (the comment block of every potential file the
     * run writes) is taken from ctx before anything inserts a default; the
     * provenance leaves out the blueprint defaults no one wrote. */
    HPOTargets trg(ctx, ctx.get<std::string>("HPOTARGET"), provenance);
    /* The search-space files the run writes (they depend on the target
     * file): an existing one the run cannot write stops it here, before
     * any data is loaded, as it stops a check. */
    run_on_every_rank("the check of the search-space files",
                      [&] { trg.check_search_space_files(); });

    /* This is more than a dispatch convenience: HPO_Host's live best/HOT
     * state is intentionally rank-local while best_pot.tadah is the global
     * cross-rank artifact. The plain PT/PA drivers own the communicator, so a
     * rank can safely keep its local incumbent when another rank wins
     * canonical publication; any other driver, and the MULTISTART and
     * PRESTAGE strategies, would drive the ranks outside that protocol, and
     * so would RESTART, whose stagnation watchdog is per rank. Refused
     * here, before any evaluation. Keep size-one MPI launches identical to
     * serial builds. */
    if (const std::string refusal =
            hpo_mpi_dispatch_refusal(hpo_size(), trg.spec);
        !refusal.empty())
        throw std::runtime_error(refusal);

    /* ---- FAILSCORE_GRADING: graded ("smooth") failure losses --------
     * ON: every failed eval returns a banded, monotone value inside
     * [0.999*FAILSCORE, FAILSCORE) instead of the constant plateau.
     * Env override TADAH_FAILSCORE_GRADING=ON|OFF (harness A/B), set
     * through set_failscore_grading so that every optimiser spec, whose
     * finite differences read it, follows it.
     * Design: INVESTIGATIONS/SMOOTH_FAILSCORE_analysis.md */
    if (const char* e = std::getenv("TADAH_FAILSCORE_GRADING")) {
        if (*e) {
            const std::string v(e);
            if (v == "ON")       trg.set_failscore_grading(true);
            else if (v == "OFF") trg.set_failscore_grading(false);
            else throw std::runtime_error(
                    "TADAH_FAILSCORE_GRADING must be ON or OFF");
        }
    }
    if (trg.FAILSCORE_GRADING) {
        /* validate_hpo_setup (below) refuses a FAILSCORE the bands do not
         * fit. */
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
        /* Under OFF the probe is still one of the evaluation's
         * simulations: its failure -- the script's, or an S that cannot be
         * read or has no number (apply_task_result) -- fails the
         * evaluation like any other's. Only the use of S is off. */
        log.warning("PC_FAILPROBE is defined but FAILSCORE_GRADING is "
                    "OFF -- its S neither grades nor gates an evaluation; "
                    "a probe that fails (its script fails, or its S cannot "
                    "be read or has no number) fails its evaluation, as "
                    "any PC_LAMMPS simulation does.");
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

    /* --validation is optional: PC_OBSERVABLE / PC_LAMMPS can carry the
     * loss on their own; validate_hpo_setup below rejects setups where
     * validation-set metrics are requested without a validation file.
     * One file: a second value is refused, not dropped. */
    const std::string val = validation_config_file(ctx);

    /* Fail fast on inconsistent weight-pinning / loss-source / strategy
     * setups: before the worker pool is sized, any PC_ISOLATION PROCESS
     * worker is spawned, any data is loaded or the optimiser starts. It
     * depends on the configuration alone (BUMPGATE, with its environment
     * override, is resolved above). */
    const bool has_train_db =
        ctx.contains("DBFILE") && ctx.size("DBFILE") > 0;
    validate_hpo_setup(ctx, trg, has_train_db, !val.empty());

    /* ---- PC_ISOLATION: pc_lammps execution mode --------------------
     * PROCESS = persistent worker-process pool (one child per fan-out
     * slot; LAMMPS globals unshared -- the free(): invalid pointer race
     * class is structurally gone). THREAD = in-process (pre-existing
     * behaviour; kept one production cycle as a deprecated fallback).
     * Env override TADAH_PC_ISOLATION lets the A/B and determinism
     * harnesses switch modes without touching configs. Design:
     * INVESTIGATIONS/PC_ISOLATION_process_pool.md. Resolved and checked
     * here, for the run and for a check alike; the run builds its pool
     * below. */
    const PcIsolation pc = resolve_pc_isolation(trg, log);

    /* Parallelisation model: THREADS is the TOTAL width of the shared
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
    /* THREADS is an integer from 1 to INT_MAX (checked at its line). The
     * run's pools live until hpo_run() returns or throws (RunPools). A
     * check builds none: it starts no worker, and gives the warnings of
     * these settings as the run does. */
    std::optional<RunPools> pools;
    int exec_width = 0;
    if (!check) {
        pools.emplace(static_cast<int>(trg.spec.num_threads.value_or(1)));
        /* The width of the pool that runs: the journal reports it, and the
         * PROCESS pool below has one worker process per executor worker
         * (a task runs on the channel of its executor worker's id). */
        exec_width = script_executor().width();
        /* Give the journal the true pool width -- the honest denominator
         * for the periodic pc_lammps pool-utilisation figure. */
        log.set_fanout_width(exec_width);
    }
    {
#ifdef TADAH_ENABLE_OPENMP
        const int omp_w = std::max(1, omp_get_max_threads());
#else
        int omp_w = 1;
        if (const char* e = std::getenv("OMP_NUM_THREADS")) {
            const int v = std::atoi(e); if (v > 0) omp_w = v;
        }
#endif
        if (!check)
            log.info("ScriptExecutor: " + std::to_string(exec_width) +
                     " workers = THREADS (total pool: fan-out + train/predict); "
                     "OMP_NUM_THREADS=" + std::to_string(omp_w) +
                     " drives only the residual OpenMP kernels (NN lists, "
                     "data-stats)");
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

        const std::string& pc_mode = pc.mode;
        const bool pc_mode_explicit = pc.explicit_mode;
        const double pc_wtimeout = pc.worker_timeout;
        if (pc_mode == "PROCESS") {
#ifdef TADAH_HPO_MPI
            /* unreachable: handled above */
#else
            if (check) {
                /* A check starts no worker process. */
            } else if (!trg.LS.empty()) {
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
            } else if (pc_mode_explicit) {
                log.info("PC_ISOLATION: PROCESS requested but no PC_LAMMPS "
                         "sims are configured -- nothing to isolate");
            } else {
                /* PROCESS is the serial build's default: nothing asked
                 * for it. */
                log.info("PC_ISOLATION: PROCESS (default) -- no PC_LAMMPS "
                         "sims are configured, nothing to isolate");
            }
#endif
        } else {
#ifndef TADAH_HPO_MPI
            /* Serial build running THREAD => it was explicitly requested
             * (the default is PROCESS): always flag the deprecated mode. */
            log.warning("PC_ISOLATION: THREAD (deprecated fallback) -- "
                        "concurrent in-process LAMMPS instances share process "
                        "globals (the free(): invalid pointer race class); "
                        "PROCESS is the supported mode. THREAD will be "
                        "removed after one production cycle.");
#else
            /* MPI build: THREAD is its only mode. The default says so in
             * the PROCESS branch above; a request names the mode too, so
             * the journal records it whichever way it was chosen. */
            if (pc_mode_explicit)
                log.info("PC_ISOLATION: THREAD (requested; the only mode of "
                         "the MPI build)");
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
            /* The spec as written, and its seconds when it is a time. */
            std::string soft = trg.MAX_TIME + " (no per-script limit)";
            if (trg.MAX_TIME_seconds) {
                std::ostringstream t; t << *trg.MAX_TIME_seconds << " s";
                soft = t.str();
                if (t.str() != trg.MAX_TIME + " s")
                    soft += " (" + trg.MAX_TIME + ")";
            }
            std::string hard;
            if (pc_mode == "PROCESS") {
                const long dl_ms = pc_worker_deadline_ms(pc_wtimeout,
                                                         trg.MAX_TIME_seconds);
                std::ostringstream w; w << static_cast<double>(dl_ms) / 1000.0;
                if (pc_wtimeout > 0.0)
                    hard = w.str() + " s (PC_WORKER_TIMEOUT)";
                else if (dl_ms < 0)
                    hard = "none";
                else
                    hard = w.str() + " s, derived as max(3t, t + 120 s) from "
                           "the per-script limit t (a line's --maxtime "
                           "derives its own)";
            } else {
                hard = "n/a (THREAD mode)";
            }
            log.info("PC_SCRIPT_TIMEOUT: " + soft + " (from " + src +
                     ") -- soft LAMMPS `timer timeout`, re-armed each "
                     "run-section; hard per-worker-task deadline: " + hard);
            if (trg.MAX_TIME_seen_new && trg.MAX_TIME_seen_legacy)
                log.warning("both PC_SCRIPT_TIMEOUT and legacy MAXTIME were "
                            "given; the later line wins (effective: " +
                            soft + " from " + src + ")");
            else if (trg.MAX_TIME_seen_legacy)
                log.info("MAXTIME is a legacy alias for PC_SCRIPT_TIMEOUT "
                         "(per-pc_script timeout, seconds); prefer "
                         "PC_SCRIPT_TIMEOUT in new configs");
        }
    }

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

    /* -- Write setup sections (a check writes neither them nor the
     * sidecars) ------------------------------------------------------ */
    if (!check) {
        log.write_files_section(ctx.get<std::string>("HPOTARGET"), val, trg);

        log.write_optimizer_section(trg.spec, param_labels);

        /* Machine-readable search-space sidecars next to the journal:
         * hpo_search_space.tsv (one row per dim) always;
         * hpo_optim_resolved.hpo (a paste-ready explicit OPTIM block) when
         * any box was derived by an AUTO rule, so a later run can pin
         * exactly what this one searched. */
        trg.write_search_space_files(".");
    }

    /* Build HPO_Host (loads train + val structures, applies load
     * transforms). A check's host loads and checks the data as the run's
     * does, and creates no result file and no archive directory. */
    HPO_Host<V,R> host(ctx, trg, val,
                       check ? HostPurpose::Check : HostPurpose::Run);
    /* This run's evaluations are numbered on from an earlier run's in the
     * process; the journal's milestones count this run's. */
    if (!check) log.set_eval_base(HPO_Host<V,R>::g_iter_value());
    /* EMBF0 (DM_REAM only): the -c potential's raw F(0) for POTENTIAL SETUP,
     * and the RESULT line's start/best values. Reads stored values only. */
    core::Context reference_context=ctx;
    models::DC_Selector reference_selector(reference_context);
    const auto reference_mode=reference_selector.dmb->embf0_mode();
    const bool has_ream=reference_mode.has_value();
    const bool embf0_zero=reference_mode==models::EmbF0Mode::Zero;
    std::optional<RawEmbeddingF0> embf0_seed;
    if (has_ream) embf0_seed = derive_raw_embedding_f0(ctx);
    /* The journal's setup sections: the run's alone (the hint below is the
     * check's too). */
    if (!check) {
        log.write_potential_section(ctx, host.train_size(), host.val_size(), elem_str,
                                    embf0_seed ? &*embf0_seed : nullptr);
        {
            HPOLog::FitSwitchReport fit;
            fit.force          = ctx.get<bool>("FORCE");
            fit.force_source   = fit_sources.force;
            fit.stress         = ctx.get<bool>("STRESS");
            fit.stress_source  = fit_sources.stress;
            fit.has_validation = !val.empty();
            fit.val_forces     = trg.use_frmse;
            fit.val_stresses   = trg.use_srmse;
            log.write_fit_switches(fit);
        }
        if (has_ream) log.arm_embf0(embf0_zero, host.embf0_diagnostic());
        /* DM_REAM density report (a bare DM_REAM many-body term): the -c
         * potential's and each WARM seed's rho_raw(r) -- and rhobar(a) under
         * RHOBARGATE -- in POTENTIAL SETUP; RESULT repeats it for the best.
         * Stored values only; never throws. */
        if (detail::context_is_bare_ream(ctx))
            log.write_density_report(detail::startup_density_report(ctx, trg),
                                     trg.RHOBARGATE.has_value());
    }
    {
        const auto& tr = host.load_transform_report();
        const auto& vr = host.val_load_transform_report();
        if (!check) {
            const std::size_t n_train_loaded =
                host.train_size() + tr.n_dropped_energy + tr.n_dropped_force;
            const std::size_t n_val_loaded =
                host.val_size() + vr.n_dropped_energy + vr.n_dropped_force;
            log.write_load_transforms_section(tr, n_train_loaded,
                                              vr, n_val_loaded);
        }

        /* EMBF0 ZERO fixes the DM_REAM isolated-atom energy at 0: fitting
         * absolute reference energies then needs another per-atom constant.
         * n_intercept_columns > 0, from stored values (no BIAS default
         * inserted). The toggles are named with their options: a -c seed
         * is a potential, whose own toggle line is not read
         * (mlip::potential_line_not_read). */
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
            log.hint(std::string(ctx.blocks("DESCRIPTOR").empty()?"EMBF0 ZERO":"REFERENCE ZERO")+" fixes the "+reference_selector.dmb->label()+" isolated-atom energy at 0 "
                     "and the model has no other per-atom constant (BIAS "
                     "false, no ESHIFT). If reference energies are absolute "
                     "(e.g. DFT totals), set ESHIFT or give the run "
                     "ESHIFT_ATOM or ESHIFT_DBATOM (--eshift-atom, "
                     "--eshift-dbatom: a potential's own line of them is not "
                     "read), all compatible with EAMFAST. BIAS true also "
                     "adds a per-atom constant, but its intercept columns, "
                     "one per ATOM species, precede the descriptor weights "
                     "(FIXINDEX/FIXWEIGHT indices shift) and it disables "
                     "EAMFAST.");
    }
    if (!check) log.write_loss_section(trg);

    /* -- IterHealthMonitor mode --------------------------------------- */
    {
        const HealthLogMode hlm = health_log_mode_from_context(ctx);
        if (!check) {
            global_iter_health_monitor().set_mode(hlm);
            global_iter_health_monitor().clear_state();
        }
    }

    /* -- Pre-flight audit ---------------------------------------------
     * Walk every configured descriptor's meta() and run the per-param /
     * cross-param HealthChecks on the training StructureDB. A fixed
     * `CGRIDMB` outside the data's support is flagged here with a
     * rho-collapse explanation BEFORE the optimiser runs. The
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
     * generation on stagnation. */
    auto policy = build_run_policy<V,R>(sp, &trg);

    /* A check ends here, every startup check passed: the summary, with
     * the warnings and hints of the startup, and no evaluation. */
    if (check) {
        write_check_summary(std::cout, ctx.get<std::string>("HPOTARGET"),
                            trg, host.train_size(), host.val_size(),
                            log.notes_without_journal());
        return EXIT_SUCCESS;
    }

    /* The stagnation threshold and top-K tracking are process-wide statics
     * that each policy arms inside run() (RESTART, PRESTAGE). A policy that
     * ended by an exception in an earlier run in this process can leave
     * them armed -- a stale window would stop a PLAIN run with a
     * StagnationStop. Every run starts disarmed. */
    HPO_Host<V,R>::set_stagnation_threshold(0);
    HPO_Host<V,R>::set_track_topk(0);

    /* Everything from here publishes the best potential or evaluates
     * candidates, so it runs under drive_run() (run_driver.h), which owns
     * every way the run can end: the result queue is flushed and the journal
     * trailer written whether the optimiser finishes, a signal stops it (the
     * return value is then 128 + the signal number), or an error escapes
     * (rethrown after the trailer). NOISE_CALIBRATION is inside for the same
     * reason: its evaluations are real, and a signal during them must end
     * the run the same way. Each optimiser run records its own termination
     * reason with HPOLog::set_status(); the trailer shows the one of the run
     * that ended the whole run (under MULTISTART the last start, under
     * PRESTAGE the refinement), or RestartPolicy's when its MAX restarts
     * were used up. */
    const int status = drive_run(
        [&]() {
            /* Bind best-potential publication to this run. HPO runs are not
             * resumable, so every run starts fresh: stale best potentials
             * from an unrelated prior objective are discarded atomically
             * before the first evaluation. MPI ranks synchronize before any
             * evaluation; a rank whose reset failed makes every rank throw
             * the same durable_best::Error, so all of them reach drive_run's
             * collective and end "FAILED -- best-potential persistence:
             * <cause>" like any other persistence failure. */
            host.prepare_best_publication();
            /* best_pot.tadah prepared (its lock taken): the DUMP and BDUMP
             * directories, checked when the target file was read, are
             * created now, just before the first evaluation, so a run that
             * stops before creates none. */
            create_archive_directories_on_every_rank(trg);
            run_noise_calibration(sp, host, log);
            log.start_progress();
            std::cout << "[HPO] " << sp.lib << ":" << sp.algo
                      << "   dim=" << sp.low.size() << "\n";
            policy->run(sp, host);
        },
        host, log,
        /* After the run's collective every rank has passed it, so no rank
         * publishes the best potential any more: its lock files go,
         * whatever the ending, before the trailer, so that a lock file that
         * stays is a WARNING in the journal. */
        [&]() { host.finish_best_publication(); });

    /* The run's pools stop when `pools` goes, here or on any exception
     * above (RunPools): the PROCESS worker processes get their SHUTDOWN
     * frames and are reaped, the executor's workers joined. */
    return status;
}

}  // namespace
} // namespace tadah::hpo
