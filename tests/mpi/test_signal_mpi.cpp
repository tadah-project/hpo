/* test_signal_mpi.cpp -- a signal that one rank of a multi-rank
 * ANNEAL_PT / ANNEAL_PA run receives ends every rank as a signal stop.
 *
 * `test_hpo_signal_mpi <scenario>` runs one driver under drive_run() on
 * every rank, with a host that behaves as HPO_Host does on a signal: it
 * throws core::ShutdownRequested at the start of an evaluation once
 * core::SignalHandler::stop_requested() is set. One rank raise()s SIGTERM
 * at the start of a chosen evaluation of its own, so the signal reaches
 * that rank alone (as when it arrives at the ranks of a job at different
 * moments, or is sent to one process). CORE's stop flag is never lowered,
 * so each scenario is a process of its own: run_signal_mpi.cmake runs every
 * scenario at 1, 2 and 4 ranks, one CTest entry per scenario.
 *
 * The scenarios and the phase the signal falls in (calibration runs on
 * rank 0's driver thread, the pool phases on worker threads):
 *   pt_seed   PT: rank 0, the seed evaluation (seed/T0 handshake)
 *   pt_t0     PT: rank 0, the first T0 calibration probe
 *   pt_pass2  PT (CLASSIC, automatic T_HOT): rank 0, the first probe of
 *             calibration pass 2, after the automatic step warm-up
 *   pt_jitter PT, INIT_MODE JITTER AUTO: the last rank, its first jitter
 *             calibration probe (every rank probes its block of
 *             dimensions, then they gather)
 *   pt_init   PT: the last rank, its first population-initialization
 *             evaluation (ReplicaPool)
 *   pt_rung   PT: the last rank, its first replica-rung evaluation
 *   pa_seed   PA: rank 0, the seed evaluation
 *   pa_t0     PA: rank 0, the first T0 calibration probe
 *   pa_init   PA: the last rank, its first population-initialization
 *             evaluation
 *   pa_sweep  PA: the last rank, its first walker-sweep evaluation
 *
 * Pins, on every rank: drive_run() returns 128 + SIGTERM (143) and does not
 * rethrow; the journal (hpo_run.log.r<rank>, hpo_run.log on one rank)
 * records "Optimiser status : SHUTDOWN (SIGTERM)" on the rank that received
 * the signal and "SHUTDOWN (SIGTERM on MPI rank <r>)", naming that rank, on
 * the others; the signal was raised on the scenario's rank alone, and at an
 * evaluation of the scenario's phase. The phase is what the host saw at that
 * evaluation (Seen, phase_of), the trigger's evaluation count aside:
 *   seed evaluation      the driver thread, the rank's first evaluation, at
 *                        the seed point (optim_init) exactly
 *   T0 calibration probe the driver thread, after the seed and before any
 *                        worker-thread evaluation, away from the seed,
 *                        outside the jitter probe phase
 *   calibration pass 2   the driver thread, outside the jitter probe phase,
 *   probe                after every non-seed chain of the rank's block had
 *                        its population-initialization evaluation and a
 *                        later pool epoch evaluated one of them again (the
 *                        automatic step warm-up)
 *   jitter calibration   the driver thread inside the INIT_MODE JITTER AUTO
 *   probe                probe phase (probe::active())
 *   population           a worker thread, the first evaluation of a non-seed
 *   initialization       chain of the rank's block (the chain:
 *                        warmstart::current_chain(), the global slot the
 *                        drivers stamp before each pool task) while another
 *                        such chain was still unevaluated
 *   pool epoch after     a worker thread, after every non-seed chain of the
 *   population           rank's block had its population-initialization
 *   initialization       evaluation: in pt_rung a replica rung (with CORANA
 *                        the driver runs no step warm-up or pilot epoch:
 *                        anneal_core.h collapses STEP_MODE to FIXED), in
 *                        pa_sweep a walker sweep (PA has no other epoch)
 * On one rank every scenario is a pin: before the fix it passes as well.
 * The phase check is a pin too: it passes before the fix as well, and a
 * trigger moved into another phase fails it (pt_rung's or pa_sweep's at the
 * last rank's first worker evaluation: "population initialization", not
 * "pool epoch after population initialization"; pt_t0's or pa_t0's at rank
 * 0's first evaluation: "seed evaluation", not "T0 calibration probe").
 * Before the fix: on 2 and 4 ranks the ranks that did not receive the signal
 * ended with an error, rethrown by drive_run() (exit status 1, "FAILED --
 * ..."): "TADAH:ANNEAL_PT: rank 0 aborted during seed/T0 calibration"
 * (pt_seed, pt_t0), "... rank 0 aborted during calibration pass 2"
 * (pt_pass2), "TADAH:ANNEAL_P?: worker failed on another MPI rank during
 * population initialization" (pt_init, pa_init), "... during replica rung"
 * (pt_rung), "... during walker sweeps" (pa_sweep), "TADAH:ANNEAL_PA: rank 0
 * aborted during seed/T0 calibration" (pa_seed, pa_t0); in pt_jitter the
 * signalled rank left the calibration before its gathers and the job hung
 * (the runner's timeout). */
#include <tadah/hpo/algorithms_pa.h>
#include <tadah/hpo/algorithms_pt.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_mpi.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/probe_phase.h>
#include <tadah/hpo/run_driver.h>
#include <tadah/hpo/warmstart_cache.h>
#include <tadah/core/checkpoint.h>
#include <tadah/core/signal_handler.h>

#include <dlib/matrix.h>
#include <mpi.h>

#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

using Vec = dlib::matrix<double, 0, 1>;

namespace {

constexpr std::size_t kDim = 4;      // >= 4 ranks each probe a dimension
constexpr std::size_t kReplicas = 8;   // PT: >= the world size
constexpr unsigned kPopulation = 16;   // PA: >= the world size

/* When the signal is raised: on rank `rank`, at the start of its
 * `nth` evaluation on the driver thread (main) or on a worker thread, or
 * (after_worker) at its first driver-thread evaluation after any
 * worker-thread one. */
struct Trigger {
  int rank = 0;
  bool main = true;
  bool after_worker = false;
  std::uint64_t nth = 1;
};

/* What the host saw at the start of the evaluation that raised the signal
 * (phase_of names its phase). */
struct Seen {
  bool on_main = false;
  bool probe_phase = false;        // probe::active(): the jitter calibration
  bool at_seed = false;            // every coordinate the seed's
  std::uint64_t main_before = 0;   // the rank's driver-thread evaluations
  std::uint64_t worker_before = 0; // ... and worker-thread ones before it
  std::size_t chain = 0;           // worker: warmstart::current_chain()
  bool chain_in_block = false;     // ... a slot of this rank's block
  unsigned chain_before = 0;       // ... its worker evaluations before it
  bool init_done = false;  // every non-seed chain of the block evaluated
  bool chain_rerun = false;  // a chain of the block evaluated twice
};

constexpr const char* kSeed = "seed evaluation";
constexpr const char* kT0 = "T0 calibration probe";
constexpr const char* kPass2 = "calibration pass 2 probe";
constexpr const char* kJitter = "jitter calibration probe";
constexpr const char* kInit = "population initialization";
constexpr const char* kAfterInit = "pool epoch after population initialization";

std::string phase_of(const Seen& s)
{
  if (s.on_main) {
    if (s.probe_phase) return kJitter;
    if (s.at_seed)
      return s.main_before == 0 && s.worker_before == 0
                 ? kSeed : "a driver-thread evaluation at the seed again";
    if (s.worker_before == 0)
      return s.main_before > 0 ? kT0
                               : "a driver-thread evaluation before the seed";
    if (s.init_done && s.chain_rerun) return kPass2;
    return "a driver-thread evaluation after population initialization "
           "with no chain evaluated again";
  }
  if (!s.chain_in_block)
    return "a worker evaluation of slot " + std::to_string(s.chain) +
           ", outside the rank's block";
  if (!s.init_done)
    return s.chain != 0 && s.chain_before == 0
               ? kInit
               : "a repeated worker evaluation of slot " +
                     std::to_string(s.chain) +
                     " before population initialization ended";
  return kAfterInit;
}

struct SignalHost {
  int rank = 0;
  Trigger trigger;
  std::thread::id driver_thread = std::this_thread::get_id();
  std::vector<double> seed;                 // the seed point (optim_init)
  std::size_t block_lo = 0, block_hi = 0;   // this rank's global slots
  mutable std::mutex mu;                    // guards everything below
  mutable std::uint64_t main_evals = 0;
  mutable std::uint64_t worker_evals = 0;
  mutable std::vector<unsigned> chain_evals;  // worker evaluations per slot
  mutable bool raised = false;
  mutable Seen seen;                        // at the raising evaluation

  double operator()(const Vec& x) const
  {
    const bool on_main = std::this_thread::get_id() == driver_thread;
    bool fire = false;
    {
      std::lock_guard<std::mutex> lock(mu);
      if (chain_evals.size() != block_hi - block_lo)
        chain_evals.assign(block_hi - block_lo, 0);
      Seen now;
      now.on_main = on_main;
      now.probe_phase = tadah::hpo::probe::active();
      now.at_seed = static_cast<std::size_t>(x.size()) == seed.size();
      for (long i = 0; now.at_seed && i < x.size(); ++i)
        now.at_seed = x(i) == seed[static_cast<std::size_t>(i)];
      now.main_before = main_evals;
      now.worker_before = worker_evals;
      if (!on_main) {
        now.chain = tadah::hpo::warmstart::current_chain();
        now.chain_in_block = now.chain >= block_lo && now.chain < block_hi;
        if (now.chain_in_block) now.chain_before = chain_evals[now.chain - block_lo];
      }
      now.init_done = true;
      for (std::size_t c = block_lo; c < block_hi; ++c) {
        const unsigned n = chain_evals[c - block_lo];
        if (c != 0 && n == 0) now.init_done = false;
        if (n >= 2) now.chain_rerun = true;
      }
      const std::uint64_t n = on_main ? ++main_evals : ++worker_evals;
      if (!on_main && now.chain_in_block) ++chain_evals[now.chain - block_lo];
      if (rank == trigger.rank && !raised && on_main == trigger.main &&
          (trigger.after_worker ? now.worker_before > 0 : n == trigger.nth)) {
        raised = true;
        seen = now;
        fire = true;
      }
    }
    if (fire) std::raise(SIGTERM);   // CORE's handler records it (install())
    /* HPO_Host::operator(): a raised stop flag ends the evaluation. */
    if (tadah::core::SignalHandler::stop_requested())
      throw tadah::core::ShutdownRequested{};
    double f = 0.0;
    for (long i = 0; i < x.size(); ++i) {
      const double d = x(i) - 0.1 * static_cast<double>(i + 1);
      f += d * d;
    }
    return f;
  }
};

/* drive_run()'s host: whether this rank produced a usable loss; the
 * final flush succeeds. */
struct DriverHost {
  struct StagnationStop {};
  bool usable = true;
  void flush_remaining(bool /*retry_io*/ = true) const {}
  bool has_usable_loss() const { return usable; }
};

tadah::hpo::OptimizerSpec base_spec(const char* algo)
{
  tadah::hpo::OptimizerSpec sp;
  sp.low.assign(kDim, -1.0);
  sp.high.assign(kDim, 1.0);
  sp.is_int.assign(kDim, false);
  sp.log_scaled.assign(kDim, false);
  sp.optim_init = {std::vector<double>(kDim, 0.2)};
  sp.lib = "TADAH";
  sp.algo = algo;
  sp.seed = 20261007;
  sp.num_threads = 2;
  sp.stop.maxeval = 100000;   // the signal ends every scenario
  sp.param_raw["CORANA"] = "ON";
  sp.param["NS"] = 1.0;
  sp.param["T_HOT"] = 1.0;
  sp.param_raw["BOUNDARY_MODE"] = "REDRAW";
  return sp;
}

std::size_t slots_of(std::size_t total, int world, int r)
{
  return total / static_cast<std::size_t>(world) +
         (static_cast<std::size_t>(r) < total % static_cast<std::size_t>(world)
              ? 1 : 0);
}

/* The first global slot of rank r's block (the drivers' slot_offset). */
std::size_t slot_offset(std::size_t total, int world, int r)
{
  std::size_t off = 0;
  for (int q = 0; q < r; ++q) off += slots_of(total, world, q);
  return off;
}

/* Evaluations the last rank makes in population initialization: one per
 * slot of its block, but none for global slot 0 (the seed, rank 0's). */
std::uint64_t init_evals_of_last_rank(std::size_t total, int world)
{
  const std::size_t n = slots_of(total, world, world - 1);
  return world == 1 ? n - 1 : n;
}

struct Scenario {
  std::string name;
  bool pt = true;
  Trigger trigger;
  const char* phase = kSeed;   // phase_of() of the raising evaluation
  std::function<void(tadah::hpo::OptimizerSpec&)> configure;
};

Scenario scenario(const std::string& name, int world)
{
  const int last = world - 1;
  Scenario s;
  s.name = name;
  s.configure = [](tadah::hpo::OptimizerSpec&) {};
  if (name == "pt_seed") {
    s.trigger = {0, true, false, 1};
    s.phase = kSeed;
  } else if (name == "pt_t0") {
    s.trigger = {0, true, false, 2};
    s.phase = kT0;
    s.configure = [](tadah::hpo::OptimizerSpec& sp) { sp.param.erase("T_HOT"); };
  } else if (name == "pt_pass2") {
    s.trigger = {0, true, true, 0};
    s.phase = kPass2;
    s.configure = [](tadah::hpo::OptimizerSpec& sp) {
      /* CLASSIC with the automatic T_HOT and the FROZEN step mode: T0 pass
       * 1, the automatic step warm-up (ReplicaPool), then pass 2 on rank 0
       * (run_pt_auto of test_anneal_mpi.cpp). */
      sp.param_raw.erase("CORANA");
      sp.param.erase("NS");
      sp.param.erase("T_HOT");
    };
  } else if (name == "pt_jitter") {
    /* The seed is rank 0's first driver-thread evaluation; the jitter
     * probes come next, on every rank's driver thread. */
    s.trigger = {last, true, false, last == 0 ? 2u : 1u};
    s.phase = kJitter;
    s.configure = [](tadah::hpo::OptimizerSpec& sp) {
      sp.param_tokens["INIT_MODE"] = {"JITTER", "AUTO", "0.05"};
      sp.param_raw["INIT_MODE"] = "JITTER";
    };
  } else if (name == "pt_init") {
    s.trigger = {last, false, false, 1};
    s.phase = kInit;
  } else if (name == "pt_rung") {
    s.trigger = {last, false, false,
                 init_evals_of_last_rank(kReplicas, world) + 1};
    s.phase = kAfterInit;
  } else if (name == "pa_seed") {
    s.pt = false;
    s.trigger = {0, true, false, 1};
    s.phase = kSeed;
  } else if (name == "pa_t0") {
    s.pt = false;
    s.trigger = {0, true, false, 2};
    s.phase = kT0;
    s.configure = [](tadah::hpo::OptimizerSpec& sp) { sp.param.erase("T_HOT"); };
  } else if (name == "pa_init") {
    s.pt = false;
    s.trigger = {last, false, false, 1};
    s.phase = kInit;
  } else if (name == "pa_sweep") {
    s.pt = false;
    s.trigger = {last, false, false,
                 init_evals_of_last_rank(kPopulation, world) + 1};
    s.phase = kAfterInit;
  } else {
    throw std::invalid_argument("unknown scenario '" + name + "'");
  }
  return s;
}

std::string read_file(const std::string& path)
{
  std::ifstream in(path);
  std::ostringstream body;
  body << in.rdbuf();
  return body.str();
}

/* "Optimiser status : <value>" of a journal ("" if it has none). */
std::string journal_status(const std::string& journal)
{
  std::istringstream in(journal);
  for (std::string line; std::getline(in, line);) {
    const auto key = line.find("Optimiser status");
    if (key == std::string::npos) continue;
    const auto colon = line.find(": ", key);
    if (colon != std::string::npos) return line.substr(colon + 2);
  }
  return {};
}

}  // namespace

int main(int argc, char** argv)
{
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  int rank = 0, world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  /* Collective: every rank's verdict, rank 0 reporting each failed one. */
  int failures = 0;
  auto check_all = [&](bool ok, const std::string& what) {
    const int bad = ok ? 0 : 1;
    std::vector<int> all(static_cast<std::size_t>(world), 0);
    MPI_Gather(&bad, 1, MPI_INT, all.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    int any = bad;
    MPI_Allreduce(&bad, &any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (any) {
      ++failures;
      if (rank == 0) {
        std::cerr << "MPI signal regression (" << world << " rank(s)): "
                  << what << " -- failed on rank(s):";
        for (int r = 0; r < world; ++r)
          if (all[static_cast<std::size_t>(r)]) std::cerr << ' ' << r;
        std::cerr << '\n';
      }
    }
  };

  try {
    if (argc != 2)
      throw std::invalid_argument("usage: test_hpo_signal_mpi <scenario>");
    const Scenario s = scenario(argv[1], world);

    /* A scratch directory per scenario and rank count: the journals and the
     * drivers' diagnostic files of concurrent CTest entries stay apart. */
    const std::filesystem::path dir =
        std::filesystem::current_path() /
        ("hpo_signal_mpi_" + s.name + "_n" + std::to_string(world));
    if (rank == 0) std::filesystem::create_directories(dir);
    MPI_Barrier(MPI_COMM_WORLD);
    std::filesystem::current_path(dir);

    tadah::core::SignalHandler::install();
    auto& log = tadah::hpo::HPOLog::instance();
    const std::string journal = "hpo_run.log" + tadah::hpo::hpo_rank_suffix();
    log.open(journal, "test", dir.string(), static_cast<int>(::getpid()));
    log.set_rank_scope(rank, world);

    auto sp = base_spec(s.pt ? "ANNEAL_PT" : "ANNEAL_PA");
    if (s.pt) {
      sp.param["REPLICAS"] = static_cast<double>(kReplicas);
    } else {
      sp.pop = kPopulation;
      sp.param["NT"] = 1.0;
      sp.param["T_MIN"] = 1e-8;
    }
    s.configure(sp);
    /* A pinned T_COLD needs the pinned T_HOT above it; with the automatic
     * T_HOT the ladder's cold end is derived (LADDER_SPAN). */
    if (s.pt && sp.param.count("T_HOT")) sp.param["T_COLD"] = 0.1;

    SignalHost host;
    host.rank = rank;
    host.trigger = s.trigger;
    host.seed = sp.optim_init.front();
    const std::size_t total = s.pt ? kReplicas : kPopulation;
    host.block_lo = slot_offset(total, world, rank);
    host.block_hi = host.block_lo + slots_of(total, world, rank);
    DriverHost dhost;

    int status = -1;
    bool rethrown = false;
    std::string what;
    try {
      status = tadah::hpo::drive_run(
          [&] {
            if (s.pt) tadah::hpo::run_anneal_pt_impl<Vec>(sp, host);
            else      tadah::hpo::run_anneal_pa_impl<Vec>(sp, host);
          },
          dhost, log);
    } catch (const std::exception& e) {
      rethrown = true;
      what = e.what();
    } catch (...) {
      rethrown = true;
      what = "non-standard exception";
    }

    const int signalled = s.trigger.rank;
    if (rethrown)
      std::cerr << "rank " << rank << ": drive_run rethrew: " << what << '\n';
    check_all(!rethrown, s.name + ": drive_run rethrew an error");
    check_all(status == 128 + SIGTERM,
              s.name + ": drive_run did not return 128 + SIGTERM");
    check_all(host.raised == (rank == signalled),
              s.name + ": the signal was not raised on rank " +
                  std::to_string(signalled) + " alone");
    const std::string phase = host.raised ? phase_of(host.seen) : "";
    if (rank == signalled && phase != s.phase)
      std::cerr << "rank " << rank << ": the signal fell in '" << phase
                << "', not '" << s.phase << "'\n";
    check_all(rank != signalled || phase == s.phase,
              s.name + ": the signal did not fall in the phase '" +
                  std::string(s.phase) + "'");
    const std::string expected =
        rank == signalled
            ? std::string("SHUTDOWN (SIGTERM)")
            : "SHUTDOWN (SIGTERM on MPI rank " + std::to_string(signalled) +
                  ")";
    const std::string got = journal_status(read_file(journal));
    if (got != expected)
      std::cerr << "rank " << rank << ": journal status '" << got
                << "', expected '" << expected << "'\n";
    check_all(got == expected,
              s.name + ": a journal does not end 'SHUTDOWN (SIGTERM)' (the "
                       "rank that received it) or 'SHUTDOWN (SIGTERM on MPI "
                       "rank " + std::to_string(signalled) + ")' (the others)");

    if (rank == 0 && failures == 0)
      std::cout << "HPO_MPI_SIGNAL " << s.name << " WORLD " << world
                << " OK\n";
  } catch (const std::exception& e) {
    std::cerr << "MPI signal test failed on rank " << rank << ": " << e.what()
              << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }

  if (failures) {
    std::cout.flush();
    std::cerr.flush();
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  MPI_Finalize();
  return 0;
}
