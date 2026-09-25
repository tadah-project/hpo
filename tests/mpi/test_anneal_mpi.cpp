#include <tadah/hpo/algorithms_pa.h>
#include <tadah/hpo/anneal_jitter.h>
#include <tadah/hpo/algorithms_pt.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_mpi.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/run_driver.h>
#include <tadah/core/checkpoint.h>
#include <tadah/core/signal_handler.h>

#include <dlib/matrix.h>
#include <mpi.h>

#include <atomic>
#include <cmath>
#include <functional>
#include <optional>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

using Vec = dlib::matrix<double, 0, 1>;

namespace {

struct Signature {
  std::uint64_t count = 0;
  std::uint64_t sum = 0;
  std::uint64_t xors = 0;
};

/* FNV-1a over a whole file. Used on hpo_corana_acceptance.tsv: the eval
 * signature above proves the TRAJECTORY is rank-invariant, but says nothing
 * about a diagnostic assembled by a gatherv -- a census that mis-ordered or
 * dropped a rank's slot block would hash differently at 2 and 4 ranks while
 * every evaluated point still matched. Returns {rows, hash}; rows is carried
 * so an empty or missing file cannot pass as agreement. */
std::pair<std::uint64_t, std::uint64_t> file_digest(const char* path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) return {0, 0};
  std::ostringstream body;
  body << in.rdbuf();
  const std::string text = body.str();
  std::uint64_t h = 1469598103934665603ULL, rows = 0;
  for (unsigned char c : text) {
    if (c == '\n') ++rows;
    h = (h ^ c) * 1099511628211ULL;
  }
  return {rows, h};
}

std::uint64_t point_hash(const Vec& x)
{
  std::uint64_t h = 1469598103934665603ULL;
  for (long i = 0; i < x.size(); ++i) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(double));
    const double value = x(i);
    std::memcpy(&bits, &value, sizeof(bits));
    h ^= bits;
    h *= 1099511628211ULL;
  }
  return h;
}

struct RecordingHost {
  int rank = 0;
  int world = 1;
  bool remote_wins = false;
  bool throw_on_last_rank = false;
  mutable std::mutex mutex;
  mutable std::vector<std::uint64_t> hashes;

  RecordingHost(int rank_in, int world_in, bool remote_wins_in = false,
                bool throw_on_last_rank_in = false)
      : rank(rank_in), world(world_in), remote_wins(remote_wins_in),
        throw_on_last_rank(throw_on_last_rank_in) {}

  double operator()(const Vec& x) const
  {
    if (throw_on_last_rank && world > 1 && rank == world - 1)
      throw std::runtime_error("injected remote worker failure");
    {
      std::lock_guard<std::mutex> lock(mutex);
      hashes.push_back(point_hash(x));
    }
    const double dx = x(0) - 0.17;
    const double dy = x(1) + 0.29;
    const double base = dx * dx + dy * dy;
    return remote_wins && world > 1 && rank == world - 1
        ? base - 1000.0 : base;
  }

  Signature global_signature() const
  {
    Signature local;
    {
      std::lock_guard<std::mutex> lock(mutex);
      local.count = static_cast<std::uint64_t>(hashes.size());
      for (const auto h : hashes) {
        local.sum += h;
        local.xors ^= h;
      }
    }
    Signature global;
    MPI_Allreduce(&local.count, &global.count, 1, MPI_UINT64_T, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local.sum, &global.sum, 1, MPI_UINT64_T, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local.xors, &global.xors, 1, MPI_UINT64_T, MPI_BXOR,
                  MPI_COMM_WORLD);
    return global;
  }
};

tadah::hpo::OptimizerSpec base_spec(const char* algo, std::uint64_t maxeval)
{
  tadah::hpo::OptimizerSpec sp;
  sp.low = {-1.0, -1.0};
  sp.high = {1.0, 1.0};
  sp.is_int.assign(2, false);
  sp.log_scaled.assign(2, false);
  sp.optim_init = {{0.2, -0.1}};
  sp.lib = "TADAH";
  sp.algo = algo;
  sp.seed = 20260809;
  sp.num_threads = 2;
  sp.stop.maxeval = maxeval;
  sp.param_raw["CORANA"] = "ON";
  sp.param["NS"] = 1.0;
  /* NT is serial/PA-only now (ANNEAL_PT runs one sweep per epoch);
   * run_pa() adds it back for the population annealer. */
  sp.param["T_HOT"] = 1.0;
  sp.param_raw["BOUNDARY_MODE"] = "REDRAW";   /* enum since 1.5.0 */
  return sp;
}

Signature run_pt(std::uint64_t cap, RecordingHost& host,
                 bool auto_t0 = false, bool stop_on_remote = false)
{
  auto sp = base_spec("ANNEAL_PT", cap);
  sp.param["REPLICAS"] = 8.0;
  sp.param["T_COLD"] = 0.1;
  if (auto_t0) sp.param.erase("T_HOT");
  if (stop_on_remote) sp.stop.stopval = -100.0;
  tadah::hpo::run_anneal_pt_impl<Vec>(sp, host);
  return host.global_signature();
}

Signature run_pt_adaptive(std::uint64_t cap, RecordingHost& host)
{
  auto sp = base_spec("ANNEAL_PT", cap);
  sp.param["REPLICAS"] = 8.0;
  sp.param["T_COLD"] = 0.1;
  sp.param["LADDER_POWER"] = 0.5;
  sp.param_raw["LADDER_ADAPT"] = "ON";
  sp.param["LADDER_ADAPT_EVERY"] = 1.0;
  sp.param["LADDER_ADAPT_GAIN"] = 1.0;
  sp.param["LADDER_ADAPT_DECAY"] = 0.6;
  tadah::hpo::run_anneal_pt_impl<Vec>(sp, host);
  return host.global_signature();
}

Signature run_pt_auto(std::uint64_t cap, RecordingHost& host)
{
  auto sp = base_spec("ANNEAL_PT", cap);
  /* param_raw, not param: CORANA became an ON/OFF token in 1.5.0 and
   * base_spec sets it there. Erasing the (now unused) numeric entry left
   * CORANA ON, and under CORANA the driver SKIPS the step warm-up
   * entirely -- so everything this case claims to cover below was in fact
   * unreachable. */
  sp.param_raw.erase("CORANA");
  sp.param.erase("NS");
  /* T_HOT is erased too: with STEP_MODE at its FROZEN default this
   * exercises the full auto path -- T0 pass 1, per-slot coordinate
   * warm-up + pilot with their cross-rank diagnostic gathers, and the
   * pass-2 re-anchoring with the hot slot's calibrated kernel (which
   * includes a bcast from the hot slot's owner rank). Rank-invariance
   * of the eval signature covers all of it. */
  sp.param.erase("T_HOT");
  sp.param["REPLICAS"] = 8.0;
  tadah::hpo::run_anneal_pt_impl<Vec>(sp, host);
  return host.global_signature();
}

Signature run_pa(std::uint64_t cap, RecordingHost& host,
                 bool auto_t0 = false, bool stop_on_remote = false)
{
  auto sp = base_spec("ANNEAL_PA", cap);
  sp.pop = 16;
  sp.param["NT"] = 1.0;
  sp.param["T_MIN"] = 1e-8;
  if (auto_t0) sp.param.erase("T_HOT");
  if (stop_on_remote) sp.stop.stopval = -100.0;
  tadah::hpo::run_anneal_pa_impl<Vec>(sp, host);
  return host.global_signature();
}

/* A host for drive_run(): whether this rank produced a usable loss, and
 * whether its final flush fails. */
struct DriverHost {
  struct StagnationStop {};
  bool usable = false;
  bool flush_fails = false;
  void flush_remaining() const
  {
    if (flush_fails) throw std::runtime_error("injected final flush failure");
  }
  bool has_usable_loss() const { return usable; }
};

void require_all(bool condition, const std::string& message, int rank)
{
  const int local = condition ? 0 : 1;
  int failed = 0;
  MPI_Allreduce(&local, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (failed) {
    if (rank == 0) std::cerr << "MPI annealing regression: " << message << '\n';
    throw std::runtime_error(message);
  }
}

}  // namespace

int main(int argc, char** argv)
{
  const std::string preinit_suffix = tadah::hpo::hpo_rank_suffix();
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  int rank = 0, world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  try {
    require_all(preinit_suffix.empty(),
                "pre-MPI-init rank suffix must be empty", rank);
    require_all(provided >= MPI_THREAD_MULTIPLE,
                "MPI_THREAD_MULTIPLE is required", rank);
    const std::string suffix = tadah::hpo::hpo_rank_suffix();
    require_all(world == 1 ? suffix.empty()
                           : suffix == ".r" + std::to_string(rank),
                "rank-local suffix is not explicit", rank);

    const std::size_t mpi_int_limit =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    require_all(tadah::hpo::hpo_mpi_checked_count(
                    mpi_int_limit, "MPI helper regression") ==
                    std::numeric_limits<int>::max(),
                "MPI count helper rejected INT_MAX", rank);
    bool count_overflow_rejected = false;
    try {
      (void)tadah::hpo::hpo_mpi_checked_count(
          mpi_int_limit + 1, "MPI helper regression");
    } catch (const std::length_error&) {
      count_overflow_rejected = true;
    }
    require_all(count_overflow_rejected,
                "MPI count helper narrowed a value above INT_MAX", rank);

    const auto displacements = tadah::hpo::hpo_mpi_checked_displacements(
        {2, 0, 3}, "MPI displacement regression");
    require_all(displacements == std::vector<int>{0, 2, 2},
                "MPI displacement helper returned wrong offsets", rank);
    bool negative_count_rejected = false;
    try {
      (void)tadah::hpo::hpo_mpi_checked_displacements(
          {1, -1}, "MPI displacement regression");
    } catch (const std::invalid_argument&) {
      negative_count_rejected = true;
    }
    require_all(negative_count_rejected,
                "MPI displacement helper accepted a negative count", rank);
    bool displacement_overflow_rejected = false;
    try {
      (void)tadah::hpo::hpo_mpi_checked_displacements(
          {std::numeric_limits<int>::max(), 1},
          "MPI displacement regression");
    } catch (const std::length_error&) {
      displacement_overflow_rejected = true;
    }
    require_all(displacement_overflow_rejected,
                "MPI displacement helper overflowed the aggregate count", rank);

    tadah::hpo::pt_detail::validate_mpi_payload_dimension(
        mpi_int_limit - 2);
    bool pt_payload_overflow_rejected = false;
    try {
      tadah::hpo::pt_detail::validate_mpi_payload_dimension(
          mpi_int_limit - 1);
    } catch (const std::length_error&) {
      pt_payload_overflow_rejected = true;
    }
    require_all(pt_payload_overflow_rejected,
                "PT accepted a dimension whose calibration payload exceeds INT_MAX",
                rank);

    tadah::hpo::pa_detail::validate_mpi_calibration_dimension(
        mpi_int_limit - 2);
    bool pa_calibration_overflow_rejected = false;
    try {
      tadah::hpo::pa_detail::validate_mpi_calibration_dimension(
          mpi_int_limit - 1);
    } catch (const std::length_error&) {
      pa_calibration_overflow_rejected = true;
    }
    require_all(
        pa_calibration_overflow_rejected,
        "PA accepted a dimension whose calibration payload exceeds INT_MAX",
        rank);

    tadah::hpo::pa_detail::validate_mpi_population_state(0, mpi_int_limit);
    bool pa_population_overflow_rejected = false;
    try {
      tadah::hpo::pa_detail::validate_mpi_population_state(
          0, mpi_int_limit + 1);
    } catch (const std::length_error&) {
      pa_population_overflow_rejected = true;
    }
    require_all(pa_population_overflow_rejected,
                "PA accepted POPULATION above the MPI int limit", rank);

    const std::size_t exact_stride_dimension = (mpi_int_limit - 1) / 2;
    tadah::hpo::pa_detail::validate_mpi_population_state(
        exact_stride_dimension, 1);
    constexpr std::size_t ordinary_stride = 7;  // 2*dim+1 for dim=3
    const std::size_t largest_population = mpi_int_limit / ordinary_stride;
    tadah::hpo::pa_detail::validate_mpi_population_state(
        3, largest_population);
    bool pa_state_overflow_rejected = false;
    try {
      tadah::hpo::pa_detail::validate_mpi_population_state(
          3, largest_population + 1);
    } catch (const std::length_error&) {
      pa_state_overflow_rejected = true;
    }
    require_all(pa_state_overflow_rejected,
                "PA accepted aggregate state above the MPI int limit", rank);

    bool pa_stride_overflow_rejected = false;
    try {
      tadah::hpo::pa_detail::validate_mpi_population_state(
          std::numeric_limits<std::size_t>::max(), 1);
    } catch (const std::length_error&) {
      pa_stride_overflow_rejected = true;
    }
    require_all(pa_stride_overflow_rejected,
                "PA overflowed size_t while forming its state stride", rank);

    if (world > 1) {
      auto oversized = base_spec("ANNEAL_PA", 137);
      oversized.pop = std::numeric_limits<unsigned>::max();
      oversized.param["NT"] = 1.0;
      oversized.param["T_MIN"] = 1e-8;
      RecordingHost untouched{rank, world};
      bool rejected_before_host = false;
      try {
        tadah::hpo::run_anneal_pa_impl<Vec>(oversized, untouched);
      } catch (const std::length_error&) {
        rejected_before_host = true;
      }
      require_all(rejected_before_host,
                  "PA did not reject an oversized MPI state in its prologue",
                  rank);
      require_all(untouched.global_signature().count == 0,
                  "PA evaluated the host before rejecting an oversized MPI state",
                  rank);
    }

    /* ---- INIT_MODE JITTER AUTO is rank-invariant ------------------------
     * The calibration splits DIMENSIONS across ranks and broadcasts the
     * result. The broadcast is the correctness-critical half: every rank
     * seeds its OWN replicas, so if each kept only its own block the ranks
     * would jitter their slots from vectors nobody else agrees on and the
     * ensemble would stop being one ensemble.
     *
     * The kernel consumes no RNG and the anchor (x0, f0) is shared, so the
     * measured sigma must be bit-identical on every rank AND independent of
     * how many ranks there are. Both are checked. */
    {
      auto jitter_spec = base_spec("ANNEAL_PT", 40);
      jitter_spec.param["REPLICAS"] = 8.0;   /* PT needs REPLICAS >= world */
      jitter_spec.param["T_COLD"] = 0.1;
      jitter_spec.param_tokens["INIT_MODE"] = {"JITTER", "AUTO", "0.05"};
      jitter_spec.param_raw["INIT_MODE"] = "JITTER";
      RecordingHost jitter_host{rank, world};
      tadah::hpo::run_anneal_pt_impl<Vec>(jitter_spec, jitter_host);

      const auto kb = tadah::hpo::anneal::Knobs::parse(
          jitter_spec, tadah::hpo::anneal::AlgoKind::PT);
      require_all(kb.init_jitter_mode ==
                      tadah::hpo::anneal::Knobs::JitterMode::Auto,
                  "AUTO jitter mode did not survive the multi-token PARAM",
                  rank);

      /* Re-run the calibration directly so the sigma vector is inspectable,
       * then compare rank 0's copy against every other rank's bit for bit. */
      Vec x0(2);
      x0(0) = 0.2; x0(1) = -0.1;
      const double f0 = jitter_host(x0);
      std::atomic<bool> stop_flag{false};
      tadah::hpo::anneal::SharedBudget budget(1000000);
      std::function<bool()> never = [] { return false; };
      tadah::hpo::anneal::StopCtx jctx{budget, stop_flag, never, std::nullopt};
      const std::vector<double> fallback(2, 0.1);
      const auto plan = tadah::hpo::anneal::run_jitter_calibration<Vec>(
          jitter_spec, kb, x0, f0, fallback, jitter_host, jctx, 0.0,
          "ANNEAL_PT", rank, world);

      /* Guard against a VACUOUS test: if the calibration abandoned (or never
       * probed), every rank would trivially hold the same fallback vector
       * and the rank-invariance check below would prove nothing. */
      require_all(!plan.abandoned,
                  "AUTO jitter calibration abandoned -- rank-invariance check "
                  "would be vacuous", rank);
      require_all(plan.probes > 0,
                  "AUTO jitter calibration spent no probes -- rank-invariance "
                  "check would be vacuous", rank);
      require_all(plan.sigma.size() == 2,
                  "AUTO jitter plan lost dimensions across the gather", rank);
      std::vector<double> from_root = plan.sigma;
      tadah::hpo::hpo_bcast(from_root);
      require_all(from_root == plan.sigma,
                  "AUTO jitter sigma differs between ranks -- the broadcast "
                  "after the per-rank gather is missing or wrong", rank);
      for (double s : plan.sigma)
        require_all(std::isfinite(s) && s > 0.0,
                    "AUTO jitter sigma is not a usable displacement", rank);
    }

    RecordingHost pt{rank, world};
    const Signature pt_sig = run_pt(113, pt);
    require_all(pt_sig.count == 113, "PT did not consume exact MAXEVAL", rank);
    /* Snapshot the Corana census this run just wrote, before the next run
     * overwrites it. base_spec() sets CORANA ON and REPLICAS 8, so the file
     * must exist with a header plus 8 slots x 2 dimensions = 17 lines --
     * asserted here so a silently missing file cannot make the cross-rank
     * comparison below vacuous. */
    const auto census_digest = file_digest("hpo_corana_acceptance.tsv");
    require_all(census_digest.first == 17,
                "Corana acceptance census is missing or the wrong size", rank);

    RecordingHost pt_adaptive{rank, world};
    const Signature pt_adaptive_sig = run_pt_adaptive(113, pt_adaptive);
    require_all(pt_adaptive_sig.count == 113,
                "adaptive PT did not consume exact MAXEVAL", rank);

    RecordingHost pt_auto{rank, world};
    const Signature pt_auto_sig = run_pt_auto(113, pt_auto);
    require_all(pt_auto_sig.count == 113,
                "automatic CLASSIC PT did not consume exact MAXEVAL", rank);

    RecordingHost pa{rank, world};
    const Signature pa_sig = run_pa(137, pa);
    require_all(pa_sig.count == 137, "PA did not consume exact MAXEVAL", rank);

    RecordingHost pt_zero{rank, world};
    require_all(run_pt(0, pt_zero).count == 0,
                "PT evaluated with MAXEVAL=0", rank);
    RecordingHost pa_zero{rank, world};
    require_all(run_pa(0, pa_zero).count == 0,
                "PA evaluated with MAXEVAL=0", rank);

    RecordingHost pt_probe{rank, world};
    require_all(run_pt(3, pt_probe, true).count == 3,
                "PT seed/T0 probes did not stop at exact MAXEVAL", rank);
    RecordingHost pa_probe{rank, world};
    require_all(run_pa(3, pa_probe, true).count == 3,
                "PA seed/T0 probes did not stop at exact MAXEVAL", rank);

    if (world > 1) {
      RecordingHost pt_remote{rank, world, true, false};
      require_all(run_pt(8, pt_remote).count == 8,
                  "PT remote winner caused a final extra evaluation", rank);
      RecordingHost pa_remote{rank, world, true, false};
      require_all(run_pa(16, pa_remote).count == 16,
                  "PA remote winner caused a final extra evaluation", rank);

      RecordingHost pt_stop{rank, world, true, false};
      const Signature pt_stop_sig = run_pt(113, pt_stop, false, true);
      require_all(pt_stop_sig.count > 1 && pt_stop_sig.count <= 8,
                  "PT remote STOPVAL was not propagated after initialization",
                  rank);
      RecordingHost pa_stop{rank, world, true, false};
      const Signature pa_stop_sig = run_pa(137, pa_stop, false, true);
      require_all(pa_stop_sig.count > 1 && pa_stop_sig.count <= 16,
                  "PA remote STOPVAL was not propagated after initialization",
                  rank);

      auto check_remote_failure = [&](bool pt_mode) {
        RecordingHost failing{rank, world, false, true};
        int caught = 0;
        try {
          if (pt_mode) (void)run_pt(8, failing);
          else         (void)run_pa(16, failing);
        } catch (const std::exception&) {
          caught = 1;
        }
        int caught_total = 0;
        MPI_Allreduce(&caught, &caught_total, 1, MPI_INT, MPI_SUM,
                      MPI_COMM_WORLD);
        require_all(caught_total == world,
                    std::string(pt_mode ? "PT" : "PA") +
                        " worker exception did not reach every rank",
                    rank);
      };
      check_remote_failure(true);
      check_remote_failure(false);
    }

    /* ---- drive_run(): one agreement on a usable loss over the ranks -----
     * drive_run() asks every rank, in one collective, whether any produced
     * a usable loss: a run in which one did ends as a completed one on
     * every rank (best_pot.tadah is global), one in which none did ends
     * with status 1 on every rank. Every ending reaches that collective
     * once, before the rank-local final flush, so a rank whose flush fails,
     * or whose run an error or a signal ends after the drivers' last
     * collective, leaves no rank waiting in it; each rank still ends its
     * own way. */
    {
      const bool last = rank == world - 1;
      struct Outcome {
        int status = -1;
        bool rethrown = false;
      };
      const auto drive = [](bool usable, bool flush_fails,
                            const std::function<void()>& body) {
        DriverHost host;
        host.usable = usable;
        host.flush_fails = flush_fails;
        Outcome o;
        try {
          o.status = tadah::hpo::drive_run(body, host,
                                           tadah::hpo::HPOLog::instance());
        } catch (...) {
          o.rethrown = true;
        }
        return o;
      };
      const auto completed = [](const Outcome& o) {
        return !o.rethrown && o.status == 0;
      };
      const std::function<void()> nothing = [] {};

      Outcome o = drive(last, false, nothing);
      require_all(completed(o),
                  "drive_run: a usable loss on one rank did not end the run "
                  "as a completed one on every rank", rank);

      o = drive(false, false, nothing);
      require_all(!o.rethrown && o.status == EXIT_FAILURE,
                  "drive_run: a run without a usable loss on any rank did "
                  "not end with status 1 on every rank", rank);

      o = drive(rank == 0, last, nothing);
      require_all(last ? o.rethrown : completed(o),
                  "drive_run: a failed final flush on the last rank did not "
                  "end that rank with its error and the others as before",
                  rank);

      o = drive(rank == 0, false, [last] {
        if (last) throw tadah::core::ShutdownRequested{};
      });
      require_all(last ? (!o.rethrown &&
                          o.status == tadah::hpo::shutdown_exit_status(
                                          tadah::core::SignalHandler::
                                              signal_seen()))
                       : completed(o),
                  "drive_run: a shutdown of the last rank after the drivers' "
                  "last collective did not end that rank with its status and "
                  "the others as before", rank);

      o = drive(rank == 0, false, [last] {
        if (last) throw std::runtime_error("injected error");
      });
      require_all(last ? o.rethrown : completed(o),
                  "drive_run: an error on the last rank after the drivers' "
                  "last collective did not end that rank with its error and "
                  "the others as before", rank);
    }

    if (rank == 0) {
      /* The harness compares signatures across 1/2/4 ranks, which proves
       * nothing if the launcher silently started N independent 1-rank jobs
       * instead of one N-rank job -- a real hazard when several MPI
       * installations are on PATH and the one matching the linked libmpi is
       * not the one CMake found. Report the world size so the driver script
       * can insist it got the ranks it asked for. */
      std::cout << "ANNEAL_MPI_WORLD " << world << '\n';
      std::cout << "ANNEAL_MPI_CENSUS " << census_digest.first << ':'
                << census_digest.second << '\n';
      std::cout << "ANNEAL_MPI_SIGNATURE PT:" << pt_sig.count << ':'
                << pt_sig.sum << ':' << pt_sig.xors
                << " PT_ADAPT:" << pt_adaptive_sig.count << ':'
                << pt_adaptive_sig.sum << ':'
                << pt_adaptive_sig.xors
                << " PT_AUTO:" << pt_auto_sig.count << ':'
                << pt_auto_sig.sum << ':' << pt_auto_sig.xors
                << " PA:" << pa_sig.count << ':' << pa_sig.sum << ':'
                << pa_sig.xors << '\n';
    }
  } catch (const std::exception& e) {
    if (rank == 0) std::cerr << "MPI annealing test failed: " << e.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }

  MPI_Finalize();
  return 0;
}
