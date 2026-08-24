#include <tadah/hpo/algorithms_pa.h>
#include <tadah/hpo/algorithms_pt.h>
#include <tadah/hpo/hpo_mpi.h>
#include <tadah/hpo/optimizer_spec.h>

#include <dlib/matrix.h>
#include <mpi.h>

#include <cstdint>
#include <cstring>
#include <iostream>
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
  sp.param["CORANA"] = 1.0;
  sp.param["NS"] = 1.0;
  /* NT is serial/PA-only now (ANNEAL_PT runs one sweep per epoch);
   * run_pa() adds it back for the population annealer. */
  sp.param["T_HOT"] = 1.0;
  sp.param["BOUNDARY_MODE"] = 1.0;
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
  sp.param["LADDER_ADAPT"] = 1.0;
  sp.param["LADDER_ADAPT_EVERY"] = 1.0;
  sp.param["LADDER_ADAPT_GAIN"] = 1.0;
  sp.param["LADDER_ADAPT_DECAY"] = 0.6;
  tadah::hpo::run_anneal_pt_impl<Vec>(sp, host);
  return host.global_signature();
}

Signature run_pt_auto(std::uint64_t cap, RecordingHost& host)
{
  auto sp = base_spec("ANNEAL_PT", cap);
  sp.param.erase("CORANA");
  sp.param.erase("NS");
  sp.param["REPLICAS"] = 8.0;
  sp.param["T_COLD"] = 0.1;
  /* STEP_MODE is intentionally omitted: exercise the FROZEN-default CLASSIC
   * coordinate warm-up and its cross-rank diagnostic gather. */
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

    RecordingHost pt{rank, world};
    const Signature pt_sig = run_pt(113, pt);
    require_all(pt_sig.count == 113, "PT did not consume exact MAXEVAL", rank);

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

    if (rank == 0) {
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
