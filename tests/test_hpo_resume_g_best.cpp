#include "catch2/catch.hpp"

#include <tadah/core/checkpoint.h>
#include <tadah/hpo/hpo_host.h>

#include <chrono>
#include <dlib/matrix.h>
#include <filesystem>
#include <limits>

/* Forward-declare LammpsRunner (global namespace, defined in
 * tadah/hpo/runners.h). We never call any of its methods, but we need
 * the name to instantiate the same HPO_Host<...> template the
 * production binary uses, so the test reaches the same static atomics. */
class LammpsRunner;

/* Bug 1 regression — HPO_Host's static comparator atomics (g_best_,
 * g_best_iter_, iter_at_last_best_, g_iter_) must be persisted across
 * a checkpoint round-trip. Without this, a resumed run starts with
 * g_best_ = +inf and the first post-resume eval is always reported as
 * a spurious "NEW BEST", overwriting best_pot.tadah with a worse-than-
 * truth potential.
 *
 * The test directly exercises the static save_globals / load_globals
 * helpers (the ICheckpointable overrides delegate to them) so we don't
 * need to construct a full HPO_Host (which requires real datasets and
 * a trainer). The atomics are template-instantiation-specific, so we
 * use the same instantiation the production binary uses
 * (HPO_Host<dlib_matrix<double,0,1>, LammpsRunner>). */

namespace {

using ParamVec = dlib::matrix<double, 0, 1>;
using Host     = tadah::hpo::HPO_Host<ParamVec, ::LammpsRunner>;

std::filesystem::path scratch_cp_path()
{
    return std::filesystem::temp_directory_path() /
        ("tadah_hpo_resume_g_best_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
         ".tadah");
}

}  // namespace

TEST_CASE("HPO_Host comparator atomics survive a checkpoint round-trip",
          "[hpo_host][resume][bug1]")
{
    const auto cp_path = scratch_cp_path();

    /* ---- write a checkpoint with known comparator state -------------- */
    Host::reset_globals_for_test();
    /* Simulate a run that found a best of 1.0 at eval 42, with the
     * stagnation tracker last reset at eval 50, and the global eval
     * counter at 100. */
    /* No public setters on the atomics (they're set by operator()).
     * We round-trip through Checkpoint by writing values manually,
     * loading, and verifying — i.e. test the save and load paths
     * separately so we don't need a side door into operator(). */
    {
        tadah::core::Checkpoint cp(cp_path.string());
        cp.section("HPO_HOST");
        cp.put_dbl ("g_best",            1.0);
        cp.put_size("g_best_iter",       42);
        cp.put_size("iter_at_last_best", 50);
        cp.put_size("g_iter",            100);
        cp.put_bool("g_best_key_valid",  true);
        cp.put_int ("g_best_rank",       2);
        cp.put_size("g_best_slot",       3);
        cp.put_size("g_best_eval",       42);
        cp.commit();
    }

    /* ---- simulate process restart — reset atomics to defaults ------- */
    Host::reset_globals_for_test();
    REQUIRE(Host::current_best() == std::numeric_limits<double>::max());
    REQUIRE(Host::current_best_iter() == 0u);
    REQUIRE(Host::current_last_best() == 0u);

    /* ---- load from checkpoint ---------------------------------------- */
    {
        tadah::core::Checkpoint cp(cp_path.string());
        REQUIRE(cp.load_from_disk());
        Host::load_globals(cp);
    }

    CHECK(Host::current_best()       == Approx(1.0));
    CHECK(Host::current_best_iter()  == 42u);
    CHECK(Host::current_last_best()  == 50u);
    REQUIRE(Host::current_best_key_for_test().has_value());
    CHECK(Host::current_best_key_for_test()->rank == 2);
    CHECK(Host::current_best_key_for_test()->slot == 3u);
    CHECK(Host::current_best_key_for_test()->eval == 42u);

    /* ---- save round-trip — values written should match what we set -- */
    {
        tadah::core::Checkpoint cp(cp_path.string());
        Host::save_globals(cp);
        cp.commit();
    }
    Host::reset_globals_for_test();
    {
        tadah::core::Checkpoint cp(cp_path.string());
        REQUIRE(cp.load_from_disk());
        Host::load_globals(cp);
    }
    CHECK(Host::current_best()      == Approx(1.0));
    CHECK(Host::current_best_iter() == 42u);
    CHECK(Host::current_last_best() == 50u);
    REQUIRE(Host::current_best_key_for_test().has_value());
    CHECK(Host::current_best_key_for_test()->rank == 2);

    /* A confirmed fresh run in the same process must not inherit the prior
     * comparator, even though eval numbering remains monotonic for the ordered
     * writer. Its first finite candidate can therefore own the new output. */
    Host::reset_best_state_for_fresh_run();
    CHECK(Host::current_best() == std::numeric_limits<double>::max());
    CHECK(Host::current_best_iter() == 0u);
    CHECK_FALSE(Host::current_best_key_for_test().has_value());
    CHECK(Host::current_last_best() == Host::g_iter_value());
    CHECK(Host::g_iter_value() == 100u);

    /* ---- a missing HPO_HOST section is a no-op (defensive) ----------- */
    Host::reset_globals_for_test();
    {
        const auto empty_cp_path = scratch_cp_path();
        tadah::core::Checkpoint cp(empty_cp_path.string());
        cp.section("OTHER_SECTION");
        cp.put_dbl("dummy", 0.0);
        cp.commit();
        tadah::core::Checkpoint reload(empty_cp_path.string());
        REQUIRE(reload.load_from_disk());
        Host::load_globals(reload);
        std::error_code ec;
        std::filesystem::remove(empty_cp_path, ec);
    }
    CHECK(Host::current_best() == std::numeric_limits<double>::max());
    CHECK(Host::current_best_iter() == 0u);

    std::error_code ec;
    std::filesystem::remove(cp_path, ec);
    Host::reset_globals_for_test();
}
