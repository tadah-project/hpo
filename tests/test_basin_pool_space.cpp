/* Basin pools under the SearchSpace: REAL-space params, distances in
 * INTERNAL coordinates, checkpoint coordinate-contract tags. */
#include "catch2/catch.hpp"
#include <tadah/hpo/basin_pool.h>
#include <tadah/hpo/basin_pool_v2.h>
#include <tadah/hpo/search_space.h>
#include <tadah/core/checkpoint.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <memory>
#include <vector>

using namespace tadah::hpo;

namespace {

SearchSpacePtr two_dim_space(bool normalise, bool log_second)
{
    DimSpec a; a.key = a.label = "A"; a.lo = 0.0;  a.hi = 10.0;
    DimSpec b; b.key = b.label = "B"; b.lo = 0.01; b.hi = 100.0;
    if (log_second) b.tf.kind = ScaleKind::Log;
    ScalePolicy pol; pol.normalise = normalise;
    return std::make_shared<const SearchSpace>(SearchSpace::build({a, b}, pol));
}

std::string scratch_checkpoint_path()
{
    return (std::filesystem::temp_directory_path() /
            ("tadah_pool_space_" +
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
             ".ckpt")).string();
}

}  // namespace

TEST_CASE("BasinPool: dedup distance is measured in internal coordinates",
          "[HPO][basins][search_space]")
{
    /* Two real points equal on A, differing on B by a factor of 10.
     * In LOG coordinates that is ln(10)/ln(1e4) ~ 0.25 of the box; in
     * raw coordinates 9.0/99.99 ~ 0.09. */
    const std::vector<double> p = {5.0, 1.0}, q = {5.0, 10.0};

    SECTION("LOG dim, normalised: merges only when dedup_dist covers 0.25") {
        BasinPool pool(0.2, 8);
        pool.set_space(two_dim_space(true, true));
        pool.add_or_merge(p, 1.0, 0);
        pool.add_or_merge(q, 2.0, 0);
        CHECK(pool.size() == 2);
        BasinPool wide(0.3, 8);
        wide.set_space(two_dim_space(true, true));
        wide.add_or_merge(p, 1.0, 0);
        wide.add_or_merge(q, 2.0, 0);
        CHECK(wide.size() == 1);
    }
    SECTION("linear dims, NONE: identical to the legacy raw normalisation") {
        BasinPool a(0.1, 8), b(0.1, 8);
        a.set_space(two_dim_space(false, false));
        b.set_normalisation({0.0, 0.01}, {10.0, 100.0});
        a.add_or_merge(p, 1.0, 0); a.add_or_merge(q, 2.0, 0);
        b.add_or_merge(p, 1.0, 0); b.add_or_merge(q, 2.0, 0);
        CHECK(a.size() == b.size());
        /* params stay REAL on the way out */
        CHECK(a.heads().front().params[1] == Approx(1.0));
    }
}

TEST_CASE("BasinPoolV2: relative_l2 through the space",
          "[HPO][basins_v2][search_space]")
{
    BasinPoolV2 pool(0.05, 8, 8);
    pool.set_space(two_dim_space(true, true));
    const double d = pool.relative_l2({5.0, 1.0}, {5.0, 10.0});
    CHECK(d == Approx(std::log(10.0) / std::log(1e4)).epsilon(1e-9));
    const double dl = pool.relative_l2({0.0, 1.0}, {10.0, 1.0});
    CHECK(dl == Approx(1.0));
}

TEST_CASE("BasinPool checkpoint carries the coordinate contract",
          "[HPO][basins][checkpoint][search_space]")
{
    tadah::core::Checkpoint cp(scratch_checkpoint_path());
    {
        BasinPool pool(0.05, 8);
        pool.set_space(two_dim_space(true, true));
        pool.add_or_merge({5.0, 1.0}, 1.0, 0);
        pool.save(cp);
    }
    CHECK(cp.get_bool("BASIN_POOL", "params_real").value_or(false));
    REQUIRE(cp.get_int("BASIN_POOL", "space_fingerprint").has_value());

    SECTION("same space loads") {
        BasinPool pool(0.05, 8);
        pool.set_space(two_dim_space(true, true));
        REQUIRE_NOTHROW(pool.load(cp));
        REQUIRE(pool.size() == 1);
        CHECK(pool.heads().front().params[1] == Approx(1.0));  /* still REAL */
    }
    SECTION("changed space is refused with a delete hint") {
        BasinPool pool(0.05, 8);
        pool.set_space(two_dim_space(true, false));   /* B no longer LOG */
        REQUIRE_THROWS_WITH(pool.load(cp),
                            Catch::Contains("different search space") &&
                            Catch::Contains("delete checkpoint.tadah"));
    }
}

TEST_CASE("BasinPool legacy checkpoint (no contract tag) is accepted only for linear spaces",
          "[HPO][basins][checkpoint][search_space]")
{
    tadah::core::Checkpoint cp(scratch_checkpoint_path());
    {
        /* emulate a pre-transform pool section: no params_real key */
        cp.section("BASIN_POOL");
        cp.put_dbl("dedup_dist", 0.05);
        cp.put_size("pool_max", 8);
        cp.put_vec("low", {0.0, 0.01});
        cp.put_vec("high", {10.0, 100.0});
        cp.put_size("n_basins", 1);
        cp.put_vec("basin_0_params", {5.0, 1.0});
        cp.put_dbl("basin_0_loss", 1.0);
        cp.put_int("basin_0_birth_round", 0);
        cp.put_int("basin_0_last_round", 0);
    }
    SECTION("all-linear: accepted") {
        BasinPool pool(0.05, 8);
        pool.set_space(two_dim_space(true, false));
        REQUIRE_NOTHROW(pool.load(cp));
        CHECK(pool.size() == 1);
    }
    SECTION("LOG dim: refused") {
        BasinPool pool(0.05, 8);
        pool.set_space(two_dim_space(true, true));
        REQUIRE_THROWS_WITH(pool.load(cp),
                            Catch::Contains("predates the search-space transform") &&
                            Catch::Contains("B"));
    }
    SECTION("no space bound: legacy normalisation restored") {
        BasinPool pool(0.05, 8);
        REQUIRE_NOTHROW(pool.load(cp));
        CHECK(pool.size() == 1);
    }
}
