/* Pinned-dim audit -- the milestone WARNING table for OPTIM dims stuck
 * ON a bound with the Corana step at its V floor.
 *
 * Two layers under test:
 *   1. anneal::pinned_dims() -- the predicate: pinned iff V[i] is at the
 *      floor AND x[i] is within one floor-step of a bound; while V can
 *      still shrink the dim must NOT be reported (wait rule).
 *   2. HPOLog arm/record/print -- marks keep their since-unit while
 *      held, unmark when the chain leaves the bound, and the table
 *      (with slots aggregation, (ln) tag, LO/HI sides) prints as a
 *      single WARNING block with each milestone SUMMARY -- and stops
 *      printing once nothing is pinned. */

#include "catch2/catch.hpp"

#include <tadah/hpo/anneal_core.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/optimizer_spec.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <dlib/matrix.h>

using namespace tadah::hpo;

namespace {

using Vec = dlib::matrix<double, 0, 1>;

std::filesystem::path scratch_log_path()
{
    return std::filesystem::temp_directory_path() /
        ("tadah_pinned_dims_" +
         std::to_string(std::chrono::steady_clock::now()
                            .time_since_epoch().count()) + ".log");
}

std::string slurp(const std::filesystem::path& p)
{
    std::ifstream f(p);
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
}

OptimizerSpec three_dim_spec()
{
    OptimizerSpec sp;
    sp.low        = {0.0, 0.0, 0.0};
    sp.high       = {1.0, 2.0, 4.0};
    sp.log_scaled = {false, false, true};
    return sp;
}

}  // namespace

TEST_CASE("anneal::pinned_dims -- floor + bound-hug predicate",
          "[pinned_dims]")
{
    const auto sp = three_dim_spec();
    anneal::Knobs kb;
    kb.corana       = true;
    kb.v_floor_frac = 1e-3;
    /* floors: dim0 = 1e-3, dim1 = 2e-3, dim2 = 4e-3 */

    anneal::ChainState<Vec> c;
    c.x = Vec(3);
    c.V = {1e-3, 2e-3, 4e-3};   /* all steps at the floor */

    SECTION("on a bound with a floor step => pinned, correct side + gap") {
        c.x = {0.0005, 1.9999, 2.0};   /* dim2 mid-box */
        const auto pins = anneal::pinned_dims(c, sp, kb);
        REQUIRE(pins.size() == 2);
        CHECK(pins[0].dim == 0);
        CHECK(!pins[0].hi);
        CHECK(pins[0].gap == Approx(0.0005));
        CHECK(pins[1].dim == 1);
        CHECK(pins[1].hi);
        CHECK(pins[1].gap == Approx(0.0001));
    }

    SECTION("wait rule: step above the floor => not pinned even ON the bound") {
        c.x = {0.0, 0.0, 0.0};
        c.V = {5e-3, 2e-3, 4e-3};      /* dim0 can still shrink */
        const auto pins = anneal::pinned_dims(c, sp, kb);
        REQUIRE(pins.size() == 2);     /* dims 1 and 2 only */
        CHECK(pins[0].dim == 1);
        CHECK(pins[1].dim == 2);
    }

    SECTION("off the bound by more than one floor-step => not pinned") {
        c.x = {0.5, 1.0, 2.0};
        CHECK(anneal::pinned_dims(c, sp, kb).empty());
    }
}

TEST_CASE("HPOLog pinned-dim audit -- mark/unmark + milestone table",
          "[pinned_dims][hpolog]")
{
    auto& log = HPOLog::instance();
    log.reset_for_test();
    const auto path = scratch_log_path();
    log.open(path.string(), "test", "test_cwd", 0);
    REQUIRE(log.is_open());

    const auto sp = three_dim_spec();
    log.arm_pin_audit(sp, 1e-3, 64, "epoch");

    /* Epoch 5: chain 3 pins dim0 LO; chain 7 pins dim0 LO + dim2 HI. */
    log.record_pin_state(3, 5, {{0, false, 0.0, 1e-3}});
    log.record_pin_state(7, 5, {{0, false, 1e-6, 1e-3}, {2, true, 0.0, 1e-3}});

    log.record_eval(100, 1.0, false);          /* milestone 1 */
    std::string s1 = slurp(path);
    CHECK(s1.find("2 OPTIM dim(s) near a bound") != std::string::npos);
    CHECK(s1.find("WARNING") != std::string::npos);
    CHECK(s1.find("2/64") != std::string::npos);   /* dim0: both chains */
    CHECK(s1.find("1/64") != std::string::npos);   /* dim2: chain 7 only */
    CHECK(s1.find("(ln)") != std::string::npos);   /* dim2 is log-scaled */
    CHECK(s1.find("LO") != std::string::npos);
    CHECK(s1.find("HI") != std::string::npos);
    CHECK(s1.find("1 epoch") != std::string::npos);
    CHECK(s1.find("persistent near-bound occupancy") != std::string::npos);
    CHECK(s1.find("may indicate") != std::string::npos);
    CHECK(s1.find("walk cannot leave") == std::string::npos);

    /* Epoch 6: chain 3 leaves its bound (unmark); chain 7 keeps dim0
     * pinned (since-unit must survive) but releases dim2. */
    log.record_pin_state(3, 6, {});
    log.record_pin_state(7, 6, {{0, false, 2e-6, 1e-3}});

    log.record_eval(200, 1.0, false);          /* milestone 2 */
    std::string s2 = slurp(path).substr(s1.size());
    CHECK(s2.find("1 OPTIM dim(s) near a bound") != std::string::npos);
    CHECK(s2.find("1/64") != std::string::npos);
    CHECK(s2.find("2 epochs") != std::string::npos);   /* held since 5 */
    CHECK(s2.find("(ln)") == std::string::npos);       /* dim2 row gone */

    /* Epoch 7: everything released -- the table must stop printing. */
    log.record_pin_state(7, 7, {});
    log.record_eval(300, 1.0, false);          /* milestone 3 */
    std::string s3 = slurp(path).substr(s1.size() + s2.size());
    CHECK(s3.find("SUMMARY") != std::string::npos);    /* milestone fired */
    CHECK(s3.find("OPTIM dim(s) pinned") == std::string::npos);

    log.finish("TEST");   /* close the singleton's file for later tests */
    log.reset_for_test();
    std::filesystem::remove(path);
}
