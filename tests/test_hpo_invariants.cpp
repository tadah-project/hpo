// The checks of a public class or function that guard its own invariants,
// whatever the caller: no target file reaches them (the parser refuses the
// value first), so they are tested here, through the class or function.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/hpo/asa_options.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/validation_metric.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

namespace {

namespace fs = std::filesystem;

struct InvariantsTempFile {
    fs::path path;
    explicit InvariantsTempFile(const std::string& body)
    : path(fs::temp_directory_path() /
           ("tadah_hpo_invariants_" +
            std::to_string(std::chrono::steady_clock::now()
                               .time_since_epoch().count()) + ".in")) {
        std::ofstream os(path);
        REQUIRE(os.good());
        os << body;
    }
    ~InvariantsTempFile() {
        std::error_code ec;
        fs::remove(path, ec);
    }
};

tadah::hpo::OptimizerSpec asa_spec()
{
    tadah::hpo::OptimizerSpec sp;
    sp.lib = "ASA";
    sp.algo = "ASA";
    sp.low = {0.0};
    sp.high = {1.0};
    sp.is_int = {false};
    sp.log_scaled = {false};
    sp.optim_init = {{0.5}};
    return sp;
}

}  // namespace

/* Pins: ValidationMetricTargets::add refuses a target or a weight that is
 * not finite, a weight below 0 and a skip cap at or below 0, whoever calls
 * it (the target-file parser refuses each at its line first).
 * Before the fix: passes as well (a pin of the invariant the parser's
 * checks duplicate). */
TEST_CASE("invariants: ValidationMetricTargets::add refuses a target, weight "
          "or cap it cannot use", "[HPO][invariants][validation_metric]")
{
    using tadah::hpo::ValidationMetricAssignment;
    using tadah::hpo::ValidationMetricTargets;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::string bad =
        "PC_ERMSE requires a finite target and nonnegative finite weight";
    for (const auto& [target, weight] :
         {std::pair<double, double>{nan, 1.0}, {0.1, nan}, {0.1, -1.0},
          {std::numeric_limits<double>::infinity(), 1.0}}) {
        INFO(target << " " << weight);
        ValidationMetricTargets t("PC_ERMSE");
        ValidationMetricAssignment a;
        a.target = target;
        a.weight = weight;
        CHECK_THROWS_WITH(t.add({}, a), bad);
        CHECK_FALSE(t.enabled());
    }
    ValidationMetricTargets t("PC_ERMSE");
    ValidationMetricAssignment a;
    a.target = 0.1;
    a.weight = 1.0;
    a.caps.abs_above = 0.0;
    CHECK_THROWS_WITH(t.add({}, a),
                      "PC_ERMSE skip thresholds must be finite and positive");
    a.caps.abs_above.reset();
    CHECK_NOTHROW(t.add({}, a));
    CHECK(t.enabled());
}

/* Pins: parse_asa_options refuses a PARAM value that is not finite, and a
 * Delta_X below 0 by the range of the option in its table, the one check
 * of the range (the branch of Delta_X held a second check of its own,
 * "must be >= 0 (0 disables tangents)", which the table's check always
 * came before).
 * Before the fix: passes as well (a pin: the table's check refused the
 * value first). */
TEST_CASE("invariants: parse_asa_options refuses a value outside the range of "
          "its option", "[HPO][invariants][asa]")
{
    auto sp = asa_spec();
    sp.param["Limit_Acceptances"] = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_WITH(tadah::hpo::parse_asa_options(sp, 1),
                      "ASA:ASA: PARAM Limit_Acceptances must be finite");
    sp = asa_spec();
    sp.param["Delta_X"] = -0.5;
    CHECK_THROWS_WITH(tadah::hpo::parse_asa_options(sp, 1),
                      Catch::StartsWith("ASA:ASA: PARAM Delta_X must be >= 0") &&
                          !Catch::Contains("disables"));
    sp = asa_spec();
    sp.param["Delta_X"] = 0.0;
    CHECK_NOTHROW(tadah::hpo::parse_asa_options(sp, 1));
}

/* Pins: OPTIM <key> (n) AUTO takes the n-th number of the key as its seed:
 * the OPTIM index counts the numbers of the key from 1 (index 0 is
 * refused at its line), so the token an index selects is a number, and the
 * AUTO box has a seed (the check that the seed was a number, "AUTO needs a
 * numeric CONFIG seed", could not fail).
 * Before the fix: passes as well (a pin of what made that check
 * unreachable). */
TEST_CASE("invariants: OPTIM AUTO takes the number its index selects as the "
          "seed", "[HPO][invariants][optim][autobox]")
{
    tadah::core::Context ctx;
    ctx.add("CGRID2B", 1.0);
    ctx.add("CGRID2B", 2.0);
    ctx.add("CGRID2B", 3.0);
    InvariantsTempFile tf("OPTIM CGRID2B (3) AUTO\nPC_ERMSE 0 1\nOPTIMIZER\n"
                          "STRATEGY\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 5\n"
                          "ENDSTRATEGY\nENDOPTIMIZER\n");
    tadah::hpo::HPOTargets trg(ctx, tf.path.string());
    REQUIRE(trg.spec.space);
    CHECK(trg.spec.space->at(0).label == "CGRID2B[3]");
    CHECK(trg.spec.space->at(0).x0 == 3.0);

    InvariantsTempFile zero("OPTIM CGRID2B (0) AUTO\nPC_ERMSE 0 1\nOPTIMIZER\n"
                            "STRATEGY\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 5\n"
                            "ENDSTRATEGY\nENDOPTIMIZER\n");
    CHECK_THROWS_WITH(tadah::hpo::HPOTargets(ctx, zero.path.string()),
                      Catch::Contains("OPTIM indices start at 1"));
}
