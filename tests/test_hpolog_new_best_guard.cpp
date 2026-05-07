#include "catch2/catch.hpp"

#include <tadah/hpo/hpo_log.h>

#include <chrono>
#include <filesystem>

/* Bug 2 regression — HPOLog::log_new_best must reject calls whose loss is
 * not strictly better than cur_best_loss_. Without the guard, an upstream
 * is_best comparator that drifted (e.g. HPO_Host::g_best_ at +infinity
 * after a resume that did not restore it) would silently overwrite the
 * journal's running best with a worse value and downstream summaries
 * would report regression as progress.
 *
 * The guard does NOT prevent the spurious comparator-side decision —
 * Bug 1's HPO_HOST checkpoint section does that. The guard's purpose
 * here is defence-in-depth so a future code path that miscalls
 * log_new_best cannot corrupt the running best. */

namespace {

std::filesystem::path scratch_log_path()
{
    return std::filesystem::temp_directory_path() /
        ("tadah_hpolog_guard_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
         ".log");
}

void log_best(double loss, std::size_t eval_num)
{
    /* No metric flags — simplest call form. */
    tadah::hpo::HPOLog::instance().log_new_best(
        eval_num, loss,
        0.0, false,   // emae
        0.0, false,   // ermse
        0.0, false,   // errmse
        0.0, false,   // frmse
        0.0, false,   // srmse
        0.0,          // wall_ms
        ""            // top_breakdown
    );
}

}  // namespace

TEST_CASE("HPOLog::log_new_best rejects non-improving losses", "[hpolog][bug2]")
{
    auto& log = tadah::hpo::HPOLog::instance();
    log.reset_for_test();
    const auto path = scratch_log_path();
    log.open(path.string(), "test", "test_cwd", 0);
    REQUIRE(log.is_open());

    /* First best at 10.0 — accepted (cur_best defaults to numeric_limits::max). */
    log_best(10.0, 1);
    CHECK(log.best_loss() == Approx(10.0));
    CHECK(log.best_eval() == 1u);

    /* Worse loss — must be rejected; running best must not regress. */
    log_best(20.0, 2);
    CHECK(log.best_loss() == Approx(10.0));
    CHECK(log.best_eval() == 1u);

    /* Equal loss — must also be rejected (strict-improvement guard). */
    log_best(10.0, 3);
    CHECK(log.best_loss() == Approx(10.0));
    CHECK(log.best_eval() == 1u);

    /* Strictly better loss — accepted. */
    log_best(5.0, 4);
    CHECK(log.best_loss() == Approx(5.0));
    CHECK(log.best_eval() == 4u);

    log.finish("TEST_DONE");
    std::error_code ec;
    std::filesystem::remove(path, ec);
}
