/* hpo_run.log SEARCH SPACE section under the SearchSpace: policy lines,
 * width ratio, per-dim rows with scale/origin/real and internal bounds. */
#include "catch2/catch.hpp"
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/search_space.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

using namespace tadah::hpo;

namespace {

std::string slurp(const std::filesystem::path& p)
{
    std::ifstream f(p);
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
}

struct JournalCapture {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("tadah_ss_journal_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".log");
    JournalCapture()
    {
        auto& log = HPOLog::instance();
        if (log.is_open()) log.finish("reopen-for-test");
        log.reset_for_test();
        log.open(path.string(), "test", "test_cwd", 0);
        REQUIRE(log.is_open());
    }
    std::string text() const { return slurp(path); }
    ~JournalCapture()
    {
        HPOLog::instance().finish("test-end");
        HPOLog::instance().reset_for_test();
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

OptimizerSpec bound_spec(bool normalise)
{
    DimSpec a; a.key = "RCUT2B";  a.label = "RCUT2B";     a.lo = 4.0;  a.hi = 6.0;
    DimSpec b; b.key = "SGRID2B"; b.label = "SGRID2B[0]"; b.lo = 0.05; b.hi = 5.0;
    DimSpec c; c.key = "FIXWEIGHT"; c.label = "FIXWEIGHT[3]"; c.lo = -26.5; c.hi = 26.5;
    c.tf.kind = ScaleKind::Asinh; c.tf.c = 1.0;
    ScalePolicy pol; pol.normalise = normalise; pol.log_ratio = 100.0;
    OptimizerSpec sp;
    sp.lib = "NLOPT"; sp.algo = "LN_SBPLX";
    sp.stop.maxeval = 60;
    sp.log_threshold = 100.0;
    sp.bind_space(std::make_shared<const SearchSpace>(SearchSpace::build({a, b, c}, pol)));
    sp.optim_init = {{5.6, 0.1, 0.0}};
    return sp;
}

}  // namespace

TEST_CASE("SEARCH SPACE section prints policy, width ratio and per-dim rows",
          "[HPO][hpolog][search_space]")
{
    SECTION("OPTIM_SCALE NONE") {
        JournalCapture j;
        auto sp = bound_spec(false);
        HPOLog::instance().write_optimizer_section(sp, sp.parameter_labels);
        const std::string t = j.text();
        CHECK(t.find("-- SEARCH SPACE") != std::string::npos);
        CHECK(t.find("OPTIM_SCALE NONE") != std::string::npos);
        CHECK(t.find("LOG_HP 100 (1 dim(s) auto-LOG)") != std::string::npos);
        CHECK(t.find("1 ASINH") != std::string::npos);
        CHECK(t.find("Real width ratio") != std::string::npos);
        CHECK(t.find("RCUT2B") != std::string::npos);
        CHECK(t.find("LOG (auto)") != std::string::npos);
        CHECK(t.find("ASINH c=1") != std::string::npos);
        CHECK(t.find("OPTIM") != std::string::npos);           /* origin column */
        CHECK(t.find("Space version         : 1") != std::string::npos);
    }
    SECTION("OPTIM_SCALE AUTO") {
        JournalCapture j;
        auto sp = bound_spec(true);
        HPOLog::instance().write_optimizer_section(sp, sp.parameter_labels);
        const std::string t = j.text();
        CHECK(t.find("OPTIM_SCALE AUTO") != std::string::npos);
        CHECK(t.find("unit box [0,1]") != std::string::npos);
        CHECK(t.find("-> 1.0 for every dim after scaling") != std::string::npos);
        /* internal bounds are [0, 1] */
        CHECK(t.find("[0.000e+00, 1.000e+00]") != std::string::npos);
    }
}

TEST_CASE("SEARCH SPACE legacy layout for a spec without a bound space",
          "[HPO][hpolog][search_space]")
{
    JournalCapture j;
    OptimizerSpec sp;
    sp.lib = "NLOPT"; sp.algo = "LN_BOBYQA";
    sp.low = {0.0, 0.0}; sp.high = {1.0, 2.0};
    sp.log_scaled = {false, true};
    HPOLog::instance().write_optimizer_section(sp, {"A", "B"});
    const std::string t = j.text();
    CHECK(t.find("(bounds in optimiser-internal space)") != std::string::npos);
    CHECK(t.find("(log-scale -- bounds are ln(value))") != std::string::npos);
}
