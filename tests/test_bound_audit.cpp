/* BoundOccupancy: marks a best point riding an OPTIM bound, in INTERNAL
 * units (so LOG dims are judged in ln), persists since_eval, and HPOLog
 * prints the milestone/final tables. */
#include "catch2/catch.hpp"
#include <tadah/hpo/bound_audit.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/search_space.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>

using namespace tadah::hpo;

namespace {

SearchSpacePtr space()
{
    DimSpec a; a.key = a.label = "RCUT2B"; a.lo = 4.0; a.hi = 6.0;
    DimSpec b; b.key = b.label = "SGRID2B"; b.lo = 0.01; b.hi = 10.0; b.tf.kind = ScaleKind::Log;
    ScalePolicy pol; pol.normalise = true;
    return std::make_shared<const SearchSpace>(SearchSpace::build({a, b}, pol));
}

std::string slurp(const std::filesystem::path& p)
{ std::ifstream f(p); std::ostringstream s; s << f.rdbuf(); return s.str(); }

}  // namespace

TEST_CASE("BoundOccupancy marks near-bound dims in internal units",
          "[HPO][bound_audit]")
{
    BoundOccupancy occ(space(), 0.02);
    REQUIRE(occ.armed());
    /* RCUT2B 5.99 -> gap 0.005 of width (HI); SGRID2B 0.0105 -> ln gap
     * ln(0.0105/0.01)/ln(1000) = 0.007 (LO) although the real gap is tiny */
    REQUIRE(occ.on_new_best(10, {5.99, 0.0105}));
    REQUIRE(occ.marks().size() == 2);
    CHECK(occ.marks()[0].dim == 0); CHECK(occ.marks()[0].hi);
    CHECK(occ.marks()[0].gap_u == Approx(0.005));
    CHECK(occ.marks()[1].dim == 1); CHECK_FALSE(occ.marks()[1].hi);
    CHECK(occ.marks()[1].gap_u == Approx(std::log(1.05) / std::log(1000.0)));
    CHECK(occ.marks()[1].real_bound == Approx(0.01));

    /* persisting mark keeps since_eval; released mark disappears */
    REQUIRE(occ.on_new_best(20, {5.995, 1.0}));
    REQUIRE(occ.marks().size() == 1);
    CHECK(occ.marks()[0].since_eval == 10);
    /* a real value far from the bound in real units but within 2% in ln:
     * SGRID2B 9.0 -> ln gap ln(10/9)/ln(1000) = 0.015 -> marked HI */
    REQUIRE(occ.on_new_best(30, {5.0, 9.0}));
    REQUIRE(occ.marks().size() == 1);
    CHECK(occ.marks()[0].dim == 1); CHECK(occ.marks()[0].hi);
    CHECK_FALSE(occ.on_new_best(40, {5.0, 1.0}));
}

TEST_CASE("HPOLog prints the bound-occupancy tables",
          "[HPO][bound_audit][hpolog]")
{
    auto& log = HPOLog::instance();
    if (log.is_open()) log.finish("reopen-for-test");
    log.reset_for_test();
    const auto path = std::filesystem::temp_directory_path() /
        ("tadah_bound_audit_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".log");
    log.open(path.string(), "test", "test_cwd", 0);
    REQUIRE(log.is_open());

    OptimizerSpec sp; sp.lib = "NLOPT"; sp.algo = "LN_SBPLX";
    sp.bind_space(space());
    log.write_optimizer_section(sp, sp.parameter_labels);

    BoundOccupancy occ(space());
    occ.on_new_best(7, {5.99, 1.0});
    log.record_bound_occupancy(occ.marks(), 7, occ.edge());
    log.record_eval(100, 1.0, tadah::hpo::EvalLoss::Usable);          /* milestone 1 */
    std::string t = slurp(path);
    CHECK(t.find("sits within 2% of an OPTIM limit on 1 dim(s)") != std::string::npos);
    CHECK(t.find("RCUT2B") != std::string::npos);
    CHECK(t.find("HINT") != std::string::npos);
    CHECK(t.find("bound pressure") != std::string::npos);

    log.finish("TEST");
    t = slurp(path);
    CHECK(t.find("Bound occupancy of the final best (eval 7)") != std::string::npos);
    CHECK(t.find("advisory") != std::string::npos);
    log.reset_for_test();
    std::error_code ec; std::filesystem::remove(path, ec);
}
