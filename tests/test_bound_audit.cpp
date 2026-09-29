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

TEST_CASE("BoundOccupancy: a hard lower bound is flagged, not read as bound "
          "pressure", "[HPO][bound_audit][hpolog]")
{
    /* AMPGRIDMB under RHOFORM NONNEG/MONOTONE: lo = 0 is the form's limit */
    DimSpec a; a.key = "AMPGRIDMB"; a.label = "AMPGRIDMB[3]"; a.lo = 0.0; a.hi = 0.5;
    DimSpec b; b.key = b.label = "RCUT2B"; b.lo = 4.0; b.hi = 6.0;
    ScalePolicy pol; pol.normalise = true;
    const auto sp = std::make_shared<const SearchSpace>(SearchSpace::build({a, b}, pol));
    BoundOccupancy occ(sp);
    occ.set_hard_lo({true, false});
    REQUIRE(occ.on_new_best(3, {0.0, 5.0}));
    REQUIRE(occ.marks().size() == 1);
    CHECK_FALSE(occ.marks()[0].hi);
    CHECK(occ.marks()[0].hard);
    /* only the lower side is hard */
    REQUIRE(occ.on_new_best(4, {0.5, 4.0}));
    REQUIRE(occ.marks().size() == 2);
    CHECK(occ.marks()[0].hi);
    CHECK_FALSE(occ.marks()[0].hard);
    CHECK_FALSE(occ.marks()[1].hard);

    auto& log = HPOLog::instance();
    const auto path = std::filesystem::temp_directory_path() /
        ("tadah_bound_audit_hard_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".log");
    OptimizerSpec spec; spec.lib = "NLOPT"; spec.algo = "LN_SBPLX";
    spec.bind_space(sp);
    auto run = [&](const std::vector<double>& best) {
        if (log.is_open()) log.finish("reopen-for-test");
        log.reset_for_test();
        log.open(path.string(), "test", "test_cwd", 0);
        REQUIRE(log.is_open());
        log.write_optimizer_section(spec, spec.parameter_labels);
        occ.on_new_best(7, best);
        log.record_bound_occupancy(occ.marks(), 7, occ.edge());
        log.record_eval(100, 1.0, tadah::hpo::EvalLoss::Usable);   /* milestone 1 */
        log.finish("TEST");
        return slurp(path);
    };
    const std::string note = "(* a hard lower bound: RHOFORM NONNEG/MONOTONE require "
                             "AMPGRIDMB >= 0, so an amplitude at 0 is an admissible "
                             "optimum, not bound pressure)";

    /* hard only: the marks and the note, no remedy at the milestone or at
     * the end */
    std::string t = run({0.0, 5.0});
    CHECK(t.find("AMPGRIDMB[3]            LO*") != std::string::npos);
    CHECK(t.find("Bound occupancy of the final best (eval 7)") != std::string::npos);
    CHECK(t.find("    " + note + "\n") != std::string::npos);
    CHECK(t.find("suggests bound pressure") == std::string::npos);
    CHECK(t.find("advisory") == std::string::npos);

    /* a soft mark next to it keeps the remedy */
    t = run({0.0, 4.0});
    CHECK(t.find("AMPGRIDMB[3]            LO*") != std::string::npos);
    CHECK(t.find("RCUT2B                  LO ") != std::string::npos);
    CHECK(t.find(note) != std::string::npos);
    CHECK(t.find("HINT") != std::string::npos);
    CHECK(t.find("advisory") != std::string::npos);

    log.reset_for_test();
    std::error_code ec; std::filesystem::remove(path, ec);
}
