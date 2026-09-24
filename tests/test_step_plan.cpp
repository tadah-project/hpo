/* Per-dimension initial-step plan (initial_step.h) and the NLopt
 * adapter's use of it (make_nlopt + get_initial_step read-back). */
#include "catch2/catch.hpp"
#include <tadah/hpo/initial_step.h>
#include <tadah/hpo/nlopt_adapter.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/search_space.h>
#include <tadah/hpo/scale_validation.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace tadah::hpo;

namespace {

/* Two dims: A real [0, 2] (width 2), B real [0, 0.008] (width 8e-3),
 * mirroring run15's CGRID2B vs AMPGRIDMB spread. */
OptimizerSpec spec(const char* algo, bool normalise)
{
    DimSpec a; a.key = "CGRID2B";   a.user_index = 1; a.label = "CGRID2B[1]";   a.lo = 0.0; a.hi = 2.0;
    DimSpec b; b.key = "AMPGRIDMB"; b.user_index = 6; b.label = "AMPGRIDMB[6]"; b.lo = 0.0; b.hi = 0.008;
    ScalePolicy pol; pol.normalise = normalise;
    OptimizerSpec sp;
    sp.lib = "NLOPT"; sp.algo = algo;
    sp.bind_space(std::make_shared<const SearchSpace>(SearchSpace::build({a, b}, pol)));
    sp.optim_init = {{1.0, 0.004}};
    return sp;
}

}  // namespace

TEST_CASE("StepPlan: STEP is a fraction of each dimension's width",
          "[HPO][step_plan]")
{
    for (bool normalise : {false, true}) {
        auto sp = spec("LN_SBPLX", normalise);
        sp.step = 0.05;
        const auto start = sp.in(sp.optim_init.front());
        const auto plan = make_step_plan(sp, start, optimizer_caps(sp));
        REQUIRE(plan.user_set);
        REQUIRE(plan.dx.size() == 2);
        CHECK(plan.frac[0] == Approx(0.05));
        CHECK(plan.frac[1] == Approx(0.05));
        CHECK(plan.dx[0] == Approx(0.05 * (sp.high[0] - sp.low[0])));
        CHECK(plan.dx[1] == Approx(0.05 * (sp.high[1] - sp.low[1])));
        if (normalise) { CHECK(plan.dx[0] == Approx(0.05)); CHECK(plan.dx[1] == Approx(0.05)); }
        else           { CHECK(plan.dx[0] == Approx(0.1));  CHECK(plan.dx[1] == Approx(4e-4)); }
        CHECK(plan.source.find("STEP 0.05") != std::string::npos);
    }
}

TEST_CASE("StepPlan: per-key and per-index overrides",
          "[HPO][step_plan]")
{
    auto sp = spec("LN_BOBYQA", false);
    sp.step = 0.1;
    sp.step_overrides.push_back({"AMPGRIDMB", {6}, 0.02});
    const auto start = sp.in(sp.optim_init.front());
    auto plan = make_step_plan(sp, start, optimizer_caps(sp));
    CHECK(plan.frac[0] == Approx(0.1));
    CHECK(plan.frac[1] == Approx(0.02));
    CHECK(plan.note[1].find("override") != std::string::npos);

    sp.step_overrides = {{"CGRID2B", {}, 0.3}};        /* whole key */
    plan = make_step_plan(sp, start, optimizer_caps(sp));
    CHECK(plan.frac[0] == Approx(0.3));
    CHECK(plan.frac[1] == Approx(0.1));

    sp.step_overrides = {{"SGRID2B", {}, 0.3}};        /* not optimised */
    REQUIRE_THROWS_WITH(make_step_plan(sp, start, optimizer_caps(sp)),
                        Catch::Contains("SGRID2B") && Catch::Contains("not optimised"));
    sp.step_overrides = {{"AMPGRIDMB", {7}, 0.3}};     /* wrong index */
    REQUIRE_THROWS_WITH(make_step_plan(sp, start, optimizer_caps(sp)),
                        Catch::Contains("index 7 is not an OPTIM'd index"));

    sp.step.reset();                                   /* overrides only, partial */
    sp.step_overrides = {{"CGRID2B", {}, 0.3}};
    REQUIRE_THROWS_WITH(make_step_plan(sp, start, optimizer_caps(sp)),
                        Catch::Contains("cover only some dimensions"));
}

TEST_CASE("StepPlan: near-bound rules are per dimension",
          "[HPO][step_plan]")
{
    SECTION("BOBYQA: only the tight dim shrinks to 0.75*slack") {
        auto sp = spec("LN_BOBYQA", false);
        sp.step = 0.25;
        sp.optim_init = {{1.0, 0.0079}};   /* B within 1e-4 of its HI bound */
        const auto start = sp.in(sp.optim_init.front());
        const auto plan = make_step_plan(sp, start, optimizer_caps(sp));
        CHECK(plan.dx[0] == Approx(0.5));                       /* untouched */
        CHECK(plan.dx[1] == Approx(0.75 * (0.008 - 0.0079)));   /* shrunk */
        CHECK(plan.note[1].find("near-bound") != std::string::npos);
        CHECK(plan.note[0].empty());
    }
    SECTION("BOBYQA: a start ON a bound keeps its step (noted)") {
        auto sp = spec("LN_BOBYQA", false);
        sp.step = 0.25;
        sp.optim_init = {{0.0, 0.004}};
        const auto start = sp.in(sp.optim_init.front());
        const auto plan = make_step_plan(sp, start, optimizer_caps(sp));
        CHECK(plan.dx[0] == Approx(0.5));
        CHECK(plan.note[0].find("on a bound") != std::string::npos);
    }
    SECTION("SBPLX: inward sign when only the upper bound is within one step") {
        auto sp = spec("LN_SBPLX", false);
        sp.step = 0.25;
        sp.optim_init = {{1.9, 0.004}};    /* A: 0.1 below HI, 1.9 above LO */
        const auto start = sp.in(sp.optim_init.front());
        const auto plan = make_step_plan(sp, start, optimizer_caps(sp));
        CHECK(plan.dx[0] == Approx(-0.5));
        CHECK(plan.frac[0] == Approx(0.25));
        CHECK(plan.note[0].find("inward") != std::string::npos);
        CHECK(plan.dx[1] == Approx(0.002));
    }
    SECTION("no start point (INNER): no near-bound rules") {
        auto sp = spec("LN_BOBYQA", false);
        sp.step = 0.25;
        const auto plan = make_step_plan(sp, {}, optimizer_caps(sp));
        CHECK(plan.dx[1] == Approx(0.002));
        CHECK(plan.note[1].empty());
    }
}

TEST_CASE("StepPlan: format_step_report lists every dimension",
          "[HPO][step_plan]")
{
    auto sp = spec("LN_SBPLX", true);
    sp.step = 0.05;
    const auto start = sp.in(sp.optim_init.front());
    const auto plan = make_step_plan(sp, start, optimizer_caps(sp));
    const std::string r = format_step_report(plan, sp, optimizer_caps(sp), start);
    CHECK(r.find("Initial step (NLOPT:LN_SBPLX)") != std::string::npos);
    CHECK(r.find("consumption:") != std::string::npos);
    CHECK(r.find("CGRID2B[1]") != std::string::npos);
    CHECK(r.find("AMPGRIDMB[6]") != std::string::npos);
    CHECK(r.find("0.0500") != std::string::npos);
}

TEST_CASE("Validator: STEP above 0.5 is rejected with the migration hint",
          "[HPO][step_plan][scale]")
{
    auto sp = spec("LN_SBPLX", false);
    sp.step = 0.7;
    REQUIRE_THROWS_WITH(validate_optimizer_scaling(sp, "OUTER", false),
                        Catch::Contains("(0, 0.5]"));
    sp.step = 0.5;
    REQUIRE_NOTHROW(validate_optimizer_scaling(sp, "OUTER", false));
    sp.step = 0.1;
    sp.step_overrides = {{"NOPE", {}, 0.1}};
    REQUIRE_THROWS_WITH(validate_optimizer_scaling(sp, "OUTER", false),
                        Catch::Contains("NOPE") && Catch::Contains("not optimised"));
}

TEST_CASE("NLopt adapter: the plan's vector reaches nlopt; default is read back",
          "[HPO][step_plan][nlopt]")
{
    SECTION("STEP set: get_initial_step returns the plan") {
        auto sp = spec("LN_BOBYQA", false);
        sp.step = 0.1;
        const auto start = sp.in(sp.optim_init.front());
        const auto plan = make_step_plan(sp, start, optimizer_caps(sp));
        nlopt::opt opt = make_nlopt(sp, 2, &plan);
        std::vector<double> dx(2, 0.0);
        opt.get_initial_step(start, dx);
        CHECK(dx[0] == Approx(0.2));
        CHECK(dx[1] == Approx(8e-4));
    }
    SECTION("STEP unset: NLopt heuristic, x-dependent, not persisted") {
        auto sp = spec("LN_SBPLX", false);
        nlopt::opt opt = make_nlopt(sp, 2, nullptr);
        opt.set_lower_bounds(sp.low);
        opt.set_upper_bounds(sp.high);
        std::vector<double> mid = {1.0, 0.004}, near = {1.9, 0.004};
        std::vector<double> dx1(2), dx2(2);
        opt.get_initial_step(mid, dx1);
        opt.get_initial_step(near, dx2);
        CHECK(dx1[0] == Approx(0.25 * 2.0));       /* 0.25*width */
        CHECK(dx1[1] == Approx(0.25 * 0.008));
        CHECK(dx2[0] == Approx(0.75 * 0.1));       /* 0.75*slack near HI */
        CHECK(dx2[1] == Approx(0.25 * 0.008));
    }
    SECTION("INNER STEP builds its own plan when the outer has none") {
        auto sp = spec("G_MLSL_LDS", false);
        sp.inner = std::make_unique<OptimizerSpec>(spec("LN_BOBYQA", false));
        sp.inner->step = 0.1;
        REQUIRE_NOTHROW(make_nlopt(sp, 2, nullptr));
    }
}

TEST_CASE("NLopt adapter: a MAXEVAL above the int NLopt counts in is refused, "
          "not wrapped", "[HPO][nlopt][strict_numbers]")
{
    /* nlopt::opt::set_maxeval takes an int, and MAXEVAL (a uint64) was
     * passed on with an implicit conversion: 4294967299 became 3 and
     * 2147483648 a negative budget, which NLopt reads as no limit.
     * HPOTargets bounds MAXEVAL at its line; a spec built another way, for
     * the run or its INNER, is refused here. */
    auto sp = spec("LN_SBPLX", false);
    sp.stop.maxeval = 2147483647u;
    CHECK(make_nlopt(sp, 2, nullptr).get_maxeval() == 2147483647);
    for (const std::uint64_t n : {std::uint64_t{2147483648u},
                                  std::uint64_t{4294967299u}}) {
        INFO("MAXEVAL " << n);
        sp.stop.maxeval = n;
        CHECK_THROWS_WITH(make_nlopt(sp, 2, nullptr),
                          "NLopt spec with MAXEVAL " + std::to_string(n) +
                          ": NLopt counts evaluations in an int (at most "
                          "2147483647)");
        auto outer = spec("G_MLSL_LDS", false);
        outer.inner = std::make_unique<OptimizerSpec>(spec("LN_SBPLX", false));
        outer.inner->stop.maxeval = n;
        CHECK_THROWS_WITH(make_nlopt(outer, 2, nullptr),
                          "NLopt spec with MAXEVAL " + std::to_string(n) +
                          ": NLopt counts evaluations in an int (at most "
                          "2147483647)");
    }
}

TEST_CASE("StepPlan: review fixes -- BOBYQA-only near-bound rule, 0.5 cap, coverage at validation",
          "[HPO][step_plan][review]")
{
    SECTION("MinCollapsed (NEWUOA_BOUND) keeps every dim's step near a bound") {
        auto sp = spec("LN_NEWUOA_BOUND", false);
        sp.step = 0.25;
        sp.optim_init = {{1.0, 0.0079}};
        const auto start = sp.in(sp.optim_init.front());
        const auto plan = make_step_plan(sp, start, optimizer_caps(sp));
        CHECK(plan.dx[1] == Approx(0.002));      /* not shrunk */
        CHECK(plan.note[1].empty());
    }
    SECTION("COBYLA is not relocated by NLopt: no shrink either") {
        auto sp = spec("LN_COBYLA", false);
        sp.step = 0.25;
        sp.optim_init = {{1.0, 0.0079}};
        const auto start = sp.in(sp.optim_init.front());
        const auto plan = make_step_plan(sp, start, optimizer_caps(sp));
        CHECK(plan.dx[1] == Approx(0.002));
    }
    SECTION("BOBYQA at STEP 0.5 stays strictly inside Powell's width >= 2*rhobeg") {
        auto sp = spec("LN_BOBYQA", false);
        sp.step = 0.5;
        const auto start = sp.in(sp.optim_init.front());
        const auto plan = make_step_plan(sp, start, optimizer_caps(sp));
        CHECK(plan.dx[0] < 0.5 * 2.0);
        CHECK(plan.dx[0] > 0.499 * 2.0);
        CHECK(plan.note[0].find("capped") != std::string::npos);
        /* and NLopt accepts the setup: optimize() on a trivial objective */
        nlopt::opt opt = make_nlopt(sp, 2, &plan);
        opt.set_lower_bounds(sp.low); opt.set_upper_bounds(sp.high);
        opt.set_min_objective([](const std::vector<double>& x, std::vector<double>&, void*) {
            return (x[0] - 1.0) * (x[0] - 1.0) + x[1] * x[1]; }, nullptr);
        opt.set_maxeval(20);
        std::vector<double> x = start; double f = 0.0;
        REQUIRE_NOTHROW(opt.optimize(x, f));
    }
    SECTION("SBPLX at STEP 0.5 is untouched (no cap)") {
        auto sp = spec("LN_SBPLX", false);
        sp.step = 0.5;
        const auto start = sp.in(sp.optim_init.front());
        const auto plan = make_step_plan(sp, start, optimizer_caps(sp));
        CHECK(plan.frac[0] == Approx(0.5));
    }
    SECTION("partial overrides without STEP are rejected by the validator, not at run time") {
        auto sp = spec("LN_SBPLX", false);
        sp.step_overrides = {{"CGRID2B", {}, 0.3}};
        REQUIRE_THROWS_WITH(validate_optimizer_scaling(sp, "OUTER", false),
                            Catch::Contains("cover only some dimensions"));
        sp.step_overrides.push_back({"AMPGRIDMB", {}, 0.2});
        REQUIRE_NOTHROW(validate_optimizer_scaling(sp, "OUTER", false));
    }
}
