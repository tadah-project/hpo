/* Per-dimension initial-step plan (initial_step.h) and the NLopt
 * adapter's use of it (make_nlopt + get_initial_step read-back). */
#include "catch2/catch.hpp"
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/initial_step.h>
#include <tadah/hpo/nlopt_adapter.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/search_space.h>
#include <tadah/hpo/scale_validation.h>

#include <dlib/matrix.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
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
    SECTION("BOBYQA: only the tight dim's start moves, every step stays") {
        /* B, within 1e-4 of its HI bound (less than half its step, 0.002),
         * starts on the bound with its full step. Before the fix: B's step
         * shrank to 0.75*slack (7.5e-5) and its start stayed. */
        auto sp = spec("LN_BOBYQA", false);
        sp.step = 0.25;
        sp.optim_init = {{1.0, 0.0079}};   /* B within 1e-4 of its HI bound */
        const auto start = sp.in(sp.optim_init.front());
        const auto plan = make_step_plan(sp, start, optimizer_caps(sp));
        CHECK(plan.given);
        CHECK(plan.dx[0] == Approx(0.5));                       /* untouched */
        CHECK(plan.dx[1] == Approx(0.002));                     /* full */
        REQUIRE(plan.start.size() == 2u);
        CHECK(plan.start[0] == 1.0);
        CHECK(plan.start[1] == 0.008);                          /* on HI */
        CHECK(plan.note[1].find("start moved onto the upper bound") !=
              std::string::npos);
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
        nlopt::opt opt = make_nlopt(sp, 2, RestartStop::Off, &plan);
        std::vector<double> dx(2, 0.0);
        opt.get_initial_step(start, dx);
        CHECK(dx[0] == Approx(0.2));
        CHECK(dx[1] == Approx(8e-4));
    }
    SECTION("STEP unset: NLopt heuristic, x-dependent, not persisted") {
        auto sp = spec("LN_SBPLX", false);
        nlopt::opt opt = make_nlopt(sp, 2, RestartStop::Off, nullptr);
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
        sp.stop.maxeval = 100;   // MLSL needs a stop (require_stop)
        sp.inner = std::make_unique<OptimizerSpec>(spec("LN_BOBYQA", false));
        sp.inner->step = 0.1;
        REQUIRE_NOTHROW(make_nlopt(sp, 2, RestartStop::Off, nullptr));
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
    CHECK(make_nlopt(sp, 2, RestartStop::Off, nullptr).get_maxeval() ==
          2147483647);
    for (const std::uint64_t n : {std::uint64_t{2147483648u},
                                  std::uint64_t{4294967299u}}) {
        INFO("MAXEVAL " << n);
        sp.stop.maxeval = n;
        CHECK_THROWS_WITH(make_nlopt(sp, 2, RestartStop::Off, nullptr),
                          "NLopt spec with MAXEVAL " + std::to_string(n) +
                          ": NLopt counts evaluations in an int (at most "
                          "2147483647)");
        auto outer = spec("G_MLSL_LDS", false);
        outer.inner = std::make_unique<OptimizerSpec>(spec("LN_SBPLX", false));
        outer.inner->stop.maxeval = n;
        CHECK_THROWS_WITH(make_nlopt(outer, 2, RestartStop::Off, nullptr),
                          "NLopt spec with MAXEVAL " + std::to_string(n) +
                          ": NLopt counts evaluations in an int (at most "
                          "2147483647)");
    }
}

TEST_CASE("StepPlan: BOBYQA-only near-bound rule, 0.5 cap, coverage at validation",
          "[HPO][step_plan][review]")
{
    /* LN_PRAXIS is the bounded MinCollapsed algorithm here; this section
     * used LN_NEWUOA_BOUND, which is refused. Before the fix: passes as
     * well (a pin of the BOBYQA-only shrink). */
    SECTION("MinCollapsed (PRAXIS) keeps every dim's step near a bound") {
        auto sp = spec("LN_PRAXIS", false);
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
        nlopt::opt opt = make_nlopt(sp, 2, RestartStop::Off, &plan);
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

namespace {

using Vec = dlib::matrix<double, 0, 1>;
struct NoRunner {};   // the GradientProvider's Runner is only a type tag

/* Records every point asked for; a bowl centred in the box of spec(). */
struct StepRecorder {
    std::vector<Vec> asked;
    double operator()(const Vec& x) {
        asked.push_back(x);
        const double a = x(0) - 1.0, b = (x(1) - 0.004) / 0.008;
        return a * a + b * b;
    }
};

/* An open journal; `text()` closes it and returns it. */
struct StepJournal {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("tadah_step_plan_" +
         std::to_string(std::chrono::steady_clock::now()
                            .time_since_epoch().count()) +
         ".log");
    StepJournal() {
        auto& log = HPOLog::instance();
        log.reset_for_test();
        log.open(path.string(), "test", "/", 0);
    }
    std::string text() {
        HPOLog::instance().finish();
        std::ifstream f(path);
        std::ostringstream s;
        s << f.rdbuf();
        return s.str();
    }
    ~StepJournal() {
        HPOLog::instance().reset_for_test();
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

/* Whether a point of `asked` has AMPGRIDMB[6] = `b` (to 1e-12). */
bool asks_b(const std::vector<Vec>& asked, double b)
{
    return std::any_of(asked.begin(), asked.end(), [b](const Vec& p) {
        return p(1) == Approx(b).epsilon(1e-12);
    });
}

/* One LN_BOBYQA search (MAXEVAL 9) of spec() from (1, `b`), a start of
 * origin `origin`, with STEP `step` (none: unset); the points asked for and
 * the journal. */
std::pair<std::vector<Vec>, std::string>
bobyqa_from(double b, std::optional<double> step, InitOrigin::Kind origin)
{
    auto sp = spec("LN_BOBYQA", false);
    sp.step = step;
    sp.optim_init = {{1.0, b}};
    sp.optim_init_origin = {InitOrigin{origin, {}}};
    sp.stop.maxeval = 9;
    StepRecorder host;
    StepJournal j;
    REQUIRE_NOTHROW(run_nlopt_impl<Vec, NoRunner>(sp, host));
    return {host.asked, j.text()};
}

}  // namespace

/* Pins: LN_BOBYQA from AMPGRIDMB[6] = 0.0079, 1e-4 below its upper bound
 * 0.008 (width 0.008), with STEP 0.25 and with STEP unset (a step of 0.25
 * of the width, 0.002, given to NLopt): the gap is below half the step, so
 * the search starts on the bound -- its first point is (1, 0.008) -- and
 * keeps its full step: a point of its first model has AMPGRIDMB[6] =
 * 0.006, one step inside. The journal gives the old and the new start ("NLOPT:LN_BOBYQA:
 * AMPGRIDMB[6] starts at 0.008 instead of 0.0079"); a CONFIG start that
 * moves gets a WARNING, a sampled (RANDOM) start does not.
 * Before the fix: the search started at 0.0079 and the step of
 * AMPGRIDMB[6] shrank to 0.75 of the gap, 7.5e-5 (with STEP unset by
 * NLopt's own default rule), as the scale of that coordinate: no point of
 * the 9 had AMPGRIDMB[6] = 0.006, and the journal had neither line. */
TEST_CASE("LN_BOBYQA: a start within half a step of a bound starts on the "
          "bound with the full step", "[HPO][step_plan][nlopt]")
{
    for (const std::optional<double> step :
         {std::optional<double>(0.25), std::optional<double>()}) {
        for (const auto origin :
             {InitOrigin::Kind::Config, InitOrigin::Kind::Random}) {
            const bool config = origin == InitOrigin::Kind::Config;
            DYNAMIC_SECTION((step ? "STEP 0.25" : "STEP unset")
                            << (config ? ", CONFIG" : ", RANDOM")) {
                const auto [asked, journal] = bobyqa_from(0.0079, step, origin);
                REQUIRE(asked.size() == 9u);
                CHECK(asked.front()(0) == 1.0);
                CHECK(asked.front()(1) == Approx(0.008).epsilon(1e-12));
                CHECK(asks_b(asked, 0.006));
                CHECK(journal.find("NLOPT:LN_BOBYQA: AMPGRIDMB[6] starts at "
                                   "0.008 instead of 0.0079") !=
                      std::string::npos);
                CHECK((journal.find("WARNING   NLOPT:LN_BOBYQA: the CONFIG "
                                    "start moves") != std::string::npos) ==
                      config);
            }
        }
    }
}

/* Pins: LN_BOBYQA (STEP 0.25: a step of 0.002 on AMPGRIDMB[6]) from
 * AMPGRIDMB[6] = 0.0068, 0.0012 below its upper bound: the gap is at least
 * half the step, so the search starts one step from the bound, at 0.006
 * (moved by 0.0008, less than half a step), and keeps its full step: a
 * point of its first model has AMPGRIDMB[6] = 0.004, one step further in.
 * Before the fix: the search started at 0.0068 and the step of
 * AMPGRIDMB[6] shrank to 0.75 of the gap, 9e-4: no point had AMPGRIDMB[6]
 * = 0.004, and the journal had no INFO line. */
TEST_CASE("LN_BOBYQA: a start between half a step and a step from a bound "
          "starts one step from it", "[HPO][step_plan][nlopt]")
{
    const auto [asked, journal] =
        bobyqa_from(0.0068, 0.25, InitOrigin::Kind::Random);
    REQUIRE(asked.size() == 9u);
    CHECK(asked.front()(0) == 1.0);
    CHECK(asked.front()(1) == Approx(0.006).epsilon(1e-12));
    CHECK(asks_b(asked, 0.004));
    CHECK(journal.find("NLOPT:LN_BOBYQA: AMPGRIDMB[6] starts at 0.006 instead "
                       "of 0.0068") != std::string::npos);
}

/* Pins: LN_COBYLA with STEP unset from AMPGRIDMB[6] = 0.0079, 1e-4 below
 * its upper bound: it is given a step of 0.25 of each width (0.002 on
 * AMPGRIDMB[6]), which it keeps as the scale of the coordinate, and its
 * first simplex turns the step inward at the bound: a point of its first
 * simplex has AMPGRIDMB[6] = 0.0059; the start stays (COBYLA takes any
 * start with its full step) and the journal has no INFO line of a moved
 * start.
 * Before the fix: NLopt's own default shrank the step of AMPGRIDMB[6] to
 * 0.75 of the gap, 7.5e-5, and COBYLA kept it as the scale of the
 * coordinate: no point had AMPGRIDMB[6] = 0.0059. */
TEST_CASE("LN_COBYLA: with STEP unset a start next to a bound keeps the full "
          "step", "[HPO][step_plan][nlopt]")
{
    auto sp = spec("LN_COBYLA", false);
    sp.optim_init = {{1.0, 0.0079}};
    sp.optim_init_origin = {InitOrigin{InitOrigin::Kind::Config, {}}};
    sp.stop.maxeval = 9;
    StepRecorder host;
    StepJournal j;
    REQUIRE_NOTHROW(run_nlopt_impl<Vec, NoRunner>(sp, host));
    const std::string journal = j.text();
    REQUIRE(host.asked.size() == 9u);
    CHECK(host.asked.front()(0) == 1.0);
    CHECK(host.asked.front()(1) == 0.0079);
    CHECK(asks_b(host.asked, 0.0059));
    CHECK(journal.find("starts at") == std::string::npos);
    CHECK(journal.find("STEP unset -- 0.25*width per dim, given to NLopt") !=
          std::string::npos);
}

namespace {

/* One search of `algo` (an MLSL or AUGLAG) on spec() from (1, `b`), a
 * CONFIG start, with STEP unset, MAXEVAL `maxeval` and the INNER block
 * `inner` (empty: none, NLopt's default local search); the points asked for
 * and the journal. */
template <class Host>
std::pair<std::vector<Vec>, std::string>
inner_search_from(const char* algo, const char* inner, double b,
                  unsigned maxeval, Host& host)
{
    auto sp = spec(algo, false);
    sp.optim_init = {{1.0, b}};
    sp.optim_init_origin = {InitOrigin{InitOrigin::Kind::Config, {}}};
    sp.stop.maxeval = maxeval;
    if (inner[0]) {
        sp.inner = std::make_unique<OptimizerSpec>();
        sp.inner->lib = "NLOPT";
        sp.inner->algo = inner;
    }
    StepJournal j;
    REQUIRE_NOTHROW(run_nlopt_impl<Vec, NoRunner>(sp, host));
    return {host.asked, j.text()};
}

/* (A - 2)^2 + ((B - 0.008) / 0.008)^2 on the box of spec(): the minimum at
 * the upper corner (2, 0.008), so MLSL's best samples, the starts of its
 * local searches, lie next to the upper bounds. */
struct UpperCornerBowl {
    std::vector<Vec> asked;
    double operator()(const Vec& x) {
        asked.push_back(x);
        const double a = x(0) - 2.0, b = (x(1) - 0.008) / 0.008;
        return a * a + b * b;
    }
};

}  // namespace

/* Pins: LN_AUGLAG with an INNER LN_BOBYQA and STEP unset (MAXEVAL 9), from
 * AMPGRIDMB[6] = 0.0079, 1e-4 below its upper bound: AUGLAG runs its one
 * local search from its start, and that search is given 0.25 of each width
 * (0.002 on AMPGRIDMB[6]) and moves the start onto the bound, as an
 * LN_BOBYQA search of its own does: the first point is (1, 0.008), a point
 * of the first model has AMPGRIDMB[6] = 0.006, one step inside, the journal
 * gives the old and the new start ("NLOPT:LN_AUGLAG (its local search
 * LN_BOBYQA): AMPGRIDMB[6] starts at 0.008 instead of 0.0079") and the
 * CONFIG start that moves gets a WARNING.
 * Before the fix: the local search was given no step, and took NLopt's
 * default from the start, which shrank the step of AMPGRIDMB[6] to 0.75 of
 * the gap, 7.5e-5, as the scale of that coordinate: the search started at
 * 0.0079, no point of the 9 had AMPGRIDMB[6] = 0.006, and the journal had
 * neither line. */
TEST_CASE("LN_AUGLAG with an INNER LN_BOBYQA: a start within half a step of "
          "a bound starts on the bound with the full step",
          "[HPO][step_plan][nlopt]")
{
    StepRecorder host;
    const auto [asked, journal] =
        inner_search_from("LN_AUGLAG", "LN_BOBYQA", 0.0079, 9, host);
    REQUIRE(asked.size() >= 5u);
    CHECK(asked.front()(0) == 1.0);
    CHECK(asked.front()(1) == Approx(0.008).epsilon(1e-12));
    CHECK(asks_b(asked, 0.006));
    CHECK(journal.find("NLOPT:LN_AUGLAG (its local search LN_BOBYQA): "
                       "AMPGRIDMB[6] starts at 0.008 instead of 0.0079") !=
          std::string::npos);
    CHECK(journal.find("WARNING   NLOPT:LN_AUGLAG (its local search "
                       "LN_BOBYQA): the CONFIG start moves") !=
          std::string::npos);
}

/* Pins: LN_AUGLAG without an INNER block (NLopt's default local search
 * LN_COBYLA) and STEP unset (MAXEVAL 9), from AMPGRIDMB[6] = 0.0079: the
 * local search is given 0.25 of each width, which it keeps as the scale of
 * the coordinate, and turns its first step inward at the bound: a point of
 * its first simplex has AMPGRIDMB[6] = 0.0059; the start stays, and the
 * journal's step line names the local search's step.
 * Before the fix: the local search took NLopt's default from the start,
 * which shrank the step of AMPGRIDMB[6] to 0.75 of the gap, 7.5e-5: no
 * point had AMPGRIDMB[6] = 0.0059, and the journal's step line was NLopt's
 * default ("STEP unset -- NLopt default per dim ..."). */
TEST_CASE("LN_AUGLAG with its default local search LN_COBYLA: a start next "
          "to a bound keeps the full step", "[HPO][step_plan][nlopt]")
{
    StepRecorder host;
    const auto [asked, journal] =
        inner_search_from("LN_AUGLAG", "", 0.0079, 9, host);
    REQUIRE(asked.size() >= 3u);
    CHECK(asked.front()(1) == 0.0079);
    CHECK(asks_b(asked, 0.0059));
    CHECK(journal.find("starts at") == std::string::npos);
    CHECK(journal.find("STEP unset -- 0.25*width per dim, given to NLopt: "
                       "LN_COBYLA keeps the first step of each dimension as "
                       "its scale") != std::string::npos);
}

/* Pins: GN_MLSL_LDS with an INNER LN_COBYLA and STEP unset (MAXEVAL 60) on
 * the bowl whose minimum is the upper corner, from (1, 0.004): every local
 * search is given 0.25 of each width (0.5 on CGRID2B[1], 0.002 on
 * AMPGRIDMB[6]), whatever its start, and turns a step inward at a bound.
 * MLSL evaluates its start and four Sobol samples, then starts a local
 * search at the sample (1.1875, 0.00775), 0.00025 below the upper bound of
 * AMPGRIDMB[6]: its first simplex asks for (1.6875, 0.00775) and (1.6875,
 * 0.00575), 0.002 inward; the local search from (1.6875, 0.00175), 0.3125
 * below the upper bound of CGRID2B[1], asks for (1.1875, 0.00175), 0.5
 * inward, and (1.6875, 0.00375). The journal's step line names the step
 * of the local search ("... given to NLopt: LN_COBYLA keeps the first step
 * of each dimension as its scale, ... (the step of the local search, which
 * GN_MLSL_LDS hands it)").
 * Before the fix: each local search took NLopt's default from its own
 * start, which shrinks a step to 0.75 of the distance to a bound closer
 * than a quarter of the width: eval 8 was (1.6875, 0.0079375), a step of
 * 1.875e-4 on AMPGRIDMB[6], and eval 32 (1.921875, 0.00175), a step of
 * 0.234375 on CGRID2B[1], then eval 33 (1.921875, 0.0030625); the journal's
 * step line was NLopt's default ("STEP unset -- NLopt default per dim
 * ..."). */
TEST_CASE("GN_MLSL_LDS with an INNER LN_COBYLA: every local search keeps the "
          "full step", "[HPO][step_plan][nlopt]")
{
    UpperCornerBowl host;
    const auto [asked, journal] =
        inner_search_from("GN_MLSL_LDS", "LN_COBYLA", 0.004, 60, host);
    const auto at = [&asked](std::size_t eval, double a, double b) {
        INFO("eval " << eval << ": (" << asked[eval - 1](0) << ", "
                     << asked[eval - 1](1) << ")");
        CHECK(asked[eval - 1](0) == Approx(a).epsilon(1e-12));
        CHECK(asked[eval - 1](1) == Approx(b).epsilon(1e-12));
    };
    REQUIRE(asked.size() == 60u);
    at(6, 1.1875, 0.00775);
    at(7, 1.6875, 0.00775);
    at(8, 1.6875, 0.00575);
    at(31, 1.6875, 0.00175);
    at(32, 1.1875, 0.00175);
    at(33, 1.6875, 0.00375);
    CHECK(journal.find("STEP unset -- 0.25*width per dim, given to NLopt: "
                       "LN_COBYLA keeps the first step of each dimension as "
                       "its scale") != std::string::npos);
    CHECK(journal.find("(the step of the local search, which GN_MLSL_LDS "
                       "hands it)") != std::string::npos);
}
