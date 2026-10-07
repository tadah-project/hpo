/* OptimizerCaps STEP policy (StepUse) -- table-driven registry facts.
 * The registry is the single source of truth for which backends honour
 * an OPTIMIZER-block STEP and how; scale_validation.cpp and the journal
 * derive their behaviour from it, so these facts are pinned here. */
#include "catch2/catch.hpp"
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/optimizer_spec.h>

#include <nlopt.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace tadah::hpo;

TEST_CASE("OptimizerCaps: StepUse classification per backend",
          "[HPO][caps][step]")
{
    struct Row { const char* lib; const char* algo; StepUse use; };
    const std::vector<Row> rows = {
        {"NLOPT", "LN_SBPLX",        StepUse::PerDimVector},
        {"NLOPT", "LN_NELDERMEAD",   StepUse::PerDimVector},
        {"NLOPT", "LN_BOBYQA",       StepUse::PerDimRescale},
        {"NLOPT", "LN_COBYLA",       StepUse::PerDimRescale},
        {"NLOPT", "LN_NEWUOA",       StepUse::MinCollapsed},
        /* LN_NEWUOA_BOUND (MinCollapsed too) is refused: see the test
         * case below. */
        {"NLOPT", "LN_PRAXIS",       StepUse::MinCollapsed},
        {"NLOPT", "LD_MMA",          StepUse::TrustHalfWidth},
        {"NLOPT", "LD_CCSAQ",        StepUse::TrustHalfWidth},
        {"NLOPT", "G_MLSL_LDS",      StepUse::ForwardedToInner},
        {"NLOPT", "G_MLSL",          StepUse::ForwardedToInner},
        {"NLOPT", "LD_AUGLAG",       StepUse::ForwardedToInner},
        {"NLOPT", "GN_CRS2_LM",      StepUse::Ignored},
        {"NLOPT", "GN_ISRES",        StepUse::Ignored},
        {"NLOPT", "GN_ESCH",         StepUse::Ignored},
        {"NLOPT", "GN_DIRECT",       StepUse::Ignored},
        {"NLOPT", "GD_STOGO",        StepUse::Ignored},
        {"NLOPT", "LD_LBFGS",        StepUse::Ignored},
        {"NLOPT", "LD_SLSQP",        StepUse::Ignored},
        {"NLOPT", "LD_TNEWTON",      StepUse::Ignored},
        {"TADAH", "GRID",            StepUse::GridCount},
        {"TADAH", "RANDOM",          StepUse::Rejected},
        {"TADAH", "ANNEAL",          StepUse::WidthFraction},
        {"TADAH", "ANNEAL_PT",       StepUse::WidthFraction},
        {"TADAH", "ANNEAL_PA",       StepUse::WidthFraction},
        {"ASA",   "ASA",             StepUse::Rejected},
        {"DLIB",  "GFS",             StepUse::Rejected},
        {"DLIB",  "BFGS",            StepUse::Rejected},
        {"CERES", "LBFGS",           StepUse::Rejected},
        {"CERES", "STEEPEST_DESCENT",StepUse::Rejected},
    };
    for (const auto& r : rows) {
        INFO(r.lib << ":" << r.algo);
        const auto& c = optimizer_caps(r.lib, r.algo);
        CHECK(c.step_use == r.use);
        CHECK(std::string(c.step_note).size() > 10);
    }
}

TEST_CASE("OptimizerCaps: step_use_consumes and names",
          "[HPO][caps][step]")
{
    CHECK(step_use_consumes(StepUse::PerDimVector));
    CHECK(step_use_consumes(StepUse::PerDimRescale));
    CHECK(step_use_consumes(StepUse::MinCollapsed));
    CHECK(step_use_consumes(StepUse::TrustHalfWidth));
    CHECK(step_use_consumes(StepUse::ForwardedToInner));
    CHECK_FALSE(step_use_consumes(StepUse::Rejected));
    CHECK_FALSE(step_use_consumes(StepUse::Ignored));
    CHECK_FALSE(step_use_consumes(StepUse::GridCount));
    CHECK_FALSE(step_use_consumes(StepUse::WidthFraction));
    CHECK(std::string(step_use_name(StepUse::PerDimRescale)).find("rescaling")
          != std::string::npos);
    CHECK(std::string(step_use_name(StepUse::GridCount)).find("GRID_POINTS")
          != std::string::npos);
}

TEST_CASE("OptimizerCaps: every registered backend carries a STEP note",
          "[HPO][caps][step]")
{
    auto all = filter_optimizer_caps([](const OptimizerCaps&) { return true; });
    REQUIRE(all.size() > 30);
    for (const auto* c : all) {
        INFO(c->lib << ":" << c->algo);
        CHECK(c->step_note != nullptr);
        CHECK(std::string(c->step_note).size() > 0);
        if (c->ignores_bounds) CHECK(std::string(c->bounds_note).size() > 0);
    }
}

TEST_CASE("OptimizerCaps: bounds-ignoring backends are flagged",
          "[HPO][caps][bounds]")
{
    CHECK(optimizer_caps("NLOPT", "LN_NEWUOA").ignores_bounds);
    /* LN_NEWUOA_BOUND, which kept to the box, is refused: see the test
     * case below. */
    CHECK_FALSE(optimizer_caps("NLOPT", "LN_BOBYQA").ignores_bounds);
    CHECK(optimizer_caps("CERES", "LBFGS").ignores_bounds);
    CHECK(optimizer_caps("CERES", "NLCG").ignores_bounds);
    CHECK_FALSE(optimizer_caps("DLIB", "BFGS").ignores_bounds);
    CHECK_FALSE(optimizer_caps("TADAH", "ANNEAL").ignores_bounds);
}

TEST_CASE("OptimizerCaps: NLOPT:LN_NEWUOA_BOUND is refused, naming the "
          "replacement", "[HPO][caps][refused]")
{
    /* NLopt 2.10 plans every LN_NEWUOA_BOUND step with a nested LD_MMA run
     * that has no time limit and sees no stop request; once a non-finite
     * value reaches its model -- HUGE_VAL, NLopt's convention for a forced
     * stop, which HPO's objective returns after every stop it makes, or a
     * model that overflows on a flat loss -- each step takes 10-60 s
     * without evaluating the loss (docs/issues/nlopt_newuoa_bound_stall.md).
     *
     * Pins: the caps lookup refuses it with the reason and the
     * replacements, LN_BOBYQA and LN_NEWUOA (not refused, as LN_BOBYQA is
     * not); no registry list offers it (filter_optimizer_caps, and through
     * it every "alternatives" list); an unregistered name keeps its own
     * message; the ALGO help says it is refused instead of describing its
     * STEP and its stops.
     * Before the fix: optimizer_caps() returned its caps (MinCollapsed,
     * keeps to the box), the filter found it, and the help printed its
     * stops and its STEP line. */
    CHECK_THROWS_WITH(optimizer_caps("NLOPT", "LN_NEWUOA_BOUND"),
                      Catch::StartsWith("NLOPT:LN_NEWUOA_BOUND is refused: ") &&
                      Catch::Contains("nested LD_MMA run") &&
                      Catch::Contains("non-finite value") &&
                      Catch::Contains("HUGE_VAL") &&
                      Catch::Contains("Use NLOPT:LN_BOBYQA") &&
                      Catch::Contains("or NLOPT:LN_NEWUOA where"));
    /* Pins: the reason says from which evaluation a stop stalls the run:
     * from its (2n+1)-th on, the last of the 2n+1 starting points NEWUOA
     * evaluates before its first step -- a stop on one of the first 2n
     * returns at once (docs/issues/nlopt_newuoa_bound_stall.md, case F with
     * STOPAT).
     * Before the review fix: "after every stop HPO makes through the
     * objective (...)", with no such bound. */
    CHECK_THROWS_WITH(optimizer_caps("NLOPT", "LN_NEWUOA_BOUND"),
                      Catch::Contains("after every stop HPO makes through "
                                      "the objective (a signal, a RESTART "
                                      "stagnation stop, an error that ends "
                                      "the run) from a run's (2n+1)-th "
                                      "evaluation on, the last of NEWUOA's "
                                      "starting points for n OPTIM "
                                      "dimensions,"));
    CHECK(filter_optimizer_caps([](const OptimizerCaps& c) {
              return c.lib == "NLOPT" && c.algo == "LN_NEWUOA_BOUND";
          }).empty());
    CHECK_NOTHROW(optimizer_caps("NLOPT", "LN_NEWUOA"));
    CHECK_NOTHROW(optimizer_caps("NLOPT", "LN_BOBYQA"));
    CHECK_THROWS_WITH(optimizer_caps("NLOPT", "NOT_AN_ALGO"),
                      "unknown optimizer NLOPT:NOT_AN_ALGO (not registered "
                      "in optimizer_caps)");

    std::ostringstream out;
    std::streambuf* const old = std::cout.rdbuf(out.rdbuf());
    OptimizerSpec::print_caps("NLOPT", "LN_NEWUOA_BOUND");
    std::cout.rdbuf(old);
    INFO("help:\n" << out.str());
    CHECK_THAT(out.str(),
               Catch::StartsWith("Options for NLOPT:LN_NEWUOA_BOUND\n"
                                 "   * REFUSED: ") &&
               Catch::Contains("Use NLOPT:LN_BOBYQA") &&
               !Catch::Contains("STEP <double>") &&
               !Catch::Contains("Stops that end a run"));
}

TEST_CASE("OptimizerCaps: backend_refusal names the refused backends, and "
          "every NLopt algorithm is registered or refused",
          "[HPO][caps][refused]")
{
    /* Pins: backend_refusal() returns the reason for (NLOPT,
     * LN_NEWUOA_BOUND) -- and the other refused backends, below -- but for
     * no registered one -- the library is part of the key -- and
     * reject_refused_backend() throws "LIB:ALGO is refused: <reason>",
     * with "<where>: " in front when given, the text optimizer_caps()
     * throws; every algorithm NLopt names is either registered or refused,
     * never both and never neither.
     * Added with the fix, which introduces both functions (the tests
     * commit could not call them). */
    const char* reason = backend_refusal("NLOPT", "LN_NEWUOA_BOUND");
    REQUIRE(reason != nullptr);
    const std::string refused =
        std::string("NLOPT:LN_NEWUOA_BOUND is refused: ") + reason;
    CHECK(backend_refusal("NLOPT", "LN_BOBYQA") == nullptr);
    CHECK(backend_refusal("NLOPT", "LN_NEWUOA") == nullptr);
    CHECK(backend_refusal("NLOPT", "NOT_AN_ALGO") == nullptr);
    CHECK(backend_refusal("TADAH", "LN_NEWUOA_BOUND") == nullptr);
    CHECK_THROWS_WITH(optimizer_caps("NLOPT", "LN_NEWUOA_BOUND"), refused);
    CHECK_THROWS_WITH(reject_refused_backend("NLOPT", "LN_NEWUOA_BOUND"),
                      refused);
    CHECK_THROWS_WITH(reject_refused_backend("NLOPT", "LN_NEWUOA_BOUND",
                                             "the EXPLORE block"),
                      "the EXPLORE block: " + refused);
    CHECK_NOTHROW(reject_refused_backend("NLOPT", "LN_BOBYQA", "anywhere"));
    CHECK(find_optimizer_caps("NLOPT", "LN_NEWUOA_BOUND") == nullptr);

    std::size_t registered = 0;
    for (int i = 0; i < NLOPT_NUM_ALGORITHMS; ++i) {
        const char* name =
            nlopt_algorithm_to_string(static_cast<nlopt_algorithm>(i));
        if (!name || !name[0]) continue;
        INFO(name);
        const bool is_registered = find_optimizer_caps("NLOPT", name) != nullptr;
        const bool is_refused = backend_refusal("NLOPT", name) != nullptr;
        CHECK(is_registered != is_refused);
        if (is_registered) ++registered;
    }
    CHECK(registered ==
          filter_optimizer_caps([](const OptimizerCaps& c) {
              return c.lib == "NLOPT";
          }).size());
}

/* NLopt 2.10 names NLOPT_LD_LBFGS_NOCEDAL (nlopt_algorithm_to_string, the
 * name the caps registry registers each NLopt algorithm under) but does not
 * implement it: nlopt_optimize() returns NLOPT_INVALID_ARGS for it (its
 * algorithm switch has no case for it, optimize.c), so every HPO run of it
 * ended "FAILED -- nlopt invalid argument" after its setup. The
 * registry-wide end-status test (test_hpo_termination.cpp) runs every
 * registered backend and found it.
 * Pins: NLopt refuses it -- so the refusal follows NLopt, and an NLopt that
 * implements it fails here --; HPO refuses it like LN_NEWUOA_BOUND:
 * backend_refusal() says NLopt does not implement it and names
 * NLOPT:LD_LBFGS, the caps lookup throws "NLOPT:NLOPT_LD_LBFGS_NOCEDAL is
 * refused: <reason>", no caps list offers it, and the ALGO help says
 * REFUSED.
 * Before the fix: registered, with no refusal; the lookup succeeded. */
TEST_CASE("OptimizerCaps: NLOPT_LD_LBFGS_NOCEDAL, which NLopt does not "
          "implement, is refused", "[HPO][caps][refused]")
{
    nlopt_opt opt = nlopt_create(NLOPT_LD_LBFGS_NOCEDAL, 1);
    REQUIRE(opt != nullptr);
    const auto square = [](unsigned, const double* x, double* g, void*) {
        if (g) g[0] = 2.0 * x[0];
        return x[0] * x[0];
    };
    nlopt_set_min_objective(opt, square, nullptr);
    nlopt_set_maxeval(opt, 10);
    double x = 0.5, f = 0.0;
    CHECK(nlopt_optimize(opt, &x, &f) == NLOPT_INVALID_ARGS);
    nlopt_destroy(opt);

    const char* reason = backend_refusal("NLOPT", "NLOPT_LD_LBFGS_NOCEDAL");
    REQUIRE(reason != nullptr);
    CHECK_THAT(std::string(reason),
               Catch::Contains("NLopt 2.10 does not implement it") &&
               Catch::Contains("Use NLOPT:LD_LBFGS"));
    CHECK_THROWS_WITH(optimizer_caps("NLOPT", "NLOPT_LD_LBFGS_NOCEDAL"),
                      std::string("NLOPT:NLOPT_LD_LBFGS_NOCEDAL is refused: ") +
                          reason);
    CHECK(find_optimizer_caps("NLOPT", "NLOPT_LD_LBFGS_NOCEDAL") == nullptr);
    CHECK(filter_optimizer_caps([](const OptimizerCaps& c) {
              return c.algo == "NLOPT_LD_LBFGS_NOCEDAL";
          }).empty());

    std::ostringstream out;
    std::streambuf* const old = std::cout.rdbuf(out.rdbuf());
    OptimizerSpec::print_caps("NLOPT", "NLOPT_LD_LBFGS_NOCEDAL");
    std::cout.rdbuf(old);
    CHECK_THAT(out.str(),
               Catch::StartsWith("Options for NLOPT:NLOPT_LD_LBFGS_NOCEDAL\n"
                                 "   * REFUSED: ") &&
               Catch::Contains("Use NLOPT:LD_LBFGS"));
}
