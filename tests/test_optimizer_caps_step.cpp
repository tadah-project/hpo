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
        /* LN_NEWUOA and LN_NEWUOA_BOUND (MinCollapsed too) are refused:
         * see the test cases below. LN_NEWUOA was a row of this table, as
         * a registered MinCollapsed backend, until it was refused. */
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
        /* CERES:LBFGS was a row before the fix (refused since); before the
         * fix the case passes as well (a pin) */
        {"CERES", "STEEPEST_DESCENT",StepUse::Rejected},
    };
    for (const auto& r : rows) {
        INFO(r.lib << ":" << r.algo);
        const auto& c = optimizer_caps(r.lib, r.algo);
        CHECK(c.step_use == r.use);
        CHECK(std::string(c.step_note).size() > 10);
    }
}

/* Pins: the STEP note of an MLSL or AUGLAG (StepUse::ForwardedToInner),
 * which `tadah explain` and the optimiser reference print, names NLopt's
 * default local search LN_COBYLA only for the algorithms NLopt runs it for
 * (default_inner: GN_MLSL, GN_MLSL_LDS, LN_AUGLAG, LN_AUGLAG_EQ); the GD_
 * and LD_ forms run LD_MMA, and AUGLAG, AUGLAG_EQ, G_MLSL and G_MLSL_LDS
 * need an INNER block. Each gives the step of an LN_BOBYQA or LN_COBYLA
 * local search without STEP as 0.25 of the width of each search parameter.
 * The note of LN_BOBYQA and LN_COBYLA says where LN_BOBYQA moves a start
 * next to a bound (onto the bound within half a step of it, otherwise to one
 * step from it) and speaks of search parameters, not dimensions.
 * Before the fix: one note served every MLSL and AUGLAG, "(the INNER
 * block's, or the default LN_COBYLA) takes 0.25 of each width", also for
 * GD_MLSL, whose default is LD_MMA; the LN_BOBYQA note said "onto the
 * bound, or to one step from it" without saying which, and "each
 * dimension takes 0.25 of its width". */
TEST_CASE("OptimizerCaps: the STEP note names a default local search only "
          "where NLopt runs one", "[HPO][caps][step]")
{
    const auto forwarded = filter_optimizer_caps([](const OptimizerCaps& c) {
        return c.step_use == StepUse::ForwardedToInner;
    });
    REQUIRE(forwarded.size() == 12u);
    for (const auto* c : forwarded) {
        INFO(c->lib << ":" << c->algo << ": " << c->step_note);
        const std::string note = c->step_note;
        const bool cobyla_default = c->default_inner &&
                                    std::string(c->default_inner) == "LN_COBYLA";
        CHECK((note.find("the default LN_COBYLA") != std::string::npos) ==
              cobyla_default);
        CHECK(note.find("takes 0.25 of the width of each search parameter") !=
              std::string::npos);
        CHECK(note.find("each width") == std::string::npos);
    }
    for (const char* algo : {"LN_BOBYQA", "LN_COBYLA"}) {
        const auto& c = optimizer_caps("NLOPT", algo);
        const std::string note = c.step_note;
        INFO(algo << ": " << note);
        CHECK(note.find("LN_BOBYQA moves a start that lies within a step of a "
                        "bound by at most half a step: onto the bound when it "
                        "lies within half a step of it, otherwise to one step "
                        "from it") != std::string::npos);
        CHECK(note.find("0.25 of the width of its box") != std::string::npos);
        CHECK(note.find("dimension") == std::string::npos);
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
    /* Pins: the backend that proposes points outside the OPTIM box is
     * CERES:STEEPEST_DESCENT, the CERES driver HPO registers, and only it
     * (Ceres' GradientProblemSolver has no bounds; such a point is scored,
     * not evaluated: box_penalty.h); its note says so; every registered
     * NLopt algorithm keeps to the box.
     * The score's slope points away from the box (box_penalty.h), so a
     * descent step leads back into it, and the note says so.
     * Before the fix: the four CERES drivers were flagged (CERES:LBFGS,
     * NLCG and WOLFE are refused now: their Wolfe line search stalls at a
     * point outside the box), and the note said such points are evaluated
     * there ("... and are evaluated there, and such a point can become the
     * best ..."); then (review) it gave the slope the wrong sign ("...
     * whose slope points back into it, it counts against MAXEVAL ..."): the
     * Contains check fails. */
    const auto unbounded = filter_optimizer_caps(
        [](const OptimizerCaps& c) { return c.ignores_bounds; });
    CHECK(format_caps_list(unbounded, unbounded.size()) ==
          "CERES:STEEPEST_DESCENT");
    CHECK_FALSE(optimizer_caps("NLOPT", "LN_BOBYQA").ignores_bounds);
    CHECK(optimizer_caps("CERES", "STEEPEST_DESCENT").ignores_bounds);
    CHECK_THAT(std::string(optimizer_caps("CERES", "STEEPEST_DESCENT").bounds_note),
               Catch::Contains("Such a point is not evaluated: it is scored "
                               "as a failure graded by its distance to the "
                               "box, whose slope points away from the box, "
                               "so a descent step leads back into it; it "
                               "counts against MAXEVAL and it is never the "
                               "best"));
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
     * replacement, LN_BOBYQA alone (not refused); no registry list offers
     * it (filter_optimizer_caps, and through it every "alternatives"
     * list); the ALGO help says it is refused instead of describing its
     * STEP and its stops.
     * Before the fix: optimizer_caps() returned its caps (MinCollapsed,
     * keeps to the box), the filter found it, and the help printed its
     * stops and its STEP line. Until LN_NEWUOA was refused too, the
     * reason ended "..., or NLOPT:LN_NEWUOA where the search may leave the
     * OPTIM box", and the EndsWith check below fails on it. */
    CHECK_THROWS_WITH(optimizer_caps("NLOPT", "LN_NEWUOA_BOUND"),
                      Catch::StartsWith("NLOPT:LN_NEWUOA_BOUND is refused: ") &&
                      Catch::Contains("nested LD_MMA run") &&
                      Catch::Contains("non-finite value") &&
                      Catch::Contains("HUGE_VAL") &&
                      Catch::EndsWith("The recommended replacement is "
                                      "NLOPT:LN_BOBYQA, Powell's "
                                      "bound-constrained successor of NEWUOA "
                                      "(its STEP rescales each dimension "
                                      "instead of setting one radius)"));
    /* Pins: the reason says from which evaluation a stop stalls the run:
     * from its (2n+1)-th on, the last of the 2n+1 starting points NEWUOA
     * evaluates before its first step -- a stop on one of the first 2n
     * returns at once (docs/issues/nlopt_newuoa_bound_stall.md, case F with
     * STOPAT).
     * Before the review fix: "after every stop HPO makes through the
     * objective (...)", with no such bound. The reason names the command,
     * tadah hpo; before the help was rewritten for a user it said "HPO",
     * and this check fails on that text. */
    CHECK_THROWS_WITH(optimizer_caps("NLOPT", "LN_NEWUOA_BOUND"),
                      Catch::Contains("after every stop tadah hpo makes through "
                                      "the objective (a signal, a RESTART "
                                      "stagnation stop, an error that ends "
                                      "the run) from a run's (2n+1)-th "
                                      "evaluation on, the last of NEWUOA's "
                                      "starting points for n OPTIM "
                                      "dimensions,"));
    CHECK(filter_optimizer_caps([](const OptimizerCaps& c) {
              return c.lib == "NLOPT" && c.algo == "LN_NEWUOA_BOUND";
          }).empty());
    CHECK_NOTHROW(optimizer_caps("NLOPT", "LN_BOBYQA"));
    /* An unregistered name has its own message: "OptimizerCaps: an unknown
     * optimiser is named with what is wrong". */

    std::ostringstream out;
    std::streambuf* const old = std::cout.rdbuf(out.rdbuf());
    OptimizerSpec::print_caps("NLOPT", "LN_NEWUOA_BOUND");
    std::cout.rdbuf(old);
    INFO("help:\n" << out.str());
    CHECK_THAT(out.str(),
               Catch::StartsWith("Options for NLOPT:LN_NEWUOA_BOUND\n"
                                 "   * REFUSED: ") &&
               Catch::Contains("The recommended replacement is "
                               "NLOPT:LN_BOBYQA") &&
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
    /* (LN_NEWUOA was checked here as not refused; it is refused now: see
     * the next test case.) */
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

/* The ALGO help of LIB:ALGO, as OptimizerSpec::print_caps prints it. */
static std::string caps_help(const std::string& lib, const std::string& algo)
{
    std::ostringstream out;
    std::streambuf* const old = std::cout.rdbuf(out.rdbuf());
    OptimizerSpec::print_caps(lib, algo);
    std::cout.rdbuf(old);
    return out.str();
}

/* A refusal's reason ends with the recommended replacement, said of the
 * optimiser ("The recommended replacement is ..."), not to the reader
 * ("Use ..."): the checks of that ending fail on the text before the help
 * was rewritten so.
 * NLopt 2.10 runs LN_NEWUOA without the OPTIM box (optimize.c calls newuoa
 * with lb = ub = NULL): a 2-D run of MAXEVAL 25 made 20 evaluations outside
 * the box, and the best potential, at RCUT2B 8.11 for a [4, 6] box, was one
 * of them (docs/issues/nlopt_newuoa_ignores_bounds.md); in 1-D NLopt refuses
 * it once the run has started ("dimension 1 must be >= 2").
 * Pins: HPO refuses it as it refuses LN_NEWUOA_BOUND (the OPTIM box is a
 * contract): backend_refusal() says why -- NLopt runs it without the box,
 * so the best potential can lie outside the bounds; it needs two OPTIM
 * dimensions -- and names NLOPT:LN_BOBYQA; the caps lookup throws
 * "NLOPT:LN_NEWUOA is refused: <reason>"; no caps list offers it; the ALGO
 * help says REFUSED. The reason says in words that NLopt passes NEWUOA no
 * bounds (it quoted NLopt's source before the help was rewritten for a
 * user: the "it passes NEWUOA no bounds" check fails on that text).
 * Before the fix: registered (ignores_bounds, a startup WARNING), with no
 * refusal: backend_refusal() was nullptr and the REQUIRE fails. */
TEST_CASE("OptimizerCaps: NLOPT:LN_NEWUOA, which ignores the OPTIM box, is "
          "refused, naming LN_BOBYQA", "[HPO][caps][refused]")
{
    const char* reason = backend_refusal("NLOPT", "LN_NEWUOA");
    REQUIRE(reason != nullptr);
    CHECK_THAT(std::string(reason),
               Catch::Contains("without the OPTIM box") &&
               Catch::Contains("it passes NEWUOA no bounds") &&
               Catch::Contains("the best potential can lie outside") &&
               Catch::Contains("two OPTIM dimensions or more") &&
               Catch::EndsWith("The recommended replacement is "
                               "NLOPT:LN_BOBYQA, Powell's "
                               "bound-constrained successor of NEWUOA (its "
                               "STEP rescales each dimension instead of "
                               "setting one radius)"));
    CHECK_THROWS_WITH(optimizer_caps("NLOPT", "LN_NEWUOA"),
                      std::string("NLOPT:LN_NEWUOA is refused: ") + reason);
    CHECK(find_optimizer_caps("NLOPT", "LN_NEWUOA") == nullptr);
    CHECK(filter_optimizer_caps([](const OptimizerCaps& c) {
              return c.algo == "LN_NEWUOA";
          }).empty());
    CHECK_THAT(caps_help("NLOPT", "LN_NEWUOA"),
               Catch::StartsWith("Options for NLOPT:LN_NEWUOA\n"
                                 "   * REFUSED: ") &&
               Catch::Contains("The recommended replacement is "
                               "NLOPT:LN_BOBYQA") &&
               !Catch::Contains("WARNING") &&
               !Catch::Contains("Stops that end a run"));
}

/* Pins: Ceres' least-squares solvers, CERES:LEVENBERG_MARQUARDT, DOGLEG and
 * SUBSPACE_DOGLEG, are refused through the refusal table, as the refused
 * NLopt algorithms are: backend_refusal() says why (they minimise a sum of
 * squared residuals, and HPO gives an optimiser one scalar loss) and names
 * CERES:STEEPEST_DESCENT, the Ceres solver HPO runs, and the
 * bound-constrained NLOPT:LD_LBFGS and DLIB:BFGS; the caps lookup throws
 * "CERES:<ALGO> is refused: <reason>", so every phase's check reports it
 * with the phase and its line (test_startup_validation_strategy.cpp); the
 * ALGO help says REFUSED. The reason names the command, tadah hpo (it said
 * "HPO" before the help was rewritten for a user, and the check of its
 * last sentence fails on that text).
 * Before the fix: the reason named "Use CERES:LBFGS, CERES:WOLFE,
 * CERES:NLCG or CERES:STEEPEST_DESCENT", three solvers refused since (their
 * Wolfe line search stalls at a point outside the OPTIM box), and
 * CERES:LBFGS was the registered solver this case checked. (Before the
 * refusal table, they were unregistered: the lookup threw "unknown
 * optimizer CERES:LEVENBERG_MARQUARDT (not registered in optimizer_caps)".) */
TEST_CASE("OptimizerCaps: the Ceres least-squares solvers are refused, naming "
          "the solvers HPO runs", "[HPO][caps][refused]")
{
    for (const char* algo :
         {"LEVENBERG_MARQUARDT", "DOGLEG", "SUBSPACE_DOGLEG"}) {
        INFO(algo);
        const char* reason = backend_refusal("CERES", algo);
        REQUIRE(reason != nullptr);
        CHECK_THAT(std::string(reason),
                   Catch::Contains("sum of squared residuals") &&
                   Catch::Contains("one scalar loss") &&
                   Catch::Contains("The recommended replacements are "
                                   "CERES:STEEPEST_DESCENT, Ceres' "
                                   "line-search solver of a scalar loss that "
                                   "tadah hpo runs, and the bound-constrained "
                                   "gradient searches NLOPT:LD_LBFGS and "
                                   "DLIB:BFGS"));
        CHECK_THROWS_WITH(optimizer_caps("CERES", algo),
                          std::string("CERES:") + algo + " is refused: " +
                              reason);
        CHECK(find_optimizer_caps("CERES", algo) == nullptr);
        CHECK_THAT(caps_help("CERES", algo),
                   Catch::StartsWith(std::string("Options for CERES:") + algo +
                                     "\n   * REFUSED: ") &&
                   Catch::Contains("The recommended replacements are "
                                   "CERES:STEEPEST_DESCENT"));
    }
    CHECK(backend_refusal("CERES", "STEEPEST_DESCENT") == nullptr);
    CHECK(backend_refusal("NLOPT", "DOGLEG") == nullptr);
}

/* Pins: a LIB:ALGO that is neither registered nor refused is named with what
 * is wrong with it, and with where the optimisers are listed: "unknown
 * optimiser NLOPT:NOT_AN_ALGO: LIB NLOPT has no ALGO NOT_AN_ALGO; `tadah
 * explain ALGO` lists the optimisers" for a library HPO knows, "unknown
 * optimiser FOO:BFGS: there is no LIB FOO (LIB is ASA, CERES, DLIB, NLOPT or
 * TADAH); ..." for one it does not (LIB is read as written: nlopt is not
 * NLOPT). The parser reports it at the line that closes the block
 * (test_startup_validation_strategy.cpp).
 * Before the fix: "unknown optimizer NLOPT:NOT_AN_ALGO (not registered in
 * optimizer_caps)" for both kinds. */
TEST_CASE("OptimizerCaps: an unknown optimiser is named with what is wrong",
          "[HPO][caps][unknown]")
{
    CHECK_THROWS_WITH(optimizer_caps("NLOPT", "NOT_AN_ALGO"),
                      "unknown optimiser NLOPT:NOT_AN_ALGO: LIB NLOPT has no "
                      "ALGO NOT_AN_ALGO; `tadah explain ALGO` lists the "
                      "optimisers");
    CHECK_THROWS_WITH(optimizer_caps("CERES", "BFGS"),
                      "unknown optimiser CERES:BFGS: LIB CERES has no ALGO "
                      "BFGS; `tadah explain ALGO` lists the optimisers");
    CHECK_THROWS_WITH(optimizer_caps("FOO", "BFGS"),
                      "unknown optimiser FOO:BFGS: there is no LIB FOO (LIB is "
                      "ASA, CERES, DLIB, NLOPT or TADAH); `tadah explain ALGO` "
                      "lists the optimisers");
    CHECK_THROWS_WITH(optimizer_caps("nlopt", "LD_LBFGS"),
                      Catch::StartsWith("unknown optimiser nlopt:LD_LBFGS: "
                                        "there is no LIB nlopt (LIB is ASA, "));
}

/* Pins: the ALGO help of CERES:STEEPEST_DESCENT (OptimizerSpec::print_caps,
 * `tadah explain CERES:STEEPEST_DESCENT`) names the solver, the stops that
 * end a run (from the caps registry: MAXEVAL, FTOL_REL, XTOL_REL), the PARAM
 * keys the driver reads -- tol_gradinf, max_iter, max_ls_iter, the finite
 * differences' fd_step / fd_kernel / diff_method, neither max_lbfgs_rank nor
 * nlcg_type --, that STEP is rejected, and that it ignores the OPTIM box and
 * that a point it proposes outside the box is not evaluated; the help of
 * CERES:LBFGS, WOLFE and NLCG says REFUSED with their reason.
 * Before the fix: the four were described (LBFGS with max_lbfgs_rank, NLCG
 * with nlcg_type), and the WARNING said a point outside the box is
 * evaluated there. */
TEST_CASE("OptimizerCaps: the ALGO help describes the CERES drivers",
          "[HPO][caps][help]")
{
    for (const char* algo : {"LBFGS", "WOLFE", "NLCG"}) {
        INFO(algo);
        CHECK_THAT(caps_help("CERES", algo),
                   Catch::StartsWith(std::string("Options for CERES:") + algo +
                                     "\n   * REFUSED: Ceres' "
                                     "line-search solver has no bounds"));
    }
    for (const char* algo : {"STEEPEST_DESCENT"}) {
        INFO(algo);
        const std::string help = caps_help("CERES", algo);
        INFO(help);
        const std::string a = algo;
        CHECK_THAT(help,
                   Catch::StartsWith("Options for CERES:" + a + "\n") &&
                   Catch::Contains("Ceres' line-search solver: steepest "
                                   "descent, Armijo line search") &&
                   Catch::Contains("   * Stops that end a run: MAXEVAL, "
                                   "FTOL_REL, XTOL_REL\n") &&
                   Catch::Contains("PARAM tol_gradinf") &&
                   Catch::Contains("PARAM max_iter") &&
                   Catch::Contains("PARAM max_ls_iter") &&
                   Catch::Contains("PARAM fd_step") &&
                   Catch::Contains("STEP is rejected: Ceres chooses the "
                                   "initial line-search step itself") &&
                   Catch::Contains("WARNING: ignores the OPTIM box") &&
                   Catch::Contains("Such a point is not evaluated") &&
                   !Catch::Contains("(no info)"));
        CHECK(help.find("PARAM max_lbfgs_rank") == std::string::npos);
        CHECK(help.find("PARAM nlcg_type") == std::string::npos);
    }
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
               Catch::Contains("The recommended replacement is "
                               "NLOPT:LD_LBFGS"));
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
               Catch::Contains("The recommended replacement is "
                               "NLOPT:LD_LBFGS"));
}
