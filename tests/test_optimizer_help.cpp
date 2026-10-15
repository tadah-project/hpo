// test_optimizer_help.cpp
//
// `tadah explain ALGO` and `tadah explain <LIB>:<ALGO>`: the optimisers of
// a `tadah hpo` target, from the caps registry (explain_optimiser,
// optimizer_help.h). tadah sets it as MLIP's explain hook.

#include "catch2/catch.hpp"

#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/optimizer_help.h>
#include <tadah/hpo/optimizer_spec.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace tadah::hpo;

namespace {

/* What explain_optimiser() writes for `name`; `known` says what it
 * returned. */
std::string explained(const std::string& name, bool* known = nullptr)
{
    std::ostringstream out;
    const bool k = explain_optimiser(name, out);
    if (known) *known = k;
    return out.str();
}

std::string caps_help(const std::string& lib, const std::string& algo)
{
    std::ostringstream out;
    OptimizerSpec::print_caps(out, lib, algo);
    return out.str();
}

}  // namespace

/* Pins: `tadah explain ALGO` (and LIB, in any letter case) prints the table
 * of the optimisers a target's LIB and ALGO can name: a "LIB <lib>" section
 * per library, sorted, with one row per registered backend, sorted, that
 * says whether it is global or local, gradient-based or derivative-free,
 * runs a local search (MLSL, AUGLAG), needs a stop or ignores the OPTIM box;
 * then the refused backends (NLOPT:LN_NEWUOA, NLOPT:LN_NEWUOA_BOUND,
 * NLOPT:NLOPT_LD_LBFGS_NOCEDAL and the Ceres least-squares solvers), and how
 * to see one: `tadah explain <LIB>:<ALGO>`. Every registered backend has its
 * row, under its library.
 * Before the fix: none compiles (explain_optimiser does not exist):
 * OptimizerSpec::print_caps was called only from HPO's tests, and `tadah
 * explain ALGO` said "tadah explain: no command or configuration key
 * 'ALGO'". */
TEST_CASE("explain_optimiser: ALGO and LIB print the table of the "
          "optimisers", "[HPO][explain]")
{
    bool known = false;
    const std::string table = explained("ALGO", &known);
    INFO(table);
    CHECK(known);
    CHECK_THAT(table,
               Catch::StartsWith("ALGO, with LIB its library, names the "
                                 "optimiser of a `tadah hpo` target") &&
               Catch::Contains("\nLIB ASA\n") &&
               Catch::Contains("\nLIB CERES\n") &&
               Catch::Contains("\nLIB DLIB\n") &&
               Catch::Contains("\nLIB NLOPT\n") &&
               Catch::Contains("\nLIB TADAH\n") &&
               Catch::Contains("Refused (`tadah explain <LIB>:<ALGO>` says "
                               "why and what to use instead):") &&
               Catch::Contains("CERES:DOGLEG") &&
               Catch::Contains("CERES:LEVENBERG_MARQUARDT") &&
               Catch::Contains("CERES:SUBSPACE_DOGLEG") &&
               Catch::Contains("NLOPT:LN_NEWUOA,") &&
               Catch::Contains("NLOPT:LN_NEWUOA_BOUND") &&
               Catch::Contains("NLOPT:NLOPT_LD_LBFGS_NOCEDAL") &&
               Catch::Contains("`tadah explain <LIB>:<ALGO>`, or `tadah "
                               "explain <ALGO>`, prints the stops, PARAM "
                               "keys and STEP use of one"));
    /* the sections in order, and every registered backend in its own */
    const auto at = [&](const std::string& s) { return table.find(s); };
    CHECK(at("\nLIB ASA\n") < at("\nLIB CERES\n"));
    CHECK(at("\nLIB CERES\n") < at("\nLIB DLIB\n"));
    CHECK(at("\nLIB DLIB\n") < at("\nLIB NLOPT\n"));
    CHECK(at("\nLIB NLOPT\n") < at("\nLIB TADAH\n"));
    CHECK(at("\nLIB TADAH\n") < at("Refused"));
    for (const auto* c : filter_optimizer_caps(
             [](const OptimizerCaps&) { return true; })) {
        INFO(c->lib << ":" << c->algo);
        const auto section = at("\nLIB " + c->lib + "\n");
        const auto row = at("\n  " + c->algo + " ");
        REQUIRE(section != std::string::npos);
        CHECK(row != std::string::npos);
        CHECK(row > section);
    }
    CHECK_THAT(table, Catch::Contains("\n  LD_LBFGS ") &&
                      Catch::Contains("local, gradient-based"));
    /* CERES:STEEPEST_DESCENT (the row checked was CERES:LBFGS, refused
     * since: listed under Refused; before the fix the case passes as well,
     * a pin) */
    CHECK_THAT(table, Catch::Contains("\n  STEEPEST_DESCENT ") &&
                      Catch::Contains("ignores the OPTIM box"));
    CHECK_THAT(table, Catch::Contains("needs a stop"));
    CHECK_THAT(table, Catch::Contains("runs a local search"));
    CHECK(table.find("\n  LN_NEWUOA ") == std::string::npos);
    CHECK(explained("algo") == table);
    CHECK(explained("LIB") == table);
    CHECK(explained("lib") == table);
}

/* Pins: `tadah explain <LIB>:<ALGO>`, or `tadah explain <ALGO>` alone, in
 * any letter case, prints that optimiser's ALGO help
 * (OptimizerSpec::print_caps): NLOPT:LD_LBFGS for "LD_LBFGS", "ld_lbfgs"
 * and "NLOPT:LD_LBFGS", CERES:STEEPEST_DESCENT, ASA:ASA for "ASA" (an ALGO
 * name before a LIB name), the REFUSED text of NLOPT:LN_NEWUOA for "NLOPT:LN_NEWUOA" and
 * "LN_NEWUOA"; a LIB name alone prints its section of the table ("NLOPT":
 * its rows, no other library's). A LIB:ALGO HPO does not have is an error
 * naming what is wrong, the startup check's text ("tadah explain: unknown
 * optimiser NLOPT:LD_FOO: LIB NLOPT has no ALGO LD_FOO; `tadah explain
 * ALGO` lists the optimisers", "... there is no LIB FOO (...)"); another
 * name is not an optimiser (false, nothing written).
 * Before the fix: none compiles (explain_optimiser does not exist). */
TEST_CASE("explain_optimiser: one optimiser, a library, and the names it "
          "does not know", "[HPO][explain]")
{
    const std::string lbfgs = caps_help("NLOPT", "LD_LBFGS");
    REQUIRE_THAT(lbfgs, Catch::StartsWith("Options for NLOPT:LD_LBFGS\n"));
    for (const char* name : {"LD_LBFGS", "ld_lbfgs", "NLOPT:LD_LBFGS",
                             "nlopt:ld_lbfgs"}) {
        INFO(name);
        bool known = false;
        CHECK(explained(name, &known) == lbfgs);
        CHECK(known);
    }
    /* CERES:STEEPEST_DESCENT (it was CERES:LBFGS, refused since; before
     * the fix the case passes as well, a pin) */
    CHECK(explained("CERES:STEEPEST_DESCENT") ==
          caps_help("CERES", "STEEPEST_DESCENT"));
    CHECK(explained("ASA") == caps_help("ASA", "ASA"));
    const std::string newuoa = explained("NLOPT:LN_NEWUOA");
    CHECK_THAT(newuoa, Catch::StartsWith("Options for NLOPT:LN_NEWUOA\n"
                                         "   * REFUSED: "));
    CHECK(explained("LN_NEWUOA") == newuoa);

    const std::string nlopt = explained("NLOPT");
    CHECK_THAT(nlopt, Catch::StartsWith("LIB NLOPT\n") &&
                      Catch::Contains("\n  LD_LBFGS ") &&
                      !Catch::Contains("LIB DLIB"));

    CHECK_THROWS_WITH(explained("NLOPT:LD_FOO"),
                      "tadah explain: unknown optimiser NLOPT:LD_FOO: LIB "
                      "NLOPT has no ALGO LD_FOO; `tadah explain ALGO` lists "
                      "the optimisers");
    CHECK_THROWS_WITH(explained("FOO:BAR"),
                      Catch::StartsWith("tadah explain: unknown optimiser "
                                        "FOO:BAR: there is no LIB FOO (LIB "
                                        "is ASA, CERES, DLIB, NLOPT or "
                                        "TADAH)"));
    bool known = true;
    CHECK(explained("bogus", &known).empty());
    CHECK_FALSE(known);
}

/* Pins: the ALGO help of the optimisers that run a local search (NLopt's
 * six MLSL and six AUGLAG forms, OptimizerCaps::accepts_inner) says what
 * their INNER block takes, in one entry wrapped at 76 columns: "INNER ...
 * ENDINNER: the local search it runs, from the points it samples in the
 * OPTIM box" (MLSL) or "from its own starting point" (AUGLAG), ": LIB NLOPT
 * and a local ALGO; a global one (`tadah explain ALGO` marks each) is
 * refused, and so is one that needs an INNER block of its own.", then
 * "Without one NLopt runs its default LN_COBYLA." (LD_MMA for the GD_ and
 * LD_ forms) or, for AUGLAG, AUGLAG_EQ, G_MLSL and G_MLSL_LDS, "The block
 * is required: NLopt has no default local optimiser for it." (owner
 * decision, MK, 2026-10-15: no nested global search). No other optimiser's
 * help has the entry.
 * Before the fix: no ALGO help says what an INNER block takes (the entry is
 * missing from all twelve): the checks of the twelve fail. */
TEST_CASE("explain_optimiser: the ALGO help of MLSL and AUGLAG says that "
          "their INNER block takes a local optimiser",
          "[HPO][explain][inner]")
{
    const std::string head = "   * INNER ... ENDINNER: ";
    /* The entry, its lines joined: from its head to the next entry. */
    const auto entry = [&](const std::string& help) {
        const auto at = help.find(head);
        if (at == std::string::npos) return std::string();
        const auto end = help.find("\n   * ", at + 1);
        std::string e = help.substr(at, end == std::string::npos
                                            ? std::string::npos
                                            : end - at);
        std::istringstream lines(e);
        std::string joined, line;
        while (std::getline(lines, line)) {
            INFO("line: '" << line << "'");
            CHECK(line.size() <= 76u);
            const auto p = line.find_first_not_of(' ');
            const std::string text =
                p == std::string::npos ? std::string() : line.substr(p);
            joined += (joined.empty() ? "" : " ") + text;
        }
        return joined;
    };
    const std::string takes =
        ": LIB NLOPT and a local ALGO; a global one (`tadah explain ALGO` "
        "marks each) is refused, and so is one that needs an INNER block of "
        "its own. ";
    const std::string required =
        "The block is required: NLopt has no default local optimiser for it.";
    std::size_t with_inner = 0;
    for (const auto* c : filter_optimizer_caps(
             [](const OptimizerCaps&) { return true; })) {
        INFO(c->lib << ":" << c->algo);
        const std::string help = caps_help(c->lib, c->algo);
        if (!c->accepts_inner) {
            CHECK(help.find("INNER ... ENDINNER") == std::string::npos);
            continue;
        }
        ++with_inner;
        const bool mlsl = c->algo.find("MLSL") != std::string::npos;
        const bool needs = c->algo == "AUGLAG" || c->algo == "AUGLAG_EQ" ||
                           c->algo == "G_MLSL" || c->algo == "G_MLSL_LDS";
        const bool derivative = c->algo.rfind("GD_", 0) == 0 ||
                                c->algo.rfind("LD_", 0) == 0;
        CHECK(entry(help) ==
              "* INNER ... ENDINNER: the local search it runs, from " +
                  std::string(mlsl ? "the points it samples in the OPTIM box"
                                   : "its own starting point") +
                  takes +
                  (needs ? required
                         : std::string("Without one NLopt runs its default ") +
                               (derivative ? "LD_MMA" : "LN_COBYLA") + "."));
    }
    CHECK(with_inner == 12u);
    CHECK_THAT(explained("NLOPT:G_MLSL_LDS"),
               Catch::Contains("\n   * INNER ... ENDINNER: the local search "
                               "it runs, from the points it\n     samples in "
                               "the OPTIM box: LIB NLOPT"));
}
