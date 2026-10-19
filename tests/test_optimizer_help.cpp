// test_optimizer_help.cpp
//
// `tadah explain ALGO` and `tadah explain <LIB>:<ALGO>`: the optimisers of
// a `tadah hpo` target, from the caps registry (explain_optimiser,
// optimizer_help.h). tadah sets it as MLIP's explain hook.

#include "catch2/catch.hpp"

#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/optimizer_help.h>
#include <tadah/hpo/optimizer_spec.h>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
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

std::string caps_help(const std::string& lib, const std::string& algo,
                      BuildNotes notes)
{
    std::ostringstream out;
    OptimizerSpec::print_caps(out, lib, algo, notes);
    return out.str();
}

/* Every LIB:ALGO, registered or refused, sorted. */
std::vector<std::pair<std::string, std::string>> every_backend()
{
    std::vector<std::pair<std::string, std::string>> names;
    for (const auto* c : filter_optimizer_caps(
             [](const OptimizerCaps&) { return true; }))
        names.emplace_back(c->lib, c->algo);
    for (const std::string& name : refused_backend_names()) {
        const auto colon = name.find(':');
        names.emplace_back(name.substr(0, colon), name.substr(colon + 1));
    }
    std::sort(names.begin(), names.end());
    return names;
}

/* What makes `text` other than text for a user, one entry per fault: the
 * module name HPO for the command (tadah hpo); a source file, patch, C++
 * name or function; a word of development jargon or history; a word that
 * addresses the reader (you, your, we, our, us, let's); American spelling
 * in prose
 * (key names, written in capitals or with an underscore, are as the
 * program reads them). */
std::vector<std::string> user_text_faults(const std::string& text)
{
    std::vector<std::string> faults;
    static const std::regex hpo("(^|[^A-Za-z0-9_])HPO([^A-Za-z0-9_]|$)");
    if (std::regex_search(text, hpo))
        faults.push_back("HPO, not tadah hpo");
    static const std::regex file("[A-Za-z0-9_]+\\.(c|cc|cpp|h|hpp|py)"
                                 "([^A-Za-z0-9_]|$)");
    std::smatch m;
    if (std::regex_search(text, m, file))
        faults.push_back("a source file: " + m.str());
    for (const char* bad : {"::", "TADAH-PATCH", "external/", "vendored",
                            "random_device", "knob", "subblock", "directive",
                            "stanza", "legacy", "previously", "no longer",
                            "pre-1."})
        if (text.find(bad) != std::string::npos)
            faults.push_back(std::string("'") + bad + "'");
    static const std::regex reader(
        "(^|[^A-Za-z])([Yy]ou|[Yy]our|[Yy]ours|[Yy]ourself|[Ww]e|[Oo]ur|"
        "[Uu]s|[Ll]et's)"
        "([^A-Za-z]|$)");
    if (std::regex_search(text, m, reader))
        faults.push_back("addresses the reader: " + m.str(2));
    /* -ize and -yze, except the words of size */
    static const std::regex word("[A-Za-z_]+");
    static const std::regex american(
        "(iz|yz)(e|es|ed|ing|er|ers|ation|ations)$|^behavior|^color|"
        "^center|^journaled$|^labeled$|^modeled$");
    static const std::set<std::string> size = {
        "size", "sizes", "sized", "sizing", "resize", "resizes", "resized",
        "resizing"};
    for (auto it = std::sregex_iterator(text.begin(), text.end(), word);
         it != std::sregex_iterator(); ++it) {
        const std::string w = it->str();
        if (w.find('_') != std::string::npos) continue;  // a key name
        bool capitals = true;
        for (const char ch : w)
            if (std::islower(static_cast<unsigned char>(ch))) capitals = false;
        if (capitals) continue;  // a key name: OPTIMIZER
        std::string lower = w;
        for (char& ch : lower)
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (size.count(lower)) continue;
        if (std::regex_search(lower, american))
            faults.push_back("American spelling: " + w);
    }
    return faults;
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
 * is required: NLopt has no default local optimiser for it." (no nested
 * global search). No other optimiser's help has the entry.
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

/* Pins: BuildNotes::Omit leaves out every line of the ALGO help that says
 * what this build lacks, so the text written for the documentation
 * (blueprint/optimizers_gen.toml) is the same in every build of one NLopt
 * version. For every LIB:ALGO, registered or refused: the Omit text has no
 * line about "this build"; print_caps without BuildNotes prints the
 * Include text, which `tadah explain` prints. In a build with Ceres the
 * two texts are the same for every LIB:ALGO. In a build without Ceres
 * (TADAH_ENABLE_CERES OFF) the Include text of CERES:STEEPEST_DESCENT has
 * the line "this build has no Ceres (TADAH_ENABLE_CERES OFF): a run is
 * refused" and its Omit text is that text without the line; every other
 * LIB:ALGO's two texts are the same.
 * Fails in a build without Ceres when print_ceres_caps prints the line
 * whatever BuildNotes says (the Omit check removed): the Omit text of
 * CERES:STEEPEST_DESCENT then has it. In a build with Ceres the line is
 * not compiled, and this case checks only that no other line depends on
 * the build. */
TEST_CASE("print_caps: BuildNotes::Omit leaves out the lines that depend on "
          "the build", "[HPO][explain][build_notes]")
{
    const std::string ceres_line =
        "   * this build has no Ceres (TADAH_ENABLE_CERES OFF): a run is "
        "refused\n";
    const auto names = every_backend();
    REQUIRE(names.size() > 50u);
    std::size_t with_line = 0;
    for (const auto& [lib, algo] : names) {
        INFO(lib << ":" << algo);
        const std::string include = caps_help(lib, algo, BuildNotes::Include);
        const std::string omit = caps_help(lib, algo, BuildNotes::Omit);
        CHECK(caps_help(lib, algo) == include);
        CHECK(omit.find("this build") == std::string::npos);
        const auto at = include.find(ceres_line);
        if (at == std::string::npos) {
            CHECK(omit == include);
            continue;
        }
        ++with_line;
        CHECK((lib == "CERES" && algo == "STEEPEST_DESCENT"));
        std::string without = include;
        without.erase(at, ceres_line.size());
        CHECK(omit == without);
    }
#ifdef TADAH_USE_CERES
    CHECK(with_line == 0u);
#else
    CHECK(with_line == 1u);
#endif
}

/* Pins: the ALGO help of every LIB:ALGO, registered or refused (the text
 * `tadah explain` prints, and the text written for the documentation), and
 * every text of the optimiser registry the documentation's copy holds (the
 * row of the table, the STEP use, the notes, the reason of a refusal) are
 * text for a user: they name the command, tadah hpo, never the module
 * HPO; no source file, patch, C++ name or function of Tadah! or of a
 * library (optimize.c, external/asa, TADAH-PATCH 7, nlopt::opt::set_param,
 * random_device); no development jargon ("knob", "Dead knobs") or history
 * ("pre-1.5.0 behaviour"); no word that addresses the reader, and a
 * refusal's reason names the recommended replacement of the optimiser
 * instead of telling the reader what to use ("Use NLOPT:LN_BOBYQA, ...");
 * British spelling in prose ("global optimiser", "equalisation", "written
 * to the journal"); key names as the program reads them.
 * Before the fix: fails on the help of ASA:ASA (external/asa, TADAH-PATCH
 * 7, "knob", "HPO host"), TADAH:ANNEAL (random_device, "Dead knobs"),
 * TADAH:ANNEAL_PT ("equalization", "pre-1.5.0"), the twenty global NLopt
 * algorithms ("global optimizer", nlopt::opt::set_param on every NLopt
 * one), the refusals of NLOPT:LN_NEWUOA (optimize.c), NLOPT:LN_NEWUOA_BOUND
 * and the Ceres least-squares solvers ("HPO"), and the notes of
 * NLOPT:GD_STOGO and the AUGLAG forms ("HPO"); every refusal on "Use". */
TEST_CASE("explain_optimiser: the ALGO help and the optimiser registry are "
          "text for a user", "[HPO][explain][user_text]")
{
    std::size_t checked = 0;
    const auto check = [&](const std::string& where, const char* text) {
        if (!text) return;
        INFO(where << ":\n" << text);
        const auto faults = user_text_faults(text);
        std::string all;
        for (const auto& f : faults) all += "\n  " + f;
        INFO("faults:" << all);
        CHECK(faults.empty());
        ++checked;
    };
    for (const auto& [lib, algo] : every_backend()) {
        const std::string name = lib + ":" + algo;
        check(name + " help", caps_help(lib, algo).c_str());
        check(name + " help (Omit)",
              caps_help(lib, algo, BuildNotes::Omit).c_str());
        if (const char* reason = backend_refusal(lib, algo)) {
            check(name + " reason", reason);
            const std::string r = reason;
            INFO(name << " reason: " << r);
            CHECK(r.rfind("Use ", 0) == std::string::npos);
            CHECK(r.find(". Use ") == std::string::npos);
            continue;
        }
        const OptimizerCaps& c = optimizer_caps(lib, algo);
        check(name + " row", optimiser_row_text(c).c_str());
        check(name + " STEP use", step_use_name(c.step_use));
        check(name + " stop_note", c.stop_note);
        check(name + " tolerance_note", c.tolerance_note);
        check(name + " untested_tolerance_note", c.untested_tolerance_note);
        check(name + " step_note", c.step_note);
        check(name + " start_note", c.start_note);
        check(name + " bounds_note", c.bounds_note);
        check(name + " default_inner", c.default_inner);
    }
    CHECK(checked > 400u);
    /* the check itself: each kind of fault is found */
    CHECK(user_text_faults("HPO runs it").size() == 1u);
    CHECK(user_text_faults("tadah hpo runs it; OPTIMIZER, HPO_X").empty());
    CHECK(user_text_faults("see asa_usr.c").size() == 1u);
    CHECK(user_text_faults("a global optimizer").size() == 1u);
    CHECK(user_text_faults("Equalized, journaled").size() == 2u);
    CHECK(user_text_faults("the block resizing keeps its size").empty());
    CHECK(user_text_faults("the primary knob").size() == 1u);
    CHECK(user_text_faults("Your run: set it if you like").size() == 1u);
    CHECK(user_text_faults("the weight and the outer search").empty());
}
