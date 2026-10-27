// test_drawn_seeds.cpp
//
// A run draws random numbers without a SEED in two places this file
// covers: the METHOD lines of an INIT block (RANDOM, LHS, SOBOL) whose block
// has no SEED, and an NLopt algorithm that draws random numbers whose block
// has no SEED (and no SEED of its INIT block, which NLopt takes in its
// place). Each draws a 64-bit seed once for the run, keeps it as the
// block's SEED (InitStrategySpec::seed, OptimizerSpec::seed) and says so on
// standard output and in the journal, with the SEED line that draws the
// same numbers again, as the RESTART block does for its points. A chain of
// CONFIG and WARM draws nothing and takes no seed.
#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <tuple>
#include <unistd.h>
#include <vector>

namespace {

namespace fs = std::filesystem;
using tadah::core::Context;
using tadah::hpo::HPOLog;
using tadah::hpo::HPOTargets;

std::string unique_name(const std::string& stem)
{
    static unsigned counter = 0;
    return stem + std::to_string(::getpid()) + "_" +
           std::to_string(counter++) + "_" +
           std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count());
}

std::string slurp(const fs::path& p)
{
    std::ifstream in(p);
    std::ostringstream os;
    os << in.rdbuf();
    return os.str();
}

/* What reading `target` gives: the parsed targets, standard output and the
 * journal, with the OPTIMISER section a run writes after reading it. */
struct Read {
    std::unique_ptr<HPOTargets> trg;
    std::string out, journal;
};

Read read_target(const std::string& target, Context ctx)
{
    const fs::path file =
        fs::temp_directory_path() / (unique_name("tadah_drawn_") + ".hpo");
    const fs::path journal =
        fs::temp_directory_path() / (unique_name("tadah_drawn_") + ".log");
    {
        std::ofstream os(file);
        REQUIRE(os.good());
        os << target;
    }
    Read r;
    auto& log = HPOLog::instance();
    if (log.is_open()) log.finish("reopen-for-test");
    log.reset_for_test();
    log.open(journal.string(), "test", "test_cwd", 0);
    std::ostringstream out;
    std::streambuf* old = std::cout.rdbuf(out.rdbuf());
    try {
        r.trg = std::make_unique<HPOTargets>(ctx, file.string());
    } catch (...) {
        std::cout.rdbuf(old);
        log.finish("test-end");
        log.reset_for_test();
        std::error_code ec;
        fs::remove(file, ec);
        fs::remove(journal, ec);
        throw;
    }
    std::cout.rdbuf(old);
    r.out = out.str();
    log.write_optimizer_section(r.trg->spec, {});
    log.finish("test-end");
    log.reset_for_test();
    r.journal = slurp(journal);
    std::error_code ec;
    fs::remove(file, ec);
    fs::remove(journal, ec);
    return r;
}

Context make_ctx()
{
    Context c;
    c.add("CGRID2B", std::vector<double>{1.0, 2.0});
    return c;
}

/* Lines 1-2 the search space, OPTIMIZER on line 3, the INIT block on line
 * 4 with `init` inside it; the STRATEGY block runs NLOPT:`algo`. */
std::string target(const std::string& init, const std::string& algo,
                   const std::string& strategy = "")
{
    return "OPTIM CGRID2B (1) 0.5 1.5\nOPTIM CGRID2B (2) 1.6 2.5\n"
           "OPTIMIZER\nINIT\n" + init + "ENDINIT\n"
           "STRATEGY\nLIB NLOPT\nALGO " + algo + "\nMAXEVAL 5\n" + strategy +
           "ENDSTRATEGY\nENDOPTIMIZER\n";
}

std::size_t count_of(const std::string& text, const std::string& needle)
{
    std::size_t n = 0;
    for (auto p = text.find(needle); p != std::string::npos;
         p = text.find(needle, p + needle.size()))
        ++n;
    return n;
}

}  // namespace

/* Pins: an INIT block without SEED whose chain draws its points (METHOD
 * SOBOL 4, METHOD RANDOM 3) draws a seed S once for the run and keeps it as
 * the block's SEED (InitStrategySpec::seed); standard output ("[HPO] ") and
 * the journal say "the INIT block on line 4 has no SEED, so its starting
 * points are drawn with a seed drawn once for the run, S (SEED S in the
 * INIT block draws the same points again)"; the same file with "SEED S" in
 * its INIT block gives the same points. Two reads draw two seeds. The
 * journal's OPTIMISER section says "SEED              : S (drawn for the
 * run)". For an NLopt algorithm that draws random numbers and has no SEED
 * of its own (GN_CRS2_LM), which takes the INIT block's SEED, the line adds
 * "; NLOPT:GN_CRS2_LM, without a SEED of its own, takes its random numbers
 * from the same seed" and the STRATEGY block draws no seed of its own.
 * Before the fix: the chain was drawn with std::random_device and no seed
 * was kept or named (the seed is unset, the line is missing). */
TEST_CASE("drawn seeds: an INIT block without SEED draws a seed once for the "
          "run and names it", "[HPO][init_strategy][seed][drawn_seed]")
{
    Context ctx = make_ctx();
    const std::regex line(
        R"(the INIT block on line 4 has no SEED, so its starting points are )"
        R"(drawn with a seed drawn once for the run, (\d+) \(SEED (\d+) in )"
        R"(the INIT block draws the same points again\))");
    for (const std::string& method :
         {std::string("METHOD SOBOL 4\n"), std::string("METHOD RANDOM 3\n")}) {
        INFO(method);
        const Read a = read_target(target(method, "LN_BOBYQA"), ctx);
        INFO("stdout:\n" << a.out << "\njournal:\n" << a.journal);
        REQUIRE(a.trg->spec.init_strategy_spec.seed.has_value());
        const std::uint64_t s = *a.trg->spec.init_strategy_spec.seed;
        std::smatch m;
        REQUIRE(std::regex_search(a.journal, m, line));
        CHECK(m[1] == std::to_string(s));
        CHECK(m[2] == std::to_string(s));
        CHECK(std::regex_search(a.out, line));
        CHECK(count_of(a.journal, "SEED              : " + std::to_string(s) +
                                      " (drawn for the run)") == 1);
        /* SEED S draws the same points again. */
        const Read b = read_target(
            target(method + "SEED " + std::to_string(s) + "\n", "LN_BOBYQA"),
            ctx);
        CHECK(b.trg->spec.optim_init == a.trg->spec.optim_init);
        CHECK(count_of(b.journal, "drawn once for the run") == 0);
        /* Another read draws another seed. */
        const Read c = read_target(target(method, "LN_BOBYQA"), ctx);
        REQUIRE(c.trg->spec.init_strategy_spec.seed.has_value());
        CHECK(*c.trg->spec.init_strategy_spec.seed != s);
    }
    {
        const Read a = read_target(target("METHOD LHS 3\n", "GN_CRS2_LM"), ctx);
        INFO("journal:\n" << a.journal);
        REQUIRE(a.trg->spec.init_strategy_spec.seed.has_value());
        const std::string s = std::to_string(*a.trg->spec.init_strategy_spec.seed);
        CHECK(count_of(a.journal,
                       "(SEED " + s + " in the INIT block draws the same points "
                       "again; NLOPT:GN_CRS2_LM, without a SEED of its own, "
                       "takes its random numbers from the same seed)") == 1);
        CHECK_FALSE(a.trg->spec.seed.has_value());
        CHECK(a.trg->spec.nlopt_seed_from_init());
    }
}

/* Pins: an NLopt algorithm that draws random numbers (GN_CRS2_LM), in a
 * block without SEED and without a SEED of its INIT block (an INIT block of
 * METHOD CONFIG, which draws nothing), draws a seed S once for the run and
 * keeps it as the block's SEED (OptimizerSpec::seed); standard output and
 * the journal say "the STRATEGY block has no SEED, and NLOPT:GN_CRS2_LM
 * draws random numbers, so it draws them with a seed drawn once for the
 * run, S (SEED S in the STRATEGY block draws the same numbers again)", and
 * the OPTIMISER section's "Random seed" is "S (drawn for the run)". The
 * same for the EXPLORE block of a PRESTAGE ("the EXPLORE block has no SEED,
 * and NLOPT:GN_ISRES draws ..."), and the lines name the block that holds
 * the optimiser's ALGO line: in a file without a STRATEGY block, whose
 * OPTIMIZER block holds LIB and ALGO, "the OPTIMIZER block has no SEED, ...
 * (SEED S in the OPTIMIZER block draws the same numbers again)". No seed is
 * drawn for an algorithm that
 * draws none (LN_BOBYQA), for a block with a SEED, or for a chain of
 * CONFIG and WARM points (no seed of the INIT block, no line).
 * Before the fix: NLopt took its own seed from the clock and the run named
 * none (OptimizerSpec::seed unset, no line); then a file without a STRATEGY
 * block was told "the STRATEGY block has no SEED ... (SEED S in the
 * STRATEGY block ...)", a block it does not have. */
TEST_CASE("drawn seeds: an NLopt algorithm that draws random numbers without "
          "a SEED draws a seed once for the run and names it",
          "[HPO][nlopt][seed][drawn_seed]")
{
    Context ctx = make_ctx();
    {
        const Read a = read_target(target("METHOD CONFIG\n", "GN_CRS2_LM"), ctx);
        INFO("stdout:\n" << a.out << "\njournal:\n" << a.journal);
        CHECK_FALSE(a.trg->spec.init_strategy_spec.seed.has_value());
        REQUIRE(a.trg->spec.seed.has_value());
        const std::string s = std::to_string(*a.trg->spec.seed);
        const std::string w =
            "the STRATEGY block has no SEED, and NLOPT:GN_CRS2_LM draws random "
            "numbers, so it draws them with a seed drawn once for the run, " +
            s + " (SEED " + s + " in the STRATEGY block draws the same numbers "
            "again)";
        CHECK(count_of(a.journal, w) == 1);
        CHECK(count_of(a.out, "[HPO] " + w + "\n") == 1);
        CHECK(count_of(a.journal, s + " (drawn for the run)") == 1);
        CHECK(count_of(a.journal, "INIT block on line 4 has no SEED") == 0);
    }
    {
        const std::string explore =
            "OPTIM CGRID2B (1) 0.5 1.5\nOPTIM CGRID2B (2) 1.6 2.5\n"
            "OPTIMIZER\nSTRATEGY\nTYPE PRESTAGE\nLIB NLOPT\nALGO LN_BOBYQA\n"
            "MAXEVAL 5\nEXPLORE\nLIB NLOPT\nALGO GN_ISRES\nMAXEVAL 5\n"
            "ENDEXPLORE\nENDSTRATEGY\nENDOPTIMIZER\n";
        const Read a = read_target(explore, ctx);
        INFO("journal:\n" << a.journal);
        REQUIRE(a.trg->spec.prestage);
        REQUIRE(a.trg->spec.prestage->seed.has_value());
        CHECK_FALSE(a.trg->spec.seed.has_value());
        const std::string s = std::to_string(*a.trg->spec.prestage->seed);
        CHECK(count_of(a.journal,
                       "the EXPLORE block has no SEED, and NLOPT:GN_ISRES draws "
                       "random numbers, so it draws them with a seed drawn once "
                       "for the run, " + s + " (SEED " + s + " in the EXPLORE "
                       "block draws the same numbers again)") == 1);
    }
    {
        /* No STRATEGY block: LIB and ALGO are in the OPTIMIZER block. */
        const std::string plain =
            "OPTIM CGRID2B (1) 0.5 1.5\nOPTIM CGRID2B (2) 1.6 2.5\n"
            "OPTIMIZER\nLIB NLOPT\nALGO GN_CRS2_LM\nMAXEVAL 5\n"
            "ENDOPTIMIZER\n";
        const Read a = read_target(plain, ctx);
        INFO("stdout:\n" << a.out << "\njournal:\n" << a.journal);
        REQUIRE(a.trg->spec.seed.has_value());
        const std::string s = std::to_string(*a.trg->spec.seed);
        const std::string w =
            "the OPTIMIZER block has no SEED, and NLOPT:GN_CRS2_LM draws "
            "random numbers, so it draws them with a seed drawn once for the "
            "run, " + s + " (SEED " + s + " in the OPTIMIZER block draws the "
            "same numbers again)";
        CHECK(count_of(a.journal, w) == 1);
        CHECK(count_of(a.out, "[HPO] " + w + "\n") == 1);
        CHECK(count_of(a.out, "STRATEGY block") == 0);
    }
    /* No seed drawn. */
    const fs::path seed_file =
        fs::temp_directory_path() / (unique_name("tadah_drawn_seed_") + ".tadah");
    {
        std::ofstream os(seed_file);
        os << "CGRID2B 1.25 2.25\n";
    }
    const std::string warm = "METHOD WARM " + seed_file.string() + "\n";
    for (const auto& [init, algo, strategy] :
         std::vector<std::tuple<std::string, std::string, std::string>>{
             {"METHOD CONFIG\n", "LN_BOBYQA", ""},
             {"METHOD CONFIG\n", "GN_CRS2_LM", "SEED 9\n"},
             {"METHOD CONFIG\n", "LN_SBPLX", ""},
             {warm, "LN_BOBYQA", ""},
             {"METHOD CONFIG\n" + warm, "LN_SBPLX", ""}}) {
        INFO(init << algo << " " << strategy);
        const Read a = read_target(target(init, algo, strategy), ctx);
        INFO("journal:\n" << a.journal);
        CHECK(count_of(a.journal, "drawn once for the run") == 0);
        CHECK(count_of(a.out, "drawn once for the run") == 0);
        CHECK_FALSE(a.trg->spec.init_strategy_spec.seed.has_value());
        if (strategy.empty()) CHECK_FALSE(a.trg->spec.seed.has_value());
    }
    std::error_code ec;
    fs::remove(seed_file, ec);
}
