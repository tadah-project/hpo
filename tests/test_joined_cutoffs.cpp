// test_joined_cutoffs.cpp
//
// The cutoff rules of tadah hpo for a knot channel (D2_Knot5, a knot
// DM_REAM density: the cutoff is the largest knot, derived at startup and
// on every evaluation, and the one-time neighbour list covers the largest
// knot an OPTIM box reaches) and for a bump channel (D2_Blip, D2_Blip5, a
// bump DM_REAM density: under a hard cutoff every support ends at or before
// the cutoff, a seed's at startup and a candidate's per evaluation; under a
// smooth one a centre beyond the cutoff is warned about) hold for each
// descriptor of a joined line (D2_mJoin, DM_mJoin) with that descriptor's
// own cutoff, the value of RCUT2B at its place, as they hold for a line of
// one descriptor. The entries of a joined line are numbered as an OPTIM line
// numbers the values of the whole key.
#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/bump_support_gate.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/rcut_derivation.h>
#include <tadah/hpo/startup_validation.h>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
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

struct TargetFile {
    fs::path path = fs::temp_directory_path() /
                    (unique_name("tadah_joined_cutoffs_") + ".hpo");
    explicit TargetFile(const std::string& body)
    {
        std::ofstream os(path);
        REQUIRE(os.good());
        os << body;
    }
    ~TargetFile()
    {
        std::error_code ec;
        fs::remove(path, ec);
    }
};

const std::string kOptimizer =
    "OPTIMIZER\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 5\nENDOPTIMIZER\n"
    "PC_ERMSE 0 1\n";

/* TYPE2B D2_mJoin of D2_Knot5 (knots: entries 1-3 of CGRID2B, its cutoff
 * the first RCUT2B value) and D2_Blip (centres: entries 4-6, widths
 * SGRID2B, half-width 2, its cutoff the second RCUT2B value). */
Context join_ctx(const char* rcut, const char* rctype, const char* cgrid,
                 const char* sgrid)
{
    Context c;
    c.add("ATOM", "Ta");
    c.add("WATOM", 1.0);
    c.add("DBFILE", "dummy.db");
    c.add("INIT2B", "true");
    c.add("TYPE2B", "D2_mJoin D2_Knot5 3 3 Ta Ta D2_Blip 3 3 Ta Ta");
    c.add("RCTYPE2B", rctype);
    c.add("RCUT2B", rcut);
    c.add("CGRID2B", cgrid);
    c.add("KDEG2B", "5 5 5");
    c.add("SGRID2B", sgrid);
    c.add("INITMB", "false");
    c.add("VERBOSE", 0);
    c.add("MODEL", "M_BLR BF_Linear");
    return c;
}

/* validate_hpo_setup() on `ctx` (changed in place) and `target`: its
 * refusal ("" when it accepts), with standard error and the journal. */
struct Startup {
    std::string error, err, journal;
};

Startup startup(Context& ctx, const std::string& target)
{
    tadah::core::PeriodicTable::initialize();
    const TargetFile tf(target);
    const fs::path journal = fs::temp_directory_path() /
                             (unique_name("tadah_joined_cutoffs_") + ".log");
    Startup s;
    auto& log = HPOLog::instance();
    if (log.is_open()) log.finish("reopen-for-test");
    log.reset_for_test();
    log.open(journal.string(), "test", "test_cwd", 0);
    std::ostringstream err;
    std::streambuf* old = std::cerr.rdbuf(err.rdbuf());
    try {
        HPOTargets trg(ctx, tf.path.string());
        tadah::hpo::validate_hpo_setup(ctx, trg, true, true);
    } catch (const std::exception& e) {
        s.error = e.what();
    }
    std::cerr.rdbuf(old);
    s.err = err.str();
    log.finish("test-end");
    log.reset_for_test();
    std::ifstream in(journal);
    std::ostringstream os;
    os << in.rdbuf();
    s.journal = os.str();
    std::error_code ec;
    fs::remove(journal, ec);
    return s;
}

}  // namespace

/* Pins: at startup, the cutoff of a D2_Knot5 of a joined line is its
 * largest knot: RCUT2B "5.3 5" with knots 2.5 3.5 4.5 becomes "4.5 5" (the
 * D2_Blip's 5 is kept), with one WARNING naming the descriptor and the
 * entry, as an OPTIM line numbers it: "tadah hpo: TYPE2B descriptor 1
 * (D2_Knot5): RCUT2B (1) 5.3 is above the largest knot (4.5): the knot
 * basis vanishes at its largest knot, so the cutoff of a D2_Knot5
 * descriptor is 4.5, which the CGRID2B knot positions set."; on every
 * evaluation the same derivation follows a moved knot
 * (apply_knot_rcut_derivation: a largest knot of 4.8 gives RCUT2B "4.8 5");
 * the one-time neighbour list covers the largest knot an OPTIM box of the
 * knot descriptor reaches (nn_safe_rcutmax 6.2 for OPTIM CGRID2B (3) 4.0
 * 6.2).
 * Before the fix: a joined line has no knot channel for these rules
 * (derived_knot_rcut reads the line's first value, D2_mJoin): at startup
 * RCUT2B stays 5.3 and nothing is said; on an evaluation the knot's move
 * to 4.8 leaves RCUT2B 4.5; and nn_safe_rcutmax is RCUT2BMAX, 5.0 (the
 * D2_Blip's stated cutoff, the larger of RCUT2B 4.5 5.0), not 6.2; then
 * the WARNING said "RCUT2B 5.3" and "the cutoff of a D2_Knot5 line". */
TEST_CASE("joined cutoffs: the cutoff of a D2_Knot5 of a joined line is "
          "derived from its largest knot", "[HPO][join][knot]")
{
    {
        Context ctx = join_ctx("5.3 5.0", "Cut_Dummy Cut_Dummy",
                               "2.5 3.5 4.5 2.5 3.0 3.5", "2 2 2");
        const Startup s = startup(ctx, "OPTIM WATOM (1) 0.1 5.0\n" + kOptimizer);
        INFO(s.error << "\njournal:\n" << s.journal);
        REQUIRE(s.error.empty());
        CHECK(ctx.get<double>("RCUT2B", 0) == 4.5);
        CHECK(ctx.get<double>("RCUT2B", 1) == 5.0);
        const std::string w =
            "tadah hpo: TYPE2B descriptor 1 (D2_Knot5): RCUT2B (1) 5.3 is "
            "above the largest knot (4.5): the knot basis vanishes at its "
            "largest knot, so the cutoff of a D2_Knot5 descriptor is 4.5, "
            "which the CGRID2B knot positions set.";
        CHECK(s.err.find("[HPO] WARNING: " + w) != std::string::npos);
        CHECK(s.journal.find(w) != std::string::npos);
    }
    {
        Context ctx = join_ctx("4.5 5.0", "Cut_Dummy Cut_Dummy",
                               "2.5 3.5 4.8 2.5 3.0 3.5", "2 2 2");
        CHECK(tadah::hpo::detail::apply_knot_rcut_derivation(ctx, true, false));
        CHECK(ctx.get<double>("RCUT2B", 0) == 4.8);
        CHECK(ctx.get<double>("RCUT2B", 1) == 5.0);
    }
    {
        tadah::core::PeriodicTable::initialize();
        Context ctx = join_ctx("4.5 5.0", "Cut_Dummy Cut_Dummy",
                               "2.5 3.5 4.5 2.5 3.0 3.5", "2 2 2");
        const TargetFile tf("OPTIM CGRID2B (3) 4.0 6.2\n" + kOptimizer);
        HPOTargets trg(ctx, tf.path.string());
        CHECK(tadah::hpo::detail::nn_safe_rcutmax(ctx, trg) == 6.2);
    }
}

/* Pins: the bump rules hold for a D2_Blip of a joined line against its own
 * cutoff (RCUT2B (2) = 5), its bumps numbered as entries of the whole key
 * (CGRID2B 4 to 6):
 *  - hard cutoff, a seed bump whose support (centre 4.5, half-width 2/2 = 1)
 *    reaches 5.5: refused at startup by the bump-support gate, naming
 *    "TYPE2B descriptor 2 (D2_Blip): CGRID2B bump 6 (centre 4.5, eta 2)
 *    reaches 5.5, past RCUT2B (2) = 5";
 *  - hard cutoff, OPTIM CGRID2B (6) 3.0 4.8: the per-candidate gate table
 *    arms bump 6 (0-based index 5 of CGRID2B) against the cutoff 5;
 *  - smooth cutoff, a seed centre 5.5 beyond the cutoff: one WARNING,
 *    "TYPE2B descriptor 2 (D2_Blip): CGRID2B bump 6 centre 5.5 lies beyond
 *    RCUT2B (2) = 5 (pathological: only its inner shoulder can contribute)
 *    — move the centre below the cutoff.".
 * Before the fix: a joined line has no bump channel for these rules
 * (bump_channel_halfwidth reads the line's first value): the seed is
 * accepted, the table is empty, and nothing is said. */
TEST_CASE("joined cutoffs: the bump supports of a D2_Blip of a joined line "
          "are held within its cutoff", "[HPO][join][bump]")
{
    {
        Context ctx = join_ctx("4.5 5.0", "Cut_Dummy Cut_Dummy",
                               "2.5 3.5 4.5 2.5 3.0 4.5", "2 2 2");
        const Startup s = startup(ctx, "OPTIM WATOM (1) 0.1 5.0\n" + kOptimizer);
        INFO(s.error);
        CHECK(s.error.find("bump-support gate") != std::string::npos);
        CHECK(s.error.find("TYPE2B descriptor 2 (D2_Blip): CGRID2B bump 6 "
                           "(centre 4.5, eta 2) reaches 5.5, past RCUT2B (2) "
                           "= 5") != std::string::npos);
    }
    {
        tadah::core::PeriodicTable::initialize();
        Context ctx = join_ctx("4.5 5.0", "Cut_Dummy Cut_Dummy",
                               "2.5 3.5 4.5 2.5 3.0 3.5", "2 2 2");
        const TargetFile tf("OPTIM CGRID2B (6) 3.0 4.8\n" + kOptimizer);
        HPOTargets trg(ctx, tf.path.string());
        const auto table = tadah::hpo::detail::build_bump_gate_table(ctx, trg);
        REQUIRE(table.size() == 1u);
        CHECK(table[0].ckey == "CGRID2B");
        REQUIRE(table[0].bumps.size() == 1u);
        CHECK(table[0].bumps[0].index == 5u);
        CHECK(table[0].bumps[0].c_flat >= 0);
        CHECK(table[0].rcut.fixed_max == 5.0);
    }
    {
        Context ctx = join_ctx("4.5 5.0", "Cut_Cos Cut_Cos",
                               "2.5 3.5 4.5 2.5 3.0 5.5", "2 2 2");
        const Startup s = startup(ctx, "OPTIM WATOM (1) 0.1 5.0\n" + kOptimizer);
        INFO(s.error << "\njournal:\n" << s.journal);
        CHECK(s.error.empty());
        const std::string w =
            "TYPE2B descriptor 2 (D2_Blip): CGRID2B bump 6 centre 5.5 lies "
            "beyond RCUT2B (2) = 5 (pathological: only its inner shoulder can "
            "contribute) — move the centre below the cutoff.";
        CHECK(s.err.find("[HPO] WARNING: " + w) != std::string::npos);
        CHECK(s.journal.find(w) != std::string::npos);
    }
}
