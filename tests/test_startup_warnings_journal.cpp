// test_startup_warnings_journal.cpp
//
// The startup warnings of the cutoff checks (validate_hpo_setup): a bump
// centre beyond the cutoff of a smooth cutoff, an OPTIM box of a bump
// centre that reaches beyond it, the knot cutoff that the run derives from
// the largest knot, an OPTIM RCUT of a knot channel. Each is a WARNING of
// the run: on standard error ("[HPO] WARNING: "), in the journal, and so
// among the warnings `tadah hpo --check` lists and counts (the notes a run
// without a journal keeps, HPOLog::notes_without_journal).
#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
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
using tadah::hpo::JournalNote;

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

std::size_t count_of(const std::string& text, const std::string& needle)
{
    std::size_t n = 0;
    for (auto p = text.find(needle); p != std::string::npos;
         p = text.find(needle, p + needle.size()))
        ++n;
    return n;
}

const std::string kOptimizer =
    "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n";

/* A two-bump D2_Blip (support half-width 2) of centres `cgrid`, widths
 * 2.0 and 1.0 and a cutoff RCUT2B 5 of form `cutoff` (Cut_Cos: smooth). */
Context blip_ctx(const char* cgrid, const char* cutoff = "Cut_Cos")
{
    Context c;
    c.add("ATOM", "Ta");
    c.add("WATOM", 1.0);
    c.add("DBFILE", "dummy.db");
    c.add("INIT2B", "true");
    c.add("TYPE2B", "D2_Blip 2 2 Ta Ta");
    c.add("CGRID2B", cgrid);
    c.add("SGRID2B", "2.0 1.0");
    c.add("RCUT2B", 5.0);
    c.add("RCTYPE2B", cutoff);
    c.add("INITMB", "false");
    c.add("VERBOSE", 0);
    c.add("MODEL", "M_BLR BF_Linear");
    return c;
}

/* A D2_Knot5 pair term of knots 5, 3 and 4.2 whose RCUT2B is `rcut2b`
 * (the run derives 5, the largest knot), hard cutoff. */
Context knot_ctx(double rcut2b)
{
    Context c;
    c.add("ATOM", "Ta");
    c.add("WATOM", 1.0);
    c.add("DBFILE", "dummy.db");
    c.add("INIT2B", "true");
    c.add("TYPE2B", "D2_Knot5 3 0 Ta Ta");
    c.add("CGRID2B", "5.0 3.0 4.2");
    c.add("RCUT2B", rcut2b);
    c.add("RCTYPE2B", "Cut_Dummy");
    c.add("INITMB", "false");
    c.add("VERBOSE", 0);
    c.add("MODEL", "M_BLR BF_Linear");
    return c;
}

/* What validate_hpo_setup() writes for `target` and `ctx`: standard error
 * and the journal of a run, and the notes of a check without a journal. */
struct Startup {
    std::string err, journal;
    std::vector<JournalNote> notes;
};

Startup startup(const Context& base, const std::string& target)
{
    tadah::core::PeriodicTable::initialize();
    const fs::path file =
        fs::temp_directory_path() / (unique_name("tadah_swj_") + ".hpo");
    const fs::path journal =
        fs::temp_directory_path() / (unique_name("tadah_swj_") + ".log");
    {
        std::ofstream os(file);
        REQUIRE(os.good());
        os << target;
    }
    Startup out;
    auto& log = HPOLog::instance();
    const auto check = [&](bool with_journal) {
        Context ctx = base;
        if (with_journal) {
            if (log.is_open()) log.finish("reopen-for-test");
            log.reset_for_test();
            log.open(journal.string(), "test", "test_cwd", 0);
        } else {
            log.reset_for_test();
            log.begin_without_journal();
        }
        std::ostringstream err;
        std::streambuf* old = std::cerr.rdbuf(err.rdbuf());
        try {
            HPOTargets trg(ctx, file.string());
            tadah::hpo::validate_hpo_setup(ctx, trg, true, true);
        } catch (...) {
            std::cerr.rdbuf(old);
            log.reset_for_test();
            throw;
        }
        std::cerr.rdbuf(old);
        if (with_journal) {
            out.err = err.str();
            log.finish("test-end");
            out.journal = slurp(journal);
        } else {
            out.notes = log.notes_without_journal();
        }
        log.reset_for_test();
    };
    check(true);
    check(false);
    std::error_code ec;
    fs::remove(file, ec);
    fs::remove(journal, ec);
    return out;
}

std::size_t warnings_with(const std::vector<JournalNote>& notes,
                          const std::string& text)
{
    std::size_t n = 0;
    for (const auto& note : notes)
        if (note.tag == "WARNING" && note.text.find(text) != std::string::npos)
            ++n;
    return n;
}

/* `w` once on standard error as a WARNING, once in the journal, once among
 * the warnings a check lists. */
void check_warned(const Startup& s, const std::string& w)
{
    INFO("journal:\n" << s.journal << "\nstderr:\n" << s.err);
    CHECK(count_of(s.err, "[HPO] WARNING: " + w) == 1);
    CHECK(count_of(s.journal, "WARNING   " + w) == 1);
    CHECK(warnings_with(s.notes, w) == 1);
}

}  // namespace

/* Pins: the startup warnings of the cutoff checks are WARNINGs of the run,
 * on standard error, in the journal and among the warnings of a check
 * without a journal (which `tadah hpo --check` lists and counts):
 *  - a bump centre beyond a smooth RCUT2B: "CGRID2B bump 1 centre 5.5 lies
 *    beyond RCUT2B = 5 (pathological: only its inner shoulder can
 *    contribute) — move the centre below the cutoff.";
 *  - a knot RCUT2B that differs from the largest knot: "tadah hpo: RCUT2B
 *    5.3 is above the largest knot (5): the knot basis vanishes at its
 *    largest knot, so the cutoff of a D2_Knot5 line is 5, which the CGRID2B
 *    knot positions set.";
 *  - an OPTIM RCUT2B of a knot channel: "OPTIM RCUT2B has no effect for the
 *    knot basis family: ...".
 * Before the fix: each went through core::Logger to standard output only,
 * with no journal line and no note for the check (the counts are 0). */
TEST_CASE("startup warnings: the cutoff checks' warnings reach standard "
          "error, the journal and tadah hpo --check",
          "[HPO][startup][journal][check]")
{
    {
        const Startup s = startup(
            blip_ctx("5.5 3.0"),
            "OPTIM SGRID2B (1) 1.0 3.0\nPC_ERMSE 0 1\n" + kOptimizer);
        check_warned(s, "CGRID2B bump 1 centre 5.5 lies beyond RCUT2B = 5 "
                        "(pathological: only its inner shoulder can "
                        "contribute) — move the centre below the cutoff.");
    }
    {
        const Startup s = startup(
            knot_ctx(5.3), "OPTIM WATOM (1) 0.1 5.0\nPC_ERMSE 0 1\n" + kOptimizer);
        check_warned(s, "tadah hpo: RCUT2B 5.3 is above the largest knot "
                        "(5): the knot basis vanishes at its largest knot, so "
                        "the cutoff of a D2_Knot5 line is 5, which the "
                        "CGRID2B knot positions set.");
    }
    {
        const Startup s = startup(
            knot_ctx(5.0), "OPTIM RCUT2B (1) 4.5 5.5\nPC_ERMSE 0 1\n" + kOptimizer);
        check_warned(s, "OPTIM RCUT2B has no effect for the knot basis "
                        "family: the effective cutoff is the largest CGRID2B "
                        "knot and RCUT2B is re-derived on every evaluation.");
    }
}

/* Pins: under a smooth cutoff, an OPTIM CGRID2B box of a bump whose high
 * bound lies beyond a fixed RCUT2B 5 is warned about as a centre beyond it
 * is: "CGRID2B bump 2: the OPTIM CGRID2B (2) high bound 5.5 lies beyond
 * RCUT2B = 5: a candidate whose centre is beyond the cutoff keeps only the
 * bump's inner shoulder (pathological); with the high bound at or below 5,
 * no candidate does." With OPTIM RCUT2B the bound is compared with the
 * box's low bound: "... lies beyond the OPTIM RCUT2B low bound 4.6: ...".
 * A box within the cutoff gives none.
 * Before the fix: a box reaching past the cutoff gave no warning. */
TEST_CASE("startup warnings: an OPTIM CGRID2B box reaching past a smooth "
          "cutoff is warned about", "[HPO][startup][bump_width]")
{
    const std::string tail = "PC_ERMSE 0 1\n" + kOptimizer;
    {
        const Startup s =
            startup(blip_ctx("4.0 3.0"), "OPTIM CGRID2B (2) 2.5 5.5\n" + tail);
        check_warned(s, "CGRID2B bump 2: the OPTIM CGRID2B (2) high bound 5.5 "
                        "lies beyond RCUT2B = 5: a candidate whose centre is "
                        "beyond the cutoff keeps only the bump's inner "
                        "shoulder (pathological); with the high bound at or "
                        "below 5, no candidate does.");
    }
    {
        const Startup s = startup(
            blip_ctx("4.0 3.0"),
            "OPTIM CGRID2B (2) 2.5 4.8\nOPTIM RCUT2B (1) 4.6 6.0\n" + tail);
        check_warned(s, "CGRID2B bump 2: the OPTIM CGRID2B (2) high bound 4.8 "
                        "lies beyond the OPTIM RCUT2B low bound 4.6: a "
                        "candidate whose centre is beyond the cutoff keeps "
                        "only the bump's inner shoulder (pathological); with "
                        "the high bound at or below 4.6, no candidate does.");
    }
    {
        const Startup s =
            startup(blip_ctx("4.0 3.0"), "OPTIM CGRID2B (2) 2.5 4.9\n" + tail);
        INFO("journal:\n" << s.journal);
        CHECK(count_of(s.journal, "high bound") == 0);
    }
}
