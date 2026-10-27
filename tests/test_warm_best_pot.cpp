// test_warm_best_pot.cpp
//
// A WARM seed that is the file the run writes its best potential to,
// best_pot.tadah in the working directory: the run removes that file before
// its first evaluation (HPO_Host::prepare_best_publication) and writes its
// best potential there, so the seed it held is gone once the run starts. The
// target file is accepted (the WARM point is read when the file is read) and
// a WARNING says so, on standard error and in the journal -- and so in the
// summary of `tadah hpo --check`, which lists the journal's warnings.
#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>

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

/* A directory of its own, the working directory while it lives. */
struct ScratchCwd {
    fs::path dir = fs::temp_directory_path() /
                   ("tadah_warm_best_" + std::to_string(::getpid()) + "_" +
                    std::to_string(std::chrono::steady_clock::now()
                                       .time_since_epoch().count()));
    fs::path old = fs::current_path();
    ScratchCwd()
    {
        fs::create_directories(dir);
        fs::current_path(dir);
    }
    ~ScratchCwd()
    {
        std::error_code ec;
        fs::current_path(old, ec);
        fs::remove_all(dir, ec);
    }
};

void write(const fs::path& p, const std::string& text)
{
    std::ofstream os(p);
    REQUIRE(os.good());
    os << text;
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

/* What reading `target` (in the working directory) writes to standard error
 * and to the journal, and the notes a check without a journal keeps. */
struct Read {
    std::string err, journal;
    std::vector<tadah::hpo::JournalNote> notes;
};

Read read_target(const std::string& target)
{
    write("t.hpo", target);
    Context ctx;
    ctx.add("CGRID2B", std::vector<double>{1.0, 2.0});
    Read out;
    auto& log = HPOLog::instance();
    const fs::path journal = fs::current_path() / "journal.log";
    {
        if (log.is_open()) log.finish("reopen-for-test");
        log.reset_for_test();
        log.open(journal.string(), "test", fs::current_path().string(), 0);
        std::ostringstream err;
        std::streambuf* old = std::cerr.rdbuf(err.rdbuf());
        try {
            HPOTargets trg(ctx, "t.hpo");
        } catch (...) {
            std::cerr.rdbuf(old);
            log.finish("test-end");
            log.reset_for_test();
            throw;
        }
        std::cerr.rdbuf(old);
        out.err = err.str();
        log.finish("test-end");
        out.journal = slurp(journal);
    }
    /* tadah hpo --check: no journal, its warnings kept for the summary. */
    log.reset_for_test();
    log.begin_without_journal();
    {
        std::ostringstream err;
        std::streambuf* old = std::cerr.rdbuf(err.rdbuf());
        try {
            HPOTargets trg(ctx, "t.hpo");
        } catch (...) {
            std::cerr.rdbuf(old);
            log.reset_for_test();
            throw;
        }
        std::cerr.rdbuf(old);
        out.notes = log.notes_without_journal();
    }
    log.reset_for_test();
    return out;
}

/* Lines 1-2 the search space, the INIT block on lines 4-6 with the METHOD
 * line on line 5. */
std::string init_warm(const std::string& file)
{
    return "OPTIM CGRID2B (1) 0.5 1.5\nOPTIM CGRID2B (2) 1.6 2.5\n"
           "OPTIMIZER\nINIT\nMETHOD WARM " + file + "\nENDINIT\n"
           "STRATEGY\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 5\nENDSTRATEGY\n"
           "ENDOPTIMIZER\n";
}

std::size_t warnings_with(const std::vector<tadah::hpo::JournalNote>& notes,
                          const std::string& text)
{
    std::size_t n = 0;
    for (const auto& note : notes)
        if (note.tag == "WARNING" && note.text == text) ++n;
    return n;
}

}  // namespace

/* Pins: METHOD WARM of the run's own best_pot.tadah in an INIT block --
 * written best_pot.tadah, ./best_pot.tadah, by its absolute path, and as a
 * symbolic link to it -- gives one WARNING, on standard error ("[HPO]
 * WARNING: "), in the journal and among the notes of a check without a
 * journal: "the INIT block on line 4: METHOD WARM <file> (line 5) reads
 * best_pot.tadah, the file this run writes its best potential to: the run
 * removes it before its first evaluation and writes its best potential
 * there, so only a copy under another name keeps this seed."; the file is
 * read (the point is the file's). A WARM seed of another name, and a
 * best_pot.tadah that is a symbolic link to it (the run replaces the link,
 * not the file it leads to), give none.
 * Before the fix: nothing was said (every count is 0). */
TEST_CASE("WARM of the run's own best_pot.tadah is warned about",
          "[HPO][init_strategy][warm][best_pot]")
{
    const ScratchCwd cwd;
    const std::string seed = "CGRID2B 1.25 2.25\n";
    write("best_pot.tadah", seed);
    write("other.tadah", seed);
    fs::create_symlink("best_pot.tadah", "link.tadah");
    const auto warning = [](const std::string& file) {
        return "the INIT block on line 4: METHOD WARM " + file +
               " (line 5) reads best_pot.tadah, the file this run writes its "
               "best potential to: the run removes it before its first "
               "evaluation and writes its best potential there, so only a "
               "copy under another name keeps this seed.";
    };
    for (const std::string& file :
         {std::string("best_pot.tadah"), std::string("./best_pot.tadah"),
          (fs::current_path() / "best_pot.tadah").string(),
          std::string("link.tadah")}) {
        INFO("METHOD WARM " << file);
        const Read r = read_target(init_warm(file));
        INFO("journal:\n" << r.journal << "\nstderr:\n" << r.err);
        CHECK(count_of(r.err, "[HPO] WARNING: " + warning(file) + "\n") == 1);
        CHECK(count_of(r.journal, warning(file)) == 1);
        CHECK(warnings_with(r.notes, warning(file)) == 1);
    }
    {
        const Read r = read_target(init_warm("other.tadah"));
        INFO("journal:\n" << r.journal << "\nstderr:\n" << r.err);
        CHECK(count_of(r.journal, "the file this run writes its best "
                                  "potential to") == 0);
        CHECK(count_of(r.err, "the file this run writes its best "
                              "potential to") == 0);
    }
    {
        /* best_pot.tadah a link to the seed: the run replaces the link. */
        fs::remove("best_pot.tadah");
        fs::create_symlink("other.tadah", "best_pot.tadah");
        const Read r = read_target(init_warm("other.tadah"));
        INFO("journal:\n" << r.journal);
        CHECK(count_of(r.journal, "the file this run writes its best "
                                  "potential to") == 0);
    }
}

/* Pins: a RESTART block's METHOD WARM best_pot.tadah (line 11, the block on
 * line 8), which each RESTART round reads again, gives one WARNING: "the
 * RESTART block on line 8: METHOD WARM best_pot.tadah (line 11) reads
 * best_pot.tadah, the file this run writes its best potential to: the run
 * removes it before its first evaluation, so each RESTART round starts
 * from the best potential the run has written by then, not from the
 * file's content at startup."
 * Before the fix: nothing was said (the counts are 0). */
TEST_CASE("WARM of the run's own best_pot.tadah in a RESTART block is warned "
          "about", "[HPO][init_strategy][warm][best_pot][restart]")
{
    const ScratchCwd cwd;
    write("best_pot.tadah", "CGRID2B 1.25 2.25\n");
    const std::string target =
        "OPTIM CGRID2B (1) 0.5 1.5\nOPTIM CGRID2B (2) 1.6 2.5\n"
        "OPTIMIZER\nSTRATEGY\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 5\n"
        "  RESTART\n    AFTER_STAGNATION 2\n    MAX 1\n"
        "    METHOD WARM best_pot.tadah\n  ENDRESTART\nENDSTRATEGY\n"
        "ENDOPTIMIZER\n";
    const std::string w =
        "the RESTART block on line 8: METHOD WARM best_pot.tadah (line 11) "
        "reads best_pot.tadah, the file this run writes its best potential "
        "to: the run removes it before its first evaluation, so each RESTART "
        "round starts from the best potential the run has written by then, "
        "not from the file's content at startup.";
    const Read r = read_target(target);
    INFO("journal:\n" << r.journal << "\nstderr:\n" << r.err);
    CHECK(count_of(r.err, "[HPO] WARNING: " + w + "\n") == 1);
    CHECK(count_of(r.journal, w) == 1);
    CHECK(warnings_with(r.notes, w) == 1);
}
