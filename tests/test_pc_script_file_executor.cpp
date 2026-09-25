// test_pc_script_file_executor.cpp
//
// The PC_LAMMPS file-based executor: scripts run through LAMMPS's own
// Input::file() on a ScriptFile rendering, so jump/label/next and
// if-then-"jump" control flow -- including BACKWARD jumps, i.e. loops --
// have native LAMMPS semantics. The old Input::one()-per-line executor
// silently discarded everything after a backward jump (rewind(stdin) is a
// no-op and the sticky label search swallowed the rest), and the stuck
// label state then poisoned the cached instance's NEXT eval: the EXTERN
// hook bound an uninitialized potential and the worker SIGSEGVed (the
// run26 storm). Covered here:
//   - backward label/next/jump loops execute and the remainder is not
//     swallowed (runner level);
//   - if-then-backward-jump (the jump executes INSIDE Input::ifthenelse,
//     invisible to any per-line driver);
//   - the run26 shape at execute_script_task level: EXTERN + backward
//     loops, repeated evals on one cached instance;
//   - poisoned-instance invalidation: a script erroring out of
//     Input::file() (which skips its FILE-handle/nested-counter cleanup)
//     must not accumulate state on the cached entry -- 20 failures would
//     hit LAMMPS's 16-level nesting cap without the erase;
//   - two-section timeout: `clear` destroys a FIRED timer, so without the
//     injected timeremain guard a timed-out section would roll into the
//     next one and the sim would misreport OK;
//   - EXTERN staging across clear (two pair_coeff * * EXTERN sections);
//   - PairTadah::coeff() hardening: EXTERN with nothing staged and an
//     unloadable potential file are clean LAMMPS errors, not null-derefs.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/pc_task_exec.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/script_file.h>
#include <tadah/tadah.lammps/lammps_tadah.h>

#include <pair_tadah.h>

#include "scratch_tmpdir.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

using tadah::hpo::ScriptFile;
using tadah::hpo::ScriptRunStatus;

namespace {

/* Kr pure-2B champion (same fixture as test_pair2b_fastpath.cpp): a real,
 * mechanically stable potential so EXTERN sections produce physics. */
const char* KR_POT_BODY =
    "ATOM              Kr\n"
    "BIAS              false\n"
    "CGRID2B           5.654180679071199478 5.411252065569847503 3.217331521801811256 3.930824515846677247 2.537779659971619584 2.250748637725905610 3.737568302103229101 2.292155276555360999 4.166118987621620029 3.895659538414396827 3.882828370167631160 5.104012407403072160 1.892622625426633620 1.861528429044043031 6.197172052531381148 6.089092122840964905 6.744489232873557505 7.244744465733268690 7.327761008518089092 8.274486017059761167 8.116941074816644175\n"
    "EWEIGHT           1.0\n"
    "FORCE             false\n"
    "FWEIGHT           1.0\n"
    "INIT2B            true\n"
    "INITMB            false\n"
    "LAMBDA            0\n"
    "MODEL             M_BLR   BF_Linear\n"
    "NORM              false\n"
    "RCTYPE2B          Cut_Dummy\n"
    "RCUT2B            8.274486017059761167\n"
    "STRESS            false\n"
    "SWEIGHT           1.0\n"
    "TYPE2B            D2_Knot5   21   0   Kr   Kr\n"
    "WATOM             1.000000000000000000\n"
    "WEIGHTS           0.002022341035452596 -0.234834468777806615 -0.176955931611558204 -0.101273818977953803 -0.640091273014962159 -0.371627635837921522 -0.122730970028676681 -0.639591836790447621 0.889938055528681660 -0.501552854167752749 -0.306677826118867403 0.142464681149275907 -0.541937682987483349 -0.085416359679197792 0.085834571424044798 0.000135043203549349 0.000000000000000000 -0.011937057559490772 0.000000000000000000 0.000000000000000000 0.000000690371680869\n";

tadah::core::Context kr_pot_ctx() {
    tadah::core::PeriodicTable::initialize();
    const std::string path = std::filesystem::temp_directory_path() /
        ("tadah_sfe_pot_" + std::to_string(::getpid()) + ".tadah");
    std::ofstream ofs(path);
    ofs << KR_POT_BODY;
    ofs.close();
    auto ctx = tadah::core::CoreUserConfigParser().parse(path);
    std::remove(path.c_str());
    return ctx;
}

LammpsRunner::LMPPtr make_instance(
    const std::vector<std::string>& extra = {}) {
    LammpsRunner runner;
    std::vector<std::string> invar =
        {"lmp", "-log", "none", "-screen", "none"};
    invar.insert(invar.end(), extra.begin(), extra.end());
    return runner.create_instance(0, invar);
}

double getd(const LammpsRunner::LMPPtr& lmp, const char* name) {
    LammpsRunner runner;
    return std::stod(runner.get_variable(lmp, name));
}

/* The rolled eos_dilute_fast2.in shape, miniaturized: an index-variable
 * walk plus a loop-variable accumulation, all three exports AFTER the
 * backward jumps (i.e. inside the region the old executor swallowed). */
std::vector<std::string> backward_loop_lines() {
    return {
        "clear",
        "units lj",
        "variable acc equal 0.0",
        "variable i loop 5",
        "label top",
        "variable acc equal $(v_acc+v_i)",
        "next i",
        "jump SELF top",
        "variable out equal $(2*v_acc)",
    };
}

std::vector<std::string> kr_loop_extern_lines() {
    return {
        "clear",
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        "atom_modify map array",
        "lattice bcc 4.1024",
        "region box block 0 5 0 5 0 5 units lattice",
        "create_box 1 box",
        "create_atoms 1 region box",
        "mass 1 83.798",
        "pair_style tadah",
        "pair_coeff * * EXTERN Kr",
        "neighbor 2.0 bin",
        "run 0 post no",
        /* scan the lattice constant like eos_dilute_fast2 scans dilation */
        "variable L0 equal $(lx)",
        "variable scan index 1.00 1.02 1.04",
        "variable npt equal 0",
        "label measure",
        "variable s equal ${scan}",
        "change_box all x final 0 $(v_s*v_L0) y final 0 $(v_s*v_L0) z final 0 $(v_s*v_L0) remap units box",
        "run 0 post no",
        "variable E_${npt} equal $(pe/atoms)",
        "variable npt equal $(v_npt+1)",
        "next scan",
        "jump SELF measure",
        /* exports live AFTER the backward jump */
        "variable n_measured equal $(v_npt)",
        "variable e_last equal $(v_E_2)",
    };
}

}  // namespace

TEST_CASE("ScriptFile renders once and unlinks on destruction",
          "[script_file]") {
    std::string path;
    {
        ScriptFile sf(backward_loop_lines());
        REQUIRE(sf.valid());
        REQUIRE_FALSE(sf.has_extern());
        path = sf.path();
        REQUIRE(std::filesystem::exists(path));
    }
    REQUIRE_FALSE(std::filesystem::exists(path));

    ScriptFile sf2(kr_loop_extern_lines());
    REQUIRE(sf2.has_extern());
}

TEST_CASE("ScriptFile leaves no file in the temporary directory, not even "
          "while it is in use", "[script_file][tempfile]") {
    /* A rendered script used to be a named file, tadah_pc_<pid>_<n>.in,
     * removed by the destructor. A process that ends without running
     * destructors -- a PC_ISOLATION worker killed at its hard deadline or
     * crashing inside LAMMPS, or killed with its parent -- left the file
     * behind, about one per full ctest run. The rendering must never own
     * a name in the temporary directory, and must stay readable through
     * path() for as long as the ScriptFile lives. */
    hpo_test::ScratchTmpdir tmp("scriptfile");
    REQUIRE(tmp.count("") == 0);

    const std::vector<std::string> lines = backward_loop_lines();
    std::string want;
    for (const auto& l : lines) want += l + "\n";
    auto read_all = [](const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        REQUIRE(in.is_open());
        return std::string(std::istreambuf_iterator<char>(in),
                           std::istreambuf_iterator<char>());
    };
    {
        ScriptFile sf(lines);
        REQUIRE(sf.valid());
        CHECK(tmp.count("") == 0);
        /* LAMMPS opens the path afresh for every evaluation. */
        CHECK(read_all(sf.path()) == want);
        CHECK(read_all(sf.path()) == want);

        ScriptFile moved(std::move(sf));
        CHECK_FALSE(sf.valid());
        REQUIRE(moved.valid());
        CHECK(read_all(moved.path()) == want);
        CHECK(tmp.count("") == 0);
    }
    CHECK(tmp.count("") == 0);
}

TEST_CASE("ScriptFile keeps one rendering per distinct script",
          "[script_file][tempfile]") {
    /* Every THREAD-mode slot renders the same scripts: one rendering per
     * distinct script, shared while any of them is alive, keeps a wide
     * THREADS run to a few descriptors. */
    const ScriptFile a(backward_loop_lines());
    const ScriptFile b(backward_loop_lines());
    const ScriptFile c(kr_loop_extern_lines());
    REQUIRE(a.valid());
    REQUIRE(b.valid());
    REQUIRE(c.valid());
    CHECK(a.path() == b.path());
    CHECK(a.path() != c.path());
    CHECK_FALSE(a.has_extern());
    CHECK(c.has_extern());
}

TEST_CASE("ScriptFile renders although files left by an earlier process "
          "hold the names it would use", "[script_file][tempfile]") {
    /* Earlier builds left tadah_pc_<pid>_<n>.in behind (a killed worker),
     * and pids are reused: by a later process, or by a process in another
     * PID namespace that shares the temporary directory. A rendering must
     * not depend on such a name being free -- a failed rendering fails the
     * script's evaluation -- and must leave files it did not create alone. */
    hpo_test::ScratchTmpdir tmp("scriptfile_stale");
    const std::string pid = std::to_string(::getpid());
    constexpr int n_stale = 64;
    auto stale_path = [&](int k) {
        return tmp.path() / ("tadah_pc_" + pid + "_" + std::to_string(k) +
                             ".in");
    };
    auto stale_text = [](int k) { return "stale " + std::to_string(k) + "\n"; };
    for (int k = 0; k < n_stale; ++k)
        std::ofstream(stale_path(k), std::ios::binary) << stale_text(k);
    REQUIRE(tmp.count("tadah_pc_") == n_stale);

    auto read_all = [](const std::filesystem::path& path) {
        std::ifstream in(path, std::ios::binary);
        REQUIRE(in.is_open());
        return std::string(std::istreambuf_iterator<char>(in),
                           std::istreambuf_iterator<char>());
    };
    {
        /* Distinct scripts: each one needs a rendering of its own. */
        std::vector<ScriptFile> files;
        std::vector<std::string> wants;
        for (int k = 0; k < 4; ++k) {
            std::vector<std::string> lines = backward_loop_lines();
            lines.push_back("variable stale_probe equal " + std::to_string(k));
            std::string want;
            for (const auto& l : lines) want += l + "\n";
            REQUIRE_NOTHROW(files.emplace_back(lines));
            wants.push_back(want);
        }
        for (std::size_t k = 0; k < files.size(); ++k) {
            REQUIRE(files[k].valid());
            CHECK(read_all(files[k].path()) == wants[k]);
        }
        CHECK(tmp.count("tadah_pc_") == n_stale);
    }
    /* The files of the earlier process are untouched. */
    CHECK(tmp.count("tadah_pc_") == n_stale);
    for (int k = 0; k < n_stale; ++k) {
        INFO("stale file " << stale_path(k));
        REQUIRE(std::filesystem::exists(stale_path(k)));
        CHECK(read_all(stale_path(k)) == stale_text(k));
    }
}

TEST_CASE("backward label/next/jump loop executes with native semantics",
          "[script_file][jump]") {
    auto lmp = make_instance();
    REQUIRE(lmp != nullptr);

    LammpsRunner runner;
    /* EXTERN-less script with an empty Context: no potential is built. */
    REQUIRE(runner.run_script(lmp, backward_loop_lines(),
                              tadah::core::Context{}));

    /* 1+2+3+4+5, then the post-loop export doubled it -- the region after
     * the backward jump was NOT swallowed. */
    REQUIRE(getd(lmp, "acc") == Approx(15.0));
    REQUIRE(getd(lmp, "out") == Approx(30.0));

    /* Same cached instance, second eval: no sticky label state. */
    REQUIRE(runner.run_script(lmp, backward_loop_lines(),
                              tadah::core::Context{}));
    REQUIRE(getd(lmp, "out") == Approx(30.0));
}

TEST_CASE("if-then with a backward jump loops correctly",
          "[script_file][jump]") {
    /* The jump executes INSIDE Input::ifthenelse -- invisible to any
     * per-line driver; only the file() executor can honor it. */
    const std::vector<std::string> lines = {
        "clear",
        "units lj",
        "variable n equal 0.0",
        "label again",
        "variable n equal $(v_n+1)",
        "if \"$(v_n) < 4.5\" then \"jump SELF again\"",
        "variable done equal $(v_n)",
    };
    auto lmp = make_instance();
    REQUIRE(lmp != nullptr);

    LammpsRunner runner;
    REQUIRE(runner.run_script(lmp, lines, tadah::core::Context{}));
    REQUIRE(getd(lmp, "n") == Approx(5.0));
    REQUIRE(getd(lmp, "done") == Approx(5.0));
}

TEST_CASE("run26 shape: EXTERN + backward loops, repeated evals on one slot",
          "[script_file][jump][extern][pc_task]") {
    auto pot_ctx = kr_pot_ctx();

    tadah::hpo::PcSlotState<LammpsRunner> st;
    const std::vector<std::string> invar =
        {"lmp", "-log", "none", "-screen", "none"};
    const std::vector<std::string> outvars = {"n_measured", "e_last"};

    for (unsigned long long iter = 1; iter <= 3; ++iter) {
        auto res = tadah::hpo::execute_script_task<LammpsRunner>(
            "kr_loop", kr_loop_extern_lines(), invar, outvars,
            {}, {}, pot_ctx, iter, {}, st);
        INFO("iter " << iter << " error: " << res.error_msg);
        REQUIRE(res.status == tadah::hpo::PcTaskStatus::OK);
        REQUIRE(res.outvar_values.size() == 2);
        /* all 3 scan points measured; E_2 exists and is a real number */
        REQUIRE(std::stod(res.outvar_values[0]) == Approx(3.0));
        REQUIRE(std::isfinite(std::stod(res.outvar_values[1])));
    }
    /* one cached instance + one rendered script, reused across evals */
    REQUIRE(st.lmp_cache.size() == 1);
}

TEST_CASE("erroring scripts do not accumulate poisoned state on the slot",
          "[script_file][pc_task][invalidate]") {
    auto pot_ctx = kr_pot_ctx();

    tadah::hpo::PcSlotState<LammpsRunner> st;
    const std::vector<std::string> invar =
        {"lmp", "-log", "none", "-screen", "none"};

    /* `jump SELF nope` reaches EOF with the label search active ->
     * "Label wasn't found" thrown out of Input::file(), which skips its
     * cleanup (open FILE handle, nested-file counter). LAMMPS caps input
     * nesting at 16 -- without invalidation the 16th retry dies. */
    const std::vector<std::string> bad = {
        "clear", "units lj", "jump SELF nope", "variable x equal 1.0"};
    const std::vector<std::string> good = backward_loop_lines();

    for (unsigned long long iter = 1; iter <= 20; ++iter) {
        auto rbad = tadah::hpo::execute_script_task<LammpsRunner>(
            "bad", bad, invar, {"x"}, {}, {}, pot_ctx, iter, {}, st);
        REQUIRE(rbad.status == tadah::hpo::PcTaskStatus::RUN_FAIL);
        /* poisoned entry erased -> the bad script never occupies the cache */
        REQUIRE(st.lmp_cache.find([&]{
            std::string k = "bad";
            for (const auto& s : invar) { k += '\x01'; k += s; }
            return k; }()) == st.lmp_cache.end());

        auto rgood = tadah::hpo::execute_script_task<LammpsRunner>(
            "good", good, invar, {"out"}, {}, {}, pot_ctx, iter, {}, st);
        REQUIRE(rgood.status == tadah::hpo::PcTaskStatus::OK);
        REQUIRE(std::stod(rgood.outvar_values[0]) == Approx(30.0));
    }
}

TEST_CASE("two-section timeout: fired timer aborts at the clear boundary",
          "[script_file][timeout]") {
    /* Section 1 melts LJ far beyond the 1 s budget; section 2 is trivial.
     * `clear` would destroy the FIRED timer, so without the injected
     * timeremain guard the script would finish and misreport OK. The
     * guard aborts with the sentinel -> TIMEOUT_ABORT. */
    const std::string body = R"(clear
units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 8 0 8 0 8
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 3.0 87287
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
neighbor 0.3 bin
neigh_modify delay 0 every 20 check no
fix 1 all nve
variable early equal 7.0
run 500000000
clear
units lj
variable late equal 1.0
)";
    /* Render through LammpsSim so the real prep pipeline injects the
     * timer lines AND the boundary guard. */
    const std::string spath = std::filesystem::temp_directory_path() /
        ("tadah_sfe_to_" + std::to_string(::getpid()) + ".in");
    { std::ofstream os(spath); os << body; }
    std::string cmd = "--script " + spath + " --varloss late 0.0 1.0";
    tadah::hpo::LammpsSim sim(cmd, "1", 1.0e6, tadah::LossFunction{});
    std::remove(spath.c_str());

    /* guard sits between the sections, right before the mid-script clear */
    bool has_guard = false;
    for (const auto& l : sim.script_lines)
        has_guard |= l.find(tadah::hpo::PC_TIMEOUT_ABORT_SENTINEL)
                     != std::string::npos;
    REQUIRE(has_guard);

    auto lmp = make_instance({"-var", "maxtime", "1 every 1"});
    REQUIRE(lmp != nullptr);

    LammpsRunner runner;
    const ScriptFile sf(sim.script_lines);
    ScriptRunStatus why = ScriptRunStatus::OK;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = runner.run_script(lmp, sf, tadah::core::Context{}, &why);
    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    REQUIRE_FALSE(ok);
    REQUIRE(why == ScriptRunStatus::TIMEOUT_ABORT);
    REQUIRE(secs < 30.0);
    /* failvar contract: early-defined variables survive the abort */
    REQUIRE(getd(lmp, "early") == Approx(7.0));
}

TEST_CASE("EXTERN staging survives clear: two sections, one staged pot",
          "[script_file][extern]") {
    auto pot_ctx = kr_pot_ctx();
    auto pot = std::make_shared<tadah::LammpsTadah>(pot_ctx);

    auto section = [](const char* a0, const char* evar) {
        return std::vector<std::string>{
            "clear", "units metal", "boundary p p p", "atom_style atomic",
            "atom_modify map array",
            std::string("lattice bcc ") + a0,
            "region box block 0 5 0 5 0 5 units lattice",
            "create_box 1 box", "create_atoms 1 region box",
            "mass 1 83.798", "pair_style tadah",
            "pair_coeff * * EXTERN Kr", "neighbor 2.0 bin",
            "run 0 post no",
            std::string("variable ") + evar + " equal $(pe/atoms)"};
    };
    std::vector<std::string> lines = section("4.1024", "e1");
    const auto s2 = section("4.2", "e2");
    lines.insert(lines.end(), s2.begin(), s2.end());

    auto lmp = make_instance();
    REQUIRE(lmp != nullptr);

    LammpsRunner runner;
    const ScriptFile sf(lines);
    REQUIRE(sf.has_extern());
    ScriptRunStatus why = ScriptRunStatus::OK;
    REQUIRE(runner.run_script(lmp, sf, pot, &why));
    REQUIRE(why == ScriptRunStatus::OK);
    REQUIRE(std::isfinite(getd(lmp, "e1")));
    REQUIRE(std::isfinite(getd(lmp, "e2")));
    REQUIRE(getd(lmp, "e1") != Approx(getd(lmp, "e2")));
}

TEST_CASE("coeff() hardening: unusable EXTERN/pot file errors cleanly",
          "[script_file][extern][coeff]") {
    tadah::core::PeriodicTable::initialize();
    auto lmp = make_instance();
    REQUIRE(lmp != nullptr);

    for (const char* cmd : {"clear", "units metal", "lattice bcc 4.1",
                            "region box block 0 1 0 1 0 1 units lattice",
                            "create_box 1 box", "create_atoms 1 region box",
                            "mass 1 83.798", "pair_style tadah"})
        lmp->input->one(cmd);

    /* EXTERN with nothing staged and no pre-load: a clean LAMMPS error
     * (previously: load()'s failure was discarded and the null potential
     * dereferenced -> SIGSEGV). */
    REQUIRE_THROWS(lmp->input->one("pair_coeff * * EXTERN Kr"));

    /* Non-EXTERN unloadable file: same contract. */
    for (const char* cmd : {"clear", "units metal", "lattice bcc 4.1",
                            "region box block 0 1 0 1 0 1 units lattice",
                            "create_box 1 box", "create_atoms 1 region box",
                            "mass 1 83.798", "pair_style tadah"})
        lmp->input->one(cmd);
    REQUIRE_THROWS(
        lmp->input->one("pair_coeff * * /nonexistent_dir/nope.tadah Kr"));
}
