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
//     unloadable potential file are clean LAMMPS errors, not null-derefs;
//   - variables read back after the script ([readback]): a LAMMPS error
//     raised while HPO evaluates a --varloss/--outvar/--hotvar/--failvar
//     formula stays inside the task, and a LAMMPS error inside the script
//     carries its message;
//   - hot values injected before the script ([inject]): an injection
//     LAMMPS refuses fails the task with its message;
//   - an instance that cannot be created and a script that cannot be
//     rendered ([create]) fail the task with the reason;
//   - the task's guard: an exception that is not a LAMMPS error fails the
//     task, in both PC_ISOLATION modes alike.

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

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
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
    "FORCE             false\n"
    "INIT2B            true\n"
    "INITMB            false\n"
    "LAMBDA            0\n"
    "MODEL             M_BLR   BF_Linear\n"
    "NORM              false\n"
    "RCTYPE2B          Cut_Dummy\n"
    "RCUT2B            8.274486017059761167\n"
    "STRESS            false\n"
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
    return runner.create_instance(invar);
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

namespace {

/* A fast script for the read-back cases: no potential to build (pair_style
 * zero, so pe is 0), a thermo_style that computes pe but not press, an
 * early violation measure, then E and Pend defined after the run. HPO
 * evaluates each read-back variable's formula after the script; `press`
 * there raises a LAMMPS error (Thermo::check_press_scalar, error->all) and
 * `1.0/(pe-pe)` a divide by zero (Variable, error->one). */
std::vector<std::string> readback_lines(const std::string& pend_formula) {
    return {"clear", "units metal", "boundary p p p", "atom_style atomic",
            "lattice bcc 4.1024", "region box block 0 2 0 2 0 2 units lattice",
            "create_box 1 box", "create_atoms 1 region box", "mass 1 83.798",
            "pair_style zero 5.0", "pair_coeff * *",
            "thermo_style custom step pe", "variable viol equal 0.25",
            "run 0 post no", "variable E equal pe",
            "variable Pend equal " + pend_formula};
}

/* The slot cache key of a script (pc_task_exec.h: the script name, then
 * '\x01' and each invar token). */
std::string slot_cache_key(const std::string& script,
                           const std::vector<std::string>& invar) {
    std::string k = script;
    for (const auto& s : invar) { k += '\x01'; k += s; }
    return k;
}

/* The two LAMMPS errors a read-back can raise, with the text each one's
 * message carries. */
struct ReadbackError {
    const char* formula;
    const char* lammps_text;
};
const ReadbackError kPressError{"press", "Thermo keyword press in variable "
                                         "requires thermo to use/init press"};
const ReadbackError kDivideError{"1.0/(pe-pe)",
                                 "Divide by 0 in variable formula"};

}  // namespace

/* Pins: a --hotvar or --failvar whose formula raises a LAMMPS error when HPO
 * reads it after the script -- `press` under a thermo_style without it
 * (error->all), `1.0/(pe-pe)` (a divide by zero, error->one) -- is returned
 * "" with its LAMMPS message in hotvar_errors (or failvar_errors: one per
 * name, "" for one that was read), the task OK with no error_msg: whether a
 * read-back error fails the task is the parent's call, which judges every
 * variable of the task in one place (apply_task_result in hpo_host_impl.hpp:
 * a --hotvar or --failvar that cannot be read fails it, journalled with its
 * message -- pinned end to end by [pc_readback_e2e] and
 * [.tadah_cli_pc_readback]). The other variables are handed on (E = 0, the
 * early failvar viol = 0.25), and the slot's LAMMPS instance is discarded,
 * so that the next script on the slot -- the same name with a readable
 * Pend -- runs on a fresh instance and succeeds with no read-back error.
 * Before the fix: PcTaskResult had no hotvar_errors or failvar_errors (the
 * file does not compile), and the worker failed the task itself --
 * RUN_FAIL, error_msg "reading --hotvar Pend after the script: <LAMMPS
 * message>", no outvar or hot value handed on -- so the parent judged no
 * other variable of the task (a --varloss without a number, a second bad hot
 * value went unnamed): with the two members alone the status check fails. (Before the read-back fix the read
 * caught the error and returned "" with status OK, and the parent skipped
 * the empty value.) */
TEST_CASE("a --hotvar or --failvar whose read-back raises a LAMMPS error "
          "is returned with its message for the parent to judge",
          "[script_file][pc_task][readback]") {
    auto pot_ctx = kr_pot_ctx();
    tadah::hpo::PcSlotState<LammpsRunner> st;
    const std::vector<std::string> invar =
        {"lmp", "-log", "none", "-screen", "none"};
    using tadah::hpo::PcTaskStatus;

    struct Case {
        const char* role;
        ReadbackError err;
    };
    for (const Case c : {Case{"--hotvar", kPressError},
                         Case{"--hotvar", kDivideError},
                         Case{"--failvar", kPressError},
                         Case{"--failvar", kDivideError}}) {
        const bool hot = std::string(c.role) == "--hotvar";
        const std::vector<std::string> hotvars =
            hot ? std::vector<std::string>{"Pend"} : std::vector<std::string>{};
        const std::vector<std::string> failvars =
            hot ? std::vector<std::string>{"viol"}
                : std::vector<std::string>{"viol", "Pend"};
        INFO(c.role << " Pend = " << c.err.formula);

        auto res = tadah::hpo::execute_script_task<LammpsRunner>(
            "rb", readback_lines(c.err.formula), invar, {"E"}, hotvars,
            failvars, pot_ctx, 1, {}, st);
        INFO("error_msg: " << res.error_msg);
        REQUIRE(res.status == PcTaskStatus::OK);
        CHECK(res.error_msg.empty());
        REQUIRE(res.outvar_values.size() == 1);
        CHECK(std::stod(res.outvar_values[0]) == 0.0);
        CHECK(res.outvar_errors.empty());
        REQUIRE(res.failvar_values.size() == failvars.size());
        CHECK(res.failvar_values[0] == "0.25");
        if (hot) {
            CHECK(res.hotvar_values == std::vector<std::string>{""});
            REQUIRE(res.hotvar_errors.size() == 1);
            CHECK_THAT(res.hotvar_errors[0],
                       Catch::Contains(c.err.lammps_text));
            CHECK(res.failvar_errors.empty());
        } else {
            CHECK(res.failvar_values[1].empty());
            CHECK(res.hotvar_errors.empty());
            REQUIRE(res.failvar_errors.size() == 2);
            CHECK(res.failvar_errors[0].empty());
            CHECK_THAT(res.failvar_errors[1],
                       Catch::Contains(c.err.lammps_text));
        }
        CHECK(st.lmp_cache.count(slot_cache_key("rb", invar)) == 0);

        /* The same script name, now readable: a fresh instance runs it. */
        auto good = tadah::hpo::execute_script_task<LammpsRunner>(
            "rb", readback_lines("pe+1.5"), invar, {"E"}, hotvars, failvars,
            pot_ctx, 1, {}, st);
        INFO("good error_msg: " << good.error_msg);
        REQUIRE(good.status == PcTaskStatus::OK);
        CHECK(good.error_msg.empty());
        CHECK(good.hotvar_errors.empty());
        CHECK(good.failvar_errors.empty());
        REQUIRE(good.outvar_values.size() == 1);
        CHECK(std::stod(good.outvar_values[0]) == 0.0);
        if (hot) {
            REQUIRE(good.hotvar_values.size() == 1);
            CHECK(std::stod(good.hotvar_values[0]) == 1.5);
        } else {
            REQUIRE(good.failvar_values.size() == 2);
            CHECK(std::stod(good.failvar_values[1]) == 1.5);
        }
        st.lmp_cache.clear();
    }
}

/* Pins: an --outvar/--varloss variable whose formula raises a LAMMPS error
 * when HPO reads it after the script stays inside execute_script_task: the
 * task reports OK with "" for that variable and the values of the others
 * (E = 0), every failvar is read (viol = 0.25), and the slot's LAMMPS
 * instance is discarded, so the next script on the slot -- the same name
 * with a readable Pend -- runs on a fresh instance and succeeds. Whether
 * the variable's read-back error fails the task is the parent's call (the
 * worker does not know which names feed the loss): a --varloss variable
 * fails it, a diagnostic --outvar is NaN with a journal line -- pinned end
 * to end, message included, by [pc_readback_e2e] and
 * [.tadah_cli_pc_readback].
 * Before the fix: the LAMMPS exception (LAMMPSException for press,
 * LAMMPSAbortException for the divide by zero) escaped execute_script_task
 * from the unguarded read, the failvars were never read and the errored
 * instance stayed cached. Under PC_ISOLATION THREAD it went on through
 * run_lammps to the evaluation's catch, whose partial Result the row
 * writer read out of bounds (SIGSEGV). Each section fails with the
 * unexpected exception. */
TEST_CASE("an --outvar or --varloss whose read-back raises a LAMMPS error "
          "is returned empty and its instance discarded",
          "[script_file][pc_task][readback]") {
    auto pot_ctx = kr_pot_ctx();
    tadah::hpo::PcSlotState<LammpsRunner> st;
    const std::vector<std::string> invar =
        {"lmp", "-log", "none", "-screen", "none"};
    using tadah::hpo::PcTaskStatus;

    for (const ReadbackError& err : {kPressError, kDivideError}) {
        INFO("Pend = " << err.formula);
        tadah::hpo::PcTaskResult res;
        REQUIRE_NOTHROW(res = tadah::hpo::execute_script_task<LammpsRunner>(
            "rb", readback_lines(err.formula), invar, {"E", "Pend"}, {},
            {"viol"}, pot_ctx, 1, {}, st));
        INFO("error_msg: " << res.error_msg);
        CHECK(res.status == PcTaskStatus::OK);
        CHECK(res.error_msg.empty());
        REQUIRE(res.outvar_values.size() == 2);
        CHECK(std::stod(res.outvar_values[0]) == 0.0);
        CHECK(res.outvar_values[1].empty());
        CHECK(res.failvar_values == std::vector<std::string>{"0.25"});
        CHECK(st.lmp_cache.count(slot_cache_key("rb", invar)) == 0);

        auto good = tadah::hpo::execute_script_task<LammpsRunner>(
            "rb", readback_lines("pe+1.5"), invar, {"E", "Pend"}, {},
            {"viol"}, pot_ctx, 1, {}, st);
        INFO("good error_msg: " << good.error_msg);
        REQUIRE(good.status == PcTaskStatus::OK);
        REQUIRE(good.outvar_values.size() == 2);
        CHECK(std::stod(good.outvar_values[1]) == 1.5);
        CHECK(st.lmp_cache.count(slot_cache_key("rb", invar)) == 1);
        st.lmp_cache.clear();
    }
}

/* Pins: a variable read back after another one's read-back error carries
 * the LAMMPS error of its own evaluation, not "has a circular dependency":
 *   - outvars E, Pend, Pdiff with Pdiff = v_Pend-1.0: the task is OK, E is
 *     read (0), Pend and Pdiff are "" and outvar_errors names, for each of
 *     them, Pend's error (Pdiff evaluates Pend);
 *   - a --failvar vdep = (v_Pend>1e9) read after the outvar Pend: vdep is
 *     "" and failvar_errors names Pend's error for it, the early failvar
 *     viol still read;
 *   - Pend declared as an --outvar and as a --hotvar: the hot value is ""
 *     and hotvar_errors names its error.
 * The task is OK in each (the parent judges a read-back error).
 * Each for `press` under a thermo_style without it and for `1.0/(pe-pe)`.
 * Before the fix: PcTaskResult had no hotvar_errors or failvar_errors (the
 * file does not compile), and the worker failed the task on a --hotvar or
 * --failvar read error (RUN_FAIL, error_msg "reading --failvar vdep after
 * the script: <Pend's error>"): with the two members alone the status check
 * of the dependent --failvar fails.
 * (Before the circular-dependency fix the first error left Pend marked as
 * being evaluated, LAMMPS Variable::eval_in_progress, so every later read
 * that reached it raised "Variable Pend: has a circular dependency".) */
TEST_CASE("a variable read back after a read-back error carries its own "
          "LAMMPS error", "[script_file][pc_task][readback]") {
    auto pot_ctx = kr_pot_ctx();
    tadah::hpo::PcSlotState<LammpsRunner> st;
    const std::vector<std::string> invar =
        {"lmp", "-log", "none", "-screen", "none"};
    using tadah::hpo::PcTaskStatus;
    using Catch::Contains;

    for (const ReadbackError& err : {kPressError, kDivideError}) {
        INFO("Pend = " << err.formula);
        std::vector<std::string> lines = readback_lines(err.formula);
        lines.push_back("variable Pdiff equal v_Pend-1.0");
        lines.push_back("variable vdep equal (v_Pend>1e9)");
        {
            INFO("a dependent --outvar");
            const auto res = tadah::hpo::execute_script_task<LammpsRunner>(
                "rb", lines, invar, {"E", "Pend", "Pdiff"}, {}, {"viol"},
                pot_ctx, 1, {}, st);
            INFO("error_msg: " << res.error_msg);
            REQUIRE(res.status == PcTaskStatus::OK);
            REQUIRE(res.outvar_values.size() == 3);
            CHECK(std::stod(res.outvar_values[0]) == 0.0);
            CHECK(res.outvar_values[1].empty());
            CHECK(res.outvar_values[2].empty());
            REQUIRE(res.outvar_errors.size() == 3);
            CHECK(res.outvar_errors[0].empty());
            for (std::size_t j : {1, 2}) {
                INFO("outvar_errors[" << j << "]");
                CHECK_THAT(res.outvar_errors[j],
                           Contains(err.lammps_text) &&
                               !Contains("circular dependency"));
            }
            CHECK(res.failvar_values == std::vector<std::string>{"0.25"});
            CHECK(st.lmp_cache.count(slot_cache_key("rb", invar)) == 0);
        }
        {
            INFO("a dependent --failvar");
            const auto res = tadah::hpo::execute_script_task<LammpsRunner>(
                "rb", lines, invar, {"E", "Pend"}, {}, {"viol", "vdep"},
                pot_ctx, 1, {}, st);
            INFO("error_msg: " << res.error_msg);
            REQUIRE(res.status == PcTaskStatus::OK);
            CHECK(res.error_msg.empty());
            CHECK(res.failvar_values == std::vector<std::string>{"0.25", ""});
            REQUIRE(res.failvar_errors.size() == 2);
            CHECK(res.failvar_errors[0].empty());
            CHECK_THAT(res.failvar_errors[1],
                       Contains(err.lammps_text) &&
                           !Contains("circular dependency"));
            CHECK(st.lmp_cache.count(slot_cache_key("rb", invar)) == 0);
        }
        {
            INFO("one name as an --outvar and a --hotvar");
            const auto res = tadah::hpo::execute_script_task<LammpsRunner>(
                "rb", lines, invar, {"E", "Pend"}, {"Pend"}, {"viol"},
                pot_ctx, 1, {}, st);
            INFO("error_msg: " << res.error_msg);
            REQUIRE(res.status == PcTaskStatus::OK);
            CHECK(res.error_msg.empty());
            CHECK(res.hotvar_values == std::vector<std::string>{""});
            REQUIRE(res.hotvar_errors.size() == 1);
            CHECK_THAT(res.hotvar_errors[0],
                       Contains(err.lammps_text) &&
                           !Contains("circular dependency"));
            CHECK(st.lmp_cache.count(slot_cache_key("rb", invar)) == 0);
        }
        st.lmp_cache.clear();
    }
}

/* Pins: every --hotvar and --failvar of a task whose read-back raises a
 * LAMMPS error carries its own message, each in its place -- hot Pend =
 * press and Q = 1.0/(pe-pe), fail viol (readable, 0.25) and R = press*2:
 * hotvar_errors {<press error>, <divide by 0>}, failvar_errors {"", <press
 * error>}, their values "" --, every other variable is handed on (E = 0),
 * and the task is OK with no error_msg, so the parent names each variable
 * at fault, with the loss and hot values it finds without a number, in one
 * journal line (pinned by [pc_readback_e2e]).
 * Before the fix: PcTaskResult had no hotvar_errors or failvar_errors (the
 * file does not compile), and the worker failed the task (RUN_FAIL) with
 * the three errors joined in its error_msg and handed on no outvar or hot
 * value, so the parent could name no other variable: with the two members
 * alone the status check fails. */
TEST_CASE("every --hotvar and --failvar whose read-back raises a LAMMPS error "
          "carries its own message",
          "[script_file][pc_task][readback]") {
    auto pot_ctx = kr_pot_ctx();
    tadah::hpo::PcSlotState<LammpsRunner> st;
    const std::vector<std::string> invar =
        {"lmp", "-log", "none", "-screen", "none"};
    std::vector<std::string> lines = readback_lines(kPressError.formula);
    lines.push_back("variable Q equal 1.0/(pe-pe)");
    lines.push_back("variable R equal press*2");
    const auto res = tadah::hpo::execute_script_task<LammpsRunner>(
        "rb", lines, invar, {"E"}, {"Pend", "Q"}, {"viol", "R"}, pot_ctx, 1,
        {}, st);
    INFO("error_msg: " << res.error_msg);
    REQUIRE(res.status == tadah::hpo::PcTaskStatus::OK);
    CHECK(res.error_msg.empty());
    REQUIRE(res.outvar_values.size() == 1);
    CHECK(std::stod(res.outvar_values[0]) == 0.0);
    CHECK(res.hotvar_values == std::vector<std::string>{"", ""});
    REQUIRE(res.hotvar_errors.size() == 2);
    CHECK_THAT(res.hotvar_errors[0],
               Catch::Contains(kPressError.lammps_text) &&
                   !Catch::Contains("circular dependency"));
    CHECK_THAT(res.hotvar_errors[1],
               Catch::Contains(kDivideError.lammps_text) &&
                   !Catch::Contains("circular dependency"));
    CHECK(res.failvar_values == std::vector<std::string>{"0.25", ""});
    REQUIRE(res.failvar_errors.size() == 2);
    CHECK(res.failvar_errors[0].empty());
    CHECK_THAT(res.failvar_errors[1],
               Catch::Contains(kPressError.lammps_text) &&
                   !Catch::Contains("circular dependency"));
    CHECK(st.lmp_cache.count(slot_cache_key("rb", invar)) == 0);
}

/* Pins: a LAMMPS error inside the script (an unknown command) fails the
 * task with the LAMMPS message in error_msg -- "ERROR: Unknown command:
 * bogus_command 1 2", which the parent journals -- after the early failvar
 * is read, and the errored instance is discarded as before.
 * Before the fix: execute_file dropped e.what(), so error_msg was empty and
 * the journal had no line about the failure; the error_msg check fails. */
TEST_CASE("a LAMMPS error inside a PC_LAMMPS script reaches the task's "
          "error message", "[script_file][pc_task][readback]") {
    auto pot_ctx = kr_pot_ctx();
    tadah::hpo::PcSlotState<LammpsRunner> st;
    const std::vector<std::string> invar =
        {"lmp", "-log", "none", "-screen", "none"};
    const std::vector<std::string> bad = {
        "clear", "units lj", "variable viol equal 0.25", "bogus_command 1 2",
        "variable x equal 1.0"};
    auto res = tadah::hpo::execute_script_task<LammpsRunner>(
        "bad", bad, invar, {"x"}, {}, {"viol"}, pot_ctx, 1, {}, st);
    INFO("error_msg: " << res.error_msg);
    REQUIRE(res.status == tadah::hpo::PcTaskStatus::RUN_FAIL);
    CHECK_THAT(res.error_msg,
               Catch::StartsWith("ERROR: Unknown command: bogus_command 1 2"));
    CHECK(res.failvar_values == std::vector<std::string>{"0.25"});
    CHECK(st.lmp_cache.count(slot_cache_key("bad", invar)) == 0);
}

/* Pins: a fired timeout fails the task without a message (the section
 * guard's sentinel is not a LAMMPS message, and a timer that stops the last
 * section raises nothing), and keeps its instance policy: discarded after
 * the guard aborted Input::file() (TIMEOUT_ABORT), kept after a clean stop
 * in the last section (TIMEOUT_CLEAN). The early failvar is read in both.
 * Before the fix: passes as well (a pin that the in-script messages leave
 * timeouts as they were). */
TEST_CASE("a timed-out PC_LAMMPS script fails without a message and keeps "
          "its instance policy", "[script_file][pc_task][timeout]") {
    auto pot_ctx = kr_pot_ctx();
    const std::string melt = R"(clear
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
variable viol equal 0.25
run 500000000
)";
    struct Case { const char* name; std::string body; bool kept; };
    for (const Case& c : {
             Case{"two_sections", melt + "clear\nunits lj\nvariable late equal 1.0\n",
                  false},
             Case{"one_section", melt + "variable late equal 1.0\n", true}}) {
        INFO(c.name);
        /* Through LammpsSim, so the real preparation injects the timer
         * lines and, between sections, the guard. */
        const std::string spath = std::filesystem::temp_directory_path() /
            ("tadah_sfe_rbto_" + std::to_string(::getpid()) + ".in");
        { std::ofstream os(spath); os << c.body; }
        std::string cmd = "--script " + spath + " --varloss late 0.0 1.0";
        tadah::hpo::LammpsSim sim(cmd, "1", 1.0e6, tadah::LossFunction{});
        std::remove(spath.c_str());

        tadah::hpo::PcSlotState<LammpsRunner> st;
        const std::vector<std::string> invar =
            {"lmp", "-log", "none", "-screen", "none", "-var", "maxtime",
             "1 every 1"};
        const auto t0 = std::chrono::steady_clock::now();
        auto res = tadah::hpo::execute_script_task<LammpsRunner>(
            c.name, sim.script_lines, invar, {"late"}, {}, {"viol"}, pot_ctx,
            1, {}, st);
        const double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        INFO("error_msg: " << res.error_msg);
        CHECK(secs < 30.0);
        REQUIRE(res.status == tadah::hpo::PcTaskStatus::RUN_FAIL);
        CHECK(res.error_msg.empty());
        CHECK(res.failvar_values == std::vector<std::string>{"0.25"});
        CHECK(st.lmp_cache.count(slot_cache_key(c.name, invar)) ==
              (c.kept ? 1u : 0u));
    }
}

/* Pins: a --failvar that cannot be read after a run that failed (here a
 * timer that stopped the last section, which keeps a healthy instance) is
 * "", the others are read (viol = 0.25), the task fails as before without
 * a message, and the instance that raised the read-back error is
 * discarded, as after a read-back error on a run that succeeded.
 * Before the fix: the read's error was swallowed and the instance kept
 * (the cache check fails). */
TEST_CASE("a --failvar that cannot be read after a failed run discards the "
          "instance", "[script_file][pc_task][readback][timeout]") {
    auto pot_ctx = kr_pot_ctx();
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
variable viol equal 0.25
variable bad equal 1.0/(v_viol-v_viol)
run 500000000
variable late equal 1.0
)";
    const std::string spath = std::filesystem::temp_directory_path() /
        ("tadah_sfe_rbfv_" + std::to_string(::getpid()) + ".in");
    { std::ofstream os(spath); os << body; }
    std::string cmd = "--script " + spath + " --varloss late 0.0 1.0";
    tadah::hpo::LammpsSim sim(cmd, "1", 1.0e6, tadah::LossFunction{});
    std::remove(spath.c_str());

    tadah::hpo::PcSlotState<LammpsRunner> st;
    const std::vector<std::string> invar =
        {"lmp", "-log", "none", "-screen", "none", "-var", "maxtime",
         "1 every 1"};
    auto res = tadah::hpo::execute_script_task<LammpsRunner>(
        "timed_out", sim.script_lines, invar, {"late"}, {}, {"viol", "bad"},
        pot_ctx, 1, {}, st);
    INFO("error_msg: " << res.error_msg);
    REQUIRE(res.status == tadah::hpo::PcTaskStatus::RUN_FAIL);
    CHECK(res.error_msg.empty());
    CHECK(res.failvar_values == std::vector<std::string>{"0.25", ""});
    CHECK(st.lmp_cache.count(slot_cache_key("timed_out", invar)) == 0);
}

/* Pins: a hot value HPO injects before the script (format_hot_inject,
 * `variable hot_a0 equal 3.25`) that LAMMPS refuses fails the task: status
 * RUN_FAIL, error_msg "injecting `variable hot_a0 equal 3.25` before the
 * script: " and the LAMMPS message, its refusal to redefine hot_a0 ("ERROR:
 * Cannot redefine variable hot_a0 with a different style" in LAMMPS 11 Feb
 * 2026, "ERROR: Cannot redefine index style variable hot_a0 as equal style
 * variable" in LAMMPS 30 Sep 2026: the check takes the start and the name),
 * the script not run, nothing read back (no
 * outvar, hot or fail value: the instance's variables are an earlier
 * evaluation's), and the slot's instance discarded:
 *   - the name defined by `-var hot_a0 3.3` on the line (index style, at
 *     the instance's creation);
 *   - a cached instance whose script, in the evaluation before, redefined
 *     the name as a string: that evaluation is OK (E2 = 3.25), the next one
 *     fails the injection without reading the failvar viol its predecessor
 *     left (0.5), and the one after runs on a fresh instance (E2 = 3.5).
 * Before the fix: every injection error was swallowed and the script ran
 * on the value it had: status OK with E2 = 3.3 (the -var value), and on
 * the cached instance E2 = 3.3 (the script's own string) instead of the
 * injected 3.5. The status checks fail. */
TEST_CASE("a hot value LAMMPS refuses to inject fails the task with the "
          "message", "[script_file][pc_task][inject]") {
    auto pot_ctx = kr_pot_ctx();
    using tadah::hpo::PcTaskStatus;
    const std::vector<std::string> base =
        {"lmp", "-log", "none", "-screen", "none"};
    const std::string inject_325 = tadah::hpo::format_hot_inject("hot_a0", 3.25);
    const std::string inject_35 = tadah::hpo::format_hot_inject("hot_a0", 3.5);
    REQUIRE(inject_325 == "variable hot_a0 equal 3.25");
    REQUIRE(inject_35 == "variable hot_a0 equal 3.5");
    const std::string refused = "ERROR: Cannot redefine ";
    /* The LAMMPS message after `prefix`: a refusal to redefine hot_a0. */
    auto refusal_of_hot_a0 = [&](const std::string& msg,
                                 const std::string& prefix) {
        CHECK_THAT(msg, Catch::StartsWith(prefix + refused));
        CHECK_THAT(msg.substr(std::min(prefix.size(), msg.size())),
                   Catch::Contains("variable hot_a0 "));
    };

    SECTION("the name defined by -var on the line") {
        tadah::hpo::PcSlotState<LammpsRunner> st;
        std::vector<std::string> invar = base;
        invar.insert(invar.end(), {"-var", "hot_a0", "3.3"});
        const auto res = tadah::hpo::execute_script_task<LammpsRunner>(
            "cold", {"clear", "units metal", "variable viol equal 0.5",
                     "variable E2 equal ${hot_a0}"},
            invar, {"E2"}, {}, {"viol"}, pot_ctx, 1, {inject_325}, st);
        INFO("error_msg: " << res.error_msg);
        INFO("outvar values: " << res.outvar_values.size()
             << (res.outvar_values.empty() ? std::string()
                                           : ", E2 = " + res.outvar_values[0]));
        REQUIRE(res.status == PcTaskStatus::RUN_FAIL);
        refusal_of_hot_a0(res.error_msg,
                          "injecting `variable hot_a0 equal 3.25` "
                          "before the script: ");
        CHECK(res.outvar_values.empty());
        CHECK(res.hotvar_values.empty());
        CHECK(res.failvar_values.empty());
        CHECK(st.lmp_cache.count(slot_cache_key("cold", invar)) == 0);
    }
    SECTION("a cached instance whose script redefined the name") {
        tadah::hpo::PcSlotState<LammpsRunner> st;
        const std::vector<std::string> lines = {
            "clear", "units metal", "variable viol equal 0.5",
            "variable E2 equal ${hot_a0}", "variable hot_a0 delete",
            "variable hot_a0 string 3.3"};
        auto first = tadah::hpo::execute_script_task<LammpsRunner>(
            "cold", lines, base, {"E2"}, {}, {"viol"}, pot_ctx, 1,
            {inject_325}, st);
        INFO("first error_msg: " << first.error_msg);
        REQUIRE(first.status == PcTaskStatus::OK);
        REQUIRE(first.outvar_values.size() == 1);
        CHECK(first.outvar_values[0] == "3.25");
        CHECK(st.lmp_cache.count(slot_cache_key("cold", base)) == 1);

        auto second = tadah::hpo::execute_script_task<LammpsRunner>(
            "cold", lines, base, {"E2"}, {}, {"viol"}, pot_ctx, 2,
            {inject_35}, st);
        INFO("second error_msg: " << second.error_msg);
        INFO("second outvar values: " << second.outvar_values.size()
             << (second.outvar_values.empty()
                     ? std::string()
                     : ", E2 = " + second.outvar_values[0]));
        REQUIRE(second.status == PcTaskStatus::RUN_FAIL);
        refusal_of_hot_a0(second.error_msg,
                          "injecting `variable hot_a0 equal 3.5` "
                          "before the script: ");
        CHECK(second.outvar_values.empty());
        CHECK(second.failvar_values.empty());
        CHECK(st.lmp_cache.count(slot_cache_key("cold", base)) == 0);

        auto third = tadah::hpo::execute_script_task<LammpsRunner>(
            "cold", lines, base, {"E2"}, {}, {"viol"}, pot_ctx, 3,
            {inject_35}, st);
        INFO("third error_msg: " << third.error_msg);
        REQUIRE(third.status == PcTaskStatus::OK);
        REQUIRE(third.outvar_values.size() == 1);
        CHECK(third.outvar_values[0] == "3.5");
        CHECK(third.failvar_values == std::vector<std::string>{"0.5"});
    }
}

/* Pins: a LAMMPS instance that cannot be created for the line's
 * command-line options (here the typo -sff) fails the task as an
 * infrastructure failure, CREATE_FAIL, with error_msg "creating the LAMMPS
 * instance: " and LAMMPS' own message ("ERROR: Invalid command-line
 * argument: -sff"), which the parent journals; no cache entry is left, so
 * the next task retries the creation.
 * Before the fix: LammpsRunner::create_instance dropped LAMMPS' exception,
 * so error_msg was empty and every evaluation failed with nothing in the
 * journal; the error_msg check fails. */
TEST_CASE("a LAMMPS instance that cannot be created fails the task with "
          "LAMMPS' message", "[script_file][pc_task][create]") {
    auto pot_ctx = kr_pot_ctx();
    tadah::hpo::PcSlotState<LammpsRunner> st;
    const std::vector<std::string> invar =
        {"lmp", "-log", "none", "-screen", "none", "-sff", "omp"};
    const auto res = tadah::hpo::execute_script_task<LammpsRunner>(
        "sff", readback_lines("pe"), invar, {"E"}, {}, {}, pot_ctx, 1, {}, st);
    INFO("error_msg: " << res.error_msg);
    REQUIRE(res.status == tadah::hpo::PcTaskStatus::CREATE_FAIL);
    CHECK_THAT(res.error_msg,
               Catch::StartsWith("creating the LAMMPS instance: ERROR: "
                                 "Invalid command-line argument: -sff"));
    CHECK(st.lmp_cache.empty());
}

namespace {

/* TMPDIR for the life of the object, restored (or unset) after. */
struct ScopedTmpdir {
    std::optional<std::string> old;
    explicit ScopedTmpdir(const std::string& value) {
        if (const char* v = std::getenv("TMPDIR")) old = std::string(v);
        ::setenv("TMPDIR", value.c_str(), 1);
    }
    ~ScopedTmpdir() {
        if (old) ::setenv("TMPDIR", old->c_str(), 1);
        else     ::unsetenv("TMPDIR");
    }
};

}  // namespace

/* Pins: a script that cannot be rendered to its temporary file -- TMPDIR
 * names a regular file, so there is no temporary directory -- fails the
 * task as an infrastructure failure, CREATE_FAIL, with error_msg
 * "rendering the script to a temporary file: " and the reason, which the
 * parent journals; no cache entry is left.
 * Before the fix: error_msg was the bare reason (the parent never
 * journalled it, which [pc_readback_e2e] pins); the StartsWith check
 * fails. */
TEST_CASE("a PC_LAMMPS script that cannot be rendered fails the task with "
          "the reason", "[script_file][pc_task][create]") {
    auto pot_ctx = kr_pot_ctx();
    const std::filesystem::path not_a_dir =
        std::filesystem::temp_directory_path() /
        ("tadah_sfe_notadir_" + std::to_string(::getpid()));
    std::ofstream(not_a_dir) << "a file, not a directory\n";
    tadah::hpo::PcSlotState<LammpsRunner> st;
    const std::vector<std::string> invar =
        {"lmp", "-log", "none", "-screen", "none"};
    tadah::hpo::PcTaskResult res;
    {
        ScopedTmpdir tmpdir(not_a_dir.string());
        /* A text no live rendering has: identical renderings are shared. */
        std::vector<std::string> lines = readback_lines("pe");
        lines.push_back("variable unrendered_" + std::to_string(::getpid()) +
                        " equal 1");
        res = tadah::hpo::execute_script_task<LammpsRunner>(
            "unrendered", lines, invar, {"E"}, {}, {}, pot_ctx, 1, {}, st);
    }
    std::filesystem::remove(not_a_dir);
    INFO("error_msg: " << res.error_msg);
    REQUIRE(res.status == tadah::hpo::PcTaskStatus::CREATE_FAIL);
    CHECK_THAT(res.error_msg,
               Catch::StartsWith("rendering the script to a temporary file: "));
    CHECK(res.error_msg.size() >
          std::string("rendering the script to a temporary file: ").size());
    CHECK(st.lmp_cache.empty());
}

namespace {

/* A runner whose run_script throws an exception that is not a LAMMPS error
 * (none of LammpsRunner's members does: they catch LAMMPS errors). It has
 * the shape execute_script_task needs of a runner, and an instance whose
 * input takes the injected lines. */
struct ThrowingRunner {
    struct Input { void one(const char*) {} };
    struct Instance {
        Input in;
        Input* input = &in;
    };
    using LMPPtr = std::unique_ptr<Instance>;
    LMPPtr create_instance(const std::vector<std::string>&,
                           std::string* = nullptr) const {
        return std::make_unique<Instance>();
    }
    bool run_script(const LMPPtr&, const ScriptFile&,
                    const std::shared_ptr<tadah::LammpsTadah>&,
                    ScriptRunStatus* = nullptr,
                    std::string* = nullptr) const {
        throw std::runtime_error("ThrowingRunner: not a LAMMPS error");
    }
    std::string get_variable(const LMPPtr&, const std::string&) const {
        return "1";
    }
};

}  // namespace

/* Pins: an exception that escapes the task's own handling -- here one
 * thrown by the runner, which no LammpsRunner member does -- fails that
 * task: RUN_FAIL with the exception's message, and the slot's instance
 * discarded. THREAD's run_one and the PROCESS worker both call
 * execute_script_task, so such a throw fails one task the same way in both
 * modes.
 * Before the fix: the exception escaped execute_script_task. Under PROCESS
 * the worker's own catch-all made it a RUN_FAIL, keeping the instance;
 * under THREAD it ended the whole evaluation through its catch (counted as
 * a NaN clamp). The test fails with the unexpected exception. */
TEST_CASE("an exception inside a PC_LAMMPS task fails that task with its "
          "message", "[pc_task][parity]") {
    auto pot_ctx = kr_pot_ctx();
    tadah::hpo::PcSlotState<ThrowingRunner> st;
    const std::vector<std::string> invar =
        {"lmp", "-log", "none", "-screen", "none"};
    tadah::hpo::PcTaskResult res;
    REQUIRE_NOTHROW(res = tadah::hpo::execute_script_task<ThrowingRunner>(
        "throws", readback_lines("pe"), invar, {"E"}, {}, {}, pot_ctx, 1,
        {"variable seed equal 1.0"}, st));
    CHECK(res.status == tadah::hpo::PcTaskStatus::RUN_FAIL);
    CHECK(res.error_msg == "ThrowingRunner: not a LAMMPS error");
    CHECK(res.outvar_values.empty());
    CHECK(st.lmp_cache.empty());
}

/* Pins: a SCRIPT_ERROR of LammpsRunner::run_script (the Context overload,
 * which builds the potential itself for an EXTERN script) hands its reason
 * to the error out-parameter: "potential construction failed: " and the
 * construction error for a Context no potential can be built from, and
 * the LAMMPS message for an error inside the script. A run that succeeds
 * leaves the parameter alone.
 * Added with the fix (an exception to tests-first): the out-parameter does
 * not exist before it, so this does not compile there. Before the fix
 * both catches dropped e.what(). */
TEST_CASE("run_script hands the reason of a script error to its error "
          "out-parameter", "[script_file][readback]") {
    tadah::core::PeriodicTable::initialize();
    LammpsRunner runner;

    SECTION("an EXTERN script whose potential cannot be built") {
        auto lmp = make_instance();
        REQUIRE(lmp != nullptr);
        const ScriptFile sf(kr_loop_extern_lines());
        REQUIRE(sf.has_extern());
        ScriptRunStatus why = ScriptRunStatus::OK;
        std::string error;
        CHECK_FALSE(runner.run_script(lmp, sf, tadah::core::Context{}, &why,
                                      &error));
        CHECK(why == ScriptRunStatus::SCRIPT_ERROR);
        CHECK_THAT(error, Catch::StartsWith("potential construction failed: "));
        CHECK(error.size() > std::string("potential construction failed: ").size());
    }
    SECTION("an unknown command") {
        auto lmp = make_instance();
        REQUIRE(lmp != nullptr);
        const ScriptFile sf(std::vector<std::string>{
            "clear", "units lj", "bogus_command 1 2"});
        ScriptRunStatus why = ScriptRunStatus::OK;
        std::string error;
        CHECK_FALSE(runner.run_script(lmp, sf, tadah::core::Context{}, &why,
                                      &error));
        CHECK(why == ScriptRunStatus::SCRIPT_ERROR);
        CHECK_THAT(error,
                   Catch::StartsWith("ERROR: Unknown command: bogus_command 1 2"));
    }
    SECTION("a script that runs") {
        auto lmp = make_instance();
        REQUIRE(lmp != nullptr);
        const ScriptFile sf(backward_loop_lines());
        ScriptRunStatus why = ScriptRunStatus::SCRIPT_ERROR;
        std::string error = "untouched";
        CHECK(runner.run_script(lmp, sf, tadah::core::Context{}, &why, &error));
        CHECK(why == ScriptRunStatus::OK);
        CHECK(error == "untouched");
    }
}
