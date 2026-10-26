// test_hpo_pc_lines_e2e.cpp
//
// The PC lines of a target file in evaluations, end to end: an HPO_Host
// in-process (PC_ISOLATION THREAD) on a small Ta training set, its
// candidates evaluated one by one. Hidden ([pc_lines_e2e][.]; CTest entry
// hpo_pc_lines_e2e, label e2e, one process): the HPO_Host statics, the
// journal and the warm-start cache are shared by the cases of a process.
//
//   - a failure signal fails its evaluation under FAILSCORE_GRADING OFF:
//     a --failvar above 0, a PC_FAILPROBE S above --gate;
//   - the a0 a PC_OBSERVABLE cohesive_energy relaxation starts from is
//     that of the chain's own previous evaluation;
//   - a point evaluated again that gives another loss is journalled and
//     counted in DIAGNOSTICS;
//   - PC_ERSQ is refused for a validation dataset whose energies per atom
//     are all equal (its R^2 has no value).

#include "tadah_test_skip.hpp"  // before catch2/catch.hpp

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/pc_fail_grading.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/warmstart_cache.h>

#include <dlib/matrix.h>

#include <unistd.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace {

namespace fs = std::filesystem;
using ParamVec = dlib::matrix<double, 0, 1>;
using Host     = tadah::hpo::HPO_Host<ParamVec, ::LammpsRunner>;
using tadah::core::Context;
using tadah::hpo::HPOLog;
using tadah::hpo::HPOTargets;

std::string unique_name(const std::string& stem) {
    return stem + std::to_string(::getpid()) + "_" +
           std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count());
}

/* A temporary working directory, the cwd while the object lives (HPO_Host
 * writes its row files to the cwd). */
struct ScratchDir {
    fs::path dir, original_cwd;
    explicit ScratchDir(const std::string& stem) {
        dir = fs::temp_directory_path() / unique_name(stem);
        fs::create_directories(dir);
        original_cwd = fs::current_path();
        fs::current_path(dir);
    }
    ~ScratchDir() {
        std::error_code ec;
        fs::current_path(original_cwd);
        fs::remove_all(dir, ec);
    }
};

/* hpo_run.log in a temporary file, read back after finish(). */
struct Journal {
    fs::path path;
    Journal() : path(fs::temp_directory_path() /
                     (unique_name("tadah_pc_lines_log_") + ".log")) {
        if (HPOLog::instance().is_open()) HPOLog::instance().finish("closed");
        HPOLog::instance().reset_for_test();
        HPOLog::instance().open(path.string(), "test", "test_cwd", 0);
    }
    ~Journal() {
        if (HPOLog::instance().is_open()) HPOLog::instance().finish("closed");
        HPOLog::instance().reset_for_test();
        std::error_code ec;
        fs::remove(path, ec);
    }
    std::string finish() {
        HPOLog::instance().finish("TEST");
        std::ifstream is(path);
        std::ostringstream oss;
        oss << is.rdbuf();
        return oss.str();
    }
};

/* One bcc Ta cell (2 atoms) of lattice constant @p a0 and energy @p energy
 * (eV), in the format of a dataset file. */
std::string bcc_entry(double a0, double energy) {
    std::ostringstream os;
    os.precision(17);
    os << "bcc_ta_" << a0 << "\n1.0 1.0 1.0\n" << energy << " 0.0\n"
       << a0 << " 0.0 0.0\n0.0 " << a0 << " 0.0\n0.0 0.0 " << a0 << "\n"
       << "0.0 0.0 0.0\n0.0 0.0 0.0\n0.0 0.0 0.0\n"
       << "Ta 0.0 0.0 0.0  0.0 0.0 0.0\n"
       << "Ta " << 0.5 * a0 << " " << 0.5 * a0 << " " << 0.5 * a0
       << "  0.0 0.0 0.0\n\n";
    return os.str();
}

/* The training set: bcc cells around a0 = 3.3, energies on a parabola,
 * so that the fitted potential has a minimum near 3.3. */
void write_train(const fs::path& path) {
    std::ofstream os(path);
    for (const double a : {3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6})
        os << bcc_entry(a, 2.0 * (4.0 * (a - 3.3) * (a - 3.3) - 8.0));
}

/* Ta D2_BP (2 functions) + BF_Linear, the model of the hpotarget's OPTIM
 * CGRID2B (1). */
Context make_ta_context(const fs::path& dbfile, const fs::path& hpo_cfg,
                        const fs::path& val_cfg) {
    Context c;
    c.add("DBFILE", dbfile.string());
    c.add("ATOM", "Ta");
    c.add("WATOM", 1);
    c.add("INIT2B", "true");
    c.add("INITMB", "false");
    c.add("TYPE2B", "D2_BP");
    c.add("TYPE2B", 2);
    c.add("TYPE2B", 2);
    c.add("TYPE2B", "Ta");
    c.add("TYPE2B", "Ta");
    c.add("CGRID2B", "2.5 3.5");
    c.add("SGRID2B", "2.0 3.0");
    c.add("RCTYPE2B", "Cut_Cos");
    c.add("RCUT2B", 4.5);
    c.add("TYPEMB", "DM_Dummy");
    c.add("TYPEMB", "Ta");
    c.add("TYPEMB", "Ta");
    c.add("RCTYPEMB", "Cut_Cos");
    c.add("RCUTMB", 4.5);
    c.add("DIMER", "false");
    c.add("DIMER", 0);
    c.add("DIMER", "false");
    c.add("FORCE", "true");
    c.add("STRESS", "false");
    c.add("ESIGMA", 1.0);
    c.add("FSIGMA", 0.1);
    c.add("FSIGMA", 0.0);
    c.add("SSIGMA", 0.5);
    c.add("SSIGMA", 0.0);
    c.add("BIAS", "false");
    c.add("NORM", "false");
    c.add("VERBOSE", 0);
    c.add("MODEL", "M_BLR");
    c.add("MODEL", "BF_Linear");
    c.add("LAMBDA", 1e-3);
    c.add("OALGO", 4);
    c.add("HPOTARGET", hpo_cfg.string());
    c.add("VALIDATION", val_cfg.string());
    return c;
}

/* A bcc Ta cell run once with the candidate potential, then @p tail. */
std::string ta_script(const std::string& tail) {
    return "clear\nunits metal\nboundary p p p\natom_style atomic\n"
           "lattice bcc 3.3\nregion box block 0 2 0 2 0 2 units lattice\n"
           "create_box 1 box\ncreate_atoms 1 region box\nmass 1 180.95\n"
           "pair_style tadah\npair_coeff * * EXTERN Ta\n"
           "thermo_style custom step pe\nrun 0 post no\n" + tail;
}

const std::string kHead =
    "OPTIMIZER\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 1\nENDOPTIMIZER\n"
    "QFLUSH 1\nFAILSCORE 1e16\nOPTIM CGRID2B (1) 2.0 3.0\n";

/* One HPO_Host in the cwd with the hpotarget kHead + @p tail (the
 * validation set: @p val_entries, one dataset); evaluates each candidate
 * CGRID2B[1] of @p xs on the chain @p chains[i] (0 when not given) and
 * returns the losses. */
std::vector<double> evaluate(const fs::path& dir, const std::string& tail,
                             const std::vector<double>& xs,
                             const std::vector<std::size_t>& chains = {},
                             const std::string& val_entries =
                                 bcc_entry(3.25, -15.9) + bcc_entry(3.35, -15.95)) {
    const fs::path train = dir / "train.db", val = dir / "config.val",
                   valdb = dir / "val.db", hpo = dir / "config.hpo";
    write_train(train);
    std::ofstream(valdb) << val_entries;
    std::ofstream(val) << "DBFILE " << valdb.string() << "\n";
    std::ofstream(hpo) << kHead << tail;
    Context ctx = make_ta_context(train, hpo, val);
    HPOTargets trg(ctx, hpo.string());
    Host host(ctx, trg, val.string());
    host.prepare_best_publication();
    std::vector<double> losses;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        tadah::hpo::warmstart::current_chain() =
            i < chains.size() ? chains[i] : 0;
        ParamVec p(1);
        p(0) = xs[i];
        double loss = 0.0;
        REQUIRE_NOTHROW(loss = host(p));
        losses.push_back(loss);
    }
    tadah::hpo::warmstart::current_chain() = 0;
    host.flush_remaining();
    return losses;
}

/* The journal line that contains @p needle ("" when none does). */
std::string line_with(const std::string& text, const std::string& needle) {
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line))
        if (line.find(needle) != std::string::npos) return line;
    return {};
}

/* The bits of @p x, to compare two losses exactly (-ffast-math). */
std::uint64_t bits(double x) {
    std::uint64_t b = 0;
    std::memcpy(&b, &x, sizeof b);
    return b;
}

}  // namespace

/* Pins: under FAILSCORE_GRADING OFF (the default) a --failvar variable
 * above 0 after the script fails the evaluation: it scores FAILSCORE
 * (1e16), and the journal says "PC_LAMMPS --failvar above 0 (<dir>/fail.in,
 * occurrence 1): bad = 1"; with bad = 0 the evaluation is usable. A
 * PC_FAILPROBE S above --gate fails it too (S = 1, --gate 0.5), journalled
 * as a FAILPROBE gate; S below --gate (--gate 2) does not.
 * Before the fix: under OFF neither signal was read, and every evaluation
 * kept its usable loss. */
TEST_CASE("a failure signal fails its evaluation under FAILSCORE_GRADING OFF",
          "[pc_lines_e2e][.]") {
    tadah::core::PeriodicTable::initialize();
    for (const char* bad : {"1", "0"}) {
        INFO("bad = " << bad);
        Host::reset_globals_for_test();
        ScratchDir sd("tadah_pc_lines_failvar_");
        std::ofstream(sd.dir / "fail.in")
            << ta_script(std::string("variable E equal pe\n"
                                     "variable bad equal ") + bad + "\n");
        Journal j;
        const auto losses = evaluate(
            sd.dir, "PC_ERMSE 0 1\nPC_LAMMPS --script " +
                        (sd.dir / "fail.in").string() +
                        " --varloss E 0 1 --failvar bad\n",
                    {2.5, 2.6});
        const std::string journal = j.finish();
        INFO(journal);
        for (const double l : losses) {
            if (std::string(bad) == "1") CHECK(l == 1e16);
            else CHECK(tadah::hpo::loss_is_feasible(l, 1e16));
        }
        const std::string line =
            line_with(journal, "fail.in, occurrence 1): bad = 1");
        if (std::string(bad) == "1")
            CHECK_THAT(line, Catch::Contains("PC_LAMMPS --failvar above 0 ("));
        else CHECK(journal.find("--failvar above 0") == std::string::npos);
        Host::reset_globals_for_test();
    }
    for (const char* gate : {"0.5", "2"}) {
        INFO("--gate " << gate);
        Host::reset_globals_for_test();
        ScratchDir sd("tadah_pc_lines_gate_");
        std::ofstream(sd.dir / "probe.in")
            << ta_script("variable S equal 1.0\n");
        std::ofstream(sd.dir / "e.in") << ta_script("variable E equal pe\n");
        Journal j;
        const auto losses = evaluate(
            sd.dir, "PC_ERMSE 0 1\nPC_LAMMPS --script " +
                        (sd.dir / "e.in").string() +
                        " --varloss E 0 1\nPC_FAILPROBE --script " +
                        (sd.dir / "probe.in").string() +
                        " --outvar S --gate " + gate + "\n",
                    {2.5, 2.6});
        const std::string journal = j.finish();
        INFO(journal);
        for (const double l : losses) {
            if (std::string(gate) == "0.5") CHECK(l == 1e16);
            else CHECK(tadah::hpo::loss_is_feasible(l, 1e16));
        }
        CHECK((journal.find("FAILPROBE gate: S=1.000000 > 0.500000") !=
               std::string::npos) == (std::string(gate) == "0.5"));
        Host::reset_globals_for_test();
    }
}

/* Pins: the relaxation of a PC_OBSERVABLE cohesive_energy term starts from
 * the a0 that the previous evaluation of its own chain relaxed to, and the
 * first evaluation of a chain from the declared --hotvar start. The
 * evaluations (chain 0, x1), (chain 0, x2), (chain 1, x1), (chain 0, x1)
 * give, bit for bit, the losses of a run with one chain: the third that
 * of the first evaluation of x1 (from the declared start), the fourth
 * that of x1 after x2 (from the a0 of x2), which differs from it (the
 * start matters).
 * Before the fix: one a0 was shared by every chain, each evaluation
 * starting where the last one of any chain ended: the third evaluation
 * started from the a0 of x2, and its loss was that of x1 after x2. */
TEST_CASE("a PC_OBSERVABLE relaxation starts from the a0 of its own chain",
          "[pc_lines_e2e][.]") {
    tadah::core::PeriodicTable::initialize();
    const std::string obs =
        "PC_ERMSE 0 1\nPC_OBSERVABLE cohesive_energy --species Ta --target "
        "8.1 --weight 1 --supercell 2 2 2 --hotvar a0 2.8 3.8 3.1 "
        "--maxiter 500 --maxeval 500\n";
    const double x1 = 2.4, x2 = 2.8;
    std::vector<double> one;
    {
        Host::reset_globals_for_test();
        ScratchDir sd("tadah_pc_lines_chain_one_");
        Journal j;
        one = evaluate(sd.dir, obs, {x1, x2, x1});
        (void)j.finish();
        Host::reset_globals_for_test();
    }
    Host::reset_globals_for_test();
    ScratchDir sd("tadah_pc_lines_chain_");
    Journal j;
    const auto l = evaluate(sd.dir, obs, {x1, x2, x1, x1}, {0, 0, 1, 0});
    const std::string journal = j.finish();
    INFO(journal);
    REQUIRE(l.size() == 4u);
    REQUIRE(one.size() == 3u);
    INFO(std::hexfloat << "losses " << l[0] << " " << l[1] << " " << l[2]
                       << " " << l[3] << "; one chain " << one[0] << " "
                       << one[1] << " " << one[2]);
    for (const double v : l) REQUIRE(tadah::hpo::loss_is_feasible(v, 1e16));
    /* Precondition: the start of x1's relaxation changes its loss. */
    REQUIRE(bits(one[0]) != bits(one[2]));
    CHECK(bits(l[0]) == bits(one[0]));
    CHECK(bits(l[1]) == bits(one[1]));
    CHECK(bits(l[2]) == bits(one[0]));
    CHECK(bits(l[3]) == bits(one[2]));
    Host::reset_globals_for_test();
}

/* Pins: a point evaluated again whose loss differs from its earlier
 * evaluation's (relative difference above 1e-9) is journalled -- "eval 2
 * evaluated the point of eval 1 again and its loss differs: " both losses,
 * and why two evaluations of one point can differ (the PC_LAMMPS reason)
 * -- and counted in DIAGNOSTICS ("Repeats with another loss : 1"); a
 * point evaluated again with the same loss is neither. The script counts
 * its runs in its LAMMPS instance, which keeps its variables from one
 * evaluation to the next.
 * Before the fix: nothing said so; the run's record kept the later loss
 * without a word. */
TEST_CASE("a point evaluated again with another loss is journalled and "
          "counted", "[pc_lines_e2e][.]") {
    tadah::core::PeriodicTable::initialize();
    for (const bool counts : {true, false}) {
        INFO("the script counts its runs: " << counts);
        Host::reset_globals_for_test();
        ScratchDir sd("tadah_pc_lines_repeat_");
        std::ofstream(sd.dir / "count.in")
            << (counts ? "variable n loop 1000\nnext n\n"
                         "variable E equal v_n\n"
                       : "variable E equal 3\n");
        Journal j;
        const auto losses = evaluate(
            sd.dir, "PC_ERMSE 0 1\nPC_LAMMPS --script " +
                        (sd.dir / "count.in").string() + " --varloss E 0 1\n",
            {2.5, 2.5});
        const std::string journal = j.finish();
        INFO(journal);
        REQUIRE(losses.size() == 2u);
        for (const double v : losses)
            REQUIRE(tadah::hpo::loss_is_feasible(v, 1e16));
        REQUIRE((losses[0] != losses[1]) == counts);
        const std::string line = line_with(
            journal, "eval 2 evaluated the point of eval 1 again and its loss "
                     "differs: ");
        const std::string diag =
            line_with(journal, "Repeats with another loss");
        if (counts) {
            CHECK_THAT(line, Catch::Contains("WARNING") &&
                             Catch::Contains("PC_LAMMPS: its scripts run on "
                                             "LAMMPS instances kept"));
            CHECK_THAT(diag, Catch::Contains(": 1 "));
        } else {
            CHECK(line.empty());
            CHECK(diag.empty());
        }
        Host::reset_globals_for_test();
    }
}

/* Pins: PC_ERSQ, the R^2 of the energies per atom of each validation
 * dataset, is refused when the host loads a validation dataset whose
 * energies per atom are all equal (one structure here): its R^2 has no
 * value. The message names the dataset. Two structures of different
 * energies per atom are accepted.
 * Before the fix: the host was built, and every evaluation produced a
 * non-finite R^2 (-inf) and failed. */
TEST_CASE("PC_ERSQ is refused for a validation dataset of equal energies",
          "[pc_lines_e2e][.]") {
    tadah::core::PeriodicTable::initialize();
    Host::reset_globals_for_test();
    ScratchDir sd("tadah_pc_lines_ersq_");
    const fs::path train = sd.dir / "train.db", val = sd.dir / "config.val",
                   valdb = sd.dir / "one.db", hpo = sd.dir / "config.hpo";
    write_train(train);
    std::ofstream(valdb) << bcc_entry(3.3, -16.0);
    std::ofstream(val) << "DBFILE " << valdb.string() << "\n";
    std::ofstream(hpo) << kHead << "PC_ERSQ 1 1\n";
    Context ctx = make_ta_context(train, hpo, val);
    HPOTargets trg(ctx, hpo.string());
    CHECK_THROWS_WITH(
        Host(ctx, trg, val.string()),
        Catch::StartsWith(
            "tadah hpo: PC_ERSQ on the validation set: the energies per atom "
            "of validation dataset 1 (" + valdb.string() + ") are all equal, "
            "so its R^2 has no value; PC_ERSQ needs two different energies "
            "per atom in every validation dataset it measures"));
    std::ofstream(valdb) << bcc_entry(3.3, -16.0) << bcc_entry(3.4, -15.9);
    HPOTargets trg2(ctx, hpo.string());
    CHECK_NOTHROW(Host(ctx, trg2, val.string()));
    Host::reset_globals_for_test();
}
