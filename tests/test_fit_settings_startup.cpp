// test_fit_settings_startup.cpp
//
// Settings of the fit that every evaluation of tadah hpo would refuse are
// refused when the run starts, before any evaluation, instead of failing
// each evaluation (FAILSCORE): the run used to start, score FAILSCORE for
// every candidate and end without a potential.
//
//  - LAMBDA of the training configuration: a negative value other than -1,
//    a second value, -1 under an OALGO other than 3, and -1 under M_KRR
//    implementation 2, with the message of the fit; an M_KRR
//    implementation other than 1 and 2, with the message of the fit. A
//    run whose
//    evaluations fit nothing (no training data, FIXINDEX pinning every
//    weight) reads no LAMBDA, and is not refused for it.
//  - An OPTIM box of EWEIGHT, FWEIGHT, SWEIGHT or LAMBDA that reaches below
//    0, at its line: a fit refuses a negative weight and a negative LAMBDA
//    other than -1, and -1 (the evidence approximation) is no value of a
//    box. OPTIM LAMBDA when the training configuration has LAMBDA -1, at
//    its line: the search would replace the evidence approximation.
//  - Under M_KRR implementation 2, OPTIM of EWEIGHT, FWEIGHT, SWEIGHT or
//    WDBFILE, which its solve ignores, with the message of the fit (MLIP
//    krr_implementation2_ignores); the settings and the training data it
//    ignores (MLIP refuse_what_krr_implementation2_ignores), and a weight
//    that is not a finite number >= 0 (MLIP require_valid_weights), once
//    the training data are read (tadah hpo --check reads them too).
//  - OPTIM of a key applied once when the run starts (LSCALE, ESHIFT,
//    ESHIFT_ATOM, ESHIFT_DBATOM, EFILTER, FFILTER, ZERO_COM_FORCE), at its
//    line: no evaluation would take a candidate value of it.
//
// The evidence approximation (LAMBDA -1, OALGO 3) of a fit that stops at
// its iteration limit, or that cannot estimate the prior and the noise
// from the targets of a candidate, is said once per run in the journal,
// hpo_run.log, and not on standard output for each evaluation; a candidate
// whose evidence approximation is refused fails (FAILSCORE, a failed
// evaluation), without the WARNING of an evaluation that threw.
//
// The model is a one-species (H) D2_BP pair term with BF_Linear (two
// weights); the data are H2 cells.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/startup_validation.h>
#include <tadah/core/element.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/data_weights.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>

#include <dlib/matrix.h>

#include <chrono>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using Catch::Matchers::Contains;
using tadah::core::Context;
using tadah::hpo::HPOTargets;
using ParamVec = dlib::matrix<double, 0, 1>;
using Host     = tadah::hpo::HPO_Host<ParamVec, ::LammpsRunner>;

struct TargetFile {
    fs::path path;
    explicit TargetFile(const std::string& body)
    : path(fs::temp_directory_path() /
           ("tadah_fit_settings_" +
            std::to_string(std::chrono::steady_clock::now()
                               .time_since_epoch().count()) + ".hpo")) {
        std::ofstream os(path);
        REQUIRE(os.good());
        os << body;
    }
    ~TargetFile() {
        std::error_code ec;
        fs::remove(path, ec);
    }
};

/* A scratch working directory: hpo_run() writes every file into the
 * current directory. */
struct ScratchDir {
    fs::path dir, original;
    ScratchDir() {
        dir = fs::temp_directory_path() /
              ("tadah_fit_settings_run_" +
               std::to_string(std::chrono::steady_clock::now()
                                  .time_since_epoch().count()));
        fs::create_directories(dir);
        original = fs::current_path();
        fs::current_path(dir);
    }
    ~ScratchDir() {
        std::error_code ec;
        fs::current_path(original, ec);
        fs::remove_all(dir, ec);
    }
};

/* Standard output into a string while it lives. */
struct CaptureCout {
    std::ostringstream text;
    std::streambuf* old;
    CaptureCout() : old(std::cout.rdbuf(text.rdbuf())) {}
    ~CaptureCout() { std::cout.rdbuf(old); }
    CaptureCout(const CaptureCout&) = delete;
    CaptureCout& operator=(const CaptureCout&) = delete;
};

void set_key(Context& c, const std::string& key, const std::string& value) {
    if (c.has_stored(key)) c.remove(key);
    c.add(key, value);
}

/* H, a D2_BP pair term of two functions, BF_Linear: two weights. */
Context model_ctx() {
    Context c;
    c.add("ATOM", "H");
    c.add("WATOM", 1.0);
    c.add("INIT2B", "true");
    c.add("INITMB", "false");
    c.add("TYPE2B", "D2_BP 2 2 H H");
    c.add("TYPEMB", "DM_Dummy H H");
    c.add("CGRID2B", "1.0 2.0");
    c.add("SGRID2B", "0.5 1.0");
    c.add("RCTYPE2B", "Cut_Cos");
    c.add("RCTYPEMB", "Cut_Cos");
    c.add("RCUT2B", 5.0);
    c.add("RCUTMB", 5.0);
    c.add("DIMER", "false 0 false");
    c.add("FORCE", "false");
    c.add("STRESS", "false");
    c.add("EWEIGHT", 1.0);
    c.add("FWEIGHT", 1.0);
    c.add("SWEIGHT", 1.0);
    c.add("BIAS", "false");
    c.add("NORM", "false");
    c.add("VERBOSE", 0);
    c.add("MODEL", "M_BLR BF_Linear");
    c.add("LAMBDA", 1e-3);
    c.add("OALGO", 4);
    return c;
}

Context krr2_ctx() {
    Context c = model_ctx();
    set_key(c, "MODEL", "M_KRR Kern_Linear 2");
    c.add("SBASIS", 2);
    return c;
}

const std::string kOptimizer =
    "OPTIMIZER\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 3\nENDOPTIMIZER\n";

/* A target file: @p lines, PC_ERMSE and an optimiser; OPTIM CGRID2B (1)
 * unless @p lines has an OPTIM line. */
std::string target(const std::string& lines = "") {
    const bool optim = lines.find("OPTIM ") != std::string::npos;
    return (optim ? "" : "OPTIM CGRID2B (1) 0.5 1.5\n") + lines +
           "PC_ERMSE 0 1\n" + kOptimizer;
}

/* What reading the target file throws, "" when it is read. */
std::string parse_error(Context c, const std::string& text) {
    TargetFile tf(text);
    try {
        HPOTargets trg(c, tf.path.string());
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

/* What the startup checks throw (the target file is read first), "" when
 * they pass. */
std::string startup_error(Context c, const std::string& text,
                          bool has_train_db = true) {
    if (has_train_db && !c.has_stored("DBFILE")) c.add("DBFILE", "dummy.db");
    TargetFile tf(text);
    try {
        HPOTargets trg(c, tf.path.string());
        tadah::hpo::validate_hpo_setup(c, trg, has_train_db, true);
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

/* What a fit of @p c refuses on four H2 cells, "" when it fits. */
std::string fit_refusal(Context c) {
    tadah::mlip::StructureDB db;
    for (std::size_t i = 0; i < 4; ++i) {
        tadah::mlip::Structure st;
        st.label = "H2 " + std::to_string(i);
        st.cell.load(6.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 6.0);
        st.add_atom(tadah::mlip::Atom(tadah::core::Element("H"), 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0));
        st.add_atom(tadah::mlip::Atom(tadah::core::Element("H"),
                                      0.7 + 0.05 * static_cast<double>(i),
                                      0.0, 0.0, 0.0, 0.0, 0.0));
        st.energy = -1.0 - 0.1 * static_cast<double>(i);
        db.add(st);
    }
    db.dbidx = {0, db.size()};
    try {
        tadah::mlip::memory::MLIPWorkspaceManager ws;
        tadah::mlip::Trainer tr(c, ws);
        tr.nnf.calc(db, tr.context);
        tr.model->train(db, tr.dc);
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

/* A dataset of @p n H2 cells (6 A cubic, the bond along x) whose
 * energies are @p energy(i), with the weights line @p weights. */
void write_h2_db(const std::string& path, std::size_t n,
                 const std::function<double(std::size_t)>& energy,
                 const std::string& weights = "1 1 1") {
    std::ofstream os(path);
    for (std::size_t i = 0; i < n; ++i) {
        os << "H2 " << i << "\n" << weights << "\n"
           << energy(i) << " 0\n"
           << "6 0 0\n0 6 0\n0 0 6\n0 0 0\n0 0 0\n0 0 0\n"
           << "H 0 0 0 0 0 0\n"
           << "H " << 0.7 + 0.05 * static_cast<double>(i) << " 0 0 0 0 0\n\n";
    }
}

/* The process-wide HPOLog and HPO_Host state of a new process. */
void fresh_process_state() {
    tadah::core::PeriodicTable::initialize();
    tadah::hpo::HPOLog::instance().reset_for_test();
    Host::reset_globals_for_test();
}

/* What hpo_run throws for @p c with @p target_text (and train.db,
 * val.db), "" when it returns; standard output into @p out. */
std::string run_error(Context c, const std::string& target_text,
                      bool check, std::string* out = nullptr) {
    std::ofstream("config.hpo") << target_text;
    std::ofstream("config.val") << "DBFILE val.db\n";
    c.add("DBFILE", "train.db");
    c.add("HPOTARGET", "config.hpo");
    c.add("VALIDATION", "config.val");
    if (check) c.add("CHECK", "true");
    fresh_process_state();
    CaptureCout cap;
    std::string err;
    try {
        tadah::hpo::hpo_run(c);
    } catch (const std::exception& e) {
        err = e.what();
    }
    if (out) *out = cap.text.str();
    return err;
}

std::string read_file(const std::string& path) {
    std::ifstream is(path);
    std::stringstream ss;
    ss << is.rdbuf();
    return ss.str();
}

std::size_t count_of(const std::string& text, const std::string& what) {
    std::size_t n = 0;
    for (std::size_t p = text.find(what); p != std::string::npos;
         p = text.find(what, p + what.size()))
        ++n;
    return n;
}

}  // namespace

/* LAMBDA -2, LAMBDA 0 1e-12, LAMBDA -1 under OALGO 1, 2 and 4, and LAMBDA
 * -1 under M_KRR implementation 2 are refused by the startup checks with
 * the message of the fit; LAMBDA -1 under OALGO 3 and LAMBDA >= 0 pass. A
 * run whose evaluations fit nothing -- no training data, or FIXINDEX
 * pinning both weights -- reads no LAMBDA and passes with LAMBDA -2 (a
 * pin).
 * Before the fix: the startup checks passed every one of them; each
 * evaluation of the run refused it and scored FAILSCORE. */
TEST_CASE("tadah hpo refuses at startup a LAMBDA that every fit refuses",
          "[HPO][fit_settings_startup][lambda]") {
    tadah::core::PeriodicTable::initialize();
    struct Case {
        std::string lambda, oalgo;
        bool krr2;
        bool refused;
    };
    const std::vector<Case> cases = {
        {"-2", "4", false, true},       {"0 1e-12", "3", false, true},
        {"-1", "1", false, true},       {"-1", "2", false, true},
        {"-1", "4", false, true},       {"-1", "1", true, true},
        {"-0.5", "1", true, true},      {"-1", "3", false, false},
        {"0", "1", false, false},       {"1e-3", "4", true, false}};
    for (const Case& k : cases) {
        DYNAMIC_SECTION("LAMBDA " << k.lambda << ", OALGO " << k.oalgo
                                  << (k.krr2 ? ", M_KRR implementation 2"
                                             : "")) {
            Context c = k.krr2 ? krr2_ctx() : model_ctx();
            set_key(c, "LAMBDA", k.lambda);
            set_key(c, "OALGO", k.oalgo);
            const std::string fit = fit_refusal(c);
            REQUIRE(fit.empty() != k.refused);
            CHECK(startup_error(c, target()) == fit);
        }
    }
    SECTION("no training data, FIXINDEX pins both weights (a pin)") {
        Context c = model_ctx();
        set_key(c, "LAMBDA", "-2");
        c.add("FIXINDEX", "1-2");
        c.add("FIXWEIGHT", "0.1 0.2");
        CHECK(startup_error(c, target(), false).empty());
    }
    SECTION("training data, FIXINDEX pins both weights (a pin)") {
        Context c = model_ctx();
        set_key(c, "LAMBDA", "-2");
        c.add("FIXINDEX", "1-2");
        c.add("FIXWEIGHT", "0.1 0.2");
        CHECK(startup_error(c, target(), true).empty());
    }
}

/* MODEL M_KRR Kern_Linear 3 is refused by the startup checks with the
 * message of the fit, naming both implementations.
 * Before the fix: the startup checks passed it; each evaluation refused it
 * with 'This KRR implementation does exist: 3'. */
TEST_CASE("tadah hpo refuses at startup an M_KRR implementation other than "
          "1 and 2",
          "[HPO][fit_settings_startup][krr]") {
    tadah::core::PeriodicTable::initialize();
    Context c = model_ctx();
    set_key(c, "MODEL", "M_KRR Kern_Linear 3");
    const std::string fit = fit_refusal(c);
    REQUIRE_THAT(fit, Contains("MODEL M_KRR Kern_Linear 3"));
    CHECK(startup_error(c, target()) == fit);
}

/* An OPTIM box of EWEIGHT, FWEIGHT, SWEIGHT or LAMBDA whose low bound is
 * below 0 is refused at its line, naming the key, the index and the low
 * bound, explicit or AUTO; a box from 0 is accepted. OPTIM LAMBDA when the
 * training configuration has LAMBDA -1 is refused at its line, naming the
 * evidence approximation.
 * Before the fix: each was accepted; the evaluations whose candidate was
 * below 0 failed, and with LAMBDA -1 the run stopped when it started, at
 * its starting point outside the box. */
TEST_CASE("An OPTIM box of a fit weight or of LAMBDA below 0 is refused at "
          "its line",
          "[HPO][fit_settings_startup][optim]") {
    tadah::core::PeriodicTable::initialize();
    for (const char* key : {"EWEIGHT", "FWEIGHT", "SWEIGHT", "LAMBDA"}) {
        DYNAMIC_SECTION(key) {
            const std::string k = key;
            const std::string bad =
                parse_error(model_ctx(), target("OPTIM " + k + " (1) -0.5 2\n"));
            CHECK_THAT(bad, Contains("Error on line 1") &&
                                Contains("OPTIM " + k + " (1): the low bound "
                                         "-0.5 is negative") &&
                                !Contains("OPTIM " + k + " (1): OPTIM"));
            CHECK(parse_error(model_ctx(), target("OPTIM " + k + " (1) 0 2\n"))
                      .empty());
        }
    }
    SECTION("an AUTO box below 0") {
        CHECK_THAT(parse_error(model_ctx(),
                               target("OPTIM EWEIGHT (1) AUTO FRAC 3\n")),
                   Contains("OPTIM EWEIGHT (1): the low bound -2 is negative"));
        CHECK(parse_error(model_ctx(), target("OPTIM EWEIGHT (1) AUTO\n"))
                  .empty());
    }
    SECTION("OPTIM LAMBDA under LAMBDA -1") {
        Context c = model_ctx();
        set_key(c, "LAMBDA", "-1");
        set_key(c, "OALGO", "3");
        CHECK_THAT(parse_error(c, target("OPTIM LAMBDA (1) 1e-6 1e-2\n")),
                   Contains("Error on line 1") &&
                       Contains("OPTIM LAMBDA: the training configuration "
                                "has LAMBDA -1") &&
                       Contains("evidence approximation"));
    }
}

/* Under M_KRR implementation 2, OPTIM EWEIGHT, FWEIGHT, SWEIGHT and WDBFILE
 * are refused by the startup checks with the text of the fit's refusal
 * (krr_implementation2_ignores); under implementation 1 they pass.
 * Before the fix: the startup checks passed them; every candidate other
 * than 1 was refused by its evaluation. */
TEST_CASE("Under M_KRR implementation 2, OPTIM of a weight is refused at "
          "startup",
          "[HPO][fit_settings_startup][krr]") {
    tadah::core::PeriodicTable::initialize();
    struct Case {
        const char* key;
        const char* instead;
    };
    for (const Case& k : {Case{"EWEIGHT", "with EWEIGHT 1"},
                          Case{"FWEIGHT", "with FWEIGHT 1"},
                          Case{"SWEIGHT", "with SWEIGHT 1"},
                          Case{"WDBFILE", "without WDBFILE"}}) {
        DYNAMIC_SECTION(k.key) {
            const std::string line =
                std::string("OPTIM ") + k.key + " (1) 0.5 2\n";
            Context c = krr2_ctx();
            if (std::string(k.key) == "WDBFILE") c.add("WDBFILE", 1.0);
            CHECK(startup_error(c, target(line)) ==
                  tadah::mlip::krr_implementation2_ignores(
                      "Kern_Linear", std::string("OPTIM ") + k.key,
                      "applies it", k.instead));
            Context c1 = model_ctx();
            set_key(c1, "MODEL", "M_KRR Kern_Linear 1");
            if (std::string(k.key) == "WDBFILE") c1.add("WDBFILE", 1.0);
            CHECK(startup_error(c1, target(line)).empty());
        }
    }
}

/* OPTIM of LSCALE, ESHIFT, ESHIFT_ATOM, ESHIFT_DBATOM, EFILTER, FFILTER
 * and ZERO_COM_FORCE is refused at its line, naming the key and that it is
 * applied once, when the run starts.
 * Before the fix: OPTIM LSCALE, ESHIFT, EFILTER and FFILTER were accepted,
 * and every evaluation trained on the data as they were prepared at
 * startup, whatever the candidate value; the others were refused with
 * another message, or as keys the configuration does not hold. */
TEST_CASE("OPTIM of a key applied once at startup is refused at its line",
          "[HPO][fit_settings_startup][optim]") {
    tadah::core::PeriodicTable::initialize();
    Context c = model_ctx();
    c.add("LSCALE", 1.0);
    c.add("ESHIFT", -0.5);
    c.add("ESHIFT_ATOM", "false");
    c.add("ESHIFT_DBATOM", "false");
    c.add("EFILTER", "-10 10");
    c.add("FFILTER", 10.0);
    c.add("ZERO_COM_FORCE", "false");
    for (const char* key : {"LSCALE", "ESHIFT", "ESHIFT_ATOM",
                            "ESHIFT_DBATOM", "EFILTER", "FFILTER",
                            "ZERO_COM_FORCE"}) {
        DYNAMIC_SECTION(key) {
            const std::string k = key;
            CHECK_THAT(parse_error(c, target("OPTIM " + k + " (1) 0.5 2\n")),
                       Contains("Error on line 1") &&
                           Contains("OPTIM " + k + " is not a valid search "
                                    "target") &&
                           Contains("applied once, when the run starts"));
        }
    }
}

/* Once the training data are read, the settings and the data that M_KRR
 * implementation 2 would ignore are refused (an eweight 0.5, EWEIGHT 2),
 * and so is a weight that is not a finite number >= 0 (EWEIGHT -1 of
 * M_BLR), by tadah hpo --check and the run alike, with the message of the
 * fit; the same setups with every weight 1 pass (a pin).
 * Before the fix: tadah hpo --check passed each of them (it fits nothing),
 * and every evaluation of the run refused it. */
TEST_CASE("tadah hpo refuses at startup the training data a fit refuses",
          "[HPO][fit_settings_startup][krr]") {
    struct Case {
        const char* name;
        bool krr2;
        const char* key;
        const char* value;
        const char* weights;
        const char* says;  // nullptr: passes
    };
    const Case cases[] = {
        {"implementation 2, eweight 0.5", true, nullptr, nullptr, "0.5 1 1",
         "eweight 0.5 of structure 1 of 4"},
        {"implementation 2, EWEIGHT 2", true, "EWEIGHT", "2", "1 1 1",
         "EWEIGHT 2 would be ignored"},
        {"M_BLR, EWEIGHT -1", false, "EWEIGHT", "-1", "1 1 1",
         "EWEIGHT -1: a weight is a finite number >= 0"},
        {"implementation 2, every weight 1 (a pin)", true, nullptr, nullptr,
         "1 1 1", nullptr},
        {"M_BLR, every weight 1 (a pin)", false, nullptr, nullptr, "1 1 1",
         nullptr}};
    for (const Case& k : cases) {
        DYNAMIC_SECTION(k.name) {
            ScratchDir sd;
            write_h2_db("train.db", 4,
                        [](std::size_t i) { return -1.0 - 0.1 * i; },
                        k.weights);
            write_h2_db("val.db", 2, [](std::size_t i) { return -1.05 - 0.1 * i; });
            Context c = k.krr2 ? krr2_ctx() : model_ctx();
            if (k.key) set_key(c, k.key, k.value);
            const std::string checked = run_error(c, target(), true);
            INFO("check: " << checked);
            if (!k.says) {
                CHECK(checked.empty());
                continue;
            }
            CHECK_THAT(checked, Contains(k.says));
            const std::string ran = run_error(c, target(), false);
            CHECK(ran == checked);
            /* Refused at startup: no evaluation wrote a row. */
            CHECK(read_file("loss.tadah").find("\n1 ") == std::string::npos);
        }
    }
}

/* LAMBDA -1 with OALGO 3 on targets that are noise to the model: the
 * evidence approximation of each evaluation stops at its iteration limit
 * (tadah train of the same data and model gives the WARNING). The journal,
 * hpo_run.log, says so once, naming the first evaluation; the evaluations
 * print nothing of it on standard output (VERBOSE 1). On targets that are
 * all zero the evidence approximation of each evaluation is refused (alpha
 * and beta are not finite): each evaluation fails as a candidate, scores
 * FAILSCORE and counts as a failed evaluation, not one that threw, and the
 * journal gives the refusal once.
 * Before the fix: each evaluation printed the WARNING on standard output
 * (3 of 3) and the journal had none; each refused evaluation was journalled as an
 * evaluation that threw ('eval-thread caught throw'), once per
 * evaluation. */
TEST_CASE("The evidence approximation of tadah hpo is said once per run in "
          "the journal",
          "[HPO][fit_settings_startup][ea]") {
    Context c = model_ctx();
    set_key(c, "LAMBDA", "-1");
    set_key(c, "OALGO", "3");
    set_key(c, "VERBOSE", "1");  // the Logger prints WARNINGs
    const std::string hpo = target("OPTIM CGRID2B (1) 0.8 1.2\n");
    SECTION("stopped at the iteration limit") {
        ScratchDir sd;
        const double noise[8] = {0.3, 0.0, 0.0, 0.3, -0.3, 0.0, 0.2, 0.1};
        write_h2_db("train.db", 8, [&](std::size_t i) { return noise[i]; });
        write_h2_db("val.db", 2, [](std::size_t i) { return 0.1 * i; });
        std::string out;
        const std::string err = run_error(c, hpo, false, &out);
        INFO("error: " << err << "\nstdout:\n" << out);
        CHECK(err.empty());
        const std::string journal = read_file("hpo_run.log");
        INFO("journal:\n" << journal);
        CHECK(count_of(journal, "stopped at its iteration limit") == 1);
        CHECK(count_of(journal, "once per run") == 1);
        CHECK(count_of(out, "stopped at its iteration limit") == 0);
    }
    SECTION("refused: targets all zero") {
        ScratchDir sd;
        write_h2_db("train.db", 4, [](std::size_t) { return 0.0; });
        write_h2_db("val.db", 2, [](std::size_t i) { return 0.1 * i; });
        std::string out;
        (void)run_error(c, hpo, false, &out);
        const std::string journal = read_file("hpo_run.log");
        INFO("journal:\n" << journal << "\nstdout:\n" << out);
        CHECK(count_of(journal, "LAMBDA -1 (evidence approximation): "
                                "iteration 1 gave alpha") == 1);
        CHECK(count_of(journal, "once per run") == 1);
        CHECK(count_of(journal, "eval-thread caught throw") == 0);
        const std::string threw_line = [&] {
            const std::size_t p = journal.find("Evaluations that threw");
            return p == std::string::npos
                       ? std::string()
                       : journal.substr(p, journal.find('\n', p) - p);
        }();
        INFO(threw_line);
        REQUIRE_FALSE(threw_line.empty());
        CHECK_THAT(threw_line, Contains(": 0"));
    }
}
