// test_hpo_error_scales.cpp
//
// tadah hpo and the error scales of the fit (ESIGMA, FSIGMA a b, SSIGMA
// a b, ENORM, ENERGY) and the priorities of the datasets (WDBFILE):
//
//  - OPTIM ESIGMA, FSIGMA (1), SSIGMA (1) and WDBFILE search positive
//    values: an AUTO box is geometric ([x0/2, 2 x0]), and a box whose low
//    bound is not > 0 is refused at its line; FSIGMA (2) and SSIGMA (2),
//    the relative parts, are linear boxes from 0 up (an AUTO box clamped at
//    0, a negative low bound refused);
//  - OPTIM ENORM and ENERGY (a choice and a switch, no continuous value)
//    and OPTIM of a retired key (EWEIGHT, FWEIGHT, SWEIGHT, WDBFILE_AUTO,
//    EWEIGHT_TEMP: with its conversion) are refused at their line;
//  - when the run starts: a configuration with a retired key (also through
//    CONFIG), and error scales a fit refuses (an ESIGMA that is not > 0, an
//    FSIGMA other than its default with FORCE false); OPTIM of the error
//    scale of a quantity the fit does not have; OPTIM of the error scale of
//    the only quantity of the fit, which scales every row alike, with
//    LAMBDA 0 or -1 (every candidate fits alike); OPTIM of an error scale
//    or of WDBFILE when no evaluation fits (FIXINDEX pins every weight);
//    GRAD_STRATEGY ANALYTICAL with OPTIM of an error scale (no analytical
//    derivative; HYBRID takes finite differences for it). A WARNING when
//    every error scale of the fit is searched with LAMBDA 0 or -1, or with
//    OPTIM LAMBDA: one direction of the search then changes nothing;
//  - the validation configuration (--validation) refuses the error scales,
//    ENORM, ENERGY and the retired keys at their line: the validation set is
//    not fitted;
//  - a WARM seed with retired key lines (a potential of an earlier version)
//    is read, with one notice listing them;
//  - PC_SRMSE (six entries in eV/A^3) is refused with the conversion to
//    PC_SRMSE_GPA (the nine entries of each tensor, in GPa), which has the
//    global and the per-dataset form of PC_ERMSE and PC_FRMSE.
//
// The model is a one-species (H) D2_BP pair term with BF_Linear (two
// weights); the data are H2 cells.

#include "catch2/catch.hpp"

#include <tadah/core/blueprint_parser.h>
#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/key_notice.h>
#include <tadah/core/periodic_table.h>
#include <tadah/core/retired_keys.h>
#include <tadah/hpo/hpo.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/startup_validation.h>

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
#include <vector>

namespace {

namespace fs = std::filesystem;
using Catch::Matchers::Contains;
using Catch::Matchers::Equals;
using tadah::core::Context;
using tadah::hpo::HPOTargets;
using ParamVec = dlib::matrix<double, 0, 1>;
using Host     = tadah::hpo::HPO_Host<ParamVec, ::LammpsRunner>;

std::string stamp() {
    return std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
}

struct TempFile {
    fs::path path;
    TempFile(const std::string& body, const std::string& ext)
    : path(fs::temp_directory_path() / ("tadah_error_scales_" + stamp() + ext)) {
        std::ofstream os(path);
        REQUIRE(os.good());
        os << body;
    }
    ~TempFile() {
        std::error_code ec;
        fs::remove(path, ec);
    }
};

/* A scratch working directory: hpo_run() writes every file into the
 * current directory. */
struct ScratchDir {
    fs::path dir, original;
    ScratchDir() {
        dir = fs::temp_directory_path() / ("tadah_error_scales_run_" + stamp());
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

/* A stream into a string while it lives. */
struct Capture {
    std::ostream& os;
    std::ostringstream text;
    std::streambuf* old;
    explicit Capture(std::ostream& s) : os(s), old(s.rdbuf(text.rdbuf())) {}
    ~Capture() { os.rdbuf(old); }
    Capture(const Capture&) = delete;
    Capture& operator=(const Capture&) = delete;
};

void set_key(Context& c, const std::string& key, const std::string& value) {
    if (c.has_stored(key)) c.remove(key);
    c.add(key, value);
}

/* The configuration lines of the model: H, a D2_BP pair term of two
 * functions, BF_Linear, OALGO 4, LAMBDA 1e-3, energies only. */
const std::vector<std::string> kModelLines = {
    "ATOM H",           "WATOM 1.0",         "INIT2B true",
    "INITMB false",     "TYPE2B D2_BP 2 2 H H", "CGRID2B 1.0 2.0",
    "SGRID2B 0.5 1.0",  "RCTYPE2B Cut_Cos",  "RCUT2B 5.0",
    "FORCE false",      "STRESS false",      "BIAS false",
    "NORM false",       "VERBOSE 0",         "MODEL M_BLR BF_Linear",
    "LAMBDA 1e-3",      "OALGO 4"};

Context model_ctx() {
    std::ostringstream text;
    for (const std::string& l : kModelLines) text << l << "\n";
    std::istringstream in(text.str());
    return tadah::core::CoreUserConfigParser().parse(in);
}

const std::string kOptimizer =
    "OPTIMIZER\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 3\nENDOPTIMIZER\n";

/* A target file: @p lines, PC_ERMSE and an optimiser; OPTIM CGRID2B (1)
 * unless @p lines has an OPTIM line. */
std::string target(const std::string& lines = "",
                   const std::string& optimizer = kOptimizer) {
    const bool optim = lines.find("OPTIM ") != std::string::npos;
    return (optim ? "" : "OPTIM CGRID2B (1) 0.5 1.5\n") + lines +
           "PC_ERMSE 0 1\n" + optimizer;
}

/* What reading the target file throws, "" when it is read. */
std::string parse_error(Context c, const std::string& text) {
    TempFile tf(text, ".hpo");
    try {
        HPOTargets trg(c, tf.path.string());
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

/* The box of index @p i (1-based) of the OPTIM key @p key of @p text. */
std::pair<double, double> box_of(Context c, const std::string& text,
                                 const std::string& key, std::size_t i) {
    TempFile tf(text, ".hpo");
    HPOTargets trg(c, tf.path.string());
    const auto& o = trg.optimap.at(key);
    REQUIRE(o.opt.at(i - 1));
    return {o.lowbound.at(i - 1), o.highbound.at(i - 1)};
}

/* What the startup checks throw (the target file is read first), "" when
 * they pass; their standard error into @p err. */
std::string startup_error(Context c, const std::string& text,
                          std::string* err = nullptr) {
    if (!c.has_stored("DBFILE")) c.add("DBFILE", "dummy.db");
    TempFile tf(text, ".hpo");
    Capture cap(std::cerr);
    std::string what;
    try {
        HPOTargets trg(c, tf.path.string());
        tadah::hpo::validate_hpo_setup(c, trg, true, true);
    } catch (const std::exception& e) {
        what = e.what();
    }
    if (err) *err = cap.text.str();
    return what;
}

/* A dataset of @p n H2 cells (6 A cubic, the bond along x). */
void write_h2_db(const std::string& path, std::size_t n, double e0) {
    std::ofstream os(path);
    for (std::size_t i = 0; i < n; ++i) {
        os << "H2 " << i << "\n1 1 1\n"
           << e0 - 0.1 * static_cast<double>(i) << " 0\n"
           << "6 0 0\n0 6 0\n0 0 6\n0 0 0\n0 0 0\n0 0 0\n"
           << "H 0 0 0 0 0 0\n"
           << "H " << 0.7 + 0.05 * static_cast<double>(i) << " 0 0 0 0 0\n\n";
    }
}

/* What hpo_run throws for @p c with @p target_text, train.db, val.db and
 * the validation configuration @p val, "" when it returns. */
std::string run_error(Context c, const std::string& target_text, bool check,
                      const std::string& val = "DBFILE val.db\n") {
    std::ofstream("config.hpo") << target_text;
    std::ofstream("config.val") << val;
    write_h2_db("train.db", 4, -1.0);
    write_h2_db("val.db", 2, -1.05);
    if (!c.has_stored("DBFILE")) c.add("DBFILE", "train.db");
    c.add("HPOTARGET", "config.hpo");
    c.add("VALIDATION", "config.val");
    if (check) c.add("CHECK", "true");
    tadah::core::PeriodicTable::initialize();
    tadah::hpo::HPOLog::instance().reset_for_test();
    Host::reset_globals_for_test();
    Capture out(std::cout);
    std::string err;
    try {
        tadah::hpo::hpo_run(c);
    } catch (const std::exception& e) {
        err = e.what();
    }
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

/* Pins: a box of ESIGMA, FSIGMA (1), SSIGMA (1) or WDBFILE whose low bound
 * is not > 0 is refused at its line, naming the key, the index, the low
 * bound and why the value is > 0; a box of FSIGMA (2) or SSIGMA (2), the
 * relative part, whose low bound is below 0 is refused, and one from 0 is
 * accepted.
 * Before the fix: the boxes from 0 and below were accepted (each candidate
 * at or below 0 failed its fit), and OPTIM WDBFILE was refused as a key
 * applied once when the run starts. */
TEST_CASE("OPTIM of an error scale or of WDBFILE searches values > 0",
          "[HPO][error_scales][optim]") {
    tadah::core::PeriodicTable::initialize();
    Context c = model_ctx();
    set_key(c, "WDBFILE", "1");
    struct Case { const char* key; const char* why; };
    for (const Case& k :
         {Case{"ESIGMA", "ESIGMA is the energy error scale, a number > 0 "
                         "(meV/atom, or meV per sqrt(atom) under ENORM "
                         "SQRT), and every fit refuses another"},
          Case{"FSIGMA", "the absolute part a of FSIGMA is the force error "
                         "scale of a structure whose forces are zero, a "
                         "number > 0 (eV/A), and every fit refuses another"},
          Case{"SSIGMA", "the absolute part a of SSIGMA is the stress error "
                         "scale of a structure whose stresses are zero, a "
                         "number > 0 (GPa), and every fit refuses another"},
          Case{"WDBFILE", "a searched WDBFILE value is the priority of a "
                          "dataset in the fit, a number > 0 (WDBFILE 0 in "
                          "the training configuration leaves a dataset out "
                          "of the fit)"}}) {
        DYNAMIC_SECTION(k.key) {
            const std::string key = k.key;
            for (const char* lo : {"0", "-0.5"}) {
                INFO("low bound " << lo);
                CHECK_THAT(parse_error(c, target("OPTIM " + key + " (1) " +
                                                 lo + " 2\n")),
                           Equals("Error on line 1: OPTIM " + key +
                                  " (1): the low bound " + lo +
                                  " is not > 0, but " + k.why +
                                  ", so the low bound of its box is > 0\n"
                                  "Line: OPTIM " + key + " (1) " + lo + " 2"));
            }
            CHECK(parse_error(c, target("OPTIM " + key + " (1) 0.05 2\n"))
                      .empty());
        }
    }
    for (const char* key : {"FSIGMA", "SSIGMA"}) {
        DYNAMIC_SECTION(key << " (2)") {
            const std::string k = key;
            CHECK_THAT(parse_error(c, target("OPTIM " + k + " (2) -0.1 0.2\n")),
                       Contains("OPTIM " + k + " (2): the low bound -0.1 is "
                                "negative, but the relative part b of " + k +
                                " is a number >= 0, and every fit refuses a "
                                "negative one, so the low bound of its box "
                                "is >= 0"));
            CHECK(parse_error(c, target("OPTIM " + k + " (2) 0 0.2\n"))
                      .empty());
        }
    }
}

/* Pins: the AUTO box of ESIGMA, FSIGMA (1), SSIGMA (1) and WDBFILE is
 * geometric, [x0/2, 2 x0] (RATIO 2), and AUTO FRAC 3 of ESIGMA, whose box
 * reaches below 0, is refused; the AUTO box of FSIGMA (2) is linear and
 * clamped at 0: [0, 0.02] with FLOOR 0.02 around b = 0, [0.02, 0.06] around
 * b = 0.04 (FRAC 0.5).
 * Before the fix: every AUTO box of these keys was linear (FRAC 0.5,
 * [0.5, 1.5] for ESIGMA 1) and AUTO FRAC 3 of ESIGMA gave [-2, 4]. */
TEST_CASE("The AUTO box of an error scale or of WDBFILE is geometric",
          "[HPO][error_scales][optim]") {
    tadah::core::PeriodicTable::initialize();
    Context c = model_ctx();
    set_key(c, "WDBFILE", "1 0.5");
    using Box = std::pair<double, double>;
    CHECK(box_of(c, target("OPTIM ESIGMA (1) AUTO\n"), "ESIGMA", 1) ==
          Box{0.5, 2.0});
    CHECK(box_of(c, target("OPTIM FSIGMA (1) AUTO\n"), "FSIGMA", 1) ==
          Box{0.05, 0.2});
    CHECK(box_of(c, target("OPTIM SSIGMA (1) AUTO\n"), "SSIGMA", 1) ==
          Box{0.25, 1.0});
    CHECK(box_of(c, target("OPTIM WDBFILE (2) AUTO\n"), "WDBFILE", 2) ==
          Box{0.25, 1.0});
    CHECK_THAT(parse_error(c, target("OPTIM ESIGMA (1) AUTO FRAC 3\n")),
               Contains("OPTIM ESIGMA (1): the low bound -2 is not > 0"));
    CHECK(box_of(c, target("OPTIM FSIGMA (2) AUTO FLOOR 0.02\n"), "FSIGMA",
                 2) == Box{0.0, 0.02});
    Context b = c;
    set_key(b, "FSIGMA", "0.1 0.04");
    const Box lin = box_of(b, target("OPTIM FSIGMA (2) AUTO\n"), "FSIGMA", 2);
    CHECK(lin.first == Approx(0.02).epsilon(1e-15));
    CHECK(lin.second == Approx(0.06).epsilon(1e-15));
}

/* Pins: OPTIM ENORM and OPTIM ENERGY are refused at their line (a choice
 * and a switch); OPTIM of a retired key is refused at its line with its
 * conversion.
 * Before the fix: OPTIM ENORM and ENERGY were read as boxes of a word
 * value, and OPTIM EWEIGHT, FWEIGHT and SWEIGHT failed with the context's
 * refusal of a retired key, WDBFILE_AUTO and EWEIGHT_TEMP as keys applied
 * once when the run starts. */
TEST_CASE("OPTIM of ENORM, ENERGY or a retired key is refused at its line",
          "[HPO][error_scales][optim]") {
    tadah::core::PeriodicTable::initialize();
    const Context c = model_ctx();
    CHECK_THAT(parse_error(c, target("OPTIM ENORM (1) 0 1\n")),
               Equals("Error on line 1: OPTIM ENORM is not a valid search "
                      "target: ENORM (ATOM or SQRT) chooses how the residual "
                      "of an energy is normalised, a choice rather than a "
                      "continuous parameter; the training configuration "
                      "sets it.\nLine: OPTIM ENORM (1) 0 1"));
    CHECK_THAT(parse_error(c, target("OPTIM ENERGY (1) 0 1\n")),
               Equals("Error on line 1: OPTIM ENERGY is not a valid search "
                      "target: ENERGY (true or false) says whether the fit "
                      "has the energies, a switch rather than a continuous "
                      "parameter; the training configuration sets it.\n"
                      "Line: OPTIM ENERGY (1) 0 1"));
    for (const char* key :
         {"EWEIGHT", "FWEIGHT", "SWEIGHT", "WDBFILE_AUTO", "EWEIGHT_TEMP"}) {
        DYNAMIC_SECTION(key) {
            const std::string k = key;
            const std::string said =
                parse_error(c, target("OPTIM " + k + " (1) 0.5 2\n"));
            CHECK_THAT(said,
                       Contains("Error on line 1: OPTIM " + k +
                                " is not a valid search target: " + k +
                                " is a retired key: ") &&
                           Contains("Replaced by: "));
        }
    }
}

/* Pins: OPTIM of the error scale of a quantity the fit does not have is
 * refused when the run starts: FSIGMA with FORCE false, SSIGMA with STRESS
 * false, ESIGMA with ENERGY false; with the quantity on it passes.
 * Before the fix: they passed, and every candidate fitted alike. */
TEST_CASE("tadah hpo refuses OPTIM of the error scale of a quantity the fit "
          "does not have",
          "[HPO][error_scales][startup]") {
    tadah::core::PeriodicTable::initialize();
    struct Case { const char* key; const char* sw; const char* what; };
    for (const Case& k : {Case{"FSIGMA", "FORCE", "forces"},
                          Case{"SSIGMA", "STRESS", "stresses"},
                          Case{"ESIGMA", "ENERGY", "energies"}}) {
        DYNAMIC_SECTION(k.key) {
            const std::string key = k.key, sw = k.sw;
            Context off = model_ctx();
            if (key == "ESIGMA") set_key(off, "FORCE", "true");
            set_key(off, sw, "false");
            const std::string line = "OPTIM " + key + " (1) 0.05 2\n";
            CHECK(startup_error(off, target(line)) ==
                  "OPTIM " + key + ": " + sw + " is false: the fit has no " +
                      k.what + ", so " + key + " would change nothing and "
                      "every candidate would give the same fit. " + sw +
                      " true fits the " + k.what + ".");
            Context on = model_ctx();
            set_key(on, sw, "true");
            CHECK(startup_error(on, target(line)).empty());
        }
    }
}

/* Pins: OPTIM of the error scale of the only quantity of the fit (ESIGMA
 * of an energy fit; FSIGMA (1) of a force fit with FSIGMA b 0) with LAMBDA
 * 0 or -1 is refused when the run starts: it scales every row alike, which
 * changes no coefficient. With LAMBDA 1e-3, with a relative part b > 0 (a
 * per-structure scale), and OPTIM FSIGMA (2) it passes.
 * Before the fix: they passed, and every candidate gave the same fit. */
TEST_CASE("tadah hpo refuses OPTIM of the error scale of the only quantity "
          "with LAMBDA 0 or -1",
          "[HPO][error_scales][startup]") {
    tadah::core::PeriodicTable::initialize();
    const std::string esigma = "OPTIM ESIGMA (1) 0.1 10\n";
    for (const auto& [lambda, oalgo] :
         std::vector<std::pair<std::string, std::string>>{{"0", "4"},
                                                           {"-1", "3"}}) {
        DYNAMIC_SECTION("LAMBDA " << lambda) {
            Context c = model_ctx();
            set_key(c, "LAMBDA", lambda);
            set_key(c, "OALGO", oalgo);
            CHECK(startup_error(c, target(esigma)) ==
                  "OPTIM ESIGMA: the energies are the only quantity of the "
                  "fit and LAMBDA is " + lambda + ": the error scale of the "
                  "only quantity scales every row of the fit alike, which "
                  "changes no coefficient, so every candidate would give "
                  "the same fit. With a LAMBDA > 0 the error scale weighs "
                  "the data against the penalty, and with forces or "
                  "stresses in the fit, the energies against them.");
            Context f = c;
            set_key(f, "ENERGY", "false");
            set_key(f, "FORCE", "true");
            CHECK_THAT(startup_error(f, target("OPTIM FSIGMA (1) 0.01 1\n")),
                       Contains("OPTIM FSIGMA: the forces are the only "
                                "quantity of the fit and LAMBDA is " + lambda));
            Context fb = f;
            set_key(fb, "FSIGMA", "0.1 0.05");
            CHECK(startup_error(fb, target("OPTIM FSIGMA (1) 0.01 1\n"))
                      .empty());
            CHECK(startup_error(f, target("OPTIM FSIGMA (2) 0 1\n")).empty());
        }
    }
    SECTION("LAMBDA 1e-3 (a pin)") {
        CHECK(startup_error(model_ctx(), target(esigma)).empty());
    }
}

/* Pins: with every error scale of the fit searched (ESIGMA and FSIGMA (1)
 * of an energy and force fit, FSIGMA b 0) and LAMBDA 0, the startup gives
 * a WARNING that one direction of the search changes nothing; with OPTIM
 * LAMBDA and ESIGMA of an energy fit too, which names the invariance of
 * the sum a fit minimises: the error scales times c with LAMBDA divided by
 * c^2 give the same fit (sum p (r / (c sigma))^2 + (LAMBDA / c^2) |w|^2 is
 * the sum of sigma and LAMBDA divided by c^2); none when one error scale
 * is fixed and LAMBDA is not searched.
 * Before the fix: no WARNING; then a WARNING that said "with LAMBDA times
 * c^2", the opposite. */
TEST_CASE("tadah hpo warns when every error scale of the fit is searched",
          "[HPO][error_scales][startup]") {
    tadah::core::PeriodicTable::initialize();
    const std::string warning =
        "every error scale of the fit is searched";
    Context c = model_ctx();
    set_key(c, "FORCE", "true");
    set_key(c, "LAMBDA", "0");
    std::string err;
    CHECK(startup_error(c, target("OPTIM ESIGMA (1) 0.1 10\n"
                                  "OPTIM FSIGMA (1) 0.01 1\n"),
                        &err)
              .empty());
    CHECK(count_of(err, warning) == 1);
    CHECK_THAT(err, Contains("ESIGMA, FSIGMA (1)") &&
                        Contains("LAMBDA is 0"));
    CHECK(startup_error(c, target("OPTIM ESIGMA (1) 0.1 10\n"), &err).empty());
    CHECK(count_of(err, warning) == 0);
    Context e = model_ctx();
    CHECK(startup_error(e, target("OPTIM ESIGMA (1) 0.1 10\n"
                                  "OPTIM LAMBDA (1) 1e-6 1e-2\n"),
                        &err)
              .empty());
    CHECK(count_of(err, warning) == 1);
    CHECK_THAT(err, Contains("OPTIM LAMBDA") &&
                        Contains("the error scales times c with LAMBDA "
                                 "divided by c^2 give the same fit"));
}

/* Pins: OPTIM WDBFILE of a run with one training dataset is refused when
 * the run starts with LAMBDA 0 or -1 (not searched): the dataset's value
 * multiplies the priority of every row of the fit alike, which changes no
 * coefficient (the WDBFILE counterpart of the error scale of the only
 * quantity). With LAMBDA 1e-3, or with two training datasets, it passes.
 * Before the fix: it passed, and every candidate gave the same fit (a run
 * of WDBFILE 1, 1.375, 0.625 and 1.1875 had one loss). */
TEST_CASE("tadah hpo refuses OPTIM WDBFILE of the only training dataset "
          "with LAMBDA 0 or -1",
          "[HPO][error_scales][startup][wdbfile]") {
    tadah::core::PeriodicTable::initialize();
    const std::string line = "OPTIM WDBFILE (1) 0.5 2\n";
    for (const auto& [lambda, oalgo] :
         std::vector<std::pair<std::string, std::string>>{{"0", "4"},
                                                           {"-1", "3"}}) {
        DYNAMIC_SECTION("LAMBDA " << lambda) {
            Context c = model_ctx();
            set_key(c, "LAMBDA", lambda);
            set_key(c, "OALGO", oalgo);
            set_key(c, "WDBFILE", "1");
            CHECK(startup_error(c, target(line)) ==
                  "OPTIM WDBFILE: the run has one training dataset and "
                  "LAMBDA is " + lambda + ": its WDBFILE value multiplies "
                  "the priority of every row of the fit alike, which "
                  "changes no coefficient, so every candidate would give "
                  "the same fit. With a LAMBDA > 0 the priority weighs the "
                  "data against the penalty, and with more training "
                  "datasets, the dataset against the others.");
            Context two = c;
            set_key(two, "DBFILE", "a.db b.db");
            set_key(two, "WDBFILE", "1 1");
            CHECK(startup_error(two, target(line)).empty());
        }
    }
    SECTION("LAMBDA 1e-3 (a pin)") {
        Context c = model_ctx();
        set_key(c, "WDBFILE", "1");
        CHECK(startup_error(c, target(line)).empty());
    }
}

/* Pins: with the WDBFILE value of every training dataset searched (two
 * datasets, WDBFILE (1) and (2)) and LAMBDA 0, the startup gives a WARNING
 * that one direction of the search changes nothing (every priority times c
 * gives the same fit); with OPTIM LAMBDA and the one dataset's WDBFILE,
 * the WARNING names the invariance: the WDBFILE values times c with LAMBDA
 * times c give the same fit; none when one dataset's value is fixed.
 * Before the fix: no WARNING. */
TEST_CASE("tadah hpo warns when the WDBFILE of every training dataset is "
          "searched",
          "[HPO][error_scales][startup][wdbfile]") {
    tadah::core::PeriodicTable::initialize();
    const std::string warning =
        "the WDBFILE value of every training dataset is searched";
    Context c = model_ctx();
    set_key(c, "LAMBDA", "0");
    set_key(c, "DBFILE", "a.db b.db");
    set_key(c, "WDBFILE", "1 1");
    std::string err;
    CHECK(startup_error(c, target("OPTIM WDBFILE (1) 0.5 2\n"
                                  "OPTIM WDBFILE (2) 0.5 2\n"),
                        &err)
              .empty());
    CHECK(count_of(err, warning) == 1);
    CHECK_THAT(err, Contains("WDBFILE (1), WDBFILE (2)") &&
                        Contains("LAMBDA is 0"));
    CHECK(startup_error(c, target("OPTIM WDBFILE (1) 0.5 2\n"), &err)
              .empty());
    CHECK(count_of(err, warning) == 0);
    Context one = model_ctx();
    set_key(one, "WDBFILE", "1");
    CHECK(startup_error(one, target("OPTIM WDBFILE (1) 0.5 2\n"
                                    "OPTIM LAMBDA (1) 1e-6 1e-2\n"),
                        &err)
              .empty());
    CHECK(count_of(err, warning) == 1);
    CHECK_THAT(err, Contains("OPTIM LAMBDA") &&
                        Contains("the WDBFILE values times c with LAMBDA "
                                 "times c give the same fit"));
}

/* Pins: GRAD_STRATEGY ANALYTICAL with OPTIM ESIGMA is refused when the run
 * starts, naming the error scale and HYBRID, which takes finite
 * differences for it; HYBRID passes.
 * Before the fix: the run started, and its first gradient stopped it. */
TEST_CASE("GRAD_STRATEGY ANALYTICAL refuses OPTIM of an error scale",
          "[HPO][error_scales][startup]") {
    tadah::core::PeriodicTable::initialize();
    const std::string line = "OPTIM ESIGMA (1) 0.1 10\n";
    auto optimizer = [](const char* strategy) {
        return std::string("OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\n"
                           "GRAD_STRATEGY ") + strategy + "\nENDOPTIMIZER\n";
    };
    CHECK_THAT(startup_error(model_ctx(), target(line, optimizer("ANALYTICAL"))),
               Contains("GRAD_STRATEGY ANALYTICAL needs the analytical "
                        "gradient of the loss, which this setup cannot "
                        "form: OPTIM ESIGMA searches an error scale of the "
                        "fit, which the analytical gradient does not "
                        "differentiate") &&
                   Contains("GRAD_STRATEGY HYBRID"));
    CHECK(startup_error(model_ctx(), target(line, optimizer("HYBRID"))).empty());
}

/* Pins: GRAD_STRATEGY ANALYTICAL with OPTIM WDBFILE, a priority of the fit
 * that each evaluation applies, is refused when the run starts, naming
 * WDBFILE and HYBRID; HYBRID passes.
 * Before the fix: the run started, and its first gradient stopped it ("no
 * analytical gradient for 1 parameter at this point"). */
TEST_CASE("GRAD_STRATEGY ANALYTICAL refuses OPTIM WDBFILE",
          "[HPO][error_scales][startup][wdbfile]") {
    tadah::core::PeriodicTable::initialize();
    const std::string line = "OPTIM WDBFILE (1) 0.5 2\n";
    auto optimizer = [](const char* strategy) {
        return std::string("OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\n"
                           "GRAD_STRATEGY ") + strategy + "\nENDOPTIMIZER\n";
    };
    Context c = model_ctx();
    set_key(c, "WDBFILE", "1");
    CHECK_THAT(startup_error(c, target(line, optimizer("ANALYTICAL"))),
               Contains("GRAD_STRATEGY ANALYTICAL needs the analytical "
                        "gradient of the loss, which this setup cannot "
                        "form: OPTIM WDBFILE searches a priority of the "
                        "fit, which the analytical gradient does not "
                        "differentiate") &&
                   Contains("GRAD_STRATEGY HYBRID"));
    CHECK(startup_error(c, target(line, optimizer("HYBRID"))).empty());
}

/* Pins: error scales that every fit refuses are refused when the run
 * starts, with the fit's message: ESIGMA -1, FSIGMA 0.2 with FORCE false.
 * Before the fix: the run started and every evaluation refused them. */
TEST_CASE("tadah hpo refuses at startup error scales every fit refuses",
          "[HPO][error_scales][startup]") {
    tadah::core::PeriodicTable::initialize();
    Context neg = model_ctx();
    set_key(neg, "ESIGMA", "-1");
    CHECK_THAT(startup_error(neg, target()),
               Contains("tadah hpo: ESIGMA -1: the energy error scale is a "
                        "number > 0"));
    Context off = model_ctx();
    set_key(off, "FSIGMA", "0.2");
    CHECK_THAT(startup_error(off, target()),
               Contains("tadah hpo: FSIGMA 0.2: with FORCE false the fit "
                        "observes no forces"));
}

/* Pins: a training configuration with retired key lines, its own or of a
 * file it includes with CONFIG, is refused when tadah hpo starts, by the
 * run and by --check alike, naming each line and its conversion (FWEIGHT
 * of the energy fit, FORCE false, converted by deleting the line); the
 * run writes no evaluation.
 * Before the fix: the run started and every evaluation refused it, and the
 * run ended without a potential ("no evaluation produced a usable
 * loss"). */
TEST_CASE("tadah hpo refuses a configuration with retired keys when it "
          "starts",
          "[HPO][error_scales][retired]") {
    ScratchDir sd;
    {
        std::ofstream os("inner.train");
        os << "FWEIGHT 100\n";
        std::ofstream outer("outer.train");
        for (const std::string& l : kModelLines) outer << l << "\n";
        outer << "EWEIGHT 1.0\nCONFIG inner.train\n";
    }
    Context c;
    {
        tadah::core::IgnoredKeyNoticeCapture quiet;
        c = tadah::core::CoreUserConfigParser().parse("outer.train");
    }
    for (const bool check : {true, false}) {
        DYNAMIC_SECTION((check ? "--check" : "the run")) {
            const std::string err = run_error(c, target(), check);
            CHECK_THAT(err,
                       Contains("tadah hpo: the configuration has keys "
                                "retired in Tadah! 1.6.0") &&
                           Contains("EWEIGHT 1.0 (line 18 of outer.train)") &&
                           Contains("FWEIGHT 100 (line 1 of inner.train; "
                                    "FORCE false here: converted by "
                                    "deleting the line)"));
            CHECK(read_file("loss.tadah").find("\n1 ") == std::string::npos);
        }
    }
}

/* Pins: the validation configuration refuses at its line ESIGMA, FSIGMA,
 * SSIGMA, ENORM and ENERGY (they set the fit, and the validation set is not
 * fitted) and each retired key, with its conversion.
 * Before the fix: they were accepted and had no effect (the retired keys
 * with a notice). */
TEST_CASE("The validation configuration refuses the error scales and the "
          "retired keys",
          "[HPO][error_scales][validation]") {
    ScratchDir sd;
    for (const char* line :
         {"ESIGMA 2", "FSIGMA 0.2", "SSIGMA 1", "ENORM SQRT", "ENERGY false",
          "EWEIGHT 1.0", "FWEIGHT 100", "SWEIGHT 1", "WDBFILE_AUTO 0.5",
          "EWEIGHT_TEMP 300"}) {
        DYNAMIC_SECTION(line) {
            const std::string l = line;
            const std::string key = l.substr(0, l.find(' '));
            const std::string err = run_error(
                model_ctx(), target(), false, "DBFILE val.db\n" + l + "\n");
            CHECK_THAT(err, Contains("Key " + key + " on line 2 of "
                                     "config.val is not accepted here: the "
                                     "validation set is not fitted"));
            if (key == "EWEIGHT" || key == "FWEIGHT" || key == "SWEIGHT" ||
                key == "WDBFILE_AUTO" || key == "EWEIGHT_TEMP")
                CHECK_THAT(err, Contains(key + " is a retired key: "));
        }
    }
}

/* Pins: the validation configuration holds DBFILE, EFILTER and FFILTER
 * (and CONFIG): tadah hpo, and tadah hpo --check, refuse every other key
 * at its line, WDBFILE (a priority of the fit) with its own reason and the
 * keys of the model, of its fit and of the run with one reason; the three
 * keys it reads pass the startup checks.
 * Before the fix: WDBFILE, LAMBDA, OALGO, RCUT2B and VERBOSE were accepted
 * and had no effect. */
TEST_CASE("The validation configuration holds DBFILE, EFILTER and FFILTER "
          "only",
          "[HPO][error_scales][validation]") {
    ScratchDir sd;
    for (const bool check : {true, false}) {
        for (const char* line : {"WDBFILE 0", "WDBFILE 2", "LAMBDA 1e-3",
                                 "OALGO 3", "RCUT2B 4.0", "VERBOSE 2"}) {
            DYNAMIC_SECTION((check ? "--check, " : "the run, ") << line) {
                const std::string l = line;
                const std::string key = l.substr(0, l.find(' '));
                const std::string err = run_error(
                    model_ctx(), target(), check, "DBFILE val.db\n" + l + "\n");
                const std::string why =
                    key == "WDBFILE"
                        ? "the validation set is not fitted; WDBFILE, the "
                          "priority of each training dataset in the fit, is "
                          "set in the training configuration, and a "
                          "validation structure without a label has "
                          "eweight, fweight or sweight 0 in its data file"
                        : "the validation configuration holds DBFILE, "
                          "EFILTER and FFILTER (and CONFIG), the validation "
                          "datasets and their filters; the training "
                          "configuration sets the model, its fit and the "
                          "run";
                CHECK(err == "Key " + key + " on line 2 of config.val is not "
                             "accepted here: " + why);
            }
        }
    }
    SECTION("DBFILE, EFILTER and FFILTER pass the startup checks") {
        const std::string err =
            run_error(model_ctx(), target(), true,
                      "DBFILE val.db\nEFILTER -2 0\nFFILTER 100\n");
        CHECK(err.empty());
    }
}

/* Pins: a WARM seed with retired key lines (a potential of an earlier
 * version) is read for its OPTIM values, with one notice that lists its
 * retired lines with the file and their line numbers.
 * Before the fix: the seed was read with no notice. */
TEST_CASE("A WARM seed with retired keys is read with one notice",
          "[HPO][error_scales][retired][warm]") {
    tadah::core::PeriodicTable::initialize();
    TempFile seed("# a potential of an earlier version\n"
                  "CGRID2B 1.2 1.9\nEWEIGHT 1.0\nFWEIGHT 100\n"
                  "SGRID2B 0.5 1.0\n",
                  ".tadah");
    const std::string text =
        "OPTIM CGRID2B (1) 0.5 1.5\nPC_ERMSE 0 1\nOPTIMIZER\n"
        "INIT\n  METHOD WARM " + seed.path.string() + "\nENDINIT\n"
        "LIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 3\nENDOPTIMIZER\n";
    TempFile tf(text, ".hpo");
    Context c = model_ctx();
    tadah::core::IgnoredKeyNoticeCapture cap;
    std::unique_ptr<HPOTargets> trg;
    REQUIRE_NOTHROW(trg = std::make_unique<HPOTargets>(c, tf.path.string()));
    std::size_t retired = 0;
    for (const std::string& n : cap.notices())
        if (n.find("Retired keys") != std::string::npos) {
            ++retired;
            CHECK_THAT(n, Contains("EWEIGHT (line 3 of " +
                                   seed.path.string() + ")") &&
                              Contains("FWEIGHT (line 4 of " +
                                       seed.path.string() + ")"));
        }
    CHECK(retired == 1);
    REQUIRE(trg->spec.optim_init.size() == 1);
    REQUIRE(trg->spec.optim_init.front().size() == 1);
    CHECK(trg->spec.optim_init.front()[0] == 1.2);
}

/* Pins: a WARM seed without a line of a searched error scale (a potential
 * of an earlier version has no FSIGMA line) does not have its value:
 * OPTIM FSIGMA (1) is a parameter not found in the seed, which
 * WARM_ON_MISMATCH handles (error: the target file is refused naming it);
 * a seed with an FSIGMA line gives its value (0.2).
 * Before the fix: the seed's FSIGMA read as the blueprint default, 0.1,
 * and WARM said it loaded every parameter. */
TEST_CASE("A WARM seed without the line of a searched key does not have "
          "its value",
          "[HPO][error_scales][warm]") {
    tadah::core::PeriodicTable::initialize();
    auto text = [](const std::string& seed) {
        return "OPTIM CGRID2B (1) 0.5 1.5\nOPTIM FSIGMA (1) 0.05 0.5\n"
               "PC_ERMSE 0 1\nOPTIMIZER\nINIT\n  METHOD WARM " + seed +
               "\n  WARM_ON_MISMATCH error\nENDINIT\nLIB NLOPT\n"
               "ALGO LN_SBPLX\nMAXEVAL 3\nENDOPTIMIZER\n";
    };
    Context c = model_ctx();
    set_key(c, "FORCE", "true");
    TempFile old_seed("CGRID2B 1.2 1.9\nFWEIGHT 100\nSGRID2B 0.5 1.0\n",
                      ".tadah");
    CHECK_THAT(parse_error(c, text(old_seed.path.string())),
               Contains("WARM: 1 parameter(s) not found in " +
                        old_seed.path.string()) &&
                   Contains("FSIGMA"));
    TempFile seed("CGRID2B 1.2 1.9\nFSIGMA 0.2 0\nSGRID2B 0.5 1.0\n",
                  ".tadah");
    TempFile tf(text(seed.path.string()), ".hpo");
    HPOTargets trg(c, tf.path.string());
    REQUIRE(trg.spec.optim_init.size() == 1);
    REQUIRE(trg.spec.optim_init.front().size() == 2);
    CHECK(trg.spec.optim_init.front()[0] == 1.2);
    CHECK(trg.spec.optim_init.front()[1] == 0.2);
}

/* Pins: PC_SRMSE is refused at its line, with the conversion to
 * PC_SRMSE_GPA; PC_SRMSE_GPA is read in the global form and, with a
 * validation set of two datasets, in the per-dataset form of PC_ERMSE and
 * PC_FRMSE.
 * Before the fix: PC_SRMSE was read (six entries in eV/A^3), PC_SRMSE_GPA
 * was an unknown key, and an indexed stress target was refused. */
TEST_CASE("PC_SRMSE is replaced by PC_SRMSE_GPA",
          "[HPO][error_scales][srmse_gpa]") {
    tadah::core::PeriodicTable::initialize();
    const Context c = model_ctx();
    CHECK_THAT(parse_error(c, target("PC_SRMSE 0.001 1\n")),
               Contains("'PC_SRMSE' was removed from the grammar: replaced "
                        "by PC_SRMSE_GPA") &&
                   Contains("160.2"));
    CHECK(parse_error(c, target("PC_SRMSE_GPA 0.5 4 --skip-above 50\n"))
              .empty());
    TempFile tf(target("PC_SRMSE_GPA (1) 0.5 4\nPC_SRMSE_GPA (2) 1 1\n"),
                ".hpo");
    Context cc = c;
    HPOTargets trg(cc, tf.path.string());
    REQUIRE_NOTHROW(trg.resolve_validation_targets({"a.db", "b.db"}));
    CHECK(trg.srmse_targets.indexed());
    REQUIRE(trg.srmse_targets.datasets().size() == 2);
    CHECK(trg.srmse_targets.datasets()[0].weight == 4.0);
    CHECK(trg.srmse_targets.datasets()[1].target == 1.0);
}

/* Pins: OPTIM of an error scale or of WDBFILE in a run whose evaluations
 * fit no coefficient (FIXINDEX pins both weights) is refused when the run
 * starts: it would change nothing; with a fit it passes.
 * Before the fix: it passed, and every candidate gave the same potential. */
TEST_CASE("tadah hpo refuses OPTIM of an error scale when no evaluation "
          "fits",
          "[HPO][error_scales][startup]") {
    tadah::core::PeriodicTable::initialize();
    Context pinned = model_ctx();
    set_key(pinned, "FIXINDEX", "1-2");
    set_key(pinned, "FIXWEIGHT", "0.1 0.2");
    set_key(pinned, "WDBFILE", "1");
    for (const char* key : {"ESIGMA", "WDBFILE"}) {
        DYNAMIC_SECTION(key) {
            const std::string k = key;
            const std::string line = "OPTIM " + k + " (1) 0.5 2\n";
            CHECK(startup_error(pinned, target(line)) ==
                  "OPTIM " + k + ": the evaluations of this run fit no "
                  "coefficient (FIXINDEX pins every weight), so " + k +
                  " would change nothing and every candidate would give "
                  "the same potential.");
            Context fitted = model_ctx();
            set_key(fitted, "WDBFILE", "1");
            CHECK(startup_error(fitted, target(line)).empty());
        }
    }
}

/* A guard: every key the blueprint marks retired is refused as an OPTIM
 * target with its conversion (HPOTargets::optim_retired_key_error lists
 * them for the registry's coverage check), so a key retired later cannot
 * be searched without a refusal. It passes as long as the list holds the
 * blueprint's retired keys; with one left out it fails (ERROR, the key of
 * --error of tadah predict, retired with the other keys of uncertainty
 * output: before the fix the list had no ERROR, and OPTIM ERROR got no
 * retired key's refusal). */
TEST_CASE("Every retired key of the blueprint is refused as an OPTIM target",
          "[HPO][error_scales][retired]") {
    tadah::core::PeriodicTable::initialize();
    std::size_t n = 0;
    for (const auto& [key, info] :
         tadah::core::BlueprintParser::getKeysMapRef()) {
        if (!info.isRetired()) continue;
        ++n;
        INFO(key);
        CHECK(HPOTargets::optim_retired_key_error(key) ==
              "OPTIM " + key + " is not a valid search target: " +
                  tadah::core::retired_key_text(key, info));
    }
    CHECK(n == 9);
    CHECK(HPOTargets::optim_retired_key_error("ESIGMA").empty());
}
