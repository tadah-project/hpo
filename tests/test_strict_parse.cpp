// test_strict_parse.cpp
//
// The strict number parsers of the hpotarget grammar (strict_parse.h) and
// the PROCESS worker deadline they feed (pc_worker_deadline_ms):
//   - parse_uint: decimal digits only, in [min, max]; no sign, fraction,
//     exponent, whitespace, hex or trailing junk, and no silent overflow;
//   - parse_finite: the whole token, finite (inf/nan/overflow rejected);
//   - next_value / expect_end / single_value: a missing value and an extra
//     token are errors;
//   - parse_lammps_timespec: off | unlimited | [[HH:]MM:]SS, a strict
//     subset of what LAMMPS' utils::timespec2seconds reads, to the same
//     seconds (LAMMPS also takes empty parts, a fourth part and stod's
//     forms: hex, inf, subnormals);
//   - parse_script_timeout: that, and a limit <= 0 is rejected;
//   - pc_worker_deadline_ms: PC_WORKER_TIMEOUT, else max(3t, t + 120 s)
//     from the parsed seconds -- the string used to go through strtod,
//     which read "2:00:00" as 2 s and "00:05:00" as no deadline.
// The parser-level contracts (messages at the key's line) are in
// test_hpo_parser_contracts.cpp.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/pc_worker_pool.h>
#include <tadah/hpo/strict_parse.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#if defined(__SSE2__)
#include <xmmintrin.h>   // _mm_getcsr, _mm_setcsr
#endif

namespace {

namespace fs = std::filesystem;
using tadah::hpo::parse_finite;
using tadah::hpo::parse_lammps_timespec;
using tadah::hpo::parse_uint;
using tadah::hpo::pc_worker_deadline_ms;

std::string error_of(const std::function<void()>& f)
{
    try {
        f();
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

/* Unique per process and per call; removed on scope exit. */
struct TempFile {
    fs::path path;
    explicit TempFile(const std::string& body, const std::string& ext)
    {
        static unsigned counter = 0;
        path = fs::temp_directory_path() /
               ("tadah_strict_parse_" + std::to_string(::getpid()) + "_" +
                std::to_string(counter++) + ext);
        std::ofstream os(path);
        os << body;
        REQUIRE(os.good());
    }
    ~TempFile() { std::error_code ec; fs::remove(path, ec); }
};

}  // namespace

TEST_CASE("parse_uint accepts decimal digits in range and nothing else",
          "[HPO][strict_parse]")
{
    CHECK(parse_uint("0", "K") == 0u);
    CHECK(parse_uint("42", "K") == 42u);
    CHECK(parse_uint("007", "K") == 7u);
    CHECK(parse_uint("18446744073709551615", "K") ==
          std::numeric_limits<std::uint64_t>::max());

    for (const std::string tok :
         {"", "abc", "5x", "x5", "2.5", "-1", "-0", "+5", "1e3", "0x10", " 5",
          "5 ", "18446744073709551616", "99999999999999999999999"}) {
        INFO("'" << tok << "'");
        CHECK(error_of([&] { parse_uint(tok, "N_STARTS", 1); }) ==
              "N_STARTS must be an integer >= 1, got '" + tok + "'");
    }

    /* Bounds, and how the message states them. */
    CHECK(parse_uint("1", "N_STARTS", 1) == 1u);
    CHECK(error_of([] { parse_uint("0", "N_STARTS", 1); }) ==
          "N_STARTS must be an integer >= 1, got '0'");
    CHECK(parse_uint("15", "DIGITS", 1, 15) == 15u);
    CHECK(error_of([] { parse_uint("16", "DIGITS", 1, 15); }) ==
          "DIGITS must be an integer from 1 to 15, got '16'");
    CHECK(error_of([] { parse_uint("0", "DIGITS", 1, 15); }) ==
          "DIGITS must be an integer from 1 to 15, got '0'");
    CHECK(error_of([] { parse_uint("0", "OUTPUT", 1,
                                   std::numeric_limits<std::uint64_t>::max(),
                                   "it is a cadence"); }) ==
          "OUTPUT must be an integer >= 1, got '0' (it is a cadence)");
}

TEST_CASE("parse_finite accepts a whole finite number and nothing else",
          "[HPO][strict_parse]")
{
    CHECK(parse_finite("1e16", "FAILSCORE") == 1e16);
    CHECK(parse_finite("-5", "FAILSCORE") == -5.0);
    CHECK(parse_finite(".5", "FAILSCORE") == 0.5);
    CHECK(parse_finite("+2.5e-3", "FAILSCORE") == 2.5e-3);
    for (const std::string tok : {"", "abc", "5x", "1.5.2", "inf", "-inf",
                                  "nan", "NaN", "infinity", "1e400", "--5",
                                  " 5x"}) {
        INFO("'" << tok << "'");
        CHECK(error_of([&] { parse_finite(tok, "FAILSCORE"); }) ==
              "FAILSCORE must be a finite number, got '" + tok + "'");
    }
}

TEST_CASE("parse_finite: the number grammar is the decimal one LAMMPS reads",
          "[HPO][strict_parse][hex]")
{
    /* LAMMPS' utils::is_double: [+-], decimal digits with at most one '.'
     * and at least one digit, then an optional exponent [eE][+-]<digits>.
     * std::stod also read hexadecimal floats (FAILSCORE 0x1p40 loaded as
     * 1.0995e+12) and skipped leading whitespace. */
    for (const std::string tok : {"0x1p40", "0X10", "0x10", "-0x1p-3",
                                  "+0x.8p1", "0x1P+4", "0x1", " 5", " 1e3"}) {
        INFO("'" << tok << "'");
        CHECK(error_of([&] { parse_finite(tok, "FAILSCORE"); }) ==
              "FAILSCORE must be a finite number, got '" + tok + "'");
    }
    /* Every other form LAMMPS takes is still a number ... */
    struct Ok { std::string tok; double v; };
    for (const auto& c : std::vector<Ok>{
             {"5", 5.0}, {"5.", 5.0}, {".5", 0.5}, {"+5", 5.0}, {"-0", -0.0},
             {"007", 7.0}, {"1e5", 1e5}, {"1E+05", 1e5}, {"1.e3", 1e3},
             {".5e-3", 5e-4}, {"-2.5E-3", -2.5e-3}, {"+.5e+1", 5.0}}) {
        INFO("'" << c.tok << "'");
        CHECK(parse_finite(c.tok, "FAILSCORE") == c.v);
    }
    /* ... and what it does not take is not. */
    for (const std::string tok : {".", "+", "-", "e5", ".e5", "1e", "1e+",
                                  "1e5.0", "+-1", "1,5", "5d0", "1_000"}) {
        INFO("'" << tok << "'");
        CHECK(error_of([&] { parse_finite(tok, "FAILSCORE"); }) ==
              "FAILSCORE must be a finite number, got '" + tok + "'");
    }
    /* A time spec's parts are numbers of the same grammar. */
    for (const std::string tok : {"0x10", "1:0x10", "0x1p1:00:00"}) {
        INFO("'" << tok << "'");
        CHECK(error_of([&] { parse_lammps_timespec(tok, "--maxtime"); }) ==
              "--maxtime must be 'off', 'unlimited' or a time [[HH:]MM:]SS, "
              "got '" + tok + "'");
    }
    CHECK(*parse_lammps_timespec("+1:.5e1", "T") == 65.0);   // 1 min + 5 s
}

TEST_CASE("parse_number and parse_positive: the one number grammar",
          "[HPO][strict_parse][hex]")
{
    using tadah::hpo::number_underflow_note;
    using tadah::hpo::parse_number;
    using tadah::hpo::parse_positive;
    double v = -1.0;
    CHECK(parse_number("+.5e+1", v));
    CHECK(v == 5.0);
    CHECK(parse_number("-0", v));
    CHECK(v == 0.0);
    CHECK(parse_number("2.2250738585072014e-308", v));   // the smallest normal
    CHECK(v == std::numeric_limits<double>::min());
    CHECK(parse_number("1.7976931348623157e308", v));    // the largest double
    /* Rejected, and `v` left alone. */
    v = 7.0;
    for (const std::string tok : {"", "0x1p4", "inf", "nan", "1e400", "-1e400",
                                  "1e-320", "1e-400", " 5", "5 ", "5x", "1e",
                                  "."}) {
        INFO("'" << tok << "'");
        CHECK_FALSE(parse_number(tok, v));
        CHECK(v == 7.0);
    }
    /* The too-small note is for an underflow only. */
    const std::string note = " (too small: its magnitude is below the "
                             "smallest normal double, 2.2250738585072014e-308)";
    CHECK(number_underflow_note("1e-320") == note);
    CHECK(number_underflow_note("-1e-400") == note);
    for (const std::string tok : {"1e400", "0x1p-1070", "abc", "5", "0"})
        CHECK(number_underflow_note(tok).empty());

    CHECK(parse_positive("60", "MAXTIME", "number of seconds") == 60.0);
    CHECK(parse_positive("1e-9", "MAXTIME") == 1e-9);
    for (const std::string tok : {"0", "-0", "-5", "abc", "inf", "0x10"}) {
        INFO("'" << tok << "'");
        CHECK(error_of([&] { parse_positive(tok, "MAXTIME",
                                            "number of seconds"); }) ==
              "MAXTIME must be a positive number of seconds, got '" + tok +
              "'");
    }
    CHECK(error_of([] { parse_positive("1e-320", "MAXTIME"); }) ==
          "MAXTIME must be a positive number, got '1e-320'" + note);
}

#if defined(__SSE2__)
TEST_CASE("parse_number rejects a subnormal whatever the floating-point mode",
          "[HPO][strict_parse]")
{
    /* A subnormal was told from zero by a floating-point comparison,
     * v != 0.0 && |v| < DBL_MIN. With denormals-are-zero on (DAZ: what the
     * start-up code of an executable linked with -ffast-math sets, and
     * Intel compilers by default) a subnormal compares equal to 0, so
     * 1e-320 was accepted. The test is on the bits of the double now, like
     * the finiteness test (metric_value_is_finite). */
    constexpr unsigned kDaz = 0x0040u, kFtz = 0x8000u;   // MXCSR bits
    const unsigned saved = _mm_getcsr();
    _mm_setcsr(saved | kDaz | kFtz);
    double v = 7.0;
    const bool subnormal = tadah::hpo::parse_number("1e-320", v);
    const bool smallest = tadah::hpo::parse_number("-4.9e-324", v);
    const std::string note = tadah::hpo::number_underflow_note("1e-320");
    double z = 7.0, n = 7.0;
    const bool zero = tadah::hpo::parse_number("-0", z);
    const bool normal = tadah::hpo::parse_number("2.2250738585072014e-308", n);
    _mm_setcsr(saved);
    CHECK_FALSE(subnormal);
    CHECK_FALSE(smallest);
    CHECK(v == 7.0);
    CHECK(note == " (too small: its magnitude is below the smallest normal "
                  "double, 2.2250738585072014e-308)");
    CHECK(zero);
    CHECK(z == 0.0);
    CHECK(normal);
    CHECK(n == std::numeric_limits<double>::min());
}
#endif

TEST_CASE("is_number_shaped: a token meant as a number",
          "[HPO][strict_parse]")
{
    /* Where a position takes a number or a word (STEP <frac> | STEP <KEY>
     * <frac>, a PARAM value, an optional PC_LAMMPS value), a token shaped
     * like a number is one, and the number grammar judges it. */
    using tadah::hpo::is_number_shaped;
    for (const std::string tok : {"5", "5x", "-5x", ".5", "-.5x", "+.", "..",
                                  "1e-320", "0x1p-3", "-0x10", "inf", "-inf",
                                  "INF", "Infinity", "nan", "NaN", "+nan"})
        CHECK(is_number_shaped(tok));
    for (const std::string tok : {"", "+", "-", ".", ".x", "-var", "x5",
                                  "WATOM", "AUTO", "inhouse", "INFO", "nano",
                                  "JITTER", "SEED"})
        CHECK_FALSE(is_number_shaped(tok));
}

TEST_CASE("single_value: exactly one token",
          "[HPO][strict_parse]")
{
    {
        std::istringstream iss(" 5 ");
        CHECK(tadah::hpo::single_value(iss, "QFLUSH") == "5");
    }
    {
        std::istringstream iss("");
        CHECK(error_of([&] { tadah::hpo::single_value(iss, "QFLUSH"); }) ==
              "QFLUSH requires a value");
    }
    {
        std::istringstream iss("5 6");
        CHECK(error_of([&] { tadah::hpo::single_value(iss, "QFLUSH"); }) ==
              "QFLUSH takes a single value (unexpected '6')");
    }
    {
        /* next_value leaves the rest to the caller. */
        std::istringstream iss("30 ENDSTRATEGY");
        CHECK(tadah::hpo::next_value(iss, "MAXEVAL") == "30");
        std::string rest;
        iss >> rest;
        CHECK(rest == "ENDSTRATEGY");
    }
}

TEST_CASE("parse_lammps_timespec reads LAMMPS time specs",
          "[HPO][strict_parse][timeout]")
{
    CHECK_FALSE(parse_lammps_timespec("off", "PC_SCRIPT_TIMEOUT"));
    CHECK_FALSE(parse_lammps_timespec("unlimited", "PC_SCRIPT_TIMEOUT"));
    CHECK(*parse_lammps_timespec("90", "T") == 90.0);
    CHECK(*parse_lammps_timespec("0.5", "T") == 0.5);
    CHECK(*parse_lammps_timespec("1:30", "T") == 90.0);          // MM:SS
    CHECK(*parse_lammps_timespec("00:05:00", "T") == 300.0);     // HH:MM:SS
    CHECK(*parse_lammps_timespec("2:00:00", "T") == 7200.0);
    CHECK(*parse_lammps_timespec("1:30:15", "T") == 5415.0);
    CHECK(*parse_lammps_timespec("1.5:00:00", "T") == 5400.0);   // as LAMMPS
    /* As LAMMPS reads them: a limit <= 0 is a time too (parse_script_timeout
     * rejects it). */
    CHECK(*parse_lammps_timespec("0", "T") == 0.0);
    CHECK(*parse_lammps_timespec("-5", "T") == -5.0);
    for (const std::string tok : {"", "bad", "OFF", "Off", "5x", "1:2:3:4",
                                  "1::2", ":5", "5:", "inf", "nan",
                                  "1:x:3", "1e400",
                                  /* finite parts, infinite total */
                                  "1e307:0:0"}) {
        INFO("'" << tok << "'");
        CHECK(error_of([&] { parse_lammps_timespec(tok, "--maxtime"); }) ==
              "--maxtime must be 'off', 'unlimited' or a time [[HH:]MM:]SS, "
              "got '" + tok + "'");
    }
}

TEST_CASE("parse_script_timeout: a positive time, 'off' or 'unlimited'",
          "[HPO][strict_parse][timeout]")
{
    using tadah::hpo::parse_script_timeout;
    CHECK_FALSE(parse_script_timeout("off", "PC_SCRIPT_TIMEOUT"));
    CHECK_FALSE(parse_script_timeout("unlimited", "PC_SCRIPT_TIMEOUT"));
    CHECK(*parse_script_timeout("90", "T") == 90.0);
    CHECK(*parse_script_timeout("0.001", "T") == 0.001);
    CHECK(*parse_script_timeout("2:00:00", "T") == 7200.0);
    /* 0 stops a script at its first timer check; with 0 or a negative
     * limit $(timeremain) is 0, so the section guard stops every
     * multi-section script. */
    for (const std::string tok : {"0", "-0", "0.0", "-5", "0:00", "00:00:00",
                                  "-1:30", "0:-5"}) {
        INFO("'" << tok << "'");
        CHECK(error_of([&] { parse_script_timeout(tok, "--maxtime"); }) ==
              "--maxtime must be a positive time, 'off' or 'unlimited' (a "
              "limit <= 0 stops every script), got '" + tok + "'");
    }
    /* A malformed spec keeps the time-spec message. */
    CHECK(error_of([] { parse_script_timeout("5x", "PC_SCRIPT_TIMEOUT"); }) ==
          "PC_SCRIPT_TIMEOUT must be 'off', 'unlimited' or a time "
          "[[HH:]MM:]SS, got '5x'");
}

TEST_CASE("pc_worker_deadline_ms: PC_WORKER_TIMEOUT, else max(3t, t + 120 s) "
          "of the parsed seconds",
          "[HPO][strict_parse][timeout][pc_worker_pool]")
{
    const auto deadline = [](double worker_timeout_s, const std::string& spec) {
        return pc_worker_deadline_ms(
            worker_timeout_s, tadah::hpo::parse_script_timeout(spec, "T"));
    };
    /* t = 300 s: max(900, 420) = 900 s. strtod read "00:05:00" as 0: no
     * deadline at all. */
    CHECK(deadline(0.0, "00:05:00") == 900'000);
    /* t = 7200 s: max(21600, 7320) = 21600 s. strtod read 2 s: the worker
     * was killed after max(6, 122) = 122 s. */
    CHECK(deadline(0.0, "2:00:00") == 21'600'000);
    /* t = 90 s: max(270, 210) = 270 s. */
    CHECK(deadline(0.0, "90") == 270'000);
    /* t = 10 s: max(30, 130) = 130 s. */
    CHECK(deadline(0.0, "10") == 130'000);
    CHECK(deadline(0.0, "off") == -1);
    CHECK(deadline(0.0, "unlimited") == -1);
    /* PC_WORKER_TIMEOUT wins, for every script. */
    CHECK(deadline(45.5, "2:00:00") == 45'500);
    CHECK(deadline(45.5, "off") == 45'500);

    /* A deadline beyond INT_MAX ms is a long like any other (readn polls
     * it in slices); t = 1431656 s gives 4294968000 ms. */
    CHECK(deadline(0.0, "1431656") == 4'294'968'000L);
    /* One that does not fit in a long is rejected rather than converted
     * (undefined behaviour); the parser stops such a limit at its line. */
    CHECK_THROWS_WITH(pc_worker_deadline_ms(1e16, std::nullopt),
                      Catch::StartsWith("pc_worker_deadline_ms: the deadline "
                                        "does not fit in 64-bit milliseconds"));
    CHECK_THROWS_WITH(pc_worker_deadline_ms(0.0, 4e15),
                      Catch::StartsWith("pc_worker_deadline_ms: the deadline "
                                        "does not fit in 64-bit milliseconds"));

    /* A limit <= 0 never reaches the deadline: the parser rejects it at its
     * line. It used to give no hard deadline without a word. */
    for (const double t : {0.0, -0.0, -5.0}) {
        for (const double worker_timeout_s : {0.0, 45.5}) {
            INFO("script timeout " << t << " s, PC_WORKER_TIMEOUT "
                 << worker_timeout_s << " s");
            CHECK_THROWS_WITH(pc_worker_deadline_ms(worker_timeout_s, t),
                              Catch::StartsWith("pc_worker_deadline_ms: the "
                                                "script timeout must be "
                                                "positive, got "));
        }
    }
}

TEST_CASE("The parsed timeout seconds reach HPOTargets and LammpsSim",
          "[HPO][strict_parse][timeout]")
{
    TempFile script("clear\nrun 0\nvariable E equal 0.0\n", ".in");
    const std::string optimizer =
        "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\nENDOPTIMIZER\n";
    const std::string sim = "PC_LAMMPS --script " + script.path.string() +
                            " --varloss E 1.0 1.0";
    auto parse = [&](const std::string& root) {
        TempFile cfg(optimizer + root, ".hpo");
        tadah::core::Context ctx;
        return tadah::hpo::HPOTargets(ctx, cfg.path.string());
    };

    {
        auto trg = parse("PC_SCRIPT_TIMEOUT 2:00:00\n" + sim + "\n" + sim +
                         " --maxtime 00:05:00\n");
        CHECK(trg.MAX_TIME == "2:00:00");
        REQUIRE(trg.MAX_TIME_seconds);
        CHECK(*trg.MAX_TIME_seconds == 7200.0);
        REQUIRE(trg.LS.size() == 2);
        /* The line captures the root value in force when it is read ... */
        CHECK(trg.LS[0].maxtime == "2:00:00");
        REQUIRE(trg.LS[0].maxtime_seconds);
        CHECK(*trg.LS[0].maxtime_seconds == 7200.0);
        /* ... and --maxtime overrides it for its line. */
        CHECK(trg.LS[1].maxtime == "00:05:00");
        REQUIRE(trg.LS[1].maxtime_seconds);
        CHECK(*trg.LS[1].maxtime_seconds == 300.0);
    }
    {
        auto trg = parse(sim + "\n");
        CHECK(trg.MAX_TIME == "off");
        CHECK_FALSE(trg.MAX_TIME_seconds);
        CHECK_FALSE(trg.LS.at(0).maxtime_seconds);
    }
}
