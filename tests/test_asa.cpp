/*
 * Unit tests for LIB ASA / ALGO ASA (vendored Ingber ASA-30.53).
 *
 * Most tests drive `run_asa_erased` with a std::vector lambda (no
 * HPO_Host); the dlib adapter path is covered once via run_asa_impl.
 * The golden-neutrality gate lives in test_asa_golden.cpp.
 */

#include "catch2/catch.hpp"

#include <tadah/core/checkpoint.h>
#include <tadah/core/context.h>
#include <tadah/hpo/algorithms_asa.h>
#include <tadah/hpo/asa_testing.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/init_strategy.h>
#include <tadah/hpo/asa_options.h>
#include <tadah/hpo/durable_best.h>
#include <tadah/hpo/fd_stencil.h>
#include <tadah/hpo/run_stop.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/warmstart_cache.h>  /* current_chain(): the block slot */

#include "anneal_test_support.h"  /* JournalCapture */

#include <dlib/matrix.h>

// ASA's headers include <omp.h> inside their extern "C" blocks (ASA_PARALLEL
// with OpenMP). GCC 15's omp.h declares C++ templates, which C linkage
// forbids; included here first, at C++ linkage, the include inside is a
// no-op (its include guard).
#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {
#include <asa/tadah_asa_hook.h>  /* tadah_asa_cost_temp mirror */
}

#include "finish_within.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using tadah::hpo::AsaHostFn;
using tadah::hpo::AsaOptions;
using tadah::hpo::OptimizerSpec;
using tadah::hpo::parse_asa_options;
using tadah::hpo::run_asa_erased;

namespace {

OptimizerSpec make_asa_spec(std::vector<double> low, std::vector<double> high)
{
    OptimizerSpec sp;
    sp.low  = std::move(low);
    sp.high = std::move(high);
    sp.is_int.assign(sp.low.size(), false);
    sp.log_scaled.assign(sp.low.size(), false);
    sp.lib  = "ASA";
    sp.algo = "ASA";
    sp.seed = 424242u;
    return sp;
}

void set_seed_point(OptimizerSpec& sp, const std::vector<double>& x)
{
    sp.optim_init.assign(1, x);
}

double quad_bowl(const std::vector<double>& x,
                 const std::vector<double>& centre)
{
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double d = x[i] - centre[i];
        s += d * d;
    }
    return s;
}

/* Thread-safe recording objective (Gener_Block > 1 runs concurrently). */
struct Recorder {
    std::function<double(const std::vector<double>&)> f;
    mutable std::mutex m;
    mutable std::vector<std::vector<double>> calls;

    AsaHostFn fn() const
    {
        return [this](const std::vector<double>& x) {
            const double v = f(x);
            std::lock_guard<std::mutex> lk(m);
            calls.push_back(x);
            return v;
        };
    }
    std::vector<std::vector<double>> sorted_calls() const
    {
        std::lock_guard<std::mutex> lk(m);
        auto c = calls;
        std::sort(c.begin(), c.end());
        return c;
    }
};

/* An objective whose every call returns less than any call before it,
 * wherever ASA samples: every generated state is a new best, so ASA accepts
 * it and restarts at 0 its count of the states generated since the last new
 * best. ASA's periodic test (every Acceptance_Frequency_Modulus = 100
 * acceptances) feeds that count into the moving average (Gener_Mov_Avr = 3
 * tests) that resizes its block of generated states, Gener_Block. */
struct AlwaysImproving {
    std::atomic<long> n{0};
    double next() { return 1.0 / (1.0 + static_cast<double>(++n)); }
};

/* Runs ASA with `sp` on `rec` under a 60 s watchdog (`what` names the run
 * if it never returns) and returns ASA's "[HPO] ..." console lines. An
 * exception from the run propagates, with the console restored. */
std::string run_bounded(const OptimizerSpec& sp, std::size_t dim,
                        const Recorder& rec, const std::string& what)
{
    const AsaHostFn fn = rec.fn();
    std::ostringstream captured;
    struct Restore {
        std::streambuf* old;
        ~Restore() { std::cout.rdbuf(old); }
    } restore{std::cout.rdbuf(captured.rdbuf())};
    hpo_test::finish_within(std::chrono::seconds(60), what,
                            [&] { run_asa_erased(sp, dim, fn); });
    return captured.str();
}

/* Runs ASA with `sp` on AlwaysImproving under a 60 s watchdog; returns the
 * evaluated points and ASA's "[HPO] ..." console lines. */
std::vector<std::vector<double>> run_always_improving(const OptimizerSpec& sp,
                                                      std::size_t dim,
                                                      std::string& console)
{
    AlwaysImproving obj;
    Recorder rec;
    rec.f = [&obj](const std::vector<double>&) { return obj.next(); };
    console = run_bounded(
        sp, dim, rec,
        "run_asa_erased on an objective whose every state is a new best");
    return rec.calls;
}

/* Runs ASA with `sp` on AlwaysImproving and returns the size of every
 * block of generated states the block hook evaluated, in order. The hook
 * evaluates a block's slots t = 0, 1, ... on the pool, setting
 * warmstart::current_chain() = t for each; with THREADS 1 the pool has one
 * worker, which evaluates them in slot order. Evaluations on asa()'s own
 * thread (the initial cost-temperature samples, reanneal tangents) are not
 * block evaluations and are left out. A slot sequence that is not
 * 0, 1, 2, ... per block is reported through `anomalies`. */
std::vector<std::size_t> block_sizes(const OptimizerSpec& sp, std::size_t dim,
                                     std::string& console,
                                     std::vector<std::string>& anomalies)
{
    AlwaysImproving obj;
    std::mutex m;
    std::vector<std::pair<std::thread::id, std::size_t>> tags;
    Recorder rec;
    rec.f = [&](const std::vector<double>&) {
        {
            std::lock_guard<std::mutex> lk(m);
            tags.emplace_back(std::this_thread::get_id(),
                              tadah::hpo::warmstart::current_chain());
        }
        return obj.next();
    };
    console = run_bounded(sp, dim, rec, "run_asa_erased counting blocks");

    /* The pool worker: the thread of the first evaluation of a slot > 0. */
    std::thread::id worker;
    for (const auto& t : tags)
        if (t.second > 0) { worker = t.first; break; }
    std::vector<std::size_t> sizes;
    for (const auto& t : tags) {
        if (t.first != worker) continue;
        if (t.second == 0) {
            sizes.push_back(1);
        } else if (sizes.empty() || t.second != sizes.back()) {
            anomalies.push_back("slot " + std::to_string(t.second) +
                                " after a block of " +
                                std::to_string(sizes.empty() ? 0
                                                             : sizes.back()));
        } else {
            ++sizes.back();
        }
    }
    return sizes;
}

/* The first journal line that contains `head` ("ASA stopped: ",
 * "ASA summary: "), or "" if none does. */
std::string journal_line(const std::string& journal, const std::string& head)
{
    std::istringstream in(journal);
    for (std::string line; std::getline(in, line);)
        if (line.find(head) != std::string::npos) return line;
    return "";
}

/* ASA's closing console line for a run that a stop ended after n host
 * evaluations: the stop's reason and ASA's own exit code. */
std::string stopped_line(const std::string& reason, std::size_t n)
{
    return "ASA stopped: " + reason + " after " + std::to_string(n) +
           " host eval(s) (asa exit: IMMEDIATE_EXIT)";
}

}  // namespace

/* ================= option parsing ================================= */

TEST_CASE("ASA options: defaults resolve to the canonical values",
          "[HPO][asa][options]")
{
    auto sp = make_asa_spec({0.0, 0.0}, {1.0, 1.0});
    const AsaOptions o = parse_asa_options(sp, 2);
    CHECK(o.limit_acceptances == 10000);
    CHECK(o.limit_generated == 99999);
    CHECK(o.limit_invalid_generated_states == 1000);
    CHECK(o.accepted_to_generated_ratio == Approx(1e-6));
    CHECK(o.cost_precision == Approx(1e-18));
    CHECK(o.maximum_cost_repeat == 5);
    CHECK(o.number_cost_samples == 5);
    CHECK(o.temperature_ratio_scale == Approx(1e-5));
    CHECK(o.cost_parameter_scale_ratio == Approx(1.0));
    CHECK(o.temperature_anneal_scale == Approx(100.0));
    CHECK(o.include_integer_parameters == 0);
    CHECK(o.user_initial_parameters == 0);  /* no INIT seed */
    CHECK(o.sequential_parameters == -1);
    CHECK(o.initial_parameter_temperature == Approx(1.0));
    CHECK(o.acceptance_frequency_modulus == 100);
    CHECK(o.generated_frequency_modulus == 10000);
    CHECK(o.reanneal_cost == 1);
    CHECK(o.reanneal_parameters == 1);
    CHECK(o.delta_x == Approx(0.001));
    CHECK(o.user_tangents == 0);
    CHECK(o.curvature_0 == 1);  /* documented TADAH deviation */
    CHECK(o.user_quench_cost_scale == Approx(1.0));
    CHECK(o.user_quench_param_scale == std::vector<double>{1.0, 1.0});
    CHECK(o.user_temperature_ratio == std::vector<double>{1.0, 1.0});
    CHECK(o.user_delta_parameter == std::vector<double>{0.001, 0.001});
    CHECK(o.user_parameter_temperature == std::vector<double>{1.0, 1.0});
    CHECK(o.multi_number == 1);
    CHECK(o.multi_specify == 1);
    CHECK(o.queue_size == 0);
    CHECK(o.gener_block == 1);
    CHECK(o.gener_block_max == 1);
    CHECK(o.gener_mov_avr == 3);

    set_seed_point(sp, {0.5, 0.5});
    CHECK(parse_asa_options(sp, 2).user_initial_parameters == 1);
}

/* Pins: PARAM Include_Integer_Parameters and User_Initial_Parameters are
 * refused as options tadah hpo does not take, with the reason: every OPTIM
 * dimension takes real values, so no integer dimension is there to include
 * ("ASA:ASA: unknown PARAM 'Include_Integer_Parameters'.
 * 'Include_Integer_Parameters' is not available: every OPTIM dimension of
 * tadah hpo takes real values, so ASA has no integer dimension to
 * include"); the start of ASA is the first starting point of the INIT
 * block, or the values of the training configuration without one, and
 * METHOD RANDOM 1 draws a random one ("... 'User_Initial_Parameters' is not
 * available: ASA starts at the first starting point of the INIT block (the
 * values of the training configuration without one); METHOD RANDOM 1 in the
 * INIT block draws a random start from its SEED"). Both values are refused,
 * with and without a starting point; the list of accepted keys names
 * neither.
 * Before the fix: Include_Integer_Parameters 0 and 1 were accepted and had
 * no effect; User_Initial_Parameters 1 was the default with an INIT point
 * and refused without one, 0 refused with one ("... contradicts the INIT
 * block: remove the INIT seed ...") and the default without. */
TEST_CASE("ASA options: Include_Integer_Parameters and "
          "User_Initial_Parameters are not available", "[HPO][asa][options]")
{
    const std::string integer =
        "ASA:ASA: unknown PARAM 'Include_Integer_Parameters'. "
        "'Include_Integer_Parameters' is not available: every OPTIM dimension "
        "of tadah hpo takes real values, so ASA has no integer dimension to "
        "include.";
    const std::string initial =
        "ASA:ASA: unknown PARAM 'User_Initial_Parameters'. "
        "'User_Initial_Parameters' is not available: ASA starts at the first "
        "starting point of the INIT block (the values of the training "
        "configuration without one); METHOD RANDOM 1 in the INIT block draws "
        "a random start from its SEED.";
    for (const bool seeded : {false, true}) {
        auto sp = make_asa_spec({0.0}, {1.0});
        if (seeded) set_seed_point(sp, {0.5});
        for (const double v : {0.0, 1.0}) {
            INFO((seeded ? "a starting point" : "no starting point")
                 << ", value " << v);
            sp.param = {{"Include_Integer_Parameters", v}};
            REQUIRE_THROWS_WITH(parse_asa_options(sp, 1), integer);
            sp.param = {{"User_Initial_Parameters", v}};
            REQUIRE_THROWS_WITH(parse_asa_options(sp, 1), initial);
        }
    }
    auto sp = make_asa_spec({0.0}, {1.0});
    sp.param = {{"Cooling_Rate", 0.9}};
    try {
        (void)parse_asa_options(sp, 1);
        FAIL("Cooling_Rate was accepted");
    } catch (const std::exception& e) {
        const std::string msg = e.what();
        CHECK(msg.find("Accepted keys:") != std::string::npos);
        CHECK(msg.find("Include_Integer_Parameters") == std::string::npos);
        CHECK(msg.find("User_Initial_Parameters") == std::string::npos);
    }
}

TEST_CASE("ASA options: every documented key is accepted",
          "[HPO][asa][options]")
{
    auto sp = make_asa_spec({0.0, 0.0}, {1.0, 1.0});
    set_seed_point(sp, {0.5, 0.5});
    sp.param = {
        {"Limit_Acceptances", 500},   {"Limit_Generated", 700},
        {"Limit_Invalid_Generated_States", -1},
        {"Accepted_To_Generated_Ratio", 1e-4},
        {"Cost_Precision", 1e-12},    {"Maximum_Cost_Repeat", 3},
        {"Number_Cost_Samples", -2},  {"Temperature_Ratio_Scale", 1e-4},
        {"Cost_Parameter_Scale_Ratio", 2.0},
        {"Temperature_Anneal_Scale", 50},
        /* Include_Integer_Parameters and User_Initial_Parameters, which this
         * list held, are refused (the case "ASA options:
         * Include_Integer_Parameters and User_Initial_Parameters are not
         * available"). */
        {"Sequential_Parameters", 0},
        {"Initial_Parameter_Temperature", 2.0},
        {"Acceptance_Frequency_Modulus", 10},
        {"Generated_Frequency_Modulus", 100},
        {"Reanneal_Cost", 0},         {"Reanneal_Parameters", 0},
        {"Delta_X", 0.01},            {"User_Tangents", 0},
        {"Curvature_0", -1},
        {"User_Quench_Cost_Scale", 2.0},
        {"User_Quench_Param_Scale", 0.9},
        {"User_Temperature_Ratio", 1.5},
        {"User_Delta_Parameter", 0.02},
        {"User_Parameter_Temperature", 3.0},
        {"Multi_Number", 3},          {"Multi_Specify", 0},
        {"Multi_Grid", 0.05},         {"Queue_Size", 10},
        {"Queue_Resolution", 0.001},  {"Gener_Block", 4},
        {"Gener_Block_Max", 8},       {"Gener_Mov_Avr", 5},
    };
    const AsaOptions o = parse_asa_options(sp, 2);
    CHECK(o.limit_acceptances == 500);
    CHECK(o.number_cost_samples == -2);
    CHECK(o.curvature_0 == -1);
    CHECK(o.user_quench_param_scale == std::vector<double>{0.9, 0.9});
    CHECK(o.user_delta_parameter == std::vector<double>{0.02, 0.02});
    CHECK(o.multi_number == 3);
    CHECK(o.multi_grid == std::vector<double>{0.05, 0.05});
    CHECK(o.queue_size == 10);
    CHECK(o.gener_block == 4);
    CHECK(o.gener_block_max == 8);
    CHECK(o.gener_mov_avr == 5);
}

TEST_CASE("ASA options: Gener_Mov_Avr 0 switches the block resizing off",
          "[HPO][asa][options]")
{
    /* Upstream's README documents Gener_Mov_Avr 0 as "Gener_Block is not
     * changed thereafter", and asa()'s periodic test resizes the block only
     * when Gener_Mov_Avr > 0, but upstream's own option check rejected
     * every value below 1 (TADAH-PATCH 7 accepts 0). */
    auto sp = make_asa_spec({0.0, 0.0}, {1.0, 1.0});
    sp.param = {{"Gener_Block", 4.0}, {"Gener_Block_Max", 4.0},
                {"Gener_Mov_Avr", 0.0}};
    const AsaOptions o = parse_asa_options(sp, 2);
    CHECK(o.gener_block == 4);
    CHECK(o.gener_block_max == 4);
    CHECK(o.gener_mov_avr == 0);

    sp.param = {{"Gener_Mov_Avr", -1.0}};
    CHECK_THROWS_WITH(parse_asa_options(sp, 2),
                      Catch::Contains("Gener_Mov_Avr"));
}

TEST_CASE("ASA options: indexed per-dim grammar", "[HPO][asa][options]")
{
    auto sp = make_asa_spec({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});

    /* [i] is 1-based: [1] is the first dimension, [dim] the last, and
     * they line up with the dim column of hpo_search_space.tsv. */
    SECTION("uniform fill + indexed override") {
        sp.param = {{"User_Quench_Param_Scale", 0.8},
                    {"User_Quench_Param_Scale[2]", 0.5}};
        const auto o = parse_asa_options(sp, 3);
        CHECK(o.user_quench_param_scale ==
              std::vector<double>{0.8, 0.5, 0.8});
    }
    SECTION("index 1 is the first dimension") {
        sp.param = {{"User_Quench_Param_Scale", 0.8},
                    {"User_Quench_Param_Scale[1]", 0.5}};
        const auto o = parse_asa_options(sp, 3);
        CHECK(o.user_quench_param_scale ==
              std::vector<double>{0.5, 0.8, 0.8});
    }
    SECTION("indexed override of a mirrored default") {
        sp.param = {{"Delta_X", 0.01}, {"User_Delta_Parameter[3]", 0.5}};
        const auto o = parse_asa_options(sp, 3);
        CHECK(o.user_delta_parameter ==
              std::vector<double>{0.01, 0.01, 0.5});
    }
    SECTION("index 0 is rejected and names the 1-based change") {
        sp.param = {{"User_Quench_Param_Scale[0]", 0.5}};
        REQUIRE_THROWS_WITH(parse_asa_options(sp, 3),
                            Catch::Contains("1-based"));
    }
    SECTION("index out of range") {
        sp.param = {{"User_Quench_Param_Scale[4]", 0.5}};
        REQUIRE_THROWS_WITH(parse_asa_options(sp, 3),
                            Catch::Contains("index out of range"));
    }
    SECTION("last dimension is addressable") {
        sp.param = {{"User_Quench_Param_Scale[3]", 0.5}};
        const auto o = parse_asa_options(sp, 3);
        CHECK(o.user_quench_param_scale ==
              std::vector<double>{1.0, 1.0, 0.5});
    }
    SECTION("malformed index") {
        sp.param = {{"User_Quench_Param_Scale[x]", 0.5}};
        REQUIRE_THROWS_WITH(parse_asa_options(sp, 3),
                            Catch::Contains("malformed"));
    }
    SECTION("an index too large for a long names the PARAM") {
        /* std::stol threw a bare "stol". */
        sp.param = {{"User_Quench_Param_Scale[99999999999999999999]", 0.5}};
        REQUIRE_THROWS_WITH(parse_asa_options(sp, 3),
                            Catch::Contains("PARAM 'User_Quench_Param_Scale"
                                            "[99999999999999999999]'"));
    }
    SECTION("indexed form on a scalar key") {
        sp.param = {{"Limit_Acceptances[0]", 10}};
        REQUIRE_THROWS_WITH(parse_asa_options(sp, 3),
                            Catch::Contains("scalar option"));
    }
}

TEST_CASE("ASA options: rejections", "[HPO][asa][options]")
{
    auto sp = make_asa_spec({0.0}, {1.0});

    SECTION("unknown key lists accepted keys") {
        sp.param = {{"Bogus_Option", 1.0}};
        REQUIRE_THROWS_WITH(parse_asa_options(sp, 1),
                            Catch::Contains("unknown PARAM") &&
                            Catch::Contains("Limit_Acceptances"));
    }
    SECTION("case-insensitive hint") {
        sp.param = {{"limit_acceptances", 10.0}};
        REQUIRE_THROWS_WITH(parse_asa_options(sp, 1),
                            Catch::Contains("case-sensitive"));
    }
    SECTION("excluded upstream name gets a tailored message") {
        sp.param = {{"Coarse_Resolution", 0.1}};
        REQUIRE_THROWS_WITH(parse_asa_options(sp, 1),
                            Catch::Contains("ASA_RESOLUTION"));
    }
    SECTION("User_Tangents is unsupported") {
        sp.param = {{"User_Tangents", 1.0}};
        REQUIRE_THROWS_WITH(parse_asa_options(sp, 1),
                            Catch::Contains("User_Tangents"));
    }
    SECTION("Number_Cost_Samples 0 / -1 invalid") {
        sp.param = {{"Number_Cost_Samples", 0.0}};
        REQUIRE_THROWS(parse_asa_options(sp, 1));
        sp.param = {{"Number_Cost_Samples", -1.0}};
        REQUIRE_THROWS(parse_asa_options(sp, 1));
    }
    SECTION("Reanneal_Cost -1 invalid") {
        sp.param = {{"Reanneal_Cost", -1.0}};
        REQUIRE_THROWS(parse_asa_options(sp, 1));
    }
    SECTION("Sequential_Parameters must be < dim") {
        sp.param = {{"Sequential_Parameters", 1.0}};
        REQUIRE_THROWS(parse_asa_options(sp, 1));
    }
    SECTION("Gener_Block_Max < Gener_Block") {
        sp.param = {{"Gener_Block", 4.0}, {"Gener_Block_Max", 2.0}};
        REQUIRE_THROWS_WITH(parse_asa_options(sp, 1),
                            Catch::Contains("Gener_Block_Max"));
    }
    SECTION("non-integral count") {
        sp.param = {{"Limit_Acceptances", 10.5}};
        REQUIRE_THROWS_WITH(parse_asa_options(sp, 1),
                            Catch::Contains("integer"));
    }
    SECTION("FTOL/XTOL point at ASA's own stops") {
        sp.param.clear();
        sp.stop.ftol_rel = 1e-6;
        REQUIRE_THROWS_WITH(parse_asa_options(sp, 1),
                            Catch::Contains("Cost_Precision"));
        sp.stop.ftol_rel.reset();
        sp.stop.xtol_abs = 1e-6;
        REQUIRE_THROWS(parse_asa_options(sp, 1));
    }
    SECTION("POPULATION / STEP / GRAD_STRATEGY / VECTOR_ARR") {
        sp.param.clear();
        sp.pop = 8;
        REQUIRE_THROWS_WITH(parse_asa_options(sp, 1),
                            Catch::Contains("POPULATION"));
        sp.pop.reset();
        sp.step = 0.1;
        REQUIRE_THROWS_WITH(parse_asa_options(sp, 1),
                            Catch::Contains("STEP"));
        sp.step.reset();
        sp.grad_strategy = tadah::hpo::GradStrategy::Numerical;
        REQUIRE_THROWS_WITH(parse_asa_options(sp, 1),
                            Catch::Contains("GRAD_STRATEGY"));
        sp.grad_strategy.reset();
        sp.vector_storage = 5;
        REQUIRE_THROWS_WITH(parse_asa_options(sp, 1),
                            Catch::Contains("VECTOR_ARR"));
    }
}

TEST_CASE("ASA exit-code names", "[HPO][asa]")
{
    using tadah::hpo::asa_exit_name;
    CHECK(std::string(asa_exit_name(0)) == "NORMAL_EXIT");
    CHECK(std::string(asa_exit_name(1)) == "P_TEMP_TOO_SMALL");
    CHECK(std::string(asa_exit_name(2)) == "C_TEMP_TOO_SMALL");
    CHECK(std::string(asa_exit_name(3)) == "COST_REPEATING");
    CHECK(std::string(asa_exit_name(4)) == "TOO_MANY_INVALID_STATES");
    CHECK(std::string(asa_exit_name(5)) == "IMMEDIATE_EXIT");
    CHECK(std::string(asa_exit_name(7)) == "INVALID_USER_INPUT");
    CHECK(std::string(asa_exit_name(8)) == "INVALID_COST_FUNCTION");
    CHECK(std::string(asa_exit_name(9)) == "INVALID_COST_FUNCTION_DERIV");
    CHECK(std::string(asa_exit_name(-1)) == "CALLOC_FAILED");
    CHECK(std::string(asa_exit_name(6)) == "UNKNOWN_EXIT_CODE");
}

/* ================= caps / dispatch ================================= */

TEST_CASE("ASA caps registry entry", "[HPO][asa][caps]")
{
    const auto& c = tadah::hpo::optimizer_caps("ASA", "ASA");
    CHECK(c.is_global);
    CHECK_FALSE(c.reads_population);
    CHECK_FALSE(c.is_gradient_based);
    CHECK_FALSE(c.accepts_inner);
    CHECK_FALSE(c.accepts_multi_seed);
    CHECK(c.supports_maxeval);
    CHECK(c.supports_maxtime);
    CHECK(c.supports_stopval);
    CHECK_FALSE(c.supports_ftol_rel);
    CHECK_FALSE(c.supports_ftol_abs);
    CHECK_FALSE(c.supports_xtol_rel);
    CHECK_FALSE(c.supports_xtol_abs);
}

TEST_CASE("ASA print_caps names the original option spellings",
          "[HPO][asa][caps]")
{
    std::ostringstream captured;
    auto* old = std::cout.rdbuf(captured.rdbuf());
    OptimizerSpec::print_caps("ASA", "ASA");
    std::cout.rdbuf(old);
    const std::string s = captured.str();
    CHECK(s.find("Temperature_Ratio_Scale") != std::string::npos);
    CHECK(s.find("Limit_Acceptances") != std::string::npos);
    CHECK(s.find("Gener_Block") != std::string::npos);
    CHECK(s.find("single-instance") != std::string::npos);
    CHECK(s.find("BSD-3") != std::string::npos);
}

/* ================= serial driver =================================== */

TEST_CASE("ASA converges on a quadratic bowl", "[HPO][asa]")
{
    auto sp = make_asa_spec({-4.0, -4.0}, {4.0, 4.0});
    set_seed_point(sp, {3.5, -3.5});
    sp.stop.maxeval = 4000;
    const std::vector<double> centre{1.0, -2.0};

    Recorder rec;
    rec.f = [&](const std::vector<double>& x) { return quad_bowl(x, centre); };
    const AsaHostFn fn = rec.fn();
    run_asa_erased(sp, 2, fn);

    REQUIRE(!rec.calls.empty());
    double best = std::numeric_limits<double>::infinity();
    for (const auto& x : rec.calls) best = std::min(best, quad_bowl(x, centre));
    CHECK(best < 0.05);
}

TEST_CASE("ASA determinism: same SEED, same eval sequence", "[HPO][asa]")
{
    auto run_once = [](unsigned seed) {
        auto sp = make_asa_spec({-2.0}, {2.0});
        set_seed_point(sp, {1.5});
        sp.seed = seed;
        sp.stop.maxeval = 300;
        Recorder rec;
        rec.f = [](const std::vector<double>& x) {
            return (x[0] - 0.3) * (x[0] - 0.3);
        };
        const AsaHostFn fn = rec.fn();
        run_asa_erased(sp, 1, fn);
        return rec.calls;
    };
    const auto a = run_once(7u);
    const auto b = run_once(7u);
    const auto c = run_once(8u);
    CHECK(a == b);
    CHECK(a != c);
}

TEST_CASE("ASA MAXEVAL is a hard cap with prompt termination",
          "[HPO][asa]")
{
    auto sp = make_asa_spec({-2.0, -2.0}, {2.0, 2.0});
    set_seed_point(sp, {1.0, 1.0});
    sp.stop.maxeval = 37;

    Recorder rec;
    rec.f = [](const std::vector<double>& x) {
        return x[0] * x[0] + x[1] * x[1];
    };
    const AsaHostFn fn = rec.fn();
    const auto t0 = std::chrono::steady_clock::now();
    run_asa_erased(sp, 2, fn);
    const double wall_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    CHECK(rec.calls.size() == 37);
    CHECK(wall_s < 30.0);
}

TEST_CASE("ASA STOPVAL stops early", "[HPO][asa]")
{
    auto sp = make_asa_spec({-2.0, -2.0}, {2.0, 2.0});
    set_seed_point(sp, {1.9, 1.9});
    sp.stop.maxeval = 100000;
    sp.stop.stopval = 1e-2;
    sp.param = {{"Limit_Generated", 100000.0},
                {"Limit_Acceptances", 100000.0}};

    Recorder rec;
    rec.f = [](const std::vector<double>& x) {
        return x[0] * x[0] + x[1] * x[1];
    };
    const AsaHostFn fn = rec.fn();
    run_asa_erased(sp, 2, fn);
    REQUIRE(!rec.calls.empty());
    CHECK(rec.calls.size() < 100000);
    const bool hit = std::any_of(
        rec.calls.begin(), rec.calls.end(),
        [](const std::vector<double>& x) {
            return x[0] * x[0] + x[1] * x[1] <= 1e-2;
        });
    CHECK(hit);
}

TEST_CASE("ASA MAXTIME stops a slow objective", "[HPO][asa]")
{
    auto sp = make_asa_spec({-2.0}, {2.0});
    set_seed_point(sp, {1.0});
    sp.stop.maxtime = 0.2;
    sp.param = {{"Limit_Generated", 1000000.0}};

    Recorder rec;
    rec.f = [](const std::vector<double>& x) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        return x[0] * x[0];
    };
    const AsaHostFn fn = rec.fn();
    const auto t0 = std::chrono::steady_clock::now();
    run_asa_erased(sp, 1, fn);
    const double wall_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    CHECK(wall_s < 30.0);
    CHECK(rec.calls.size() < 10000);
}

TEST_CASE("ASA integer parameter dims stay integral", "[HPO][asa]")
{
    auto sp = make_asa_spec({-5.0, -5.0}, {5.0, 5.0});
    sp.is_int[0] = true;
    set_seed_point(sp, {2.0, 1.5});
    sp.stop.maxeval = 400;

    Recorder rec;
    rec.f = [](const std::vector<double>& x) {
        return x[0] * x[0] + x[1] * x[1];
    };
    const AsaHostFn fn = rec.fn();
    run_asa_erased(sp, 2, fn);
    REQUIRE(!rec.calls.empty());
    for (const auto& x : rec.calls)
        CHECK(x[0] == std::floor(x[0]));
}

TEST_CASE("ASA hard failures map to invalid states, run survives",
          "[HPO][asa]")
{
    auto sp = make_asa_spec({-2.0}, {2.0});
    set_seed_point(sp, {1.0});
    sp.stop.maxeval = 500;
    sp.failscore = 1e6;
    /* Reanneal_Parameters 0: a failure during reanneal-tangent
     * computation is an upstream ABORT (exit 9, tested below); here we
     * exercise failures during normal sampling only. Bypass the
     * Cost_Precision plateau and slow annealing so the 500-eval budget
     * is the binding stop. */
    sp.param = {{"Maximum_Cost_Repeat", 0.0},
                {"Temperature_Ratio_Scale", 0.5},
                {"Reanneal_Parameters", 0.0}};

    std::atomic<int> n{0};
    Recorder rec;
    rec.f = [&](const std::vector<double>& x) {
        /* every 5th eval fails hard */
        return (++n % 5 == 0) ? 1e6 : x[0] * x[0];
    };
    const AsaHostFn fn = rec.fn();
    run_asa_erased(sp, 1, fn);
    CHECK(rec.calls.size() == 500);  /* failures still consume budget */
}

TEST_CASE("ASA host failure during reanneal tangents aborts gracefully "
          "(upstream INVALID_COST_FUNCTION_DERIV)",
          "[HPO][asa]")
{
    auto sp = make_asa_spec({-2.0}, {2.0});
    set_seed_point(sp, {1.0});
    sp.stop.maxeval = 500;
    sp.failscore = 1e6;
    /* A periodic test, and so the reanneal tangents, every 10
     * acceptances. With the default 100 this run never computed the
     * tangents: it ended at P_TEMP_TOO_SMALL after 469 evaluations, and
     * the check below on the evaluation count passed without the case
     * this test is about. */
    sp.param = {{"Acceptance_Frequency_Modulus", 10.0}};

    std::atomic<int> n{0};
    Recorder rec;
    rec.f = [&](const std::vector<double>& x) {
        return (++n % 5 == 0) ? 1e6 : x[0] * x[0];
    };
    const AsaHostFn fn = rec.fn();
    /* defaults keep Reanneal_Parameters 1 -> the every-5th failure
     * eventually lands inside cost_derivatives(): upstream exits with
     * INVALID_COST_FUNCTION_DERIV. The driver must report it (not
     * throw, not hang). */
    anneal_test::JournalCapture jc;
    REQUIRE_NOTHROW(run_asa_erased(sp, 1, fn));
    CHECK(rec.calls.size() < 500);
    /* Pins: with no stop of tadah's pending, the stop line names the host
     * failure during the tangents and ASA's own exit code.
     * Before the fix: passes as well (a pin of the text that a stop of
     * tadah's inside the tangents must not take); without
     * Acceptance_Frequency_Modulus 10 it fails (P_TEMP_TOO_SMALL). */
    const std::string stop = journal_line(jc.read(), "ASA stopped: ");
    INFO(stop);
    CHECK(stop.find("ASA stopped: a host failure occurred while ASA was "
                    "computing reanneal tangents "
                    "(INVALID_COST_FUNCTION_DERIV)") != std::string::npos);
    CHECK(stop.find("(asa exit: INVALID_COST_FUNCTION_DERIV)") !=
          std::string::npos);
}

TEST_CASE("ASA: a stop that falls in the reanneal tangents is not reported "
          "as INVALID_COST_FUNCTION_DERIV", "[HPO][asa]")
{
    /* Pins: once a stop of tadah's is pending (here MAXEVAL), ASA's
     * callbacks return invalid states without evaluating; inside the
     * reanneal tangents (cost_derivatives(), Reanneal_Parameters 1 by
     * default) ASA takes that for a failed derivative and asa() returns
     * INVALID_COST_FUNCTION_DERIV. The summary and the stop line then say
     * "IMMEDIATE_EXIT requested during the reanneal tangents (asa_out:
     * INVALID_COST_FUNCTION_DERIV)" after the stop's own reason. Periodic
     * tests every 3 acceptances make the tangents frequent, so some
     * MAXEVAL in 8..40 falls inside them (REQUIRE below).
     * Before the fix: those runs read "ASA stopped: MAXEVAL reached after N
     * host eval(s) (asa exit: INVALID_COST_FUNCTION_DERIV)" and the summary
     * "exit=INVALID_COST_FUNCTION_DERIV", the code of a failed evaluation
     * during the tangents. */
    std::size_t in_tangents = 0;
    for (std::uint64_t maxeval = 8; maxeval <= 40; ++maxeval) {
        auto sp = make_asa_spec({-2.0}, {2.0});
        set_seed_point(sp, {1.0});
        sp.stop.maxeval = maxeval;
        sp.param = {{"Acceptance_Frequency_Modulus", 3.0},
                    {"Curvature_0", 1.0}};
        Recorder rec;
        rec.f = [](const std::vector<double>& x) { return x[0] * x[0]; };
        const AsaHostFn fn = rec.fn();
        anneal_test::JournalCapture jc;
        REQUIRE_NOTHROW(run_asa_erased(sp, 1, fn));
        const std::string journal = jc.read();
        const std::string stop = journal_line(journal, "ASA stopped: ");
        const std::string summary = journal_line(journal, "ASA summary: ");
        INFO("MAXEVAL " << maxeval << "\n" << stop << "\n" << summary);
        REQUIRE_FALSE(stop.empty());
        CHECK(stop.find("ASA stopped: MAXEVAL reached after " +
                        std::to_string(maxeval) + " host eval(s)") !=
              std::string::npos);
        for (const std::string* line : {&stop, &summary}) {
            const bool deriv =
                line->find("INVALID_COST_FUNCTION_DERIV") != std::string::npos;
            const bool tangents =
                line->find("IMMEDIATE_EXIT requested during the reanneal "
                           "tangents (asa_out: INVALID_COST_FUNCTION_DERIV)") !=
                std::string::npos;
            CHECK(deriv == tangents);
        }
        if (stop.find("during the reanneal tangents") != std::string::npos)
            ++in_tangents;
    }
    /* The scan exercises the case. */
    REQUIRE(in_tangents > 0);
}

TEST_CASE("ASA exceptions are marshalled out and the mutex is released",
          "[HPO][asa]")
{
    auto sp = make_asa_spec({-2.0}, {2.0});
    set_seed_point(sp, {1.0});
    sp.stop.maxeval = 1000;

    struct Boom {};
    std::atomic<int> n{0};
    const AsaHostFn thrower = [&](const std::vector<double>& x) -> double {
        if (++n >= 20) throw Boom{};
        return x[0] * x[0];
    };
    REQUIRE_THROWS_AS(run_asa_erased(sp, 1, thrower), Boom);

    /* an immediately following run works (single-instance mutex freed) */
    Recorder rec;
    rec.f = [](const std::vector<double>& x) { return x[0] * x[0]; };
    const AsaHostFn fn = rec.fn();
    sp.stop.maxeval = 50;
    REQUIRE_NOTHROW(run_asa_erased(sp, 1, fn));
    CHECK(rec.calls.size() == 50);
}

TEST_CASE("ASA names a shutdown request or a RESTART stagnation stop, not "
          "an exception", "[HPO][asa]")
{
    /* Pins: the stop line names the exception that ends the run by what it
     * is -- a shutdown request (with the signal, when one was seen), the
     * RESTART stagnation stop, or an evaluation's error with its message
     * -- and counts every evaluation the host completed: the one that
     * raised the stagnation stop (or an error) completed and is in
     * loss.tadah, while a shutdown request comes before an evaluation
     * starts. The exception still propagates.
     * Before the fix: every section read "ASA stopped: exception raised
     * during evaluation after 19 host eval(s)". */
    auto sp = make_asa_spec({-2.0}, {2.0});
    set_seed_point(sp, {1.0});
    sp.stop.maxeval = 1000;
    std::atomic<int> n{0};
    std::function<void()> raise;
    const AsaHostFn host = [&](const std::vector<double>& x) -> double {
        if (++n == 20) raise();
        return x[0] * x[0];
    };
    anneal_test::JournalCapture jc;
    std::string expected;
    SECTION("a shutdown request") {
        raise = [] { throw tadah::core::ShutdownRequested{}; };
        CHECK_THROWS_AS(run_asa_erased(sp, 1, host),
                        tadah::core::ShutdownRequested);
        expected = "ASA stopped: shutdown requested after 19 host eval(s)";
    }
    SECTION("the RESTART stagnation stop") {
        raise = [] { throw tadah::hpo::StagnationStop{}; };
        CHECK_THROWS_AS(run_asa_erased(sp, 1, host),
                        tadah::hpo::StagnationStop);
        expected = "ASA stopped: RESTART stagnation stop (no new best within "
                   "AFTER_STAGNATION evaluations) after 20 host eval(s)";
    }
    SECTION("an error") {
        raise = [] { throw std::runtime_error("boom"); };
        CHECK_THROWS_WITH(run_asa_erased(sp, 1, host), "boom");
        expected = "ASA stopped: an evaluation raised an exception (boom) "
                   "after 20 host eval(s)";
    }
    CHECK(n.load() == 20);
    const std::string stop = journal_line(jc.read(), "ASA stopped: ");
    INFO(stop);
    CHECK(stop.find(expected) != std::string::npos);
    CHECK(stop.find("exception raised during evaluation") ==
          std::string::npos);
}

TEST_CASE("ASA writes its upstream journal (asa_out*)", "[HPO][asa]")
{
    auto sp = make_asa_spec({-1.0}, {1.0});
    set_seed_point(sp, {0.5});
    sp.stop.maxeval = 30;
    Recorder rec;
    rec.f = [](const std::vector<double>& x) { return x[0] * x[0]; };
    const AsaHostFn fn = rec.fn();
    run_asa_erased(sp, 1, fn);
    bool found = false;
    for (const auto& e : std::filesystem::directory_iterator(".")) {
        const auto name = e.path().filename().string();
        if (name.rfind("asa_out", 0) == 0) { found = true; break; }
    }
    CHECK(found);
}

TEST_CASE("ASA dlib adapter path (run_asa_impl)", "[HPO][asa]")
{
    using Vec = dlib::matrix<double, 0, 1>;
    struct StubHost {
        mutable std::size_t n = 0;
        double operator()(const Vec& x) const {
            ++n;
            double s = 0.0;
            for (long i = 0; i < x.size(); ++i) s += x(i) * x(i);
            return s;
        }
    };
    auto sp = make_asa_spec({-1.0, -1.0}, {1.0, 1.0});
    set_seed_point(sp, {0.5, -0.5});
    sp.stop.maxeval = 60;
    StubHost host;
    tadah::hpo::run_asa_impl<Vec>(sp, host);
    CHECK(host.n == 60);
}

TEST_CASE("ASA block runs mirror the live cost temperature",
          "[HPO][asa][parallel]")
{
    auto sp = make_asa_spec({-2.0, -2.0}, {2.0, 2.0});
    set_seed_point(sp, {1.0, -1.0});
    sp.num_threads = 2;  /* installs the hook -> mirror refreshed */
    sp.stop.maxeval = 200;
    sp.param = {{"Gener_Block", 4.0}, {"Gener_Block_Max", 4.0}};
    Recorder rec;
    rec.f = [](const std::vector<double>& x) {
        return x[0] * x[0] + x[1] * x[1];
    };
    const AsaHostFn fn = rec.fn();
    run_asa_erased(sp, 2, fn);
    /* the driver resets the mirror to NaN per run; the patched spawn
     * region must have refreshed it with a real (positive) acceptance
     * temperature by the first block */
    CHECK(std::isfinite(tadah_asa_cost_temp));
    CHECK(tadah_asa_cost_temp > 0.0);
}

TEST_CASE("ASA P_TEMP_TOO_SMALL names the underflowed dimension",
          "[HPO][asa]")
{
    /* 1-D at default schedule anneals the parameter temperature below
     * EPS_DOUBLE within a few hundred generated states -> exit 1; the
     * stop line must name the dimension via its OPTIM label. */
    auto sp = make_asa_spec({-2.0}, {2.0});
    set_seed_point(sp, {1.0});
    sp.parameter_labels = {"RCUT2B"};
    sp.stop.maxeval = 5000;
    sp.param = {{"Maximum_Cost_Repeat", 0.0}};

    Recorder rec;
    rec.f = [](const std::vector<double>& x) { return x[0] * x[0]; };
    const AsaHostFn fn = rec.fn();

    std::ostringstream captured;
    auto* old = std::cout.rdbuf(captured.rdbuf());
    run_asa_erased(sp, 1, fn);
    std::cout.rdbuf(old);

    const std::string s = captured.str();
    INFO(s);
    CHECK(s.find("P_TEMP_TOO_SMALL") != std::string::npos);
    CHECK(s.find("[dim 1: RCUT2B]") != std::string::npos);
}

TEST_CASE("ASA keeps sampling after a periodic test that follows a new "
          "best", "[HPO][asa]")
{
    /* The moving average starts at {1, 1, 1}. At acceptance 100 the count
     * since the last new best is 0, and {1, 1, 0} averaged to 2/3, which
     * truncated to a block of 0 states: asa() then looped for ever without
     * generating or evaluating a state, so neither MAXEVAL nor a signal
     * could end the run. With the default Gener_Block_Max 1 the block
     * must stay at one state. */
    auto sp = make_asa_spec({0.0, 0.0}, {1.0, 1.0});
    set_seed_point(sp, {0.5, 0.5});
    sp.stop.maxeval = 400;
    std::string console;
    const auto calls = run_always_improving(sp, 2, console);
    INFO(console);
    CHECK(calls.size() == 400);
    CHECK(console.find("ASA stopped: MAXEVAL reached after 400 host eval(s)")
          != std::string::npos);
}

TEST_CASE("ASA requires an OPTIM dimension", "[HPO][asa]")
{
    OptimizerSpec sp;
    sp.lib = "ASA"; sp.algo = "ASA";
    const AsaHostFn fn = [](const std::vector<double>&) { return 0.0; };
    REQUIRE_THROWS_WITH(run_asa_erased(sp, 0, fn),
                        Catch::Contains("OPTIM dimension"));
}

/* ================= block-parallel hook ============================= */

TEST_CASE("ASA Gener_Block=1 through the hook matches the serial path",
          "[HPO][asa][parallel]")
{
    auto run_once = [](bool force_hook) {
        auto sp = make_asa_spec({-2.0, -2.0}, {2.0, 2.0});
        set_seed_point(sp, {1.0, -1.0});
        sp.seed = 99u;
        sp.stop.maxeval = 400;
        if (force_hook) {
            /* THREADS>1 installs the pool + hook while Gener_Block and
             * its cap stay 1 (a cap above 1 would arm upstream's
             * ADAPTIVE block resizing and legitimately change the
             * sampled sequence). Blocks of one state flow through the
             * hook and must reproduce the serial path exactly. */
            sp.num_threads = 2;
        }
        Recorder rec;
        rec.f = [](const std::vector<double>& x) {
            return (x[0] - 0.5) * (x[0] - 0.5) + x[1] * x[1];
        };
        const AsaHostFn fn = rec.fn();
        run_asa_erased(sp, 2, fn);
        return rec.calls;
    };
    CHECK(run_once(false) == run_once(true));
}

TEST_CASE("ASA Gener_Block>1 results are THREADS-invariant",
          "[HPO][asa][parallel]")
{
    auto run_once = [](unsigned threads) {
        auto sp = make_asa_spec({-2.0, -2.0}, {2.0, 2.0});
        set_seed_point(sp, {1.0, -1.0});
        sp.seed = 1234u;
        sp.num_threads = threads;
        sp.stop.maxeval = 600;
        sp.param = {{"Gener_Block", 4.0}, {"Gener_Block_Max", 4.0}};
        Recorder rec;
        rec.f = [](const std::vector<double>& x) {
            return (x[0] + 0.7) * (x[0] + 0.7) +
                   (x[1] - 0.2) * (x[1] - 0.2);
        };
        const AsaHostFn fn = rec.fn();
        run_asa_erased(sp, 2, fn);
        return rec.sorted_calls();
    };
    const auto t1 = run_once(1);
    const auto t2 = run_once(2);
    const auto t4 = run_once(4);
    CHECK(t1 == t2);
    CHECK(t1 == t4);
    CHECK(!t1.empty());
}

TEST_CASE("ASA block runs keep at least one state per block when the "
          "periodic tests follow new bests", "[HPO][asa][parallel]")
{
    /* Blocks of two states go through the hook (Gener_Block_Max > 1
     * installs it); one worker keeps the evaluation order, and so the
     * costs, deterministic. The moving average starts at {2, 2, 2}; with a
     * periodic test every 7 acceptances, tests that follow new bests feed
     * it counts of 1 and then 0 until it truncates to a block of 0 states
     * well within 300 evaluations (for SEED 1 to 8), and the hook was then
     * called with empty blocks for ever. */
    auto sp = make_asa_spec({0.0, 0.0}, {1.0, 1.0});
    set_seed_point(sp, {0.5, 0.5});
    sp.seed = 1u;
    sp.num_threads = 1;
    sp.stop.maxeval = 300;
    sp.param = {{"Gener_Block", 2.0}, {"Gener_Block_Max", 2.0},
                {"Acceptance_Frequency_Modulus", 7.0}};
    std::string console;
    const auto calls = run_always_improving(sp, 2, console);
    INFO(console);
    CHECK(calls.size() == 300);
    CHECK(console.find("ASA stopped: MAXEVAL reached after 300 host "
                       "eval(s)") != std::string::npos);
}

TEST_CASE("ASA Gener_Mov_Avr 0 keeps every generated block at Gener_Block",
          "[HPO][asa][parallel]")
{
    /* With Gener_Block = Gener_Block_Max = 4 the moving average of the
     * periodic tests still resizes the block, down to 1 (TADAH-PATCH 5
     * floor), on an objective whose every state is a new best. With
     * Gener_Mov_Avr 0 no periodic test resizes it: every block holds 4
     * states, except the last one, which MAXEVAL may cut short. */
    auto sp = make_asa_spec({0.0, 0.0}, {1.0, 1.0});
    set_seed_point(sp, {0.5, 0.5});
    sp.seed = 1u;
    sp.num_threads = 1;
    sp.stop.maxeval = 300;
    sp.param = {{"Gener_Block", 4.0}, {"Gener_Block_Max", 4.0},
                {"Gener_Mov_Avr", 0.0},
                {"Acceptance_Frequency_Modulus", 7.0}};
    std::string console;
    std::vector<std::string> anomalies;
    const auto sizes = block_sizes(sp, 2, console, anomalies);
    INFO(console);
    CHECK(anomalies.empty());
    REQUIRE(sizes.size() > 20);
    std::size_t evaluated = 0;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        INFO("block " << i << " of " << sizes.size());
        if (i + 1 < sizes.size()) CHECK(sizes[i] == 4);
        else CHECK((sizes[i] >= 1 && sizes[i] <= 4));
        evaluated += sizes[i];
    }
    CHECK(evaluated <= 300);
    CHECK(console.find("ASA stopped: MAXEVAL reached after 300 host "
                       "eval(s)") != std::string::npos);

    /* Control: the same run at the default window (Gener_Mov_Avr 3)
     * shrinks blocks, so the count above would see a resize. */
    sp.param.erase("Gener_Mov_Avr");
    std::vector<std::string> control_anomalies;
    const auto resized = block_sizes(sp, 2, console, control_anomalies);
    CHECK(control_anomalies.empty());
    REQUIRE(resized.size() > 1);
    CHECK(std::any_of(resized.begin(), resized.end() - 1,
                      [](std::size_t n) { return n < 4; }));
}

/* ================= stops inside ASA's sampling loops =============== */

/* Outside its main loop ASA samples states in three retry loops: the
 * Number_Cost_Samples states of the initial cost temperature, the random
 * start drawn with User_Initial_Parameters 0 (the default without an INIT
 * seed), and the |Reanneal_Cost| states resampled at a periodic test when
 * |Reanneal_Cost| > 1. Each draws states until one is valid, bounded only
 * by Limit_Invalid_Generated_States. Once a stop is pending, every callback
 * returns an invalid state without evaluating, and upstream ignores
 * Immediate_Exit while it samples the cost temperature: the loop ran on
 * until the invalid-state limit, and with the documented -1 (no limit) for
 * ever, deaf to MAXEVAL, MAXTIME, STOPVAL, an evaluation's exception and
 * signals. A stop must end the run at the callback that raised it, with
 * ASA's IMMEDIATE_EXIT, in every loop and whatever the limit. */

TEST_CASE("ASA stops at MAXEVAL inside the initial cost-temperature "
          "sampling", "[HPO][asa]")
{
    auto bowl = [](const std::vector<double>& x) {
        return x[0] * x[0] + x[1] * x[1];
    };
    /* The default Number_Cost_Samples 5 samples the cost temperature at
     * evaluations 1 to 5; the stop is raised by the 4th callback. */
    for (const double limit : {-1.0, 1000.0}) {
        auto sp = make_asa_spec({-2.0, -2.0}, {2.0, 2.0});
        set_seed_point(sp, {1.0, 1.0});
        sp.stop.maxeval = 3;
        sp.param = {{"Limit_Invalid_Generated_States", limit}};
        Recorder rec;
        rec.f = bowl;
        const std::string console = run_bounded(
            sp, 2, rec,
            "ASA with MAXEVAL 3 and Limit_Invalid_Generated_States " +
                std::to_string(static_cast<int>(limit)));
        INFO("Limit_Invalid_Generated_States " << limit << "\n" << console);
        CHECK(rec.calls.size() == 3);
        CHECK(console.find(stopped_line("MAXEVAL reached", 3)) !=
              std::string::npos);
    }
    /* A block run samples the cost temperature through the same serial
     * callback. */
    auto sp = make_asa_spec({-2.0, -2.0}, {2.0, 2.0});
    set_seed_point(sp, {1.0, 1.0});
    sp.num_threads = 4;
    sp.stop.maxeval = 3;
    sp.param = {{"Limit_Invalid_Generated_States", -1.0},
                {"Gener_Block", 4.0}, {"Gener_Block_Max", 4.0}};
    Recorder rec;
    rec.f = bowl;
    const std::string console = run_bounded(
        sp, 2, rec,
        "ASA block run (THREADS 4, Gener_Block 4) with MAXEVAL 3 and "
        "Limit_Invalid_Generated_States -1");
    INFO(console);
    CHECK(rec.calls.size() == 3);
    CHECK(console.find(stopped_line("MAXEVAL reached", 3)) !=
          std::string::npos);
}

TEST_CASE("ASA stops at STOPVAL, MAXTIME and an evaluation's exception "
          "inside the initial cost-temperature sampling", "[HPO][asa]")
{
    const tadah::hpo::ParamMap no_limit = {
        {"Limit_Invalid_Generated_States", -1.0}};

    SECTION("STOPVAL: the first evaluation meets it") {
        auto sp = make_asa_spec({-2.0}, {2.0});
        set_seed_point(sp, {1.0});
        sp.stop.stopval = 1e9;
        sp.param = no_limit;
        Recorder rec;
        rec.f = [](const std::vector<double>& x) { return x[0] * x[0]; };
        const std::string console = run_bounded(
            sp, 1, rec,
            "ASA with STOPVAL met at the first evaluation and "
            "Limit_Invalid_Generated_States -1");
        INFO(console);
        CHECK(rec.calls.size() == 1);
        CHECK(console.find(stopped_line("STOPVAL reached", 1)) !=
              std::string::npos);
    }
    SECTION("MAXTIME: passes while 100000 cost samples are drawn") {
        auto sp = make_asa_spec({-2.0}, {2.0});
        set_seed_point(sp, {1.0});
        sp.stop.maxtime = 0.05;
        sp.param = no_limit;
        sp.param["Number_Cost_Samples"] = 100000.0;
        Recorder rec;
        rec.f = [](const std::vector<double>& x) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            return x[0] * x[0];
        };
        const std::string console = run_bounded(
            sp, 1, rec,
            "ASA with MAXTIME 0.05 s inside Number_Cost_Samples 100000 and "
            "Limit_Invalid_Generated_States -1");
        INFO(console);
        CHECK(rec.calls.size() < 100000);
        CHECK(console.find(stopped_line("MAXTIME exceeded",
                                        rec.calls.size())) !=
              std::string::npos);
    }
    SECTION("an exception thrown by the third evaluation") {
        auto sp = make_asa_spec({-2.0}, {2.0});
        set_seed_point(sp, {1.0});
        sp.param = no_limit;
        struct Boom {};
        std::atomic<int> n{0};
        Recorder rec;
        rec.f = [&n](const std::vector<double>& x) -> double {
            if (++n == 3) throw Boom{};
            return x[0] * x[0];
        };
        REQUIRE_THROWS_AS(
            run_bounded(sp, 1, rec,
                        "ASA with an exception at the third evaluation and "
                        "Limit_Invalid_Generated_States -1"),
            Boom);
        CHECK(n.load() == 3);
        CHECK(rec.calls.size() == 2);
    }
}

TEST_CASE("ASA stops at MAXEVAL while it draws a random start "
          "(User_Initial_Parameters 0)", "[HPO][asa]")
{
    /* No INIT seed: User_Initial_Parameters resolves to 0, and after the 5
     * cost-temperature samples ASA draws states until one is valid; the
     * 6th callback raises the stop. */
    auto sp = make_asa_spec({-2.0, -2.0}, {2.0, 2.0});
    sp.stop.maxeval = 5;
    sp.param = {{"Limit_Invalid_Generated_States", -1.0}};
    Recorder rec;
    rec.f = [](const std::vector<double>& x) {
        return x[0] * x[0] + x[1] * x[1];
    };
    const std::string console = run_bounded(
        sp, 2, rec,
        "ASA drawing a random start with MAXEVAL 5 and "
        "Limit_Invalid_Generated_States -1");
    INFO(console);
    CHECK(console.find("User_Initial_Parameters=0") != std::string::npos);
    CHECK(rec.calls.size() == 5);
    CHECK(console.find(stopped_line("MAXEVAL reached", 5)) !=
          std::string::npos);
}

TEST_CASE("ASA stops at MAXEVAL inside the Reanneal_Cost resampling of a "
          "periodic test", "[HPO][asa]")
{
    /* On an objective whose every state is a new best ASA accepts every
     * state, so where it evaluates does not depend on the draws: 5
     * cost-temperature samples, the INIT start, then one state per
     * acceptance; with a periodic test every 10 acceptances and
     * Reanneal_Parameters 0 (no tangent evaluations), the test after
     * evaluation 16 resamples Reanneal_Cost 5 states at evaluations 17 to
     * 21. MAXEVAL 16 to 20 raise the stop at each of those callbacks. */
    for (const double limit : {-1.0, 1000.0}) {
        for (std::uint64_t maxeval = 16; maxeval <= 20; ++maxeval) {
            auto sp = make_asa_spec({0.0, 0.0}, {1.0, 1.0});
            set_seed_point(sp, {0.5, 0.5});
            sp.stop.maxeval = maxeval;
            sp.param = {{"Limit_Invalid_Generated_States", limit},
                        {"Acceptance_Frequency_Modulus", 10.0},
                        {"Reanneal_Parameters", 0.0},
                        {"Reanneal_Cost", 5.0}};
            AlwaysImproving obj;
            Recorder rec;
            rec.f = [&obj](const std::vector<double>&) { return obj.next(); };
            const std::string console = run_bounded(
                sp, 2, rec,
                "ASA with MAXEVAL " + std::to_string(maxeval) +
                    " inside the Reanneal_Cost 5 resampling and "
                    "Limit_Invalid_Generated_States " +
                    std::to_string(static_cast<int>(limit)));
            INFO("MAXEVAL " << maxeval << ", Limit_Invalid_Generated_States "
                 << limit << "\n" << console);
            CHECK(rec.calls.size() == maxeval);
            CHECK(console.find(stopped_line("MAXEVAL reached", maxeval)) !=
                  std::string::npos);
        }
    }
}

/* ASA's block evaluations catch host exceptions themselves
 * (AsaRunCtx::stash), so the ReplicaPool's ranking never saw them and the
 * FIRST one won: a StagnationStop caught first masked a concurrent best
 * potential that could not be saved (and a PRESTAGE exploration, which
 * swallows the stop, optimised on for the rest of its budget). The stash
 * keeps the one of the highest rank (run_stop.h). */
TEST_CASE("ASA keeps the most severe of concurrent block exceptions",
          "[HPO][asa][parallel][failure_handling]")
{
    auto run_with = [](std::function<void()> first, std::function<void()> second) {
        auto sp = make_asa_spec({-2.0, -2.0}, {2.0, 2.0});
        set_seed_point(sp, {1.0, -1.0});
        sp.num_threads = 4;
        sp.stop.maxeval = 400;
        sp.param = {{"Gener_Block", 4.0}, {"Gener_Block_Max", 4.0}};
        const auto main_id = std::this_thread::get_id();
        std::atomic<int> block_calls{0}, arrived{0};
        const AsaHostFn fn = [&](const std::vector<double>& x) -> double {
            /* Serial calls (cost-temperature sampling) run on this thread;
             * block evaluations on the pool's workers. */
            if (std::this_thread::get_id() != main_id) {
                const int n = ++block_calls;
                if (n <= 2) {
                    ++arrived;
                    const auto t0 = std::chrono::steady_clock::now();
                    while (arrived.load() < 2 &&
                           std::chrono::steady_clock::now() - t0 <
                               std::chrono::seconds(5))
                        std::this_thread::yield();
                    if (n == 1) first();
                    std::this_thread::sleep_for(std::chrono::milliseconds(30));
                    second();
                }
            }
            return x[0] * x[0] + x[1] * x[1];
        };
        run_asa_erased(sp, 2, fn);
    };
    auto stagnation = [] { throw tadah::hpo::StagnationStop{}; };
    auto shutdown   = [] { throw tadah::core::ShutdownRequested{}; };
    auto durable    = [] { throw tadah::hpo::durable_best::Error("disk full"); };

    CHECK_THROWS_AS(run_with(stagnation, durable), tadah::hpo::durable_best::Error);
    CHECK_THROWS_AS(run_with(stagnation, shutdown), tadah::core::ShutdownRequested);
    CHECK_THROWS_AS(run_with(shutdown, stagnation), tadah::core::ShutdownRequested);
    /* Equal severity: the first stays. */
    CHECK_THROWS_AS(run_with(stagnation, stagnation), tadah::hpo::StagnationStop);
}

TEST_CASE("ASA refuses a box its finite-difference probes cannot fit, before "
          "it evaluates anything", "[HPO][asa][fd_bounds]")
{
    /* ASA's reannealing and exit tangents evaluate (1 + Delta) x, or
     * (1 - Delta) x when that lies outside the box (cost_derivatives). In a
     * box of relative width 1e-6 both lie outside it for Delta_X 0.001:
     * they were evaluated there. */
    auto sp = make_asa_spec({1.0}, {1.000001});
    sp.stop.maxeval = 20;
    std::size_t calls = 0;
    const AsaHostFn eval = [&calls](const std::vector<double>& x) {
        ++calls;
        return (x[0] - 1.0000005) * (x[0] - 1.0000005);
    };
    REQUIRE_THROWS_WITH(run_asa_erased(sp, 1, eval),
                        Catch::StartsWith("ASA:ASA: ASA's finite-difference "
                                          "probes") &&
                            Catch::Contains("the default Delta_X 0.001") &&
                            Catch::Contains("Delta_X up to 4.99999e-07 fits "
                                            "it"));
    CHECK(calls == 0u);
    /* a Delta that fits: the run goes ahead */
    sp.param = {{"Delta_X", 4e-7}};
    REQUIRE_NOTHROW(run_asa_erased(sp, 1, eval));
    CHECK(calls > 0u);
}

namespace {

/* The points ASA's cost_derivatives evaluates to differentiate at x (the
 * probes of one coordinate), computed as asa.c computes them: the tangent
 * (1 + Delta) x, or (1 - Delta) x when that lies outside [lo, hi]; with
 * `curvature` (Curvature_0 0 or -1) the diagonal curvature -- x - 2 Delta
 * |x| and x - Delta |x| when x + Delta |x| lies above the box, x + 2 Delta
 * |x| and x + Delta |x| when x - Delta |x| lies below it, and else the two
 * points it tested, x + Delta |x| and x - Delta |x| (TADAH-PATCH 8: upstream
 * evaluated (1 + Delta) x and (1 - Delta) x there, which can round an ulp
 * beyond a bound whatever the Delta). */
std::vector<double> asa_probes(double x, double delta, bool curvature,
                               double lo, double hi)
{
    std::vector<double> out;
    double d = delta;
    double p = (1.0 + d) * x;
    if (p > hi || p < lo) {
        d = -d;
        p = (1.0 + d) * x;
    }
    out.push_back(p);
    if (!curvature) return out;
    if (x + delta * std::fabs(x) > hi) {
        out.push_back(x - 2.0 * delta * std::fabs(x));
        out.push_back(x - delta * std::fabs(x));
    } else if (x - delta * std::fabs(x) < lo) {
        out.push_back(x + 2.0 * delta * std::fabs(x));
        out.push_back(x + delta * std::fabs(x));
    } else {
        out.push_back(x + delta * std::fabs(x));
        out.push_back(x - delta * std::fabs(x));
    }
    return out;
}

/* The x of [lo, hi] whose probes the test below checks: the bounds, the
 * doubles around the points where a probe meets a bound, and a sweep. */
std::vector<double> asa_probe_points(double lo, double hi, double delta,
                                     std::mt19937_64& rng)
{
    std::vector<double> xs = {lo, hi};
    for (double c : {hi / (1 + delta), lo / (1 - delta), hi / (1 - delta),
                     lo / (1 + delta), hi / (1 + 2 * delta),
                     lo / (1 - 2 * delta), lo / (1 + 2 * delta),
                     hi / (1 - 2 * delta), lo, hi, 0.0}) {
        if (!std::isfinite(c)) continue;
        double x = c;
        for (int k = 0; k < 200; ++k) x = std::nextafter(x, -HUGE_VAL);
        for (int k = 0; k < 401; ++k, x = std::nextafter(x, HUGE_VAL))
            if (x >= lo && x <= hi) xs.push_back(x);
    }
    std::uniform_real_distribution<double> u(lo, hi);
    for (int k = 0; k < 500; ++k) xs.push_back(u(rng));
    return xs;
}

}  // namespace

TEST_CASE("ASA probes stay in the box at the largest Delta the check "
          "accepts", "[HPO][asa][fd_bounds]")
{
    /* asa_probe_fit decided on the real probes, (1 +- Delta) x and x +-
     * Delta |x|; ASA evaluates them in double. At the largest Delta it
     * accepted, a probe ASA computed could lie a few ulps outside the box
     * where the real one met its bound: x - 2 Delta |x| at x = hi below lo
     * in these boxes (the exit curvature, Curvature_0 0). The check now
     * bounds the probes as computed. */
    using tadah::hpo::asa_largest_fitting_delta;
    const std::vector<std::pair<double, double>> boxes = {
        {-0.0077981798734812714, 0.12877021024572782},
        {-0.011093119698940893, 21.577341420066467},
        {-2.8872224549622936, 29.237396712996667},
        {-0.013895641427448169, 0.027586208707990213}};
    std::mt19937_64 rng(11);
    for (const auto& [lo, hi] : boxes) {
        CAPTURE(lo, hi);
        const auto delta = asa_largest_fitting_delta(lo, hi, true);
        REQUIRE(delta);
        std::size_t outside = 0;
        for (double q : asa_probes(hi, *delta, true, lo, hi))
            if (q < lo || q > hi) ++outside;
        CHECK(outside == 0u);
    }
    /* boxes of every sign and width, with and without the curvature */
    std::uniform_real_distribution<double> e(-6.0, 6.0), w(-14.0, 2.0);
    std::size_t cases = 0, outside = 0;
    for (int it = 0; it < 400; ++it) {
        double lo = 0.0, hi = 0.0;
        switch (it % 4) {
            case 0: lo = std::exp(e(rng)); hi = lo * (1 + std::exp(w(rng))); break;
            case 1: hi = -std::exp(e(rng)); lo = hi * (1 + std::exp(w(rng))); break;
            case 2: lo = -std::exp(e(rng)); hi = std::exp(e(rng)); break;
            default: lo = 0.0; hi = std::exp(e(rng)); break;
        }
        for (bool curvature : {false, true}) {
            const auto delta = asa_largest_fitting_delta(lo, hi, curvature);
            if (!delta) continue;
            ++cases;
            for (double x : asa_probe_points(lo, hi, *delta, rng))
                for (double q : asa_probes(x, *delta, curvature, lo, hi))
                    if (q < lo || q > hi) {
                        if (outside == 0)
                            UNSCOPED_INFO("lo " << lo << " hi " << hi
                                          << " Delta " << *delta << " x " << x
                                          << " probe " << q);
                        ++outside;
                    }
        }
    }
    CHECK(cases > 700u);
    CHECK(outside == 0u);
    /* the unit box of OPTIM_SCALE AUTO: the tangents up to Delta 1, the
     * curvature up to 1/2, exactly (asa.rst) */
    CHECK(asa_largest_fitting_delta(0.0, 1.0, false) == std::optional<double>(1.0));
    CHECK(asa_largest_fitting_delta(0.0, 1.0, true) == std::optional<double>(0.5));
}

TEST_CASE("ASA's exit curvature evaluates the points it tests (TADAH-PATCH 8)",
          "[HPO][asa][fd_bounds]")
{
    /* Where x + Delta |x| and x - Delta |x| both lie in the box, the
     * middle branch of cost_derivatives' diagonal curvature (Curvature_0 0
     * or -1) evaluated (1 + Delta) x and (1 - Delta) x instead, which round
     * differently: at x = 0.58058058058058049 in [0.58, 1.16] with the
     * default Delta_X 0.001, x - Delta |x| = 0.58 exactly, but (1 - Delta) x
     * = 0.57999999999999985, an ulp below the box, and ASA evaluated it at
     * its exit (fixed by patching the bundled ASA). The box passes
     * asa_probe_fit, so no Delta could have ruled it out. */
    const double lo = 0.58, hi = 1.16, x0 = 0.58058058058058049;
    const double delta = 0.001;
    /* the middle branch at x0, and upstream's probe outside the box */
    REQUIRE(x0 + delta * std::fabs(x0) <= hi);
    REQUIRE(x0 - delta * std::fabs(x0) >= lo);
    REQUIRE((1.0 - delta) * x0 < lo);
    auto sp = make_asa_spec({lo}, {hi});
    set_seed_point(sp, {x0});
    /* the exit curvature, at the best point: x0, where the loss is 0 */
    sp.param = {{"Curvature_0", 0.0}, {"Limit_Generated", 30.0}};
    Recorder rec;
    rec.f = [x0](const std::vector<double>& x) {
        return (x[0] - x0) * (x[0] - x0);
    };
    const AsaHostFn fn = rec.fn();
    run_asa_erased(sp, 1, fn);
    std::size_t outside = 0;
    bool probed_low = false, probed_high = false;
    for (const auto& x : rec.calls) {
        if (x[0] < lo || x[0] > hi) ++outside;
        if (x[0] == x0 - delta * std::fabs(x0)) probed_low = true;
        if (x[0] == x0 + delta * std::fabs(x0)) probed_high = true;
    }
    CHECK(probed_low);
    CHECK(probed_high);
    CHECK(outside == 0u);
}

namespace {

/* The numbers the starting-point method RANDOM draws for `seed`, in the
 * order it draws them: the coordinates of its points in a one-dimensional
 * unit box (low + u (high - low) = u). */
std::vector<double> random_method_draws(std::uint64_t seed, std::size_t n)
{
    namespace fs = std::filesystem;
    const fs::path cfg = fs::temp_directory_path() /
        ("tadah_asa_stream_" + std::to_string(seed) + "_" +
         std::to_string(std::chrono::steady_clock::now()
                            .time_since_epoch().count()) + ".hpo");
    { std::ofstream(cfg) << "OPTIMIZER\nLIB DLIB\nALGO BFGS\nENDOPTIMIZER\n"; }
    tadah::core::Context ctx;
    const tadah::hpo::HPOTargets trg(ctx, cfg.string());
    std::error_code ec;
    fs::remove(cfg, ec);

    OptimizerSpec unit;
    unit.low = {0.0};
    unit.high = {1.0};
    unit.is_int = {false};
    unit.log_scaled = {false};
    tadah::hpo::InitStrategySpec ispec;
    ispec.chain.push_back({"RANDOM", {std::to_string(n)}});
    const auto method = tadah::hpo::build_init_strategy(ispec);
    const tadah::hpo::InitContext ic{unit, trg, seed};
    std::vector<double> u;
    for (const auto& p : method->generate(ic, n)) u.push_back(p.at(0));
    REQUIRE(u.size() == n);
    return u;
}

/* The plain-C callbacks of asa_testing::run_raw_featured: a recording
 * quadratic bowl, and a generator that hands out `g_draws` in order. */
std::vector<double> g_draws;
std::size_t g_next_draw = 0;
std::vector<std::vector<double>> g_raw_calls;

double raw_bowl(const double* x, long dim, int* cost_flag)
{
    g_raw_calls.emplace_back(x, x + dim);
    *cost_flag = 1;
    return (x[0] - 0.3) * (x[0] - 0.3) + (x[1] + 0.2) * (x[1] + 0.2);
}

double raw_draw()
{
    /* Past the end the stream is not the method's any more: 2 is no
     * uniform number in [0, 1), which the case reports. */
    return g_next_draw < g_draws.size() ? g_draws[g_next_draw++] : 2.0;
}

}  // namespace

/* Every RNG consumer of a seed draws a stream of its own
 * (seed_stream.h): ASA's random numbers for SEED s are not those of the
 * starting-point method RANDOM (an INIT or RESTART block's METHOD RANDOM)
 * for the same seed, which TADAH:RANDOM and ASA both drew -- each seeded
 * std::mt19937_64 with the integer and read it through uniform(0, 1) -- so
 * equal seeds of two keys gave the same numbers (an INIT block's SEED and
 * the optimiser's SEED, a RESTART round's seed and the optimiser's round
 * seed).
 * Pins: ASA under SEED 2026 from (1, -1) in [-2, 2]^2 (Limit_Generated 40,
 * default options otherwise) evaluates other states than the bundled
 * asa() given, with the same options, the RANDOM method's numbers for seed
 * 2026 in order (asa_testing::run_raw_featured), within the first 10.
 * Before the fix: the first 10 states (and every later one) were the same:
 * ASA drew the RANDOM method's numbers. */
TEST_CASE("ASA draws random numbers of its own and not the RANDOM method's for "
          "the same seed", "[HPO][asa][seed]")
{
    constexpr std::uint64_t kSeed = 2026;
    const std::vector<double> lo = {-2.0, -2.0}, hi = {2.0, 2.0},
                              x0 = {1.0, -1.0};

    auto sp = make_asa_spec(lo, hi);
    set_seed_point(sp, x0);
    sp.seed = kSeed;
    sp.stop.maxeval = 1000;
    sp.param = {{"Limit_Generated", 40.0}};
    Recorder rec;
    rec.f = [](const std::vector<double>& x) {
        return quad_bowl(x, {0.3, -0.2});
    };
    run_asa_erased(sp, 2, rec.fn());

    g_draws = random_method_draws(kSeed, 4000);
    g_next_draw = 0;
    g_raw_calls.clear();
    tadah::hpo::asa_testing::run_raw_featured(&raw_bowl, &raw_draw, 2,
                                              lo.data(), hi.data(),
                                              x0.data(), 40);
    /* The method's numbers lasted the reference run. */
    REQUIRE(g_next_draw < g_draws.size());

    REQUIRE(rec.calls.size() >= 10u);
    REQUIRE(g_raw_calls.size() >= 10u);
    const std::vector<std::vector<double>> asa(rec.calls.begin(),
                                               rec.calls.begin() + 10);
    const std::vector<std::vector<double>> method(g_raw_calls.begin(),
                                                  g_raw_calls.begin() + 10);
    CHECK(asa != method);
}
