/*
 * Unit tests for LIB ASA / ALGO ASA (vendored Ingber ASA-30.53).
 *
 * Most tests drive `run_asa_erased` with a std::vector lambda (no
 * HPO_Host); the dlib adapter path is covered once via run_asa_impl.
 * The golden-neutrality gate lives in test_asa_golden.cpp.
 */

#include "catch2/catch.hpp"

#include <tadah/hpo/algorithms_asa.h>
#include <tadah/hpo/asa_options.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/optimizer_spec.h>

#include <dlib/matrix.h>

extern "C" {
#include <asa/tadah_asa_hook.h>  /* tadah_asa_cost_temp mirror */
}

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <sstream>
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
        {"Include_Integer_Parameters", 1},
        {"User_Initial_Parameters", 1},
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
    SECTION("User_Initial_Parameters contradictions") {
        sp.param = {{"User_Initial_Parameters", 1.0}};
        REQUIRE_THROWS_WITH(parse_asa_options(sp, 1),
                            Catch::Contains("INIT"));
        set_seed_point(sp, {0.5});
        sp.param = {{"User_Initial_Parameters", 0.0}};
        REQUIRE_THROWS_WITH(parse_asa_options(sp, 1),
                            Catch::Contains("INIT"));
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
    CHECK_FALSE(c.is_population);
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
    CHECK(c.ladder_ftol == tadah::hpo::OptimizerCaps::FtolKind::None);
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
    REQUIRE_NOTHROW(run_asa_erased(sp, 1, fn));
    CHECK(rec.calls.size() < 500);
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
