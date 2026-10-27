/*
 * Locking tests for kernel/driver correctness fixes that are not
 * already pinned elsewhere:
 *   - the two-pass T0 calibration (journal lines, skip conditions, and
 *     exact budget accounting),
 *   - the ChainObserver seam (pre-swap rung ends, DEO swap attempts,
 *     degenerate flags),
 *   - the dedicated warm-up RNG streams (chain streams untouched by
 *     calibration),
 *   - the interrupted-warm-up best-scale restore,
 *   - the one-time ANNEAL_PT diagnostics block,
 *   - serial AUTO-length budget accounting.
 */

#include "catch2/catch.hpp"

#include <tadah/hpo/algorithms.h>
#include <tadah/hpo/algorithms_pt.h>
#include <tadah/hpo/anneal_core.h>
#include <tadah/hpo/optimizer_spec.h>

#include <dlib/matrix.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Vec = dlib::matrix<double, 0, 1>;

tadah::hpo::OptimizerSpec p2_pt_spec(std::vector<double> low,
                                     std::vector<double> high)
{
    tadah::hpo::OptimizerSpec sp;
    sp.low = std::move(low);
    sp.high = std::move(high);
    sp.is_int.assign(sp.low.size(), false);
    sp.log_scaled.assign(sp.low.size(), false);
    sp.lib = "TADAH";
    sp.algo = "ANNEAL_PT";
    sp.optim_init.assign(1, std::vector<double>(sp.low.size(), 0.0));
    return sp;
}

struct P2Stub {
    std::function<double(const Vec&)> f;
    mutable std::mutex m;
    mutable std::size_t n = 0;
    double operator()(const Vec& x) const
    {
        std::lock_guard<std::mutex> lg(m);
        ++n;
        return f(x);
    }
};

struct P2Log {
    std::filesystem::path path;
    P2Log()
        : path(std::filesystem::temp_directory_path() /
               ("anneal_p2_log_" +
                std::to_string(std::chrono::steady_clock::now()
                                   .time_since_epoch().count()) +
                ".log"))
    {
        tadah::hpo::HPOLog::instance().reset_for_test();
        tadah::hpo::HPOLog::instance().open(path.string(), "test", "/", 0);
    }
    ~P2Log()
    {
        tadah::hpo::HPOLog::instance().finish("test-end");
        tadah::hpo::HPOLog::instance().reset_for_test();
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    std::string read() const
    {
        std::ifstream in(path);
        std::ostringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }
};

/* The journal's stop line carries the driver's own eval count:
 * "... stopped: <reason> after N host eval(s) ...". */
std::size_t journal_eval_count(const std::string& log)
{
    std::smatch m;
    REQUIRE(std::regex_search(
        log, m, std::regex(R"(stopped: .* after (\d+) host eval\(s\))")));
    return static_cast<std::size_t>(std::stoull(m[1]));
}

struct RecordingObserver final : tadah::hpo::anneal::ChainObserver {
    struct Rung { int slot; double T; double f; };
    struct Swap {
        std::size_t epoch, pair;
        bool accepted, degenerate;
        double t_lo, t_hi, f_lo, f_hi;
    };
    std::mutex m;
    std::vector<Rung> rungs;
    std::vector<Swap> swaps;
    void on_rung_end(int slot, double T, const std::vector<double>&,
                     double f) override
    {
        std::lock_guard<std::mutex> lg(m);
        rungs.push_back(Rung{slot, T, f});
    }
    void on_swap_attempt(std::size_t epoch, std::size_t pair, bool accepted,
                         bool degenerate, double t_lo, double t_hi,
                         double f_lo, double f_hi) override
    {
        std::lock_guard<std::mutex> lg(m);
        swaps.push_back(
            Swap{epoch, pair, accepted, degenerate, t_lo, t_hi, f_lo, f_hi});
    }
};

}  // namespace

TEST_CASE("ANNEAL_PT two-pass calibration journals both passes and "
          "accounts every eval",
          "[HPO][anneal_pt][calibration][pass2][lock]")
{
    using namespace tadah::hpo;
    P2Log lr;
    auto sp = p2_pt_spec({-1.0}, {1.0});   /* STEP_MODE FROZEN default */
    sp.seed = 8101;
    sp.num_threads = 1;
    sp.stop.maxeval = 400;
    sp.param["REPLICAS"] = 2.0;

    P2Stub stub{[](const Vec& x) { return x(0) * x(0); }};
    run_anneal_pt_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    CHECK(log.find("ANNEAL_PT calibration pass 2: T_HOT=")
          != std::string::npos);
    /* It was "pass2/pass1 ratio=". */
    CHECK(log.find("T_HOT of pass 2 / pass 1 = ") != std::string::npos);
    CHECK(log.find("ladder T per slot after calibration pass 2")
          != std::string::npos);
    /* Budget accounting is exact across seed, pass-1 probes, init,
     * warm-up, pilot, pass-2 probes, and production epochs. */
    CHECK(stub.n == journal_eval_count(log));
    CHECK(stub.n <= 400);
}

TEST_CASE("ANNEAL_PT skips the second calibration pass when the first "
          "kernel is the production kernel",
          "[HPO][anneal_pt][calibration][pass2]")
{
    using namespace tadah::hpo;

    SECTION("explicit T_HOT/T_COLD: no calibration at all") {
        P2Log lr;
        auto sp = p2_pt_spec({-1.0}, {1.0});
        sp.seed = 8102;
        sp.num_threads = 1;
        sp.stop.maxeval = 200;
        sp.param["REPLICAS"] = 2.0;
        sp.param["T_HOT"] = 1.0;
        sp.param["T_COLD"] = 0.1;
        P2Stub stub{[](const Vec& x) { return x(0) * x(0); }};
        run_anneal_pt_impl<Vec>(sp, stub);
        CHECK(lr.read().find("calibration pass 2") == std::string::npos);
    }
    SECTION("STEP_MODE FIXED: pass 1's kernel is the production kernel") {
        P2Log lr;
        auto sp = p2_pt_spec({-1.0}, {1.0});
        sp.param_raw["STEP_MODE"] = "FIXED";
        sp.seed = 8103;
        sp.num_threads = 1;
        sp.stop.maxeval = 200;
        sp.param["REPLICAS"] = 2.0;
        P2Stub stub{[](const Vec& x) { return x(0) * x(0); }};
        run_anneal_pt_impl<Vec>(sp, stub);
        const std::string log = lr.read();
        CHECK(log.find("T0=") != std::string::npos);   /* pass 1 ran */
        CHECK(log.find("calibration pass 2") == std::string::npos);
    }
    SECTION("CORANA: V starts at the pass-1 kernel and adapts live") {
        P2Log lr;
        auto sp = p2_pt_spec({-1.0}, {1.0});
        sp.param_raw["CORANA"] = "ON";
        sp.seed = 8104;
        sp.num_threads = 1;
        sp.stop.maxeval = 200;
        sp.param["REPLICAS"] = 2.0;
        P2Stub stub{[](const Vec& x) { return x(0) * x(0); }};
        run_anneal_pt_impl<Vec>(sp, stub);
        CHECK(lr.read().find("calibration pass 2") == std::string::npos);
    }
}

TEST_CASE("ANNEAL_PT observer sees pre-swap rung ends and DEO swap "
          "attempts",
          "[HPO][anneal_pt][observer][lock]")
{
    using namespace tadah::hpo;
    P2Log lr;
    RecordingObserver obs;
    auto sp = p2_pt_spec({-1.0}, {1.0});
    sp.param_raw["STEP_MODE"] = "FIXED";
    sp.anneal_observer = &obs;
    sp.seed = 8105;
    sp.num_threads = 2;
    sp.stop.maxeval = 4 + 4 * 10;   /* seed + 3 init + 10 epochs of K=4 */
    sp.param["REPLICAS"] = 4.0;
    sp.param["T_HOT"] = 1.0;
    sp.param["T_COLD"] = 0.1;

    std::atomic<int> k{0};
    P2Stub stub{[&k](const Vec& x) {
        return x(0) * x(0) + 1e-6 * (++k);   /* strictly distinct losses */
    }};
    run_anneal_pt_impl<Vec>(sp, stub);

    /* Every epoch reports each of the K slots exactly once, pre-swap. */
    REQUIRE((obs.rungs.size() % 4 == 0));
    const std::size_t epochs = obs.rungs.size() / 4;
    CHECK(epochs == 10);
    for (std::size_t e = 0; e < epochs; ++e) {
        std::vector<int> slots;
        for (std::size_t s = 0; s < 4; ++s)
            slots.push_back(obs.rungs[4 * e + s].slot);
        std::sort(slots.begin(), slots.end());
        CHECK(slots == std::vector<int>{0, 1, 2, 3});
    }
    /* DEO parity: each attempt's pair parity equals the exchange
     * round's parity; temperatures are the adjacent slot pair; distinct
     * losses mean no attempt is degenerate. */
    REQUIRE_FALSE(obs.swaps.empty());
    for (const auto& s : obs.swaps) {
        CHECK((s.pair % 2 == s.epoch % 2));
        CHECK(s.t_lo < s.t_hi);
        CHECK_FALSE(s.degenerate);
        CHECK(s.f_lo != s.f_hi);
    }
}

TEST_CASE("constant losses mark every observed swap attempt degenerate",
          "[HPO][anneal_pt][observer][degenerate][lock]")
{
    using namespace tadah::hpo;
    P2Log lr;
    RecordingObserver obs;
    auto sp = p2_pt_spec({-1.0}, {1.0});
    sp.param_raw["STEP_MODE"] = "FIXED";
    sp.anneal_observer = &obs;
    sp.seed = 8106;
    sp.num_threads = 1;
    sp.stop.maxeval = 40;
    sp.param["REPLICAS"] = 2.0;
    sp.param["T_HOT"] = 1.0;
    sp.param["T_COLD"] = 0.1;

    P2Stub stub{[](const Vec&) { return 1.0; }};
    run_anneal_pt_impl<Vec>(sp, stub);

    REQUIRE_FALSE(obs.swaps.empty());
    for (const auto& s : obs.swaps) {
        CHECK(s.degenerate);
        CHECK(s.accepted);   /* exp(0) = 1: correct, information-free */
    }
}

TEST_CASE("serial ANNEAL observer sees each rung end on the cooling "
          "schedule",
          "[HPO][anneal][observer][lock]")
{
    using namespace tadah::hpo;
    P2Log lr;
    RecordingObserver obs;
    OptimizerSpec sp;
    sp.low = {-1.0};
    sp.high = {1.0};
    sp.is_int = {false};
    sp.log_scaled = {false};
    sp.lib = "TADAH";
    sp.algo = "ANNEAL";
    sp.optim_init.assign(1, std::vector<double>{0.0});
    sp.anneal_observer = &obs;
    sp.seed = 8107;
    sp.stop.maxeval = 41;
    sp.param["T_INIT"] = 1.0;
    sp.param["T_MIN"] = 0.5;
    sp.param["RT"] = 0.9;
    sp.param["CHAIN_LENGTH"] = 5.0;
    sp.param_raw["REANNEAL"] = "OFF";

    P2Stub stub{[](const Vec& x) { return x(0) * x(0); }};
    run_anneal_impl<Vec>(sp, stub);

    /* T_INIT=1, T_MIN=0.5, RT=0.9 -> rungs at T = 1, .9, .81, ...,
     * stopping when T < 0.5: 7 rungs; slot is always 0 and the
     * temperatures decrease geometrically. */
    REQUIRE(obs.rungs.size() == 7);
    for (std::size_t r = 0; r < obs.rungs.size(); ++r) {
        CHECK(obs.rungs[r].slot == 0);
        CHECK(obs.rungs[r].T == Approx(std::pow(0.9, double(r))));
    }
    CHECK(stub.n == journal_eval_count(lr.read()));
}

TEST_CASE("warm-up and pilot draw only from their own stream: the chain "
          "stream is untouched",
          "[HPO][anneal_pt][step_auto][rng][lock]")
{
    using namespace tadah::hpo;
    auto sp = p2_pt_spec({-1.0}, {1.0});
    sp.stop.maxeval = 100000;
    const auto kb = anneal::Knobs::parse(sp, anneal::AlgoKind::PT);
    REQUIRE(kb.step_mode == anneal_enums::STEP_MODE::FROZEN);

    anneal::ChainState<Vec> c;
    c.rng = anneal::make_stream(42, anneal::SALT_CHAIN, 0);
    const std::mt19937_64 chain_stream_before = c.rng;
    c.x = Vec(1);
    c.x = 0.0;
    c.f = 0.0;
    c.V = {0.1};

    anneal::SharedBudget budget(100000);
    std::atomic<bool> stop{false};
    const std::function<bool()> never = []() { return false; };
    anneal::StopCtx ctx{budget, stop, never, std::nullopt};

    pt_detail::SlotCalibration sc;
    sc.rng = anneal::make_stream(42, anneal::SALT_WARMUP, 0);
    auto host = [](const Vec& x) { return x(0) * x(0); };

    pt_detail::coordinate_warmup(c, 0.5, sp, kb, host, ctx, 100000, sc);
    pt_detail::full_vector_pilot(c, 0.5, sp, kb, host, ctx, 100000, sc);

    CHECK(c.rng == chain_stream_before);
}

TEST_CASE("interrupted coordinate warm-up continues at the best tested "
          "scale, not the last probe",
          "[HPO][anneal_pt][step_auto][interrupt][lock]")
{
    using namespace tadah::hpo;
    auto sp = p2_pt_spec({-1.0}, {1.0});
    sp.stop.maxeval = 100000;
    const auto kb = anneal::Knobs::parse(sp, anneal::AlgoKind::PT);

    anneal::ChainState<Vec> c;
    c.rng = anneal::make_stream(7, anneal::SALT_CHAIN, 0);
    c.x = Vec(1);
    c.x = 0.0;
    c.f = 0.0;
    const double v0 = 0.1;
    c.V = {v0};

    anneal::SharedBudget budget(100000);
    std::atomic<bool> stop{false};
    const std::function<bool()> never = []() { return false; };
    anneal::StopCtx ctx{budget, stop, never, std::nullopt};

    pt_detail::SlotCalibration sc;
    sc.rng = anneal::make_stream(7, anneal::SALT_WARMUP, 0);
    /* A cusp objective at a freezing temperature: every uphill move is
     * rejected, so batch 1 (4 attempts) is a gross-low miss and the
     * probe scale for batch 2 is v0/10. The allowance interrupts batch
     * 2 mid-way; the kernel must return to the best TESTED scale (v0),
     * not the untested v0/10 probe. */
    auto host = [](const Vec& x) { return 1e9 * std::fabs(x(0)); };
    pt_detail::coordinate_warmup(c, 1e-9, sp, kb, host, ctx, 6, sc);

    REQUIRE(sc.dims.size() == 1);
    CHECK((sc.dims[0].status & pt_detail::StepInterrupted) != 0);
    CHECK(c.V[0] == Approx(v0));
    CHECK(sc.dims[0].coordinate_scale == Approx(v0));
}

TEST_CASE("ANNEAL_PT states its fixed diagnostic thresholds once at "
          "startup",
          "[HPO][anneal_pt][log][diagnostics]")
{
    using namespace tadah::hpo;
    P2Log lr;
    auto sp = p2_pt_spec({-1.0}, {1.0});
    sp.param_raw["STEP_MODE"] = "FIXED";
    sp.seed = 8108;
    sp.num_threads = 1;
    sp.stop.maxeval = 30;
    sp.param["REPLICAS"] = 2.0;
    sp.param["T_HOT"] = 1.0;
    sp.param["T_COLD"] = 0.1;
    P2Stub stub{[](const Vec& x) { return x(0) * x(0); }};
    run_anneal_pt_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    CHECK(log.find("ANNEAL_PT diagnostics: step-health acceptance band "
                   "[5%, 50%] per SWAP_WINDOW") != std::string::npos);
    CHECK(log.find("transport window = 100*(REPLICAS-1)")
          != std::string::npos);
    CHECK(log.find("excluded from ladder adaptation") != std::string::npos);
}

TEST_CASE("serial AUTO lengths account the budget exactly",
          "[HPO][anneal][auto][budget][lock]")
{
    using namespace tadah::hpo;
    P2Log lr;
    OptimizerSpec sp;
    sp.low = {-1.0};
    sp.high = {1.0};
    sp.is_int = {false};
    sp.log_scaled = {false};
    sp.lib = "TADAH";
    sp.algo = "ANNEAL";
    sp.optim_init.assign(1, std::vector<double>{0.5});
    sp.seed = 8109;
    sp.stop.maxeval = 300;   /* CHAIN_LENGTH resolves from this budget */

    P2Stub stub{[](const Vec& x) { return x(0) * x(0); }};
    run_anneal_impl<Vec>(sp, stub);

    const std::string log = lr.read();
    CHECK(log.find("ANNEAL auto CHAIN_LENGTH: remaining budget")
          != std::string::npos);
    CHECK(stub.n == journal_eval_count(log));
    CHECK(stub.n <= 300);
}
