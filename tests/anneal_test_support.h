#pragma once
/* Shared fixtures for the annealing test suites: journal capture, a
 * thread-safe counting stub host, a recording ChainObserver, and spec
 * builders. Header-only; used by the P2 lock tests and the statistical
 * validation suites. */

#include <tadah/hpo/anneal_core.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/optimizer_spec.h>

#include <dlib/matrix.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace anneal_test {

using Vec = dlib::matrix<double, 0, 1>;

inline tadah::hpo::OptimizerSpec pt_spec(std::vector<double> low,
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

inline tadah::hpo::OptimizerSpec serial_spec(std::vector<double> low,
                                             std::vector<double> high)
{
    auto sp = pt_spec(std::move(low), std::move(high));
    sp.algo = "ANNEAL";
    return sp;
}

/* Thread-safe stub host counting its calls. */
struct CountingHost {
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

/* Captures the run journal into a temp file for string assertions. */
struct JournalCapture {
    std::filesystem::path path;
    JournalCapture()
        : path(std::filesystem::temp_directory_path() /
               ("anneal_journal_" +
                std::to_string(std::chrono::steady_clock::now()
                                   .time_since_epoch().count()) +
                ".log"))
    {
        tadah::hpo::HPOLog::instance().reset_for_test();
        tadah::hpo::HPOLog::instance().open(path.string(), "test", "/", 0);
    }
    ~JournalCapture()
    {
        tadah::hpo::HPOLog::instance().finish("test-end");
        tadah::hpo::HPOLog::instance().reset_for_test();
        if (std::getenv("TADAH_TEST_DUMP_LOG"))
            std::cout << "\n===== " << path.string() << " =====\n" << read();
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

/* The journal's stop line carries the driver's own eval count. */
inline std::size_t journal_eval_count(const std::string& log)
{
    std::smatch m;
    if (!std::regex_search(
            log, m, std::regex(R"(stopped: .* after (\d+) host eval\(s\))")))
        return static_cast<std::size_t>(-1);
    return static_cast<std::size_t>(std::stoull(m[1]));
}

/* Recording observer: every pre-swap rung end and every exchange
 * attempt (thread-safe; rung ends arrive from pool workers). */
struct RecordingObserver final : tadah::hpo::anneal::ChainObserver {
    struct Rung { int slot; double T; std::vector<double> x; double f; };
    struct Swap {
        std::size_t epoch, pair;
        bool accepted, degenerate;
        double t_lo, t_hi, f_lo, f_hi;
    };
    std::mutex m;
    std::vector<Rung> rungs;
    std::vector<Swap> swaps;
    bool keep_x = false;   /* x copies are large; opt in when needed */
    void on_rung_end(int slot, double T, const std::vector<double>& x,
                     double f) override
    {
        std::lock_guard<std::mutex> lg(m);
        rungs.push_back(Rung{slot, T,
                             keep_x ? x : std::vector<double>{}, f});
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

}  // namespace anneal_test
