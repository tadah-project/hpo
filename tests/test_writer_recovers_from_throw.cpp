#include "catch2/catch.hpp"

#include <tadah/hpo/hpo_log.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <typeinfo>

/* Bug 4 regression — the HPO writer thread's exclusive section must
 * survive a per-iter exception without leaving writer_token_ stuck set.
 * The production code's exclusive section in HPO_Host::submit_result is
 * coupled to many heavy-weight runtime objects (HPOTargets, contexts,
 * StructureDB, ofstreams of fixed paths, the moodycamel queue), so this
 * test exercises the SAME try/catch + erase-on-throw + token-clear
 * pattern in isolation — using the real HPOLog singleton — to verify
 * the recovery semantics. The end-to-end test (test_resume_idempotence)
 * exercises the full path through submit_result.
 *
 * What the production writer does, and what this test mirrors:
 *
 *   try {
 *       lock io_mtx_;
 *       drain g_q_ -> io_buf_;
 *       while (io_buf_ has next_flush_) {
 *           try { write iter body; erase + advance }
 *           catch (...) { log warning; erase + advance }   // <-- bug 4 fix
 *       }
 *   } catch (...) { log warning; }
 *   writer_token_.clear();
 *
 * The bug was: no try/catch. A per-iter throw unwound past
 * writer_token_.clear() and every subsequent submit short-circuited.
 * The fix is the inner try/catch (so a single bad iter doesn't stall
 * forever) plus the outer try/catch (belt-and-suspenders). */

namespace {

struct FakeResult {
    std::size_t iter = 0;
    bool        throw_on_write = false;
};

/* Mirror of the production writer's exclusive section. The bug-4
 * recovery pattern lives in the inner and outer try/catch. */
struct WriterSim {
    std::map<std::size_t, FakeResult> io_buf;
    std::atomic_flag                  writer_token = ATOMIC_FLAG_INIT;
    std::atomic<std::size_t>          next_flush{1};
    std::size_t                       last_written_iter = 0;
    std::vector<std::size_t>          dropped_iters;

    void run_writer() {
        if (writer_token.test_and_set(std::memory_order_acquire)) return;

        try {
            while (io_buf.count(next_flush)) {
                try {
                    const auto& e = io_buf[next_flush];
                    if (e.throw_on_write) {
                        throw std::runtime_error(
                            "Key not found in the Context object: NUMERIC");
                    }
                    last_written_iter = e.iter;
                    io_buf.erase(next_flush++);
                } catch (const std::exception& ex) {
                    dropped_iters.push_back(next_flush.load());
                    tadah::hpo::HPOLog::instance().warning(
                        std::string("HPO writer: dropping iter=") +
                        std::to_string(next_flush.load()) +
                        " (exception type=" + typeid(ex).name() + "): " +
                        ex.what());
                    io_buf.erase(next_flush++);
                }
            }
        } catch (const std::exception& ex) {
            tadah::hpo::HPOLog::instance().warning(
                std::string("HPO writer: outer catch: ") + ex.what());
        }

        writer_token.clear(std::memory_order_release);
    }
};

std::filesystem::path scratch_log_path()
{
    return std::filesystem::temp_directory_path() /
        ("tadah_writer_sim_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
         ".log");
}

}  // namespace

TEST_CASE("Writer recovers from a per-iter throw and keeps draining",
          "[writer][bug4]")
{
    auto& log = tadah::hpo::HPOLog::instance();
    log.reset_for_test();
    const auto log_path = scratch_log_path();
    log.open(log_path.string(), "test", "test_cwd", 0);
    REQUIRE(log.is_open());

    WriterSim sim;
    /* Populate 100 iters; iter 50 is the throwing one. */
    for (std::size_t i = 1; i <= 100; ++i) {
        FakeResult r;
        r.iter = i;
        r.throw_on_write = (i == 50);
        sim.io_buf.emplace(i, std::move(r));
    }

    sim.run_writer();

    /* All iters drained. Bug-4 signature would be: writer dies at iter
     * 50, next_flush_ stays at 50, io_buf retains iters 50..100, and
     * writer_token_ stays set. */
    CHECK(sim.io_buf.empty());
    CHECK(sim.next_flush.load() == 101);
    CHECK(sim.last_written_iter == 100);
    CHECK(sim.dropped_iters.size() == 1);
    CHECK(sim.dropped_iters.front() == 50);

    /* Token must be free for the next writer to acquire. */
    CHECK_FALSE(sim.writer_token.test_and_set(std::memory_order_acquire));
    sim.writer_token.clear(std::memory_order_release);

    log.finish("TEST_DONE");

    /* Journal should mention the dropped iter. */
    std::ifstream in(log_path);
    REQUIRE(in.is_open());
    std::string body((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    in.close();
    CHECK(body.find("dropping iter=50") != std::string::npos);
    CHECK(body.find("NUMERIC") != std::string::npos);

    std::error_code ec;
    std::filesystem::remove(log_path, ec);
}

TEST_CASE("Writer drains entire buffer when no iter throws",
          "[writer][bug4]")
{
    auto& log = tadah::hpo::HPOLog::instance();
    log.reset_for_test();
    const auto log_path = scratch_log_path();
    log.open(log_path.string(), "test", "test_cwd", 0);
    REQUIRE(log.is_open());

    WriterSim sim;
    for (std::size_t i = 1; i <= 50; ++i) {
        FakeResult r;
        r.iter = i;
        sim.io_buf.emplace(i, std::move(r));
    }
    sim.run_writer();

    CHECK(sim.io_buf.empty());
    CHECK(sim.next_flush.load() == 51);
    CHECK(sim.dropped_iters.empty());

    log.finish("TEST_DONE");
    std::error_code ec;
    std::filesystem::remove(log_path, ec);
}

TEST_CASE("Writer can be re-driven after a throw without losing forward progress",
          "[writer][bug4]")
{
    auto& log = tadah::hpo::HPOLog::instance();
    log.reset_for_test();
    const auto log_path = scratch_log_path();
    log.open(log_path.string(), "test", "test_cwd", 0);
    REQUIRE(log.is_open());

    WriterSim sim;
    /* First batch — iters 1..10 with iter 5 throwing. */
    for (std::size_t i = 1; i <= 10; ++i) {
        FakeResult r;
        r.iter = i;
        r.throw_on_write = (i == 5);
        sim.io_buf.emplace(i, std::move(r));
    }
    sim.run_writer();
    CHECK(sim.next_flush.load() == 11);
    CHECK(sim.dropped_iters.size() == 1);

    /* Second batch — iters 11..20, none throwing. The writer should
     * pick up where it left off (next_flush_=11) and drain cleanly. */
    for (std::size_t i = 11; i <= 20; ++i) {
        FakeResult r;
        r.iter = i;
        sim.io_buf.emplace(i, std::move(r));
    }
    sim.run_writer();
    CHECK(sim.io_buf.empty());
    CHECK(sim.next_flush.load() == 21);
    CHECK(sim.dropped_iters.size() == 1);   // unchanged

    log.finish("TEST_DONE");
    std::error_code ec;
    std::filesystem::remove(log_path, ec);
}
