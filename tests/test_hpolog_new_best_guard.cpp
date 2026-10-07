#include "catch2/catch.hpp"

#include <tadah/hpo/hpo_log.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

/* Bug 2 regression — HPOLog::log_new_best must reject calls whose loss is
 * not strictly better than cur_best_loss_. Without the guard, an upstream
 * is_best comparator that drifted out of sync with the journal would
 * silently overwrite the journal's running best with a worse value and
 * downstream summaries would report regression as progress.
 *
 * The guard does NOT prevent the spurious comparator-side decision; its
 * purpose is defence-in-depth so a future code path that miscalls
 * log_new_best cannot corrupt the running best. */

namespace {

std::filesystem::path scratch_log_path()
{
    return std::filesystem::temp_directory_path() /
        ("tadah_hpolog_guard_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
         ".log");
}

std::string read_log(const std::filesystem::path& path)
{
    std::ifstream is(path);
    std::ostringstream oss;
    oss << is.rdbuf();
    return oss.str();
}

void log_best(double loss, std::size_t eval_num)
{
    /* No metric flags — simplest call form. */
    tadah::hpo::HPOLog::instance().log_new_best(
        eval_num, loss,
        0.0, false,   // emae
        0.0, false,   // ermse
        0.0, false,   // errmse
        0.0, false,   // frmse
        0.0, false,   // srmse
        0.0,          // wall_ms
        ""            // top_breakdown
    );
}

}  // namespace

TEST_CASE("HPOLog diagnostics state observations without inferring causes",
          "[hpolog][diagnostics]")
{
    auto& log = tadah::hpo::HPOLog::instance();

    const auto zero_path = scratch_log_path();
    log.reset_for_test();
    log.open(zero_path.string(), "test", "test_cwd", 0);
    log.finish("ZERO_EVALS");
    const std::string zero = read_log(zero_path);
    CHECK(zero.find("No evaluations recorded; NaN/Inf rate is unavailable.")
          != std::string::npos);
    CHECK(zero.find("well-configured") == std::string::npos);
    std::error_code zero_ec;
    std::filesystem::remove(zero_path, zero_ec);

    const auto mixed_path = scratch_log_path();
    log.reset_for_test();
    log.open(mixed_path.string(), "test", "test_cwd", 0);
    log.record_eval(1, 1.0, tadah::hpo::EvalLoss::Usable);
    log.record_eval(2, 1.0, tadah::hpo::EvalLoss::NanClamped);
    log.log_nan_clamped(2, "", "");
    log.finish("MIXED");
    const std::string mixed = read_log(mixed_path);
    CHECK(mixed.find("50.0% of evaluations produced a non-finite loss")
          != std::string::npos);
    CHECK(mixed.find("This counter does not identify the cause.")
          != std::string::npos);
    CHECK(mixed.find("evaluation produced NaN/Inf") != std::string::npos);
    CHECK(mixed.find("most evaluations") == std::string::npos);
    std::error_code mixed_ec;
    std::filesystem::remove(mixed_path, mixed_ec);

    CHECK(tadah::hpo::nlopt_result_str(-1) == "FAILURE -- generic failure");
}

TEST_CASE("HPOLog diagnostics judge every unusable loss, not only NaN/Inf "
          "clampings and graded failures", "[hpolog][diagnostics]")
{
    /* The verdict used to read the graded-failure count alone, so a run
     * whose losses were all NaN/Inf clamped or FAILSCORE said "[ok] Every
     * evaluation produced a usable loss".
     * Pins also: an evaluation that threw (EvalLoss::Threw) counts as an
     * unusable loss and under "Evaluations that threw", with a [!] line,
     * and not as a NaN/Inf clamping.
     * Before the fix: does not compile (EvalLoss has no Threw; HPOLog no
     * n_threw()); the host recorded a throw as EvalLoss::NanClamped (the
     * runtime proof is "No usable loss: evaluations that throw count as
     * unusable losses", test_hpo_termination.cpp). */
    using tadah::hpo::EvalLoss;
    auto& log = tadah::hpo::HPOLog::instance();
    struct Case {
        const char* what;
        std::vector<EvalLoss> evals;
        const char* count;     // the "Unusable losses" value
        const char* verdict;   // expected line, or nullptr for none
    };
    const Case cases[] = {
        {"all usable", {EvalLoss::Usable, EvalLoss::Usable},
         "0  (0.0%)", "[ok] Every evaluation produced a usable loss."},
        {"all NaN/Inf clamped", {EvalLoss::NanClamped, EvalLoss::NanClamped},
         "2  (100.0%)", "[!] No evaluation produced a usable loss."},
        {"all FAILSCORE", {EvalLoss::Failed, EvalLoss::Failed, EvalLoss::Failed},
         "3  (100.0%)", "[!] No evaluation produced a usable loss."},
        {"a majority failed",
         {EvalLoss::Usable, EvalLoss::Failed, EvalLoss::NanClamped},
         "2  (66.7%)", "[!] The majority of evaluations failed."},
        {"a minority failed",
         {EvalLoss::Usable, EvalLoss::Usable, EvalLoss::Failed},
         "1  (33.3%)", nullptr},
        /* An evaluation that threw is unusable, and is not a NaN/Inf
         * clamping (the counts below). */
        {"all threw", {EvalLoss::Threw, EvalLoss::Threw},
         "2  (100.0%)", "[!] No evaluation produced a usable loss."},
        {"a usable loss, a NaN/Inf clamping and a throw",
         {EvalLoss::Usable, EvalLoss::NanClamped, EvalLoss::Threw},
         "2  (66.7%)", "[!] The majority of evaluations failed."},
    };
    for (const auto& c : cases) {
        DYNAMIC_SECTION(c.what) {
            const auto path = scratch_log_path();
            log.reset_for_test();
            log.open(path.string(), "test", "test_cwd", 0);
            std::size_t n = 0;
            for (const auto e : c.evals) log.record_eval(++n, 1.0, e);
            const auto count = [&c](EvalLoss k) {
                return static_cast<std::size_t>(
                    std::count(c.evals.begin(), c.evals.end(), k));
            };
            CHECK(log.n_unusable() ==
                  static_cast<std::size_t>(std::count_if(
                      c.evals.begin(), c.evals.end(),
                      [](EvalLoss e) { return e != EvalLoss::Usable; })));
            CHECK(log.n_nan() == count(EvalLoss::NanClamped));
            CHECK(log.n_threw() == count(EvalLoss::Threw));
            log.finish("TEST_DONE");
            const std::string text = read_log(path);
            CHECK(text.find("Unusable losses         : " +
                            std::string(c.count)) != std::string::npos);
            CHECK(text.find("NaN/Inf clampings       : " +
                            std::to_string(count(EvalLoss::NanClamped)) +
                            "  (") != std::string::npos);
            CHECK(text.find("Evaluations that threw  : " +
                            std::to_string(count(EvalLoss::Threw)) +
                            "  (") != std::string::npos);
            CHECK((text.find("raised an exception and scored FAIL_SCORE") !=
                   std::string::npos) == (count(EvalLoss::Threw) > 0));
            for (const char* line :
                 {"[ok] Every evaluation produced a usable loss.",
                  "[!] No evaluation produced a usable loss.",
                  "[!] The majority of evaluations failed."}) {
                const bool expected =
                    c.verdict && std::string(c.verdict) == line;
                INFO(line);
                CHECK((text.find(line) != std::string::npos) == expected);
            }
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    }
}

TEST_CASE("HPOLog diagnostics of a multi-rank run judge this rank's "
          "evaluations and name the rank", "[hpolog][diagnostics]")
{
    /* Each rank keeps its own journal, which counts its own evaluations,
     * while whether the run produced a usable loss is decided over every
     * rank: a rank whose evaluations all failed ends as a completed run
     * when another rank's did not. Its verdict must not read as the
     * run's. */
    using tadah::hpo::EvalLoss;
    auto& log = tadah::hpo::HPOLog::instance();
    struct Case {
        const char* what;
        std::vector<EvalLoss> evals;
        const char* verdict;
    };
    const Case cases[] = {
        {"all usable", {EvalLoss::Usable, EvalLoss::Usable},
         "[ok] Every evaluation on MPI rank 1 (of 2) produced a usable loss."},
        {"none usable", {EvalLoss::Failed, EvalLoss::NanClamped},
         "[!] No evaluation on MPI rank 1 (of 2) produced a usable loss."},
        {"a majority failed",
         {EvalLoss::Usable, EvalLoss::Failed, EvalLoss::Failed},
         "[!] The majority of evaluations on MPI rank 1 (of 2) failed."},
    };
    for (const auto& c : cases) {
        DYNAMIC_SECTION(c.what) {
            const auto path = scratch_log_path();
            log.reset_for_test();
            log.open(path.string(), "test", "test_cwd", 0);
            log.set_rank_scope(1, 2);
            std::size_t n = 0;
            for (const auto e : c.evals) log.record_eval(++n, 1.0, e);
            log.finish("TEST_DONE");
            const std::string text = read_log(path);
            CHECK(text.find(c.verdict) != std::string::npos);
            for (const char* unscoped :
                 {"[ok] Every evaluation produced a usable loss.",
                  "[!] No evaluation produced a usable loss.",
                  "[!] The majority of evaluations failed."}) {
                INFO(unscoped);
                CHECK(text.find(unscoped) == std::string::npos);
            }
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    }
    /* A single-rank journal (the default, and what reset_for_test()
     * restores) keeps the plain verdict. */
    log.reset_for_test();
    const auto path = scratch_log_path();
    log.open(path.string(), "test", "test_cwd", 0);
    log.record_eval(1, 1.0, EvalLoss::Failed);
    log.finish("TEST_DONE");
    const std::string text = read_log(path);
    CHECK(text.find("[!] No evaluation produced a usable loss.") !=
          std::string::npos);
    CHECK(text.find("MPI rank") == std::string::npos);
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST_CASE("HPOLog's milestone SUMMARY counts the evaluations that threw "
          "apart from the NaN rate", "[hpolog][diagnostics]")
{
    /* Pins: the SUMMARY line at evaluation 100 gives the NaN rate of the
     * NaN/Inf clampings alone (2 of 100: 2.0%) and the evaluations that
     * threw as "threw=3"; a window without a throw prints no threw=.
     * Before the fix: does not compile (EvalLoss has no Threw); the host
     * recorded a throw as a NaN/Inf clamping, so the SUMMARY's NaN rate
     * included it. */
    using tadah::hpo::EvalLoss;
    auto& log = tadah::hpo::HPOLog::instance();
    for (const std::size_t threw : {std::size_t{3}, std::size_t{0}}) {
        DYNAMIC_SECTION(threw << " evaluation(s) threw") {
            const auto path = scratch_log_path();
            log.reset_for_test();
            log.open(path.string(), "test", "test_cwd", 0);
            for (std::size_t n = 1; n <= 100; ++n) {
                EvalLoss e = EvalLoss::Usable;
                if (n == 10 || n == 20) e = EvalLoss::NanClamped;
                else if (n > 20 && n <= 20 + threw) e = EvalLoss::Threw;
                log.record_eval(n, 1.0, e);
            }
            log.finish("TEST_DONE");
            const std::string text = read_log(path);
            std::istringstream in(text);
            std::string summary;
            for (std::string line; std::getline(in, line);)
                if (line.find("SUMMARY  ---    100 evals done") !=
                    std::string::npos) { summary = line; break; }
            INFO(text);
            REQUIRE_FALSE(summary.empty());
            CHECK(summary.find("NaN rate=2.0%") != std::string::npos);
            CHECK((summary.find("threw=") != std::string::npos) ==
                  (threw > 0));
            if (threw > 0)
                CHECK(summary.find("threw=3") != std::string::npos);
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    }
}

TEST_CASE("HPOLog::log_new_best rejects non-improving losses", "[hpolog][bug2]")
{
    auto& log = tadah::hpo::HPOLog::instance();
    log.reset_for_test();
    const auto path = scratch_log_path();
    log.open(path.string(), "test", "test_cwd", 0);
    REQUIRE(log.is_open());

    /* First best at 10.0 — accepted (cur_best defaults to numeric_limits::max). */
    log_best(10.0, 1);
    CHECK(log.best_loss() == Approx(10.0));
    CHECK(log.best_eval() == 1u);

    /* Worse loss — must be rejected; running best must not regress. */
    log_best(20.0, 2);
    CHECK(log.best_loss() == Approx(10.0));
    CHECK(log.best_eval() == 1u);

    /* Equal loss — must also be rejected (strict-improvement guard). */
    log_best(10.0, 3);
    CHECK(log.best_loss() == Approx(10.0));
    CHECK(log.best_eval() == 1u);

    /* Strictly better loss — accepted. */
    log_best(5.0, 4);
    CHECK(log.best_loss() == Approx(5.0));
    CHECK(log.best_eval() == 4u);

    log.finish("TEST_DONE");
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST_CASE("HPOLog I/O diagnostics vouch for the best potential only when it "
          "was persisted", "[hpolog][diagnostics][io_retry]")
{
    /* The note claimed "the best potential ... is complete" on a
     * run that ended because it could not be persisted. */
    auto& log = tadah::hpo::HPOLog::instance();
    auto run = [&](bool persist_failed, bool with_best) {
        const auto path = scratch_log_path();
        log.reset_for_test();
        log.open(path.string(), "test", "test_cwd", 0);
        log.record_eval(1, 1.0, tadah::hpo::EvalLoss::Usable);
        if (with_best) log_best(1.0, 1);
        log.record_io_retry_writer(2);
        log.record_rows_lost(3);
        if (persist_failed) log.record_best_persistence_failure();
        log.finish(persist_failed ? "FAILED -- best-potential persistence: x"
                                  : "COMPLETED");
        const std::string text = read_log(path);
        std::error_code ec;
        std::filesystem::remove(path, ec);
        return text;
    };

    const std::string ok = run(false, true);
    CHECK(ok.find("Transient I/O retries   : 2 (best-potential 0, per-eval "
                  "logs/archives 2); rows lost 3; archives lost 0") != std::string::npos);
    CHECK(ok.find("best potential is published separately and is complete")
          != std::string::npos);

    const std::string failed = run(true, true);
    CHECK(failed.find("is complete") == std::string::npos);
    CHECK(failed.find("The best potential could not be persisted either")
          != std::string::npos);

    const std::string none = run(false, false);
    CHECK(none.find("is complete") == std::string::npos);
    CHECK(none.find("not written after retrying; see the WARNING entries.")
          != std::string::npos);
    log.reset_for_test();
}
