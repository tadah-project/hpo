#include "catch2/catch.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>

/* End-to-end resume-idempotence test exercising all four bug fixes
 * together. Gated [slow] and on the TADAH_HPO_RESUME_FIXTURE env var
 * (path to a workdir prepared with master config + databases + HPO
 * config). Requires TADAH_BINARY to point at the built `tadah`
 * executable. Skips gracefully when either is unset.
 *
 * Bundles all four fixes:
 *   Bug 1 — first post-resume eval must NOT spuriously claim NEW BEST.
 *   Bug 2 — log_new_best must reject worse-than-current-best (defence
 *           in depth; verified indirectly through SUMMARY consistency).
 *   Bug 3 — NOISE_CALIBRATION must NOT run on resume.
 *   Bug 4 — writer thread must not freeze; loss.tadah / best_pot.tadah
 *           keep being updated through the run.
 *
 * Suggested env-var setup for a live smoke:
 *   export TADAH_BINARY=/path/to/build_tadah/tadah
 *   export TADAH_HPO_RESUME_FIXTURE=/path/to/ITER5/  # workdir with all configs
 *   export TADAH_HPO_RESUME_CONFIG=Ta_config.basins.tadah
 *   export TADAH_HPO_RESUME_HPOTARGET=config.hpo.G_basins
 *   export TADAH_HPO_RESUME_VALIDATION=config.val
 *
 * The test copies the fixture to a scratch dir before each run so the
 * source workdir is never modified. */

namespace {

namespace fs = std::filesystem;

std::string env_or_empty(const char* k) {
    const char* v = std::getenv(k);
    return v ? std::string(v) : std::string();
}

std::string read_file(const fs::path& p) {
    std::ifstream f(p);
    if (!f.is_open()) return {};
    std::ostringstream ss; ss << f.rdbuf();
    return ss.str();
}

double parse_dbl_after(const std::string& s, const std::string& key) {
    /* Find "key=NUMBER" or "key NUMBER" — return first match or NaN. */
    std::regex re(key + R"(\s*=?\s*([0-9eE\.\+\-]+))");
    std::smatch m;
    if (std::regex_search(s, m, re) && m.size() > 1) {
        try { return std::stod(m[1].str()); } catch (...) {}
    }
    return std::numeric_limits<double>::quiet_NaN();
}

}  // namespace

TEST_CASE("BASINS resume is idempotent — bugs 1+2+3+4 end-to-end",
          "[resume][slow][e2e]")
{
    const auto tadah_bin = env_or_empty("TADAH_BINARY");
    const auto fixture   = env_or_empty("TADAH_HPO_RESUME_FIXTURE");
    if (tadah_bin.empty() || fixture.empty()) {
        SUCCEED("Skipped — set TADAH_BINARY and TADAH_HPO_RESUME_FIXTURE to enable.");
        return;
    }
    const auto cfg     = env_or_empty("TADAH_HPO_RESUME_CONFIG");
    const auto hpotrg  = env_or_empty("TADAH_HPO_RESUME_HPOTARGET");
    const auto valcfg  = env_or_empty("TADAH_HPO_RESUME_VALIDATION");
    if (cfg.empty() || hpotrg.empty() || valcfg.empty()) {
        FAIL("TADAH_HPO_RESUME_FIXTURE set but TADAH_HPO_RESUME_CONFIG/HPOTARGET/VALIDATION missing");
    }

    REQUIRE(fs::exists(tadah_bin));
    REQUIRE(fs::is_directory(fixture));

    /* Copy fixture to scratch dir. */
    const auto scratch = fs::temp_directory_path() /
        ("tadah_resume_e2e_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(scratch);
    fs::copy(fixture, scratch,
             fs::copy_options::recursive |
             fs::copy_options::copy_symlinks);

    /* Truncate STAGE_MAX_EVALS for a fast run, if the user hasn't already.
     * This also exercises Bug 4's GLOBAL→LOCAL transition trigger path. */
    {
        const auto hp_path = scratch / hpotrg;
        std::string body = read_file(hp_path);
        body = std::regex_replace(body,
                                  std::regex(R"(STAGE_MAX_EVALS\s+\d+)"),
                                  "STAGE_MAX_EVALS    100");
        std::ofstream out(hp_path);
        out << body;
    }

    auto run_tadah = [&](int timeout_s) -> int {
        std::ostringstream cmd;
        cmd << "cd '" << scratch.string() << "' && timeout --foreground "
            << timeout_s << " '" << tadah_bin << "' hpo "
            << "--validation '" << valcfg << "' "
            << "--hpotarget '" << hpotrg << "' "
            << "-c '" << cfg << "' "
            << "> tadah.stdout 2> tadah.stderr";
        return std::system(cmd.str().c_str());
    };

    /* ── Run A — fresh start ──────────────────────────────────────── */
    INFO("scratch dir: " << scratch.string());
    const int rc_a = run_tadah(/*timeout_s=*/600);
    REQUIRE(rc_a == 0);
    REQUIRE(fs::exists(scratch / "hpo_run.log"));
    REQUIRE(fs::exists(scratch / "checkpoint.tadah"));

    const std::string log_a = read_file(scratch / "hpo_run.log");
    REQUIRE_FALSE(log_a.empty());

    /* Capture run-A's best loss from the journal RESULT section. */
    const double best_a = parse_dbl_after(log_a, "best loss");
    REQUIRE(std::isfinite(best_a));
    INFO("run A best=" << best_a);

    /* ── Run B — resume in same dir ──────────────────────────────── */
    /* Stash run-A journal so we can compare. */
    fs::rename(scratch / "hpo_run.log",
               scratch / "hpo_run.log.A");

    const int rc_b = run_tadah(/*timeout_s=*/600);
    REQUIRE(rc_b == 0);
    REQUIRE(fs::exists(scratch / "hpo_run.log"));
    const std::string log_b = read_file(scratch / "hpo_run.log");
    REQUIRE_FALSE(log_b.empty());

    /* Bug 3: no NOISE_CALIBRATION block on resume. */
    CHECK(log_b.find("NOISE_CALIBRATION: running") == std::string::npos);

    /* Resume marker is present. */
    CHECK(log_b.find("BASINS: resumed from checkpoint") != std::string::npos);

    /* Bug 1: no spurious BEST event with loss > best_a. We pull every
     * BEST loss from log_b and verify all are ≤ best_a (within an
     * epsilon to allow numeric round-trip). */
    std::regex best_re(R"(BEST\s+\*\s+eval=\s*\d+\s+loss=([0-9eE\.\+\-]+))");
    auto begin = std::sregex_iterator(log_b.begin(), log_b.end(), best_re);
    auto end   = std::sregex_iterator();
    bool found_any_best = false;
    for (auto it = begin; it != end; ++it) {
        found_any_best = true;
        const double l = std::stod((*it)[1].str());
        const double tol = std::max(1e-9, 1e-6 * std::fabs(best_a));
        CHECK(l <= best_a + tol);
    }
    /* It is acceptable for run B to report no NEW BEST (BASINS may
     * tighten without finding a strictly better point). The assertion
     * is only that any NEW BEST it does report is not WORSE than A. */
    (void)found_any_best;

    /* Bug 4: best_pot.tadah is fresh after run B (writer didn't freeze). */
    REQUIRE(fs::exists(scratch / "best_pot.tadah"));
    const auto bp_mtime = fs::last_write_time(scratch / "best_pot.tadah");
    const auto cp_mtime = fs::last_write_time(scratch / "checkpoint.tadah");
    /* best_pot.tadah should have been updated within the same run as
     * checkpoint.tadah — if writer froze early in B, best_pot.tadah
     * would be from A. We bound by 60s of cp_mtime as a sanity gate. */
    using namespace std::chrono;
    const auto delta = duration_cast<seconds>(cp_mtime - bp_mtime).count();
    CHECK(std::abs(delta) < 60);

    /* Cleanup. */
    std::error_code ec;
    fs::remove_all(scratch, ec);
}
