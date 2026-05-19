/*
 * Guards tests/integration/strategy_matrix.manifest against grammar
 * drift: every entry's <hpo body> must parse cleanly under the current
 * OPTIMIZER block grammar.
 *
 * The integration matrix itself (strategy_matrix.sh) is opt-in and
 * gated on TADAH_HPO_INTEGRATION_BUNDLE, so without this test a stale
 * manifest goes unnoticed until someone runs the slow suite. This test
 * needs no bundle -- it only exercises the parser.
 */

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/hpo/hpo_targets.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

/* Expand the manifest's \n escapes -- mirrors `printf '%b'` in
 * strategy_matrix.sh. The manifest only ever uses \n. */
std::string expand_escapes(const std::string& in)
{
    std::string out;
    out.reserve(in.size());
    for (std::size_t i = 0; i < in.size(); ++i) {
        if (in[i] == '\\' && i + 1 < in.size() && in[i + 1] == 'n') {
            out.push_back('\n');
            ++i;
        } else {
            out.push_back(in[i]);
        }
    }
    return out;
}

std::string trim(const std::string& s)
{
    const auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    const auto b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

}  // namespace

TEST_CASE("strategy_matrix.manifest entries parse under current grammar",
          "[HPO][integration][manifest]")
{
    namespace fs = std::filesystem;
    /* __FILE__ is an absolute path (CMake GLOBs the test sources), so
     * the manifest is found regardless of the test's working dir. */
    const fs::path manifest =
        fs::path(__FILE__).parent_path() / "integration" /
        "strategy_matrix.manifest";

    std::ifstream in(manifest);
    REQUIRE(in.good());

    std::size_t entries = 0;
    std::string line;
    while (std::getline(in, line)) {
        const std::string t = trim(line);
        if (t.empty() || t[0] == '#') continue;

        const auto bar = t.find('|');
        REQUIRE(bar != std::string::npos);
        const std::string name = trim(t.substr(0, bar));
        const std::string body = expand_escapes(t.substr(bar + 1));
        REQUIRE_FALSE(name.empty());

        const auto cfg = fs::temp_directory_path() /
            ("tadah_manifest_" + name + "_" +
             std::to_string(std::chrono::steady_clock::now()
                                .time_since_epoch().count()) + ".hpo");
        {
            std::ofstream os(cfg);
            REQUIRE(os.good());
            os << body;
        }

        tadah::core::Context ctx;
        INFO("manifest entry: " << name);
        CHECK_NOTHROW(tadah::hpo::HPOTargets(ctx, cfg.string()));

        std::error_code ec;
        fs::remove(cfg, ec);
        ++entries;
    }
    /* Keep this in sync when entries are added to the manifest. */
    CHECK(entries == 6);
}
