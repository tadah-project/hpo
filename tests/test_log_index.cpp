// test_log_index.cpp
//
// The radix boundary (1.5.0). Two guards, because per-string checks cannot
// catch a site nobody remembered:
//
//   * unit tests for logidx itself, including the contract that grid()'s cell
//     callback receives the ZERO-BASED index (callers use it to index arrays
//     and std::map, so converting it would reintroduce the off-by-one);
//   * a negative regex sweep over a whole rendered journal, which fails on
//     ANY surviving 0-based slot/pair/walker label.
//
// The machine-artefact side is pinned too: `# HPO_SOURCE_SLOT` is written
// 1-based but MUST parse back to the internal 0-based value, or
// durable_best::publish_if_better throws "payload metadata does not match its
// comparison key" on every publish.

#include "catch2/catch.hpp"

#include <tadah/hpo/durable_best.h>
#include <tadah/hpo/log_index.h>

#include <regex>
#include <sstream>
#include <string>
#include <vector>

using namespace tadah::hpo;

TEST_CASE("logidx: radix, widths and row headers", "[HPO][log][radix]")
{
    CHECK(logidx::ui(0) == 1u);
    CHECK(logidx::ui(15) == 16u);
    CHECK(logidx::uis(0) == "1");

    /* Width follows the RENDERED label: 10 slots print 1..10 and need two
     * columns. Deriving it from `count - 1` (the pre-1.5.0 form) would give
     * one column and misalign the row against its value columns. */
    CHECK(logidx::width(4) == 1);
    CHECK(logidx::width(9) == 1);
    CHECK(logidx::width(10) == 2);
    CHECK(logidx::width(100) == 3);

    CHECK(logidx::range(0, 7, 2) == "[ 1- 8]");
    CHECK(logidx::range(8, 9, 2) == "[ 9-10]");
    CHECK(logidx::range(0, 3, 1) == "[1-4]");
}

TEST_CASE("logidx: from_ui inverts ui and rejects a malformed index",
          "[HPO][log][radix]")
{
    for (std::size_t z : {std::size_t{0}, std::size_t{1}, std::size_t{31}})
        CHECK(logidx::from_ui(logidx::ui(z), "test") == z);
    /* 0 is not a valid 1-based index -- catching it here is what stops a
     * size_t underflow propagating into a comparison key. */
    CHECK_THROWS_WITH(logidx::from_ui(0, "widget id"),
                      Catch::Contains("malformed widget id"));
}

TEST_CASE("logidx::grid hands the cell callback a ZERO-BASED index",
          "[HPO][log][radix]")
{
    std::vector<std::size_t> seen;
    const std::string g = logidx::grid(
        "HDR:", 10, [&](std::ostringstream& o, std::size_t i) {
            seen.push_back(i);
            o << "  " << i;
        });

    REQUIRE(seen.size() == 10u);
    for (std::size_t i = 0; i < seen.size(); ++i) CHECK(seen[i] == i);

    CHECK(g.find("HDR:") == 0);
    CHECK(g.find("[ 1- 8]") != std::string::npos);
    CHECK(g.find("[ 9-10]") != std::string::npos);
}

TEST_CASE("logidx: dim labels agree with the 1-based hpo_search_space.tsv",
          "[HPO][log][radix]")
{
    CHECK(logidx::dim_label(0) == "dim[1]");
    CHECK(logidx::param_label(0) == "param[1]");
}

TEST_CASE("durable_best: HPO_SOURCE_SLOT is 1-based on disk, 0-based in the key",
          "[HPO][log][radix][durable_best]")
{
    /* FORMAT 2 payload: slot 4 on disk is slot 3 in memory. */
    const std::string payload =
        "# HPO_BEST_FORMAT 2\n"
        "# HPO_BEST_LOSS 1.5e+00\n"
        "# HPO_SOURCE_RANK 0\n"
        "# HPO_SOURCE_SLOT 4\n"
        "# HPO_SOURCE_EVAL 42\n";
    const auto key = durable_best::parse_key(payload);
    REQUIRE(key.has_value());
    CHECK(key->slot == 3u);
    CHECK(key->eval == 42u);

    /* FORMAT 1 files stay readable and stay 0-based, so a Key parsed from an
     * old best_pot.tadah still orders correctly against a new one. */
    std::string legacy = payload;
    legacy.replace(legacy.find("FORMAT 2"), 8, "FORMAT 1");
    legacy.replace(legacy.find("SLOT 4"), 6, "SLOT 3");
    const auto old_key = durable_best::parse_key(legacy);
    REQUIRE(old_key.has_value());
    CHECK(old_key->slot == 3u);

    /* A FORMAT 2 payload claiming slot 0 is malformed, not slot -1. */
    std::string bad = payload;
    bad.replace(bad.find("SLOT 4"), 6, "SLOT 0");
    CHECK_FALSE(durable_best::parse_key(bad).has_value());
}

TEST_CASE("radix sweep: a rendered grid carries no 0-based label",
          "[HPO][log][radix]")
{
    /* The guard that catches a site nobody remembered. */
    std::ostringstream doc;
    doc << logidx::grid("slots:", 4,
                        [](std::ostringstream& o, std::size_t i) {
                            o << "  " << logidx::uis(i);
                        })
        << "\n"
        << logidx::grid("pairs:", 3,
                        [](std::ostringstream& o, std::size_t i) {
                            o << "  " << logidx::uis(i);
                        });

    const std::regex zero_based(
        R"(\b(slot|pair|walker)[ =]0\b|\[0-|\(0=coldest)");
    CHECK_FALSE(std::regex_search(doc.str(), zero_based));
    CHECK(doc.str().find("[1-4]") != std::string::npos);
    CHECK(doc.str().find("[1-3]") != std::string::npos);
}
