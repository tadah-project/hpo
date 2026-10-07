// Catch2 v2 unit tests for the per-iter health monitor (Phase 6).
//
// Verifies:
//   * parse_health_log_mode handles full / summary / off (case-insensitive)
//   * compute_iter_health partitions weights by ColumnLayout, computes
//     max-abs / median-abs / nz-fraction per block, and sets
//     weights_have_nan_inf when a non-finite slips through.
//   * IterHealthMonitor in summary mode rate-limits warnings to one
//     per (block, kind) per window.
//   * IterHealthMonitor in full mode emits every offending iter.
//   * Off mode is a no-op.
//   * Every weight is checked for NaN/Inf, whatever the layout; a layout
//     that does not describe the weights is reported as such.
//   * Records from concurrent evaluations are all counted.

#include "catch2/catch.hpp"

#include <tadah/core/aed_type.h>
#include <tadah/core/core_types.h>
#include <tadah/core/logger.h>
#include <tadah/hpo/iter_health.h>
#include <tadah/mlip/design_matrix/column_block.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace tadah::hpo;
using tadah::core::t_type;
using tadah::mlip::ColumnBlock;
using tadah::mlip::ColumnLayout;

static ColumnLayout make_two_block_layout() {
  ColumnLayout L;
  ColumnBlock a; a.descriptor_label = "D2"; a.col_offset = 0;  a.col_count = 4;
  ColumnBlock b; b.descriptor_label = "DM"; b.col_offset = 4;  b.col_count = 6;
  L.blocks.push_back(a);
  L.blocks.push_back(b);
  return L;
}

/* What core::Logger prints on standard output at WARNING (the level of the
 * tadah CLI; a test process starts at ERROR) while this lives. */
struct LoggerCapture {
  std::ostringstream out;
  std::streambuf* saved_buf;
  tadah::core::LogLevel saved_level;
  LoggerCapture()
      : saved_buf(std::cout.rdbuf(out.rdbuf())),
        saved_level(tadah::core::Logger::instance().getLogLevel()) {
    tadah::core::Logger::instance().setLogLevel(tadah::core::LogLevel::WARNING);
  }
  ~LoggerCapture() {
    tadah::core::Logger::instance().setLogLevel(saved_level);
    std::cout.rdbuf(saved_buf);
  }
  std::string text() const { return out.str(); }
};

TEST_CASE("iter_health: parse_health_log_mode", "[iter_health]") {
  REQUIRE(parse_health_log_mode("off")     == HealthLogMode::Off);
  REQUIRE(parse_health_log_mode("OFF")     == HealthLogMode::Off);
  REQUIRE(parse_health_log_mode("summary") == HealthLogMode::Summary);
  REQUIRE(parse_health_log_mode("Full")    == HealthLogMode::Full);
  REQUIRE_THROWS(parse_health_log_mode("verbose"));
}

TEST_CASE("iter_health: compute partitions weights by layout", "[iter_health]") {
  auto layout = make_two_block_layout();
  // 10-dim weight vector: D2 block all 1.0 (max=1, median=1, nz=1.0),
  // DM block 4 zeros + 2 large => max=10, median=0, nz=2/6.
  t_type w(10);
  for (std::size_t i = 0; i < 4; ++i) w[i] = 1.0;
  for (std::size_t i = 4; i < 8; ++i) w[i] = 0.0;
  w[8] = 10.0; w[9] = -7.0;

  auto rec = compute_iter_health(/*iter=*/42, w, layout);
  REQUIRE(rec.iter == 42);
  REQUIRE(rec.block_max_abs.size() == 2);
  REQUIRE(rec.block_max_abs[0] == Approx(1.0));
  REQUIRE(rec.block_max_abs[1] == Approx(10.0));
  REQUIRE(rec.block_nz_fraction[0] == Approx(1.0));
  REQUIRE(rec.block_nz_fraction[1] == Approx(2.0 / 6.0));
  REQUIRE_FALSE(rec.weights_have_nan_inf);
}

TEST_CASE("iter_health: NaN flips weights_have_nan_inf", "[iter_health]") {
  auto layout = make_two_block_layout();
  t_type w(10);
  for (std::size_t i = 0; i < 10; ++i) w[i] = 1.0;
  w[5] = std::numeric_limits<double>::quiet_NaN();
  auto rec = compute_iter_health(0, w, layout);
  REQUIRE(rec.weights_have_nan_inf);
}

TEST_CASE("iter_health: zero block triggers warning", "[iter_health]") {
  auto layout = make_two_block_layout();
  t_type w(10);
  // D2 block alive; DM block all zero.
  for (std::size_t i = 0; i < 4; ++i) w[i] = 1.0;
  for (std::size_t i = 4; i < 10; ++i) w[i] = 0.0;
  auto rec = compute_iter_health(0, w, layout);

  IterHealthMonitor mon(HealthLogMode::Full);
  mon.record(rec, layout);
  REQUIRE(mon.warning_count(1, "max_abs") == 1);
  REQUIRE(mon.warning_count(0, "max_abs") == 0);
}

TEST_CASE("iter_health: summary mode rate-limits per window", "[iter_health]") {
  auto layout = make_two_block_layout();
  t_type w(10);
  for (std::size_t i = 0; i < 10; ++i) w[i] = 0.0;  // both blocks zero

  IterHealthMonitor mon(HealthLogMode::Summary, /*window=*/100);
  for (std::size_t it = 0; it < 50; ++it) {
    auto rec = compute_iter_health(it, w, layout);
    mon.record(rec, layout);
  }
  // 50 evals, all dead, summary mode with window=100 -> exactly 1
  // warning per (block, kind) so far.
  REQUIRE(mon.warning_count(0, "max_abs") == 1);
  REQUIRE(mon.warning_count(1, "max_abs") == 1);

  // Cross the window boundary -> one more warning per (block, kind).
  for (std::size_t it = 50; it < 200; ++it) {
    auto rec = compute_iter_health(it, w, layout);
    mon.record(rec, layout);
  }
  // After iter=100 the window resets; iter=150 also resets again.
  // Implementation emits when iter >= last + window. Sequence: warn at
  // iter=0 (counts++). Skip until iter=100 (=0+100). Skip until iter=200.
  // We saw iters 0..199 -> warnings at 0 and 100 only.
  REQUIRE(mon.warning_count(0, "max_abs") == 2);
}

TEST_CASE("iter_health: full mode emits every iter", "[iter_health]") {
  auto layout = make_two_block_layout();
  t_type w(10);
  for (std::size_t i = 0; i < 10; ++i) w[i] = 0.0;

  IterHealthMonitor mon(HealthLogMode::Full);
  for (std::size_t it = 0; it < 30; ++it) {
    auto rec = compute_iter_health(it, w, layout);
    mon.record(rec, layout);
  }
  REQUIRE(mon.warning_count(0, "max_abs") == 30);
  REQUIRE(mon.warning_count(1, "max_abs") == 30);
}

TEST_CASE("iter_health: off mode is a no-op", "[iter_health]") {
  auto layout = make_two_block_layout();
  t_type w(10);
  for (std::size_t i = 0; i < 10; ++i) w[i] = 0.0;

  IterHealthMonitor mon(HealthLogMode::Off);
  for (std::size_t it = 0; it < 30; ++it) {
    auto rec = compute_iter_health(it, w, layout);
    mon.record(rec, layout);
  }
  REQUIRE(mon.warning_count(0, "max_abs") == 0);
}

/* Pins: a non-finite weight is found wherever it is: an Inf at weight 11
 * of 12, past the 10 columns of the layout's two blocks, sets
 * weights_have_nan_inf.
 * Before the fix: only the weights inside the blocks were scanned, so the
 * flag stayed false. */
TEST_CASE("iter_health: a non-finite weight outside the layout's blocks is "
          "found", "[iter_health]") {
  auto layout = make_two_block_layout();   // 10 columns
  t_type w(12);
  for (std::size_t i = 0; i < 12; ++i) w[i] = 1.0;
  w[11] = std::numeric_limits<double>::infinity();
  auto rec = compute_iter_health(0, w, layout);
  REQUIRE(rec.weights_have_nan_inf);
}

/* Pins: a layout that does not describe the weights (10 columns for 5
 * finite weights) is not taken for NaN/Inf weights: weights_have_nan_inf is
 * false, and record() in full mode prints "iter_health: eval 7 -- the
 * model has 5 weights but its column layout has 10 columns: the per-block
 * checks are skipped." and no block warning (neither block's max_abs
 * count moves) and no NaN/Inf line.
 * Before the fix (this case was "layout/weights size mismatch sets nan_inf
 * flag" and required the flag): the block past the weights set
 * weights_have_nan_inf as a sentinel, so record() printed "weights contain
 * NaN/Inf entries" for finite weights, and warned that the block it could
 * not read (max_abs left 0) had only zero weights. */
TEST_CASE("iter_health: a layout that does not describe the weights is "
          "reported as such, not as NaN/Inf weights", "[iter_health]") {
  auto layout = make_two_block_layout();   // claims 10 cols
  t_type w(5);                              // only 5 weights
  for (std::size_t i = 0; i < 5; ++i) w[i] = 1.0;
  auto rec = compute_iter_health(7, w, layout);
  CHECK_FALSE(rec.weights_have_nan_inf);

  IterHealthMonitor mon(HealthLogMode::Full);
  std::string printed;
  {
    LoggerCapture cap;
    mon.record(rec, layout);
    printed = cap.text();
  }
  INFO(printed);
  CHECK(printed.find("iter_health: eval 7 -- the model has 5 weights but "
                     "its column layout has 10 columns: the per-block "
                     "checks are skipped.") != std::string::npos);
  CHECK(printed.find("NaN/Inf") == std::string::npos);
  CHECK(mon.warning_count(0, "max_abs") == 0);
  CHECK(mon.warning_count(1, "max_abs") == 0);
}

/* Pins: a layout of as many columns as weights whose blocks do not
 * partition them (block 1 starts at column 5, after a column no block
 * owns) is reported for what it is: weights_have_nan_inf false, and
 * record() in full mode prints "iter_health: eval 3 -- the column layout
 * of the model's 10 weights is not contiguous from column 0: the per-block
 * checks are skipped." and no block warning.
 * Before the fix: the message gave both counts, "the model has 10 weights
 * but its column layout has 10 columns", which says no mismatch. */
TEST_CASE("iter_health: a layout whose blocks do not partition the weights "
          "is reported as such", "[iter_health]") {
  ColumnLayout layout;
  ColumnBlock a; a.descriptor_label = "D2"; a.col_offset = 0; a.col_count = 4;
  ColumnBlock b; b.descriptor_label = "DM"; b.col_offset = 5; b.col_count = 6;
  layout.blocks.push_back(a);
  layout.blocks.push_back(b);
  t_type w(10);
  for (std::size_t i = 0; i < 10; ++i) w[i] = 0.0;
  auto rec = compute_iter_health(3, w, layout);
  CHECK_FALSE(rec.weights_have_nan_inf);

  IterHealthMonitor mon(HealthLogMode::Full);
  std::string printed;
  {
    LoggerCapture cap;
    mon.record(rec, layout);
    printed = cap.text();
  }
  INFO(printed);
  CHECK(printed.find("iter_health: eval 3 -- the column layout of the "
                     "model's 10 weights is not contiguous from column 0: "
                     "the per-block checks are skipped.") != std::string::npos);
  CHECK(mon.warning_count(0, "max_abs") == 0);
  CHECK(mon.warning_count(1, "max_abs") == 0);
}

/* Pins: one monitor recorded from 8 threads at once, as HPO_Host's
 * concurrent evaluations (ANNEAL_PT/PA replicas, DLIB:GFS batches) record
 * into the process' monitor, 400 records each of a dead DM block in full
 * mode: every warning is counted, 3200 for block 1, none for block 0, and
 * the case completes.
 * Before the fix: record() changed its maps without a lock (the header
 * asked callers to funnel records through one thread, which the host never
 * did): a data race, which lost counts in each of 9 runs (2730 to 3021 of
 * 3200). */
TEST_CASE("iter_health: records from concurrent evaluations are all "
          "counted", "[iter_health]") {
  auto layout = make_two_block_layout();
  t_type w(10);
  for (std::size_t i = 0; i < 4; ++i) w[i] = 1.0;
  for (std::size_t i = 4; i < 10; ++i) w[i] = 0.0;

  IterHealthMonitor mon(HealthLogMode::Full);
  constexpr std::size_t kThreads = 8, kRecords = 400;
  std::vector<std::thread> threads;
  for (std::size_t t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      for (std::size_t k = 0; k < kRecords; ++k) {
        const auto rec = compute_iter_health(t * kRecords + k, w, layout);
        mon.record(rec, layout);
      }
    });
  }
  for (auto& th : threads) th.join();
  CHECK(mon.warning_count(1, "max_abs") == kThreads * kRecords);
  CHECK(mon.warning_count(0, "max_abs") == 0);
}
