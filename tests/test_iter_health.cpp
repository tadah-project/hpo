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

#include "catch2/catch.hpp"

#include <tadah/core/aed_type.h>
#include <tadah/core/core_types.h>
#include <tadah/hpo/iter_health.h>
#include <tadah/mlip/design_matrix/column_block.h>

#include <cmath>
#include <limits>

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

TEST_CASE("iter_health: layout/weights size mismatch sets nan_inf flag", "[iter_health]") {
  auto layout = make_two_block_layout();   // claims 10 cols
  t_type w(5);                              // only 5 weights
  for (std::size_t i = 0; i < 5; ++i) w[i] = 1.0;
  auto rec = compute_iter_health(0, w, layout);
  REQUIRE(rec.weights_have_nan_inf);   // sentinel for structural mismatch
}
