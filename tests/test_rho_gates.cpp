// test_rho_gates.cpp
//
// The DM_REAM density gates (rho_gates.h):
//   - RHOGATE / RHOBARGATE grammar (hpo_targets.cpp) and the startup
//     checks of validate_hpo_setup (bare DM_REAM, RHOGATE SIGNED-only, R_LO
//     and the first RHOBARGATE shell against the knot-aware reachable
//     RCUTMB, the CONFIG/WARM seed checks under SEED ERROR | WARN);
//   - the gate verdicts on the nine campaign fixtures (rho_gate_fixtures.h):
//     rho_min / rho_max bit for bit against expected.tsv, the verdicts and v
//     at tol 1e-5, 6e-6, 4e-6, 1e-8 and 0, and RHOBARGATE's 1307 / 373
//     rises on ALL_E_s5166_ON;
//   - saturation, abstention, arming, the cached verdict of an unarmed
//     gate, the combined grading, the counters, the report lines, and that
//     a gate never changes the candidate context;
//   - hidden host-level cases, run explicitly:
//       ./test_hpo "[rhogate_e2e]"   ./test_hpo "[rhobargate_e2e]"
//   - a hidden micro-benchmark: ./test_hpo "[rho_gates_bench]"

#include "catch2/catch.hpp"
#include "rho_gate_fixtures.h"
#include "rhoform_test_support.h"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/bump_support_gate.h>
#include <tadah/hpo/hpo.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/pc_fail_grading.h>
#include <tadah/hpo/rho_gates.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/startup_validation.h>
#include <tadah/models/eam/eam_lattice.h>

#include <dlib/matrix.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace dt = tadah::hpo::detail;
using tadah::core::Context;
using tadah::core::PeriodicTable;
using tadah::hpo::GateKind;
using tadah::hpo::HPOLog;
using tadah::hpo::HPOTargets;
using tadah::hpo::validate_hpo_setup;
using ParamVec = dlib::matrix<double, 0, 1>;
using Host     = tadah::hpo::HPO_Host<ParamVec, ::LammpsRunner>;
using rhoform_test::ForceRebuildGuard;
using rhoform_test::Journal;
using rhoform_test::kOptimizer;
using rhoform_test::make_ream_ctx;
using rhoform_test::ream_seed;
using rhoform_test::ScratchDir;
using rhoform_test::targets_error;
using rhoform_test::TempFile;
using rhoform_test::write_ta_db;

constexpr double kSat = std::numeric_limits<double>::max();

std::uint64_t bits(double v) {
  std::uint64_t u;
  std::memcpy(&u, &v, sizeof u);
  return u;
}

/* The campaign fixture's density block (RHOFORM SIGNED: its amplitudes are
 * signed), as MODELS/tests/test_ream_fixtures.cpp builds it. */
Context fixture_ctx(const rho_gate_fixtures::Fixture& f) {
  Context c;
  c.add("VERBOSE", 0);
  c.add("ATOM", "Ta");
  c.add("WATOM", f.watom);
  c.add("INITMB", "true");
  c.add("RCUTMB", f.rcutmb);
  c.add("RCTYPEMB", "Cut_Dummy");
  c.add("TYPEMB", "DM_REAM 3 20 20 20 3 3 0 0 Ta Ta");
  c.add("CGRIDMB", f.cgrid);
  c.add("SGRIDMB", f.sgrid);
  c.add("AMPGRIDMB", f.amp);
  c.add("CEMBFUNC", "0.0 0.5 1.0");
  c.add("SEMBFUNC", "1.0 1.0 1.0");
  c.add("RHOFORM", "SIGNED");
  return c;
}

const rho_gate_fixtures::Fixture& fixture(const std::string& name) {
  for (const auto& f : rho_gate_fixtures::kFixtures)
    if (name == f.name) return f;
  FAIL("no fixture " << name);
  return rho_gate_fixtures::kFixtures[0];
}

dt::RhoGateTable rho_table(double tol, std::optional<double> r_lo,
                           std::size_t npts = 20001) {
  dt::RhoGateTable t;
  t.spec.tol = tol;
  t.spec.r_lo = r_lo;
  t.spec.npts = npts;
  t.armed = true;
  return t;
}

dt::RhoBarGateTable rhobar_table(const std::string& lattice, double a0,
                                 double f_lo, double f_hi, std::size_t npts,
                                 double rcut) {
  dt::RhoBarGateTable t;
  t.spec.lattice = lattice;
  t.spec.a0 = a0;
  t.spec.f_lo = f_lo;
  t.spec.f_hi = f_hi;
  t.spec.npts = npts;
  t.shells = tadah::models::eam_lattice::shells(
      lattice, rcut / (f_lo * a0) * (1.0 + 1e-12));
  t.f = tadah::models::ReamDensity::linspace(f_lo, f_hi, npts);
  for (double x : t.f) t.a.push_back(a0 * x);
  t.armed = true;
  return t;
}

/* The error validate_density_gates throws for (ctx, target) ("" when none). */
std::string gates_error(Context ctx, const std::string& target) {
  TempFile tf(target);
  HPOTargets trg(ctx, tf.path.string());
  try {
    dt::validate_density_gates(ctx, trg);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

}  // namespace

/* ======================================================================== */

TEST_CASE("RHOGATE / RHOBARGATE: strict grammar", "[hpo][rhogate][grammar]") {
  PeriodicTable::initialize();
  const std::string base = "OPTIM AMPGRIDMB (3) -0.5 0.5\nPC_ERMSE 0 1\n" +
                           std::string(kOptimizer);
  auto parse = [&](const std::string& lines) {
    Context ctx = make_ream_ctx("SIGNED");
    TempFile tf(base + lines);
    return std::make_unique<HPOTargets>(ctx, tf.path.string());
  };

  SECTION("absent: no gate") {
    auto trg = parse("");
    CHECK_FALSE(trg->RHOGATE);
    CHECK_FALSE(trg->RHOBARGATE);
  }
  SECTION("RHOGATE: defaults and every option, keywords case-insensitive") {
    auto a = parse("RHOGATE 1e-5\n");
    REQUIRE(a->RHOGATE);
    CHECK(a->RHOGATE->tol == 1e-5);
    CHECK_FALSE(a->RHOGATE->r_lo);
    CHECK(a->RHOGATE->npts == 20001);
    CHECK_FALSE(a->RHOGATE->seed_warn);
    auto b = parse("RHOGATE 0 seed warn npts 2000 r_lo 1.7168\n");
    REQUIRE(b->RHOGATE);
    CHECK(b->RHOGATE->tol == 0.0);
    CHECK(*b->RHOGATE->r_lo == 1.7168);
    CHECK(b->RHOGATE->npts == 2000);
    CHECK(b->RHOGATE->seed_warn);
    CHECK(parse("RHOGATE 1e-5 NPTS 2 SEED Error\n")->RHOGATE->npts == 2);
    CHECK(parse("RHOGATE 1e-5 NPTS 10000000\n")->RHOGATE->npts == 10000000);
  }
  SECTION("RHOBARGATE: defaults and options") {
    auto a = parse("RHOBARGATE BCC 3.304 0.75 1.10\n");
    REQUIRE(a->RHOBARGATE);
    CHECK(a->RHOBARGATE->lattice == "bcc");
    CHECK(a->RHOBARGATE->a0 == 3.304);
    CHECK(a->RHOBARGATE->f_lo == 0.75);
    CHECK(a->RHOBARGATE->f_hi == 1.10);
    CHECK(a->RHOBARGATE->npts == 1001);
    CHECK_FALSE(a->RHOBARGATE->seed_warn);
    auto b = parse("RHOBARGATE hcp 2.9 0.9 1.1 Npts 3 SEED WARN\n");
    CHECK(b->RHOBARGATE->lattice == "hcp");
    CHECK(b->RHOBARGATE->npts == 3);
    CHECK(b->RHOBARGATE->seed_warn);
  }
  SECTION("refused: numbers, ranges, keywords, duplicates, trailing tokens") {
    struct Case { const char* line; const char* what; };
    for (const Case& c : {
             Case{"RHOGATE\n", "the tolerance is missing"},
             Case{"RHOGATE -1\n", "RHOGATE tol must be >= 0"},
             Case{"RHOGATE nan\n", "RHOGATE tol must be a finite number"},
             Case{"RHOGATE inf\n", "RHOGATE tol must be a finite number"},
             Case{"RHOGATE 1e-5x\n", "RHOGATE tol must be a finite number"},
             Case{"RHOGATE 1e-5 NPTS 1\n", "RHOGATE NPTS must be an integer from 2 to 10000000"},
             Case{"RHOGATE 1e-5 NPTS 2e3\n", "RHOGATE NPTS must be an integer"},
             Case{"RHOGATE 1e-5 NPTS 10000001\n", "RHOGATE NPTS must be an integer"},
             Case{"RHOGATE 1e-5 NPTS 2.5\n", "RHOGATE NPTS must be an integer"},
             Case{"RHOGATE 1e-5 R_LO -1\n", "RHOGATE R_LO must be >= 0"},
             Case{"RHOGATE 1e-5 R_LO nan\n", "RHOGATE R_LO must be a finite number"},
             Case{"RHOGATE 1e-5 SEED MAYBE\n", "RHOGATE SEED must be ERROR or WARN"},
             Case{"RHOGATE 1e-5 NPTS 100 NPTS 200\n", "NPTS given more than once"},
             Case{"RHOGATE 1e-5 TOL 2\n", "unexpected token 'TOL'"},
             Case{"RHOGATE 1e-5 NPTS\n", "NPTS needs a value"},
             Case{"RHOGATE 1e-5 1e-6\n", "unexpected token '1e-6'"},
             Case{"RHOGATE 1e-5\nRHOGATE 1e-6\n", "RHOGATE given more than once"},
             Case{"RHOBARGATE bcc 3.304 0.75\n", "the lattice, a0, f_lo and f_hi are required"},
             Case{"RHOBARGATE sc 3.304 0.75 1.1\n", "lattice must be bcc, fcc or hcp"},
             Case{"RHOBARGATE bcc 0 0.75 1.1\n", "RHOBARGATE a0 must be > 0"},
             Case{"RHOBARGATE bcc inf 0.75 1.1\n", "RHOBARGATE a0 must be a finite number"},
             Case{"RHOBARGATE bcc 3.3 1.1 0.75\n", "needs 0 < f_lo < f_hi"},
             Case{"RHOBARGATE bcc 3.3 0 1.1\n", "needs 0 < f_lo < f_hi"},
             Case{"RHOBARGATE bcc 3.3 0.9 0.9\n", "needs 0 < f_lo < f_hi"},
             Case{"RHOBARGATE bcc 3.3 0.9 1.1 NPTS 2\n", "RHOBARGATE NPTS must be an integer from 3 to 10000000"},
             Case{"RHOBARGATE bcc 3.3 0.9 1.1 R_LO 1\n", "unexpected token 'R_LO'"},
             Case{"RHOBARGATE bcc 3.3 0.9 1.1 SEED WARN extra\n", "unexpected token 'extra'"},
             Case{"RHOBARGATE bcc 3.3 0.9 1.1\nRHOBARGATE fcc 4 0.9 1.1\n",
                  "RHOBARGATE given more than once"}}) {
      CAPTURE(c.line);
      const std::string e = targets_error(make_ream_ctx("SIGNED"), base + c.line);
      CHECK_THAT(e, Catch::Contains(c.what));
    }
  }
}

TEST_CASE("RHOGATE on the campaign fixtures: bitwise extrema, verdicts and v",
          "[hpo][rhogate][fixtures]") {
  PeriodicTable::initialize();
  REQUIRE(sizeof(rho_gate_fixtures::kFixtures) /
              sizeof(rho_gate_fixtures::kFixtures[0]) == 9);
  for (std::size_t k = 0; k < 9; ++k) {
    const auto& f = rho_gate_fixtures::kFixtures[k];
    const auto& ref = rho_gate_fixtures::kRhoGate[k];
    REQUIRE(std::string(f.name) == ref.name);
    INFO(f.name);
    const Context ctx = fixture_ctx(f);
    const std::vector<char> before = ctx.serialize();
    for (int t = 0; t < 5; ++t) {
      const double tol = rho_gate_fixtures::kTols[t];
      INFO("tol " << tol);
      const dt::RhoGateResult r = dt::eval_rho_gate(rho_table(tol, 1.7168), ctx);
      REQUIRE_FALSE(r.abstain);
      CHECK_FALSE(r.saturated);
      CHECK(r.r_lo == 1.7168);
      CHECK(r.r_hi == std::stod(f.rcutmb));
      CHECK(bits(r.rho_max) == bits(f.rho_max));
      CHECK(bits(r.r_max) == bits(f.r_at_max));
      CHECK(bits(r.r_min) == bits(f.r_at_min));
      if (std::string(f.name) == "D_M3_s301_w4_best") {
        /* the tool takes f_c(RCUTMB) = 1, tadah's Cut_Dummy 0 */
        CHECK(f.rho_min > 0.0);
        CHECK(bits(r.rho_min) == bits(0.0));
        CHECK(r.r_min == r.r_hi);
      } else {
        CHECK(bits(r.rho_min) == bits(f.rho_min));
      }
      CHECK(r.fail == ref.fail[t]);
      INFO("v " << std::hexfloat << r.v << " expected " << ref.v[t]);
      CHECK(bits(r.v) == bits(ref.v[t]));
      if (!r.fail) CHECK(r.v == 0.0);
    }
    /* the gate reads stored values only: the context is unchanged */
    CHECK(ctx.serialize() == before);
  }
}

TEST_CASE("RHOBARGATE on the campaign fixtures: ALL_E rises at 1307 of 3501 "
          "(373 of 1001) steps, the others never",
          "[hpo][rhogate][rhobargate][fixtures]") {
  PeriodicTable::initialize();
  for (const auto& f : rho_gate_fixtures::kFixtures) {
    INFO(f.name);
    const Context ctx = fixture_ctx(f);
    const bool all_e = std::string(f.name) == "ALL_E_s5166_ON";
    for (const std::size_t npts : {std::size_t(3501), std::size_t(1001)}) {
      INFO("NPTS " << npts);
      const dt::RhoBarGateResult r = dt::eval_rhobar_gate(
          rhobar_table("bcc", 3.304, 0.75, 1.10, npts, std::stod(f.rcutmb)), ctx);
      REQUIRE_FALSE(r.abstain);
      CHECK_FALSE(r.saturated);
      CHECK(r.fail == all_e);
      if (!all_e) {
        CHECK(r.rises == 0);
        CHECK(r.v == 0.0);
        continue;
      }
      CHECK(r.rises == (npts == 3501 ? 1307u : 373u));
      CHECK(r.first_rise == 0.75);
      /* gate_refs.py: the same sums in the same order */
      CHECK(bits(r.v) == bits(npts == 3501 ? 0x1.44ec0985f0a76p+1
                                           : 0x1.44f221173eb9fp+1));
    }
  }
}

TEST_CASE("Density gates on the campaign fixtures are gauge invariant: "
          "AMPGRIDMB x lambda or WATOM x lambda keeps every verdict",
          "[hpo][rhogate][rhobargate][fixtures]") {
  PeriodicTable::initialize();
  auto scaled = [](const char* list, double lambda) {
    std::istringstream is(list);
    std::ostringstream os;
    os.precision(17);
    double x;
    bool first = true;
    while (is >> x) {
      os << (first ? "" : " ") << x * lambda;
      first = false;
    }
    return os.str();
  };
  for (const auto& f : rho_gate_fixtures::kFixtures) {
    INFO(f.name);
    const Context base = fixture_ctx(f);
    const double rc = std::stod(f.rcutmb);
    const auto r0 = dt::eval_rho_gate(rho_table(1e-5, 1.7168), base);
    const auto b0 = dt::eval_rhobar_gate(
        rhobar_table("bcc", 3.304, 0.75, 1.10, 1001, rc), base);
    for (const double lambda : {0.95, 1.05}) {
      INFO("lambda " << lambda);
      Context amp = fixture_ctx(f);
      amp.remove("AMPGRIDMB");
      amp.add("AMPGRIDMB", scaled(f.amp, lambda));
      Context w = fixture_ctx(f);
      w.remove("WATOM");
      w.add("WATOM", scaled(f.watom, lambda));
      for (const Context* c : {&amp, &w}) {
        const auto r = dt::eval_rho_gate(rho_table(1e-5, 1.7168), *c);
        REQUIRE_FALSE(r.abstain);
        CHECK(r.fail == r0.fail);
        if (r0.fail) CHECK(std::fabs(r.v - r0.v) <= 1e-12 * r0.v);
        const auto b = dt::eval_rhobar_gate(
            rhobar_table("bcc", 3.304, 0.75, 1.10, 1001, rc), *c);
        REQUIRE_FALSE(b.abstain);
        CHECK(b.fail == b0.fail);
        CHECK(b.rises == b0.rises);
        if (b0.fail) CHECK(std::fabs(b.v - b0.v) <= 1e-12 * b0.v);
      }
    }
  }
}

TEST_CASE("Density gates: saturation, abstention and the knife edge",
          "[hpo][rhogate][rhobargate]") {
  PeriodicTable::initialize();
  auto ream = [](const char* form, const char* amps) {
    return make_ream_ctx(form, amps);
  };

  SECTION("RHOGATE saturates at DBL_MAX") {
    /* rho_max <= 0 on the window */
    auto r = dt::eval_rho_gate(rho_table(1e-5, std::nullopt),
                               ream("SIGNED", "-0.5 -0.3 -0.1"));
    CHECK(r.fail);
    CHECK(r.saturated);
    CHECK(r.v == kSat);
    CHECK_THAT(r.reason, Catch::Contains("rho_max <= 0"));
    /* a density pair weight <= 0 */
    Context w = ream("SIGNED", "0.5 0.3 0.1");
    w.remove("WATOM");
    w.add("WATOM", -1.0);
    r = dt::eval_rho_gate(rho_table(1e-5, std::nullopt), w);
    CHECK(r.saturated);
    CHECK_THAT(r.reason, Catch::Contains("density pair weight W = -1"));
    /* pair-triple WATOM with a zero pair */
    Context wt = ream("SIGNED", "0.5 0.3 0.1");
    wt.remove("WATOM");
    wt.add("WATOM", "Ta Ta 0");
    CHECK(dt::eval_rho_gate(rho_table(1e-5, std::nullopt), wt).saturated);
    /* ... but a check_weights = false evaluation ignores it */
    CHECK_FALSE(dt::eval_rho_gate(rho_table(1e-5, std::nullopt), wt, false).fail);
    /* an empty window */
    r = dt::eval_rho_gate(rho_table(1e-5, 6.0), ream("SIGNED", "0.5 0.3 0.1"));
    CHECK(r.saturated);
    CHECK_THAT(r.reason, Catch::Contains("window"));
    /* a non-finite value */
    r = dt::eval_rho_gate(rho_table(1e-5, std::nullopt), ream("SIGNED", "0.5 inf 0.1"));
    CHECK(r.saturated);
    CHECK_THAT(r.reason, Catch::Contains("non-finite"));
  }
  SECTION("a density DM_REAM would refuse: the gate abstains") {
    Context z = ream("SIGNED", "0.5 0.3 0.1");
    z.remove("SGRIDMB");
    z.add("SGRIDMB", "3.0 0.0 3.0");   // DM_REAM: zero entry in SGRIDMB
    const auto r = dt::eval_rho_gate(rho_table(1e-5, std::nullopt), z);
    CHECK(r.abstain);
    CHECK_FALSE(r.fail);
    CHECK_THAT(r.reason, Catch::Contains("zero entry in SGRIDMB"));
    Context n = ream("SIGNED", "0.5 0.3 0.1");
    n.remove("RCTYPEMB");
    CHECK(dt::eval_rho_gate(rho_table(1e-5, std::nullopt), n).abstain);
    CHECK(dt::eval_rhobar_gate(rhobar_table("bcc", 3.304, 0.98, 1.1, 101, 5.5), z)
              .abstain);
  }
  SECTION("RHOBARGATE saturates on an all-zero rhobar and a bad weight") {
    auto r = dt::eval_rhobar_gate(rhobar_table("bcc", 3.304, 0.98, 1.1, 101, 5.5),
                                  ream("NONNEG", "0 0 0"));
    CHECK(r.saturated);
    CHECK(r.v == kSat);
    CHECK_THAT(r.reason, Catch::Contains("rhobar(a) = 0"));
    Context w = ream("SIGNED", "0.5 0.3 0.1");
    w.remove("WATOM");
    w.add("WATOM", 0.0);
    CHECK(dt::eval_rhobar_gate(rhobar_table("bcc", 3.304, 0.98, 1.1, 101, 5.5), w)
              .saturated);
  }
  SECTION("a flat step is a rise: the knife edge fails at v = 0") {
    /* MONOTONE, one bump far outside: S = 1 on [0, RCUTMB), so rho is a
     * constant and rhobar(a) a non-increasing step function of the
     * neighbour count within RCUTMB */
    Context c = ream("MONOTONE", "0.5 0.0 0.0");
    c.remove("CGRIDMB");
    c.add("CGRIDMB", "10.0 10.0 10.0");
    const auto r =
        dt::eval_rhobar_gate(rhobar_table("bcc", 3.304, 0.98, 1.1, 101, 5.5), c);
    CHECK(r.fail);
    CHECK_FALSE(r.saturated);
    CHECK(r.rises > 0);
    CHECK(r.v == 0.0);
  }
  SECTION("MONOTONE is non-increasing, NONNEG is not (the e2e density)") {
    auto t = rhobar_table("bcc", 3.304, 0.98, 1.10, 1001, 5.5);
    CHECK_FALSE(dt::eval_rhobar_gate(t, ream("MONOTONE", "0.5 1.0 0.1")).fail);
    const auto nn = dt::eval_rhobar_gate(t, ream("NONNEG", "0.5 1.0 0.1"));
    CHECK(nn.fail);
    CHECK(nn.rises == 896);
    CHECK(bits(nn.v) == bits(0x1.2e6fea5fb8946p-3));   // gate_refs.py
    const auto sg = dt::eval_rhobar_gate(t, ream("SIGNED", "0.5 1.0 0.1"));
    CHECK(bits(sg.v) == bits(nn.v));   // NONNEG == SIGNED for a_n >= 0
  }
}

TEST_CASE("Density gates read WATOM and RCUTMBMAX with the number grammar "
          "of the model", "[hpo][rhogate][rhobargate]") {
  // The gates read WATOM (D_Base::density_pair_weights) and RCUTMBMAX with
  // strtod, the model with istringstream: "0x1p-1" was a weight of 0.5 in
  // a gate and 0 in the model, and a hexadecimal RCUTMBMAX a cutoff the
  // model never used. Both now read the decimal grammar of the files.
  PeriodicTable::initialize();
  const auto t = rho_table(1e-5, std::nullopt);
  const auto tb = rhobar_table("bcc", 3.304, 0.98, 1.1, 101, 5.5);
  Context w = make_ream_ctx("SIGNED", "0.5 0.3 0.1");
  w.remove("WATOM");
  w.add("WATOM", std::string("0x1p-1"));
  const auto r = dt::eval_rho_gate(t, w);
  CHECK(r.fail);
  CHECK(r.saturated);
  CHECK_THAT(r.reason, Catch::Contains("WATOM entry 1 ('0x1p-1') is not a "
                                       "decimal number"));
  CHECK(dt::eval_rhobar_gate(tb, w).saturated);
  // a decimal weight passes as before
  Context ok = make_ream_ctx("SIGNED", "0.5 0.3 0.1");
  ok.remove("WATOM");
  ok.add("WATOM", std::string("0.5"));
  CHECK_FALSE(dt::eval_rho_gate(t, ok).fail);

  Context rc = make_ream_ctx("SIGNED", "0.5 0.3 0.1");
  // the raw token (add/addInternal would re-derive RCUTMBMAX from RCUTMB)
  rc.storage_update("RCUTMBMAX", {"0x1.6p2"});  // 5.5 to strtod
  REQUIRE(rc.stored_value("RCUTMBMAX").value_or("") == "0x1.6p2");
  const auto a = dt::eval_rho_gate(t, rc);
  CHECK(a.abstain);
  CHECK_FALSE(a.fail);
  CHECK_THAT(a.reason, Catch::Contains("RCUTMBMAX ('0x1.6p2') is not a "
                                       "decimal number"));
}

TEST_CASE("Density gates: arming, the reachable RCUTMB and the cached verdict",
          "[hpo][rhogate]") {
  PeriodicTable::initialize();
  auto build = [](Context ctx, const std::string& lines) {
    TempFile tf(lines + "PC_ERMSE 0 1\n" + kOptimizer);
    HPOTargets trg(ctx, tf.path.string());
    return dt::build_density_gates(ctx, trg);
  };
  const Context ctx = make_ream_ctx("SIGNED");

  SECTION("armed on AMPGRIDMB / CGRIDMB / SGRIDMB / RCUTMB, case-insensitive") {
    for (const char* optim : {"OPTIM AMPGRIDMB (3) -0.5 0.5\n",
                              "OPTIM cgridmb (1) 2.5 3.0\n",
                              "OPTIM SGRIDMB (2) 2.5 3.5\n",
                              "OPTIM RcutMB (1) 5.0 5.5\n"}) {
      CAPTURE(optim);
      const auto g = build(ctx, std::string(optim) +
                                    "RHOGATE 1e-5\nRHOBARGATE bcc 3.304 0.98 1.1\n");
      REQUIRE(g.rho);
      REQUIRE(g.rhobar);
      CHECK(g.rho->armed);
      CHECK(g.rhobar->armed);
      CHECK_FALSE(g.rho->fixed);
      CHECK(g.rho->armed_on.size() == 1);
    }
  }
  SECTION("unarmed: the cached -c verdict is reused, no scan per candidate") {
    const auto g = build(ctx, "OPTIM CEMBFUNC (1) -0.5 0.1\nRHOGATE 1e-5\n"
                              "RHOBARGATE bcc 3.304 0.98 1.1\n");
    REQUIRE(g.rho);
    CHECK_FALSE(g.rho->armed);
    CHECK_FALSE(g.rho->watom_armed);
    REQUIRE(g.rho->fixed);
    CHECK_FALSE(g.rho->fixed->fail);
    REQUIRE(g.rhobar->fixed);
    const std::size_t scans = dt::density_gate_scans_for_test();
    for (int i = 0; i < 3; ++i) {
      const auto v = dt::check_density_gates(g, ctx);
      REQUIRE(v.rho);
      REQUIRE(v.rhobar);
      CHECK_FALSE(v.rho->fail);
      CHECK_FALSE(v.rhobar->fail);
    }
    CHECK(dt::density_gate_scans_for_test() == scans);
  }
  SECTION("OPTIM WATOM alone: the pair-weight check per candidate") {
    const auto g = build(ctx, "OPTIM WATOM (1) -1 2\nRHOGATE 1e-5\n");
    REQUIRE(g.rho);
    CHECK_FALSE(g.rho->armed);
    CHECK(g.rho->watom_armed);
    Context neg = ctx;
    neg.remove("WATOM");
    neg.add("WATOM", -0.5);
    const std::size_t scans = dt::density_gate_scans_for_test();
    const auto v = dt::check_density_gates(g, neg);
    CHECK(v.rho->fail);
    CHECK(v.rho->saturated);
    CHECK_FALSE(dt::check_density_gates(g, ctx).rho->fail);
    CHECK(dt::density_gate_scans_for_test() == scans);
  }
  SECTION("bump channel: the RCUTMB range over its OPTIM box") {
    TempFile tf("OPTIM rcutmb (1) 5.0 6.0\nPC_ERMSE 0 1\n" + std::string(kOptimizer));
    Context c = ctx;
    HPOTargets trg(c, tf.path.string());
    const auto r = dt::knot_aware_reachable_rcut(ctx, trg);
    CHECK_FALSE(r.knot);
    CHECK(r.min_reachable == 5.0);
    CHECK(r.max_reachable == 6.0);
  }
  SECTION("knot channel: the largest knot over the CGRIDMB boxes") {
    Context k = make_ream_ctx("SIGNED", "0.01 0.02 0.3");
    k.remove("TYPEMB");
    k.add("TYPEMB", "DM_REAM 2 3 0 3 5 0 0 0 Ta Ta");   // no width grids
    k.remove("SGRIDMB");
    k.remove("SEMBFUNC");
    k.remove("CGRIDMB");
    k.add("CGRIDMB", "3.0 4.0 5.0");
    k.remove("RCUTMB");
    k.add("RCUTMB", 5.0);
    for (const auto& c : {std::make_tuple("OPTIM CGRIDMB (3) 4.5 5.5\n", 4.5, 5.5),
                          std::make_tuple("OPTIM CGRIDMB (2) 3.5 5.2\n", 5.0, 5.2),
                          std::make_tuple("OPTIM RCUTMB (1) 4.0 6.0\n", 5.0, 5.0)}) {
      CAPTURE(std::get<0>(c));
      TempFile tf(std::string(std::get<0>(c)) + "PC_ERMSE 0 1\n" + kOptimizer);
      Context kk = k;
      HPOTargets trg(kk, tf.path.string());
      const auto r = dt::knot_aware_reachable_rcut(k, trg);
      CHECK(r.knot);
      CHECK(r.min_reachable == std::get<1>(c));
      CHECK(r.max_reachable == std::get<2>(c));
    }
    /* R_LO against the smallest reachable knot */
    const std::string e = gates_error(
        k, "OPTIM CGRIDMB (3) 4.5 5.5\nRHOGATE 1e-5 R_LO 4.6\nPC_ERMSE 0 1\n" +
               std::string(kOptimizer));
    CHECK_THAT(e, Catch::Contains("RHOGATE R_LO 4.6 A must lie below") &&
                      Catch::Contains("RCUTMB = 4.5000 A (the largest CGRIDMB knot"));
  }
}

TEST_CASE("Density gates: startup refusals and seed checks",
          "[hpo][rhogate][rhobargate][startup]") {
  PeriodicTable::initialize();
  const std::string tail = "PC_ERMSE 0 1\n" + std::string(kOptimizer);

  SECTION("RHOGATE is for RHOFORM SIGNED only") {
    for (const char* form : {"NONNEG", "MONOTONE"}) {
      CAPTURE(form);
      const std::string e = gates_error(make_ream_ctx(form),
                                        "OPTIM AMPGRIDMB (3) 0 0.5\nRHOGATE 1e-5\n" + tail);
      CHECK_THAT(e, Catch::Contains("RHOGATE: RHOFORM " + std::string(form)) &&
                        Catch::Contains("the form guarantees rho >= 0"));
      /* RHOBARGATE is fine under every form */
      CHECK(gates_error(make_ream_ctx(form),
                        "OPTIM AMPGRIDMB (3) 0 0.5\nRHOBARGATE bcc 3.304 0.98 1.1\n" +
                            tail) == "");
    }
  }
  SECTION("a bare DM_REAM is required: no many-body term, DM_mJoin") {
    Context ctx = make_ream_ctx("SIGNED");
    TempFile tf("OPTIM AMPGRIDMB (3) -0.5 0.5\nRHOGATE 1e-5\n" + tail);
    HPOTargets trg(ctx, tf.path.string());
    Context no_mb = ctx;
    no_mb.remove("INITMB");
    no_mb.add("INITMB", "false");
    REQUIRE_THROWS_WITH(dt::validate_density_gates(no_mb, trg),
                        Catch::Contains("no DM_REAM many-body term"));
    Context mjoin = ctx;
    mjoin.remove("TYPEMB");
    mjoin.add("TYPEMB", "DM_mJoin DM_REAM 3 3 3 3 5 5 0 0 Ta Ta");
    REQUIRE_THROWS_WITH(dt::validate_density_gates(mjoin, trg),
                        Catch::Contains("DM_mJoin is not supported"));
  }
  SECTION("R_LO and the first RHOBARGATE shell inside the smallest RCUTMB") {
    CHECK_THAT(gates_error(make_ream_ctx("SIGNED"),
                           "OPTIM RCUTMB (1) 5.0 6.0\nRHOGATE 1e-5 R_LO 5.0\n" + tail),
               Catch::Contains("RHOGATE R_LO 5 A must lie below the smallest "
                               "cutoff a candidate can have, RCUTMB = 5.0000 A"));
    CHECK(gates_error(make_ream_ctx("SIGNED"),
                      "OPTIM RCUTMB (1) 5.0 6.0\nRHOGATE 1e-5 R_LO 4.99\n" + tail) == "");
    CHECK_THAT(gates_error(make_ream_ctx("SIGNED"),
                           "OPTIM AMPGRIDMB (2) 0 2\nRHOBARGATE bcc 3.304 0.98 3.0\n" +
                               tail),
               Catch::Contains("RHOBARGATE: the first bcc shell at a = f_hi a0") &&
                   Catch::Contains("beyond the smallest cutoff"));
  }
  SECTION("the -c pot as the CONFIG seed: SEED ERROR throws, SEED WARN warns") {
    const Context bad = make_ream_ctx("SIGNED", "0.5 0.3 -0.2");
    const std::string e =
        gates_error(bad, "OPTIM AMPGRIDMB (3) -0.5 0.5\nRHOGATE 1e-5\n" + tail);
    CHECK_THAT(e, Catch::Contains("RHOGATE: seed -c pot (INIT CONFIG) fails the "
                                  "rho(r) sign gate: rho_min -1.9966e-01 at r "
                                  "4.4025 A, rho_max 5.0036e-01, rel -3.9903e-01 < "
                                  "-tol = -1.0000e-05 on [0.8250, 5.5000] A (20001 "
                                  "points), v = 4.4515e-01.") &&
                      Catch::Contains("add SEED WARN"));
    Journal j;
    CHECK(gates_error(bad, "OPTIM AMPGRIDMB (3) -0.5 0.5\nRHOGATE 1e-5 SEED WARN\n" +
                               tail) == "");
    const std::string txt = j.finish();
    CHECK_THAT(txt, Catch::Contains("WARNING") &&
                        Catch::Contains("SEED WARN: the run proceeds") &&
                        Catch::Contains("RHOGATE: rho(r) sign gate, tol 1.0000e-05, "
                                        "window [0.15 RCUTMB, RCUTMB] of each "
                                        "candidate, 20001-point grid, SEED WARN; "
                                        "armed per candidate on AMPGRIDMB[3]") &&
                        Catch::Contains("seeds: -c pot (INIT CONFIG) FAIL"));
  }
  SECTION("unarmed: a failing -c density is an error under SEED WARN too") {
    for (const char* mode : {"", " SEED WARN"}) {
      CAPTURE(mode);
      const std::string e = gates_error(
          make_ream_ctx("SIGNED", "0.5 0.3 -0.2"),
          "OPTIM CEMBFUNC (1) -0.5 0.1\nRHOGATE 1e-5" + std::string(mode) + "\n" + tail);
      CHECK_THAT(e, Catch::Contains("RHOGATE: the -c density fails the rho(r) sign "
                                    "gate") &&
                        Catch::Contains("every evaluation would be rejected"));
    }
    CHECK_THAT(gates_error(make_ream_ctx("SIGNED"),
                           "OPTIM CEMBFUNC (1) -0.5 0.1\n"
                           "RHOBARGATE bcc 3.304 0.75 1.1 SEED WARN\n" + tail),
               Catch::Contains("RHOBARGATE: the -c density fails the lattice "
                               "rhobar(a) gate") &&
                   Catch::Contains("first at a/a0 = 0.7500, v = 1.9432e-01"));
  }
  SECTION("a density pair weight no candidate can make positive is an error "
          "under SEED WARN too, armed or not") {
    Context wneg = make_ream_ctx("SIGNED");
    wneg.remove("WATOM");
    wneg.add("WATOM", -1);
    const std::string lhs = "OPTIMIZER\nINIT\nSTRATEGY LHS 8\nSEED 3\nENDINIT\n"
                            "LIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n";
    for (const char* gate : {"RHOGATE 1e-5 SEED WARN\n",
                             "RHOBARGATE bcc 3.304 0.98 1.1 SEED WARN\n"}) {
      CAPTURE(gate);
      /* armed with WATOM fixed; the -c pot outside the seed set; unarmed
       * with an OPTIM WATOM box below 0 */
      for (const std::string& setup :
           {"OPTIM AMPGRIDMB (3) -0.5 0.5\n" + tail,
            "OPTIM AMPGRIDMB (3) -0.5 0.5\nPC_ERMSE 0 1\n" + lhs,
            "OPTIM WATOM (1) -2 -0.5\n" + tail}) {
        CAPTURE(setup);
        const std::string e = gates_error(wneg, gate + setup);
        CHECK_THAT(e, Catch::Contains("(WATOM) is not finite and > 0 on every "
                                      "candidate") &&
                          Catch::Contains("every evaluation would be rejected, "
                                          "under SEED ERROR or WARN"));
        /* it stands in for the per-seed and unarmed messages */
        CHECK(e.find('\n') == std::string::npos);
      }
    }
    /* an OPTIM WATOM box reaching above 0: only the failing seed is named */
    const std::string reach = "OPTIM WATOM (1) -2 1\n" + tail;
    CHECK_THAT(gates_error(wneg, "RHOGATE 1e-5\n" + reach),
               Catch::Contains("RHOGATE: seed -c pot (INIT CONFIG) fails the rho(r) "
                               "sign gate: a density pair weight W = -1 (WATOM)"));
    Journal j;
    CHECK(gates_error(wneg, "RHOGATE 1e-5 SEED WARN\n" + reach) == "");
    CHECK_THAT(j.finish(),
               Catch::Contains("SEED WARN: the run proceeds; if the optimiser "
                               "starts from this seed, that evaluation is "
                               "rejected before training (band D)."));
  }
  SECTION("WARM seeds: only the failing seed is named") {
    TempFile good(ream_seed("0.5 0.3 0.05"));
    TempFile bad(ream_seed("0.5 0.3 -0.1"));
    const std::string e = gates_error(
        make_ream_ctx("SIGNED"),
        "OPTIMIZER\nINIT\nSTRATEGY WARM " + good.path.string() + "\nSTRATEGY WARM " +
            bad.path.string() + "\nENDINIT\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\n"
            "ENDOPTIMIZER\nOPTIM AMPGRIDMB (3) -0.5 0.5\nRHOGATE 1e-5\nPC_ERMSE 0 1\n");
    CHECK_THAT(e, Catch::Contains("seed INIT WARM " + bad.path.string() +
                                  " (its OPTIM dimensions over the -c "
                                  "potential) fails"));
    CHECK(e.find(good.path.string()) == std::string::npos);
  }
  SECTION("sampled points are ordinary candidates; a failing -c pot outside "
          "the seed set is a WARNING") {
    Journal j;
    TempFile good(ream_seed("0.5 0.3 0.05"));
    /* LHS only: some samples fail (a3 < 0), and the -c pot (a3 -0.2) is not
     * an INIT start */
    CHECK(gates_error(make_ream_ctx("SIGNED", "0.5 0.3 -0.2"),
                      "OPTIMIZER\nINIT\nSTRATEGY LHS 8\nSEED 3\nENDINIT\nLIB DLIB\n"
                      "ALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n"
                      "OPTIM AMPGRIDMB (3) -0.5 0.5\nRHOGATE 1e-5\nPC_ERMSE 0 1\n") == "");
    CHECK(gates_error(make_ream_ctx("SIGNED", "0.5 0.3 -0.2"),
                      "OPTIMIZER\nINIT\nSTRATEGY WARM " + good.path.string() +
                          "\nENDINIT\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\nENDOPTIMIZER\n"
                          "OPTIM AMPGRIDMB (3) -0.5 0.5\nRHOGATE 1e-5\nPC_ERMSE 0 1\n") ==
          "");
    const std::string txt = j.finish();
    CHECK_THAT(txt, Catch::Contains("RHOGATE: -c pot fails the rho(r) sign gate") &&
                        Catch::Contains("It is not an INIT start: the run proceeds") &&
                        Catch::Contains("INIT WARM " + good.path.string() +
                                        " (its OPTIM dimensions over the -c "
                                        "potential) PASS"));
  }
  SECTION("validate_hpo_setup runs the gate checks") {
    Context ctx = make_ream_ctx("SIGNED", "0.5 0.3 -0.2");
    TempFile tf("OPTIM AMPGRIDMB (3) -0.5 0.5\nRHOGATE 1e-5\nPC_EMAE 0 1\n" +
                std::string(kOptimizer));
    HPOTargets trg(ctx, tf.path.string());
    REQUIRE_THROWS_WITH(validate_hpo_setup(ctx, trg, false, true),
                        Catch::Contains("RHOGATE: seed -c pot (INIT CONFIG)"));
  }
}

TEST_CASE("Density gates: combined grading and the reject text",
          "[hpo][rhogate]") {
  PeriodicTable::initialize();
  const Context ctx = make_ream_ctx("SIGNED");
  TempFile tf("OPTIM AMPGRIDMB (2,3) -2 2\nRHOGATE 1e-5\n"
              "RHOBARGATE bcc 3.304 0.98 1.1\nPC_ERMSE 0 1\n" + std::string(kOptimizer));
  Context c = ctx;
  HPOTargets trg(c, tf.path.string());
  const dt::DensityGates g = dt::build_density_gates(ctx, trg);
  auto verdict = [&](double a2, double a3) {
    return dt::check_density_gates(g, dt::seed_context(ctx, trg, {a2, a3}, g));
  };
  const std::string bump = "bump-support gate reject (c + w/eta past the "
                           "cutoff, violation=0.800000, graded band D)";

  SECTION("all pass: no reject") {
    CHECK_FALSE(dt::combine_gate_verdicts(0.0, "", g, verdict(0.3, 0.1)));
  }
  SECTION("BUMPGATE alone keeps its v and its text") {
    const auto r = dt::combine_gate_verdicts(0.8, bump, g, verdict(0.3, 0.1));
    REQUIRE(r);
    CHECK(r->kind == GateKind::Bump);
    CHECK(r->v == 0.8);
    CHECK(r->what == bump);
  }
  SECTION("RHOGATE alone") {
    auto d = verdict(0.3, -0.2);
    d.rhobar.reset();   // (this density's rhobar(a) rises too)
    const auto r = dt::combine_gate_verdicts(0.0, "", g, d);
    REQUIRE(r);
    CHECK(r->kind == GateKind::RhoSign);
    CHECK(r->v == d.rho->v);
    CHECK(r->what == "rho-sign gate reject (rho_min/rho_max = -3.9903e-01 at r = "
                     "4.4025 A, tol 1.0000e-05, v = 4.4515e-01)");
  }
  SECTION("RHOBARGATE alone") {
    const auto r = dt::combine_gate_verdicts(0.0, "", g, verdict(1.0, 0.1));
    REQUIRE(r);
    CHECK(r->kind == GateKind::RhoBar);
    CHECK(bits(r->v) == bits(0x1.2e6fea5fb8946p-3));
    CHECK(r->what == "lattice-rhobar gate reject (896 rises over a/a0 [0.98, 1.1] "
                     "on bcc, first at a/a0 = 0.9800, v = 1.4767e-01)");
  }
  SECTION("several: the first is counted, the sum grades") {
    const auto d = verdict(1.0, -0.2);
    REQUIRE(d.rho->fail);
    REQUIRE(d.rhobar->fail);
    auto r = dt::combine_gate_verdicts(0.8, bump, g, d);
    REQUIRE(r);
    CHECK(r->kind == GateKind::Bump);
    CHECK(r->v == 0.8 + d.rho->v + d.rhobar->v);
    CHECK_THAT(r->what, Catch::StartsWith(bump + "; also failing: rho-sign gate (v = ") &&
                            Catch::Contains("), lattice-rhobar gate (v = ") &&
                            Catch::Contains("); graded by the summed v = "));
    r = dt::combine_gate_verdicts(0.0, "", g, d);
    CHECK(r->kind == GateKind::RhoSign);
    CHECK(r->v == d.rho->v + d.rhobar->v);
  }
  SECTION("saturation propagates and stays finite") {
    dt::DensityGateVerdict d = verdict(0.3, 0.1);
    d.rho->fail = d.rho->saturated = true;
    d.rho->v = kSat;
    d.rhobar->fail = d.rhobar->saturated = true;
    d.rhobar->v = kSat;
    const auto r = dt::combine_gate_verdicts(0.8, bump, g, d);
    REQUIRE(r);
    CHECK(r->v == kSat);
    CHECK_THAT(r->what, Catch::Contains("graded by the summed v = saturated"));
  }
  SECTION("an abstaining gate passes") {
    dt::DensityGateVerdict d = verdict(0.3, 0.1);
    d.rho->abstain = true;
    CHECK_FALSE(dt::combine_gate_verdicts(0.0, "", g, d));
  }
}

TEST_CASE("HPOLog: per-gate counters, SUMMARY and DIAGNOSTICS lines, reset",
          "[hpo][rhogate][journal]") {
  Journal j;
  auto& log = HPOLog::instance();
  CHECK(log.next_gate_note(GateKind::RhoSign) == 1);
  CHECK(log.next_gate_note(GateKind::RhoSign) == 2);
  CHECK(log.next_gate_note(GateKind::Bump) == 1);
  /* 100 evaluations (the first milestone SUMMARY): 1 bump, 2 rho-sign, 3
   * rhobar rejects, each after its record_fail_band */
  const GateKind kinds[6] = {GateKind::Bump, GateKind::RhoSign, GateKind::RhoSign,
                             GateKind::RhoBar, GateKind::RhoBar, GateKind::RhoBar};
  for (std::size_t i = 1; i <= 100; ++i) {
    log.record_eval(i, 1.0,
                    i <= 6 ? tadah::hpo::EvalLoss::Failed : tadah::hpo::EvalLoss::Usable);
    if (i <= 6) {
      log.record_fail_band(static_cast<int>(tadah::hpo::FailBand::D));
      log.record_gate_reject(kinds[i - 1]);
    }
  }
  CHECK(log.gate_rejects(GateKind::Bump) == 1);
  CHECK(log.gate_rejects(GateKind::RhoSign) == 2);
  CHECK(log.gate_rejects(GateKind::RhoBar) == 3);
  const std::string txt = j.finish();
  CHECK_THAT(txt, Catch::Contains("D=6) bumpgate=1 rhogate=2 rhobargate=3") &&
                      Catch::Contains("    of which bump-support gate rejects: 1 "
                                      "(c + w/eta past RCUT, rejected before "
                                      "training)\n") &&
                      Catch::Contains("    of which rho-sign gate rejects: 2 "
                                      "(rho_min < -tol*rho_max on the RHOGATE "
                                      "window, rejected before training)\n") &&
                      Catch::Contains("    of which lattice-rhobar gate rejects: 3 "
                                      "(rhobar(a) not strictly decreasing, "
                                      "rejected before training)\n"));
  log.reset_for_test();
  for (GateKind k : {GateKind::Bump, GateKind::RhoSign, GateKind::RhoBar}) {
    CHECK(log.gate_rejects(k) == 0);
    CHECK(log.next_gate_note(k) == 1);
  }
  log.reset_for_test();

  SECTION("a bump-only run prints exactly the BUMPGATE lines of tadah 1.5.0") {
    Journal j2;
    for (std::size_t i = 1; i <= 100; ++i) {
      log.record_eval(i, 1.0,
                      i == 1 ? tadah::hpo::EvalLoss::Failed : tadah::hpo::EvalLoss::Usable);
      if (i == 1) {
        log.record_fail_band(static_cast<int>(tadah::hpo::FailBand::D));
        log.record_bump_gate_reject();
      }
    }
    const std::string t2 = j2.finish();
    CHECK_THAT(t2, Catch::Contains("D=1) bumpgate=1\n") &&
                       Catch::Contains("    of which bump-support gate rejects: 1 "
                                       "(c + w/eta past RCUT, rejected before "
                                       "training)\n"));
    CHECK(t2.find(" rhogate=") == std::string::npos);   // (the log's path has "rhogate")
    CHECK(t2.find("rho-sign") == std::string::npos);
  }
}

TEST_CASE("Density report: startup rows, RESULT, not derivable",
          "[hpo][rhogate][journal]") {
  PeriodicTable::initialize();

  SECTION("fixture summaries") {
    dt::DensityGates none;
    const auto theta = dt::summarise_density(none, fixture_ctx(fixture("theta_star_ON")));
    REQUIRE(theta.derivable);
    CHECK(theta.form == "SIGNED");
    CHECK(theta.weights == "W = 1");
    CHECK(theta.npts == 20001);
    CHECK(theta.r_lo == 0.15 * theta.r_hi);
    CHECK_FALSE(theta.non_increasing);   // the tail rises back to 0 at RCUTMB
    CHECK(theta.max_drho > 0.0);
    const std::string row = dt::format_density_rho(theta);
    CHECK_THAT(row, Catch::StartsWith("[0.8373, 5.5819] A, 20001 pts: min ") &&
                        Catch::Contains("; non-increasing on window: no; W = 1"));
    /* MONOTONE: non-increasing to rounding */
    Context mono = make_ream_ctx("MONOTONE", "0.5 1.0 0.1");
    const auto m = dt::summarise_density(none, mono);
    REQUIRE(m.derivable);
    CHECK(m.non_increasing);
    CHECK(dt::describe_density_form(m.form) ==
          "RHOFORM MONOTONE -- rho = W max(f_c, 0) sum a_n S_n, a_n >= 0 (rho >= 0 "
          "and non-increasing by construction)");
  }
  SECTION("an incomplete context: not derivable") {
    Context c = make_ream_ctx("SIGNED");
    c.remove("AMPGRIDMB");
    const auto s = dt::summarise_density(dt::DensityGates{}, c);
    CHECK_FALSE(s.derivable);
    CHECK(dt::format_density_rho(s) == "not derivable: DM_REAM density: AMPGRIDMB "
                                       "is missing.");
  }
  SECTION("startup rows, the RESULT line for the best and the resume case") {
    Context ctx = make_ream_ctx("SIGNED");
    TempFile seed(ream_seed("0.5 0.3 0.05"));
    TempFile tf("OPTIMIZER\nINIT\nSTRATEGY CONFIG\nSTRATEGY WARM " +
                seed.path.string() + "\nENDINIT\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\n"
                "ENDOPTIMIZER\nOPTIM AMPGRIDMB (3) -0.5 0.5\nRHOGATE 1e-5 R_LO 1.7\n"
                "RHOBARGATE bcc 3.304 0.98 1.1\nPC_ERMSE 0 1\n");
    HPOTargets trg(ctx, tf.path.string());
    const auto entries = dt::startup_density_report(ctx, trg);
    REQUIRE(entries.size() == 2);
    CHECK(entries[0].label == "-c pot");
    CHECK(entries[1].label == "WARM 1");
    CHECK_THAT(entries[1].note, Catch::Contains(seed.path.string()));
    {
      Journal j;
      HPOLog::instance().write_density_report(entries, true);
      const dt::DensityGates g = dt::build_density_gates(ctx, trg);
      HPOLog::instance().log_new_best(7, 1.5, 0, false, 0, false, 0, false, 0, false,
                                      0, false, 0.0, "");
      HPOLog::instance().record_density_best(7, 1.5, dt::summarise_density(g, ctx));
      /* a later, worse record is ignored */
      HPOLog::instance().record_density_best(8, 2.0, dt::summarise_density(g, ctx));
      const std::string txt = j.finish();
      CHECK_THAT(txt, Catch::Contains("  Density form          : RHOFORM SIGNED -- ") &&
                          Catch::Contains("  Density rho(r), -c pot: [1.7000, 5.5000] A, "
                                          "20001 pts: min ") &&
                          Catch::Contains("  Density rho(r), WARM 1: ") &&
                          Catch::Contains("  Density rhobar, -c pot: bcc a0 3.304 A, "
                                          "a/a0 [0.98, 1.1], 1001 pts: strictly "
                                          "decreasing (0 rising steps)") &&
                          Catch::Contains("  Density rho(r)          : best (eval 7) "
                                          "[1.7000, 5.5000] A") &&
                          Catch::Contains("; RHOGATE tol 1.0000e-05: PASS") &&
                          Catch::Contains("  Density rhobar          : best (eval 7) "
                                          "bcc a0 3.304 A"));
    }
    {
      /* A best without a density summary: the host records one for every
       * new best unless it could not build the density (the -c pot row of
       * the report then says "not derivable: <reason>"). HPO runs are not
       * resumed, so no best comes from an earlier session. */
      Journal j;
      HPOLog::instance().write_density_report(entries, false);
      HPOLog::instance().log_new_best(3, 1.0, 0, false, 0, false, 0, false, 0, false,
                                      0, false, 0.0, "");
      const std::string txt = j.finish();
      CHECK_THAT(txt, Catch::Contains("  Density rho(r)          : n/a (not "
                                      "derivable)"));
      CHECK(txt.find("predates") == std::string::npos);
      CHECK(txt.find("Density rhobar          :") == std::string::npos);
    }
    {
      Journal j;   /* not armed: no RESULT density line */
      HPOLog::instance().log_new_best(3, 1.0, 0, false, 0, false, 0, false, 0, false,
                                      0, false, 0.0, "");
      CHECK(j.finish().find("Density rho(r)") == std::string::npos);
    }
  }
}

/* ------------------------------------------------------------------------
 * Host-level cases (thread-local trainer cache, like the other HPO_Host
 * e2e cases): ./test_hpo "[rhogate_e2e]" / "[rhobargate_e2e]"
 * ------------------------------------------------------------------------ */
namespace {

/* The trainer's potential as text: its key set and values. */
std::string trainer_pot() {
  const auto* tr = Host::tls_trainer_for_test();
  REQUIRE(tr != nullptr);
  auto pot = const_cast<tadah::mlip::Trainer*>(tr)->get_param_file();
  const std::vector<char> s = pot.serialize();
  return std::string(s.begin(), s.end());
}

std::set<std::string> pot_keys(const std::string& pot) {
  std::set<std::string> k;
  std::istringstream is(pot);
  std::string line;
  while (std::getline(is, line)) {
    std::istringstream ls(line);
    std::string key;
    if (ls >> key) k.insert(key);
  }
  return k;
}

/* One e2e run directory: the validation set and a config.hpo per target. */
struct E2E {
  ScratchDir sd{"tadah_rhogate_e2e_"};
  fs::path val_cfg;
  E2E() {
    const fs::path val_db = sd.dir / "val.db";
    write_ta_db(val_db);
    val_cfg = sd.dir / "config.val";
    std::ofstream os(val_cfg);
    os << "DBFILE " << val_db.string() << "\n";
  }
  fs::path target(const std::string& name, const std::string& body) const {
    const fs::path p = sd.dir / name;
    std::ofstream os(p);
    os << "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\nENDOPTIMIZER\n"
          "LOSS L2\nPC_EMAE 0 1\nFAILSCORE 1e9\n" << body;
    return p;
  }
};

double graded(double v) {
  return tadah::hpo::graded_failscore(1e9, tadah::hpo::FailBand::D, v, 1.0);
}

}  // namespace

TEST_CASE("HPO_Host RHOGATE: graded reject before training, silent pass",
          "[rhogate_e2e][.]") {
  PeriodicTable::initialize();
  E2E e;
  const std::string rho_optim = "FAILSCORE_GRADING ON\nOPTIM AMPGRIDMB (3) -0.5 0.5\n";

  SECTION("reject, tolerance, v and the journal") {
    Journal j;
    Host::reset_globals_for_test();
    ForceRebuildGuard rebuild;
    const fs::path cfg = e.target("gate.hpo", rho_optim + "RHOGATE 1e-5\n");
    Context ctx = make_ream_ctx("SIGNED");
    ctx.add("HPOTARGET", cfg.string());
    HPOTargets trg(ctx, cfg.string());
    REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, false, true));
    Host host(ctx, trg, e.val_cfg.string());
    ParamVec p(1);
    p(0) = -0.2;
    double loss = 0.0;
    REQUIRE_NOTHROW(loss = host(p));
    CHECK(loss == graded(0x1.c7d66cd418f32p-2));   // gate_refs.py: 0.3990 + 0.0461
    CHECK(loss >= 0.999e9);
    CHECK(loss < 1e9);
    /* no training happened: every dimension is finite-differenced */
    ParamVec g(1);
    std::vector<int> needs_fd;
    REQUIRE(host.analytical_gradient(p, g, needs_fd));
    CHECK(needs_fd.size() == 1);
    CHECK(HPOLog::instance().gate_rejects(GateKind::RhoSign) == 1);
    /* a negative excursion inside the tolerance is silent */
    p(0) = -1e-6;
    CHECK(tadah::hpo::loss_is_feasible(host(p), 1e9));
    p(0) = -1e-4;
    CHECK(host(p) == graded(0x1.6310101179698p-13));   // v_peak 1.6064e-4
    CHECK(HPOLog::instance().gate_rejects(GateKind::RhoSign) == 2);
    host.flush_remaining();
    const std::string txt = j.finish();
    CHECK_THAT(txt, Catch::Contains("eval 1: rho-sign gate reject (rho_min/rho_max = "
                                    "-3.9903e-01 at r = 4.4025 A, tol 1.0000e-05, "
                                    "v = 4.4515e-01); total rejects=1") &&
                        Catch::Contains("; total rejects=2") &&
                        Catch::Contains("D=2 (pot build / gated)") &&
                        Catch::Contains("    of which rho-sign gate rejects: 2 "));
    Host::reset_globals_for_test();
  }
  SECTION("a pass is bitwise the run without the gate; ctx and pot unchanged") {
    for (bool rebuild : {false, true}) {
      CAPTURE(rebuild);
      std::unique_ptr<ForceRebuildGuard> guard;
      if (rebuild) guard = std::make_unique<ForceRebuildGuard>();
      double loss[2];
      std::string pot[2];
      for (int gated = 0; gated < 2; ++gated) {
        /* the first host builds its trainer; without rebuild the second
         * trains on the cached one */
        std::optional<ForceRebuildGuard> first;
        if (gated == 0) first.emplace();
        Journal j;
        Host::reset_globals_for_test();
        /* one file name: HPOTARGET is part of the potential */
        const fs::path cfg = e.target("pass.hpo",
                                      rho_optim + (gated ? "RHOGATE 1e-5\n" : ""));
        Context ctx = make_ream_ctx("SIGNED");
        ctx.add("HPOTARGET", cfg.string());
        HPOTargets trg(ctx, cfg.string());
        Host host(ctx, trg, e.val_cfg.string());
        ParamVec p(1);
        p(0) = 0.1;
        loss[gated] = host(p);
        pot[gated] = trainer_pot();
        host.flush_remaining();
        if (gated) CHECK(HPOLog::instance().gate_rejects(GateKind::RhoSign) == 0);
        Host::reset_globals_for_test();
      }
      CHECK(tadah::hpo::loss_is_feasible(loss[0], 1e9));
      CHECK(bits(loss[0]) == bits(loss[1]));
      CHECK(pot_keys(pot[0]) == pot_keys(pot[1]));
      CHECK(pot[0] == pot[1]);
    }
  }
  SECTION("FAILSCORE_GRADING OFF: the flat FAILSCORE") {
    Host::reset_globals_for_test();
    ForceRebuildGuard rebuild;
    const fs::path cfg = e.target("off.hpo", "OPTIM AMPGRIDMB (3) -0.5 0.5\nRHOGATE 1e-5\n");
    Context ctx = make_ream_ctx("SIGNED");
    ctx.add("HPOTARGET", cfg.string());
    HPOTargets trg(ctx, cfg.string());
    Host host(ctx, trg, e.val_cfg.string());
    ParamVec p(1);
    p(0) = -0.2;
    CHECK(host(p) == 1e9);
    host.flush_remaining();
    Host::reset_globals_for_test();
  }
  SECTION("unarmed: the cached verdict rejects every candidate, no rescan") {
    Journal j;
    Host::reset_globals_for_test();
    ForceRebuildGuard rebuild;
    const fs::path cfg = e.target("unarmed.hpo", "FAILSCORE_GRADING ON\n"
                                                 "OPTIM CEMBFUNC (1) -0.5 0.1\n"
                                                 "RHOGATE 1e-5 SEED WARN\n");
    Context ctx = make_ream_ctx("SIGNED", "0.5 0.3 -0.2");
    ctx.add("HPOTARGET", cfg.string());
    HPOTargets trg(ctx, cfg.string());
    /* startup refuses it under either SEED mode ... */
    REQUIRE_THROWS_WITH(validate_hpo_setup(ctx, trg, false, true),
                        Catch::Contains("every evaluation would be rejected"));
    /* ... and a host built anyway rejects every candidate alike */
    Host host(ctx, trg, e.val_cfg.string());
    const std::size_t scans = dt::density_gate_scans_for_test();
    ParamVec p(1);
    p(0) = -0.3;
    const double l1 = host(p);
    p(0) = 0.05;
    const double l2 = host(p);
    CHECK(l1 == graded(0x1.c7d66cd418f32p-2));
    CHECK(l2 == l1);
    CHECK(dt::density_gate_scans_for_test() == scans);
    host.flush_remaining();
    Host::reset_globals_for_test();
  }
  SECTION("BUMPGATE and RHOGATE: the first failing gate is counted, the sum "
          "grades; a bump-only reject is as in tadah 1.5.0") {
    const std::string boxes = "FAILSCORE_GRADING ON\nOPTIM AMPGRIDMB (3) -0.5 0.5\n"
                              "OPTIM CGRIDMB (3) 4.0 5.2\nOPTIM SGRIDMB (3) 2.0 4.0\n";
    for (int gated = 0; gated < 2; ++gated) {
      CAPTURE(gated);
      Journal j;
      Host::reset_globals_for_test();
      ForceRebuildGuard rebuild;
      const fs::path cfg = e.target("both.hpo", boxes + (gated ? "RHOGATE 1e-5\n" : ""));
      Context ctx = make_ream_ctx("SIGNED");
      ctx.add("HPOTARGET", cfg.string());
      HPOTargets trg(ctx, cfg.string());
      REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, false, true));
      const auto bump_tbl = dt::build_bump_gate_table(ctx, trg);
      REQUIRE_FALSE(bump_tbl.empty());
      Host host(ctx, trg, e.val_cfg.string());
      /* p = (AMPGRIDMB[3], CGRIDMB[3], SGRIDMB[3]): reach 5.2 + 3/2 > 5.5 */
      ParamVec p(3);
      p(0) = 0.1; p(1) = 5.2; p(2) = 2.0;
      const double v_bump = dt::bump_gate_violation(bump_tbl, p);
      REQUIRE(v_bump > 0.0);
      /* bump only (rho >= 0): tadah 1.5.0's loss and line, with or without
       * RHOGATE */
      CHECK(host(p) == graded(v_bump));
      const std::string line1 =
          "eval 1: bump-support gate reject (c + w/eta past the cutoff, "
          "violation=" + std::to_string(v_bump) +
          ", graded band D); total rejects=1\n";
      p(0) = -0.2;   /* now RHOGATE fails too */
      const double l2 = host(p);
      if (gated) {
        const dt::DensityGates g = dt::build_density_gates(ctx, trg);
        const auto d = dt::check_density_gates(
            g, dt::seed_context(ctx, trg, {-0.2, 5.2, 2.0}, g));
        REQUIRE(d.rho->fail);
        CHECK(l2 == graded(v_bump + d.rho->v));
      } else {
        CHECK(l2 == graded(v_bump));
      }
      CHECK(HPOLog::instance().gate_rejects(GateKind::Bump) == 2);
      CHECK(HPOLog::instance().gate_rejects(GateKind::RhoSign) == 0);
      host.flush_remaining();
      const std::string txt = j.finish();
      CHECK_THAT(txt, Catch::Contains(line1) &&
                          Catch::Contains("    of which bump-support gate rejects: 2 "));
      CHECK(txt.find("of which rho-sign") == std::string::npos);
      CHECK((txt.find("also failing: rho-sign gate") != std::string::npos) == (gated == 1));
      Host::reset_globals_for_test();
    }
  }
  SECTION("the RESULT density line of the best") {
    Journal j;
    Host::reset_globals_for_test();
    ForceRebuildGuard rebuild;
    const fs::path cfg = e.target("result.hpo", rho_optim + "RHOGATE 1e-5\n");
    Context ctx = make_ream_ctx("SIGNED");
    ctx.add("HPOTARGET", cfg.string());
    HPOTargets trg(ctx, cfg.string());
    HPOLog::instance().write_density_report(dt::startup_density_report(ctx, trg), false);
    {
      Host host(ctx, trg, e.val_cfg.string());
      ParamVec p(1);
      p(0) = 0.1;
      host(p);
      host.flush_remaining();
    }
    const std::string txt = j.finish();
    CHECK_THAT(txt, Catch::Contains("  Density rho(r), -c pot: [0.8250, 5.5000] A") &&
                        Catch::Contains("  Density rho(r)          : best (eval ") &&
                        Catch::Contains("; RHOGATE tol 1.0000e-05: PASS\n"));
    Host::reset_globals_for_test();
  }
}

TEST_CASE("HPO_Host RHOBARGATE: graded reject before training",
          "[rhobargate_e2e][.]") {
  PeriodicTable::initialize();
  E2E e;
  const std::string optim = "FAILSCORE_GRADING ON\nOPTIM AMPGRIDMB (2) 0.0 2.0\n";

  SECTION("reject and silent pass") {
    double gated_loss[2] = {0.0, 0.0};
    for (int gated = 1; gated >= 0; --gated) {
      Journal j;
      Host::reset_globals_for_test();
      ForceRebuildGuard rebuild;
      const fs::path cfg = e.target("rb.hpo", optim + (gated ? "RHOBARGATE bcc 3.304 "
                                                               "0.98 1.10\n" : ""));
      Context ctx = make_ream_ctx("SIGNED");
      ctx.add("HPOTARGET", cfg.string());
      HPOTargets trg(ctx, cfg.string());
      REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, false, true));
      Host host(ctx, trg, e.val_cfg.string());
      ParamVec p(1);
      p(0) = 0.05;
      gated_loss[gated] = host(p);
      if (gated) {
        p(0) = 1.0;
        CHECK(host(p) == graded(0x1.2e6fea5fb8946p-3));
        CHECK(HPOLog::instance().gate_rejects(GateKind::RhoBar) == 1);
      }
      host.flush_remaining();
      const std::string txt = j.finish();
      if (gated)
        CHECK_THAT(txt, Catch::Contains("eval 2: lattice-rhobar gate reject (896 "
                                        "rises over a/a0 [0.98, 1.1] on bcc, first "
                                        "at a/a0 = 0.9800, v = 1.4767e-01); total "
                                        "rejects=1") &&
                            Catch::Contains("    of which lattice-rhobar gate "
                                            "rejects: 1 "));
      Host::reset_globals_for_test();
    }
    CHECK(tadah::hpo::loss_is_feasible(gated_loss[1], 1e9));
    CHECK(bits(gated_loss[1]) == bits(gated_loss[0]));
  }
  SECTION("the seed fails the wider window: a startup error naming the rise") {
    const fs::path cfg = e.target("wide.hpo", optim + "RHOBARGATE bcc 3.304 0.75 1.10\n");
    Context ctx = make_ream_ctx("SIGNED");
    ctx.add("HPOTARGET", cfg.string());
    HPOTargets trg(ctx, cfg.string());
    REQUIRE_THROWS_WITH(validate_hpo_setup(ctx, trg, false, true),
                        Catch::Contains("RHOBARGATE: seed -c pot (INIT CONFIG) fails "
                                        "the lattice rhobar(a) gate: rhobar rises at "
                                        "264 of 1000 steps on bcc (a0 3.304 A, a/a0 "
                                        "[0.75, 1.1], 1001 points), first at a/a0 = "
                                        "0.7500, v = 1.9432e-01."));
  }
  SECTION("NONNEG fails like SIGNED, MONOTONE passes") {
    for (const char* form : {"NONNEG", "MONOTONE"}) {
      CAPTURE(form);
      Host::reset_globals_for_test();
      ForceRebuildGuard rebuild;
      const fs::path cfg = e.target("form.hpo", optim + "RHOBARGATE bcc 3.304 0.98 1.10\n");
      Context ctx = make_ream_ctx(form);
      ctx.add("HPOTARGET", cfg.string());
      HPOTargets trg(ctx, cfg.string());
      REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, false, true));
      Host host(ctx, trg, e.val_cfg.string());
      ParamVec p(1);
      p(0) = 1.0;
      const double l = host(p);
      CHECK(tadah::hpo::loss_is_feasible(l, 1e9) == (std::string(form) == "MONOTONE"));
      host.flush_remaining();
      Host::reset_globals_for_test();
    }
  }
  SECTION("both density gates fail: counted under rhogate only") {
    Journal j;
    Host::reset_globals_for_test();
    ForceRebuildGuard rebuild;
    const fs::path cfg = e.target("both.hpo", "FAILSCORE_GRADING ON\n"
                                              "OPTIM AMPGRIDMB (2,3) -2 2\n"
                                              "RHOGATE 1e-5\n"
                                              "RHOBARGATE bcc 3.304 0.98 1.10\n");
    Context ctx = make_ream_ctx("SIGNED");
    ctx.add("HPOTARGET", cfg.string());
    HPOTargets trg(ctx, cfg.string());
    REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, false, true));
    Host host(ctx, trg, e.val_cfg.string());
    ParamVec p(2);
    p(0) = 1.0;
    p(1) = -0.2;
    const double l = host(p);
    const dt::DensityGates g = dt::build_density_gates(ctx, trg);
    const auto d = dt::check_density_gates(g, dt::seed_context(ctx, trg, {1.0, -0.2}, g));
    CHECK(l == graded(d.rho->v + d.rhobar->v));
    CHECK(HPOLog::instance().gate_rejects(GateKind::RhoSign) == 1);
    CHECK(HPOLog::instance().gate_rejects(GateKind::RhoBar) == 0);
    host.flush_remaining();
    CHECK_THAT(j.finish(), Catch::Contains("rho-sign gate reject (") &&
                               Catch::Contains("; also failing: lattice-rhobar gate (v = "));
    Host::reset_globals_for_test();
  }
}

TEST_CASE("Density gates: cost per candidate", "[rho_gates_bench][.]") {
  PeriodicTable::initialize();
  const Context ctx = fixture_ctx(fixture("theta_star_ON"));
  const double rcut = std::stod(fixture("theta_star_ON").rcutmb);
  auto time_ms = [](auto&& fn) {
    const int reps = 200;
    fn();   // warm-up
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) fn();
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count() / reps;
  };
  struct Row { const char* what; double ms; };
  std::vector<Row> rows;
  for (const auto& w : {std::make_pair("RHOGATE [1.7168, RCUTMB], 20001 pts",
                                       std::optional<double>(1.7168)),
                        std::make_pair("RHOGATE [0.15 RCUTMB, RCUTMB], 20001 pts",
                                       std::optional<double>())}) {
    const auto t = rho_table(1e-5, w.second);
    rows.push_back({w.first, time_ms([&] { (void)dt::eval_rho_gate(t, ctx); })});
  }
  for (const std::size_t n : {std::size_t(1001), std::size_t(3501)}) {
    const auto t = rhobar_table("bcc", 3.304, 0.75, 1.10, n, rcut);
    rows.push_back({n == 1001 ? "RHOBARGATE bcc [0.75, 1.10], 1001 pts"
                              : "RHOBARGATE bcc [0.75, 1.10], 3501 pts",
                    time_ms([&] { (void)dt::eval_rhobar_gate(t, ctx); })});
  }
  std::cout << "density gate cost per candidate (theta_star_ON, 20 Blip5 bumps):\n";
  for (const auto& r : rows)
    std::cout << "  " << r.what << ": " << r.ms << " ms\n";
  for (const auto& r : rows) CHECK(r.ms < 50.0);   // a sanity bound, not the target
}
