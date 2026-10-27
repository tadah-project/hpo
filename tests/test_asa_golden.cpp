/* The ASA golden-neutrality gate (see external/asa/TADAH-PATCHES.md).
 *
 * Every compile-time gate enabled in tadah_asa_config.h claims to be
 * trajectory-neutral at its default option values. This test PROVES it:
 * the featured build and the pristine all-gates-off build run the same
 * deterministic problem with the same explicit LCG, and the recorded
 * trajectories must match bitwise. If this test fails after a gate
 * change: flip the offending gate off, remove its PARAM keys, document
 * the exclusion (the verify-or-drop procedure). */

#include "catch2/catch.hpp"

#include "asa_golden_common.h"
#include "finish_within.h"
#include <tadah/hpo/asa_testing.h>

#include <chrono>
#include <map>
#include <string>
#include <vector>

TEST_CASE("ASA golden gate: featured == pristine at default options",
          "[HPO][asa][golden]")
{
    constexpr std::size_t dim = 3;
    const double lo[dim] = {-4.0, -4.0, -4.0};
    const double hi[dim] = {4.0, 4.0, 4.0};
    const double x0[dim] = {2.5, -1.5, 0.5};
    const long limit_generated = 2000;

    asa_golden::reset(777);
    const auto p = run_golden_pristine(dim, lo, hi, x0, limit_generated);
    const asa_golden::Trajectory ptraj = asa_golden::traj();  /* copy */

    asa_golden::reset(777);
    const auto f = tadah::hpo::asa_testing::run_raw_featured(
        &asa_golden::golden_cost, &asa_golden::golden_rng, dim, lo, hi, x0,
        limit_generated);
    const asa_golden::Trajectory& ftraj = asa_golden::traj();

    INFO("pristine evals=" << ptraj.f.size()
         << " featured evals=" << ftraj.f.size());
    REQUIRE(ptraj.f.size() > 100);  /* the run actually ran */
    REQUIRE(ptraj.f.size() == ftraj.f.size());
    CHECK(ptraj.flat_x == ftraj.flat_x);  /* bitwise double compare */
    CHECK(ptraj.f == ftraj.f);
    CHECK(p.exit_status == f.exit_status);
    CHECK(p.n_accepted == f.n_accepted);
    CHECK(p.n_generated == f.n_generated);
    CHECK(p.best_x == f.best_x);
    CHECK(p.final_cost == f.final_cost);
}

TEST_CASE("ASA golden gate: featured == pristine when every state is a new "
          "best", "[HPO][asa][golden]")
{
    /* Only the featured build resizes its block of generated states: its
     * periodic test (every Acceptance_Frequency_Modulus = 100 acceptances)
     * averages, over the last Gener_Mov_Avr = 3 tests, the number of states
     * generated since the last new best. Here that number is 0 at every
     * test, and the average {1, 1, 0} truncated to a block of 0 states at
     * acceptance 100: the featured asa() then looped for ever while the
     * pristine one went on. At Gener_Block = Gener_Block_Max = 1 the block
     * must stay at one state, which is canonical serial ASA. */
    constexpr std::size_t dim = 2;
    const double lo[dim] = {0.0, 0.0};
    const double hi[dim] = {1.0, 1.0};
    const double x0[dim] = {0.5, 0.5};
    const long limit_generated = 600;

    asa_golden::reset(777);
    const auto p = run_golden_pristine(dim, lo, hi, x0, limit_generated,
                                       &asa_golden::improving_cost);
    const asa_golden::Trajectory ptraj = asa_golden::traj();  /* copy */

    asa_golden::reset(777);
    tadah::hpo::asa_testing::RawResult f;
    hpo_test::finish_within(
        std::chrono::seconds(60),
        "the featured asa() on a cost whose every state is a new best",
        [&] {
            f = tadah::hpo::asa_testing::run_raw_featured(
                &asa_golden::improving_cost, &asa_golden::golden_rng, dim,
                lo, hi, x0, limit_generated);
        });
    const asa_golden::Trajectory& ftraj = asa_golden::traj();

    INFO("pristine evals=" << ptraj.f.size()
         << " featured evals=" << ftraj.f.size()
         << " pristine accepted=" << p.n_accepted);
    REQUIRE(p.n_accepted > 200);  /* past two periodic tests */
    REQUIRE(ptraj.f.size() == ftraj.f.size());
    CHECK(ptraj.flat_x == ftraj.flat_x);  /* bitwise double compare */
    CHECK(ptraj.f == ftraj.f);
    CHECK(p.exit_status == f.exit_status);
    CHECK(p.n_accepted == f.n_accepted);
    CHECK(p.n_generated == f.n_generated);
    CHECK(p.best_x == f.best_x);
    CHECK(p.final_cost == f.final_cost);
}

namespace {

/* The periodic tests of a run of the featured asa() with its reannealing
 * on (Reanneal_Parameters 1, the default): each test evaluates the
 * tangents of the dimensions (Locate_Cost 3), so each maximal run of such
 * calls is one test. Returns, per test, the counts of accepted and of
 * generated states when it ran. */
std::vector<tadah::hpo::asa_testing::RawCall>
periodic_tests(const tadah::hpo::asa_testing::RawResult& r)
{
    std::vector<tadah::hpo::asa_testing::RawCall> tests;
    bool in_test = false;
    for (const auto& c : r.calls) {
        const bool tangent = c.locate_cost == 3;
        if (tangent && !in_test) tests.push_back(c);
        in_test = tangent;
    }
    return tests;
}

/* The featured asa() on golden_cost in three dimensions with the PARAM
 * lines `params`, for 400 generated states (the run ends with the block
 * that passes 400), and Maximum_Cost_Repeat 0: no periodic test ends the
 * run before its reannealing. */
tadah::hpo::asa_testing::RawResult
periodic_run(std::map<std::string, double> params)
{
    params["Maximum_Cost_Repeat"] = 0.0;
    constexpr std::size_t dim = 3;
    const double lo[dim] = {-4.0, -4.0, -4.0};
    const double hi[dim] = {4.0, 4.0, 4.0};
    const double x0[dim] = {2.5, -1.5, 0.5};
    asa_golden::reset(777);
    return tadah::hpo::asa_testing::run_raw_featured(
        &asa_golden::golden_cost, &asa_golden::golden_rng, dim, lo, hi, x0,
        400, params);
}

}  // namespace

/* Pins: ASA:ASA with PARAM Acceptance_Frequency_Modulus 5 (and
 * Generated_Frequency_Modulus 0, so that only the acceptances call the
 * periodic tests) runs a periodic test each time the count of accepted
 * states reaches a multiple of 5: the k-th test follows the 5k-th
 * acceptance -- in serial mode (Gener_Block 1) right after it, at exactly
 * 5k accepted states; in block mode (Gener_Block 4, Gener_Mov_Avr 0, every
 * block 4 states) after the block that holds it, at 5k to 5k + 3 accepted
 * states (a block accepts at most 4). Every multiple of 5 that the run
 * reaches gets its test.
 * Before the fix: serial mode passes as well; in block mode the test ran
 * only when the block's last state was accepted and brought the count to a
 * multiple of 5: of the multiples a block passed, only those reached by its
 * last state got a test. */
TEST_CASE("ASA block mode: a periodic test follows every "
          "Acceptance_Frequency_Modulus acceptances", "[HPO][asa][golden]")
{
    for (const double block : {1.0, 4.0}) {
        DYNAMIC_SECTION("Gener_Block " << block) {
            const auto r = periodic_run({{"Gener_Block", block},
                                         {"Gener_Mov_Avr", 0.0},
                                         {"Acceptance_Frequency_Modulus", 5.0},
                                         {"Generated_Frequency_Modulus", 0.0}});
            const auto tests = periodic_tests(r);
            INFO("accepted states " << r.n_accepted << ", generated "
                 << r.n_generated << ", periodic tests " << tests.size());
            REQUIRE(r.n_accepted >= 40);
            CHECK(tests.size() == static_cast<std::size_t>(r.n_accepted / 5));
            for (std::size_t k = 1; k <= tests.size(); ++k) {
                INFO("periodic test " << k);
                const long a = tests[k - 1].n_accepted;
                const long first = 5 * static_cast<long>(k);
                CHECK(a >= first);
                CHECK(a <= first + (block > 1.0 ? 3 : 0));
            }
        }
    }
}

/* Pins: ASA:ASA in block mode (Gener_Block 4, Gener_Mov_Avr 0) with PARAM
 * Generated_Frequency_Modulus 42 (and Acceptance_Frequency_Modulus 0) runs
 * a periodic test after each block in which the count of generated states
 * reaches a multiple of 42: the k-th test at 42k to 42k + 3 generated
 * states (a block generates 4), one for each multiple of 42 the run
 * reaches.
 * Before the fix: the test ran only when a block ended on a multiple of 42
 * -- a block of 4 states ends on 4j, so only on the multiples of 84: every
 * other test was missing. */
TEST_CASE("ASA block mode: a periodic test follows every "
          "Generated_Frequency_Modulus generated states", "[HPO][asa][golden]")
{
    const auto r = periodic_run({{"Gener_Block", 4.0},
                                 {"Gener_Mov_Avr", 0.0},
                                 {"Acceptance_Frequency_Modulus", 0.0},
                                 {"Generated_Frequency_Modulus", 42.0}});
    const auto tests = periodic_tests(r);
    INFO("generated states " << r.n_generated << ", periodic tests "
         << tests.size());
    REQUIRE(r.n_generated >= 400);
    CHECK(tests.size() == static_cast<std::size_t>(r.n_generated / 42));
    for (std::size_t k = 1; k <= tests.size(); ++k) {
        INFO("periodic test " << k);
        const long g = tests[k - 1].n_generated;
        const long first = 42 * static_cast<long>(k);
        CHECK(g >= first);
        CHECK(g <= first + 3);
    }
}
