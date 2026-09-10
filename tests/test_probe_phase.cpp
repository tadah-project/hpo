// test_probe_phase.cpp
//
// The off-budget probe phase (1.5.0), the infrastructure the INIT_MODE JITTER
// AUTO calibration runs inside. Every case here corresponds to something that
// silently breaks if the phase is not bracketed:
//
//   * StagnationStop firing a restart partway through a calibration, because
//     the annealers run inside RestartPolicy and hundreds of deliberately
//     non-improving probes look exactly like a stalled search;
//   * checkpoints written with g_iter advanced over evals the annealer has no
//     state behind;
//   * probes poisoning a chain's warm-start lineage -- the failure class
//     behind the run15/16/17 artefact.
//
// The warm-start case also pins the thing every reviewer suggests and which
// is a defect: namespacing by offsetting warmstart::current_chain(). That
// value becomes EvalResult::chain_id, which is written into a FIXED-WIDTH
// SLOT column and used as the durable-best comparison key, so a large offset
// corrupts artefacts rather than isolating a cache.

#include "catch2/catch.hpp"

#include <tadah/hpo/anneal_core.h>
#include <tadah/hpo/probe_phase.h>
#include <tadah/hpo/warmstart_cache.h>

#include <string>

using namespace tadah::hpo;

TEST_CASE("probe::Scope brackets the phase and nests", "[HPO][probe_phase]")
{
    CHECK_FALSE(probe::active());
    {
        probe::Scope outer;
        CHECK(probe::active());
        {
            probe::Scope inner;      /* nested: still active, no double-clear */
            CHECK(probe::active());
        }
        CHECK(probe::active());      /* inner must NOT have ended the phase */
    }
    CHECK_FALSE(probe::active());
}

TEST_CASE("probe::Scope clears the phase when an exception unwinds",
          "[HPO][probe_phase]")
{
    /* A probe that throws must not leave the process permanently convinced
     * it is calibrating -- stagnation and checkpointing would stay off for
     * the rest of the run. */
    try {
        probe::Scope s;
        REQUIRE(probe::active());
        throw std::runtime_error("probe blew up");
    } catch (const std::exception&) {
    }
    CHECK_FALSE(probe::active());
}

TEST_CASE("warm-start: probe entries do not collide with a chain's",
          "[HPO][probe_phase][warmstart]")
{
    const std::string script = "probe_phase_test.in";
    const std::string var = "hot_a0";
    const std::size_t chain = 3;

    warmstart::set(chain, script, var, 1.0);          /* production value */
    {
        probe::Scope s;
        /* A probe must not SEE the chain's value as its own cached start... */
        const auto seen = warmstart::get(chain, script, var);
        CHECK((!seen.has_value() || *seen != 1.0));
        warmstart::set(chain, script, var, 99.0);     /* ...nor overwrite it */
    }
    const auto after = warmstart::get(chain, script, var);
    REQUIRE(after.has_value());
    CHECK(*after == 1.0);      /* the chain's lineage is intact */
}

TEST_CASE("warm-start: the probe key space is reserved, not current_chain()",
          "[HPO][probe_phase][warmstart]")
{
    /* current_chain() must be untouched by the phase: it flows into
     * EvalResult::chain_id -> the SLOT column / durable-best key. */
    warmstart::current_chain() = 7;
    {
        probe::Scope s;
        CHECK(warmstart::current_chain() == 7u);
        warmstart::set(warmstart::current_chain(), "s.in", "v", 2.0);
        CHECK(warmstart::current_chain() == 7u);
    }
    CHECK(warmstart::current_chain() == 7u);
    /* And the probe's write did not land under the real chain key. */
    CHECK_FALSE(warmstart::get(7, "s.in", "v").has_value());
    warmstart::current_chain() = 0;
}

TEST_CASE("jitter status text names each band distinctly",
          "[HPO][probe_phase][jitter]")
{
    using namespace tadah::hpo::anneal;
    CHECK(jitter_status_text(JitterOk) == "ok");
    /* A feasibility wall (BUMPGATE) is not a cliff and not insensitivity:
     * they call for different actions, so they must not read alike. */
    CHECK(jitter_status_text(JitterGateReject) == "gate-reject");
    CHECK(jitter_status_text(JitterCliff) == "cliff");
    CHECK(jitter_status_text(JitterInsensitive) == "insensitive");
    CHECK(jitter_status_text(JitterSupportCapped | JitterInsensitive)
          == "support-capped+insensitive");
}
