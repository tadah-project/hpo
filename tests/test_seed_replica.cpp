// test_seed_replica.cpp
//
// anneal::seed_replica -- the replica/walker initialisation kernel shared by
// ANNEAL_PT and ANNEAL_PA (1.5.0; the two drivers carried near-identical
// copies before, and PA compared INIT_MODE against a raw 0).
//
// The risk in extracting it is not the coordinates, it is how much of the
// chain's private RNG stream initialisation CONSUMES. Every PT/PA trajectory
// at a fixed SEED is reproducible only while that is stable, and it is
// invisible to any assertion on x alone: two implementations can produce
// identical starting points while leaving the stream in different places, and
// the run then diverges on the first production proposal.
//
// So the oracle here is an inline transcription of the pre-extraction loop
// (copied verbatim from algorithms_pt.h), and every case checks the stream
// POSITION afterwards by drawing from both generators and comparing.

#include "catch2/catch.hpp"

#include <tadah/hpo/anneal_core.h>
#include <tadah/hpo/anneal_keys_gen.h>
#include <tadah/hpo/optimizer_spec.h>

#include <dlib/matrix.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

using namespace tadah::hpo;
using Vec = dlib::matrix<double, 0, 1>;

namespace {

/* VERBATIM transcription of the loop body that lived in algorithms_pt.h
 * before seed_replica existed. Do not "simplify" it -- its value is that it
 * is the old code. */
void reference_seed_replica(anneal::ChainState<Vec>& c, std::uint64_t seed,
                            std::size_t k, const Vec& x0,
                            const std::vector<double>& proposal_scale,
                            const OptimizerSpec& sp, const anneal::Knobs& kb)
{
    const std::size_t dim = sp.low.size();
    c.rng = anneal::make_stream(seed, anneal::SALT_CHAIN, k);
    c.V   = proposal_scale;
    c.x   = Vec(dim);
    if (k == 0 || kb.init_mode == anneal_enums::INIT_MODE::JITTER) {
        std::normal_distribution<double> nd(0.0, 1.0);
        for (std::size_t i = 0; i < dim; ++i) {
            long double v = static_cast<long double>(x0(i));
            if (k > 0) {
                auto draw = [&]() {
                    return static_cast<long double>(x0(i)) +
                           static_cast<long double>(proposal_scale[i]) *
                               nd(c.rng);
                };
                v = draw();
                (void)anneal::constrain_proposal(
                    v, sp.low[i], sp.high[i], kb.boundary_mode, draw);
            }
            c.x(i) = static_cast<double>(v);
        }
    } else {
        for (std::size_t i = 0; i < dim; ++i)
            c.x(i) = std::uniform_real_distribution<>(sp.low[i],
                                                      sp.high[i])(c.rng);
    }
    c.best_x = c.x;
}

OptimizerSpec spec(std::size_t dim, anneal::AlgoKind kind)
{
    OptimizerSpec sp;
    sp.lib  = "TADAH";
    sp.algo = kind == anneal::AlgoKind::PT ? "ANNEAL_PT" : "ANNEAL_PA";
    sp.low.assign(dim, -1.0);
    sp.high.assign(dim, 1.0);
    sp.is_int.assign(dim, false);
    sp.log_scaled.assign(dim, false);
    sp.stop.maxeval = 1000;
    return sp;
}

/* Identical stream POSITION, not merely identical coordinates. */
void require_same_stream(std::mt19937_64 a, std::mt19937_64 b)
{
    for (int i = 0; i < 8; ++i) REQUIRE(a() == b());
}

}  // namespace

TEST_CASE("seed_replica matches the pre-extraction loop, stream included",
          "[HPO][anneal_core][seed_replica]")
{
    constexpr std::uint64_t kSeed = 20260910u;
    const std::size_t dim = 5;

    /* A seed close to the upper wall so REDRAW/REFLECT actually fire and the
     * data-dependent consumption path is exercised rather than skipped. */
    Vec x0(static_cast<long>(dim));
    for (std::size_t i = 0; i < dim; ++i) x0(static_cast<long>(i)) = 0.92;

    for (const char* mode : {"JITTER", "UNIFORM"}) {
        for (const char* boundary : {"REFLECT", "REDRAW"}) {
            for (std::size_t k : {std::size_t{0}, std::size_t{1},
                                  std::size_t{7}}) {
                INFO("INIT_MODE=" << mode << " BOUNDARY_MODE=" << boundary
                                  << " k=" << k);
                auto sp = spec(dim, anneal::AlgoKind::PT);
                sp.param_raw["INIT_MODE"] = mode;
                sp.param_raw["BOUNDARY_MODE"] = boundary;
                /* Wide enough that a 0.92 start leaves the box often. */
                sp.param["STEP_FRAC"] = 0.4;
                const auto kb = anneal::Knobs::parse(sp, anneal::AlgoKind::PT);
                const auto scale = anneal::make_initial_steps(sp, kb);

                anneal::ChainState<Vec> got, want;
                anneal::seed_replica(got, kSeed, k, x0, scale, scale, sp, kb);
                reference_seed_replica(want, kSeed, k, x0, scale, sp, kb);

                REQUIRE(got.x.size() == want.x.size());
                for (long i = 0; i < got.x.size(); ++i)
                    CHECK(got.x(i) == want.x(i));
                for (long i = 0; i < got.best_x.size(); ++i)
                    CHECK(got.best_x(i) == want.best_x(i));
                REQUIRE(got.V == want.V);
                require_same_stream(got.rng, want.rng);
            }
        }
    }
}

TEST_CASE("seed_replica: slot 0 is the seed and consumes nothing",
          "[HPO][anneal_core][seed_replica]")
{
    constexpr std::uint64_t kSeed = 7u;
    const std::size_t dim = 4;
    Vec x0(static_cast<long>(dim));
    for (std::size_t i = 0; i < dim; ++i) x0(static_cast<long>(i)) = 0.25;

    for (const char* mode : {"JITTER", "UNIFORM"}) {
        INFO("INIT_MODE=" << mode);
        auto sp = spec(dim, anneal::AlgoKind::PT);
        sp.param_raw["INIT_MODE"] = mode;
        const auto kb = anneal::Knobs::parse(sp, anneal::AlgoKind::PT);
        const auto scale = anneal::make_initial_steps(sp, kb);

        anneal::ChainState<Vec> c;
        anneal::seed_replica(c, kSeed, 0, x0, scale, scale, sp, kb);

        /* Exactly the seed, under BOTH modes: slot 0 ignores INIT_MODE. */
        for (long i = 0; i < c.x.size(); ++i) CHECK(c.x(i) == x0(i));

        /* And the stream is untouched -- a fresh one must still agree. The
         * normal_distribution is constructed even here, so this pins that
         * constructing it does not by itself consume. */
        require_same_stream(c.rng,
                            anneal::make_stream(kSeed, anneal::SALT_CHAIN, 0));
    }
}

TEST_CASE("seed_replica: one normal_distribution per replica, not per dim",
          "[HPO][anneal_core][seed_replica]")
{
    /* std::normal_distribution generates variates in PAIRS and caches the
     * second. Constructing it inside the dimension loop would therefore
     * consume twice as much of the stream while still producing plausible
     * coordinates -- the exact silent divergence this guards. */
    constexpr std::uint64_t kSeed = 99u;
    const std::size_t dim = 6;
    Vec x0(static_cast<long>(dim));
    for (std::size_t i = 0; i < dim; ++i) x0(static_cast<long>(i)) = 0.0;

    auto sp = spec(dim, anneal::AlgoKind::PT);
    sp.param_raw["INIT_MODE"] = "JITTER";
    sp.param["STEP_FRAC"] = 0.05;      /* small: no boundary redraws */
    const auto kb = anneal::Knobs::parse(sp, anneal::AlgoKind::PT);
    const auto scale = anneal::make_initial_steps(sp, kb);

    anneal::ChainState<Vec> c;
    anneal::seed_replica(c, kSeed, 3, x0, scale, scale, sp, kb);

    /* Reproduce the shared-distribution consumption exactly. */
    auto ref = anneal::make_stream(kSeed, anneal::SALT_CHAIN, 3);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (std::size_t i = 0; i < dim; ++i) (void)nd(ref);
    require_same_stream(c.rng, ref);

    /* A per-dimension distribution would NOT match (it draws a fresh pair
     * every time), which is what makes the check above meaningful. */
    auto per_dim = anneal::make_stream(kSeed, anneal::SALT_CHAIN, 3);
    for (std::size_t i = 0; i < dim; ++i) {
        std::normal_distribution<double> fresh(0.0, 1.0);
        (void)fresh(per_dim);
    }
    CHECK(per_dim() != c.rng());
}

TEST_CASE("seed_replica: PA and PT initialise identically",
          "[HPO][anneal_core][seed_replica]")
{
    /* PA used to compare `kb.init_mode == 0` against a raw literal while PT
     * used the named enum. Same kernel now, so UNIFORM must mean UNIFORM in
     * both -- previously an enum reorder would have silently split them. */
    constexpr std::uint64_t kSeed = 4242u;
    const std::size_t dim = 3;
    Vec x0(static_cast<long>(dim));
    for (std::size_t i = 0; i < dim; ++i) x0(static_cast<long>(i)) = 0.1;

    for (const char* mode : {"JITTER", "UNIFORM"}) {
        INFO("INIT_MODE=" << mode);
        auto pt = spec(dim, anneal::AlgoKind::PT);
        auto pa = spec(dim, anneal::AlgoKind::PA);
        pt.param_raw["INIT_MODE"] = mode;
        pa.param_raw["INIT_MODE"] = mode;
        const auto kpt = anneal::Knobs::parse(pt, anneal::AlgoKind::PT);
        const auto kpa = anneal::Knobs::parse(pa, anneal::AlgoKind::PA);

        anneal::ChainState<Vec> a, b;
        anneal::seed_replica(a, kSeed, 2, x0,
                             anneal::make_initial_steps(pt, kpt),
                             anneal::make_initial_steps(pt, kpt), pt, kpt);
        anneal::seed_replica(b, kSeed, 2, x0,
                             anneal::make_initial_steps(pa, kpa),
                             anneal::make_initial_steps(pa, kpa), pa, kpa);
        for (long i = 0; i < a.x.size(); ++i) CHECK(a.x(i) == b.x(i));
        require_same_stream(a.rng, b.rng);
    }
}

TEST_CASE("INIT_MODE JITTER <frac>: magnitude decoupled from STEP_FRAC",
          "[HPO][anneal_core][seed_replica][jitter]")
{
    /* The defect this fixes: one number used to set both the production step
     * and how far the warm start was thrown away. */
    const std::size_t dim = 4;
    Vec x0(static_cast<long>(dim));
    for (std::size_t i = 0; i < dim; ++i) x0(static_cast<long>(i)) = 0.0;

    auto sp = spec(dim, anneal::AlgoKind::PT);
    sp.param_raw["INIT_MODE"] = "JITTER";
    sp.param_tokens["INIT_MODE"] = {"JITTER", "0.01"};
    sp.param["STEP_FRAC"] = 0.4;               /* deliberately far apart */
    const auto kb = anneal::Knobs::parse(sp, anneal::AlgoKind::PT);
    REQUIRE(kb.init_jitter_mode == anneal::Knobs::JitterMode::Fixed);
    CHECK(kb.init_jitter_frac == Approx(0.01));

    const auto proposal = anneal::make_initial_steps(sp, kb);
    const auto jitter   = anneal::make_jitter_sigma(sp, kb, proposal);
    /* width is 2 here (bounds -1..1) */
    CHECK(proposal[0] == Approx(0.8));         /* STEP_FRAC * width */
    CHECK(jitter[0]   == Approx(0.02));        /* frac * width      */

    anneal::ChainState<Vec> c;
    anneal::seed_replica(c, 11u, 2, x0, proposal, jitter, sp, kb);
    /* V is the PRODUCTION step, untouched by the jitter magnitude. */
    CHECK(c.V[0] == Approx(0.8));
}

TEST_CASE("INIT_MODE JITTER <frac>: sigma is a hard per-coordinate bound",
          "[HPO][anneal_core][seed_replica][jitter]")
{
    /* Why the uniform draw: the magnitude is a BOUND. A Gaussian with the
     * same sigma leaves ~32% of coordinates outside it, so the guarantee the
     * user was given would not hold. Checked over many replicas x dims. */
    const std::size_t dim = 8;
    Vec x0(static_cast<long>(dim));
    for (std::size_t i = 0; i < dim; ++i) x0(static_cast<long>(i)) = 0.0;

    auto sp = spec(dim, anneal::AlgoKind::PT);
    sp.param_raw["INIT_MODE"] = "JITTER";
    sp.param_tokens["INIT_MODE"] = {"JITTER", "0.05"};
    const auto kb = anneal::Knobs::parse(sp, anneal::AlgoKind::PT);
    const auto proposal = anneal::make_initial_steps(sp, kb);
    const auto jitter   = anneal::make_jitter_sigma(sp, kb, proposal);

    std::size_t checked = 0;
    for (std::size_t k = 1; k <= 40; ++k) {
        anneal::ChainState<Vec> c;
        anneal::seed_replica(c, 2026u, k, x0, proposal, jitter, sp, kb);
        for (std::size_t i = 0; i < dim; ++i) {
            const double d = std::fabs(c.x(static_cast<long>(i)) -
                                       x0(static_cast<long>(i)));
            CHECK(d <= jitter[i] * (1.0 + 1e-12));
            ++checked;
        }
    }
    CHECK(checked == 40u * dim);
}

TEST_CASE("INIT_MODE extras: every malformed form is a hard error",
          "[HPO][anneal_core][seed_replica][jitter]")
{
    auto make = [](std::vector<std::string> toks) {
        auto sp = spec(3, anneal::AlgoKind::PT);
        sp.param_raw["INIT_MODE"] = toks.front();
        sp.param_tokens["INIT_MODE"] = std::move(toks);
        return sp;
    };
    using anneal::Knobs;
    const auto PT = anneal::AlgoKind::PT;

    /* a magnitude on UNIFORM is meaningless */
    REQUIRE_THROWS_WITH(Knobs::parse(make({"UNIFORM", "0.01"}), PT),
                        Catch::Contains("applies to JITTER only"));
    /* AUTO without its tolerance */
    REQUIRE_THROWS_WITH(Knobs::parse(make({"JITTER", "AUTO"}), PT),
                        Catch::Contains("needs a relative loss tolerance"));
    /* non-numeric tolerance / fraction */
    REQUIRE_THROWS_WITH(Knobs::parse(make({"JITTER", "AUTO", "wide"}), PT),
                        Catch::Contains("must be a finite number"));
    REQUIRE_THROWS_WITH(Knobs::parse(make({"JITTER", "AUTO", "0"}), PT),
                        Catch::Contains("tolerance must be > 0"));
    /* fraction out of range: it is a fraction of the OPTIM interval */
    for (const char* bad : {"0", "-0.1", "1.5"})
        REQUIRE_THROWS_WITH(Knobs::parse(make({"JITTER", bad}), PT),
                            Catch::Contains("must be in (0,1]"));
    /* serial ANNEAL runs one chain, so there is nothing to spread */
    auto ser = spec(3, anneal::AlgoKind::Serial);
    ser.algo = "ANNEAL";
    ser.param_tokens["INIT_MODE"] = {"JITTER", "0.01"};
    REQUIRE_THROWS_WITH(Knobs::parse(ser, anneal::AlgoKind::Serial),
                        Catch::Contains("single chain"));
}

TEST_CASE("INIT_MODE JITTER with no extras is bit-identical to pre-1.5.0",
          "[HPO][anneal_core][seed_replica][jitter]")
{
    /* The compatibility guarantee: absent a magnitude, STEP mode keeps the
     * Gaussian draw off the same stream, so existing SEEDs reproduce. */
    const std::size_t dim = 5;
    Vec x0(static_cast<long>(dim));
    for (std::size_t i = 0; i < dim; ++i) x0(static_cast<long>(i)) = 0.3;

    auto sp = spec(dim, anneal::AlgoKind::PT);
    sp.param_raw["INIT_MODE"] = "JITTER";
    const auto kb = anneal::Knobs::parse(sp, anneal::AlgoKind::PT);
    REQUIRE(kb.init_jitter_mode == anneal::Knobs::JitterMode::Step);
    const auto scale = anneal::make_initial_steps(sp, kb);
    CHECK(anneal::make_jitter_sigma(sp, kb, scale) == scale);

    anneal::ChainState<Vec> got, want;
    anneal::seed_replica(got, 5u, 3, x0, scale, scale, sp, kb);
    reference_seed_replica(want, 5u, 3, x0, scale, sp, kb);
    for (long i = 0; i < got.x.size(); ++i) CHECK(got.x(i) == want.x(i));
    require_same_stream(got.rng, want.rng);
}
