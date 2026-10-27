/*
 * Seed anchoring through the optimiser adapters.
 *
 * run_inner() hands every adapter a copy of its spec whose search space is
 * anchored on the spec's start points (anchor_seed_points). Pinned here:
 * the FIRST evaluation each adapter makes receives exactly the seed's
 * doubles -- the adapters map the start with sp.in() and evaluate
 * sp.out(u0) on the untouched u0 -- while the unanchored spec still sees
 * the plain map's rounded values, so the check is not vacuous. The
 * adapters are driven through their HostT-generic implementations with a
 * stub host; the HPO_Host-bound ones (DLIB, TADAH RANDOM/GRID) use the
 * same sp.in()/sp.out() pair, and the e2e cases in
 * test_hpo_fixweight_e2e.cpp run run_inner() and `tadah hpo` themselves.
 */

#include "catch2/catch.hpp"

#include <tadah/hpo/optimizer_caps.h>   // OptimizerCaps::reads_seed
#include <tadah/hpo/algorithms.h>
#include <tadah/hpo/algorithms_asa.h>
#include <tadah/hpo/algorithms_pa.h>
#include <tadah/hpo/algorithms_pt.h>
#include <tadah/hpo/ceres_adapter.h>
#include <tadah/hpo/gradient_provider.h>
#include <tadah/hpo/nlopt_adapter.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/search_space.h>

#include <dlib/matrix.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

using Vec = dlib::matrix<double, 0, 1>;
using namespace tadah::hpo;

namespace {

bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }

/* The coordinate system of a space: dimensions, policy, version and, per
 * dimension, its scale, real box and internal box. Seed anchors are not
 * part of it. */
bool same_coordinates(const tadah::hpo::SearchSpace& a,
                      const tadah::hpo::SearchSpace& b)
{
    if (a.dim() != b.dim() || a.normalised() != b.normalised() ||
        a.version() != b.version())
        return false;
    for (std::size_t i = 0; i < a.dim(); ++i)
        if (a.describe_dim(i) != b.describe_dim(i) ||
            !same_bits(a.at(i).lo, b.at(i).lo) ||
            !same_bits(a.at(i).hi, b.at(i).hi) ||
            !same_bits(a.internal_lo(i), b.internal_lo(i)) ||
            !same_bits(a.internal_hi(i), b.internal_hi(i)))
            return false;
    return true;
}

struct NoRunner {};   // GradientProvider's Runner is only a type tag

/* Seeds whose plain x -> u -> x round trip under OPTIM_SCALE AUTO is not
 * bit-exact: a small weight in an explicit wide box, a grid centre in an
 * AUTO FRAC 0 FLOOR 0.5 box, and a LOG box. */
const double kC0 = 0.100000000000000006;
const std::vector<double> kSeed = {-0.000287486722631648, kC0, 1.2};

SearchSpacePtr seed_space()
{
    auto dim = [](const char* label, double lo, double hi, ScaleKind k) {
        DimSpec d;
        d.key = label; d.label = label; d.lo = lo; d.hi = hi; d.tf.kind = k;
        return d;
    };
    ScalePolicy pol; pol.normalise = true;
    return std::make_shared<const SearchSpace>(SearchSpace::build(
        {dim("FIXWEIGHT[13]", -300.0, 300.0, ScaleKind::Linear),
         dim("CGRID2B[1]", kC0 - 0.5, kC0 + 0.5, ScaleKind::Linear),
         dim("SGRID2B[1]", 0.5, 2.0, ScaleKind::Log)}, pol));
}

OptimizerSpec seed_spec(const std::string& lib, const std::string& algo)
{
    OptimizerSpec sp;
    sp.lib = lib;
    sp.algo = algo;
    sp.scale.normalise = true;
    sp.bind_space(seed_space());
    sp.optim_init = {kSeed};
    sp.stop.maxeval = 40;
    /* SEED for an optimiser that reads it (one that draws random
     * numbers, OptimizerCaps::reads_seed): the others refuse it. */
    const auto* caps = tadah::hpo::find_optimizer_caps(sp.lib, sp.algo);
    if (caps && caps->reads_seed) sp.seed = 7;
    sp.num_threads = 1;
    return sp;
}

/* Records every REAL point it is asked to evaluate; a smooth bowl whose
 * minimum is away from the seed, so every algorithm moves. */
struct RecordingHost {
    mutable std::mutex m;
    mutable std::vector<std::vector<double>> xs;
    double operator()(const Vec& x) const
    {
        std::vector<double> v(static_cast<std::size_t>(x.size()));
        double s = 0.0;
        for (long i = 0; i < x.size(); ++i) {
            v[static_cast<std::size_t>(i)] = x(i);
            const double d = x(i) - 0.3 * static_cast<double>(i + 1);
            s += d * d;
        }
        std::lock_guard<std::mutex> lg(m);
        xs.push_back(std::move(v));
        return 1.0 + s;
    }
};

using Runner = std::function<void(const OptimizerSpec&, RecordingHost&)>;

std::vector<std::vector<double>> evals(const OptimizerSpec& sp, const Runner& run)
{
    RecordingHost host;
    run(sp, host);
    REQUIRE_FALSE(host.xs.empty());
    return host.xs;
}

std::vector<double> first_eval(const OptimizerSpec& sp, const Runner& run)
{
    return evals(sp, run).front();
}

void check_first_eval_is_seed(const OptimizerSpec& plain, const Runner& run)
{
    const std::optional<OptimizerSpec> anchored = anchor_seed_points(plain);
    REQUIRE(anchored.has_value());
    const std::vector<double> x = first_eval(*anchored, run);
    REQUIRE(x.size() == kSeed.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        CAPTURE(i, x[i], kSeed[i]);
        CHECK(same_bits(x[i], kSeed[i]));
    }
}

/* Position of the first evaluation equal to the seed bit for bit, or
 * npos. */
std::size_t seed_position(const std::vector<std::vector<double>>& xs)
{
    for (std::size_t k = 0; k < xs.size(); ++k) {
        bool eq = xs[k].size() == kSeed.size();
        for (std::size_t i = 0; eq && i < kSeed.size(); ++i)
            eq = same_bits(xs[k][i], kSeed[i]);
        if (eq) return k;
    }
    return std::string::npos;
}

}  // namespace

TEST_CASE("anchor_seed_points: an anchored copy whenever a seed lies in the box",
          "[HPO][seed_anchor]")
{
    const OptimizerSpec sp = seed_spec("NLOPT", "LN_SBPLX");
    /* every seed coordinate is inexact under the plain map */
    for (std::size_t i = 0; i < kSeed.size(); ++i)
        REQUIRE_FALSE(same_bits(sp.space->to_real(i, sp.space->to_internal(i, kSeed[i])),
                                kSeed[i]));

    const auto a = anchor_seed_points(sp);
    REQUIRE(a.has_value());
    CHECK(a->space != sp.space);
    CHECK(a->space->anchor_count() == kSeed.size());
    CHECK(sp.space->anchor_count() == 0u);            // the shared space is untouched
    CHECK(a->lib == sp.lib);
    CHECK(a->algo == sp.algo);
    CHECK(a->optim_init == sp.optim_init);
    CHECK(a->low == sp.low);
    CHECK(a->high == sp.high);
    CHECK(same_coordinates(*a->space, *sp.space));
    const auto back = a->out(a->in(kSeed));
    for (std::size_t i = 0; i < kSeed.size(); ++i) CHECK(same_bits(back[i], kSeed[i]));

    /* A seed the plain map already returns is anchored too: an optimiser
     * that rounds its start (BOBYQA, COBYLA, PRAXIS) evaluates a u next to
     * u0, which only the anchor window maps back to the seed. */
    OptimizerSpec exact = sp;
    exact.optim_init = {{0.0, kC0 - 0.5, 2.0}};       // box points map exactly
    const auto ae = anchor_seed_points(exact);
    REQUIRE(ae.has_value());
    CHECK(ae->space->anchor_count() == 3u);

    /* No space, no seed, or no seed coordinate inside its box: no copy. */
    OptimizerSpec bare;
    bare.optim_init = {kSeed};
    CHECK_FALSE(anchor_seed_points(bare).has_value());
    OptimizerSpec none = sp;
    none.optim_init.clear();
    CHECK_FALSE(anchor_seed_points(none).has_value());
    OptimizerSpec outside = sp;
    outside.optim_init = {{301.0, kC0 + 1.0, 3.0}};
    CHECK_FALSE(anchor_seed_points(outside).has_value());

    /* Idempotent: anchoring an anchored spec reproduces its anchors. */
    const auto again = anchor_seed_points(*a);
    REQUIRE(again.has_value());
    CHECK(again->space->anchor_count() == a->space->anchor_count());
}

TEST_CASE("seed anchors: every adapter's first evaluation is the seed bitwise",
          "[HPO][seed_anchor]")
{
    SECTION("unanchored control: the plain map rounds the seed") {
        const std::vector<double> x = first_eval(
            seed_spec("NLOPT", "LN_SBPLX"), [](const OptimizerSpec& s, RecordingHost& h) {
                run_nlopt_impl<Vec, NoRunner>(s, h);
            });
        std::size_t differ = 0;
        for (std::size_t i = 0; i < x.size(); ++i)
            if (!same_bits(x[i], kSeed[i])) ++differ;
        CHECK(differ == kSeed.size());
    }
    /* LN_NEWUOA_BOUND, which was in this list, is refused (the NLopt
     * adapter's caps lookup throws). Before the fix: passes as well. */
    for (const char* algo : {"LN_SBPLX", "LN_NELDERMEAD", "LN_BOBYQA", "LN_COBYLA",
                             "LN_PRAXIS"}) {
        DYNAMIC_SECTION("NLOPT:" << algo) {
            check_first_eval_is_seed(seed_spec("NLOPT", algo),
                [](const OptimizerSpec& s, RecordingHost& h) {
                    run_nlopt_impl<Vec, NoRunner>(s, h);
                });
        }
    }
    SECTION("TADAH:ANNEAL") {
        check_first_eval_is_seed(seed_spec("TADAH", "ANNEAL"),
            [](const OptimizerSpec& s, RecordingHost& h) { run_anneal_impl<Vec>(s, h); });
    }
    SECTION("TADAH:ANNEAL_PT") {
        OptimizerSpec sp = seed_spec("TADAH", "ANNEAL_PT");
        sp.stop.maxeval = 400;
        sp.param["REPLICAS"] = 4.0;
        check_first_eval_is_seed(sp,
            [](const OptimizerSpec& s, RecordingHost& h) { run_anneal_pt_impl<Vec>(s, h); });
    }
    SECTION("TADAH:ANNEAL_PA") {
        OptimizerSpec sp = seed_spec("TADAH", "ANNEAL_PA");
        sp.stop.maxeval = 400;
        check_first_eval_is_seed(sp,
            [](const OptimizerSpec& s, RecordingHost& h) { run_anneal_pa_impl<Vec>(s, h); });
    }
    SECTION("ASA:ASA") {
        /* ASA first draws Number_Cost_Samples (default 5) random states for
         * its initial cost temperature, then evaluates the user's initial
         * parameters. */
        const Runner asa = [](const OptimizerSpec& s, RecordingHost& h) {
            run_asa_impl<Vec>(s, h);
        };
        const OptimizerSpec plain = seed_spec("ASA", "ASA");
        CHECK(seed_position(evals(plain, asa)) == std::string::npos);
        const auto anchored = anchor_seed_points(plain);
        REQUIRE(anchored.has_value());
        CHECK(seed_position(evals(*anchored, asa)) == 5u);
    }
#ifdef TADAH_USE_CERES
    SECTION("CERES:STEEPEST_DESCENT") {
        /* a pin (the section ran CERES:LBFGS, refused since) */
        check_first_eval_is_seed(seed_spec("CERES", "STEEPEST_DESCENT"),
            [](const OptimizerSpec& s, RecordingHost& h) {
                tadah::hpo::detail::run_ceres_gradient_problem<Vec, NoRunner>(
                    s, h, ceres::STEEPEST_DESCENT, ceres::ARMIJO,
                    "STEEPEST_DESCENT");
            });
    }
#endif
}

/* A pinned-weight seed searched as OPTIM FIXWEIGHT (2-11) in explicit
 * [-300, 300] boxes. With no STEP, NLopt's default initial steps differ
 * per dimension, so BOBYQA
 * and COBYLA rescale the start (u -> (u/s)*s) and PRAXIS maps it through
 * atanh/tanh: their first evaluation is a few ulp off u0 in some
 * dimension (FIXWEIGHT(8) for BOBYQA/COBYLA, FIXWEIGHT(2) -- whose plain
 * round trip is exact -- for PRAXIS). The anchor window still evaluates
 * the seed bit for bit. */
TEST_CASE("seed anchors: NLopt BOBYQA/COBYLA/PRAXIS evaluate a rounded start "
          "at the seed", "[HPO][seed_anchor]")
{
    const std::vector<double> seed = {
        -216.764741590923819, 108.326725555511771, -43.3325794917411713,
        8.56326925583498166,  11.7937235812636061, -8.12615760784936469,
        -26.560116371452601,  -27.7050793839341019, -16.8854852768916643,
        -5.55896250563770522};
    std::vector<DimSpec> dims;
    for (std::size_t i = 0; i < seed.size(); ++i) {
        DimSpec d;
        d.key = "FIXWEIGHT";
        d.label = "FIXWEIGHT[" + std::to_string(i + 2) + "]";
        d.lo = -300.0;
        d.hi = 300.0;
        dims.push_back(d);
    }
    ScalePolicy pol; pol.normalise = true;
    auto space = std::make_shared<const SearchSpace>(SearchSpace::build(dims, pol));
    const Runner nlopt = [](const OptimizerSpec& s, RecordingHost& h) {
        run_nlopt_impl<Vec, NoRunner>(s, h);
    };
    for (const char* algo : {"LN_BOBYQA", "LN_COBYLA", "LN_PRAXIS"}) {
        DYNAMIC_SECTION("NLOPT:" << algo) {
            OptimizerSpec sp;
            sp.lib = "NLOPT";
            sp.algo = algo;
            sp.scale.normalise = true;
            sp.bind_space(space);
            sp.optim_init = {seed};
            sp.stop.maxeval = 3;
            sp.num_threads = 1;

            /* control: without anchors the first evaluation is the plain
             * map of a u that is NOT u0 in some dimension */
            const std::vector<double> x_plain = first_eval(sp, nlopt);
            std::size_t perturbed = 0;
            for (std::size_t i = 0; i < seed.size(); ++i)
                if (!same_bits(x_plain[i],
                               space->to_real(i, space->to_internal(i, seed[i]))))
                    ++perturbed;
            CHECK(perturbed > 0u);

            const auto anchored = anchor_seed_points(sp);
            REQUIRE(anchored.has_value());
            const std::vector<double> x = first_eval(*anchored, nlopt);
            REQUIRE(x.size() == seed.size());
            for (std::size_t i = 0; i < seed.size(); ++i) {
                CAPTURE(i, x[i], seed[i]);
                CHECK(same_bits(x[i], seed[i]));
            }
        }
    }
}

/* The window is absolute, 4 ulp of the internal box. Under
 * OPTIM_SCALE NONE a LIN dimension's map is the identity, and with
 * max(|lo|, |hi|) >= 2^24 four ulp of the box reach the Ceres kernel's
 * smallest step sqrt(eps) = 2^-26: both points of the central difference
 * mapped to a small seed and the gradient component came out 0 (the
 * dimension then stays at its seed for the whole local run). A small
 * seed's window is capped at 2^-27, so every finite-difference kernel sees
 * the true slope (with the default fd_step). */
TEST_CASE("seed anchors: the window never swallows a finite-difference step",
          "[HPO][seed_anchor]")
{
    const double inf = std::numeric_limits<double>::infinity();
    const double x0 = 1e-3;
    struct Bowl {   // (x - 7)^2 in REAL space: slope 2*(x0 - 7) at the seed
        double operator()(const Vec& x) const { return (x(0) - 7.0) * (x(0) - 7.0); }
    };
    for (double hi : {1e7, 2e7, 5e7}) {
        CAPTURE(hi);
        DimSpec d;
        d.key = d.label = "FIXWEIGHT[1]";
        d.lo = 0.0;
        d.hi = hi;
        ScalePolicy pol;
        pol.normalise = false;                     // OPTIM_SCALE NONE
        OptimizerSpec sp;
        sp.lib = "DLIB";
        sp.algo = "BFGS";
        sp.bind_space(std::make_shared<const SearchSpace>(SearchSpace::build({d}, pol)));
        sp.optim_init = {{x0}};
        const auto a = anchor_seed_points(sp);
        REQUIRE(a.has_value());

        const double ulp_box = std::nextafter(hi, inf) - hi;
        const double tol = a->space->seed_anchor_tolerance(0, x0);   // NONE: u0 = x0
        CHECK(tol == std::min(4 * ulp_box, kSeedAnchorMaxTol));
        CHECK(tol < std::sqrt(std::numeric_limits<double>::epsilon()));
        /* the seed itself, and a start rounded by up to the window */
        CHECK(same_bits(a->space->to_real(0, x0), x0));
        CHECK(same_bits(a->space->to_real(0, x0 + 0.5 * tol), x0));
        /* the Ceres kernel's smallest step is outside the window */
        const double step = std::sqrt(std::numeric_limits<double>::epsilon());
        CHECK(a->space->to_real(0, x0 + step) == x0 + step);
        CHECK(a->space->to_real(0, x0 - step) == x0 - step);

        /* NUMERICAL gradient at the seed with each kernel (Ceres is the
         * default under NONE) */
        for (const char* kernel : {"ceres", "inhouse"}) {
            CAPTURE(kernel);
            OptimizerSpec k = *a;
            k.param_raw["fd_kernel"] = kernel;
            Bowl host;
            std::atomic<unsigned long> counter{0};
            auto gp = make_gradient_provider<Vec, NoRunner>(k, host, &counter, 0);
            Vec x;
            x.set_size(1);
            x(0) = x0;
            const Vec u = k.in(x);
            Vec g;
            REQUIRE(gp.gradient(u, g));
            REQUIRE(g.size() == 1);
            CHECK(g(0) == Approx(2.0 * (x0 - 7.0)).epsilon(1e-6));
        }
    }
}

/* The 2^-27 cap applied to every anchor, so under OPTIM_SCALE NONE a
 * seed with |x0| >= 2^26 in dimension 1 or later -- where NLopt BOBYQA and
 * COBYLA rescale the start by up to 1 ulp(x0) -- was evaluated 1 ulp off.
 * Each anchor's window is now at least 2 ulp of its own u0. 48 large seeds
 * in [0, 2^30] behind a unit dimension: without anchors some first
 * evaluations are off the seed; with them none is. */
TEST_CASE("seed anchors: NLopt BOBYQA/COBYLA evaluate a large NONE seed exactly",
          "[HPO][seed_anchor]")
{
    auto dim = [](const char* label, double lo, double hi) {
        DimSpec d;
        d.key = label; d.label = label; d.lo = lo; d.hi = hi;
        return d;
    };
    ScalePolicy pol; pol.normalise = false;          // OPTIM_SCALE NONE
    auto space = std::make_shared<const SearchSpace>(SearchSpace::build(
        {dim("FIXWEIGHT[1]", 0.0, 1.0), dim("FIXWEIGHT[2]", 0.0, 0x1p30)}, pol));
    const Runner nlopt = [](const OptimizerSpec& s, RecordingHost& h) {
        run_nlopt_impl<Vec, NoRunner>(s, h);
    };
    for (const char* algo : {"LN_BOBYQA", "LN_COBYLA"}) {
        DYNAMIC_SECTION("NLOPT:" << algo) {
            std::size_t off_plain = 0, off_anchored = 0;
            for (int k = 0; k < 48; ++k) {
                /* 84139161.677173063 and 47 more in
                 * [2^26, 2^29), irrationally spaced */
                const double x1 = k == 0 ? 84139161.677173063
                                         : std::ldexp(1.0 + std::fmod(0.6180339887498949 * k, 1.0),
                                                      26 + k % 3) + 0.3183098861837907 * k;
                const std::vector<double> seed = {0.3, x1};
                OptimizerSpec sp;
                sp.lib = "NLOPT";
                sp.algo = algo;
                sp.scale.normalise = false;
                sp.bind_space(space);
                sp.optim_init = {seed};
                sp.stop.maxeval = 1;
                sp.num_threads = 1;
                const std::vector<double> xp = first_eval(sp, nlopt);
                if (!same_bits(xp[1], x1)) ++off_plain;
                const auto anchored = anchor_seed_points(sp);
                REQUIRE(anchored.has_value());
                const std::vector<double> x = first_eval(*anchored, nlopt);
                REQUIRE(x.size() == 2u);
                CAPTURE(k, x1, x[1]);
                CHECK(same_bits(x[0], seed[0]));
                if (!same_bits(x[1], x1)) ++off_anchored;
            }
            CHECK(off_plain > 0u);       // the window is not vacuous here
            CHECK(off_anchored == 0u);
        }
    }
}
