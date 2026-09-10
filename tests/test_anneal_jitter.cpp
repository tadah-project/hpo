// test_anneal_jitter.cpp
//
// The INIT_MODE JITTER AUTO calibration kernel, tested against closed-form
// objectives with no host, no pool and no MPI -- the reason anneal_jitter.h
// takes a probe callable instead of an HPO_Host.
//
// The oracle is exact rather than asymptotic. For L(x) = sum_i d_i x_i^2 with
// every coordinate seeded at c, displacing dimension j by s gives
//     dL = d_j * ((c+s)^2 - c^2) = d_j * (2cs + s^2)
// so the band edge dL = p*L0 is the positive root of
//     d_j s^2 + 2 c d_j s - p L0 = 0   =>   s* = -c + sqrt(c^2 + p L0 / d_j).
// (The often-quoted s* ~ 1/sqrt(d_j) is only the s >> c limit of that; using
// the limit as the expectation would make the test wrong in the other
// regime, so the exact root is used throughout.)
//
// sigma is the largest displacement actually MEASURED in band, so it is
// bounded above by s* and below by the search's resolution -- the assertions
// are one-sided on top and resolution-aware below, never "approximately s*".

#include "catch2/catch.hpp"

#include <tadah/hpo/anneal_jitter.h>
#include <tadah/hpo/optimizer_spec.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

using namespace tadah::hpo;

namespace {

OptimizerSpec box(std::size_t dim, double lo, double hi)
{
    OptimizerSpec sp;
    sp.low.assign(dim, lo);
    sp.high.assign(dim, hi);
    sp.is_int.assign(dim, false);
    sp.log_scaled.assign(dim, false);
    return sp;
}

/* Exact band edge for the separable quadratic described above. */
double band_edge(double d_j, double c, double p, double L0)
{
    return -c + std::sqrt(c * c + p * L0 / d_j);
}

}  // namespace

TEST_CASE("calibrate_jitter recovers the analytic band edge over 4 decades",
          "[HPO][anneal_jitter]")
{
    const std::vector<double> d{1.0, 10.0, 100.0, 1000.0};
    const std::size_t dim = d.size();
    const double c = 0.1;                       /* seed, same in every dim */
    auto sp = box(dim, -1.0, 1.0);

    std::vector<double> x0(dim, c);
    double L0 = 0.0;
    for (std::size_t i = 0; i < dim; ++i) L0 += d[i] * c * c;

    auto loss_with = [&](std::size_t j, double v) {
        double s = 0.0;
        for (std::size_t i = 0; i < dim; ++i)
            s += d[i] * (i == j ? v * v : c * c);
        return s;
    };

    for (double p : {0.01, 0.05, 0.2}) {
        anneal::JitterOptions opt;
        opt.tol = p;
        opt.max_probes = 16;                    /* 8 per side */

        const auto plan = anneal::calibrate_jitter(
            sp, x0, L0, opt, std::vector<double>(dim, 0.0), loss_with);

        REQUIRE_FALSE(plan.abandoned);
        CHECK(plan.threshold == Approx(L0 + p * L0));

        for (std::size_t j = 0; j < dim; ++j) {
            const double want = band_edge(d[j], c, p, L0);
            INFO("p=" << p << " dim=" << j << " d=" << d[j]
                      << " sigma=" << plan.sigma[j] << " edge=" << want);
            /* Never claims more than it measured... */
            CHECK(plan.sigma[j] <= want * (1.0 + 1e-9));
            /* ...and gets within one refinement rung of it. */
            CHECK(plan.sigma[j] >= want / 1.6);
        }
        /* Stiffer dimensions must tolerate less: the whole point. */
        for (std::size_t j = 1; j < dim; ++j)
            CHECK(plan.sigma[j] < plan.sigma[j - 1]);
    }
}

TEST_CASE("calibrate_jitter: sigma is a measured bound, never interpolated",
          "[HPO][anneal_jitter]")
{
    /* Every returned sigma must itself pass the band test when replayed. */
    const std::vector<double> d{1.0, 250.0, 7.0};
    const std::size_t dim = d.size();
    const double c = 0.2;
    auto sp = box(dim, -1.0, 1.0);
    std::vector<double> x0(dim, c);
    double L0 = 0.0;
    for (std::size_t i = 0; i < dim; ++i) L0 += d[i] * c * c;

    auto loss_with = [&](std::size_t j, double v) {
        double s = 0.0;
        for (std::size_t i = 0; i < dim; ++i)
            s += d[i] * (i == j ? v * v : c * c);
        return s;
    };

    anneal::JitterOptions opt;
    opt.tol = 0.05;
    const auto plan = anneal::calibrate_jitter(
        sp, x0, L0, opt, std::vector<double>(dim, 0.0), loss_with);

    for (std::size_t j = 0; j < dim; ++j) {
        for (double sign : {+1.0, -1.0}) {
            const double f = loss_with(j, x0[j] + sign * plan.sigma[j]);
            INFO("dim " << j << " sign " << sign);
            CHECK(f <= plan.threshold * (1.0 + 1e-9));
        }
    }
}

TEST_CASE("calibrate_jitter: an insensitive dimension is capped, not maximal",
          "[HPO][anneal_jitter]")
{
    /* A coordinate the objective ignores stays in band at every scale. It
     * gets the cap and says so, rather than silently claiming the whole box. */
    auto sp = box(2, -1.0, 1.0);
    std::vector<double> x0{0.0, 0.0};
    const double L0 = 1.0;

    auto loss_with = [&](std::size_t j, double v) {
        return j == 0 ? 1.0 + 100.0 * v * v : 1.0;   /* dim 1 is inert */
    };

    anneal::JitterOptions opt;
    opt.tol = 0.05;
    const auto plan = anneal::calibrate_jitter(
        sp, x0, L0, opt, std::vector<double>(2, 0.0), loss_with);

    CHECK((plan.dims[1].status & anneal::JitterInsensitive) != 0u);
    CHECK((plan.dims[1].status & anneal::JitterSupportCapped) != 0u);
    CHECK(plan.sigma[1] == Approx(opt.max_frac * 2.0));   /* cap, not width */
    /* One screen probe per side: the two-sign screen resolves an inert
     * coordinate immediately, without any refinement. */
    CHECK(plan.dims[1].probes == 2u);
    /* The sensitive dimension is not capped. */
    CHECK((plan.dims[0].status & anneal::JitterInsensitive) == 0u);
    CHECK(plan.sigma[0] < plan.sigma[1]);
}

TEST_CASE("calibrate_jitter: a seed on a bound is probed one-sided",
          "[HPO][anneal_jitter]")
{
    /* The recurring pattern in this project: dims pinned at a wall. Probing
     * outward would leave the box, so only the side with room is measured. */
    auto sp = box(1, 0.0, 1.0);
    std::vector<double> x0{1.0};             /* exactly on the upper bound */
    const double L0 = 1.0;
    auto loss_with = [&](std::size_t, double v) { return 1.0 + (1.0 - v); };

    anneal::JitterOptions opt;
    opt.tol = 0.05;
    const auto plan = anneal::calibrate_jitter(
        sp, x0, L0, opt, std::vector<double>(1, 0.0), loss_with);

    CHECK((plan.dims[0].status & anneal::JitterOneSided) != 0u);
    CHECK(plan.sigma[0] > 0.0);
    CHECK(x0[0] - plan.sigma[0] >= sp.low[0]);   /* stayed inside the box */
}

TEST_CASE("calibrate_jitter: an unusable anchor abandons instead of guessing",
          "[HPO][anneal_jitter]")
{
    auto sp = box(3, -1.0, 1.0);
    sp.failscore = 1e9;
    std::vector<double> x0(3, 0.0);
    const std::vector<double> fallback{0.1, 0.2, 0.3};

    std::size_t calls = 0;
    auto counting = [&](std::size_t, double) { ++calls; return 1.0; };

    anneal::JitterOptions opt;
    opt.tol = 0.05;

    /* A FAILSCORE seed gives a relative tolerance nothing to be relative to.
     * Note 1e9 * 1.05 would still "pass" a naive band test at every scale --
     * which is exactly how this would silently hand back a half-box jitter. */
    for (double bad : {1.0e9, 0.0}) {
        calls = 0;
        const auto plan = anneal::calibrate_jitter(sp, x0, bad, opt,
                                                   fallback, counting);
        INFO("L0 = " << bad);
        CHECK(plan.abandoned);
        CHECK(calls == 0u);                  /* spends nothing */
        CHECK(plan.sigma == fallback);
        for (const auto& d : plan.dims)
            CHECK((d.status & anneal::JitterNoAnchor) != 0u);
    }
}

TEST_CASE("calibrate_jitter: a cliff is out of band and is flagged",
          "[HPO][anneal_jitter]")
{
    auto sp = box(1, -1.0, 1.0);
    sp.failscore = 1e9;
    std::vector<double> x0{0.0};
    const double L0 = 1.0;

    /* Feasible near the seed, a hard cliff beyond 0.05. */
    auto loss_with = [&](std::size_t, double v) {
        return std::fabs(v) > 0.05 ? 1e9 : 1.0;
    };

    anneal::JitterOptions opt;
    opt.tol = 0.05;
    opt.max_probes = 12;
    const auto plan = anneal::calibrate_jitter(
        sp, x0, L0, opt, std::vector<double>(1, 0.0), loss_with);

    CHECK((plan.dims[0].status & anneal::JitterCliff) != 0u);
    CHECK(plan.sigma[0] <= 0.05);       /* never steps over the cliff */
    CHECK(plan.sigma[0] > 0.0);
}

TEST_CASE("calibrate_jitter: an improving direction is not punished",
          "[HPO][anneal_jitter]")
{
    /* The band is one-sided in the WORSE direction: shrinking sigma because
     * a coordinate improved the loss would penalise the directions most
     * worth exploring. */
    auto sp = box(1, -1.0, 1.0);
    std::vector<double> x0{0.0};
    const double L0 = 1.0;
    auto downhill = [&](std::size_t, double v) { return 1.0 - 0.5 * v * v; };

    anneal::JitterOptions opt;
    opt.tol = 0.05;
    const auto plan = anneal::calibrate_jitter(
        sp, x0, L0, opt, std::vector<double>(1, 0.0), downhill);

    CHECK(plan.sigma[0] == Approx(opt.max_frac * 2.0));
    CHECK((plan.dims[0].status & anneal::JitterInsensitive) != 0u);
}

TEST_CASE("calibrate_jitter: probes are bounded and the result is RNG-free",
          "[HPO][anneal_jitter]")
{
    const std::size_t dim = 20;
    auto sp = box(dim, -1.0, 1.0);
    std::vector<double> x0(dim, 0.1);
    double L0 = 0.0;
    for (std::size_t i = 0; i < dim; ++i) L0 += (i + 1.0) * 0.01;
    auto loss_with = [&](std::size_t j, double v) {
        double s = 0.0;
        for (std::size_t i = 0; i < dim; ++i)
            s += (i + 1.0) * (i == j ? v * v : 0.01);
        return s;
    };

    anneal::JitterOptions opt;
    opt.tol = 0.05;
    opt.max_probes = 6;

    const auto a = anneal::calibrate_jitter(
        sp, x0, L0, opt, std::vector<double>(dim, 0.0), loss_with);
    const auto b = anneal::calibrate_jitter(
        sp, x0, L0, opt, std::vector<double>(dim, 0.0), loss_with);

    CHECK(a.sigma == b.sigma);                    /* deterministic */
    CHECK(a.probes == b.probes);
    /* The bound includes the interior audit, which costs one extra probe on
     * each sampled dimension and is counted in the preflight estimate too. */
    CHECK(a.probes <= dim * opt.max_probes +
                          std::min(opt.audit_sample, dim));
    /* Per dimension the search itself is still capped; the audit probe is
     * charged to the plan, not to the dimension's search budget. */
    for (const auto& d : a.dims) CHECK(d.probes <= opt.max_probes);
}

TEST_CASE("calibrate_jitter: measured probes-per-dimension at realistic scale",
          "[HPO][anneal_jitter][cost]")
{
    /* Cost is the deciding factor for AUTO on a real search space (the Ta
     * Peierls problem is 816 dimensions of PC_LAMMPS relaxations), and every
     * figure in the design was an ESTIMATE. Measure it instead, on a spread
     * of stiffnesses spanning six decades so the number is not flattered by
     * an easy landscape. */
    const std::size_t dim = 120;
    auto sp = box(dim, -1.0, 1.0);
    const double c = 0.1;
    std::vector<double> x0(dim, c);

    std::vector<double> d(dim);
    for (std::size_t i = 0; i < dim; ++i)
        d[i] = std::pow(10.0, static_cast<double>(i % 7) - 1.0);  /* 0.1 .. 1e5 */

    double L0 = 0.0;
    for (std::size_t i = 0; i < dim; ++i) L0 += d[i] * c * c;
    auto loss_with = [&](std::size_t j, double v) {
        double s = 0.0;
        for (std::size_t i = 0; i < dim; ++i)
            s += d[i] * (i == j ? v * v : c * c);
        return s;
    };

    anneal::JitterOptions opt;
    opt.tol = 0.05;
    opt.max_probes = 8;                       /* the shipped default shape */
    const auto plan = anneal::calibrate_jitter(
        sp, x0, L0, opt, std::vector<double>(dim, 0.0), loss_with);

    const double per_dim = static_cast<double>(plan.probes) / dim;
    WARN("measured probes/dim = " << per_dim
         << "  (total " << plan.probes << " for " << dim << " dims)");
    CHECK(per_dim <= static_cast<double>(opt.max_probes));
    /* Every dimension still lands inside its band -- cheapness that gave up
     * accuracy would be worthless. */
    for (std::size_t j = 0; j < dim; ++j)
        CHECK(loss_with(j, x0[j] + plan.sigma[j]) <= plan.threshold * (1 + 1e-9));
}

namespace {

/* A grouped search space, like a real basis: several families whose members
 * share a sensitivity scale, with the families spanning decades. */
struct Grouped {
    std::size_t dim;
    std::vector<double> d;
    std::vector<std::string> labels;
    double c = 0.1;
    double L0 = 0.0;

    explicit Grouped(std::size_t per_family, std::size_t families)
    {
        dim = per_family * families;
        for (std::size_t f = 0; f < families; ++f) {
            const double stiff = std::pow(10.0, static_cast<double>(f));
            for (std::size_t m = 0; m < per_family; ++m) {
                d.push_back(stiff);
                labels.push_back("KEY" + std::to_string(f) + "[" +
                                 std::to_string(m + 1) + "]");
            }
        }
        for (std::size_t i = 0; i < dim; ++i) L0 += d[i] * c * c;
    }

    double loss(std::size_t j, double v) const
    {
        double s = 0.0;
        for (std::size_t i = 0; i < dim; ++i)
            s += d[i] * (i == j ? v * v : c * c);
        return s;
    }
};

}  // namespace

TEST_CASE("INIT_JITTER_SAMPLE: cost falls, every family is still touched",
          "[HPO][anneal_jitter][sample]")
{
    Grouped g(20, 4);                     /* 80 dims, 4 families */
    auto sp = box(g.dim, -1.0, 1.0);
    sp.parameter_labels = g.labels;
    std::vector<double> x0(g.dim, g.c);
    auto probe = [&](std::size_t j, double v) { return g.loss(j, v); };

    anneal::JitterOptions full;
    full.tol = 0.05;
    full.audit_sample = 0;                /* isolate the sampling lever */
    const auto all = anneal::calibrate_jitter(
        sp, x0, g.L0, full, std::vector<double>(g.dim, 0.0), probe);

    anneal::JitterOptions few = full;
    few.sample = 8;                       /* 2 per family */
    const auto sampled = anneal::calibrate_jitter(
        sp, x0, g.L0, few, std::vector<double>(g.dim, 0.0), probe);

    /* The point of the lever. */
    CHECK(sampled.probes < all.probes / 5);

    std::size_t inferred = 0, measured = 0;
    for (const auto& d : sampled.dims)
        (d.status & anneal::JitterInferred) ? ++inferred : ++measured;
    CHECK(measured == 8u);
    CHECK(inferred == g.dim - 8u);

    /* Round-robin across FAMILIES, not the first N coordinates: a prefix
     * sample of a grouped basis would measure one family and infer the rest
     * from a scale that has nothing to do with them. */
    std::vector<bool> family_seen(4, false);
    for (std::size_t i = 0; i < g.dim; ++i)
        if (!(sampled.dims[i].status & anneal::JitterInferred))
            family_seen[i / 20] = true;
    for (bool seen : family_seen) CHECK(seen);

    /* An inferred sigma should land near its family's measured scale --
     * within an order of magnitude, since the families differ by decades. */
    for (std::size_t f = 0; f < 4; ++f) {
        double meas = 0.0;
        for (std::size_t i = f * 20; i < (f + 1) * 20; ++i)
            if (!(sampled.dims[i].status & anneal::JitterInferred))
                meas = std::max(meas, sampled.sigma[i]);
        REQUIRE(meas > 0.0);
        for (std::size_t i = f * 20; i < (f + 1) * 20; ++i) {
            INFO("family " << f << " dim " << i);
            CHECK(sampled.sigma[i] <= meas * 10.0);
            CHECK(sampled.sigma[i] >= meas / 10.0);
        }
    }
}

TEST_CASE("INIT_JITTER_SAMPLE 0 measures everything", "[HPO][anneal_jitter][sample]")
{
    Grouped g(5, 2);
    auto sp = box(g.dim, -1.0, 1.0);
    sp.parameter_labels = g.labels;
    std::vector<double> x0(g.dim, g.c);

    anneal::JitterOptions opt;
    opt.tol = 0.05;
    opt.sample = 0;
    opt.audit_sample = 0;
    const auto plan = anneal::calibrate_jitter(
        sp, x0, g.L0, opt, std::vector<double>(g.dim, 0.0),
        [&](std::size_t j, double v) { return g.loss(j, v); });

    for (const auto& d : plan.dims)
        CHECK((d.status & anneal::JitterInferred) == 0u);
}

TEST_CASE("interior audit catches a loss that is not monotone in displacement",
          "[HPO][anneal_jitter][audit]")
{
    /* sigma is the largest displacement measured IN band, so it certifies the
     * grid points that were tested -- not the interval below them. A well
     * between the grid points is exactly the case where that distinction
     * matters, and the audit is what stops the bound being overclaimed. */
    auto sp = box(1, -1.0, 1.0);
    std::vector<double> x0{0.0};
    const double L0 = 1.0;
    const double thr = 1.05;

    /* In band at the cap and near the seed, but NOT in between -- so the
     * screen accepts the cap as sigma while sigma/2 sits in the well. */
    auto tricky = [&](std::size_t, double v) {
        const double a = std::fabs(v);
        return (a > 0.3 && a < 0.7) ? 2.0 : 1.0;
    };

    anneal::JitterOptions opt;
    opt.tol = 0.05;
    opt.audit_sample = 1;
    const auto plan = anneal::calibrate_jitter(
        sp, x0, L0, opt, std::vector<double>(1, 0.0), tricky);

    CHECK(thr > 1.0);
    CHECK(plan.audited == 1u);
    CHECK(plan.non_monotone == 1u);
    CHECK((plan.dims[0].status & anneal::JitterNonMonotone) != 0u);
}

TEST_CASE("interior audit stays quiet on a well-behaved objective",
          "[HPO][anneal_jitter][audit]")
{
    Grouped g(4, 2);
    auto sp = box(g.dim, -1.0, 1.0);
    sp.parameter_labels = g.labels;
    std::vector<double> x0(g.dim, g.c);

    anneal::JitterOptions opt;
    opt.tol = 0.05;
    opt.audit_sample = 8;
    const auto plan = anneal::calibrate_jitter(
        sp, x0, g.L0, opt, std::vector<double>(g.dim, 0.0),
        [&](std::size_t j, double v) { return g.loss(j, v); });

    CHECK(plan.audited > 0u);
    CHECK(plan.non_monotone == 0u);   /* a quadratic IS monotone in |dx| */
}

TEST_CASE("cost levers: measured saving at a realistic search-space size",
          "[HPO][anneal_jitter][cost]")
{
    /* The decision this informs is whether AUTO is affordable on the real
     * 816-dimension Ta problem, where one objective evaluation is a set of
     * LAMMPS relaxations. Report measured probe counts, not estimates. */
    Grouped g(102, 8);                 /* 816 dims, 8 families */
    auto sp = box(g.dim, -1.0, 1.0);
    sp.parameter_labels = g.labels;
    std::vector<double> x0(g.dim, g.c);
    auto probe = [&](std::size_t j, double v) { return g.loss(j, v); };

    anneal::JitterOptions full;
    full.tol = 0.05;
    full.max_probes = 8;
    const auto all = anneal::calibrate_jitter(
        sp, x0, g.L0, full, std::vector<double>(g.dim, 0.0), probe);

    anneal::JitterOptions few = full;
    few.sample = 64;
    const auto sampled = anneal::calibrate_jitter(
        sp, x0, g.L0, few, std::vector<double>(g.dim, 0.0), probe);

    WARN("816 dims: full = " << all.probes << " probes ("
         << (static_cast<double>(all.probes) / g.dim) << "/dim);  "
         "INIT_JITTER_SAMPLE 64 = " << sampled.probes << " probes");

    CHECK(all.probes <= g.dim * full.max_probes + full.audit_sample);
    CHECK(sampled.probes * 8 < all.probes);   /* the lever earns its keep */
}
