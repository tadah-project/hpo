/* Width-invariant backends must evaluate the same REAL points under
 * OPTIM_SCALE NONE and OPTIM_SCALE AUTO (same seed): the unit-box map is
 * an affine change of variables their proposals already scale by width. */
#include "catch2/catch.hpp"
#include <tadah/hpo/anneal_core.h>
#include <tadah/hpo/algorithms.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/search_space.h>
#include "anneal_test_support.h"

#include <cmath>
#include <memory>
#include <vector>

using namespace tadah::hpo;

namespace {

using Vec = anneal_test::Vec;

OptimizerSpec anneal_spec(bool normalise)
{
    DimSpec a; a.key = a.label = "A"; a.lo = -2.0; a.hi = 6.0;
    DimSpec b; b.key = b.label = "B"; b.lo = 0.0;  b.hi = 0.01;   /* 800x narrower */
    ScalePolicy pol; pol.normalise = normalise;
    OptimizerSpec sp;
    sp.lib = "TADAH"; sp.algo = "ANNEAL";
    sp.bind_space(std::make_shared<const SearchSpace>(SearchSpace::build({a, b}, pol)));
    sp.optim_init = {{1.0, 0.004}};
    sp.seed = 20260907;
    sp.stop.maxeval = 120;
    sp.param["T_INIT"] = 1.0;
    sp.param["T_MIN"] = 0.05;
    sp.param["RT"] = 0.8;
    sp.param["CHAIN_LENGTH"] = 10.0;
    sp.param_raw["REANNEAL"] = "OFF";
    sp.param["STEP_FRAC"] = 0.2;
    return sp;
}

struct RecordingHost {
    std::vector<std::vector<double>> calls;
    double operator()(const Vec& x)
    {
        std::vector<double> p(static_cast<std::size_t>(x.size()));
        for (long i = 0; i < x.size(); ++i) p[static_cast<std::size_t>(i)] = x(i);
        calls.push_back(p);
        /* a bowl in REAL space; the narrow dim matters as much as the wide one */
        return std::pow(x(0) - 2.0, 2.0) + std::pow((x(1) - 0.003) / 0.001, 2.0);
    }
};

}  // namespace

TEST_CASE("Serial ANNEAL evaluates the same real points under NONE and AUTO",
          "[HPO][anneal][search_space][invariance]")
{
    RecordingHost none_host, auto_host;
    run_anneal_impl<Vec>(anneal_spec(false), none_host);
    run_anneal_impl<Vec>(anneal_spec(true),  auto_host);
    REQUIRE(none_host.calls.size() == auto_host.calls.size());
    REQUIRE(none_host.calls.size() > 20);
    for (std::size_t k = 0; k < none_host.calls.size(); ++k) {
        REQUIRE(none_host.calls[k].size() == 2);
        CHECK(none_host.calls[k][0] == Approx(auto_host.calls[k][0]).epsilon(1e-9).margin(1e-12));
        CHECK(none_host.calls[k][1] == Approx(auto_host.calls[k][1]).epsilon(1e-9).margin(1e-14));
    }
}

TEST_CASE("resolvable_coordinate_scale has a width-relative floor at the lower bound",
          "[HPO][anneal][search_space]")
{
    /* x = 0 exactly (the unit box's lower bound): the ULP of 0 is a
     * subnormal; the floor must be eps * width instead. */
    const double s0 = anneal::resolvable_coordinate_scale(0.0, 0.0, 1.0);
    CHECK(s0 == Approx(std::numeric_limits<double>::epsilon()));
    /* interior point: unchanged ULP semantics up to the floor */
    const double s1 = anneal::resolvable_coordinate_scale(0.5, 0.0, 1.0);
    CHECK(s1 >= std::numeric_limits<double>::epsilon() * 0.5);
    CHECK(s1 <= 2.0 * std::numeric_limits<double>::epsilon());
}
