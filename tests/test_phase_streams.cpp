// test_phase_streams.cpp
//
// The random numbers of a PRESTAGE refinement are its own: an optimiser
// with a SEED drew the same streams in the refinement as in the
// exploration, or in a PLAIN run, given the same seed (seed_stream.h). A
// PRESTAGE whose exploration and refinement both ran TADAH:RANDOM with
// SEED 7 refined by evaluating the exploration's draws again, and under
// RESTART the seeds HPO derives for a round (the refinement SEED + r*S,
// the exploration EXPLORE SEED + r) met for SEEDs that differ (SEED 7 and
// EXPLORE SEED 8: round 1's refinement drew round 0's exploration).
// test_hpo_termination.cpp has the end-to-end cases (TADAH:RANDOM); these
// drive each other seeded optimiser on a stub host.

#include "catch2/catch.hpp"

#include <tadah/hpo/algorithms.h>
#include <tadah/hpo/algorithms_asa.h>
#include <tadah/hpo/algorithms_pa.h>
#include <tadah/hpo/algorithms_pt.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/nlopt_adapter.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/phase_specs.h>

#include "anneal_test_support.h"

#include <dlib/matrix.h>

#include <cstddef>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace {

using Vec = dlib::matrix<double, 0, 1>;
using Point = std::vector<double>;
using tadah::hpo::OptimizerSpec;

/* Records every point it is asked for, in order (one worker each run). */
struct Recorder {
    mutable std::mutex m;
    mutable std::vector<Point> calls;
    double operator()(const Vec& x) const
    {
        Point p(static_cast<std::size_t>(x.size()));
        for (long i = 0; i < x.size(); ++i) p[static_cast<std::size_t>(i)] = x(i);
        return (*this)(p);
    }
    double operator()(const Point& p) const
    {
        {
            std::lock_guard<std::mutex> lk(m);
            calls.push_back(p);
        }
        double s = 0.0;
        for (std::size_t i = 0; i < p.size(); ++i) {
            const double d = p[i] - 0.3 + 0.1 * static_cast<double>(i);
            s += d * d;
        }
        return s;
    }
};

struct NoRunner {};

/* A spec on [-1, 1]^2 starting at (0.5, -0.5), SEED 7, MAXEVAL 40. */
OptimizerSpec base_spec(const std::string& lib, const std::string& algo)
{
    OptimizerSpec sp;
    sp.low = {-1.0, -1.0};
    sp.high = {1.0, 1.0};
    sp.is_int.assign(2, false);
    sp.log_scaled.assign(2, false);
    sp.lib = lib;
    sp.algo = algo;
    sp.optim_init = {{0.5, -0.5}};
    sp.seed = 7;
    sp.stop.maxeval = 40;
    sp.num_threads = 1;
    return sp;
}

/* The points `sp` evaluates, in order. */
std::vector<Point> run(const OptimizerSpec& sp)
{
    Recorder rec;
    if (sp.lib == "ASA") {
        tadah::hpo::run_asa_erased(
            sp, sp.low.size(),
            [&rec](const std::vector<double>& x) { return rec(x); });
    } else if (sp.lib == "NLOPT") {
        tadah::hpo::run_nlopt_impl<Vec, NoRunner>(sp, rec);
    } else if (sp.algo == "ANNEAL") {
        tadah::hpo::run_anneal_impl<Vec>(sp, rec);
    } else if (sp.algo == "ANNEAL_PT") {
        tadah::hpo::run_anneal_pt_impl<Vec>(sp, rec);
    } else {
        tadah::hpo::run_anneal_pa_impl<Vec>(sp, rec);
    }
    return rec.calls;
}

}  // namespace

/* Pins: for TADAH:ANNEAL, ANNEAL_PT, ANNEAL_PA, ASA and NLOPT:GN_CRS2_LM
 * with SEED 7 from (0.5, -0.5) (MAXEVAL 40, one worker), two runs of the
 * spec evaluate the same points (the stream is the seed's), and the PRESTAGE
 * refinement spec built from it (prestage_refinement_spec, the same SEED
 * and start point) evaluates other points: its random numbers are not the
 * search's for the same seed.
 * Before the fix: the refinement evaluated exactly the points of the spec
 * outside it, in the same order (the same streams for the same seed). */
TEST_CASE("PRESTAGE refinement: a seeded optimiser draws other random "
          "numbers than with the same SEED outside the refinement",
          "[HPO][seed][prestage][phase_streams]")
{
    anneal_test::JournalCapture journal;
    struct Case { const char* lib; const char* algo; };
    for (const Case c : {Case{"TADAH", "ANNEAL"}, Case{"TADAH", "ANNEAL_PT"},
                         Case{"TADAH", "ANNEAL_PA"}, Case{"ASA", "ASA"},
                         Case{"NLOPT", "GN_CRS2_LM"}}) {
        SECTION(std::string(c.lib) + ":" + c.algo) {
            OptimizerSpec sp = base_spec(c.lib, c.algo);
            if (sp.algo == "ANNEAL") sp.param["CHAIN_LENGTH"] = 4.0;
            if (sp.algo == "ANNEAL_PT") {
                sp.param["REPLICAS"] = 4.0;
                sp.param["T_HOT"] = 1.0;
            }
            if (sp.algo == "ANNEAL_PA") {
                sp.pop = 8;
                sp.param["CHAIN_LENGTH"] = 2.0;
                sp.param["T_HOT"] = 1.0;
            }
            if (sp.lib == "ASA") sp.param["Limit_Generated"] = 40.0;

            const std::vector<Point> search = run(sp);
            REQUIRE(search.size() >= 10u);
            /* The seed decides the stream: the run is reproducible. */
            CHECK(run(sp) == search);

            const OptimizerSpec refine =
                tadah::hpo::prestage_refinement_spec(sp, sp.optim_init);
            REQUIRE(refine.seed == sp.seed);
            REQUIRE(refine.optim_init == sp.optim_init);
            const std::vector<Point> refined = run(refine);
            REQUIRE(refined.size() >= 10u);
            INFO("search:  " << ::Catch::Detail::stringify(search));
            INFO("refined: " << ::Catch::Detail::stringify(refined));
            CHECK(refined != search);
        }
    }
}
