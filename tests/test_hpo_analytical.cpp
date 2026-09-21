// test_hpo_analytical.cpp
//
// Phase 2A.4 tests for the HPO analytical-gradient orchestration. Three
// thrusts:
//
//   1. LossFunction::loss_prime parity vs. central-difference for every
//      loss type. The chain rule  dL/dypred = -(dL/dr)  is sign-checked.
//
//   2. analytics_grad::* parity vs. central-difference for every metric.
//      Each helper is fed a small (truth, predicted) StructureDB pair
//      and the per-cell sum of d(metric)/d(pred_cell) is integrated by
//      a small finite perturbation; the sum should match the FD
//      derivative of the metric.
//
//   3. make_param_ref_table walks HPOTargets::optimap in the same order
//      as HPO_Host::param_labels_ — needed so the orchestrator's flat
//      θ index matches what the optimiser sees.
//
// The Layer-3 end-to-end ANALYTICAL-vs-NUMERICAL parity through a
// running HPO_Host is staged behind 2A.4 wiring; the descriptor-side
// dPhi_train/dθ + dPhi_val/dθ assembly is not yet wired into the host's
// forward eval. Until then, HYBRID + HPO_Host::analytical_gradient is
// equivalent to NUMERICAL by construction (the orchestrator pushes
// every index onto needs_fd, and the masked-FD pass overwrites every
// entry); covering that here would just retest the GradientProvider
// dispatch already exercised in test_hpo.cpp.

#include "catch2/catch.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <tadah/core/context.h>
#include <tadah/core/element.h>
#include <tadah/core/periodic_table.h>
#include <tadah/core/aeds_type.h>
#include <tadah/hpo/dloss_dtheta.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/sims_grad.h>

#include <Eigen/Dense>
#include <tadah/mlip/analytics/analytics_grad.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/structure_db.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#if defined(TADAH_ENABLE_OPENMP)
#   include <omp.h>
#endif

using tadah::LossFunction;
using tadah::mlip::PredCellKind;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;

namespace {

// Central-difference reference for any single-arg unary function f.
double central_diff(std::function<double(double)> f, double x, double h = 1e-6) {
    return (f(x + h) - f(x - h)) / (2.0 * h);
}

// A tiny, deterministic (truth, predicted) StructureDB pair we can perturb.
// 3 structures, 2 atoms each. Nonzero forces, stresses, distinct truth/pred
// values so every metric has a non-trivial gradient.
StructureDB make_truth() {
    StructureDB db;
    db.dbidx = {0, 3};
    db.structures.resize(3);
    for (std::size_t i = 0; i < 3; ++i) {
        Structure& s = db.structures[i];
        s.energy = 1.0 + 0.5 * static_cast<double>(i);
        s.atoms.resize(2);
        for (std::size_t a = 0; a < 2; ++a) {
            for (std::size_t k = 0; k < 3; ++k) {
                s.atoms[a].force(k) = 0.1 * (a + 1) * (k + 1)
                                    - 0.05 * static_cast<double>(i);
            }
        }
        for (std::size_t x = 0; x < 3; ++x)
            for (std::size_t y = 0; y < 3; ++y)
                s.stress(x, y) = 0.01 * (x + 1) * (y + 1) * (i + 1);
    }
    return db;
}

StructureDB make_pred(double offset_e, double offset_f, double offset_s) {
    StructureDB db = make_truth();  // start from truth; perturb fields
    for (std::size_t i = 0; i < db.size(); ++i) {
        Structure& s = db.structures[i];
        s.energy += offset_e * (1.0 + 0.13 * static_cast<double>(i));
        for (std::size_t a = 0; a < 2; ++a) {
            for (std::size_t k = 0; k < 3; ++k) {
                s.atoms[a].force(k) += offset_f
                                     * (1.0 + 0.07 * static_cast<double>(i + a + k));
            }
        }
        for (std::size_t x = 0; x < 3; ++x) {
            for (std::size_t y = 0; y < 3; ++y) {
                s.stress(x, y) += offset_s
                                * (1.0 + 0.03 * static_cast<double>(i + x + y));
            }
        }
    }
    return db;
}

// Apply a delta to a single (i, kind, comp) cell of pred. Used to FD a
// single partial of any metric.
void perturb_pred_cell(StructureDB& pred, std::size_t i,
                       PredCellKind kind, std::size_t comp, double delta) {
    Structure& s = pred.structures[i];
    switch (kind) {
        case PredCellKind::ENERGY:
            s.energy += delta; break;
        case PredCellKind::FORCE: {
            const std::size_t a = comp / 3;
            const std::size_t k = comp % 3;
            s.atoms[a].force(k) += delta;
            break;
        }
        case PredCellKind::STRESS: {
            std::size_t cnt = 0;
            for (std::size_t x = 0; x < 3 && cnt <= comp; ++x) {
                for (std::size_t y = x; y < 3; ++y, ++cnt) {
                    if (cnt == comp) {
                        s.stress(x, y) += delta;
                        if (x != y) s.stress(y, x) += delta;  // symmetric
                        return;
                    }
                }
            }
            break;
        }
    }
}

}  // namespace

/* ====================================================================
 * 1. LossFunction::loss_prime parity vs. central-difference.
 * ==================================================================== */

TEST_CASE("LossFunction::loss_prime matches central-difference per type",
          "[HPO][loss_prime][parity]")
{
    struct Case {
        LossFunction::Type type;
        double y;
        double ypred;
        double w;
        std::vector<double> params;
    };

    const std::vector<Case> cases = {
        // Below-kink, above-kink, and a few off-zero values for each.
        {LossFunction::Type::L1,       1.0,  0.7, 1.0, {}},
        {LossFunction::Type::L1,      -2.0, -2.5, 0.5, {}},
        {LossFunction::Type::L2,       1.0,  0.7, 1.0, {}},
        {LossFunction::Type::L2,       0.0, -0.4, 1.0, {}},
        {LossFunction::Type::LOG_COSH, 1.0,  0.5, 1.0, {}},
        {LossFunction::Type::LOG_COSH, 2.5,  3.0, 0.7, {}},
        {LossFunction::Type::MAPE,     1.0,  0.7, 1.0, {}},
        {LossFunction::Type::HUBER,    1.0,  0.5, 1.0, {1.0}},  // |r|<delta
        {LossFunction::Type::HUBER,    1.0, -2.0, 1.0, {1.0}},  // |r|>delta
        {LossFunction::Type::TUKEY,    1.0,  0.7, 1.0, {1.5}},  // |r|<c
        // RMSLE only for positive y/ypred (log requires it).
        {LossFunction::Type::RMSLE,    2.0,  1.7, 1.0, {}},
    };

    for (const auto& c : cases) {
        LossFunction lf(c.type, c.params);
        const double a_prime = lf.loss_prime(c.y, c.ypred, c.w);
        const double fd = central_diff(
            [&](double yp) { return lf.loss(c.y, yp, c.w); }, c.ypred, 1e-6);
        INFO("type=" << LossFunction::to_string(c.type)
             << " y=" << c.y << " ypred=" << c.ypred
             << " analytical=" << a_prime << " FD=" << fd);
        // 1e-5 tolerance: FD's truncation error is O(h^2 f''') and the
        // step is 1e-6, so 1e-12 truncation + 1e-10 round-off -> 1e-10
        // theoretical. We allow 1e-5 absolute to cover Tukey's near-c
        // edge case where f''' is non-trivial.
        CHECK(std::abs(a_prime - fd) < 1e-5);
    }
}

/* ====================================================================
 * 2. analytics_grad::* parity vs. central-difference.
 * ==================================================================== */

TEST_CASE("analytics_grad::emae partial sum matches FD",
          "[MLIP][analytics_grad][parity]")
{
    StructureDB truth = make_truth();
    StructureDB pred  = make_pred(0.05, 0.0, 0.0);

    // Reference metric.
    auto compute_emae = [&](const StructureDB& p) {
        double s = 0.0;
        for (std::size_t i = 0; i < truth.size(); ++i)
            s += std::abs(truth(i).energy - p(i).energy)
               / static_cast<double>(truth(i).natoms());
        return s / static_cast<double>(truth.size());
    };

    // For each (i, ENERGY, 0) cell, FD the partial and compare.
    for (std::size_t i = 0; i < truth.size(); ++i) {
        const double h = 1e-5;
        StructureDB p_plus  = pred;  perturb_pred_cell(p_plus,  i, PredCellKind::ENERGY, 0,  h);
        StructureDB p_minus = pred;  perturb_pred_cell(p_minus, i, PredCellKind::ENERGY, 0, -h);
        const double fd = (compute_emae(p_plus) - compute_emae(p_minus)) / (2.0 * h);

        // Capture the analytical partial for this cell.
        double a_partial = 0.0;
        tadah::mlip::analytics_grad::emae(truth, pred, /*scale=*/1.0,
            [&](std::size_t ii, PredCellKind kind, std::size_t /*comp*/, double v) {
                if (ii == i && kind == PredCellKind::ENERGY) a_partial = v;
            });

        INFO("emae i=" << i << " analytical=" << a_partial << " FD=" << fd);
        CHECK(std::abs(a_partial - fd) < 1e-7);
    }
}

TEST_CASE("analytics_grad::ermse partial matches FD",
          "[MLIP][analytics_grad][parity]")
{
    StructureDB truth = make_truth();
    StructureDB pred  = make_pred(0.05, 0.0, 0.0);

    auto compute_ermse = [&](const StructureDB& p) {
        double s = 0.0;
        for (std::size_t i = 0; i < truth.size(); ++i) {
            const double e = (truth(i).energy - p(i).energy)
                           / static_cast<double>(truth(i).natoms());
            s += e * e;
        }
        return std::sqrt(s / static_cast<double>(truth.size()));
    };

    for (std::size_t i = 0; i < truth.size(); ++i) {
        const double h = 1e-5;
        StructureDB p_plus  = pred;  perturb_pred_cell(p_plus,  i, PredCellKind::ENERGY, 0,  h);
        StructureDB p_minus = pred;  perturb_pred_cell(p_minus, i, PredCellKind::ENERGY, 0, -h);
        const double fd = (compute_ermse(p_plus) - compute_ermse(p_minus)) / (2.0 * h);

        double a_partial = 0.0;
        tadah::mlip::analytics_grad::ermse(truth, pred, /*scale=*/1.0,
            [&](std::size_t ii, PredCellKind kind, std::size_t /*comp*/, double v) {
                if (ii == i && kind == PredCellKind::ENERGY) a_partial = v;
            });

        INFO("ermse i=" << i << " analytical=" << a_partial << " FD=" << fd);
        CHECK(std::abs(a_partial - fd) < 1e-7);
    }
}

TEST_CASE("analytics_grad::ersq partial matches FD",
          "[MLIP][analytics_grad][parity]")
{
    StructureDB truth = make_truth();
    StructureDB pred  = make_pred(0.05, 0.0, 0.0);

    // R^2 metric per Statistics::r_sq:
    //   1 - SS_res / SS_tot
    auto compute_ersq = [&](const StructureDB& p) {
        const std::size_t N = truth.size();
        double mean_obs = 0.0;
        for (std::size_t i = 0; i < N; ++i)
            mean_obs += truth(i).energy / static_cast<double>(truth(i).natoms());
        mean_obs /= static_cast<double>(N);
        double SS_res = 0.0, SS_tot = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            const double n   = static_cast<double>(truth(i).natoms());
            const double obs = truth(i).energy / n;
            const double prd = p(i).energy / n;
            SS_res += (obs - prd) * (obs - prd);
            SS_tot += (obs - mean_obs) * (obs - mean_obs);
        }
        return 1.0 - SS_res / SS_tot;
    };

    for (std::size_t i = 0; i < truth.size(); ++i) {
        const double h = 1e-5;
        StructureDB p_plus  = pred;  perturb_pred_cell(p_plus,  i, PredCellKind::ENERGY, 0,  h);
        StructureDB p_minus = pred;  perturb_pred_cell(p_minus, i, PredCellKind::ENERGY, 0, -h);
        const double fd = (compute_ersq(p_plus) - compute_ersq(p_minus)) / (2.0 * h);

        double a_partial = 0.0;
        tadah::mlip::analytics_grad::ersq(truth, pred, /*scale=*/1.0,
            [&](std::size_t ii, PredCellKind kind, std::size_t /*comp*/, double v) {
                if (ii == i && kind == PredCellKind::ENERGY) a_partial = v;
            });

        INFO("ersq i=" << i << " analytical=" << a_partial << " FD=" << fd);
        CHECK(std::abs(a_partial - fd) < 1e-7);
    }
}

TEST_CASE("analytics_grad::frmse partial matches FD",
          "[MLIP][analytics_grad][parity]")
{
    StructureDB truth = make_truth();
    StructureDB pred  = make_pred(0.0, 0.05, 0.0);

    auto compute_frmse = [&](const StructureDB& p) {
        std::size_t N_F = 0;
        double s = 0.0;
        for (std::size_t i = 0; i < truth.size(); ++i)
            for (std::size_t a = 0; a < truth(i).natoms(); ++a)
                for (std::size_t k = 0; k < 3; ++k) {
                    const double d = truth(i).atoms[a].force(k)
                                   - p(i).atoms[a].force(k);
                    s += d * d; ++N_F;
                }
        return std::sqrt(s / static_cast<double>(N_F));
    };

    // FD a single (0, FORCE, 0) cell.
    {
        const double h = 1e-5;
        StructureDB p_plus  = pred;  perturb_pred_cell(p_plus,  0, PredCellKind::FORCE, 0,  h);
        StructureDB p_minus = pred;  perturb_pred_cell(p_minus, 0, PredCellKind::FORCE, 0, -h);
        const double fd = (compute_frmse(p_plus) - compute_frmse(p_minus)) / (2.0 * h);

        double a_partial = 0.0;
        tadah::mlip::analytics_grad::frmse(truth, pred, /*scale=*/1.0,
            [&](std::size_t i, PredCellKind kind, std::size_t comp, double v) {
                if (i == 0 && kind == PredCellKind::FORCE && comp == 0) a_partial = v;
            });

        INFO("frmse [0,F,0] analytical=" << a_partial << " FD=" << fd);
        CHECK(std::abs(a_partial - fd) < 1e-7);
    }
}

TEST_CASE("analytics_grad::srmse partial matches FD",
          "[MLIP][analytics_grad][parity]")
{
    StructureDB truth = make_truth();
    StructureDB pred  = make_pred(0.0, 0.0, 0.05);

    auto compute_srmse = [&](const StructureDB& p) {
        std::size_t N_S = 0;
        double s = 0.0;
        for (std::size_t i = 0; i < truth.size(); ++i) {
            for (std::size_t x = 0; x < 3; ++x) {
                for (std::size_t y = x; y < 3; ++y) {
                    const double d = truth(i).stress(x, y) - p(i).stress(x, y);
                    s += d * d; ++N_S;
                }
            }
        }
        return std::sqrt(s / static_cast<double>(N_S));
    };

    // FD the (0, STRESS, 0) cell — that is xx for structure 0.
    {
        const double h = 1e-5;
        StructureDB p_plus  = pred;  perturb_pred_cell(p_plus,  0, PredCellKind::STRESS, 0,  h);
        StructureDB p_minus = pred;  perturb_pred_cell(p_minus, 0, PredCellKind::STRESS, 0, -h);
        const double fd = (compute_srmse(p_plus) - compute_srmse(p_minus)) / (2.0 * h);

        double a_partial = 0.0;
        tadah::mlip::analytics_grad::srmse(truth, pred, /*scale=*/1.0,
            [&](std::size_t i, PredCellKind kind, std::size_t comp, double v) {
                if (i == 0 && kind == PredCellKind::STRESS && comp == 0) a_partial = v;
            });

        INFO("srmse [0,S,0] analytical=" << a_partial << " FD=" << fd);
        CHECK(std::abs(a_partial - fd) < 1e-7);
    }
}

/* ====================================================================
 * 2b. Multi-segment analytics_grad::* parity tests.
 *
 * fill_validation aggregates Analytics::calc_* outputs as v.mean() — the
 * arithmetic mean of per-segment metric values. For S > 1 the gradient
 * helpers must produce d(scalar_avg)/d(cell) where
 *
 *   scalar_avg = (1/S) * sum_{s=0..S-1} metric_s
 *
 * Fixture: 10 structures, dbidx = {0, 4, 7, 10} (S = 3 segments of sizes
 * 4, 3, 3). truth/pred are offset just enough to avoid zero-derivative
 * pathologies on any segment.
 * ==================================================================== */

namespace {

StructureDB make_truth_multiseg() {
    StructureDB db;
    db.dbidx = {0, 4, 7, 10};
    db.structures.resize(10);
    for (std::size_t i = 0; i < 10; ++i) {
        Structure& s = db.structures[i];
        // Vary natoms across structures to stress the per-segment N_s
        // and per-cell natoms_i factors in the chain rule.
        const std::size_t na = 2 + (i % 3);
        s.atoms.resize(na);
        s.energy = 1.0 + 0.3 * static_cast<double>(i)
                       + 0.15 * static_cast<double>(na);
        for (std::size_t a = 0; a < na; ++a) {
            for (std::size_t k = 0; k < 3; ++k) {
                s.atoms[a].force(k) = 0.07 * static_cast<double>(a + 1)
                                    * static_cast<double>(k + 1)
                                    - 0.04 * static_cast<double>(i);
            }
        }
        for (std::size_t x = 0; x < 3; ++x)
            for (std::size_t y = 0; y < 3; ++y)
                s.stress(x, y) = 0.013 * static_cast<double>((x + 1) * (y + 1))
                               * static_cast<double>(i + 1);
    }
    return db;
}

StructureDB make_pred_multiseg(double dE, double dF, double dS) {
    StructureDB db = make_truth_multiseg();
    for (std::size_t i = 0; i < db.size(); ++i) {
        Structure& s = db.structures[i];
        s.energy += dE * (1.0 + 0.11 * static_cast<double>(i));
        for (std::size_t a = 0; a < s.natoms(); ++a) {
            for (std::size_t k = 0; k < 3; ++k) {
                s.atoms[a].force(k) += dF
                                     * (1.0 + 0.05 * static_cast<double>(i + a + k));
            }
        }
        for (std::size_t x = 0; x < 3; ++x) {
            for (std::size_t y = 0; y < 3; ++y) {
                s.stress(x, y) += dS
                                * (1.0 + 0.02 * static_cast<double>(i + x + y));
            }
        }
    }
    return db;
}

// Reference metric: mean over segments of (per-segment emae).
double compute_emae_multiseg(const StructureDB& truth, const StructureDB& pred) {
    const std::size_t S = truth.dbidx.size() - 1;
    double acc = 0.0;
    for (std::size_t s = 0; s < S; ++s) {
        const std::size_t lo = truth.dbidx[s];
        const std::size_t hi = truth.dbidx[s + 1];
        double seg = 0.0;
        for (std::size_t i = lo; i < hi; ++i)
            seg += std::abs(truth(i).energy - pred(i).energy)
                 / static_cast<double>(truth(i).natoms());
        acc += seg / static_cast<double>(hi - lo);
    }
    return acc / static_cast<double>(S);
}

double compute_ermse_multiseg(const StructureDB& truth, const StructureDB& pred) {
    const std::size_t S = truth.dbidx.size() - 1;
    double acc = 0.0;
    for (std::size_t s = 0; s < S; ++s) {
        const std::size_t lo = truth.dbidx[s];
        const std::size_t hi = truth.dbidx[s + 1];
        double sumsq = 0.0;
        for (std::size_t i = lo; i < hi; ++i) {
            const double e = (truth(i).energy - pred(i).energy)
                           / static_cast<double>(truth(i).natoms());
            sumsq += e * e;
        }
        acc += std::sqrt(sumsq / static_cast<double>(hi - lo));
    }
    return acc / static_cast<double>(S);
}

double compute_errmse_multiseg(const StructureDB& truth, const StructureDB& pred) {
    const std::size_t S = truth.dbidx.size() - 1;
    double acc = 0.0;
    for (std::size_t s = 0; s < S; ++s) {
        const std::size_t lo = truth.dbidx[s];
        const std::size_t hi = truth.dbidx[s + 1];
        double sumsq = 0.0;
        std::size_t N_eff = 0;
        for (std::size_t i = lo; i < hi; ++i) {
            if (truth(i).energy == 0.0) continue;
            const double rr = (truth(i).energy - pred(i).energy)
                            / truth(i).energy;
            sumsq += rr * rr;
            ++N_eff;
        }
        if (N_eff > 0)
            acc += std::sqrt(sumsq / static_cast<double>(N_eff));
    }
    return acc / static_cast<double>(S);
}

double compute_ersq_multiseg(const StructureDB& truth, const StructureDB& pred) {
    const std::size_t S = truth.dbidx.size() - 1;
    double acc = 0.0;
    for (std::size_t s = 0; s < S; ++s) {
        const std::size_t lo = truth.dbidx[s];
        const std::size_t hi = truth.dbidx[s + 1];
        const std::size_t N_s = hi - lo;
        double mean_obs = 0.0;
        for (std::size_t i = lo; i < hi; ++i)
            mean_obs += truth(i).energy / static_cast<double>(truth(i).natoms());
        mean_obs /= static_cast<double>(N_s);
        double SS_res = 0.0, SS_tot = 0.0;
        for (std::size_t i = lo; i < hi; ++i) {
            const double n   = static_cast<double>(truth(i).natoms());
            const double obs = truth(i).energy / n;
            const double prd = pred (i).energy / n;
            SS_res += (obs - prd) * (obs - prd);
            SS_tot += (obs - mean_obs) * (obs - mean_obs);
        }
        if (SS_tot > 0.0) acc += 1.0 - SS_res / SS_tot;
    }
    return acc / static_cast<double>(S);
}

double compute_frmse_multiseg(const StructureDB& truth, const StructureDB& pred) {
    const std::size_t S = truth.dbidx.size() - 1;
    double acc = 0.0;
    for (std::size_t s = 0; s < S; ++s) {
        const std::size_t lo = truth.dbidx[s];
        const std::size_t hi = truth.dbidx[s + 1];
        double sumsq = 0.0;
        std::size_t N_F = 0;
        for (std::size_t i = lo; i < hi; ++i) {
            for (std::size_t a = 0; a < truth(i).natoms(); ++a) {
                for (std::size_t k = 0; k < 3; ++k) {
                    const double d = truth(i).atoms[a].force(k)
                                   - pred(i).atoms[a].force(k);
                    sumsq += d * d;
                    ++N_F;
                }
            }
        }
        if (N_F > 0) acc += std::sqrt(sumsq / static_cast<double>(N_F));
    }
    return acc / static_cast<double>(S);
}

double compute_srmse_multiseg(const StructureDB& truth, const StructureDB& pred) {
    const std::size_t S = truth.dbidx.size() - 1;
    double acc = 0.0;
    for (std::size_t s = 0; s < S; ++s) {
        const std::size_t lo = truth.dbidx[s];
        const std::size_t hi = truth.dbidx[s + 1];
        double sumsq = 0.0;
        std::size_t N_S = 0;
        for (std::size_t i = lo; i < hi; ++i) {
            for (std::size_t x = 0; x < 3; ++x) {
                for (std::size_t y = x; y < 3; ++y) {
                    const double d = truth(i).stress(x, y) - pred(i).stress(x, y);
                    sumsq += d * d;
                    ++N_S;
                }
            }
        }
        if (N_S > 0) acc += std::sqrt(sumsq / static_cast<double>(N_S));
    }
    return acc / static_cast<double>(S);
}

}  // namespace

TEST_CASE("analytics_grad::emae multi-segment parity vs FD",
          "[MLIP][analytics_grad][parity][multiseg]")
{
    StructureDB truth = make_truth_multiseg();
    StructureDB pred  = make_pred_multiseg(0.05, 0.0, 0.0);

    for (std::size_t i = 0; i < truth.size(); ++i) {
        const double h = 1e-5;
        StructureDB p_plus  = pred; perturb_pred_cell(p_plus,  i, PredCellKind::ENERGY, 0,  h);
        StructureDB p_minus = pred; perturb_pred_cell(p_minus, i, PredCellKind::ENERGY, 0, -h);
        const double fd = (compute_emae_multiseg(truth, p_plus)
                         - compute_emae_multiseg(truth, p_minus)) / (2.0 * h);

        double a_partial = 0.0;
        tadah::mlip::analytics_grad::emae(truth, pred, /*scale=*/1.0,
            [&](std::size_t ii, PredCellKind kind, std::size_t, double v) {
                if (ii == i && kind == PredCellKind::ENERGY) a_partial = v;
            });
        INFO("emae multiseg i=" << i << " a=" << a_partial << " fd=" << fd);
        CHECK(std::abs(a_partial - fd) < 1e-7);
    }
}

TEST_CASE("analytics_grad::ermse multi-segment parity vs FD",
          "[MLIP][analytics_grad][parity][multiseg]")
{
    StructureDB truth = make_truth_multiseg();
    StructureDB pred  = make_pred_multiseg(0.05, 0.0, 0.0);

    for (std::size_t i = 0; i < truth.size(); ++i) {
        const double h = 1e-5;
        StructureDB p_plus  = pred; perturb_pred_cell(p_plus,  i, PredCellKind::ENERGY, 0,  h);
        StructureDB p_minus = pred; perturb_pred_cell(p_minus, i, PredCellKind::ENERGY, 0, -h);
        const double fd = (compute_ermse_multiseg(truth, p_plus)
                         - compute_ermse_multiseg(truth, p_minus)) / (2.0 * h);

        double a_partial = 0.0;
        tadah::mlip::analytics_grad::ermse(truth, pred, /*scale=*/1.0,
            [&](std::size_t ii, PredCellKind kind, std::size_t, double v) {
                if (ii == i && kind == PredCellKind::ENERGY) a_partial = v;
            });
        INFO("ermse multiseg i=" << i << " a=" << a_partial << " fd=" << fd);
        CHECK(std::abs(a_partial - fd) < 1e-7);
    }
}

TEST_CASE("analytics_grad::errmse multi-segment parity vs FD",
          "[MLIP][analytics_grad][parity][multiseg]")
{
    StructureDB truth = make_truth_multiseg();
    StructureDB pred  = make_pred_multiseg(0.05, 0.0, 0.0);

    for (std::size_t i = 0; i < truth.size(); ++i) {
        const double h = 1e-5;
        StructureDB p_plus  = pred; perturb_pred_cell(p_plus,  i, PredCellKind::ENERGY, 0,  h);
        StructureDB p_minus = pred; perturb_pred_cell(p_minus, i, PredCellKind::ENERGY, 0, -h);
        const double fd = (compute_errmse_multiseg(truth, p_plus)
                         - compute_errmse_multiseg(truth, p_minus)) / (2.0 * h);

        double a_partial = 0.0;
        tadah::mlip::analytics_grad::errmse(truth, pred, /*scale=*/1.0,
            [&](std::size_t ii, PredCellKind kind, std::size_t, double v) {
                if (ii == i && kind == PredCellKind::ENERGY) a_partial = v;
            });
        INFO("errmse multiseg i=" << i << " a=" << a_partial << " fd=" << fd);
        CHECK(std::abs(a_partial - fd) < 1e-6);
    }
}

TEST_CASE("analytics_grad::ersq multi-segment parity vs FD",
          "[MLIP][analytics_grad][parity][multiseg]")
{
    StructureDB truth = make_truth_multiseg();
    StructureDB pred  = make_pred_multiseg(0.05, 0.0, 0.0);

    for (std::size_t i = 0; i < truth.size(); ++i) {
        const double h = 1e-5;
        StructureDB p_plus  = pred; perturb_pred_cell(p_plus,  i, PredCellKind::ENERGY, 0,  h);
        StructureDB p_minus = pred; perturb_pred_cell(p_minus, i, PredCellKind::ENERGY, 0, -h);
        const double fd = (compute_ersq_multiseg(truth, p_plus)
                         - compute_ersq_multiseg(truth, p_minus)) / (2.0 * h);

        double a_partial = 0.0;
        tadah::mlip::analytics_grad::ersq(truth, pred, /*scale=*/1.0,
            [&](std::size_t ii, PredCellKind kind, std::size_t, double v) {
                if (ii == i && kind == PredCellKind::ENERGY) a_partial = v;
            });
        INFO("ersq multiseg i=" << i << " a=" << a_partial << " fd=" << fd);
        CHECK(std::abs(a_partial - fd) < 1e-6);
    }
}

TEST_CASE("analytics_grad::frmse multi-segment parity vs FD",
          "[MLIP][analytics_grad][parity][multiseg]")
{
    StructureDB truth = make_truth_multiseg();
    StructureDB pred  = make_pred_multiseg(0.0, 0.05, 0.0);

    // FD a representative subset of force cells across all three segments.
    struct Probe { std::size_t i; std::size_t a; std::size_t k; };
    const Probe probes[] = {
        {0, 0, 0}, {0, 1, 2},  // segment 0
        {4, 0, 1}, {5, 1, 0},  // segment 1
        {7, 0, 2}, {9, 1, 1},  // segment 2
    };
    for (const auto& pr : probes) {
        const std::size_t comp = 3 * pr.a + pr.k;
        const double h = 1e-5;
        StructureDB p_plus  = pred; perturb_pred_cell(p_plus,  pr.i, PredCellKind::FORCE, comp,  h);
        StructureDB p_minus = pred; perturb_pred_cell(p_minus, pr.i, PredCellKind::FORCE, comp, -h);
        const double fd = (compute_frmse_multiseg(truth, p_plus)
                         - compute_frmse_multiseg(truth, p_minus)) / (2.0 * h);

        double a_partial = 0.0;
        tadah::mlip::analytics_grad::frmse(truth, pred, /*scale=*/1.0,
            [&](std::size_t ii, PredCellKind kind, std::size_t cc, double v) {
                if (ii == pr.i && kind == PredCellKind::FORCE && cc == comp) a_partial = v;
            });
        INFO("frmse multiseg i=" << pr.i << " comp=" << comp
             << " a=" << a_partial << " fd=" << fd);
        CHECK(std::abs(a_partial - fd) < 1e-7);
    }
}

TEST_CASE("analytics_grad::srmse multi-segment parity vs FD",
          "[MLIP][analytics_grad][parity][multiseg]")
{
    StructureDB truth = make_truth_multiseg();
    StructureDB pred  = make_pred_multiseg(0.0, 0.0, 0.05);

    struct Probe { std::size_t i; std::size_t comp; };
    const Probe probes[] = {
        {0, 0}, {0, 3},   // segment 0
        {4, 0}, {5, 2},   // segment 1
        {7, 1}, {9, 5},   // segment 2
    };
    for (const auto& pr : probes) {
        const double h = 1e-5;
        StructureDB p_plus  = pred; perturb_pred_cell(p_plus,  pr.i, PredCellKind::STRESS, pr.comp,  h);
        StructureDB p_minus = pred; perturb_pred_cell(p_minus, pr.i, PredCellKind::STRESS, pr.comp, -h);
        const double fd = (compute_srmse_multiseg(truth, p_plus)
                         - compute_srmse_multiseg(truth, p_minus)) / (2.0 * h);

        double a_partial = 0.0;
        tadah::mlip::analytics_grad::srmse(truth, pred, /*scale=*/1.0,
            [&](std::size_t ii, PredCellKind kind, std::size_t cc, double v) {
                if (ii == pr.i && kind == PredCellKind::STRESS && cc == pr.comp) a_partial = v;
            });
        INFO("srmse multiseg i=" << pr.i << " comp=" << pr.comp
             << " a=" << a_partial << " fd=" << fd);
        CHECK(std::abs(a_partial - fd) < 1e-7);
    }
}

TEST_CASE("analytics_grad multi-segment: dead-flat segment emits zero",
          "[MLIP][analytics_grad][edge][multiseg]")
{
    // Construct a fixture where segment 1 has truth == pred exactly:
    // gradient on every cell of segment 1 should be zero (smooth limit at
    // a flat minimum), other segments should still emit non-zero partials.
    StructureDB truth = make_truth_multiseg();
    StructureDB pred  = make_pred_multiseg(0.05, 0.0, 0.0);
    // Force segment 1 (indices 4, 5, 6) pred energies to match truth.
    for (std::size_t i = 4; i < 7; ++i) pred.structures[i].energy = truth(i).energy;

    bool any_seg1_nonzero = false;
    std::size_t count_seg0_or_2 = 0;
    bool any_seg0_or_2_nonzero = false;
    tadah::mlip::analytics_grad::ermse(truth, pred, /*scale=*/1.0,
        [&](std::size_t ii, PredCellKind kind, std::size_t, double v) {
            if (kind != PredCellKind::ENERGY) return;
            if (ii >= 4 && ii < 7) {
                if (v != 0.0) any_seg1_nonzero = true;
            } else {
                ++count_seg0_or_2;
                if (std::abs(v) > 1e-12) any_seg0_or_2_nonzero = true;
            }
        });
    CHECK_FALSE(any_seg1_nonzero);
    CHECK(any_seg0_or_2_nonzero);
    // 7 structures in segments 0 and 2 (4 + 3). The ermse helper emits a
    // partial for every (i, ENERGY, 0) in non-dead segments.
    CHECK(count_seg0_or_2 == 7);
}

TEST_CASE("analytics_grad multi-segment: empty segment is skipped safely",
          "[MLIP][analytics_grad][edge][multiseg]")
{
    // dbidx = {0, 4, 4, 7} — segment 1 is empty (lo == hi).
    StructureDB truth;
    truth.dbidx = {0, 4, 4, 7};
    truth.structures.resize(7);
    StructureDB pred;
    pred.dbidx = {0, 4, 4, 7};
    pred.structures.resize(7);
    for (std::size_t i = 0; i < 7; ++i) {
        truth.structures[i].atoms.resize(2);
        pred .structures[i].atoms.resize(2);
        truth.structures[i].energy = 1.0 + 0.2 * static_cast<double>(i);
        pred .structures[i].energy = truth(i).energy + 0.05;
    }
    // Helpers must not throw, divide by zero, or emit for the empty seg.
    std::size_t calls = 0;
    REQUIRE_NOTHROW(
        tadah::mlip::analytics_grad::emae(truth, pred, /*scale=*/1.0,
            [&](std::size_t, PredCellKind, std::size_t, double v) {
                CHECK(std::isfinite(v));
                ++calls;
            }));
    // 7 ENERGY cells, all in non-empty segments.
    CHECK(calls == 7);
}

/* ====================================================================
 * 3. compute_dloss_dtheta orchestrator routes everything to needs_fd.
 *
 * Phase 2A.4 deliverable; the actual analytical evaluation is staged
 * behind the descriptor-side dPhi assembly. This test pins the
 * contract so a future implementer who fills the orchestrator must
 * also update the test.
 * ==================================================================== */

TEST_CASE("compute_dloss_dtheta currently routes every index to needs_fd",
          "[HPO][dloss_dtheta][routing]")
{
    // compute_dloss_dtheta is a template; we exercise it with a tiny
    // stub vector type below (VecAdapter) rather than pulling in dlib
    // just to verify the routing contract.

    // Build a minimal plan and a synthetic param-ref table.
    std::vector<tadah::hpo::ParamRef> prefs;
    {
        tadah::hpo::ParamRef p0; p0.theta_name = "CGRID2B"; p0.arg_pos = 1; p0.flat_idx = 0;
        tadah::hpo::ParamRef p1; p1.theta_name = "RCUT2B";  p1.arg_pos = -1; p1.flat_idx = 1;
        prefs.push_back(p0); prefs.push_back(p1);
    }
    tadah::models::FactorHandle fh;
    fh.valid = true;
    fh.n     = 3;
    fh.algo  = tadah::models::Algorithm::NormalEquations;
    fh.AtA.resize(9, 0.0);

    tadah::hpo::DLossDThetaPlan plan;
    plan.factor = &fh;
    plan.prefs  = &prefs;

    // Use dlib::matrix-style aliasing via Eigen-free shim — Catch test only
    // needs ::set_size + operator(). The dlib type used elsewhere in this
    // suite is `dlib::matrix<double, 0, 1>` but bringing it in here would
    // pull a heavy header. Use a tiny adapter struct.
    struct VecAdapter {
        std::vector<double> data;
        long size() const { return static_cast<long>(data.size()); }
        void set_size(long n) { data.assign(static_cast<std::size_t>(n), 0.0); }
        double  operator()(long i) const { return data[static_cast<std::size_t>(i)]; }
        double& operator()(long i)       { return data[static_cast<std::size_t>(i)]; }
    };

    VecAdapter x;  x.set_size(2);  x.data = {0.5, 3.0};
    VecAdapter g;
    std::vector<int> needs_fd;
    REQUIRE(tadah::hpo::compute_dloss_dtheta(plan, x, g, needs_fd));
    REQUIRE(g.size() == 2);
    CHECK(g(0) == 0.0);
    CHECK(g(1) == 0.0);
    REQUIRE(needs_fd.size() == 2);
    CHECK(needs_fd[0] == 0);
    CHECK(needs_fd[1] == 1);
}

/* The IFT building block (detail::ift_rhs + LinearRegressor::backsolve)
 * is exercised at Layer-2 in MODELS/tests/test_linear_regressor_ift.cpp
 * for the LAPACK-backed path. The HPO-side hand-rolled Cholesky path
 * is exercised below — same math, different LAPACK availability:
 * HPO TUs are compiled with -DTADAH_USE_DLIB which suppresses
 * tadah::core::dpotrf_, so dloss_dtheta.h provides cholesky_upper /
 * cholesky_solve_uut in pure C++ instead.
 *
 * A Layer-3 end-to-end ANALYTICAL-vs-NUMERICAL parity test through a
 * live HPO_Host is the documented next sub-task; it requires the
 * descriptor-walk dPhi assembly to land first. */

namespace {

// Build a small SPD matrix in column-major n*n layout: A = R^T R + epsI.
std::vector<double> spd_matrix(int n, std::uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> R(static_cast<std::size_t>(n) * n);
    for (std::size_t k = 0; k < R.size(); ++k) R[k] = dist(rng);
    std::vector<double> A(static_cast<std::size_t>(n) * n, 0.0);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            double s = 0.0;
            for (int k = 0; k < n; ++k)
                s += R[k + std::size_t(i) * n] * R[k + std::size_t(j) * n];
            A[i + std::size_t(j) * n] = s;
        }
    for (int i = 0; i < n; ++i) A[i + std::size_t(i) * n] += 0.1;  // bump diag
    return A;
}

// Build phi(m × n) row-major where column 1 scales by exp(theta); other
// columns are theta-independent. Mirrors the MODELS-side
// build_phi(base, theta) used by test_linear_regressor_ift.cpp.
std::vector<double> build_phi_rowmajor(const std::vector<double>& base_rowmajor,
                                       std::size_t m, std::size_t n,
                                       double theta)
{
    std::vector<double> Phi = base_rowmajor;
    const double s = std::exp(theta);
    for (std::size_t i = 0; i < m; ++i)
        Phi[i * n + 1] *= s;
    return Phi;
}

// Augment phi with a sqrt(lambda)*I block at the bottom (n_bias columns
// unaugmented). Returns the (m+n) × n augmented matrix in row-major.
std::vector<double> augment_ridge(const std::vector<double>& phi_unaug,
                                  std::size_t m, std::size_t n,
                                  double lambda, std::size_t n_bias)
{
    std::vector<double> P((m + n) * n, 0.0);
    for (std::size_t r = 0; r < m; ++r)
        for (std::size_t c = 0; c < n; ++c)
            P[r * n + c] = phi_unaug[r * n + c];
    if (lambda > 0.0) {
        const double s = std::sqrt(lambda);
        for (std::size_t k = n_bias; k < n; ++k)
            P[(m + k) * n + k] = s;
    }
    return P;
}

std::vector<double> augment_T(const std::vector<double>& T_unaug,
                              std::size_t m, std::size_t n_aug)
{
    std::vector<double> T(m + n_aug, 0.0);
    for (std::size_t i = 0; i < m; ++i) T[i] = T_unaug[i];
    return T;
}

// Solve (Phi^T Phi) W = Phi^T T via the same path the HPO TU has
// available — compute_normal_equations + cholesky_upper + cholesky_solve_uut.
// Used by the FD reference; lets the test exercise the full HPO-side
// math chain at every theta perturbation.
tadah::core::t_type fit_via_hpo_chol(const std::vector<double>& phi_aug,
                                     const std::vector<double>& t_aug,
                                     std::size_t                n_rows,
                                     std::size_t                n_cols)
{
    std::vector<double> AtA;
    tadah::hpo::detail::compute_normal_equations(
        phi_aug.data(), n_rows, n_cols, AtA);
    REQUIRE(tadah::hpo::detail::cholesky_upper(AtA.data(),
                                               static_cast<int>(n_cols)));
    tadah::core::t_type Atb(n_cols);
    for (std::size_t j = 0; j < n_cols; ++j) Atb[j] = 0.0;
    for (std::size_t r = 0; r < n_rows; ++r)
        for (std::size_t c = 0; c < n_cols; ++c)
            Atb[c] += phi_aug[r * n_cols + c] * t_aug[r];
    tadah::hpo::detail::cholesky_solve_uut(AtA.data(),
                                           static_cast<int>(n_cols),
                                           Atb.ptr());
    return Atb;
}

bool close_rel(double a, double b, double rtol = 1e-6, double atol = 1e-9)
{
    return std::abs(a - b) <= atol + rtol * std::max(std::abs(a), std::abs(b));
}

}  // namespace

TEST_CASE("cholesky_upper / cholesky_solve_uut round-trip an SPD matrix",
          "[HPO][dloss_dtheta][cholesky]")
{
    const int n = 5;
    auto A = spd_matrix(n, /*seed=*/123);
    auto A_orig = A;

    REQUIRE(tadah::hpo::detail::cholesky_upper(A.data(), n));

    // Verify U^T U == A_orig (only the upper triangle is canonical).
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i <= j; ++i) {
            double s = 0.0;
            for (int k = 0; k <= std::min(i, j); ++k)
                s += A[k + std::size_t(i) * n] * A[k + std::size_t(j) * n];
            INFO("i=" << i << " j=" << j);
            CHECK(close_rel(s, A_orig[i + std::size_t(j) * n], 1e-12, 1e-12));
        }
    }

    // Solve A x = b for a known x by computing b = A x.
    std::vector<double> x_true{0.7, -1.2, 0.4, 1.9, -0.3};
    std::vector<double> b(n, 0.0);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            b[i] += A_orig[i + std::size_t(j) * n] * x_true[j];

    std::vector<double> x = b;
    tadah::hpo::detail::cholesky_solve_uut(A.data(), n, x.data());
    for (int i = 0; i < n; ++i) {
        INFO("i=" << i);
        CHECK(close_rel(x[i], x_true[i], 1e-10, 1e-12));
    }
}

TEST_CASE("compute_dWdtheta matches FD on full re-solve (OLS, lambda=0)",
          "[HPO][dloss_dtheta][ift][parity]")
{
    // Mirrors test_linear_regressor_ift.cpp's analytical-IFT case but
    // routes through the HPO-side hand-rolled Cholesky chain. Phi(theta)
    // scales column 1 by exp(theta); dPhi/dtheta picks the same column.
    const std::size_t m = 16, n = 4;
    const std::size_t n_rows_aug = m + n;  // OLS still pads zero ridge rows
    const double      theta_0 = 0.3;

    std::mt19937 rng(7);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> base(m * n);
    for (auto& v : base) v = dist(rng);
    std::vector<double> T_data(m);
    {
        std::mt19937 rng2(8);
        for (auto& v : T_data) v = dist(rng2);
    }

    auto phi_aug_at = [&](double theta) {
        auto Phi = build_phi_rowmajor(base, m, n, theta);
        return augment_ridge(Phi, m, n, /*lambda=*/0.0, /*n_bias=*/0);
    };
    auto t_aug = augment_T(T_data, m, n);

    /* Forward fit at theta_0 — produce W* + factor for the analytical
     * pass. We use the HPO-side chol chain to build the factor; that is
     * the same path compute_dWdtheta will use, so this also covers the
     * "factor produced via HPO helpers" branch used by the on-demand
     * factor-reconstruction strategy in HPO_Host::analytical_gradient. */
    auto phi_aug_0 = phi_aug_at(theta_0);
    auto W_star = fit_via_hpo_chol(phi_aug_0, t_aug, n_rows_aug, n);

    /* Build dPhi/dtheta. Only column 1 is non-zero; equal to
     * phi_unaug[*, 1] (since phi_unaug[*, 1] = base[*, 1]*exp(theta_0)
     * and d/dtheta of that is itself). Augmentation rows are zero. */
    std::vector<double> dphi_aug(n_rows_aug * n, 0.0);
    {
        const double s = std::exp(theta_0);
        for (std::size_t r = 0; r < m; ++r)
            dphi_aug[r * n + 1] = base[r * n + 1] * s;
    }

    /* Plan: factor (computed by us, since compute_dWdtheta takes a
     * FactorHandle but here we don't have an OLS workspace — we
     * re-create the AtA + factor directly through compute_normal_equations
     * + cholesky_upper). Bind the factor's AtA buffer. */
    tadah::models::FactorHandle factor;
    {
        std::vector<double> AtA;
        tadah::hpo::detail::compute_normal_equations(
            phi_aug_0.data(), n_rows_aug, n, AtA);
        REQUIRE(tadah::hpo::detail::cholesky_upper(AtA.data(),
                                                   static_cast<int>(n)));
        factor.AtA   = std::move(AtA);
        factor.n     = static_cast<int>(n);
        factor.algo  = tadah::models::Algorithm::NormalEquations;
        factor.valid = true;
    }

    std::vector<tadah::hpo::ParamRef> prefs(1);
    prefs[0].theta_name = "THETA";
    prefs[0].arg_pos    = -1;
    prefs[0].flat_idx   = 0;

    std::vector<tadah::hpo::PerThetaInputs> per_theta(1);
    per_theta[0].dPhi_train_aug = dphi_aug.data();

    tadah::hpo::DLossDThetaPlan plan;
    plan.factor        = &factor;
    plan.weights       = &W_star;
    plan.phi_train_aug = phi_aug_0.data();
    plan.t_train_aug   = t_aug.data();
    plan.n_aug_rows    = n_rows_aug;
    plan.prefs         = &prefs;
    plan.per_theta     = &per_theta;

    std::vector<double> dW(prefs.size() * n, 0.0);
    std::vector<int>    needs_fd_unused;
    REQUIRE(tadah::hpo::detail::compute_dWdtheta(plan, dW, needs_fd_unused));
    REQUIRE(needs_fd_unused.empty());

    /* FD reference: solve at theta_0 ± h, finite-difference the weights. */
    const double h = 1e-5;
    auto Wp = fit_via_hpo_chol(phi_aug_at(theta_0 + h), t_aug, n_rows_aug, n);
    auto Wm = fit_via_hpo_chol(phi_aug_at(theta_0 - h), t_aug, n_rows_aug, n);
    for (std::size_t j = 0; j < n; ++j) {
        const double a = dW[j];
        const double b = (Wp[j] - Wm[j]) / (2.0 * h);
        INFO("j=" << j << " analytical=" << a << " FD=" << b);
        CHECK(close_rel(a, b, 1e-6, 1e-9));
    }
}

TEST_CASE("compute_dWdtheta matches FD on full re-solve (ridge, lambda>0)",
          "[HPO][dloss_dtheta][ift][parity][ridge]")
{
    const std::size_t m = 14, n = 3;
    const std::size_t n_rows_aug = m + n;
    const double      lambda  = 0.05;
    const double      theta_0 = -0.2;
    const std::size_t n_bias  = 0;

    std::mt19937 rng(200);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> base(m * n);
    for (auto& v : base) v = dist(rng);
    std::vector<double> T_data(m);
    {
        std::mt19937 rng2(201);
        for (auto& v : T_data) v = dist(rng2);
    }

    auto phi_aug_at = [&](double theta) {
        auto Phi = build_phi_rowmajor(base, m, n, theta);
        return augment_ridge(Phi, m, n, lambda, n_bias);
    };
    auto t_aug = augment_T(T_data, m, n);

    auto phi_aug_0 = phi_aug_at(theta_0);
    auto W_star = fit_via_hpo_chol(phi_aug_0, t_aug, n_rows_aug, n);

    /* Augmentation rows are theta-independent → dphi_aug is zero in the
     * bottom block. Only column 1 of the data block is non-zero. */
    std::vector<double> dphi_aug(n_rows_aug * n, 0.0);
    {
        const double s = std::exp(theta_0);
        for (std::size_t r = 0; r < m; ++r)
            dphi_aug[r * n + 1] = base[r * n + 1] * s;
    }

    tadah::models::FactorHandle factor;
    {
        std::vector<double> AtA;
        tadah::hpo::detail::compute_normal_equations(
            phi_aug_0.data(), n_rows_aug, n, AtA);
        REQUIRE(tadah::hpo::detail::cholesky_upper(AtA.data(),
                                                   static_cast<int>(n)));
        factor.AtA   = std::move(AtA);
        factor.n     = static_cast<int>(n);
        factor.algo  = tadah::models::Algorithm::NormalEquations;
        factor.valid = true;
    }

    std::vector<tadah::hpo::ParamRef> prefs(1);
    prefs[0].theta_name = "THETA";
    prefs[0].arg_pos    = -1;
    prefs[0].flat_idx   = 0;

    std::vector<tadah::hpo::PerThetaInputs> per_theta(1);
    per_theta[0].dPhi_train_aug = dphi_aug.data();

    tadah::hpo::DLossDThetaPlan plan;
    plan.factor        = &factor;
    plan.weights       = &W_star;
    plan.phi_train_aug = phi_aug_0.data();
    plan.t_train_aug   = t_aug.data();
    plan.n_aug_rows    = n_rows_aug;
    plan.prefs         = &prefs;
    plan.per_theta     = &per_theta;

    std::vector<double> dW(prefs.size() * n, 0.0);
    std::vector<int>    needs_fd_unused;
    REQUIRE(tadah::hpo::detail::compute_dWdtheta(plan, dW, needs_fd_unused));
    REQUIRE(needs_fd_unused.empty());

    const double h = 1e-5;
    auto Wp = fit_via_hpo_chol(phi_aug_at(theta_0 + h), t_aug, n_rows_aug, n);
    auto Wm = fit_via_hpo_chol(phi_aug_at(theta_0 - h), t_aug, n_rows_aug, n);
    for (std::size_t j = 0; j < n; ++j) {
        const double a = dW[j];
        const double b = (Wp[j] - Wm[j]) / (2.0 * h);
        INFO("j=" << j << " analytical=" << a << " FD=" << b);
        CHECK(close_rel(a, b, 1e-6, 1e-9));
    }
}

TEST_CASE("compute_dWdtheta routes nullptr dPhi indices to needs_fd",
          "[HPO][dloss_dtheta][ift][hybrid]")
{
    const std::size_t m = 6, n = 2;
    const std::size_t n_rows_aug = m + n;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> base(m * n);
    for (auto& v : base) v = dist(rng);
    std::vector<double> T_data(m);
    for (auto& v : T_data) v = dist(rng);

    auto phi = build_phi_rowmajor(base, m, n, /*theta=*/0.0);
    auto phi_aug = augment_ridge(phi, m, n, 0.0, 0);
    auto t_aug   = augment_T(T_data, m, n);
    auto W_star  = fit_via_hpo_chol(phi_aug, t_aug, n_rows_aug, n);

    tadah::models::FactorHandle factor;
    {
        std::vector<double> AtA;
        tadah::hpo::detail::compute_normal_equations(
            phi_aug.data(), n_rows_aug, n, AtA);
        REQUIRE(tadah::hpo::detail::cholesky_upper(AtA.data(),
                                                   static_cast<int>(n)));
        factor.AtA   = std::move(AtA);
        factor.n     = static_cast<int>(n);
        factor.algo  = tadah::models::Algorithm::NormalEquations;
        factor.valid = true;
    }

    /* Two thetas; only the first has analytical inputs. */
    std::vector<double> dphi(n_rows_aug * n, 0.0);
    for (std::size_t r = 0; r < m; ++r) dphi[r * n + 0] = 1.0;

    std::vector<tadah::hpo::ParamRef> prefs(2);
    prefs[0].theta_name = "A"; prefs[0].arg_pos = -1; prefs[0].flat_idx = 0;
    prefs[1].theta_name = "B"; prefs[1].arg_pos = -1; prefs[1].flat_idx = 1;

    std::vector<tadah::hpo::PerThetaInputs> per_theta(2);
    per_theta[0].dPhi_train_aug = dphi.data();
    per_theta[1].dPhi_train_aug = nullptr;     // → needs_fd

    tadah::hpo::DLossDThetaPlan plan;
    plan.factor        = &factor;
    plan.weights       = &W_star;
    plan.phi_train_aug = phi_aug.data();
    plan.t_train_aug   = t_aug.data();
    plan.n_aug_rows    = n_rows_aug;
    plan.prefs         = &prefs;
    plan.per_theta     = &per_theta;

    std::vector<double> dW(prefs.size() * n, 0.0);
    std::vector<int>    needs_fd;
    REQUIRE(tadah::hpo::detail::compute_dWdtheta(plan, dW, needs_fd));
    REQUIRE(needs_fd.size() == 1);
    CHECK(needs_fd[0] == 1);
    /* Row 1 of dW is left zero — caller's responsibility to FD it. */
    CHECK(dW[1 * n + 0] == 0.0);
    CHECK(dW[1 * n + 1] == 0.0);
}

namespace {

/* Local ScopedTempFile copy — test_hpo.cpp's lives in its own TU's
 * anonymous namespace and is therefore not visible here. Same body,
 * different file scope. */
struct ScopedTempFileLocal {
    std::filesystem::path path;
    explicit ScopedTempFileLocal(const std::string& body)
    : path(std::filesystem::temp_directory_path() /
           ("tadah_hpo_anal_" +
            std::to_string(std::chrono::steady_clock::now()
                                                .time_since_epoch().count()) +
            ".in"))
    {
        std::ofstream os(path);
        REQUIRE(os.good());
        os << body;
        os.close();
        REQUIRE(os.good());
    }
    ~ScopedTempFileLocal() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

/* Build a tiny truth StructureDB with N H-atom structures. Energies and
 * positions are filled deterministically; FORCE / STRESS are not used
 * (Phase 2A first cut: energy-only). natoms is fixed at 1 to make
 * EMAE math trivial: emae = mean(|E_truth - E_pred|) * 1000. */
StructureDB make_synthetic_truth(std::size_t N) {
    using namespace tadah::mlip;
    using tadah::core::Element;
    StructureDB db;
    db.dbidx = {0, N};
    db.structures.resize(N);
    for (std::size_t i = 0; i < N; ++i) {
        Structure& s = db.structures[i];
        s.add_atom(Atom(Element("H"),
                        0.1 * static_cast<double>(i + 1), 0.0, 0.0,
                        0.0, 0.0, 0.0));
        s.energy = 1.0 + 0.5 * static_cast<double>(i);
    }
    return db;
}

/* Build pred = Phi_val · W per validation row. In the energy-only
 * synthetic setup, each structure has one energy row; row i = struct i. */
StructureDB make_pred_from_phi(const StructureDB&        truth,
                               const std::vector<double>& phi_val,
                               std::size_t                n_cols,
                               const tadah::core::t_type& W)
{
    StructureDB pred = make_synthetic_truth(truth.size());  // start from truth
    for (std::size_t i = 0; i < truth.size(); ++i) {
        double e = 0.0;
        for (std::size_t c = 0; c < n_cols; ++c)
            e += phi_val[i * n_cols + c] * W[c];
        pred.structures[i].energy = e;
    }
    return pred;
}

/* Build phi_val(theta) row-major (n_struct × n_cols), column 2 scaled
 * by exp(theta). Uses a *different* column from the train-side scaling
 * (which moves col 1) so the validation prediction is not algebraically
 * compensated by W*(theta) — gives a non-trivial dy_pred/dtheta. */
std::vector<double>
build_phi_val(const std::vector<double>& base_rowmajor,
              std::size_t n_struct, std::size_t n_cols,
              double theta)
{
    std::vector<double> Phi = base_rowmajor;
    const double s = std::exp(theta);
    for (std::size_t i = 0; i < n_struct; ++i)
        Phi[i * n_cols + 2] *= s;
    return Phi;
}

/* Compute emae (post-scale, meV/atom) given truth/pred, mirroring
 * mlip::Analytics::calc_e_mae() composed with fill_validation's 1000x
 * scaling. With natoms=1 per structure: emae = mean(|E_t - E_p|) * 1000. */
double emae_from_dbs(const StructureDB& truth, const StructureDB& pred) {
    REQUIRE(truth.size() == pred.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < truth.size(); ++i) {
        const double n = static_cast<double>(truth(i).natoms());
        sum += std::abs(truth(i).energy - pred(i).energy) / n;
    }
    return 1000.0 * sum / static_cast<double>(truth.size());
}

}  // namespace

TEST_CASE("compute_dloss_dtheta chain rule matches FD on synthetic energy-only "
          "EMAE/L2 plan",
          "[HPO][dloss_dtheta][chain_rule][parity]")
{
    using namespace tadah::mlip;

    /* Periodic table must be initialised before atom symbol lookup. */
    tadah::core::PeriodicTable::initialize();

    /* Synthetic dimensions: 6 train rows + 4-augmented = 10 aug rows;
     * 4 weight columns; 3 validation structures (1 H atom each). */
    const std::size_t m = 6, n_cols = 4;
    const std::size_t n_aug = m + n_cols;
    const std::size_t n_val = 3;
    const double      theta_0 = 0.3;
    const double      lambda  = 0.0;  /* OLS for parity simplicity */

    /* HPOTargets via a temp config — minimal valid OPTIMIZER + LOSS L2 +
     * one PC_EMAE so use_emae=true and trg.emae / w_emae land. */
    ScopedTempFileLocal cfg(R"(OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 1
ENDOPTIMIZER
LOSS L2
PC_EMAE 0 1
)");
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    REQUIRE(trg.use_emae);
    CHECK(trg.emae == 0.0);
    CHECK(trg.w_emae == 1.0);

    /* Build a well-conditioned synthetic problem: pick a "true" W and
     * derive both T_train and truth-val energies from Phi(theta_0) · W.
     * Add small noise to truth-val so emae > 0 robustly (avoids the
     * |·| kink at perfect fit). The result: emae ~ 0.1 meV/atom, loss
     * ~ 0.01 meV²/atom² — well within double-precision FD reach. */
    std::mt19937 rng_train(11);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> base_train(m * n_cols);
    for (auto& v : base_train) v = dist(rng_train);

    std::mt19937 rng_val(22);
    std::vector<double> base_val(n_val * n_cols);
    for (auto& v : base_val) v = dist(rng_val);

    /* True W: deterministic, modest magnitude. */
    const std::vector<double> W_true = {0.3, -0.5, 0.2, 0.4};

    /* T_train = Phi_train(theta_0) · W_true (no noise → train fits W_true
     * exactly at theta_0). */
    std::vector<double> T_data(m, 0.0);
    {
        const double s = std::exp(theta_0);
        for (std::size_t r = 0; r < m; ++r) {
            for (std::size_t c = 0; c < n_cols; ++c) {
                const double phi_rc = (c == 1)
                    ? base_train[r * n_cols + c] * s
                    : base_train[r * n_cols + c];
                T_data[r] += phi_rc * W_true[c];
            }
        }
    }

    auto phi_train_aug_at = [&](double theta) {
        std::vector<double> Phi(n_aug * n_cols, 0.0);
        const double s = std::exp(theta);
        for (std::size_t r = 0; r < m; ++r)
            for (std::size_t c = 0; c < n_cols; ++c)
                Phi[r * n_cols + c] =
                    (c == 1) ? base_train[r * n_cols + c] * s
                             : base_train[r * n_cols + c];
        if (lambda > 0.0) {
            const double sl = std::sqrt(lambda);
            for (std::size_t k = 0; k < n_cols; ++k)
                Phi[(m + k) * n_cols + k] = sl;
        }
        return Phi;
    };
    auto t_aug = [&]() {
        std::vector<double> T(n_aug, 0.0);
        for (std::size_t i = 0; i < m; ++i) T[i] = T_data[i];
        return T;
    }();

    /* fit_via_hpo_chol from the IFT-test helpers above is reused. */
    auto fit_W = [&](double theta) {
        return fit_via_hpo_chol(phi_train_aug_at(theta), t_aug, n_aug, n_cols);
    };

    /* Build truth-val energies first so loss_at can capture them — the
     * test perturbs theta and must hold truth fixed (only the
     * predicted energies move). Truth energies are derived from
     * (Phi_val(theta_0) · W_true) plus small noise to keep emae > 0
     * (avoids the |r| kink) yet small (FD signal stays above
     * double-precision noise). */
    auto phi_val_0 = build_phi_val(base_val, n_val, n_cols, theta_0);
    auto truth_db  = make_synthetic_truth(n_val);
    {
        std::mt19937 rng_noise(33);
        std::uniform_real_distribution<double> noise(-0.001, 0.001);
        for (std::size_t i = 0; i < n_val; ++i) {
            double e = 0.0;
            for (std::size_t c = 0; c < n_cols; ++c)
                e += phi_val_0[i * n_cols + c] * W_true[c];
            truth_db.structures[i].energy = e + noise(rng_noise);
        }
    }

    /* loss(theta) — the function the analytical gradient differentiates.
     * truth is captured by reference so theta perturbation only moves
     * the prediction (matches the analytical chain rule's contract). */
    auto loss_at = [&](double theta) {
        const auto W   = fit_W(theta);
        const auto pv  = build_phi_val(base_val, n_val, n_cols, theta);
        const auto pred  = make_pred_from_phi(truth_db, pv, n_cols, W);
        const double emae = emae_from_dbs(truth_db, pred);
        /* Mirror fit_loss: r.loss_sum = LossFn(emae, target, w);
         * loss_count = 1 (only EMAE active); r.loss = r.loss_sum / 1. */
        return trg.globalLoss.loss(emae, trg.emae, trg.w_emae);
    };

    /* Forward state at theta_0 — populates the plan. */
    auto W_star    = fit_W(theta_0);
    auto phi_aug_0 = phi_train_aug_at(theta_0);
    auto pred_db   = make_pred_from_phi(truth_db, phi_val_0, n_cols, W_star);
    const double emae_0 = emae_from_dbs(truth_db, pred_db);

    /* dPhi/dtheta — column 1 of the data block; augmentation rows zero. */
    std::vector<double> dphi_train(n_aug * n_cols, 0.0);
    {
        const double s = std::exp(theta_0);
        for (std::size_t r = 0; r < m; ++r)
            dphi_train[r * n_cols + 1] = base_train[r * n_cols + 1] * s;
    }
    std::vector<double> dphi_val(n_val * n_cols, 0.0);
    {
        /* Phi_val scales col 2 by exp(theta) (see build_phi_val above);
         * dPhi_val/dtheta picks the same column (since d/dtheta of
         * base * exp(theta) = base * exp(theta) at the col 2 entries). */
        const double s = std::exp(theta_0);
        for (std::size_t r = 0; r < n_val; ++r)
            dphi_val[r * n_cols + 2] = base_val[r * n_cols + 2] * s;
    }

    /* Build the factor in the same way the future on-demand path will. */
    tadah::models::FactorHandle factor;
    {
        std::vector<double> AtA;
        tadah::hpo::detail::compute_normal_equations(
            phi_aug_0.data(), n_aug, n_cols, AtA);
        REQUIRE(tadah::hpo::detail::cholesky_upper(AtA.data(),
                                                   static_cast<int>(n_cols)));
        factor.AtA   = std::move(AtA);
        factor.n     = static_cast<int>(n_cols);
        factor.algo  = tadah::models::Algorithm::NormalEquations;
        factor.valid = true;
    }

    std::vector<tadah::hpo::ParamRef> prefs(1);
    prefs[0].theta_name = "THETA";  prefs[0].arg_pos = -1;
    prefs[0].flat_idx   = 0;

    std::vector<tadah::hpo::PerThetaInputs> per_theta(1);
    per_theta[0].dPhi_train_aug = dphi_train.data();
    per_theta[0].dPhi_val       = dphi_val.data();

    tadah::hpo::DLossDThetaPlan plan;
    plan.factor        = &factor;
    plan.weights       = &W_star;
    plan.phi_train_aug = phi_aug_0.data();
    plan.t_train_aug   = t_aug.data();
    plan.n_aug_rows    = n_aug;
    plan.phi_val       = phi_val_0.data();
    plan.n_val_rows    = n_val;
    plan.truth_val     = &truth_db;
    plan.pred_val      = &pred_db;
    plan.trg           = &trg;
    plan.metrics.emae  = emae_0;
    plan.loss_count    = 1;        /* one active metric: EMAE */
    plan.prefs         = &prefs;
    plan.per_theta     = &per_theta;

    struct VecAdapter {
        std::vector<double> data;
        long size() const { return static_cast<long>(data.size()); }
        void set_size(long n) { data.assign(static_cast<std::size_t>(n), 0.0); }
        double  operator()(long i) const { return data[static_cast<std::size_t>(i)]; }
        double& operator()(long i)       { return data[static_cast<std::size_t>(i)]; }
    };
    VecAdapter x;  x.set_size(1);  x.data = {theta_0};
    VecAdapter g;
    std::vector<int> needs_fd;
    REQUIRE(tadah::hpo::compute_dloss_dtheta(plan, x, g, needs_fd));
    REQUIRE(g.size() == 1);
    REQUIRE(needs_fd.empty());

    /* FD reference. */
    const double h = 1e-5;
    const double dl_fd = (loss_at(theta_0 + h) - loss_at(theta_0 - h))
                       / (2.0 * h);

    INFO("analytical=" << g(0) << "  FD=" << dl_fd);
    INFO("emae_0=" << emae_0
         << " loss(theta_0)=" << loss_at(theta_0)
         << " loss(theta_+h)=" << loss_at(theta_0 + h)
         << " loss(theta_-h)=" << loss_at(theta_0 - h));
    INFO("W_star[0]=" << W_star[0] << " W_star[1]=" << W_star[1]
         << " W_star[2]=" << W_star[2] << " W_star[3]=" << W_star[3]);
    /* Direct dmetric/dtheta and dW/dtheta for diagnosis. */
    {
        std::vector<double> dW_dbg;
        std::vector<int>    nfd_dbg;
        REQUIRE(tadah::hpo::detail::compute_dWdtheta(plan, dW_dbg, nfd_dbg));
        INFO("dW/dtheta=[" << dW_dbg[0] << "," << dW_dbg[1]
             << "," << dW_dbg[2] << "," << dW_dbg[3] << "]");
        for (std::size_t i = 0; i < n_val; ++i) {
            double dyp = 0.0;
            for (std::size_t c = 0; c < n_cols; ++c)
                dyp += dphi_val[i * n_cols + c] * W_star[c]
                     + phi_val_0[i * n_cols + c] * dW_dbg[c];
            INFO("cell " << i << " truth=" << truth_db(i).energy
                 << " pred=" << pred_db(i).energy
                 << " dy_pred/dtheta=" << dyp);
        }
    }
    /* Layer-3 tolerance per the plan: 1e-4 relative. */
    const double rel = std::abs(g(0) - dl_fd)
                     / std::max({std::abs(g(0)), std::abs(dl_fd), 1e-12});
    CHECK(rel < 1e-4);
}

/* ---------------------------------------------------------------------- *
 * Layer-3 live-trainer parity test
 *
 * Exercises compute_analytical_gradient_from_trainer end-to-end against
 * a hand-constructed Trainer. Avoids the cost of constructing a full
 * HPO_Host (which would need DBFILE inputs + a config.hpo) by:
 *
 *   - Building train/val StructureDBs in-memory (one 3-atom H cell each)
 *   - Running NNFinder on both
 *   - Constructing a Trainer from a minimal Context
 *   - Training via Trainer::train()
 *   - Building a tiny HPOTargets via a temp config (PC_EMAE 0 1, LOSS L2,
 *     OPTIM CGRID2B (1) <low> <high>)
 *
 * The FD reference re-trains a fresh Trainer at theta_0 ± h, runs predict
 * on the same val_stdb, computes EMAE, and central-differences the loss.
 * Tolerance: 1e-3 relative (loosened from 1e-4 to absorb the descriptor-
 * walker's wfactor scaling and Cholesky reconstruction round-off; the
 * synthetic chain-rule test above pins the math to 1e-4).
 * ---------------------------------------------------------------------- */

#include <tadah/core/element.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/trainer.h>

namespace {

using tadah::core::Context;
using tadah::core::Element;
using tadah::mlip::Atom;
using tadah::mlip::NNFinder;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::Trainer;

/* Minimal Context with just enough keys for Trainer to construct,
 * train, and predict on a single-species (H) D2_BP + Cut_Cos energy-only
 * setup with BF_Linear, OALGO=4, NORM=false, BIAS=false. */
Context make_minimal_trainer_context(const std::vector<double>& cgrid,
                                     const std::vector<double>& sgrid,
                                     double rcut)
{
    Context c;
    /* Species */
    c.add("ATOM",   "H");
    c.add("WATOM",  1);
    /* Descriptors */
    c.add("INIT2B", "true");
    c.add("INITMB", "false");
    c.add("TYPE2B", "D2_BP");
    c.add("TYPE2B", static_cast<int>(cgrid.size()));
    c.add("TYPE2B", static_cast<int>(sgrid.size()));
    c.add("TYPE2B", "H"); c.add("TYPE2B", "H");
    c.add("TYPEMB", "DM_Dummy");
    c.add("TYPEMB", "H"); c.add("TYPEMB", "H");
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < cgrid.size(); ++i) {
            if (i) ss << " ";
            ss << cgrid[i];
        }
        c.add("CGRID2B", ss.str());
    }
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < sgrid.size(); ++i) {
            if (i) ss << " ";
            ss << sgrid[i];
        }
        c.add("SGRID2B", ss.str());
    }
    c.add("RCTYPE2B", "Cut_Cos");
    c.add("RCTYPEMB", "Cut_Cos");
    c.add("RCUT2B",   rcut);                /* sets RCUT2BMAX automatically */
    c.add("RCUTMB",   rcut);
    c.add("DIMER",    "false");
    c.add("DIMER",    0);
    c.add("DIMER",    "false");
    /* Geometry-aware caps the host fills from optimap; here we hard-code
     * to the same rcut since the test doesn't optimise RCUT2B. */
    /* Output controls */
    c.add("FORCE",    "false");
    c.add("STRESS",   "false");
    c.add("EWEIGHT",  1.0);
    c.add("FWEIGHT",  1.0);
    c.add("SWEIGHT",  1.0);
    c.add("BIAS",     "false");
    c.add("NORM",     "false");
    c.add("VERBOSE",  0);
    /* Solver */
    c.add("MODEL",    "M_BLR");
    c.add("MODEL",    "BF_Linear");
    /* Small ridge keeps Phi^T Phi positive-definite — without it the
     * 4-row 4-column system from D2_BP[2,2] on a few 3-atom H cells is
     * singular under OALGO=4 (NormalEquations). */
    c.add("LAMBDA",   1e-3);
    c.add("OALGO",    4);
    return c;
}

/* Build a 3-atom H cell with shifted positions per index so train and
 * val are slightly different (so the fit doesn't algebraically collapse
 * to perfect identity at every theta — the algebraic-cancellation pitfall
 * we hit in the synthetic test). */
Structure make_h3_cell(double offset_x) {
    Structure st;
    st.cell.load(20.0, 0.0, 0.0,
                 0.0, 20.0, 0.0,
                 0.0, 0.0, 20.0);
    st.add_atom(Atom(Element("H"), 0.0 + offset_x, 0.0, 0.0, 0.0, 0.0, 0.0));
    st.add_atom(Atom(Element("H"), 1.6 + offset_x, 0.0, 0.0, 0.0, 0.0, 0.0));
    st.add_atom(Atom(Element("H"), 0.0 + offset_x, 2.1, 0.0, 0.0, 0.0, 0.0));
    return st;
}

/* Create a StructureDB of N H3 cells, each shifted by 0.05*i. truth-side
 * energies are filled by the caller (depend on θ via the descriptor
 * eval). */
StructureDB make_h3_stdb(std::size_t N, double rcut) {
    StructureDB db;
    db.dbidx = {0, N};
    db.structures.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        db.structures.push_back(make_h3_cell(0.05 * static_cast<double>(i)));
    }
    NNFinder finder;
    for (std::size_t i = 0; i < N; ++i) finder.calc(db.structures[i], rcut);
    return db;
}

/* Solve for predicted energies from the trained model on a StructureDB.
 * Mirrors what M_Tadah_Base::predict does for energy-only configs. */
std::vector<double> predict_energies(Trainer& host, const StructureDB& db) {
    std::vector<double> e(db.size(), 0.0);
    for (std::size_t s = 0; s < db.size(); ++s) {
        tadah::mlip::StDescriptors std_d = host.dc.calc(db(s));
        e[s] = host.model->epredict(std_d);
    }
    return e;
}

/* Compute scaled EMAE matching fill_validation: 1000 * mean(|truth-pred|/natoms). */
double emae_scaled(const StructureDB& truth,
                   const std::vector<double>& pred_e)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < truth.size(); ++i) {
        const double n = static_cast<double>(truth(i).natoms());
        sum += std::abs(truth(i).energy - pred_e[i]) / n;
    }
    return 1000.0 * sum / static_cast<double>(truth.size());
}

}  // namespace

TEST_CASE("compute_analytical_gradient_from_trainer matches FD on live trainer "
          "(D2_BP + BF_Linear + EMAE/L2)",
          "[HPO][analytical_gradient][live_trainer][parity]")
{
    using tadah::mlip::memory::MLIPWorkspaceManager;
    tadah::core::PeriodicTable::initialize();

    /* Synthetic problem dimensions. */
    const std::vector<double> cgrid_0 = {0.5, 1.4};
    const std::vector<double> sgrid_0 = {3.0, 5.0};
    const double rcut    = 3.0;
    const double theta_0 = cgrid_0[0];   /* the parameter under test */

    /* Train and val StructureDBs. */
    auto train_stdb = make_h3_stdb(/*N=*/4, rcut);
    auto val_stdb   = make_h3_stdb(/*N=*/3, rcut);
    /* Different offset for val so its descriptor distribution differs
     * from train (otherwise W*(θ) algebraically compensates the val Phi
     * and the gradient signal vanishes — same pitfall as the synthetic
     * test). Shift each val structure by an extra 0.07. */
    for (auto& s : val_stdb.structures) {
        for (auto& a : s.atoms) a.position[1] += 0.07;
    }
    {
        NNFinder finder;
        for (auto& s : val_stdb.structures) finder.calc(s, rcut);
    }

    /* Truth energies for train and val. Use small fixed values so emae
     * is non-trivial without driving Phi^T Phi to instability. The
     * particular numbers are arbitrary — what matters is that they are
     * not algebraically aligned with Phi(θ_0)·W*. */
    for (std::size_t i = 0; i < train_stdb.size(); ++i)
        train_stdb.structures[i].energy = 0.05 * static_cast<double>(i + 1);
    for (std::size_t i = 0; i < val_stdb.size(); ++i)
        val_stdb.structures[i].energy = 0.07 * static_cast<double>(i + 1);

    /* Single trainer instance for the analytical pass. Stack-allocated;
     * destroyed at scope exit. */
    MLIPWorkspaceManager ws_main;
    Context c_main = make_minimal_trainer_context(cgrid_0, sgrid_0, rcut);
    Trainer host(c_main, ws_main);
    host.train(train_stdb);

    /* Build HPOTargets with EMAE only + OPTIM CGRID2B (1) so prefs has
     * a single entry pointing at cgrid[0]. */
    ScopedTempFileLocal cfg(R"(OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 1
ENDOPTIMIZER
LOSS L2
PC_EMAE 0 1
OPTIM CGRID2B (1) 0.0 4.0
)");
    /* HPOTargets's OPTIM parser cross-checks the optim variable against
     * the master Context — `OPTIM CGRID2B (1) ...` requires CGRID2B
     * already declared. Reuse the trainer-side context so the targets
     * parser sees a consistent view. */
    Context ctx_for_targets =
        make_minimal_trainer_context(cgrid_0, sgrid_0, rcut);
    tadah::hpo::HPOTargets trg(ctx_for_targets, cfg.path.string());
    REQUIRE(trg.use_emae);
    REQUIRE(trg.optimap.count("CGRID2B"));

    /* Build prefs flat table. */
    auto prefs = tadah::hpo::make_param_ref_table(trg);
    REQUIRE(prefs.size() == 1);
    CHECK(prefs[0].theta_name == "CGRID2B");
    CHECK(prefs[0].arg_pos == 1);

    /* Analytical gradient. */
    std::vector<double> g_anal;
    std::vector<int>    needs_fd;
    REQUIRE(tadah::hpo::compute_analytical_gradient_from_trainer(
        host, train_stdb, val_stdb, trg, prefs, g_anal, needs_fd));
    REQUIRE(g_anal.size() == 1);
    REQUIRE(needs_fd.empty());

    /* FD reference: re-fit at θ_0 ± h and compute the loss. */
    auto loss_at = [&](double theta_val) {
        std::vector<double> cg = cgrid_0;
        cg[0] = theta_val;
        MLIPWorkspaceManager ws_h;
        Context c_h = make_minimal_trainer_context(cg, sgrid_0, rcut);
        Trainer t_h(c_h, ws_h);
        t_h.train(train_stdb);
        const auto pred_e = predict_energies(t_h, val_stdb);
        const double emae = emae_scaled(val_stdb, pred_e);
        return trg.globalLoss.loss(emae, trg.emae, trg.w_emae);
    };

    const double h = 1e-5;
    const double dl_fd = (loss_at(theta_0 + h) - loss_at(theta_0 - h))
                       / (2.0 * h);

    INFO("analytical=" << g_anal[0] << "  FD=" << dl_fd);
    INFO("loss(theta_0)=" << loss_at(theta_0));

    const double rel = std::abs(g_anal[0] - dl_fd)
                     / std::max({std::abs(g_anal[0]), std::abs(dl_fd), 1e-12});
    CHECK(rel < 1e-3);
}

namespace {

/* Variant of make_minimal_trainer_context that swaps the master model
 * from M_BLR/BF_Linear to M_KRR/Kern_Linear/modelN=1. Everything else is
 * identical — Kern_Linear's Phi reduces to sum-of-AEDs (see
 * DM_Kern_Linear::calc_phi_energy_row) and M_KRR_Train::train skips EKM
 * when kernel.get_label() == "Kern_Linear", so the train-side math is
 * byte-identical to M_BLR/BF_Linear; only the model-factory branch and
 * the predict-side `kernel.epredict` wrapper change. The HPO analytical
 * gate accepts both labels equivalently (Phase 2B B-HPO lift). */
Context make_minimal_trainer_context_kern_linear(
    const std::vector<double>& cgrid,
    const std::vector<double>& sgrid,
    double rcut)
{
    Context c = make_minimal_trainer_context(cgrid, sgrid, rcut);
    /* Swap MODEL tokens: M_BLR/BF_Linear → M_KRR/Kern_Linear/modelN=1. */
    c.remove("MODEL");
    c.add("MODEL", "M_KRR");
    c.add("MODEL", "Kern_Linear");
    c.add("MODEL", 1);
    return c;
}

}  // namespace

TEST_CASE("compute_analytical_gradient_from_trainer matches FD on live trainer "
          "(D2_BP + Kern_Linear + EMAE/L2) — Phase 2B (B-HPO)",
          "[HPO][analytical_gradient][live_trainer][parity][kern_linear]")
{
    using tadah::mlip::memory::MLIPWorkspaceManager;
    tadah::core::PeriodicTable::initialize();

    /* Same shape as the BF_Linear variant — Kern_Linear's design matrix
     * is identical (sum-of-AEDs without EKM), so we expect identical
     * gradient numerics within the FD tolerance. */
    const std::vector<double> cgrid_0 = {0.5, 1.4};
    const std::vector<double> sgrid_0 = {3.0, 5.0};
    const double rcut    = 3.0;
    const double theta_0 = cgrid_0[0];

    auto train_stdb = make_h3_stdb(/*N=*/4, rcut);
    auto val_stdb   = make_h3_stdb(/*N=*/3, rcut);
    for (auto& s : val_stdb.structures) {
        for (auto& a : s.atoms) a.position[1] += 0.07;
    }
    {
        NNFinder finder;
        for (auto& s : val_stdb.structures) finder.calc(s, rcut);
    }
    for (std::size_t i = 0; i < train_stdb.size(); ++i)
        train_stdb.structures[i].energy = 0.05 * static_cast<double>(i + 1);
    for (std::size_t i = 0; i < val_stdb.size(); ++i)
        val_stdb.structures[i].energy = 0.07 * static_cast<double>(i + 1);

    MLIPWorkspaceManager ws_main;
    Context c_main = make_minimal_trainer_context_kern_linear(cgrid_0, sgrid_0, rcut);
    Trainer host(c_main, ws_main);
    host.train(train_stdb);

    /* Sanity: MODEL[1] must be the lifted "Kern_Linear" gate value. */
    REQUIRE(host.context.template get<std::string>("MODEL", 1) == "Kern_Linear");

    ScopedTempFileLocal cfg(R"(OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 1
ENDOPTIMIZER
LOSS L2
PC_EMAE 0 1
OPTIM CGRID2B (1) 0.0 4.0
)");
    Context ctx_for_targets =
        make_minimal_trainer_context_kern_linear(cgrid_0, sgrid_0, rcut);
    tadah::hpo::HPOTargets trg(ctx_for_targets, cfg.path.string());
    REQUIRE(trg.use_emae);
    REQUIRE(trg.optimap.count("CGRID2B"));

    auto prefs = tadah::hpo::make_param_ref_table(trg);
    REQUIRE(prefs.size() == 1);
    CHECK(prefs[0].theta_name == "CGRID2B");
    CHECK(prefs[0].arg_pos == 1);

    std::vector<double> g_anal;
    std::vector<int>    needs_fd;
    REQUIRE(tadah::hpo::compute_analytical_gradient_from_trainer(
        host, train_stdb, val_stdb, trg, prefs, g_anal, needs_fd));
    REQUIRE(g_anal.size() == 1);
    /* Critical: under the new gate, Kern_Linear must NOT fall back to FD
     * — analytical path must produce a real value. */
    REQUIRE(needs_fd.empty());

    /* FD reference: re-fit M_KRR/Kern_Linear at θ_0 ± h. */
    auto loss_at = [&](double theta_val) {
        std::vector<double> cg = cgrid_0;
        cg[0] = theta_val;
        MLIPWorkspaceManager ws_h;
        Context c_h = make_minimal_trainer_context_kern_linear(cg, sgrid_0, rcut);
        Trainer t_h(c_h, ws_h);
        t_h.train(train_stdb);
        const auto pred_e = predict_energies(t_h, val_stdb);
        const double emae = emae_scaled(val_stdb, pred_e);
        return trg.globalLoss.loss(emae, trg.emae, trg.w_emae);
    };

    const double h = 1e-5;
    const double dl_fd = (loss_at(theta_0 + h) - loss_at(theta_0 - h))
                       / (2.0 * h);

    INFO("analytical=" << g_anal[0] << "  FD=" << dl_fd);
    INFO("loss(theta_0)=" << loss_at(theta_0));

    const double rel = std::abs(g_anal[0] - dl_fd)
                     / std::max({std::abs(g_anal[0]), std::abs(dl_fd), 1e-12});
    CHECK(rel < 1e-3);
}

TEST_CASE("compute_analytical_gradient_from_trainer falls back for non-linear kernel "
          "(MODEL Kern_RBF) — Phase 2B (B-HPO) gate",
          "[HPO][analytical_gradient][live_trainer][kern_rbf][routing]")
{
    using tadah::mlip::memory::MLIPWorkspaceManager;
    tadah::core::PeriodicTable::initialize();

    /* Smoke test of the gate routing: with MODEL[1] = "Kern_RBF" the
     * helper must route every θ index onto needs_fd (no analytical
     * support yet — EKM projection differentiation lands in Phase 2C).
     * This documents the current restriction and catches regressions
     * where someone forgets to add the new kernel to the gate. */
    const std::vector<double> cgrid_0 = {0.5, 1.4};
    const std::vector<double> sgrid_0 = {3.0, 5.0};
    const double rcut = 3.0;

    auto train_stdb = make_h3_stdb(/*N=*/4, rcut);
    auto val_stdb   = make_h3_stdb(/*N=*/3, rcut);
    for (std::size_t i = 0; i < train_stdb.size(); ++i)
        train_stdb.structures[i].energy = 0.05 * static_cast<double>(i + 1);
    for (std::size_t i = 0; i < val_stdb.size(); ++i)
        val_stdb.structures[i].energy = 0.07 * static_cast<double>(i + 1);

    /* Build a minimal context — but don't actually construct an M_KRR
     * with Kern_RBF (would require SBASIS / EKM setup). Instead, take
     * a BF_Linear-trained Trainer and rewrite host.context's MODEL[1]
     * after the fact to simulate the gate-routing check. The helper
     * reads ctx.get<std::string>("MODEL", 1) directly and falls back
     * before invoking any walker, so the actual model object doesn't
     * have to match the relabeled string. */
    MLIPWorkspaceManager ws;
    Context c = make_minimal_trainer_context(cgrid_0, sgrid_0, rcut);
    Trainer host(c, ws);
    host.train(train_stdb);
    host.context.remove("MODEL");
    host.context.add("MODEL", "M_KRR");
    host.context.add("MODEL", "Kern_RBF");
    host.context.add("MODEL", 1);

    ScopedTempFileLocal cfg(R"(OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 1
ENDOPTIMIZER
LOSS L2
PC_EMAE 0 1
OPTIM CGRID2B (1) 0.0 4.0
)");
    Context ctx_for_targets =
        make_minimal_trainer_context(cgrid_0, sgrid_0, rcut);
    tadah::hpo::HPOTargets trg(ctx_for_targets, cfg.path.string());
    auto prefs = tadah::hpo::make_param_ref_table(trg);
    REQUIRE(prefs.size() == 1);

    std::vector<double> g_anal;
    std::vector<int>    needs_fd;
    REQUIRE(tadah::hpo::compute_analytical_gradient_from_trainer(
        host, train_stdb, val_stdb, trg, prefs, g_anal, needs_fd));
    /* All indices fell back to FD. */
    REQUIRE(needs_fd.size() == prefs.size());
    /* Helper still pre-zeros g_out before falling back; the caller's
     * HYBRID FD pass overwrites these entries. */
    for (double v : g_anal) CHECK(v == 0.0);
}

/* ------------------------------------------------------------------------
 * Phase 2C (C-step) — FORCE=true live-trainer parity for the analytical
 * gradient. With FORCE=true the trainer fits W* against both energy and
 * force targets via the augmented Phi_train (1 energy + 3·natoms force
 * rows per struct, then ridge). The validation side stays energy-only
 * (use_frmse=false), so the loss is EMAE/L2 — the same shape as the
 * BF_Linear baseline test above, but exercising the force-row IFT chain.
 * ------------------------------------------------------------------------ */

namespace {

/* Variant of make_minimal_trainer_context with FORCE=true. The trainer's
 * DesignMatrix builds force rows, fills force targets, and OLS fits W*
 * against the joint energy+force least-squares. */
Context make_minimal_trainer_context_force(
    const std::vector<double>& cgrid,
    const std::vector<double>& sgrid,
    double rcut)
{
    Context c = make_minimal_trainer_context(cgrid, sgrid, rcut);
    c.remove("FORCE");
    c.add("FORCE", "true");
    return c;
}

/* Build a 3-atom H cell with non-zero force targets per atom. Forces
 * decoupled from positions — they are independent training targets so
 * W*(θ) has to compromise between fitting energy and force, which is
 * what we want to exercise the force-row IFT chain. */
Structure make_h3_cell_with_forces(double offset_x, double force_scale) {
    Structure st;
    st.cell.load(20.0, 0.0, 0.0,
                 0.0, 20.0, 0.0,
                 0.0, 0.0, 20.0);
    /* Atom positions same as make_h3_cell. Last 3 ctor args are
     * (fx, fy, fz). Force pattern: per-atom non-zero, modest magnitude. */
    st.add_atom(Atom(Element("H"), 0.0 + offset_x, 0.0, 0.0,
                     +0.10 * force_scale, -0.05 * force_scale, 0.0));
    st.add_atom(Atom(Element("H"), 1.6 + offset_x, 0.0, 0.0,
                     -0.08 * force_scale, +0.03 * force_scale, 0.0));
    st.add_atom(Atom(Element("H"), 0.0 + offset_x, 2.1, 0.0,
                     -0.02 * force_scale, +0.02 * force_scale, 0.0));
    return st;
}

StructureDB make_h3_force_stdb(std::size_t N, double rcut, double force_scale) {
    StructureDB db;
    db.dbidx = {0, N};
    db.structures.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        db.structures.push_back(
            make_h3_cell_with_forces(0.05 * static_cast<double>(i), force_scale));
    }
    NNFinder finder;
    for (std::size_t i = 0; i < N; ++i) finder.calc(db.structures[i], rcut);
    return db;
}

}  // namespace

TEST_CASE("compute_analytical_gradient_from_trainer matches FD on live trainer "
          "(D2_BP + BF_Linear + FORCE=true + EMAE/L2) — Phase 2C (C-step)",
          "[HPO][analytical_gradient][live_trainer][parity][force]")
{
    using tadah::mlip::memory::MLIPWorkspaceManager;
    tadah::core::PeriodicTable::initialize();

    const std::vector<double> cgrid_0 = {0.5, 1.4};
    const std::vector<double> sgrid_0 = {3.0, 5.0};
    const double rcut    = 3.0;
    const double theta_0 = cgrid_0[0];

    auto train_stdb = make_h3_force_stdb(/*N=*/4, rcut, /*force_scale=*/1.0);
    auto val_stdb   = make_h3_force_stdb(/*N=*/3, rcut, /*force_scale=*/0.5);
    for (auto& s : val_stdb.structures) {
        for (auto& a : s.atoms) a.position[1] += 0.07;
    }
    {
        NNFinder finder;
        for (auto& s : val_stdb.structures) finder.calc(s, rcut);
    }
    for (std::size_t i = 0; i < train_stdb.size(); ++i)
        train_stdb.structures[i].energy = 0.05 * static_cast<double>(i + 1);
    for (std::size_t i = 0; i < val_stdb.size(); ++i)
        val_stdb.structures[i].energy = 0.07 * static_cast<double>(i + 1);

    MLIPWorkspaceManager ws_main;
    Context c_main = make_minimal_trainer_context_force(cgrid_0, sgrid_0, rcut);
    Trainer host(c_main, ws_main);
    host.train(train_stdb);

    /* Sanity: FORCE must be set in the trainer context. */
    REQUIRE(host.context.template get<bool>("FORCE"));

    ScopedTempFileLocal cfg(R"(OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 1
ENDOPTIMIZER
LOSS L2
PC_EMAE 0 1
OPTIM CGRID2B (1) 0.0 4.0
)");
    Context ctx_for_targets =
        make_minimal_trainer_context_force(cgrid_0, sgrid_0, rcut);
    tadah::hpo::HPOTargets trg(ctx_for_targets, cfg.path.string());
    REQUIRE(trg.use_emae);
    /* No use_frmse — val side stays energy-only in this first cut. */
    REQUIRE_FALSE(trg.use_frmse);

    auto prefs = tadah::hpo::make_param_ref_table(trg);
    REQUIRE(prefs.size() == 1);

    std::vector<double> g_anal;
    std::vector<int>    needs_fd;
    REQUIRE(tadah::hpo::compute_analytical_gradient_from_trainer(
        host, train_stdb, val_stdb, trg, prefs, g_anal, needs_fd));
    REQUIRE(g_anal.size() == 1);
    /* Critical: with FORCE=true the gate must now accept and produce
     * an analytical value — no FD fallback for D2_BP CGRID2B[1]. */
    REQUIRE(needs_fd.empty());

    auto loss_at = [&](double theta_val) {
        std::vector<double> cg = cgrid_0;
        cg[0] = theta_val;
        MLIPWorkspaceManager ws_h;
        Context c_h = make_minimal_trainer_context_force(cg, sgrid_0, rcut);
        Trainer t_h(c_h, ws_h);
        t_h.train(train_stdb);
        const auto pred_e = predict_energies(t_h, val_stdb);
        const double emae = emae_scaled(val_stdb, pred_e);
        return trg.globalLoss.loss(emae, trg.emae, trg.w_emae);
    };

    const double h = 1e-5;
    const double dl_fd = (loss_at(theta_0 + h) - loss_at(theta_0 - h))
                       / (2.0 * h);

    INFO("analytical=" << g_anal[0] << "  FD=" << dl_fd);
    INFO("loss(theta_0)=" << loss_at(theta_0));

    const double rel = std::abs(g_anal[0] - dl_fd)
                     / std::max({std::abs(g_anal[0]), std::abs(dl_fd), 1e-12});
    CHECK(rel < 1e-3);
}

/* ------------------------------------------------------------------------
 * Phase 2B (C-val) — use_frmse live-trainer parity for the analytical
 * gradient. FORCE=true on the trainer (joint energy+force LSQ), AND
 * use_frmse=true on the validation side so the loss aggregates a force
 * RMSE term in addition to EMAE. The validation Phi gains 3·natoms
 * force rows per struct and the orchestrator's metric chain rule loops
 * over FORCE cells via val_cell_to_row + analytics_grad::frmse.
 * ------------------------------------------------------------------------ */

namespace {

/* Compute scaled FRMSE matching fill_validation: 1.0 * rmse(F_pred - F_truth)
 * over all 3·natoms cells. */
double frmse_scaled(const StructureDB& truth, const StructureDB& pred) {
    std::size_t N_F = 0;
    double sumsq = 0.0;
    for (std::size_t s = 0; s < truth.size(); ++s) {
        for (std::size_t a = 0; a < truth(s).natoms(); ++a) {
            for (std::size_t k = 0; k < 3; ++k) {
                const double d = truth(s).atoms[a].force(k)
                               - pred (s).atoms[a].force(k);
                sumsq += d * d;
                ++N_F;
            }
        }
    }
    if (N_F == 0) return 0.0;
    return 1.0 * std::sqrt(sumsq / static_cast<double>(N_F));
}

/* Full predict (energy + force) for a trained host on a StructureDB.
 * Mirrors what fill_validation feeds into Analytics: pred_val with
 * atom.force populated. */
StructureDB predict_full(Trainer& host, const StructureDB& db) {
    StructureDB pred;
    pred.dbidx = db.dbidx;
    pred.structures.resize(db.size());
    for (std::size_t s = 0; s < db.size(); ++s) {
        const auto& st = db(s);
        tadah::mlip::StDescriptors std_d = host.dc.calc(st);
        pred.structures[s] = host.model->predict(host.context, std_d, st);
    }
    return pred;
}

}  // namespace

TEST_CASE("compute_analytical_gradient_from_trainer matches FD on live trainer "
          "(D2_BP + BF_Linear + FORCE=true + EMAE+FRMSE/L2) — Phase 2B (C-val)",
          "[HPO][analytical_gradient][live_trainer][parity][force][frmse]")
{
    using tadah::mlip::memory::MLIPWorkspaceManager;
    tadah::core::PeriodicTable::initialize();

    const std::vector<double> cgrid_0 = {0.5, 1.4};
    const std::vector<double> sgrid_0 = {3.0, 5.0};
    const double rcut    = 3.0;
    const double theta_0 = cgrid_0[0];

    auto train_stdb = make_h3_force_stdb(/*N=*/4, rcut, /*force_scale=*/1.0);
    auto val_stdb   = make_h3_force_stdb(/*N=*/3, rcut, /*force_scale=*/0.5);
    for (auto& s : val_stdb.structures) {
        for (auto& a : s.atoms) a.position[1] += 0.07;
    }
    {
        NNFinder finder;
        for (auto& s : val_stdb.structures) finder.calc(s, rcut);
    }
    for (std::size_t i = 0; i < train_stdb.size(); ++i)
        train_stdb.structures[i].energy = 0.05 * static_cast<double>(i + 1);
    for (std::size_t i = 0; i < val_stdb.size(); ++i)
        val_stdb.structures[i].energy = 0.07 * static_cast<double>(i + 1);

    MLIPWorkspaceManager ws_main;
    Context c_main = make_minimal_trainer_context_force(cgrid_0, sgrid_0, rcut);
    Trainer host(c_main, ws_main);
    host.train(train_stdb);

    REQUIRE(host.context.template get<bool>("FORCE"));

    /* PC_FRMSE turns on use_frmse in HPOTargets. The configured target
     * is 0 (typical for an RMSE: drive toward 0) and weight 1. */
    ScopedTempFileLocal cfg(R"(OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 1
ENDOPTIMIZER
LOSS L2
PC_EMAE 0 1
PC_FRMSE 0 1
OPTIM CGRID2B (1) 0.0 4.0
)");
    Context ctx_for_targets =
        make_minimal_trainer_context_force(cgrid_0, sgrid_0, rcut);
    tadah::hpo::HPOTargets trg(ctx_for_targets, cfg.path.string());
    REQUIRE(trg.use_emae);
    REQUIRE(trg.use_frmse);

    auto prefs = tadah::hpo::make_param_ref_table(trg);
    REQUIRE(prefs.size() == 1);

    std::vector<double> g_anal;
    std::vector<int>    needs_fd;
    REQUIRE(tadah::hpo::compute_analytical_gradient_from_trainer(
        host, train_stdb, val_stdb, trg, prefs, g_anal, needs_fd));
    REQUIRE(g_anal.size() == 1);
    /* Gate must accept use_frmse=true now — no FD fallback. */
    REQUIRE(needs_fd.empty());

    auto loss_at = [&](double theta_val) {
        std::vector<double> cg = cgrid_0;
        cg[0] = theta_val;
        MLIPWorkspaceManager ws_h;
        Context c_h = make_minimal_trainer_context_force(cg, sgrid_0, rcut);
        Trainer t_h(c_h, ws_h);
        t_h.train(train_stdb);
        const auto pred_full = predict_full(t_h, val_stdb);
        std::vector<double> pred_e(val_stdb.size(), 0.0);
        for (std::size_t i = 0; i < val_stdb.size(); ++i)
            pred_e[i] = pred_full(i).energy;
        const double emae  = emae_scaled (val_stdb, pred_e);
        const double frmse = frmse_scaled(val_stdb, pred_full);
        /* Mirror fit_loss: sum loss(metric, target, w) over active
         * metrics, then divide by loss_count. Two metrics here. */
        const double l = trg.globalLoss.loss(emae,  trg.emae,  trg.w_emae)
                       + trg.globalLoss.loss(frmse, trg.frmse_targets.global().target, trg.frmse_targets.global().weight);
        return l / 2.0;
    };

    const double h = 1e-5;
    const double dl_fd = (loss_at(theta_0 + h) - loss_at(theta_0 - h))
                       / (2.0 * h);

    INFO("analytical=" << g_anal[0] << "  FD=" << dl_fd);
    INFO("loss(theta_0)=" << loss_at(theta_0));

    const double rel = std::abs(g_anal[0] - dl_fd)
                     / std::max({std::abs(g_anal[0]), std::abs(dl_fd), 1e-12});
    CHECK(rel < 1e-3);
}

/* ------------------------------------------------------------------------
 * P2 (PROMPT_NEXT_NLOPT_LBFGS.md / Option A) — PC_LAMMPS falls back to FD
 *
 * sims_loss from LAMMPS scripts is not differentiable analytically (LAMMPS
 * is opaque to the chain rule). The orchestrator's gate detects
 * !trg.LS.empty() and routes every theta index to needs_fd so HYBRID's
 * masked FD pass picks up both fit_loss and sims_loss components
 * consistently. Documented as Option A — partial-analytical (Option B,
 * Phase 2C) is the natural follow-up.
 * ------------------------------------------------------------------------ */
TEST_CASE("compute_analytical_gradient_from_trainer falls back when "
          "trg.LS is non-empty (PC_LAMMPS Option A)",
          "[HPO][analytical_gradient][live_trainer][lammps_fallback]")
{
    using tadah::mlip::memory::MLIPWorkspaceManager;
    tadah::core::PeriodicTable::initialize();

    /* Use the same single-arm D2_BP setup that the BF_Linear baseline
     * test passes on — the gate must override an otherwise-accepted
     * config when LS is non-empty. */
    const std::vector<double> cgrid_0 = {0.5, 1.4};
    const std::vector<double> sgrid_0 = {3.0, 5.0};
    const double rcut    = 3.0;

    auto train_stdb = make_h3_stdb(/*N=*/4, rcut);
    auto val_stdb   = make_h3_stdb(/*N=*/3, rcut);
    for (auto& s : val_stdb.structures) {
        for (auto& a : s.atoms) a.position[1] += 0.07;
    }
    {
        NNFinder finder;
        for (auto& s : val_stdb.structures) finder.calc(s, rcut);
    }
    for (std::size_t i = 0; i < train_stdb.size(); ++i)
        train_stdb.structures[i].energy = 0.05 * static_cast<double>(i + 1);
    for (std::size_t i = 0; i < val_stdb.size(); ++i)
        val_stdb.structures[i].energy = 0.07 * static_cast<double>(i + 1);

    MLIPWorkspaceManager ws_main;
    Context c_main = make_minimal_trainer_context(cgrid_0, sgrid_0, rcut);
    Trainer host(c_main, ws_main);
    host.train(train_stdb);

    ScopedTempFileLocal cfg(R"(OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 1
ENDOPTIMIZER
LOSS L2
PC_EMAE 0 1
OPTIM CGRID2B (1) 0.0 4.0
)");
    Context ctx_for_targets =
        make_minimal_trainer_context(cgrid_0, sgrid_0, rcut);
    tadah::hpo::HPOTargets trg(ctx_for_targets, cfg.path.string());
    REQUIRE(trg.use_emae);

    /* Inject a synthetic LAMMPS sim into the targets so the gate fires.
     * Default-constructed LammpsSim is fine — the helper never actually
     * runs LAMMPS; it only checks LS.empty(). */
    REQUIRE(trg.LS.empty());
    trg.LS.emplace_back();
    REQUIRE_FALSE(trg.LS.empty());

    auto prefs = tadah::hpo::make_param_ref_table(trg);
    REQUIRE(prefs.size() == 1);

    std::vector<double> g_anal;
    std::vector<int>    needs_fd;
    REQUIRE(tadah::hpo::compute_analytical_gradient_from_trainer(
        host, train_stdb, val_stdb, trg, prefs, g_anal, needs_fd));

    /* Critical: every theta index must be on needs_fd — the gate must
     * not silently compute an analytical fit_loss gradient while
     * dropping the sims_loss contribution. */
    REQUIRE(g_anal.size() == prefs.size());
    REQUIRE(needs_fd.size() == prefs.size());
    for (std::size_t i = 0; i < prefs.size(); ++i) {
        CHECK(g_anal[i] == 0.0);
        CHECK(needs_fd[i] == static_cast<int>(prefs[i].flat_idx));
    }
}

/* ------------------------------------------------------------------------
 * Phase 3.7B — `compute_analytical_gradient_from_trainer` accepts a
 * SimsGradProvider callback and folds sims_loss into the analytical
 * gradient when one is supplied. The null-provider path remains the
 * pre-3.7B fall_back_all_fd (covered by the Option A test above);
 * here we verify both halves of the new branch:
 *
 *  (1) provider supplied + zero-observable input ⇒ behaves like the
 *      empty-trg.LS path: needs_fd empty, g_anal == fit-only gradient.
 *      Demonstrates the wiring runs compute_dloss_dtheta as before
 *      and that compute_sims_grad_dWchain adds nothing when there is
 *      nothing to add.
 *
 *  (2) provider supplied + one cohesive-E observable ⇒ needs_fd empty,
 *      g_anal differs from (1) by a measurable amount (the sims_grad
 *      contribution). The exact sign/magnitude depends on the live
 *      trainer's dW/dθ and the synthetic descriptor inputs; we assert
 *      the difference is non-zero and finite.
 * ------------------------------------------------------------------------ */
TEST_CASE("compute_analytical_gradient_from_trainer wires the "
          "SimsGradProvider into compute_sims_grad_dWchain (Phase 3.7B)",
          "[HPO][analytical_gradient][live_trainer][sims_grad_provider]")
{
    using tadah::mlip::memory::MLIPWorkspaceManager;
    tadah::core::PeriodicTable::initialize();

    /* Same trainer scaffolding as the Option A test. */
    const std::vector<double> cgrid_0 = {0.5, 1.4};
    const std::vector<double> sgrid_0 = {3.0, 5.0};
    const double rcut = 3.0;

    auto train_stdb = make_h3_stdb(/*N=*/4, rcut);
    auto val_stdb   = make_h3_stdb(/*N=*/3, rcut);
    for (auto& s : val_stdb.structures) {
        for (auto& a : s.atoms) a.position[1] += 0.07;
    }
    {
        NNFinder finder;
        for (auto& s : val_stdb.structures) finder.calc(s, rcut);
    }
    for (std::size_t i = 0; i < train_stdb.size(); ++i)
        train_stdb.structures[i].energy = 0.05 * static_cast<double>(i + 1);
    for (std::size_t i = 0; i < val_stdb.size(); ++i)
        val_stdb.structures[i].energy = 0.07 * static_cast<double>(i + 1);

    MLIPWorkspaceManager ws_main;
    Context c_main = make_minimal_trainer_context(cgrid_0, sgrid_0, rcut);
    Trainer host(c_main, ws_main);
    host.train(train_stdb);

    ScopedTempFileLocal cfg(R"(OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 1
ENDOPTIMIZER
LOSS L2
PC_EMAE 0 1
OPTIM CGRID2B (1) 0.0 4.0
)");
    Context ctx_for_targets =
        make_minimal_trainer_context(cgrid_0, sgrid_0, rcut);
    tadah::hpo::HPOTargets trg(ctx_for_targets, cfg.path.string());
    trg.LS.emplace_back();
    REQUIRE_FALSE(trg.LS.empty());

    auto prefs = tadah::hpo::make_param_ref_table(trg);
    REQUIRE(prefs.size() == 1);

    const std::size_t n_desc = host.DCS.descriptor_size();
    REQUIRE(n_desc > 0);

    /* Synthetic per-script context. Held alive in the test scope so the
     * SimsGradScriptInput's pointers remain valid until the helper
     * finishes consuming them. */
    constexpr int nlocal = 4;
    constexpr int natoms = 4;
    tadah::core::aeds_type2 aeds(n_desc, nlocal);
    for (int a = 0; a < nlocal; ++a)
        for (std::size_t c = 0; c < n_desc; ++c)
            aeds(c, a) = 1.0 + 0.1 * a + 0.2 * static_cast<double>(c);

    std::vector<double> dF_dW(
        static_cast<std::size_t>(3 * nlocal) * n_desc, 0.0);
    for (int a = 0; a < nlocal; ++a)
        for (int axis = 0; axis < 3; ++axis)
            for (std::size_t c = 0; c < n_desc; ++c)
                dF_dW[(static_cast<std::size_t>(3 * a + axis)) * n_desc + c]
                    = 0.05 * (a + 1) + 0.03 * (axis + 1)
                    + 0.07 * static_cast<double>(c + 1);

    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(3 * nlocal, 3 * nlocal) * 10.0;

    auto make_input = [&](bool with_observable) {
        tadah::hpo::SimsGradScriptInput in;
        in.ctx.nlocal   = nlocal;
        in.ctx.natoms   = natoms;
        in.ctx.n_desc   = static_cast<int>(n_desc);
        in.ctx.box_dofs = 0;
        in.ctx.aeds     = &aeds;
        in.ctx.lx = in.ctx.ly = 3.301; in.ctx.lz = 6.602;
        in.H            = H;
        in.dF_dW        = dF_dW.data();
        if (with_observable) {
            in.expressions = {"c_eatoms/atoms"};
            in.v_observed  = {0.6};
            in.t_targets   = {0.5};
            in.w_weights   = {1.0};
            in.loss_fns    = {tadah::LossFunction(
                                tadah::LossFunction::Type::L2)};
        }
        return in;
    };

    /* Case 1: null provider — Option A behaviour preserved. */
    std::vector<double> g_null;
    std::vector<int>    nfd_null;
    REQUIRE(tadah::hpo::compute_analytical_gradient_from_trainer(
        host, train_stdb, val_stdb, trg, prefs, g_null, nfd_null));
    REQUIRE(g_null.size() == prefs.size());
    REQUIRE(nfd_null.size() == prefs.size());
    CHECK(g_null[0] == 0.0);

    /* Case 2: provider returns a script with NO observables. The fit
     * side still computes its analytical gradient and the sims chain
     * is a no-op — needs_fd must come out empty and g_anal non-zero. */
    int provider_calls_empty = 0;
    auto provider_empty =
        [&](std::vector<tadah::hpo::SimsGradScriptInput>& out) {
            ++provider_calls_empty;
            out.clear();
            out.push_back(make_input(/*with_observable=*/false));
            return true;
        };
    std::vector<double> g_empty;
    std::vector<int>    nfd_empty;
    REQUIRE(tadah::hpo::compute_analytical_gradient_from_trainer(
        host, train_stdb, val_stdb, trg, prefs, g_empty, nfd_empty,
        provider_empty));
    CHECK(provider_calls_empty == 1);
    REQUIRE(g_empty.size() == prefs.size());
    CHECK(nfd_empty.empty());
    CHECK(std::isfinite(g_empty[0]));

    /* Case 3: provider returns a script WITH the cohesive-E observable.
     * The chain rule folds a non-zero sims_grad contribution into
     * g_anal — the result differs from Case 2 by a measurable amount.
     * Magnitude depends on the live dW/dθ; just assert non-zero diff. */
    int provider_calls_obs = 0;
    auto provider_obs =
        [&](std::vector<tadah::hpo::SimsGradScriptInput>& out) {
            ++provider_calls_obs;
            out.clear();
            out.push_back(make_input(/*with_observable=*/true));
            return true;
        };
    std::vector<double> g_obs;
    std::vector<int>    nfd_obs;
    REQUIRE(tadah::hpo::compute_analytical_gradient_from_trainer(
        host, train_stdb, val_stdb, trg, prefs, g_obs, nfd_obs,
        provider_obs));
    CHECK(provider_calls_obs == 1);
    REQUIRE(g_obs.size() == prefs.size());
    CHECK(nfd_obs.empty());
    CHECK(std::isfinite(g_obs[0]));
    /* The sims_grad contribution shifts the gradient. Without it the
     * helper would silently drop the LAMMPS-observable term — this
     * differs-from-no-observable check is the load-bearing wiring
     * assertion. */
    CHECK(std::abs(g_obs[0] - g_empty[0]) > 1e-12);

    /* Case 4: provider reports failure ⇒ all-or-nothing fall back to
     * FD (matches Phase 3.5's all-or-nothing contract). */
    int provider_calls_fail = 0;
    auto provider_fail =
        [&](std::vector<tadah::hpo::SimsGradScriptInput>& out) {
            ++provider_calls_fail;
            out.clear();
            return false;
        };
    std::vector<double> g_fail;
    std::vector<int>    nfd_fail;
    REQUIRE(tadah::hpo::compute_analytical_gradient_from_trainer(
        host, train_stdb, val_stdb, trg, prefs, g_fail, nfd_fail,
        provider_fail));
    CHECK(provider_calls_fail == 1);
    REQUIRE(nfd_fail.size() == prefs.size());
    CHECK(g_fail[0] == 0.0);
}

/* ------------------------------------------------------------------------
 * P3 (PROMPT_NEXT_NLOPT_LBFGS.md) — composite-stack smoke test
 *
 * D2_Blip + DM_Blip + Cut_Poly + BF_Linear + energy-only (FORCE=false,
 * STRESS=false). Mirrors the user's first-run config (minus PC_LAMMPS,
 * which routes through the P2 fallback gate). Exercises:
 *   - Cross-arm theta ownership: CGRID2B[1] is owned by D2_Blip only,
 *     CGRIDMB[1] is owned by DM_Blip only. The walker iterates both
 *     arms; only the owning arm writes per theta.
 *   - Cut_Poly fully analytical for both RCUT2B and RCUTMB chains
 *     (calc_drcut family complete after Phase 2B C-step).
 *   - Composite descriptor stack composes correctly end-to-end —
 *     pre-validates the Q4 risk (single dmb pointer in walker).
 *
 * Asserts the gate accepts (needs_fd.empty()) and parity vs FD within
 * 1e-3 relative for one theta from each arm.
 * ------------------------------------------------------------------------ */

namespace {

/* Minimal Context with both D2_Blip + Cut_Poly (RCTYPE2B) AND
 * DM_Blip + Cut_Poly (RCTYPEMB) active. Energy-only (FORCE=false,
 * STRESS=false). BF_Linear + LAMBDA + OALGO=4 — matches the analytical
 * gate's hard requirements. */
Context make_composite_blip_context(const std::vector<double>& cgrid_2b,
                                    const std::vector<double>& sgrid_2b,
                                    const std::vector<double>& cgrid_mb,
                                    const std::vector<double>& sgrid_mb,
                                    double rcut,
                                    int    Lmax = 2)
{
    Context c;
    c.add("ATOM",   "H");
    c.add("WATOM",  1);

    /* D2 arm — D2_Blip + Cut_Poly. */
    c.add("INIT2B", "true");
    c.add("TYPE2B", "D2_Blip");
    c.add("TYPE2B", static_cast<int>(cgrid_2b.size()));
    c.add("TYPE2B", static_cast<int>(sgrid_2b.size()));
    c.add("TYPE2B", "H"); c.add("TYPE2B", "H");
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < cgrid_2b.size(); ++i) {
            if (i) ss << " ";
            ss << cgrid_2b[i];
        }
        c.add("CGRID2B", ss.str());
    }
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < sgrid_2b.size(); ++i) {
            if (i) ss << " ";
            ss << sgrid_2b[i];
        }
        c.add("SGRID2B", ss.str());
    }
    c.add("RCTYPE2B", "Cut_Poly");
    c.add("RCUT2B",   rcut);

    /* DM arm — DM_Blip + Cut_Poly. */
    c.add("INITMB", "true");
    c.add("TYPEMB", "DM_Blip");
    c.add("TYPEMB", Lmax);
    c.add("TYPEMB", static_cast<int>(cgrid_mb.size()));
    c.add("TYPEMB", static_cast<int>(sgrid_mb.size()));
    c.add("TYPEMB", "H"); c.add("TYPEMB", "H");
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < cgrid_mb.size(); ++i) {
            if (i) ss << " ";
            ss << cgrid_mb[i];
        }
        c.add("CGRIDMB", ss.str());
    }
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < sgrid_mb.size(); ++i) {
            if (i) ss << " ";
            ss << sgrid_mb[i];
        }
        c.add("SGRIDMB", ss.str());
    }
    c.add("RCTYPEMB", "Cut_Poly");
    c.add("RCUTMB",   rcut);

    /* Common dimer/output/solver keys. */
    c.add("DIMER",    "false");
    c.add("DIMER",    0);
    c.add("DIMER",    "false");
    c.add("FORCE",    "false");
    c.add("STRESS",   "false");
    c.add("EWEIGHT",  1.0);
    c.add("FWEIGHT",  1.0);
    c.add("SWEIGHT",  1.0);
    c.add("BIAS",     "false");
    c.add("NORM",     "false");
    c.add("VERBOSE",  0);
    c.add("MODEL",    "M_BLR");
    c.add("MODEL",    "BF_Linear");
    c.add("LAMBDA",   1e-3);
    c.add("OALGO",    4);
    return c;
}

}  // namespace

TEST_CASE("compute_analytical_gradient_from_trainer matches FD on live trainer "
          "(D2_Blip + DM_Blip + Cut_Poly + BF_Linear, energy-only)",
          "[HPO][analytical_gradient][live_trainer][parity][composite][blip]")
{
    using tadah::mlip::memory::MLIPWorkspaceManager;
    tadah::core::PeriodicTable::initialize();

    /* Tiny grids — keep N_cols small. Blip support is |x_c|<2, so cgrid
     * values within rcut and sgrid >> 1 keeps (r-cgrid)/sgrid inside
     * support for r ~ 1.6 .. 2.5. Same grid for both arms to keep the
     * test compact. */
    const std::vector<double> cgrid_2b_0 = {0.5, 1.4};
    const std::vector<double> sgrid_2b_0 = {3.0, 5.0};
    const std::vector<double> cgrid_mb_0 = {0.5, 1.4};
    const std::vector<double> sgrid_mb_0 = {3.0, 5.0};
    const double rcut = 3.0;

    auto train_stdb = make_h3_stdb(/*N=*/4, rcut);
    auto val_stdb   = make_h3_stdb(/*N=*/3, rcut);
    for (auto& s : val_stdb.structures) {
        for (auto& a : s.atoms) a.position[1] += 0.07;
    }
    {
        NNFinder finder;
        for (auto& s : val_stdb.structures) finder.calc(s, rcut);
    }
    for (std::size_t i = 0; i < train_stdb.size(); ++i)
        train_stdb.structures[i].energy = 0.05 * static_cast<double>(i + 1);
    for (std::size_t i = 0; i < val_stdb.size(); ++i)
        val_stdb.structures[i].energy = 0.07 * static_cast<double>(i + 1);

    MLIPWorkspaceManager ws_main;
    Context c_main = make_composite_blip_context(
        cgrid_2b_0, sgrid_2b_0, cgrid_mb_0, sgrid_mb_0, rcut);
    Trainer host(c_main, ws_main);
    host.train(train_stdb);

    /* Sanity: both arms active in the trainer context. */
    REQUIRE(host.context.template get<bool>("INIT2B"));
    REQUIRE(host.context.template get<bool>("INITMB"));
    REQUIRE_FALSE(host.context.template get<bool>("FORCE"));

    /* Optimise two thetas — one from each arm. Both have closed forms:
     * D2_Blip owns CGRID2B (Phase 2B-A1); DM_Blip owns CGRIDMB
     * (Phase 2B-A3). The walker iterates both arms; only the owning
     * arm should write per theta. */
    ScopedTempFileLocal cfg(R"(OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 1
ENDOPTIMIZER
LOSS L2
PC_EMAE 0 1
OPTIM CGRID2B (1) 0.0 4.0
OPTIM CGRIDMB (1) 0.0 4.0
)");
    Context ctx_for_targets = make_composite_blip_context(
        cgrid_2b_0, sgrid_2b_0, cgrid_mb_0, sgrid_mb_0, rcut);
    tadah::hpo::HPOTargets trg(ctx_for_targets, cfg.path.string());
    REQUIRE(trg.use_emae);
    REQUIRE(trg.optimap.count("CGRID2B"));
    REQUIRE(trg.optimap.count("CGRIDMB"));

    auto prefs = tadah::hpo::make_param_ref_table(trg);
    REQUIRE(prefs.size() == 2);

    /* The order depends on std::map iteration (lexicographic). */
    std::size_t idx_2b = prefs.size(), idx_mb = prefs.size();
    for (std::size_t i = 0; i < prefs.size(); ++i) {
        if (prefs[i].theta_name == "CGRID2B") idx_2b = i;
        if (prefs[i].theta_name == "CGRIDMB") idx_mb = i;
    }
    REQUIRE(idx_2b < prefs.size());
    REQUIRE(idx_mb < prefs.size());
    CHECK(prefs[idx_2b].arg_pos == 1);
    CHECK(prefs[idx_mb].arg_pos == 1);

    std::vector<double> g_anal;
    std::vector<int>    needs_fd;
    REQUIRE(tadah::hpo::compute_analytical_gradient_from_trainer(
        host, train_stdb, val_stdb, trg, prefs, g_anal, needs_fd));
    REQUIRE(g_anal.size() == 2);

    /* Critical: the gate must accept the composite D2+DM stack with
     * Cut_Poly under energy-only HYBRID — no FD fallback. */
    REQUIRE(needs_fd.empty());

    /* FD reference: re-fit a fresh Trainer at theta_0 ± h on each arm
     * and central-difference the loss. Mirrors the BF_Linear baseline. */
    auto loss_at = [&](const std::vector<double>& cg2b,
                       const std::vector<double>& cgmb) {
        MLIPWorkspaceManager ws_h;
        Context c_h = make_composite_blip_context(
            cg2b, sgrid_2b_0, cgmb, sgrid_mb_0, rcut);
        Trainer t_h(c_h, ws_h);
        t_h.train(train_stdb);
        const auto pred_e = predict_energies(t_h, val_stdb);
        const double emae = emae_scaled(val_stdb, pred_e);
        return trg.globalLoss.loss(emae, trg.emae, trg.w_emae);
    };

    const double h = 1e-5;

    /* CGRID2B[1] partial. */
    {
        std::vector<double> cg2b_p = cgrid_2b_0;  cg2b_p[0] = cgrid_2b_0[0] + h;
        std::vector<double> cg2b_m = cgrid_2b_0;  cg2b_m[0] = cgrid_2b_0[0] - h;
        const double dl_fd = (loss_at(cg2b_p, cgrid_mb_0) -
                              loss_at(cg2b_m, cgrid_mb_0)) / (2.0 * h);
        const double rel = std::abs(g_anal[idx_2b] - dl_fd)
                         / std::max({std::abs(g_anal[idx_2b]),
                                     std::abs(dl_fd), 1e-12});
        INFO("CGRID2B[1] analytical=" << g_anal[idx_2b] << "  FD=" << dl_fd);
        CHECK(rel < 1e-3);
    }

    /* CGRIDMB[1] partial. */
    {
        std::vector<double> cgmb_p = cgrid_mb_0;  cgmb_p[0] = cgrid_mb_0[0] + h;
        std::vector<double> cgmb_m = cgrid_mb_0;  cgmb_m[0] = cgrid_mb_0[0] - h;
        const double dl_fd = (loss_at(cgrid_2b_0, cgmb_p) -
                              loss_at(cgrid_2b_0, cgmb_m)) / (2.0 * h);
        const double rel = std::abs(g_anal[idx_mb] - dl_fd)
                         / std::max({std::abs(g_anal[idx_mb]),
                                     std::abs(dl_fd), 1e-12});
        INFO("CGRIDMB[1] analytical=" << g_anal[idx_mb] << "  FD=" << dl_fd);
        CHECK(rel < 1e-3);
    }
}

/* ------------------------------------------------------------------------
 * Phase 2C-DM — Layer-3 end-to-end FORCE parity on composite DM_mBlip
 * stack. Mirrors the user's BASINSv2_blip6_lambda_fix shape (DM_mBlip
 * with F_Blip embedding, D2_Blip + Cut_Cos, BF_Linear) but reduced to
 * tiny grids for test cost. PC_EMAE+PC_FRMSE drive both energy- and
 * force-row chains through `assemble_dphi_train_force_rows` (D2 + DM).
 *
 * Asserts:
 *   - The gate accepts (needs_fd.empty()) with FORCE=true.
 *   - g_anal matches central-difference within 1e-3 relative for one
 *     theta from each arm (CGRID2B[1] from D2_Blip, CGRIDMB[1] from
 *     DM_mBlip).
 * ------------------------------------------------------------------------ */

namespace {

/* Composite Context with both D2_Blip + DM_mBlip (= DM_mEAD<F_Blip>)
 * arms, FORCE=true, BF_Linear, OALGO=4. DM_mBlip enforces CGRIDMB and
 * SGRIDMB of size 1 (F_Blip is dimension-expanding) — descriptor
 * dimension per L comes from CEMBFUNC.size(). */
Context make_composite_mblip_force_context(const std::vector<double>& cgrid_2b,
                                           const std::vector<double>& sgrid_2b,
                                           double cgrid_mb_val,
                                           double sgrid_mb_val,
                                           const std::vector<double>& cembfunc,
                                           const std::vector<double>& sembfunc,
                                           double rcut,
                                           int    Lmax = 1)
{
    Context c;
    c.add("ATOM",   "H");
    c.add("WATOM",  1);

    /* D2 arm — D2_Blip + Cut_Cos. */
    c.add("INIT2B", "true");
    c.add("TYPE2B", "D2_Blip");
    c.add("TYPE2B", static_cast<int>(cgrid_2b.size()));
    c.add("TYPE2B", static_cast<int>(sgrid_2b.size()));
    c.add("TYPE2B", "H"); c.add("TYPE2B", "H");
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < cgrid_2b.size(); ++i) {
            if (i) ss << " ";
            ss << cgrid_2b[i];
        }
        c.add("CGRID2B", ss.str());
    }
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < sgrid_2b.size(); ++i) {
            if (i) ss << " ";
            ss << sgrid_2b[i];
        }
        c.add("SGRID2B", ss.str());
    }
    c.add("RCTYPE2B", "Cut_Cos");
    c.add("RCUT2B",   rcut);

    /* DM arm — DM_mBlip + Cut_Cos. TYPEMB layout for DM_mEAD<F_Blip>:
     *   label + Lmax + size(cgrid) + size(sgrid) +
     *   size(cembfunc) + size(sembfunc) + atom pairs. */
    c.add("INITMB", "true");
    c.add("TYPEMB", "DM_mBlip");
    c.add("TYPEMB", Lmax);
    c.add("TYPEMB", 1);
    c.add("TYPEMB", 1);
    c.add("TYPEMB", static_cast<int>(cembfunc.size()));
    c.add("TYPEMB", static_cast<int>(sembfunc.size()));
    c.add("TYPEMB", "H"); c.add("TYPEMB", "H");
    c.add("CGRIDMB", cgrid_mb_val);
    c.add("SGRIDMB", sgrid_mb_val);
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < cembfunc.size(); ++i) {
            if (i) ss << " ";
            ss << cembfunc[i];
        }
        c.add("CEMBFUNC", ss.str());
    }
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < sembfunc.size(); ++i) {
            if (i) ss << " ";
            ss << sembfunc[i];
        }
        c.add("SEMBFUNC", ss.str());
    }
    c.add("RCTYPEMB", "Cut_Cos");
    c.add("RCUTMB",   rcut);

    c.add("DIMER",    "false");
    c.add("DIMER",    0);
    c.add("DIMER",    "false");
    c.add("FORCE",    "true");
    c.add("STRESS",   "false");
    c.add("EWEIGHT",  1.0);
    c.add("FWEIGHT",  1.0);
    c.add("SWEIGHT",  1.0);
    c.add("BIAS",     "false");
    c.add("NORM",     "false");
    c.add("VERBOSE",  0);
    c.add("MODEL",    "M_BLR");
    c.add("MODEL",    "BF_Linear");
    c.add("LAMBDA",   1e-3);
    c.add("OALGO",    4);
    return c;
}

}  // namespace

TEST_CASE("compute_analytical_gradient_from_trainer matches FD on live trainer "
          "(D2_Blip + DM_mBlip + FORCE=true + EMAE+FRMSE/L2) — Phase 2C-DM",
          "[HPO][analytical_gradient][live_trainer][parity][composite][force][mblip]")
{
    using tadah::mlip::memory::MLIPWorkspaceManager;
    tadah::core::PeriodicTable::initialize();

    const std::vector<double> cgrid_2b_0 = {0.5, 1.4};
    const std::vector<double> sgrid_2b_0 = {3.0, 5.0};
    const double cgrid_mb_0 = 0.7;
    const double sgrid_mb_0 = 1.5;
    const std::vector<double> cembfunc = {0.10, 0.25};
    const std::vector<double> sembfunc = {3.0, 3.0};
    const double rcut = 3.0;
    const int    Lmax = 1;

    auto train_stdb = make_h3_force_stdb(/*N=*/4, rcut, /*force_scale=*/1.0);
    auto val_stdb   = make_h3_force_stdb(/*N=*/3, rcut, /*force_scale=*/0.5);
    for (auto& s : val_stdb.structures) {
        for (auto& a : s.atoms) a.position[1] += 0.07;
    }
    {
        NNFinder finder;
        for (auto& s : val_stdb.structures) finder.calc(s, rcut);
    }
    for (std::size_t i = 0; i < train_stdb.size(); ++i)
        train_stdb.structures[i].energy = 0.05 * static_cast<double>(i + 1);
    for (std::size_t i = 0; i < val_stdb.size(); ++i)
        val_stdb.structures[i].energy = 0.07 * static_cast<double>(i + 1);

    MLIPWorkspaceManager ws_main;
    Context c_main = make_composite_mblip_force_context(
        cgrid_2b_0, sgrid_2b_0, cgrid_mb_0, sgrid_mb_0,
        cembfunc, sembfunc, rcut, Lmax);
    Trainer host(c_main, ws_main);
    host.train(train_stdb);

    REQUIRE(host.context.template get<bool>("INIT2B"));
    REQUIRE(host.context.template get<bool>("INITMB"));
    REQUIRE(host.context.template get<bool>("FORCE"));

    ScopedTempFileLocal cfg(R"(OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 1
ENDOPTIMIZER
LOSS L2
PC_EMAE  0 1
PC_FRMSE 0 1
OPTIM CGRID2B (1) 0.0 4.0
OPTIM CGRIDMB (1) 0.0 4.0
)");
    Context ctx_for_targets = make_composite_mblip_force_context(
        cgrid_2b_0, sgrid_2b_0, cgrid_mb_0, sgrid_mb_0,
        cembfunc, sembfunc, rcut, Lmax);
    tadah::hpo::HPOTargets trg(ctx_for_targets, cfg.path.string());
    REQUIRE(trg.use_emae);
    REQUIRE(trg.use_frmse);
    REQUIRE(trg.optimap.count("CGRID2B"));
    REQUIRE(trg.optimap.count("CGRIDMB"));

    auto prefs = tadah::hpo::make_param_ref_table(trg);
    REQUIRE(prefs.size() == 2);

    std::size_t idx_2b = prefs.size(), idx_mb = prefs.size();
    for (std::size_t i = 0; i < prefs.size(); ++i) {
        if (prefs[i].theta_name == "CGRID2B") idx_2b = i;
        if (prefs[i].theta_name == "CGRIDMB") idx_mb = i;
    }
    REQUIRE(idx_2b < prefs.size());
    REQUIRE(idx_mb < prefs.size());

    std::vector<double> g_anal;
    std::vector<int>    needs_fd;
    REQUIRE(tadah::hpo::compute_analytical_gradient_from_trainer(
        host, train_stdb, val_stdb, trg, prefs, g_anal, needs_fd));
    REQUIRE(g_anal.size() == 2);

    /* Critical: the gate must accept the composite D2_Blip + DM_mBlip
     * stack with FORCE=true + use_frmse=true — no FD fallback. */
    REQUIRE(needs_fd.empty());

    auto loss_at = [&](const std::vector<double>& cg2b, double cgmb_val) {
        MLIPWorkspaceManager ws_h;
        Context c_h = make_composite_mblip_force_context(
            cg2b, sgrid_2b_0, cgmb_val, sgrid_mb_0,
            cembfunc, sembfunc, rcut, Lmax);
        Trainer t_h(c_h, ws_h);
        t_h.train(train_stdb);
        const auto pred_full = predict_full(t_h, val_stdb);
        std::vector<double> pred_e(val_stdb.size(), 0.0);
        for (std::size_t i = 0; i < val_stdb.size(); ++i)
            pred_e[i] = pred_full(i).energy;
        const double emae  = emae_scaled (val_stdb, pred_e);
        const double frmse = frmse_scaled(val_stdb, pred_full);
        return (trg.globalLoss.loss(emae,  trg.emae,  trg.w_emae)
              + trg.globalLoss.loss(frmse, trg.frmse_targets.global().target, trg.frmse_targets.global().weight)) / 2.0;
    };

    const double h = 1e-5;

    /* CGRID2B[1] partial. */
    {
        std::vector<double> cg2b_p = cgrid_2b_0;  cg2b_p[0] = cgrid_2b_0[0] + h;
        std::vector<double> cg2b_m = cgrid_2b_0;  cg2b_m[0] = cgrid_2b_0[0] - h;
        const double dl_fd = (loss_at(cg2b_p, cgrid_mb_0) -
                              loss_at(cg2b_m, cgrid_mb_0)) / (2.0 * h);
        const double rel = std::abs(g_anal[idx_2b] - dl_fd)
                         / std::max({std::abs(g_anal[idx_2b]),
                                     std::abs(dl_fd), 1e-12});
        INFO("CGRID2B[1] analytical=" << g_anal[idx_2b] << "  FD=" << dl_fd);
        CHECK(rel < 1e-3);
    }

    /* CGRIDMB[1] partial. */
    {
        const double cgmb_p = cgrid_mb_0 + h;
        const double cgmb_m = cgrid_mb_0 - h;
        const double dl_fd = (loss_at(cgrid_2b_0, cgmb_p) -
                              loss_at(cgrid_2b_0, cgmb_m)) / (2.0 * h);
        const double rel = std::abs(g_anal[idx_mb] - dl_fd)
                         / std::max({std::abs(g_anal[idx_mb]),
                                     std::abs(dl_fd), 1e-12});
        INFO("CGRIDMB[1] analytical=" << g_anal[idx_mb] << "  FD=" << dl_fd);
        CHECK(rel < 1e-3);
    }
}

/* ------------------------------------------------------------------------
 * Phase 2C-WATOM — Layer-3 end-to-end parity with WATOM[1] in OPTIM.
 * Mirrors the Phase 2C-DM composite test but adds WATOM[1] alongside
 * CGRID2B[1] and CGRIDMB[1]. The gate must accept WATOM (no FD fallback)
 * and the analytical gradient must match FD within 1e-3 relative.
 * Asserts the user's BASINSv2_blip6 config shape (D2_Blip + DM_mBlip +
 * FORCE=true + EMAE+FRMSE) runs 100% analytical.
 * ------------------------------------------------------------------------ */

namespace {

Context make_composite_mblip_force_context_with_watom(
        const std::vector<double>& cgrid_2b,
        const std::vector<double>& sgrid_2b,
        double cgrid_mb_val,
        double sgrid_mb_val,
        const std::vector<double>& cembfunc,
        const std::vector<double>& sembfunc,
        double rcut,
        double watom,
        int    Lmax = 1)
{
    Context c;
    c.add("ATOM",   "H");
    c.add("WATOM",  watom);
    c.add("INIT2B", "true");
    c.add("TYPE2B", "D2_Blip");
    c.add("TYPE2B", static_cast<int>(cgrid_2b.size()));
    c.add("TYPE2B", static_cast<int>(sgrid_2b.size()));
    c.add("TYPE2B", "H"); c.add("TYPE2B", "H");
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < cgrid_2b.size(); ++i) { if (i) ss << " "; ss << cgrid_2b[i]; }
        c.add("CGRID2B", ss.str());
    }
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < sgrid_2b.size(); ++i) { if (i) ss << " "; ss << sgrid_2b[i]; }
        c.add("SGRID2B", ss.str());
    }
    c.add("RCTYPE2B", "Cut_Cos");
    c.add("RCUT2B",   rcut);

    c.add("INITMB", "true");
    c.add("TYPEMB", "DM_mBlip");
    c.add("TYPEMB", Lmax);
    c.add("TYPEMB", 1);
    c.add("TYPEMB", 1);
    c.add("TYPEMB", static_cast<int>(cembfunc.size()));
    c.add("TYPEMB", static_cast<int>(sembfunc.size()));
    c.add("TYPEMB", "H"); c.add("TYPEMB", "H");
    c.add("CGRIDMB", cgrid_mb_val);
    c.add("SGRIDMB", sgrid_mb_val);
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < cembfunc.size(); ++i) { if (i) ss << " "; ss << cembfunc[i]; }
        c.add("CEMBFUNC", ss.str());
    }
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < sembfunc.size(); ++i) { if (i) ss << " "; ss << sembfunc[i]; }
        c.add("SEMBFUNC", ss.str());
    }
    c.add("RCTYPEMB", "Cut_Cos");
    c.add("RCUTMB",   rcut);
    c.add("DIMER",    "false");
    c.add("DIMER",    0);
    c.add("DIMER",    "false");
    c.add("FORCE",    "true");
    c.add("STRESS",   "false");
    c.add("EWEIGHT",  1.0);
    c.add("FWEIGHT",  1.0);
    c.add("SWEIGHT",  1.0);
    c.add("BIAS",     "false");
    c.add("NORM",     "false");
    c.add("VERBOSE",  0);
    c.add("MODEL",    "M_BLR");
    c.add("MODEL",    "BF_Linear");
    c.add("LAMBDA",   1e-3);
    c.add("OALGO",    4);
    return c;
}

}  // namespace

TEST_CASE("compute_analytical_gradient_from_trainer matches FD on live trainer "
          "(D2_Blip + DM_mBlip + WATOM + FORCE=true + EMAE+FRMSE) — Phase 2C-WATOM",
          "[HPO][analytical_gradient][live_trainer][parity][composite][force][mblip][watom]")
{
    using tadah::mlip::memory::MLIPWorkspaceManager;
    tadah::core::PeriodicTable::initialize();

    const std::vector<double> cgrid_2b_0 = {0.5, 1.4};
    const std::vector<double> sgrid_2b_0 = {3.0, 5.0};
    const double cgrid_mb_0 = 0.7;
    const double sgrid_mb_0 = 1.5;
    const std::vector<double> cembfunc = {0.10, 0.25};
    const std::vector<double> sembfunc = {3.0, 3.0};
    const double rcut = 3.0;
    const int    Lmax = 1;
    const double watom_0 = 1.0;

    auto train_stdb = make_h3_force_stdb(/*N=*/4, rcut, /*force_scale=*/1.0);
    auto val_stdb   = make_h3_force_stdb(/*N=*/3, rcut, /*force_scale=*/0.5);
    for (auto& s : val_stdb.structures) {
        for (auto& a : s.atoms) a.position[1] += 0.07;
    }
    {
        NNFinder finder;
        for (auto& s : val_stdb.structures) finder.calc(s, rcut);
    }
    for (std::size_t i = 0; i < train_stdb.size(); ++i)
        train_stdb.structures[i].energy = 0.05 * static_cast<double>(i + 1);
    for (std::size_t i = 0; i < val_stdb.size(); ++i)
        val_stdb.structures[i].energy = 0.07 * static_cast<double>(i + 1);

    MLIPWorkspaceManager ws_main;
    Context c_main = make_composite_mblip_force_context_with_watom(
        cgrid_2b_0, sgrid_2b_0, cgrid_mb_0, sgrid_mb_0,
        cembfunc, sembfunc, rcut, watom_0, Lmax);
    Trainer host(c_main, ws_main);
    host.train(train_stdb);

    REQUIRE(host.context.template get<bool>("INIT2B"));
    REQUIRE(host.context.template get<bool>("INITMB"));
    REQUIRE(host.context.template get<bool>("FORCE"));

    ScopedTempFileLocal cfg(R"(OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 1
ENDOPTIMIZER
LOSS L2
PC_EMAE  0 1
PC_FRMSE 0 1
OPTIM CGRID2B (1) 0.0 4.0
OPTIM CGRIDMB (1) 0.0 4.0
OPTIM WATOM   (1) 0.2 10.0
)");
    Context ctx_for_targets = make_composite_mblip_force_context_with_watom(
        cgrid_2b_0, sgrid_2b_0, cgrid_mb_0, sgrid_mb_0,
        cembfunc, sembfunc, rcut, watom_0, Lmax);
    tadah::hpo::HPOTargets trg(ctx_for_targets, cfg.path.string());
    REQUIRE(trg.use_emae);
    REQUIRE(trg.use_frmse);
    REQUIRE(trg.optimap.count("CGRID2B"));
    REQUIRE(trg.optimap.count("CGRIDMB"));
    REQUIRE(trg.optimap.count("WATOM"));

    auto prefs = tadah::hpo::make_param_ref_table(trg);
    REQUIRE(prefs.size() == 3);

    std::size_t idx_2b = prefs.size(), idx_mb = prefs.size(), idx_w = prefs.size();
    for (std::size_t i = 0; i < prefs.size(); ++i) {
        if (prefs[i].theta_name == "CGRID2B") idx_2b = i;
        if (prefs[i].theta_name == "CGRIDMB") idx_mb = i;
        if (prefs[i].theta_name == "WATOM")   idx_w  = i;
    }
    REQUIRE(idx_2b < prefs.size());
    REQUIRE(idx_mb < prefs.size());
    REQUIRE(idx_w  < prefs.size());

    std::vector<double> g_anal;
    std::vector<int>    needs_fd;
    REQUIRE(tadah::hpo::compute_analytical_gradient_from_trainer(
        host, train_stdb, val_stdb, trg, prefs, g_anal, needs_fd));
    REQUIRE(g_anal.size() == 3);

    /* Critical: gate must accept WATOM analytically. */
    REQUIRE(needs_fd.empty());

    auto loss_at = [&](const std::vector<double>& cg2b, double cgmb_val, double w_val) {
        MLIPWorkspaceManager ws_h;
        Context c_h = make_composite_mblip_force_context_with_watom(
            cg2b, sgrid_2b_0, cgmb_val, sgrid_mb_0,
            cembfunc, sembfunc, rcut, w_val, Lmax);
        Trainer t_h(c_h, ws_h);
        t_h.train(train_stdb);
        const auto pred_full = predict_full(t_h, val_stdb);
        std::vector<double> pred_e(val_stdb.size(), 0.0);
        for (std::size_t i = 0; i < val_stdb.size(); ++i)
            pred_e[i] = pred_full(i).energy;
        const double emae  = emae_scaled (val_stdb, pred_e);
        const double frmse = frmse_scaled(val_stdb, pred_full);
        return (trg.globalLoss.loss(emae,  trg.emae,  trg.w_emae)
              + trg.globalLoss.loss(frmse, trg.frmse_targets.global().target, trg.frmse_targets.global().weight)) / 2.0;
    };

    const double h = 1e-5;

    {
        std::vector<double> cg2b_p = cgrid_2b_0;  cg2b_p[0] = cgrid_2b_0[0] + h;
        std::vector<double> cg2b_m = cgrid_2b_0;  cg2b_m[0] = cgrid_2b_0[0] - h;
        const double dl_fd = (loss_at(cg2b_p, cgrid_mb_0, watom_0) -
                              loss_at(cg2b_m, cgrid_mb_0, watom_0)) / (2.0 * h);
        const double rel = std::abs(g_anal[idx_2b] - dl_fd)
                         / std::max({std::abs(g_anal[idx_2b]),
                                     std::abs(dl_fd), 1e-12});
        INFO("CGRID2B[1] analytical=" << g_anal[idx_2b] << "  FD=" << dl_fd);
        CHECK(rel < 1e-3);
    }

    {
        const double cgmb_p = cgrid_mb_0 + h;
        const double cgmb_m = cgrid_mb_0 - h;
        const double dl_fd = (loss_at(cgrid_2b_0, cgmb_p, watom_0) -
                              loss_at(cgrid_2b_0, cgmb_m, watom_0)) / (2.0 * h);
        const double rel = std::abs(g_anal[idx_mb] - dl_fd)
                         / std::max({std::abs(g_anal[idx_mb]),
                                     std::abs(dl_fd), 1e-12});
        INFO("CGRIDMB[1] analytical=" << g_anal[idx_mb] << "  FD=" << dl_fd);
        CHECK(rel < 1e-3);
    }

    /* The critical WATOM[1] partial. */
    {
        const double dl_fd = (loss_at(cgrid_2b_0, cgrid_mb_0, watom_0 + h) -
                              loss_at(cgrid_2b_0, cgrid_mb_0, watom_0 - h)) / (2.0 * h);
        const double rel = std::abs(g_anal[idx_w] - dl_fd)
                         / std::max({std::abs(g_anal[idx_w]),
                                     std::abs(dl_fd), 1e-12});
        INFO("WATOM[1] analytical=" << g_anal[idx_w] << "  FD=" << dl_fd);
        CHECK(rel < 1e-3);
    }
}

/* Phase 2D reproducibility: the per-θ loop in
 * compute_analytical_gradient_from_trainer is wrapped in
 * `#pragma omp parallel for schedule(dynamic)`. Each iteration owns its
 * dphi_train_buf[i] / dphi_val_buf[i] / per_theta[i] slot and a
 * thread-local scratch buffer; the descriptor walkers are const on the
 * shared state. No reductions happen across threads in the loop, so the
 * gradient values must be byte-identical regardless of OMP_NUM_THREADS.
 *
 * The composite [force][mblip][watom] case from Phase 2C-WATOM is the
 * stress-test target: it exercises both arms (D2 + DM), both walker
 * pairs (energy + force, train + val), and both per-pair and per-atom
 * scratch usage. */
TEST_CASE("compute_analytical_gradient_from_trainer reproducibility under "
          "varying OMP thread counts (Phase 2D)",
          "[HPO][analytical_gradient][live_trainer][reproducibility][composite][force][mblip][watom][omp]")
{
    using tadah::mlip::memory::MLIPWorkspaceManager;
    tadah::core::PeriodicTable::initialize();

    const std::vector<double> cgrid_2b_0 = {0.5, 1.4};
    const std::vector<double> sgrid_2b_0 = {3.0, 5.0};
    const double cgrid_mb_0 = 0.7;
    const double sgrid_mb_0 = 1.5;
    const std::vector<double> cembfunc = {0.10, 0.25};
    const std::vector<double> sembfunc = {3.0, 3.0};
    const double rcut = 3.0;
    const int    Lmax = 1;
    const double watom_0 = 1.0;

    auto train_stdb = make_h3_force_stdb(/*N=*/4, rcut, /*force_scale=*/1.0);
    auto val_stdb   = make_h3_force_stdb(/*N=*/3, rcut, /*force_scale=*/0.5);
    for (auto& s : val_stdb.structures) {
        for (auto& a : s.atoms) a.position[1] += 0.07;
    }
    {
        NNFinder finder;
        for (auto& s : val_stdb.structures) finder.calc(s, rcut);
    }
    for (std::size_t i = 0; i < train_stdb.size(); ++i)
        train_stdb.structures[i].energy = 0.05 * static_cast<double>(i + 1);
    for (std::size_t i = 0; i < val_stdb.size(); ++i)
        val_stdb.structures[i].energy = 0.07 * static_cast<double>(i + 1);

    MLIPWorkspaceManager ws_main;
    Context c_main = make_composite_mblip_force_context_with_watom(
        cgrid_2b_0, sgrid_2b_0, cgrid_mb_0, sgrid_mb_0,
        cembfunc, sembfunc, rcut, watom_0, Lmax);
    Trainer host(c_main, ws_main);
    host.train(train_stdb);

    ScopedTempFileLocal cfg(R"(OPTIMIZER
LIB DLIB
ALGO BFGS
MAXEVAL 1
ENDOPTIMIZER
LOSS L2
PC_EMAE  0 1
PC_FRMSE 0 1
OPTIM CGRID2B (1) 0.0 4.0
OPTIM CGRIDMB (1) 0.0 4.0
OPTIM WATOM   (1) 0.2 10.0
)");
    Context ctx_for_targets = make_composite_mblip_force_context_with_watom(
        cgrid_2b_0, sgrid_2b_0, cgrid_mb_0, sgrid_mb_0,
        cembfunc, sembfunc, rcut, watom_0, Lmax);
    tadah::hpo::HPOTargets trg(ctx_for_targets, cfg.path.string());
    auto prefs = tadah::hpo::make_param_ref_table(trg);
    REQUIRE(prefs.size() == 3);

    /* Run once with 1 thread (the serial reference). */
#if defined(TADAH_ENABLE_OPENMP)
    omp_set_num_threads(1);
#endif
    std::vector<double> g_serial;
    std::vector<int>    needs_fd_serial;
    REQUIRE(tadah::hpo::compute_analytical_gradient_from_trainer(
        host, train_stdb, val_stdb, trg, prefs, g_serial, needs_fd_serial));
    REQUIRE(g_serial.size() == 3);
    REQUIRE(needs_fd_serial.empty());

    /* Run again with OMP_NUM_THREADS=8 (or omp_get_max_threads if smaller). */
#if defined(TADAH_ENABLE_OPENMP)
    const int target_threads = std::min(8, omp_get_max_threads());
    omp_set_num_threads(target_threads);
#endif
    std::vector<double> g_parallel;
    std::vector<int>    needs_fd_parallel;
    REQUIRE(tadah::hpo::compute_analytical_gradient_from_trainer(
        host, train_stdb, val_stdb, trg, prefs, g_parallel, needs_fd_parallel));
    REQUIRE(g_parallel.size() == 3);
    REQUIRE(needs_fd_parallel.empty());

    /* Per-θ buffers don't cross thread boundaries — the downstream
     * compute_dWdtheta is sequential and consumes each pre-built buffer
     * independently. Result must be byte-identical. */
    for (std::size_t i = 0; i < g_serial.size(); ++i) {
        INFO("theta " << i << " (" << prefs[i].theta_name << "): "
             << "serial=" << g_serial[i] << "  parallel=" << g_parallel[i]);
        CHECK(g_serial[i] == g_parallel[i]);
    }

    /* Phase 2F: cost-descending sort over prefs reorders OMP dispatch
     * but writes go to dphi_*_buf[i] / per_theta[i] in descriptor-side
     * indexing, so the orchestrator's chain-rule reduction is identical.
     * The compute_analytical_gradient_from_trainer body keeps a static
     * cost cache: first call here populated it (g_serial), so g_parallel
     * already used cost-descending sort. Add a third call with OMP=8
     * (cache now warm AGAIN, sort stable) to confirm successive calls
     * remain byte-identical — guards against any state leaking between
     * calls through the cache or the order vector. */
    std::vector<double> g_parallel_again;
    std::vector<int>    needs_fd_parallel_again;
    REQUIRE(tadah::hpo::compute_analytical_gradient_from_trainer(
        host, train_stdb, val_stdb, trg, prefs,
        g_parallel_again, needs_fd_parallel_again));
    REQUIRE(g_parallel_again.size() == 3);
    REQUIRE(needs_fd_parallel_again.empty());
    for (std::size_t i = 0; i < g_serial.size(); ++i) {
        INFO("theta " << i << " (" << prefs[i].theta_name << "): "
             << "g_serial=" << g_serial[i]
             << "  g_parallel_again=" << g_parallel_again[i]);
        CHECK(g_serial[i] == g_parallel_again[i]);
    }

#if defined(TADAH_ENABLE_OPENMP)
    /* Restore so other tests aren't affected. */
    omp_set_num_threads(omp_get_max_threads());
#endif
}


TEST_CASE("Indexed ERMSE and FRMSE live trainer gradients match finite differences",
          "[pc_dataset][analytical_gradient][live_trainer]")
{
    using tadah::mlip::memory::MLIPWorkspaceManager;
    tadah::core::PeriodicTable::initialize();
    const std::vector<double> cgrid{0.5, 1.4}, sgrid{3.0, 5.0};
    const double rcut = 3.0, theta = cgrid[0];
    auto train = make_h3_force_stdb(4, rcut, 1.0);
    auto val = make_h3_force_stdb(3, rcut, 0.5);
    val.dbidx = {0, 1, 3}; // unequal datasets: average datasets, not structures
    for (std::size_t i = 0; i < train.size(); ++i) train(i).energy = 0.05*(i+1);
    for (std::size_t i = 0; i < val.size(); ++i) val(i).energy = 0.07*(i+1);
    MLIPWorkspaceManager ws;
    Context ctx = make_minimal_trainer_context_force(cgrid, sgrid, rcut);
    Trainer host(ctx, ws);
    host.train(train);
    for (const auto* loss_name : {"L1", "L2", "LOG_COSH", "MAPE", "HUBER", "RMSLE", "TUKEY"}) {
        for (int mode = 0; mode < 3; ++mode) {
            INFO("LOSS=" << loss_name << " mode=" << mode);
            const bool energy = mode != 1, force = mode != 0;
            std::string body = "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\nENDOPTIMIZER\nLOSS "
                + std::string(loss_name) + "\nOPTIM CGRID2B (1) 0.0 4.0\n";
            if (energy) body += "PC_ERMSE (1) 15 0.003\nPC_ERMSE (2) 5 0.007\n";
            if (force) body += "PC_FRMSE (2) 0.08 0.5\nPC_FRMSE (1) 0.02 2\n";
            ScopedTempFileLocal cfg(body);
            auto target_ctx = make_minimal_trainer_context_force(cgrid, sgrid, rcut);
            tadah::hpo::HPOTargets trg(target_ctx, cfg.path.string());
            trg.resolve_validation_targets({"first.db", "second.db"});
            const auto prefs = tadah::hpo::make_param_ref_table(trg);
            std::vector<double> grad;
            std::vector<int> needs_fd;
            REQUIRE(tadah::hpo::compute_analytical_gradient_from_trainer(
                host, train, val, trg, prefs, grad, needs_fd));
            REQUIRE(needs_fd.empty());
            REQUIRE(grad.size() == 1);
            auto loss_at = [&](double x) {
                auto cg = cgrid; cg[0] = x;
                MLIPWorkspaceManager w;
                auto c = make_minimal_trainer_context_force(cg, sgrid, rcut);
                Trainer t(c, w); t.train(train);
                const auto pred = predict_full(t, val);
                tadah::mlip::Analytics a(val, pred);
                const auto e = a.calc_e_rmse(), f = a.calc_f_rmse();
                double sum = 0;
                if (energy) sum += (trg.globalLoss.loss(e[0]*1000, 15, 0.003)
                                  + trg.globalLoss.loss(e[1]*1000, 5, 0.007)) / 2;
                if (force) sum += (trg.globalLoss.loss(f[0], 0.02, 2)
                                 + trg.globalLoss.loss(f[1], 0.08, 0.5)) / 2;
                return sum / (int(energy) + int(force));
            };
            const double h = 1e-5;
            const double fd = (loss_at(theta+h) - loss_at(theta-h)) / (2*h);
            CHECK(grad[0] == Approx(fd).epsilon(1e-3).margin(1e-7));
        }
    }
}
