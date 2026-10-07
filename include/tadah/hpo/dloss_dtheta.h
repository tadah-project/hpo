#pragma once
/*
 * dloss_dtheta.h - central HPO-side analytical-gradient driver.
 *
 * Phase 2A.4 lands the orchestration scaffold and the IFT chain-rule
 * building block (ift_rhs). The end-to-end metric aggregation across
 * (truth, pred) StructureDB pairs is staged behind the descriptor-side
 * dPhi assembly that lives in HPO_Host (next sub-task).
 *
 * Components in this header:
 *
 *   ParamRef + make_param_ref_table : flatten optimap into a per-index
 *     view so the orchestrator can decode "what theta does index i
 *     refer to?" without re-walking the std::map.
 *
 *   PerThetaInputs : caller-supplied per-theta design-matrix
 *     derivatives. Filled by the descriptor walker (HPO_Host
 *     integration) or by hand in tests.
 *
 *   DLossDThetaPlan : references-only bundle of live state for the
 *     orchestrator (factor handle, weights, validation StructureDB,
 *     param-ref table, optional per-theta inputs).
 *
 *   detail::ift_rhs : per-theta IFT right-hand side assembly,
 *     suitable for feeding into LinearRegressor::backsolve to obtain
 *     dW_star/dtheta. Validated by Layer-3 parity test.
 *
 *   compute_dloss_dtheta : top-level entry. When per-theta inputs are
 *     missing it routes everything to needs_fd (HYBRID-mode default).
 *     When supplied it computes dW/dtheta via IFT and writes the
 *     dot-product (Phi_val_aggregate . dW/dtheta) to g -- the second
 *     half of dy/dtheta. The first half (the dPhi_val_row . W
 *     contribution) is added by the caller via per_theta.dy_feature
 *     when supplied. The metric chain-rule sum is still TODO; until
 *     it lands the orchestrator's analytical contribution covers the
 *     core IFT math, with metric aggregation deferred (caller passes
 *     the metric scalar separately or HYBRID's masked-FD pass picks
 *     up the full chain).
 *
 * IFT chain rule for ridge regression with fixed lambda:
 *
 *     Aug(Phi) = [ Phi ; sqrt(lambda)*I_no_bias ]
 *     Aug(T)   = [  T  ;          0             ]
 *     factor stores U with U^T U = Aug(Phi)^T Aug(Phi)
 *
 *     residual_aug   = Aug(T) - Aug(Phi) W_star
 *     rhs(theta)     = (dAug(Phi)/dtheta)^T residual_aug
 *                      - Aug(Phi)^T (dAug(Phi)/dtheta) W_star
 *     dW_star/dtheta = backsolve(factor, rhs(theta))
 */

#include <tadah/core/core_types.h>
#include <tadah/core/intercept_layout.h>
#include <tadah/hpo/fit_switch_guard.h>
#include <tadah/hpo/analytical_gate.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/sims_grad.h>
#include <tadah/mlip/analytics/analytics.h>
#include <tadah/mlip/analytics/analytics_grad.h>
#include <tadah/mlip/analytics/dphi_dtheta_walk.h>
#include <tadah/mlip/detail/nn_guard.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>
#include <tadah/models/dc_selector.h>
#include <tadah/models/factor_handle.h>
/* linear_regressor.h is intentionally NOT included here to avoid
 * pulling ols.h's LAPACK template instantiations into HPO TUs (which
 * are compiled with -DTADAH_USE_DLIB; lapack.h's extern "C" block
 * is suppressed under that flag, so an HPO TU that references
 * tadah::core::dgesdd_ etc. fails to compile). The orchestrator's
 * IFT helpers expose ift_rhs only; callers who need to drive the
 * backsolve themselves include linear_regressor.h on the MODELS-side
 * test or HPO_Host integration layer. */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(TADAH_ENABLE_OPENMP)
#   include <omp.h>
#endif

namespace tadah::hpo {

/* ParamRef -- flatten `optimap` into a per-index lookup so the
 * orchestrator can decode "what theta does index i refer to?" without
 * re-walking the map. Built once per gradient call by
 * make_param_ref_table from HPOTargets. */
struct ParamRef {
    /** The OPTIM key (upper-case), e.g. "CGRID2B", "SGRID2B", "RCUT2B".
     *  Matches the keys D2_Base::calc_dDdtheta expects. */
    std::string theta_name;
    /** 1-based index for vector parameters. Always positive -- descriptor
     *  closed forms for vector keys (CGRID*, SGRID*, CEMBFUNC, SEMBFUNC,
     *  ...) require `arg_pos >= 1`. Scalar keys (RCUT*, RCUTENV) ignore
     *  arg_pos and match on theta_name alone, so 1 is fine there too. */
    int         arg_pos = 1;
    /** Flat index in the optimiser's theta vector. */
    std::size_t flat_idx = 0;
};

/** Build a flat param-ref table mirroring HPO_Host::param_labels_.
 *  The returned vector is the size of the optimised theta vector and
 *  ordered the same way as `param_labels_` (and the optimiser's x).
 *
 *  Walks `optimap` in iteration order (std::map: lexicographic by
 *  key), and within each key emits one entry per opt[i]==true position. */
inline std::vector<ParamRef>
make_param_ref_table(const HPOTargets& trg)
{
    std::vector<ParamRef> out;
    std::size_t flat = 0;
    for (const auto& kv : trg.optimap) {
        const std::string& key = kv.first;
        const auto&        op  = kv.second.opt;
        for (std::size_t i = 0; i < op.size(); ++i) {
            if (!op[i]) continue;
            ParamRef r;
            r.theta_name = key;
            r.arg_pos    = static_cast<int>(i + 1);
            r.flat_idx   = flat++;
            out.push_back(std::move(r));
        }
    }
    return out;
}

/* PerThetaInputs -- caller-supplied per-theta design-matrix derivatives
 * for one theta index. The orchestrator uses these to build the IFT
 * RHS and chain through to dy_pred/dtheta. The owning vector lives in
 * DLossDThetaPlan and is indexed in flat-theta order.
 *
 * Layout note: dPhi_train_aug, phi_train_aug, and dPhi_val are flat
 * row-major arrays. The training-side buffers have shape
 *   (n_aug_rows x n_cols)
 * with n_cols = factor.n. The validation-side dPhi_val has shape
 *   (n_val_rows x n_cols)
 * matching the validation Phi layout (no ridge augmentation). */
struct PerThetaInputs {
    /** Augmented dPhi_train/dtheta -- flat row-major (n_aug_rows x n_cols).
     *  Augmentation rows are typically zero (lambda is fixed and the
     *  ridge block does not depend on theta). nullptr => orchestrator
     *  pushes the index onto needs_fd. */
    const double* dPhi_train_aug = nullptr;

    /** dPhi_val/dtheta -- flat row-major (n_val_rows x n_cols). Used by
     *  the metric chain rule to build the dPhi_val_row . W contribution
     *  to dy_pred/dtheta. nullptr is permitted (zero contribution). */
    const double* dPhi_val = nullptr;
};

/* MetricValues -- measured (post-scale) metric values from
 * Analytics + fill_validation, fed into the chain rule alongside the
 * targets in HPOTargets. Mirrors the EvalResult fields populated by
 * fill_validation in the live host; tests build this struct directly. */
struct MetricValues {
    double emae   = 0.0;
    double ermse  = 0.0;
    double errmse = 0.0;
    double ersq   = 0.0;
    double frmse  = 0.0;
    double srmse  = 0.0;
    std::vector<double> v_ermse, v_frmse;
};

/* DLossDThetaPlan -- references-only bundle of live state passed to the
 * orchestrator. Trivially copyable; do not extend lifetimes beyond the
 * gradient call. */
struct DLossDThetaPlan {
    /** Cached Cholesky factor of (Phi^T Phi + lambda I_no_bias) from
     *  the train-side fit. NormalEquations only (Phase 2A scope). */
    const tadah::models::FactorHandle*       factor = nullptr;

    /** Trained weight vector (length factor->n). */
    const tadah::core::t_type*               weights = nullptr;

    /** Augmented training Phi (snapshot from dm.getPhi() after train).
     *  Flat row-major (n_aug_rows x factor.n). */
    const double*                            phi_train_aug = nullptr;
    const double*                            t_train_aug   = nullptr;
    std::size_t                              n_aug_rows    = 0;

    /** Validation Phi (no ridge augmentation), flat row-major
     *  (n_val_rows x factor.n). Used by the metric chain rule to build
     *  the Phi_val_row . dW/dtheta contribution. Required when
     *  per_theta is populated (i.e., when the orchestrator is asked
     *  to assemble dy_pred/dtheta analytically); ignored otherwise.
     *
     *  Layout: per-struct interleaved when val_has_force is true --
     *  for each struct s, 1 energy row + 3.natoms(s) force rows.
     *  When val_has_force is false, just 1 energy row per struct
     *  (n_val_rows == truth_val->size()). */
    const double*                            phi_val   = nullptr;
    std::size_t                              n_val_rows = 0;

    /** Per-struct row offset map for the validation Phi. When
     *  val_has_force is false, val_energy_row_for_struct[s] == s.
     *  When val_has_force is true, val_energy_row_for_struct[s] is
     *  the absolute row index of struct s's energy row in phi_val,
     *  and val_force_row_start_for_struct[s] == val_energy_row_for_struct[s]+1
     *  is the absolute row of (atom 0, axis 0). Force cell (a, k)
     *  lives at val_force_row_start_for_struct[s] + 3.a + k.
     *
     *  Both vectors may be nullptr -- in that case val_cell_to_row falls
     *  back to the energy-only convention (row = struct_idx) and
     *  rejects FORCE / STRESS cells with a tailored error. */
    const std::vector<std::size_t>*          val_energy_row_for_struct      = nullptr;
    const std::vector<std::size_t>*          val_force_row_start_for_struct = nullptr;
    bool                                     val_has_force                  = false;

    /** Active descriptor selector -- present for future cross-cutting
     *  theta aggregation (RCUT2B/RCUTMB across composite descriptors). */
    const tadah::models::DC_Selector*        dcs = nullptr;

    /** Validation truth + predicted StructureDB. Both have the same
     *  size and segment layout (single-segment in Phase 2A). */
    const tadah::mlip::StructureDB*          truth_val = nullptr;
    const tadah::mlip::StructureDB*          pred_val  = nullptr;

    /** Targets configuration (metric switches / weights / loss fn). */
    const HPOTargets*                        trg = nullptr;

    /** Measured metric values (post-scale, matching r.emae etc. in
     *  EvalResult). The orchestrator feeds these into loss_prime to
     *  obtain dL/dmetric for the outer chain step. */
    MetricValues                             metrics{};

    /** Number of loss terms the host averages -- matches r.loss_count:
     *  the active validation metrics, and every PC_OBSERVABLE line and
     *  PC_LAMMPS varloss term. The orchestrator divides the assembled g
     *  by this at the end so the result matches r.loss = r.loss_sum /
     *  loss_count; the observable and LAMMPS chains added afterwards are
     *  divided by it too. */
    std::size_t                              loss_count = 0;

    /** Param-ref table built by make_param_ref_table. Size == g.size(). */
    const std::vector<ParamRef>*             prefs = nullptr;

    /** Per-theta inputs. When null or empty the orchestrator routes
     *  every index to needs_fd (Phase 2A.4 default). */
    const std::vector<PerThetaInputs>*       per_theta = nullptr;

    /** Set true if the eval was clamped (NaN/FAIL_SCORE) -- orchestrator
     *  returns false to the caller, signalling graceful termination. */
    bool                                     was_nan_clamped = false;
};

namespace detail {

/* Compute the IFT RHS for a single theta:
 *
 *   rhs = dPhi^T (T - Phi W) - Phi^T (dPhi W)
 *
 * All inputs are augmented (rows include the ridge augmentation block).
 * Output rhs has size n_cols. Validated against FD via the Layer-3
 * parity test (test_hpo_analytical.cpp).
 *
 * Inputs are flat row-major arrays of size (n_rows * n_cols). */
inline void
ift_rhs(const double*               phi_aug,
        const double*               dphi_aug,
        const double*               t_aug,
        const tadah::core::t_type&  W,
        std::size_t                 n_rows,
        std::size_t                 n_cols,
        tadah::core::t_type&        rhs)
{
    rhs.resize(n_cols);
    for (std::size_t j = 0; j < n_cols; ++j) rhs[j] = 0.0;

    std::vector<double> residual(n_rows, 0.0);
    std::vector<double> dPhiW(n_rows, 0.0);
    for (std::size_t r = 0; r < n_rows; ++r) {
        double phiW = 0.0, dPW = 0.0;
        for (std::size_t c = 0; c < n_cols; ++c) {
            phiW += phi_aug [r * n_cols + c] * W[c];
            dPW  += dphi_aug[r * n_cols + c] * W[c];
        }
        residual[r] = t_aug[r] - phiW;
        dPhiW[r]    = dPW;
    }

    for (std::size_t r = 0; r < n_rows; ++r) {
        for (std::size_t c = 0; c < n_cols; ++c) {
            rhs[c] += dphi_aug[r * n_cols + c] * residual[r];
            rhs[c] -= phi_aug [r * n_cols + c] * dPhiW [r];
        }
    }
}

/* The IFT building block (ift_rhs above + LinearRegressor::backsolve)
 * is the chain-rule core. Composing them is intentionally left to the
 * caller because LinearRegressor::backsolve lives behind a header
 * (linear_regressor.h -> ols.h) that fails to compile in HPO TUs under
 * -DTADAH_USE_DLIB (lapack.h's extern "C" block is suppressed). The
 * MODELS-side test_linear_regressor_ift.cpp validates the composition
 * directly; the HPO_Host integration sub-task uses the hand-rolled
 * Cholesky pair below to do the same math without dragging in
 * linear_regressor.h. */

/* Hand-rolled in-place Cholesky factorisation of a symmetric positive-
 * definite matrix `A` stored column-major in an n*n flat buffer. On
 * success, the upper triangle of `A` holds U with U^T U = A_in.
 *
 * Layout matches LAPACK dpotrf_(uplo='U'): column-major, only the upper
 * triangle is read/written; the lower triangle is left untouched.
 *
 * Returns true on success, false if the matrix is not positive definite
 * (in which case the caller should treat the iter as a clamp event).
 *
 * Why hand-rolled: HPO TUs are compiled with -DTADAH_USE_DLIB which
 * suppresses tadah::core's dpotrf_ declaration (the dlib build path
 * supplies its own LAPACK headers and we have to avoid redeclaration).
 * For the small AtA matrices Phase 2A targets (n_cols on the order of
 * tens), a pure-C++ O(n^3/6) Cholesky is inexpensive -- the descriptor
 * walk and predict path dominate. A LAPACK replacement is a future
 * optimisation if this ever shows up in profiles. */
inline bool cholesky_upper(double* A, int n)
{
    for (int j = 0; j < n; ++j) {
        double diag = A[j + std::size_t(j) * n];
        for (int k = 0; k < j; ++k) {
            const double u_kj = A[k + std::size_t(j) * n];
            diag -= u_kj * u_kj;
        }
        if (!(diag > 0.0)) return false;
        const double u_jj = std::sqrt(diag);
        A[j + std::size_t(j) * n] = u_jj;
        const double inv = 1.0 / u_jj;
        for (int i = j + 1; i < n; ++i) {
            double s = A[j + std::size_t(i) * n];
            for (int k = 0; k < j; ++k)
                s -= A[k + std::size_t(j) * n]
                   * A[k + std::size_t(i) * n];
            A[j + std::size_t(i) * n] = s * inv;
        }
    }
    return true;
}

/* Solve U^T U x = b in place using the upper-triangular factor `U`
 * (column-major n*n, output of cholesky_upper). Equivalent to
 * dpotrs_(uplo='U', N=n, NRHS=1) but pure C++ -- same rationale as
 * cholesky_upper. */
inline void cholesky_solve_uut(const double* U, int n, double* x)
{
    /* Forward substitution: solve U^T y = b, y overwrites x. */
    for (int i = 0; i < n; ++i) {
        double s = x[i];
        for (int k = 0; k < i; ++k)
            s -= U[k + std::size_t(i) * n] * x[k];
        x[i] = s / U[i + std::size_t(i) * n];
    }
    /* Back substitution: solve U z = y, z overwrites x. */
    for (int i = n - 1; i >= 0; --i) {
        double s = x[i];
        for (int k = i + 1; k < n; ++k)
            s -= U[i + std::size_t(k) * n] * x[k];
        x[i] = s / U[i + std::size_t(i) * n];
    }
}

/* Build A^T A in column-major n_cols*n_cols layout from a row-major
 * augmented Phi (n_rows x n_cols). Writes the full matrix (both
 * triangles) so cholesky_upper can read either; matches the dsyrk_+
 * implicit-symmetry convention without depending on LAPACK. */
inline void compute_normal_equations(const double* phi_aug,
                                     std::size_t   n_rows,
                                     std::size_t   n_cols,
                                     std::vector<double>& AtA_out)
{
    AtA_out.assign(n_cols * n_cols, 0.0);
    for (std::size_t r = 0; r < n_rows; ++r) {
        for (std::size_t i = 0; i < n_cols; ++i) {
            const double xi = phi_aug[r * n_cols + i];
            for (std::size_t j = 0; j < n_cols; ++j) {
                AtA_out[i + j * n_cols] += xi * phi_aug[r * n_cols + j];
            }
        }
    }
}

/* Compute dW_star/dtheta for every theta index whose PerThetaInputs
 * carries a non-null dPhi_train_aug. Indices with nullptr are pushed
 * onto `needs_fd`.
 *
 * `dWdtheta_out` is sized to (prefs.size() x n_cols), row-major; rows
 * for needs_fd indices are left zero.
 *
 * Uses one Cholesky factorisation of (Phi_aug^T Phi_aug) shared across
 * all theta partials -- N_theta back-substitutions vs the N_theta full
 * re-solves a finite-difference path would do. Returns false if the
 * matrix is not positive definite (caller should clamp). */
inline bool
compute_dWdtheta(const DLossDThetaPlan&            plan,
                 std::vector<double>&              dWdtheta_out,
                 std::vector<int>&                 needs_fd)
{
    if (!plan.factor || !plan.factor->valid)
        throw std::runtime_error(
            "compute_dWdtheta: FactorHandle is missing or invalid.");
    if (!plan.weights)
        throw std::runtime_error("compute_dWdtheta: weights missing");
    if (!plan.phi_train_aug || !plan.t_train_aug)
        throw std::runtime_error(
            "compute_dWdtheta: phi_train_aug or t_train_aug missing");
    if (!plan.prefs)
        throw std::runtime_error("compute_dWdtheta: prefs missing");

    const std::size_t n_cols = static_cast<std::size_t>(plan.factor->n);
    const std::size_t n_rows = plan.n_aug_rows;
    const auto&       prefs  = *plan.prefs;
    const auto&       W      = *plan.weights;

    /* Build U (Cholesky factor of A^T A) once, shared across all theta. */
    std::vector<double> U;
    compute_normal_equations(plan.phi_train_aug, n_rows, n_cols, U);
    if (!cholesky_upper(U.data(), static_cast<int>(n_cols))) return false;

    dWdtheta_out.assign(prefs.size() * n_cols, 0.0);
    needs_fd.clear();

    if (!plan.per_theta || plan.per_theta->size() != prefs.size()) {
        /* No analytical inputs available -- fall back to FD for everything. */
        for (std::size_t i = 0; i < prefs.size(); ++i)
            needs_fd.push_back(static_cast<int>(prefs[i].flat_idx));
        return true;
    }

    tadah::core::t_type rhs;
    for (std::size_t i = 0; i < prefs.size(); ++i) {
        const PerThetaInputs& in = (*plan.per_theta)[i];
        if (!in.dPhi_train_aug) {
            needs_fd.push_back(static_cast<int>(prefs[i].flat_idx));
            continue;
        }
        ift_rhs(plan.phi_train_aug, in.dPhi_train_aug, plan.t_train_aug,
                W, n_rows, n_cols, rhs);
        cholesky_solve_uut(U.data(), static_cast<int>(n_cols), rhs.ptr());
        for (std::size_t c = 0; c < n_cols; ++c)
            dWdtheta_out[i * n_cols + c] = rhs[c];
    }
    return true;
}

}  // namespace detail

/* Map a (struct_idx, kind, comp) cell -- as emitted by analytics_grad's
 * set_dydp callback -- to its row in the validation Phi.
 *
 * Phase 2A first cut handled ENERGY cells only with row = struct_idx.
 * Phase 2B (C-val) adds FORCE-cell support via the per-struct row maps:
 *
 *   - ENERGY: when energy_row_for_struct is non-null, the row is
 *     (*energy_row_for_struct)[struct_idx]. When null, row = struct_idx
 *     (preserves the energy-only convention used by tests built against
 *     the Phase 2A signature).
 *   - FORCE : requires force_row_start_for_struct (otherwise throws).
 *     row = (*force_row_start_for_struct)[struct_idx] + comp, where
 *     comp = 3*a + k as emitted by analytics_grad::frmse.
 *   - STRESS: still rejected -- stress chain rule needs positions in
 *     the Phi row (Phase 2B sub-step C-stress).
 */
inline std::size_t
val_cell_to_row(const tadah::mlip::StructureDB&         truth_val,
                std::size_t                             struct_idx,
                tadah::mlip::PredCellKind               kind,
                std::size_t                             comp,
                const std::vector<std::size_t>*         energy_row_for_struct      = nullptr,
                const std::vector<std::size_t>*         force_row_start_for_struct = nullptr)
{
    if (struct_idx >= truth_val.size())
        throw std::out_of_range(
            "compute_dloss_dtheta: struct_idx out of range");
    if (kind == tadah::mlip::PredCellKind::ENERGY) {
        return energy_row_for_struct
            ? (*energy_row_for_struct)[struct_idx]
            : struct_idx;
    }
    if (kind == tadah::mlip::PredCellKind::FORCE) {
        if (!force_row_start_for_struct)
            throw std::runtime_error(
                "compute_dloss_dtheta: FORCE cell emitted but the orchestrator "
                "did not build force rows in Phi_val -- pass val_has_force=true "
                "and val_force_row_start_for_struct to enable use_frmse for "
                "the analytical path.");
        return (*force_row_start_for_struct)[struct_idx] + comp;
    }
    throw std::runtime_error(
        "compute_dloss_dtheta: STRESS cells are not yet supported by the "
        "analytical path; disable use_srmse for the analytical strategy or "
        "fall back to GRAD_STRATEGY=NUMERICAL. (Phase 2B sub-step C-stress.)");
}

/* Orchestrator entry point.
 *
 *   plan      : live state references; see DLossDThetaPlan.
 *   x         : current theta vector (real-space; sp.out applied
 *               upstream). Currently unused inside the orchestrator --
 *               the plan carries everything needed.
 *   g         : output gradient; resized to plan.prefs->size() and
 *               pre-zeroed.
 *   needs_fd  : output list of theta indices the orchestrator could
 *               not supply analytically -- passed back to the HYBRID
 *               strategy for masked central-difference fallback.
 *
 * Returns: false if plan.was_nan_clamped is set; true otherwise.
 *
 * Routing:
 *   - When `plan.per_theta` is null/empty, every index is pushed onto
 *     `needs_fd` (HYBRID's masked-FD pass picks them up).
 *   - When `plan.per_theta[i].dPhi_train_aug` is nullptr, that index
 *     is pushed onto `needs_fd`.
 *   - For all other indices the chain rule runs:
 *
 *       dW_star/dtheta_i  = backsolve(factor, ift_rhs(...))     (compute_dWdtheta)
 *       dy_pred[c]/dtheta_i = (dPhi_val[c]/dtheta_i) . W_star
 *                             + Phi_val[c] . (dW_star/dtheta_i)
 *       dmetric_m/dtheta_i = sum_c (dmetric_m/dy_pred[c])
 *                                 . dy_pred[c]/dtheta_i
 *       g[i] = (1/loss_count) sum_m loss_prime(metric_m, target_m, w_m)
 *                                  . dmetric_m/dtheta_i
 */
template<class Vec>
inline bool compute_dloss_dtheta(const DLossDThetaPlan& plan,
                                 const Vec&             /*x*/,
                                 Vec&                   g,
                                 std::vector<int>&      needs_fd,
                                 std::vector<double>*   dWdtheta_out = nullptr)
{
    if (plan.was_nan_clamped) return false;
    if (!plan.prefs)
        throw std::runtime_error("compute_dloss_dtheta: prefs missing");
    if (!plan.factor || !plan.factor->valid)
        throw std::runtime_error(
            "compute_dloss_dtheta: FactorHandle is missing or invalid. "
            "The HPO host must call train_with_factor under "
            "GRAD_STRATEGY=ANALYTICAL/HYBRID with OALGO=4 before "
            "requesting an analytical gradient.");

    const auto& prefs = *plan.prefs;
    g.set_size(static_cast<long>(prefs.size()));
    for (long i = 0; i < g.size(); ++i) g(i) = 0.0;
    needs_fd.clear();
    needs_fd.reserve(prefs.size());

    /* Without per_theta inputs, route everything to FD. */
    if (!plan.per_theta || plan.per_theta->size() != prefs.size()) {
        for (std::size_t i = 0; i < prefs.size(); ++i)
            needs_fd.push_back(static_cast<int>(prefs[i].flat_idx));
        if (dWdtheta_out) dWdtheta_out->clear();
        return true;
    }

    /* IFT chain: build dW/dtheta for every theta whose dPhi_train_aug
     * is supplied. Indices with nullptr dPhi_train_aug end up in
     * needs_fd_internal -- we propagate that to the caller's list.
     *
     * Phase 3.7B: when the caller wants to chain dW/dtheta into a sims_grad
     * computation afterwards (LAMMPS-observable adjoint), it passes a
     * non-null `dWdtheta_out` and reads the buffer once we return. The
     * compute happens once either way -- local storage when out is null,
     * caller's buffer otherwise. */
    std::vector<double> dWdtheta_local;
    std::vector<double>& dWdtheta = dWdtheta_out ? *dWdtheta_out : dWdtheta_local;
    std::vector<int>    needs_fd_internal;
    if (!detail::compute_dWdtheta(plan, dWdtheta, needs_fd_internal)) {
        /* Cholesky failed -> factor non-PD; signal as NaN-clamp. */
        if (dWdtheta_out) dWdtheta_out->clear();
        return false;
    }
    /* Carry through indices the IFT layer punted to FD. */
    for (int idx : needs_fd_internal) needs_fd.push_back(idx);

    /* Per-theta indices that have analytical dPhi_train_aug -- these are
     * the ones we run the metric chain rule on. */
    if (!plan.weights || !plan.trg || !plan.truth_val || !plan.pred_val)
        throw std::runtime_error(
            "compute_dloss_dtheta: weights/trg/truth_val/pred_val "
            "missing -- required when per_theta is populated.");

    const auto& W       = *plan.weights;
    const auto& trg     = *plan.trg;
    const auto& truth   = *plan.truth_val;
    const auto& pred    = *plan.pred_val;
    const std::size_t n_cols = static_cast<std::size_t>(plan.factor->n);

    /* Validate phi_val presence when at least one theta is analytical
     * and there are validation rows to chain through (a run without a
     * validation set -- PC_OBSERVABLE alone -- has none, and no metric to
     * differentiate). If the chain rule is trivial (Phi_val . dW = 0
     * everywhere) the caller can still pass phi_val=nullptr; we'd then
     * only chain through dPhi_val . W. But that's an edge case and an
     * explicit pointer check catches the more common omission. */
    bool any_analytical = false;
    for (const auto& pt : *plan.per_theta) {
        if (pt.dPhi_train_aug != nullptr) { any_analytical = true; break; }
    }
    if (any_analytical && plan.n_val_rows > 0 && !plan.phi_val)
        throw std::runtime_error(
            "compute_dloss_dtheta: phi_val missing -- required for the "
            "Phi_val . dW/dtheta term in the metric chain rule.");

    /* Iterate every analytical theta. The metric chain rule loop is
     * O(N_active_metrics * N_val_cells) per theta; for energy-only
     * single-segment validation this is tiny. */
    for (std::size_t ti = 0; ti < prefs.size(); ++ti) {
        const PerThetaInputs& pt = (*plan.per_theta)[ti];
        if (!pt.dPhi_train_aug) continue;       /* in needs_fd already */

        const std::size_t flat = prefs[ti].flat_idx;
        const double*     dW   = dWdtheta.data() + ti * n_cols;

        /* Pre-compute dy_pred[cell]/dtheta_i = dPhi_val_row . W +
         * Phi_val_row . dW. We do this lazily inside the set_dydp
         * callback (one row per call); for energy-only ENERGY cells
         * the row is just the struct index. */
        auto compute_dy_pred = [&](std::size_t row) -> double {
            double dyp = 0.0;
            if (pt.dPhi_val) {
                for (std::size_t c = 0; c < n_cols; ++c)
                    dyp += pt.dPhi_val[row * n_cols + c] * W[c];
            }
            for (std::size_t c = 0; c < n_cols; ++c)
                dyp += plan.phi_val[row * n_cols + c] * dW[c];
            return dyp;
        };

        /* For each active metric: dmetric/dtheta_i = sum_c (dmetric/dy[c]) *
         * (dy[c]/dtheta_i). Then accumulate
         * loss_prime(metric, target, w) * dmetric/dtheta_i into g[flat].
         *
         * The set_dydp lambda is identical for every metric -- it captures
         * dmetric_dtheta_i by reference and accumulates per cell. We
         * reset dmetric_dtheta_i before each analytics_grad::* call. */
        double      dmetric_dtheta_i = 0.0;
        auto set_dydp = [&](std::size_t struct_idx,
                            tadah::mlip::PredCellKind kind,
                            std::size_t comp,
                            double partial) {
            const std::size_t row =
                val_cell_to_row(truth, struct_idx, kind, comp,
                                plan.val_energy_row_for_struct,
                                plan.val_force_row_start_for_struct);
            dmetric_dtheta_i += partial * compute_dy_pred(row);
        };

        const long flat_l = static_cast<long>(flat);

        // HPO evaluates loss(measured, target, weight); differentiate its
        // first argument, including the measured-value denominator of MAPE.
        auto dl_dmeasured = [&](double measured, double target, double w) {
            return trg.globalLoss.loss_prime_observed(measured, target, w);
        };
        auto dataset_chain = [&](const std::vector<double>& slopes) {
            return [&, slopes](std::size_t dataset, std::size_t struct_idx,
                               tadah::mlip::PredCellKind kind, std::size_t comp,
                               double partial) {
                const auto row = val_cell_to_row(truth, struct_idx, kind, comp,
                    plan.val_energy_row_for_struct, plan.val_force_row_start_for_struct);
                g(flat_l) += slopes.at(dataset) * partial * compute_dy_pred(row);
            };
        };

        if (trg.use_emae) {
            dmetric_dtheta_i = 0.0;
            tadah::mlip::analytics_grad::emae(truth, pred, /*scale=*/1000.0,
                                              set_dydp);
            g(flat_l) += dl_dmeasured(plan.metrics.emae, trg.emae, trg.w_emae)
                       * dmetric_dtheta_i;
        }
        if (trg.use_ermse) {
            const auto slopes = trg.ermse_targets.slopes(trg.globalLoss,
                plan.metrics.ermse, plan.metrics.v_ermse);
            tadah::mlip::analytics_grad::ermse_per_dataset(truth, pred, 1000.0,
                                                          dataset_chain(slopes));
        }
        if (trg.use_errmse) {
            dmetric_dtheta_i = 0.0;
            tadah::mlip::analytics_grad::errmse(truth, pred, /*scale=*/1000.0,
                                                set_dydp);
            g(flat_l) += dl_dmeasured(plan.metrics.errmse, trg.errmse, trg.w_errmse)
                       * dmetric_dtheta_i;
        }
        if (trg.use_ersq) {
            dmetric_dtheta_i = 0.0;
            tadah::mlip::analytics_grad::ersq(truth, pred, /*scale=*/1.0,
                                              set_dydp);
            g(flat_l) += dl_dmeasured(plan.metrics.ersq, trg.ersq, trg.w_ersq)
                       * dmetric_dtheta_i;
        }
        /* use_frmse -- Phase 2B (C-val) lift. Requires plan.val_has_force
         * and plan.val_force_row_start_for_struct to be set by the
         * orchestrator: analytics_grad::frmse emits FORCE cells that
         * val_cell_to_row resolves through those row maps.
         *
         * Scale: fill_validation calls calc_f_rmse() with scale=1.0 (the
         * forces are already in eV/A and frmse is reported in eV/A). */
        if (trg.use_frmse) {
            if (!plan.val_has_force || !plan.val_force_row_start_for_struct)
                throw std::runtime_error(
                    "compute_dloss_dtheta: use_frmse is active but the "
                    "orchestrator did not build val-side force rows. "
                    "compute_analytical_gradient_from_trainer wires this "
                    "automatically when trg.use_frmse=true.");
            const auto slopes = trg.frmse_targets.slopes(trg.globalLoss,
                plan.metrics.frmse, plan.metrics.v_frmse);
            tadah::mlip::analytics_grad::frmse_per_dataset(truth, pred, 1.0,
                                                          dataset_chain(slopes));
        }
        /* use_srmse remains deferred -- analytics_grad::srmse emits STRESS
         * cells which val_cell_to_row rejects (stress requires positions
         * in the Phi row; Phase 2B sub-step C-stress). */
    }

    /* Normalise by loss_count to match r.loss = r.loss_sum / loss_count. */
    if (plan.loss_count > 0) {
        const double inv = 1.0 / static_cast<double>(plan.loss_count);
        for (long i = 0; i < g.size(); ++i) g(i) *= inv;
    }
    return true;
}

/* ------------------------------------------------------------------- *
 *  SimsGradProvider -- host-supplied callback that produces one
 *  SimsGradScriptInput per script in trg.LS, captured at the current
 *  iteration's relaxed state. Returns false if any script could not be
 *  evaluated (LAMMPS instance failed, fail-score clamp, missing variable,
 *  ...); the caller then falls back to all-FD per Phase 3.5's
 *  all-or-nothing policy.
 *
 *  Default (nullptr) -> preserves the pre-3.7B behaviour: when
 *  `trg.LS` is non-empty, `compute_analytical_gradient_from_trainer`
 *  routes every theta to FD via `fall_back_all_fd`. Non-null -> the helper
 *  asks the provider to capture per-script inputs and chains them
 *  through `compute_sims_grad_dWchain` to fold sims_loss gradients
 *  into `g_out`.
 *
 *  The provider returned vector must have `inputs.size() == trg.LS.size()`
 *  with inputs[k] corresponding to LS[k]. Empty trg.LS -> provider is
 *  not invoked.
 * ------------------------------------------------------------------- */
using SimsGradProvider =
    std::function<bool(std::vector<SimsGradScriptInput>& inputs)>;

/* ------------------------------------------------------------------- *
 *  ObservableGradProvider -- typed-observable (PC_OBSERVABLE) analogue
 *  of SimsGradProvider. Host-supplied callback that evaluates every
 *  entry of `trg.observables` at the current iteration's relaxed state
 *  and fills one ObservableGradInput per observable.
 *
 *  Returns false on any per-observable failure (relaxer failure, NaN
 *  energies, model not evaluable); the helper then falls all theta back
 *  to FD per the same all-or-nothing contract that governs
 *  SimsGradProvider.
 *
 *  The provider's returned vector must have
 *  `inputs.size() == trg.observables.size()` with inputs[k]
 *  corresponding to trg.observables[k]. Empty trg.observables -> the
 *  provider is not invoked.
 * ------------------------------------------------------------------- */
using ObservableGradProvider =
    std::function<bool(std::vector<ObservableGradInput>& inputs)>;

/* ------------------------------------------------------------------- *
 *  compute_analytical_gradient_from_trainer
 *
 *  Free helper that takes a live, trained mlip::Trainer plus the
 *  training and validation StructureDBs and produces the analytical
 *  loss gradient for each registered theta. Indices the descriptor walker
 *  cannot supply analytically (or where Phase 2A first-cut restrictions
 *  fail) end up in `needs_fd` so the HYBRID strategy's masked-FD pass
 *  can pick them up.
 *
 *  This is the body of HPO_Host::analytical_gradient extracted as a
 *  free function so it is testable without constructing a full
 *  HPO_Host (which requires real DBFILE inputs and a config.hpo).
 *  HPO_Host::analytical_gradient is a thin wrapper that builds prefs,
 *  retrieves the per-thread trainer cache, and forwards.
 *
 *  Restrictions for the analytical path: STRESS=false, NORM=false,
 *  OALGO=4, LAMBDA != -1, MODEL[1] in {BF_Linear, Kern_Linear},
 *  use_srmse=false. FORCE=true is supported on the train side
 *  (Phase 2B C-step), and use_frmse=true on the val side (Phase 2B C-val)
 *  whatever FORCE is: the val-side descriptors are computed under
 *  validation_switches(trg), so they carry fd_ij whenever use_frmse is
 *  set. On any restriction violation the helper routes
 *  everything to needs_fd and returns true. Non-linear kernels
 *  (Kern_RBF / Kern_Polynomial / Kern_Sigmoid) use EKM projection in
 *  M_KRR_Train, so the analytical chain requires dEKM/dtheta --
 *  deferred to Phase 2C.
 *
 *  `sims_grad_provider` (Phase 3.7B): default-null. When `trg.LS` is
 *  non-empty AND a provider is supplied, the helper drives
 *  `compute_sims_grad_dWchain` on the provider's outputs to fold
 *  LAMMPS-observable gradients into `g_out`. Null provider with non-
 *  empty LS -> falls back to all-FD (byte-identical to pre-3.7B).
 *
 *  `obs_grad_provider` (PC_OBSERVABLE D4b): default-null. When
 *  `trg.observables` is non-empty AND a provider is supplied, the
 *  helper drives `compute_observables_grad_dWchain` on the provider's
 *  outputs to fold typed-observable (cohesive_energy, ...) gradients
 *  into `g_out`. Null provider with non-empty observables -> falls
 *  back to all-FD. Sims and observable chains are independent -- both
 *  fire when both providers are present, and a failure in either
 *  routes every theta to FD (all-or-nothing across both tiers; per-tier
 *  Phase A masking is deferred).
 *
 *  g_out is sized to prefs.size() and pre-zeroed by this helper.
 *  Returns false only when an internal step failed in a way the
 *  optimiser should treat as a clamp event (today: Cholesky non-PD).
 */
inline bool
compute_analytical_gradient_from_trainer(
    mlip::Trainer&                            host,
    const mlip::StructureDB&                  train_stdb,
    const mlip::StructureDB&                  val_stdb,
    const HPOTargets&                         trg,
    const std::vector<ParamRef>&              prefs,
    std::vector<double>&                      g_out,
    std::vector<int>&                         needs_fd,
    const SimsGradProvider&                   sims_grad_provider = nullptr,
    const ObservableGradProvider&             obs_grad_provider  = nullptr)
{
    g_out.assign(prefs.size(), 0.0);
    needs_fd.clear();

    auto fall_back_all_fd = [&]() {
        for (std::size_t i = 0; i < prefs.size(); ++i)
            needs_fd.push_back(static_cast<int>(prefs[i].flat_idx));
        return true;
    };

    /* Phase-timing instrumentation. Enabled when env TADAH_PROFILE_GRADIENT=1
     * is set; emits one stderr line per gradient call breaking down the
     * cost of train Phi rebuild, val Phi build, per-theta loop, and downstream
     * chain rule. Costs nothing when the env var is unset (one getenv per
     * gradient call). */
    const bool profile_grad = [](){
        const char* e = std::getenv("TADAH_PROFILE_GRADIENT");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    using clk = std::chrono::high_resolution_clock;
    auto t_start = clk::now();
    auto t_train_phi = t_start, t_chol = t_start, t_val_phi = t_start,
         t_dm_cache = t_start, t_per_theta = t_start,
         t_chain_rule = t_start;
    auto ms = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    auto& ctx = host.context;

    /* The lists must cover the trainer's cutoff. The loops below compute
     * descriptors inside OpenMP regions; checked here, a short list is an
     * exception the caller catches (it then uses finite differences). */
    {
        const double rc = host.dc.neighbour_cutoff();
        mlip::detail::require_db_nn_cutoff(
            train_stdb, rc, "compute_analytical_gradient_from_trainer");
        mlip::detail::require_db_nn_cutoff(
            val_stdb, rc, "compute_analytical_gradient_from_trainer");
    }

    /* PC_LAMMPS interaction.
     *
     * Pre-Phase 3.7B (Option A from PROMPT_NEXT_NLOPT_LBFGS.md): any
     * trg.LS entry forced a full FD fall-back because sims_loss had no
     * analytical chain. Phase 3 IFT-adjoint changes that -- when the
     * caller supplies a `sims_grad_provider`, the LAMMPS-observable
     * gradient is folded in below via `compute_sims_grad_dWchain`. Null
     * provider keeps the historical fall-back (HYBRID's masked FD picks
     * up both fit_loss and sims_loss components).
     *
     * The fall-back path is byte-identical to the pre-3.7B behaviour
     * when (a) trg.LS is non-empty AND (b) sims_grad_provider is null. */
    if (!trg.LS.empty() && !sims_grad_provider)
        return fall_back_all_fd();

    /* The settings the analytical gradient covers (analytical_gate.h, the
     * rule validate_hpo_setup applies to GRAD_STRATEGY ANALYTICAL):
     *   - training data with every weight fitted;
     *   - STRESS false, no PC_SRMSE: stress rows (a position factor
     *     (ri[x] - rj[x]) in the Phi row) are a separate sub-step;
     *     FORCE=true is supported on the train side (Phase 2C C-step), and
     *     PC_FRMSE on the validation side (C-val) whatever FORCE is;
     *   - NORM false, OALGO 4 with a fixed LAMBDA >= 0;
     *   - MODEL[1] BF_Linear or Kern_Linear. Phase 2A restricted Phi
     *     assembly to BF_Linear (Phi[s, c] = sum_atoms aed_a[c]); Phase 2B
     *     (B-HPO) extends to Kern_Linear: DM_Kern_Linear::calc_phi_energy_row
     *     reduces to the same sum-of-AEDs Phi, and EKM projection is skipped
     *     for Kern_Linear in M_KRR_Train::train1. Non-linear kernels
     *     (Kern_RBF / Kern_Polynomial / Kern_Sigmoid) need a kernel-arm
     *     walker and dEKM/dtheta -- deferred to Phase 2C (the kernel-arm
     *     walker, assemble_dphi_train_energy_rows_d2_kern in MLIP, is
     *     exercised by Layer-1.5 parity tests but not consumed here). */
    if (!analytical_gradient_blocker(ctx, trg, train_stdb.size() > 0).empty())
        return fall_back_all_fd();

    /* Rebuild phi_train_aug from train_stdb + dc.calc rather than
     * reading host.dm.getPhi(). Trainer::dm is *not* populated by
     * Trainer::train (that builds M_BLR's private internal desmat,
     * which is inaccessible from here). The cost is one extra
     * dc.calc(train_stdb) pass per gradient call -- same cost as the
     * predict() pass in operator(), and amortised over all theta partials.
     *
     * Layout matches DesignMatrix's per-struct interleaved post-scale
     * output:
     *   - Per struct s: 1 energy row, then (if FORCE) 3.natoms force
     *     rows for atoms 0..natoms-1 x axis 0..2.
     *   - After all data rows: n_cols ridge augmentation rows
     *     (sqrt(lambda) on the no-bias diagonal).
     *
     * Row offsets:
     *   - energy_row[s]      = sum_{s'<s} (1 + (FORCE ? 3.natoms(s') : 0))
     *   - force_row_start[s] = energy_row[s] + 1   (only used if FORCE)
     *   - ridge block starts at m_data. */
    const bool        use_force = ctx.template get<bool>("FORCE");
    const std::size_t n_cols    = host.DCS.descriptor_size();
    /* The descriptors read below (host.dc, DSIZE entries per atom) must
     * have the DC_Selector width n_cols the design matrix is built with
     * (one intercept column per ATOM species with BIAS true; the Trainer
     * constructor requires it, mlip::require_descriptor_layout). */
    if (n_cols != ctx.template get<std::size_t>("DSIZE"))
        throw std::logic_error(
            "compute_analytical_gradient_from_trainer: the DC_Selector "
            "descriptor has " + std::to_string(n_cols) + " entries, the "
            "trainer's DSIZE is " +
            std::to_string(ctx.template get<std::size_t>("DSIZE")));
    const double      eweightglob = ctx.template get<double>("EWEIGHT");
    const double      fweightglob = ctx.template get<double>("FWEIGHT");

    std::vector<std::size_t> energy_row_for_struct(train_stdb.size(), 0);
    std::vector<std::size_t> force_row_start_for_struct(train_stdb.size(), 0);
    {
        std::size_t row_cursor = 0;
        for (std::size_t s = 0; s < train_stdb.size(); ++s) {
            energy_row_for_struct[s]      = row_cursor;
            force_row_start_for_struct[s] = row_cursor + 1;
            row_cursor += 1;
            if (use_force) row_cursor += 3 * train_stdb(s).natoms();
        }
    }
    const std::size_t m_data    = energy_row_for_struct.empty()
        ? 0
        : (energy_row_for_struct.back() + 1
           + (use_force ? 3 * train_stdb(train_stdb.size() - 1).natoms() : 0));
    const std::size_t n_aug_rows = m_data + n_cols;

    std::vector<double> phi_train_aug(n_aug_rows * n_cols, 0.0);
    std::vector<double> t_train_aug  (n_aug_rows,           0.0);

    /* Per-struct Phi rebuild is embarrassingly parallel: each iteration
     * writes to its own energy_row_for_struct[s] and force_row_start[s]
     * slots in phi_train_aug / t_train_aug (offsets are unique per s by
     * construction). host.dc.calc is const, holds no mutable state, and
     * returns a fresh StDescriptors per call (already used in parallel
     * by DescriptorsCalc::calc(stdb) in MLIP). */
    /* An exception may not leave the OpenMP loop (std::terminate): each
     * iteration keeps its own, and the lowest index's is rethrown after it. */
    std::vector<std::exception_ptr> train_phi_errors(train_stdb.size());
#if defined(TADAH_ENABLE_OPENMP)
#   pragma omp parallel for schedule(dynamic)
#endif
    for (std::size_t s = 0; s < train_stdb.size(); ++s) {
        try {
            const auto&        st = train_stdb(s);
            mlip::StDescriptors std_d = host.dc.calc(st);

            /* Energy row: sum atoms' aed, scaled by sqrt(eweight*eweightglob). */
            const double w_e = std::sqrt(st.eweight * eweightglob);
            const std::size_t e_off = energy_row_for_struct[s] * n_cols;
            for (std::size_t a = 0; a < std_d.naed(); ++a) {
                const tadah::core::aed_type& aed = std_d.get_aed(a);
                for (std::size_t c = 0; c < n_cols; ++c)
                    phi_train_aug[e_off + c] += aed[c];
            }
            for (std::size_t c = 0; c < n_cols; ++c)
                phi_train_aug[e_off + c] *= w_e;
            t_train_aug[energy_row_for_struct[s]] = st.energy * w_e;

            if (use_force) {
                /* Force rows: mirror dm_bf_linear::calc_phi_force_rows
                 *   Phi(row+k, c) -= (fij(c,k) - fji(c,k))
                 * where fij = st_d.fd[a][jj], fji = st_d.fd[j][aa]. Both
                 * sides were already axis-expanded by dc.calc when
                 * FORCE=true (descriptors_calc.hpp:226-238). */
                const double w_f = std::sqrt(st.fweight * fweightglob);
                const std::size_t f_start = force_row_start_for_struct[s];
                for (std::size_t a = 0; a < st.natoms(); ++a) {
                    for (std::size_t jj = 0; jj < std_d.fd[a].size(); ++jj) {
                        const std::size_t j  = st.nn->idx[a][jj];
                        const std::size_t aa = st.get_nn_iindex(a, j, jj);
                        const tadah::core::fd_type& fij = std_d.fd[a][jj];
                        const tadah::core::fd_type& fji = std_d.fd[j][aa];
                        for (std::size_t k = 0; k < 3; ++k) {
                            const std::size_t row_off = (f_start + 3 * a + k) * n_cols;
                            for (std::size_t c = 0; c < n_cols; ++c)
                                phi_train_aug[row_off + c] -= (fij(c, k) - fji(c, k));
                        }
                    }
                    /* Force-row scaling and force targets per axis. */
                    for (std::size_t k = 0; k < 3; ++k) {
                        const std::size_t row = f_start + 3 * a + k;
                        for (std::size_t c = 0; c < n_cols; ++c)
                            phi_train_aug[row * n_cols + c] *= w_f;
                        t_train_aug[row] = st(a).force[k] * w_f;
                    }
                }
            }
        } catch (...) {
            train_phi_errors[s] = std::current_exception();
        }
    }
    for (const auto& e : train_phi_errors)
        if (e) std::rethrow_exception(e);

    /* Ridge augmentation block -- written into the bottom n_cols rows. */
    const double      lambda = ctx.template get<double>("LAMBDA", 0);
    const std::size_t n_bias = tadah::core::n_intercept_columns(ctx);
    if (lambda > 0.0) {
        const double s = std::sqrt(lambda);
        for (std::size_t k = n_bias; k < n_cols; ++k)
            phi_train_aug[(m_data + k) * n_cols + k] = s;
    }

    if (profile_grad) t_train_phi = clk::now();

    tadah::core::t_type W_star = host.model->get_weights();

    /* Build the Cholesky factor. */
    tadah::models::FactorHandle factor;
    {
        std::vector<double> AtA;
        detail::compute_normal_equations(
            phi_train_aug.data(), n_aug_rows, n_cols, AtA);
        if (!detail::cholesky_upper(AtA.data(), static_cast<int>(n_cols)))
            return fall_back_all_fd();
        factor.AtA    = std::move(AtA);
        factor.n      = static_cast<int>(n_cols);
        factor.n_bias = n_bias;
        factor.algo   = tadah::models::Algorithm::NormalEquations;
        factor.valid  = true;
    }

    if (profile_grad) t_chol = clk::now();

    /* Validation Phi layout. Per-struct interleaved when use_frmse is
     * active (mirrors the train-side layout): 1 energy row + 3.natoms
     * force rows per struct. When use_frmse is off, a single energy row
     * per struct (energy_row_for_struct_val[s] == s -- preserves the
     * Phase 2A scalar layout). The validation descriptors carry their
     * force parts whenever use_frmse is set, whatever FORCE training used:
     * the loop below runs under validation_switches(trg), as the
     * prediction in HPO_Host::operator() does. */
    const bool use_frmse_val = trg.use_frmse;

    std::vector<std::size_t> energy_row_for_struct_val(val_stdb.size(), 0);
    std::vector<std::size_t> force_row_start_for_struct_val(val_stdb.size(), 0);
    std::size_t n_val_rows = 0;
    {
        std::size_t row_cursor = 0;
        for (std::size_t s = 0; s < val_stdb.size(); ++s) {
            energy_row_for_struct_val[s]      = row_cursor;
            force_row_start_for_struct_val[s] = row_cursor + 1;
            row_cursor += 1;
            if (use_frmse_val) row_cursor += 3 * val_stdb(s).natoms();
        }
        n_val_rows = row_cursor;
    }

    /* Build phi_val and pred_val via dc.calc -> sum AEDs (energy rows)
     * and the antisymmetric fij - fji pattern (force rows, when
     * use_frmse_val). Force prediction is phi_val_force_row . W*. */
    mlip::StructureDB pred_val;
    pred_val.dbidx = val_stdb.dbidx;
    pred_val.structures.resize(val_stdb.size());

    std::vector<double> phi_val(n_val_rows * n_cols, 0.0);
    /* Same per-struct parallelism as the train-side build: pred_val.structures
     * is pre-sized and each iteration writes only to its own [s] slot;
     * phi_val rows are non-overlapping by row-offset construction. */
    /* An exception may not leave the OpenMP loop (std::terminate): each
     * iteration keeps its own, and the lowest index's is rethrown after it. */
    std::vector<std::exception_ptr> val_phi_errors(val_stdb.size());
    {
    /* dc.calc reads FORCE/STRESS from the trainer's context: set them to
     * what validation measures for this loop only (outside the parallel
     * region), and restore the training ones after it. */
    const auto val_sw = detail::validation_switches(trg);
    detail::ScopedFitSwitches val_switches(ctx, val_sw.force, val_sw.stress);
#if defined(TADAH_ENABLE_OPENMP)
#   pragma omp parallel for schedule(dynamic)
#endif
    for (std::size_t s = 0; s < val_stdb.size(); ++s) {
        try {
            const auto& st = val_stdb(s);
            mlip::StDescriptors std_d = host.dc.calc(st);

            const std::size_t e_off = energy_row_for_struct_val[s] * n_cols;
            for (std::size_t a = 0; a < std_d.naed(); ++a) {
                const tadah::core::aed_type& aed = std_d.get_aed(a);
                for (std::size_t c = 0; c < n_cols; ++c)
                    phi_val[e_off + c] += aed[c];
            }
            double pred_e = 0.0;
            for (std::size_t c = 0; c < n_cols; ++c)
                pred_e += phi_val[e_off + c] * W_star[c];
            pred_val.structures[s].atoms.resize(st.natoms());
            pred_val.structures[s].energy = pred_e;

            if (use_frmse_val) {
                /* Force rows + per-atom force predictions. */
                const std::size_t f_start = force_row_start_for_struct_val[s];
                for (std::size_t a = 0; a < st.natoms(); ++a) {
                    for (std::size_t jj = 0; jj < std_d.fd[a].size(); ++jj) {
                        const std::size_t j  = st.nn->idx[a][jj];
                        const std::size_t aa = st.get_nn_iindex(a, j, jj);
                        const tadah::core::fd_type& fij = std_d.fd[a][jj];
                        const tadah::core::fd_type& fji = std_d.fd[j][aa];
                        for (std::size_t k = 0; k < 3; ++k) {
                            const std::size_t row_off = (f_start + 3 * a + k) * n_cols;
                            for (std::size_t c = 0; c < n_cols; ++c)
                                phi_val[row_off + c] -= (fij(c, k) - fji(c, k));
                        }
                    }
                    for (std::size_t k = 0; k < 3; ++k) {
                        const std::size_t row = f_start + 3 * a + k;
                        double pf = 0.0;
                        for (std::size_t c = 0; c < n_cols; ++c)
                            pf += phi_val[row * n_cols + c] * W_star[c];
                        pred_val.structures[s].atoms[a].force[k] = pf;
                    }
                }
            }
        } catch (...) {
            val_phi_errors[s] = std::current_exception();
        }
    }
    }  // the training FORCE/STRESS are restored here
    for (const auto& e : val_phi_errors)
        if (e) std::rethrow_exception(e);

    if (profile_grad) t_val_phi = clk::now();

    /* Compute metric values via Analytics + the same scaling as
     * fill_validation. Active-metric count goes into loss_count. */
    DLossDThetaPlan plan;
    {
        mlip::Analytics a(val_stdb, pred_val);
        std::size_t lc = 0;
        if (trg.use_emae)   { plan.metrics.emae   = a.calc_e_mae().mean()   * 1000.0; ++lc; }
        if (trg.use_ermse) {
            const auto v = a.calc_e_rmse();
            plan.metrics.ermse = v.mean() * 1000.0;
            for (std::size_t i = 0; i < v.size(); ++i) plan.metrics.v_ermse.push_back(v[i] * 1000.0);
            ++lc;
        }
        if (trg.use_errmse) { plan.metrics.errmse = a.calc_e_rrmse().mean() * 1000.0; ++lc; }
        if (trg.use_ersq)   { plan.metrics.ersq   = a.calc_e_r_sq().mean()  * 1.0;    ++lc; }
        if (trg.use_frmse) {
            const auto v = a.calc_f_rmse();
            /* The average over the measured datasets (a dataset without
             * forces is not measured: nan, slope 0). */
            const auto& measured = trg.frmse_targets.measured();
            plan.metrics.frmse = measured.empty() ? v.mean()
                                                  : mlip::measured_mean(v, measured);
            for (std::size_t i = 0; i < v.size(); ++i) plan.metrics.v_frmse.push_back(v[i]);
            ++lc;
        }
        /* The host averages every loss term (fit_loss, sims_loss,
         * obs_loss count into one r.loss_count). */
        for (const auto& lsim : trg.LS) lc += lsim.i_varloss.size();
        lc += trg.observables.size();
        plan.loss_count = lc;
    }

    /* DM rho/F'(rho) cache (Phase 2E P3). The DM walker's per-atom pre-pass
     * (calc_rho over neighbours + calc_aed to fill F'(rho) tail) is
     * theta-INVARIANT -- neither calc_rho nor calc_aed take a theta_name. Build
     * the cache once per gradient call (train + val) and reuse it across
     * all theta. build_dm_rho_cache parallelises internally.
     *
     * Bench knob: TADAH_DISABLE_DM_CACHE=1 skips the cache build and routes
     * the per-theta loop through the non-cached combined walker. Used by P5 to
     * measure the cache's marginal contribution on the live config; the
     * cache itself stays opt-in via this knob until we have empirical data
     * showing it consistently wins (the cache's memory-traffic vs
     * compute-savings trade-off depends on the descriptor's per-atom rho
     * size and the validation-set scale). */
    const bool disable_dm_cache = [](){
        const char* e = std::getenv("TADAH_DISABLE_DM_CACHE");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    mlip::DMRhoCache dm_cache_train;
    mlip::DMRhoCache dm_cache_val;
    if (host.DCS.dmb && !disable_dm_cache) {
        mlip::build_dm_rho_cache(train_stdb, *host.DCS.dmb,
                                 dm_cache_train, n_cols);
        mlip::build_dm_rho_cache(val_stdb,   *host.DCS.dmb,
                                 dm_cache_val,   n_cols);
    }

    if (profile_grad) t_dm_cache = clk::now();

    /* Per-theta buffers (own storage; plan holds raw pointers into these).
     * Train side is laid out in the interleaved Phi_aug shape; val side
     * matches the n_val_rows-wide phi_val layout (with force rows when
     * use_frmse_val is true). */
    std::vector<std::vector<double>> dphi_train_buf(prefs.size());
    std::vector<std::vector<double>> dphi_val_buf  (prefs.size());
    std::vector<PerThetaInputs>      per_theta     (prefs.size());

    /* Per-theta loop is embarrassingly parallel: each iteration i writes to
     * its own dphi_train_buf[i] / dphi_val_buf[i] / per_theta[i] slot,
     * and the descriptor walkers are const on host.DCS, train_stdb,
     * val_stdb. Scratch buffers are declared INSIDE the loop body so
     * each thread owns its own copy under OMP (heap churn is ~29 allocs
     * per gradient call, negligible vs the descriptor walk cost).
     *
     * Cost-aware scheduling (Phase 2F): on a config with n_theta close
     * to n_threads (a production blip6 config: 29 theta / 26 threads) and a long tail of
     * near-equal-cost heavy theta, plain schedule(dynamic) can leave a
     * heavy theta in the queue at wave-2 -- the cheap-theta threads pick it up
     * and the wall doubles. Sort theta by descending last-call cost (a
     * static cache keyed by theta_name[arg_pos]) so the heaviest enter
     * the queue first; threads converge on the heavy wave together,
     * and the tail becomes a batch of cheap theta that fits in any gap.
     * Iteration writes still go to dphi_*_buf[i] / per_theta[i] (the
     * descriptor-side i), so caller-visible buffer layout is
     * unchanged -- sorting only reorders the OMP dispatch sequence. */
    static std::unordered_map<std::string, double> theta_cost_cache;
    auto cost_key = [](const ParamRef& p) {
        std::string k = p.theta_name;
        k.push_back('[');
        k += std::to_string(p.arg_pos);
        k.push_back(']');
        return k;
    };
    std::vector<std::size_t> theta_order(prefs.size());
    std::iota(theta_order.begin(), theta_order.end(), std::size_t{0});
    {
        std::vector<double> theta_cost_hint(prefs.size(), 0.0);
        for (std::size_t i = 0; i < prefs.size(); ++i) {
            const auto it = theta_cost_cache.find(cost_key(prefs[i]));
            theta_cost_hint[i] = (it == theta_cost_cache.end()) ? 0.0 : it->second;
        }
        /* stable_sort: first call (all costs == 0) keeps unsorted prefs
         * order, paying the tail cost once; subsequent calls converge. */
        std::stable_sort(theta_order.begin(), theta_order.end(),
                         [&](std::size_t a, std::size_t b) {
                             return theta_cost_hint[a] > theta_cost_hint[b];
                         });
    }
    /* Per-theta timing. Always recorded -- the cache that drives next-call
     * scheduling is fed from here, so we cannot gate this on profile_grad.
     * clk::now() is ~30 ns on Linux clock_gettime(MONOTONIC); cost vs the
     * ms-scale per-theta work is invisible. */
    std::vector<double> per_theta_ms(prefs.size(), 0.0);
    /* Per-theta queue start offset (ms from per-theta loop entry) -- lets the
     * profile output identify which theta are dispatched LAST, the metric
     * that matters for tail behaviour. Filled under profile_grad. */
    std::vector<double> per_theta_start_ms(prefs.size(), 0.0);
    const auto t_loop_start = clk::now();
    /* An exception may not leave the OpenMP loop (std::terminate): each
     * iteration keeps its own, and the lowest index's is rethrown after it. */
    std::vector<std::exception_ptr> per_theta_errors(theta_order.size());
#if defined(TADAH_ENABLE_OPENMP)
#   pragma omp parallel for schedule(dynamic, 1)
#endif
    for (std::size_t ord = 0; ord < theta_order.size(); ++ord) {
        try {
            const std::size_t i = theta_order[ord];
            const auto t_th_start = clk::now();
            if (profile_grad) per_theta_start_ms[i] = ms(t_loop_start, t_th_start);
            /* Thread-local scratch (one struct's energy row, dense). */
            std::vector<double> energy_scratch_train(train_stdb.size() * n_cols, 0.0);
            std::vector<double> energy_scratch_val  (val_stdb.size()   * n_cols, 0.0);

            dphi_train_buf[i].assign(n_aug_rows * n_cols, 0.0);
            dphi_val_buf  [i].assign(n_val_rows * n_cols, 0.0);

            /* Combined energy + force walker with rho cache (Phase 2E P1+P3):
             * one pass per side sharing the per-pair iteration (D2) and the
             * per-atom rho/drho pre-pass (DM) across the energy and force
             * derivative kernels, AND reusing the theta-independent rho/F'(rho)
             * field across all theta via dm_cache_{train,val}. Pre-Phase-2E this
             * was four walker calls per theta each rebuilding the rho pre-pass;
             * Phase 2E folds the rho pre-pass into a single one-shot build
             * outside the per-theta loop.
             *
             * Train-side energy rows go to energy_scratch_train (dense
             * n_struct x n_cols, copied later into the interleaved
             * dphi_train_buf with weight scaling). Force rows go directly
             * into the interleaved dphi_train_buf at the force-row offsets. */
            const auto pr_train = disable_dm_cache
                ? mlip::assemble_dphi_train_both_rows(
                      train_stdb, host.DCS,
                      prefs[i].theta_name, prefs[i].arg_pos,
                      energy_scratch_train.data(),
                      use_force ? dphi_train_buf[i].data() : nullptr,
                      n_cols, force_row_start_for_struct)
                : mlip::assemble_dphi_train_both_rows_cached(
                      train_stdb, host.DCS, dm_cache_train,
                      prefs[i].theta_name, prefs[i].arg_pos,
                      energy_scratch_train.data(),
                      use_force ? dphi_train_buf[i].data() : nullptr,
                      n_cols, force_row_start_for_struct);
            const bool ok_e_train = pr_train.first;
            const bool ok_f_train = pr_train.second;

            /* Val-side: same cached walker on val_stdb with dm_cache_val.
             * No weight scaling (val Phi has no eweight/fweight factors --
             * they only weigh the training fit). force_row_start_for_struct_val
             * is consumed only when use_frmse_val is true. */
            const auto pr_val = disable_dm_cache
                ? mlip::assemble_dphi_train_both_rows(
                      val_stdb, host.DCS,
                      prefs[i].theta_name, prefs[i].arg_pos,
                      energy_scratch_val.data(),
                      use_frmse_val ? dphi_val_buf[i].data() : nullptr,
                      n_cols, force_row_start_for_struct_val)
                : mlip::assemble_dphi_train_both_rows_cached(
                      val_stdb, host.DCS, dm_cache_val,
                      prefs[i].theta_name, prefs[i].arg_pos,
                      energy_scratch_val.data(),
                      use_frmse_val ? dphi_val_buf[i].data() : nullptr,
                      n_cols, force_row_start_for_struct_val);
            const bool ok_e_val = pr_val.first;
            const bool ok_f_val = pr_val.second;

            /* Train-side ownership requires BOTH energy and (if FORCE) force
             * walkers to claim the theta. A descriptor that owns theta for
             * energy but not yet for force (e.g. D2_LJ, D2_ZBL today: only
             * calc_dDdtheta is implemented) would otherwise produce a
             * dphi_train_aug with correct energy rows but zero force rows,
             * silently giving the IFT a wrong gradient. Force the whole
             * theta to needs_fd instead.
             *
             * Val-side ownership: when use_frmse_val is true, both energy
             * and force walkers must claim the theta for val too -- same
             * argument applies (a missing val force walker would silently
             * zero the dPhi_val force rows). */
            /* No validation structures (a PC_OBSERVABLE-only run): there are
             * no validation rows to differentiate, so the val side has nothing
             * to claim. */
            const bool ok_train = ok_e_train && (!use_force      || ok_f_train);
            const bool ok_val   = val_stdb.size() == 0 ||
                                  (ok_e_val && (!use_frmse_val || ok_f_val));
            if (!ok_train || !ok_val) continue;     /* -> needs_fd */

            /* Copy energy scratch rows into dphi_train_buf at the energy
             * row offsets and scale by w_e. */
            for (std::size_t s = 0; s < train_stdb.size(); ++s) {
                const double w_e = std::sqrt(train_stdb(s).eweight * eweightglob);
                const std::size_t src_off = s * n_cols;
                const std::size_t dst_off = energy_row_for_struct[s] * n_cols;
                for (std::size_t c = 0; c < n_cols; ++c)
                    dphi_train_buf[i][dst_off + c] = energy_scratch_train[src_off + c] * w_e;
            }
            if (use_force) {
                /* Scale force rows by w_f. */
                for (std::size_t s = 0; s < train_stdb.size(); ++s) {
                    const double w_f = std::sqrt(train_stdb(s).fweight * fweightglob);
                    const std::size_t f_start = force_row_start_for_struct[s];
                    for (std::size_t a = 0; a < train_stdb(s).natoms(); ++a) {
                        for (std::size_t k = 0; k < 3; ++k) {
                            const std::size_t row_off = (f_start + 3 * a + k) * n_cols;
                            for (std::size_t c = 0; c < n_cols; ++c)
                                dphi_train_buf[i][row_off + c] *= w_f;
                        }
                    }
                }
            }

            /* Copy val energy scratch rows into dphi_val_buf at the
             * energy-row offsets. No weight scaling on the val side. */
            for (std::size_t s = 0; s < val_stdb.size(); ++s) {
                const std::size_t src_off = s * n_cols;
                const std::size_t dst_off = energy_row_for_struct_val[s] * n_cols;
                for (std::size_t c = 0; c < n_cols; ++c)
                    dphi_val_buf[i][dst_off + c] = energy_scratch_val[src_off + c];
            }
            /* Val force rows were written directly by the walker into the
             * interleaved positions -- no copy needed, and no weight scaling. */

            per_theta[i].dPhi_train_aug = dphi_train_buf[i].data();
            per_theta[i].dPhi_val       = dphi_val_buf  [i].data();

            per_theta_ms[i] = ms(t_th_start, clk::now());
        } catch (...) {
            per_theta_errors[ord] = std::current_exception();
        }
    }
    for (const auto& e : per_theta_errors)
        if (e) std::rethrow_exception(e);

    if (profile_grad) t_per_theta = clk::now();

    /* Feed the scheduling cache for the next gradient call. Skipped theta
     * (continue branches above) keep per_theta_ms[i] ~= 0 -- fine, they
     * route to FD anyway and will sort to the tail next call. */
    for (std::size_t i = 0; i < prefs.size(); ++i)
        theta_cost_cache[cost_key(prefs[i])] = per_theta_ms[i];

    plan.factor        = &factor;
    plan.weights       = &W_star;
    plan.phi_train_aug = phi_train_aug.data();
    plan.t_train_aug   = t_train_aug.data();
    plan.n_aug_rows    = n_aug_rows;
    plan.phi_val       = phi_val.data();
    plan.n_val_rows    = n_val_rows;
    plan.val_energy_row_for_struct      = &energy_row_for_struct_val;
    plan.val_force_row_start_for_struct = use_frmse_val
        ? &force_row_start_for_struct_val
        : nullptr;
    plan.val_has_force = use_frmse_val;
    plan.dcs           = &host.DCS;
    plan.truth_val     = &val_stdb;
    plan.pred_val      = &pred_val;
    plan.trg           = &trg;
    plan.prefs         = &prefs;
    plan.per_theta     = &per_theta;

    /* The orchestrator works on a Vec template -- wrap g_out in a
     * minimal adapter (mirrors the synthetic-test VecAdapter style). */
    struct VecAdapterFlat {
        std::vector<double>* data;
        long size() const { return static_cast<long>(data->size()); }
        void set_size(long n) { data->assign(static_cast<std::size_t>(n), 0.0); }
        double& operator()(long i) { return (*data)[static_cast<std::size_t>(i)]; }
        double  operator()(long i) const { return (*data)[static_cast<std::size_t>(i)]; }
    };
    VecAdapterFlat x_dummy{ nullptr };
    /* x is unused inside compute_dloss_dtheta; pass an empty adapter. */
    std::vector<double> x_buf;
    x_dummy.data = &x_buf;
    VecAdapterFlat g_adapter{ &g_out };

    /* Capture dW/dtheta when either chain needs it. Both PC_LAMMPS
     * (Phase 3.7B) and PC_OBSERVABLE (D4b) consume the same dW/dtheta
     * matrix -- gather both flags upfront. The extra cost is one
     * allocation; the actual back-solves happen inside
     * compute_dloss_dtheta either way. */
    const bool want_sims_grad = sims_grad_provider && !trg.LS.empty();
    const bool want_obs_grad  = obs_grad_provider && !trg.observables.empty();
    const bool want_dWdtheta  = want_sims_grad || want_obs_grad;
    std::vector<double> dWdtheta;
    const bool ok = compute_dloss_dtheta(
        plan, x_dummy, g_adapter, needs_fd,
        want_dWdtheta ? &dWdtheta : nullptr);

    /* Chain rule for both LAMMPS-script (Tier 3, Phase 3.7B) and
     * typed-observable (Tier 1/2, D4b) gradients. Runs only when the
     * caller supplied at least one provider AND fit-side
     * compute_dloss_dtheta succeeded; the IFT Cholesky failure path
     * (ok=false) already routes the optimiser to a clamp, so we don't
     * bother chaining downstream losses in that case. */
    if (ok && want_dWdtheta && dWdtheta.size() == prefs.size() * n_cols) {
        /* Build the set of theta for which the fit-side produced an
         * analytical dW/dtheta. needs_fd already lists indices that fell back
         * to FD; the complement is the candidate set for analytical
         * chain rules. Shared by both sims and observable chains so
         * neither tier silently bypasses the FD set. */
        std::unordered_set<int> fd_set(needs_fd.begin(), needs_fd.end());
        std::vector<int> analytical_theta;
        analytical_theta.reserve(prefs.size());
        std::vector<int> theta_row_to_g_idx(prefs.size(), -1);
        for (std::size_t i = 0; i < prefs.size(); ++i) {
            const int flat = static_cast<int>(prefs[i].flat_idx);
            if (fd_set.count(flat) == 0) {
                analytical_theta.push_back(static_cast<int>(i));
                theta_row_to_g_idx[i] = static_cast<int>(i);
            }
        }

        auto fall_back_to_fd = [&]() {
            g_out.assign(prefs.size(), 0.0);
            needs_fd.clear();
            for (std::size_t i = 0; i < prefs.size(); ++i)
                needs_fd.push_back(static_cast<int>(prefs[i].flat_idx));
            return true;
        };

        /* The chains below add each LAMMPS / observable term's
         * dL_k/dtheta; the host loss is the MEAN of every term
         * (plan.loss_count of them), so their sum is divided by it once
         * both have run -- compute_dloss_dtheta already divided the
         * validation-metric part. */
        std::vector<double> chain(prefs.size(), 0.0);

        if (want_sims_grad) {
            std::vector<SimsGradScriptInput> sims_inputs;
            bool provider_ok = false;
            try {
                provider_ok = sims_grad_provider(sims_inputs);
            } catch (const std::exception& e) {
                /* Provider threw -- treat as failure and fall back to FD
                 * for everything (per the helper's general "graceful
                 * degrade" pattern; analytical_gradient surfaces this via
                 * its catch around our call). */
                (void)e;
                provider_ok = false;
            }
            const bool inputs_well_formed =
                provider_ok && sims_inputs.size() == trg.LS.size();
            if (!inputs_well_formed) return fall_back_to_fd();

            /* Phase A masking: compute_sims_grad_dWchain skips entries
             * with `failed=true` (per-script failures from the provider,
             * e.g. LAMMPS run_script divergence on random-init pots).
             * Failed scripts contribute zero to the chain; this is
             * mathematically correct in the FAIL_SCORE=constant region
             * (dFAIL_SCORE/dtheta = 0) and prevents a single Tier-3 script
             * crash from poisoning Tier-1 (PC_OBSERVABLE) analytical
             * gradients via the all-FD gate. */
            std::vector<int> needs_fd_sims;
            const bool sims_ok = compute_sims_grad_dWchain(
                sims_inputs, dWdtheta, n_cols,
                theta_row_to_g_idx, analytical_theta,
                chain, needs_fd_sims);
            if (!sims_ok || !needs_fd_sims.empty()) return fall_back_to_fd();
        }

        if (want_obs_grad) {
            std::vector<ObservableGradInput> obs_inputs;
            bool provider_ok = false;
            try {
                provider_ok = obs_grad_provider(obs_inputs);
            } catch (const std::exception& e) {
                /* Same "provider throws -> all-FD" treatment as the sims
                 * branch. The Observable evaluator runs entirely in
                 * Tadah (no LAMMPS), so the throw set here is mostly
                 * relaxer divergence / NN-finder failure / linear-model
                 * weight mismatch. */
                (void)e;
                provider_ok = false;
            }
            const bool inputs_well_formed =
                provider_ok && obs_inputs.size() == trg.observables.size();
            if (!inputs_well_formed) return fall_back_to_fd();

            /* The value's own dependence on theta at fixed weights: an
             * observable value is linear in model energies
             * (ObservableEvalResult::energy_terms), each E = Phi_row . W_star
             * over its structure's descriptors, so dO/dtheta|_W =
             * sum_k coeff_k * (dPhi_row_k/dtheta . W_star) -- the energy-row
             * walker the training rows use. The relaxed geometry's own
             * dependence on theta adds nothing at a stationary point, and
             * dO/dW . dW_star/dtheta is the chain's other term. Without it the
             * gradient missed the (usually dominant, and mostly cancelling)
             * direct term. A value with dO/dW but no energy terms has an
             * unknown direct term: finite differences for every theta. */
            for (auto& in : obs_inputs) {
                if (in.failed || in.constant || in.dO_dW.size() == 0)
                    continue;
                if (in.energy_terms.empty()) return fall_back_to_fd();
                mlip::StructureDB terms;
                for (const auto& t : in.energy_terms) terms.add(t.structure);
                std::vector<double> rows(terms.size() * n_cols, 0.0);
                in.dO_dtheta_explicit.assign(prefs.size(), 0.0);
                for (const int ti : analytical_theta) {
                    const auto i = static_cast<std::size_t>(ti);
                    std::fill(rows.begin(), rows.end(), 0.0);
                    /* The walker claims a theta for a structure only where
                     * a neighbour pair contributes (none for the isolated
                     * atom's pair terms): an unclaimed row is zero. */
                    (void)mlip::assemble_dphi_train_both_rows(
                        terms, host.DCS, prefs[i].theta_name, prefs[i].arg_pos,
                        rows.data(), /*dphi_force_out=*/nullptr, n_cols,
                        /*force_row_start_per_struct=*/{});
                    double d = 0.0;
                    for (std::size_t k = 0; k < terms.size(); ++k) {
                        double dE = 0.0;
                        for (std::size_t c = 0; c < n_cols; ++c)
                            dE += rows[k * n_cols + c] * W_star[c];
                        d += in.energy_terms[k].coeff * dE;
                    }
                    in.dO_dtheta_explicit[i] = d;
                }
            }

            /* Phase A masking applies symmetrically to PC_OBSERVABLE:
             * a per-observable `failed=true` (relaxer divergence,
             * non-finite value, NN-finder fault) -> skip-and-zero
             * contribution rather than poisoning the whole gradient. */
            std::vector<int> needs_fd_obs;
            const bool obs_ok = compute_observables_grad_dWchain(
                obs_inputs, dWdtheta, n_cols,
                theta_row_to_g_idx, analytical_theta,
                chain, needs_fd_obs);
            if (!obs_ok || !needs_fd_obs.empty()) return fall_back_to_fd();
        }

        const double inv = plan.loss_count > 0
            ? 1.0 / static_cast<double>(plan.loss_count) : 1.0;
        for (std::size_t i = 0; i < g_out.size(); ++i)
            g_out[i] += inv * chain[i];
    }

    if (profile_grad) {
        t_chain_rule = clk::now();
        std::fprintf(stderr,
            "[GRADPROF] total=%.1fms | train_phi=%.1f | chol=%.2f | "
            "val_phi=%.1f | dm_cache=%.1f | per_theta=%.1f | "
            "chain_rule=%.1f | n_train=%zu n_val=%zu n_theta=%zu "
            "n_cols=%zu\n",
            ms(t_start, t_chain_rule),
            ms(t_start,     t_train_phi),
            ms(t_train_phi, t_chol),
            ms(t_chol,      t_val_phi),
            ms(t_val_phi,   t_dm_cache),
            ms(t_dm_cache,  t_per_theta),
            ms(t_per_theta, t_chain_rule),
            train_stdb.size(), val_stdb.size(), prefs.size(), n_cols);
        /* Rank per-theta wall times. The few longest dominate the parallel
         * loop wall-clock when n_theta < n_threads (one thread sits on
         * a slow theta while others finish their batch and idle). */
        std::vector<std::size_t> order(prefs.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&](std::size_t a, std::size_t b){
                      return per_theta_ms[a] > per_theta_ms[b];
                  });
        std::fprintf(stderr, "[GRADPROF-theta] top-6 thetas by wall:");
        for (std::size_t k = 0; k < std::min<std::size_t>(6, order.size()); ++k) {
            const std::size_t i = order[k];
            std::fprintf(stderr, " %s[%d]=%.1fms",
                         prefs[i].theta_name.c_str(), prefs[i].arg_pos,
                         per_theta_ms[i]);
        }
        std::fprintf(stderr, "\n");
        /* Queue-tail diagnostic: which theta were dispatched LAST? Under
         * Phase 2F cost-descending sort, the tail must be cheap theta --
         * if a heavy theta shows up here, the sort isn't doing its job
         * (e.g. cold cache, or cost variance > expected). */
        std::vector<std::size_t> start_order(prefs.size());
        std::iota(start_order.begin(), start_order.end(), std::size_t{0});
        std::sort(start_order.begin(), start_order.end(),
                  [&](std::size_t a, std::size_t b){
                      return per_theta_start_ms[a] > per_theta_start_ms[b];
                  });
        std::fprintf(stderr, "[GRADPROF-theta] last-3 dispatched (queue tail):");
        for (std::size_t k = 0; k < std::min<std::size_t>(3, start_order.size()); ++k) {
            const std::size_t i = start_order[k];
            std::fprintf(stderr, " %s[%d] start=%.1fms wall=%.1fms",
                         prefs[i].theta_name.c_str(), prefs[i].arg_pos,
                         per_theta_start_ms[i], per_theta_ms[i]);
        }
        std::fprintf(stderr, "\n");
    }

    return ok;
}

}  // namespace tadah::hpo
