#include <tadah/hpo/scale_validation.h>
#include <tadah/hpo/log_index.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/phase_specs.h>        // with_block_line
#include <tadah/hpo/initial_step.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/validation_metric.h>   // metric_value_is_finite

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tadah::hpo {

namespace {

std::string spec_name(const OptimizerSpec& sp, const std::string& where)
{
    std::string s = sp.lib + ":" + sp.algo;
    if (!where.empty()) s += " (" + where + ")";
    return s;
}

std::string dim_label(const OptimizerSpec& sp, std::size_t i)
{
    if (i < sp.parameter_labels.size() && !sp.parameter_labels[i].empty())
        return sp.parameter_labels[i];
    return logidx::dim_label(i);
}

bool is_log(const OptimizerSpec& sp, std::size_t i)
{
    return i < sp.log_scaled.size() && sp.log_scaled[i];
}

void warn(const std::string& msg)
{
    std::cerr << "[HPO] " << msg << "\n";
    HPOLog::instance().warning(msg);
}

void check_step_value(double v, const std::string& ctx)
{
    if (!metric_value_is_finite(v) || !(v > 0.0)) {
        std::ostringstream oss;
        oss << ctx << ": STEP must be > 0 and finite (got " << v << ")";
        throw std::runtime_error(oss.str());
    }
    if (v > kStepFracMax) {
        std::ostringstream oss;
        oss << ctx << ": STEP must be a fraction of the OPTIM width in (0, "
            << kStepFracMax << "] (got " << v << "); values above " << kStepFracMax
            << " cannot satisfy BOBYQA's width >= 2*step rule and exceed NLopt's "
               "own 0.25 default";
        throw std::runtime_error(oss.str());
    }
}

void validate_start_points(const OptimizerSpec& sp, const std::string& where)
{
    if (sp.space) {
        for (std::size_t p = 0; p < sp.optim_init.size(); ++p) {
            const auto& x = sp.optim_init[p];
            const std::string what = "INIT[" + where + "] starting point " +
                                     std::to_string(p + 1);
            if (x.size() != sp.space->dim()) {
                std::ostringstream oss;
                oss << what << " has " << x.size() << " coordinate(s), expected "
                    << sp.space->dim();
                throw std::runtime_error(oss.str());
            }
            for (std::size_t i = 0; i < x.size(); ++i) {
                if (sp.space->at(i).tf.kind == ScaleKind::Log && !(x[i] > 0.0)) {
                    std::ostringstream oss;
                    oss << what << ": " << sp.space->labels()[i] << " = " << x[i]
                        << " must be > 0 on a log-scaled dimension (LOG)";
                    throw std::runtime_error(oss.str());
                }
            }
            sp.space->require_inside_real(x, what);
        }
        return;
    }
    const std::size_t dim = sp.low.size();
    if (sp.high.size() != dim)
        throw std::runtime_error(where + ": mismatched lower/upper bound counts");
    for (std::size_t p = 0; p < sp.optim_init.size(); ++p) {
        const auto& x = sp.optim_init[p];
        if (x.size() != dim) {
            std::ostringstream oss;
            oss << "INIT[" << where << "] starting point " << (p + 1)
                << " has " << x.size() << " coordinate(s), expected " << dim;
            throw std::runtime_error(oss.str());
        }
        for (std::size_t i = 0; i < dim; ++i) {
            const double v = x[i];
            if (!std::isfinite(v)) {
                std::ostringstream oss;
                oss << "INIT[" << where << "] starting point " << (p + 1)
                    << ": " << dim_label(sp, i) << " = " << v
                    << " is not finite";
                throw std::runtime_error(oss.str());
            }
            double t = v;
            double lo = sp.low[i], hi = sp.high[i];
            if (is_log(sp, i)) {
                if (!(v > 0.0)) {
                    std::ostringstream oss;
                    oss << "INIT[" << where << "] starting point " << (p + 1)
                        << ": " << dim_label(sp, i) << " = " << v
                        << " must be > 0 on a log-scaled dimension (LOG_HP)";
                    throw std::runtime_error(oss.str());
                }
                t  = std::log(v);
                lo = std::exp(sp.low[i]);
                hi = std::exp(sp.high[i]);
            }
            const double tlo = sp.low[i], thi = sp.high[i];
            if (t < tlo || t > thi) {
                std::ostringstream oss;
                oss << "INIT[" << where << "] starting point " << (p + 1)
                    << ": " << dim_label(sp, i) << " = " << v
                    << " lies outside its OPTIM bounds [" << lo << ", " << hi
                    << "]; move the seed inside the box or widen the box";
                throw std::runtime_error(oss.str());
            }
        }
    }
}

}  // namespace

void validate_optimizer_scaling(const OptimizerSpec& sp,
                                const std::string& where,
                                bool check_start_points)
{
    /* HPOTargets::read() requires LIB and ALGO in every block, INNER
     * included. A spec built outside the parser may leave them "NOT SET";
     * it has no caps entry to check against, and the run rejects it when it
     * looks the optimiser up (run_inner, require_driver). */
    const bool has_backend = (sp.lib != "NOT SET" && sp.algo != "NOT SET");
    if (has_backend) {
        const OptimizerCaps& caps = optimizer_caps(sp);
        const std::string name = spec_name(sp, where);

        /* A refusal of the block's STEP names the line the block closes at
         * (OptimizerSpec::block_line), as the parser's refusals of a block
         * do (with_block_line); a spec built outside the parser has none. */
        try {
            const bool consumes = step_use_consumes(caps.step_use);
            if ((sp.step || !sp.step_overrides.empty()) && !consumes) {
                std::string msg = name + " does not use STEP";
                if (caps.step_note && caps.step_note[0])
                    msg += std::string("; ") + caps.step_note;
                throw std::runtime_error(msg);
            }
            if (sp.step) check_step_value(*sp.step, name);
            for (const auto& ov : sp.step_overrides)
                check_step_value(ov.frac, name + " STEP " + ov.key);
            /* Override targets must exist (unknown key / index are errors)
             * and overrides without a block-wide STEP must cover every
             * dim. */
            if (!sp.step_overrides.empty() && sp.space) {
                resolve_step_overrides(sp);
                require_step_coverage(sp);
            }
        } catch (const std::exception& e) {
            throw std::runtime_error(with_block_line(sp, e.what()));
        }

        if (caps.step_use == StepUse::ForwardedToInner && sp.step && sp.inner &&
            (sp.inner->step || !sp.inner->step_overrides.empty())) {
            std::ostringstream oss;
            oss << name << ": STEP " << *sp.step << " overrides the INNER "
                << sp.inner->lib << ":" << sp.inner->algo << " STEP";
            if (sp.inner->step) oss << " " << *sp.inner->step;
            else                oss << " overrides";
            oss << " (NLopt forwards the outer initial step to the local "
                   "optimiser); keep one of them";
            warn(oss.str());
        }
        /* An optimiser that ignores the bounds: one WARNING per run and
         * backend (this check runs at parse time, at startup and when each
         * phase starts). */
        if (caps.ignores_bounds && caps.bounds_note && caps.bounds_note[0] &&
            HPOLog::instance().first_in_run("ignores the OPTIM bounds: " +
                                            sp.lib + ":" + sp.algo))
            warn(name + ": " + caps.bounds_note);
    }

    if (check_start_points) validate_start_points(sp, where);
}

void audit_search_space_widths(const OptimizerSpec& sp)
{
    const std::size_t dim = sp.low.size();
    if (dim == 0 || sp.high.size() != dim) return;
    double wmin = std::numeric_limits<double>::infinity();
    double wmax = 0.0;
    std::size_t imin = 0, imax = 0;
    std::vector<std::size_t> pinned;
    for (std::size_t i = 0; i < dim; ++i) {
        const double w = sp.high[i] - sp.low[i];
        if (w < wmin) { wmin = w; imin = i; }
        if (w > wmax) { wmax = w; imax = i; }
        const double mag = std::max({std::fabs(sp.low[i]), std::fabs(sp.high[i]), 1.0});
        if (!(w > 1e-12 * mag)) pinned.push_back(i);
    }
    {
        std::ostringstream oss;
        oss << "search space: " << dim << " dim(s); OPTIM width min "
            << wmin << " (" << dim_label(sp, imin) << ") max " << wmax
            << " (" << dim_label(sp, imax) << "); max/min ratio "
            << (wmin > 0.0 ? wmax / wmin : std::numeric_limits<double>::infinity())
            << " (optimiser-internal units)";
        HPOLog::instance().info(oss.str());
    }
    for (std::size_t i : pinned) {
        std::ostringstream oss;
        oss << "search space: " << dim_label(sp, i) << " has OPTIM width "
            << (sp.high[i] - sp.low[i]) << " -- below 1e-12 of its magnitude, "
               "i.e. effectively pinned; drop the OPTIM line or widen the box";
        warn(oss.str());
    }
}

void run_scale_validation(const OptimizerSpec& root, bool check_start_points)
{
    const bool csp = check_start_points;
    /* An INNER search is a local NLopt algorithm: an INNER block of another
     * LIB is refused at startup as not NLopt (require_inner_backend),
     * whatever it names, so it has no scaling of its own to check here
     * (its caps would be looked up for a refused or unknown optimiser). */
    const auto nlopt_inner = [](const OptimizerSpec& s) {
        return s.inner && s.inner->lib == "NLOPT";
    };
    validate_optimizer_scaling(root, "OUTER", csp);
    if (nlopt_inner(root)) validate_optimizer_scaling(*root.inner, "INNER", csp);
    if (root.prestage) {
        validate_optimizer_scaling(*root.prestage, "PRESTAGE", csp);
        if (nlopt_inner(*root.prestage))
            validate_optimizer_scaling(*root.prestage->inner, "PRESTAGE.INNER", csp);
    }

    if (csp) audit_search_space_widths(root);
}

}  // namespace tadah::hpo
