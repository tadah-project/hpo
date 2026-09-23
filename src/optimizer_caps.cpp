#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/optimizer_spec.h>

#include <nlopt.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace tadah::hpo {

namespace {

struct CapsKey {
    std::string lib;
    std::string algo;
    bool operator==(const CapsKey& o) const noexcept {
        return lib == o.lib && algo == o.algo;
    }
};
struct CapsKeyHash {
    std::size_t operator()(const CapsKey& k) const noexcept {
        return std::hash<std::string>{}(k.lib) * 1315423911u
             ^ std::hash<std::string>{}(k.algo);
    }
};

using Registry = std::unordered_map<CapsKey, OptimizerCaps, CapsKeyHash>;

void add(Registry& r, OptimizerCaps c)
{
    CapsKey k{c.lib, c.algo};
    r.emplace(k, std::move(c));
}

Registry build_registry()
{
    Registry r;

    /* TADAH built-ins ------------------------------------------------- */
    {
        OptimizerCaps c;
        c.lib = "TADAH";  c.algo = "RANDOM";
        c.is_global       = true;
        c.supports_maxeval = true;
        c.supports_maxtime = true;
        c.supports_stopval = true;
        c.step_use  = StepUse::Rejected;
        c.step_note = "uniform sampling in the OPTIM box has no initial step; remove STEP";
        add(r, c);
    }
    {
        OptimizerCaps c;
        c.lib = "TADAH";  c.algo = "GRID";
        c.is_global       = true;
        c.supports_maxeval = false;
        c.step_use  = StepUse::GridCount;
        c.step_note = "STEP is not the grid resolution; use PARAM GRID_POINTS <n> "
                      "(integer >= 2, default 20 points per dimension)";
        add(r, c);
    }
    {
        OptimizerCaps c;
        c.lib = "TADAH";  c.algo = "ANNEAL";
        c.is_global       = true;
        c.supports_maxeval = true;
        c.supports_maxtime = true;
        c.supports_stopval = true;
        /* eps-NEPS plateau rule (both SA variants): NEPS consecutive
         * rungs whose best improves by less than the tolerance stop the
         * run. */
        c.supports_ftol_rel = true;
        c.supports_ftol_abs = true;
        c.step_use  = StepUse::WidthFraction;
        c.step_note = "set PARAM STEP_FRAC (and PARAM STEP_MODE for ANNEAL_PT) instead";
        add(r, c);
    }
    {
        OptimizerCaps c;
        c.lib = "TADAH";  c.algo = "ANNEAL_PT";
        c.is_global       = true;
        c.is_population   = true;   /* replica ladder */
        c.supports_maxeval = true;
        c.supports_maxtime = true;
        c.supports_stopval = true;
        /* Corana sec.-4 eps-NEPS plateau evaluated on the coldest slot
         * + the global best. */
        c.supports_ftol_rel = true;
        c.supports_ftol_abs = true;
        c.step_use  = StepUse::WidthFraction;
        c.step_note = "set PARAM STEP_FRAC (and PARAM STEP_MODE for ANNEAL_PT) instead";
        add(r, c);
    }
    {
        OptimizerCaps c;
        c.lib = "TADAH";  c.algo = "ANNEAL_PA";
        c.is_global       = true;
        c.is_population   = true;
        c.supports_maxeval = true;
        c.supports_maxtime = true;
        c.supports_stopval = true;
        /* Corana sec.-4 eps-NEPS plateau evaluated on the population
         * best + the global best. */
        c.supports_ftol_rel = true;
        c.supports_ftol_abs = true;
        c.step_use  = StepUse::WidthFraction;
        c.step_note = "set PARAM STEP_FRAC (and PARAM STEP_MODE for ANNEAL_PT) instead";
        add(r, c);
    }

    /* ASA (vendored Ingber ASA-30.53, external/asa) ------------------- */
    {
        OptimizerCaps c;
        c.lib = "ASA";  c.algo = "ASA";
        c.is_global       = true;
        c.supports_maxeval = true;
        c.supports_maxtime = true;
        c.supports_stopval = true;
        /* No ftol/xtol: ASA's own plateau stops are PARAM Cost_Precision
         * + Maximum_Cost_Repeat and Accepted_To_Generated_Ratio. */
        c.step_use  = StepUse::Rejected;
        c.step_note = "the generating distribution width is driven by the per-dim "
                      "temperatures (Initial_Parameter_Temperature, Temperature_Ratio_Scale)";
        add(r, c);
    }

    /* DLIB ----------------------------------------------------------- */
    {
        OptimizerCaps c;
        c.lib = "DLIB";  c.algo = "GFS";
        c.is_global          = true;
        c.accepts_multi_seed = true;
        c.supports_maxeval   = true;
        c.supports_ftol_abs  = true;     // solver_eps -- absolute-style
        c.step_use  = StepUse::Rejected;
        c.step_note = "MaxLIPO+TR sizes its own trust region from the samples; remove STEP";
        add(r, c);
    }
    {
        OptimizerCaps c;
        c.lib = "DLIB";  c.algo = "BFGS";
        c.is_gradient_based  = true;
        c.supports_maxeval   = true;
        c.supports_ftol_abs  = true;     // objective_delta_stop_strategy
        c.step_use  = StepUse::Rejected;
        c.step_note = "the line search starts from -grad; the finite-difference step is PARAM fd_step";
        add(r, c);
    }

    /* CERES -- GradientProblemSolver family (LBFGS / WOLFE / NLCG /
     * STEEPEST_DESCENT). All four are local, gradient-based, accept
     * function/parameter tolerance (relative) and a gradient-norm
     * tolerance via PARAM tol_gradinf. The Solver-family (LM, DOGLEG,
     * SUBSPACE_DOGLEG) is intentionally NOT registered: those algorithms
     * are degenerate over a scalar loss and require residual-mode loss
     * exposure plus autodiff to be useful -- deferred to a later phase.
     * The dispatch in run_policy_base.h short-circuits those names with
     * a tailored message before this caps lookup runs. */
    {
        const char* ceres_grad_algos[] = {
            "LBFGS", "WOLFE", "NLCG", "STEEPEST_DESCENT"
        };
        for (const char* a : ceres_grad_algos) {
            OptimizerCaps c;
            c.lib                = "CERES";
            c.algo               = a;
            c.is_gradient_based  = true;
            c.supports_maxeval   = true;
            c.supports_ftol_rel  = true;
            c.supports_xtol_rel  = true;
            c.step_use           = StepUse::Rejected;
            c.step_note          = "Ceres chooses the initial line-search step itself; remove "
                                   "STEP (PARAM max_ls_iter caps the line-search iterations)";
            c.ignores_bounds     = true;
            c.bounds_note        = "Ceres GradientProblemSolver enforces no box constraints: "
                                   "iterates may leave the OPTIM box (log-scaled dims stay "
                                   "positive; linear dims are unbounded)";
            add(r, c);
        }
    }

    /* NLopt -- enumerate the catalogue. Every NLopt algorithm honours
     * the same StopCriteria via nlopt::opt::set_*; algorithm-class
     * flags are inferred from the canonical name prefix. */
    for (int i = 0; i < NLOPT_NUM_ALGORITHMS; ++i) {
        const char* name = nlopt_algorithm_to_string(static_cast<nlopt_algorithm>(i));
        if (!name || !name[0]) continue;
        OptimizerCaps c;
        c.lib = "NLOPT";  c.algo = name;
        c.supports_maxeval  = true;
        c.supports_maxtime  = true;
        c.supports_stopval  = true;
        c.supports_ftol_rel = true;
        c.supports_ftol_abs = true;
        c.supports_xtol_rel = true;
        c.supports_xtol_abs = true;

        const std::string s = name;
        if (!s.empty() && s[0] == 'G')                    c.is_global = true;
        if (s.size() >= 2 && s[1] == 'D')                 c.is_gradient_based = true;
        if (s.find("MLSL")   != std::string::npos ||
            s.find("AUGLAG") != std::string::npos)        c.accepts_inner = true;
        if (s.find("CRS")    != std::string::npos ||
            s.find("ESCH")   != std::string::npos ||
            s.find("ISRES")  != std::string::npos ||
            s.find("MLSL")   != std::string::npos)        c.is_population = true;

        /* Initial-step consumption, verified against nlopt 2.10
         * src/api/optimize.c (see StepUse doc in optimizer_caps.h). */
        auto has = [&](const char* sub) { return s.find(sub) != std::string::npos; };
        if (has("SBPLX") || has("NELDERMEAD")) {
            c.step_use  = StepUse::PerDimVector;
            c.step_note = "per-dim first simplex edges; no coordinate rescaling; later "
                          "step scale is one L1 progress ratio in [0.1,10] applied to "
                          "every dim; the per-dim x-stop guard is |x|-relative (use XTOL_ABS)";
        } else if (has("BOBYQA") || has("COBYLA")) {
            c.step_use  = StepUse::PerDimRescale;
            c.step_note = "coordinates rescaled by the step vector (rhobeg = step in every "
                          "dim after rescaling); rhoend = XTOL_REL*rhobeg, floored by XTOL_ABS";
        } else if (has("NEWUOA") || has("PRAXIS")) {
            c.step_use  = StepUse::MinCollapsed;
            c.step_note = "uses one radius = min over dims of |step|";
        } else if (has("MMA") || has("CCSAQ")) {
            c.step_use  = StepUse::TrustHalfWidth;
            c.step_note = "per-dim initial trust-region half-width (default 0.5*width)";
        } else if (has("MLSL") || has("AUGLAG")) {
            c.step_use  = StepUse::ForwardedToInner;
            c.step_note = "forwarded to the INNER local optimiser (an OUTER STEP overrides "
                          "an INNER STEP)";
        } else {
            c.step_use  = StepUse::Ignored;
            c.step_note = "this algorithm never reads an initial step (population, "
                          "branch-and-bound, or line-search method); remove STEP";
        }
        if (s == "LN_NEWUOA") {
            c.ignores_bounds = true;
            c.bounds_note    = "LN_NEWUOA is called without bounds (nlopt optimize.c passes "
                               "lb=ub=NULL): evaluations may leave the OPTIM box; use "
                               "LN_NEWUOA_BOUND or LN_BOBYQA";
        }

        add(r, c);
    }

    return r;
}

const Registry& registry()
{
    static const Registry r = build_registry();
    return r;
}

}  // namespace

const char* step_use_name(StepUse u)
{
    switch (u) {
        case StepUse::Rejected:         return "rejected (no initial-step concept)";
        case StepUse::Ignored:          return "ignored by the algorithm (rejected)";
        case StepUse::PerDimVector:     return "per-dim vector, raw simplex edges";
        case StepUse::PerDimRescale:    return "per-dim vector -> coordinate rescaling";
        case StepUse::MinCollapsed:     return "collapsed to min over dims of |step|";
        case StepUse::TrustHalfWidth:   return "per-dim trust-region half-width";
        case StepUse::ForwardedToInner: return "forwarded to the INNER optimiser";
        case StepUse::GridCount:        return "grid point count (use PARAM GRID_POINTS)";
        case StepUse::WidthFraction:    return "family-owned width fraction (PARAM STEP_FRAC)";
    }
    return "?";
}

bool step_use_consumes(StepUse u)
{
    switch (u) {
        case StepUse::PerDimVector:
        case StepUse::PerDimRescale:
        case StepUse::MinCollapsed:
        case StepUse::TrustHalfWidth:
        case StepUse::ForwardedToInner:
            return true;
        default:
            return false;
    }
}

const OptimizerCaps& optimizer_caps(const std::string& lib,
                                    const std::string& algo)
{
    const auto& reg = registry();
    auto it = reg.find({lib, algo});
    if (it == reg.end())
        throw std::runtime_error("unknown optimizer " + lib + ":" + algo +
                                 " (not registered in optimizer_caps)");
    return it->second;
}

const OptimizerCaps& optimizer_caps(const OptimizerSpec& sp)
{
    return optimizer_caps(sp.lib, sp.algo);
}

std::vector<const OptimizerCaps*> filter_optimizer_caps(CapsPredicate pred)
{
    std::vector<const OptimizerCaps*> out;
    for (const auto& kv : registry()) {
        if (pred(kv.second)) out.push_back(&kv.second);
    }
    return out;
}

std::string format_caps_list(const std::vector<const OptimizerCaps*>& caps,
                             std::size_t max_show)
{
    if (caps.empty()) return "<none registered>";
    std::string s;
    const std::size_t n_show = std::min(max_show, caps.size());
    for (std::size_t i = 0; i < n_show; ++i) {
        if (i) s += ", ";
        s += caps[i]->lib;
        s += ":";
        s += caps[i]->algo;
    }
    if (n_show < caps.size()) {
        s += ", ... (";
        s += std::to_string(caps.size() - n_show);
        s += " more)";
    }
    return s;
}

}  // namespace tadah::hpo
