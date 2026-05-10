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
        add(r, c);
    }
    {
        OptimizerCaps c;
        c.lib = "TADAH";  c.algo = "GRID";
        c.is_global       = true;
        c.supports_maxeval = false;
        add(r, c);
    }
    {
        OptimizerCaps c;
        c.lib = "TADAH";  c.algo = "ANNEAL";
        c.is_global       = true;
        c.supports_maxeval = true;
        c.supports_maxtime = true;
        c.supports_stopval = true;
        add(r, c);
    }

    /* DLIB ----------------------------------------------------------- */
    {
        OptimizerCaps c;
        c.lib = "DLIB";  c.algo = "GFS";
        c.is_global          = true;
        c.accepts_multi_seed = true;
        c.supports_maxeval   = true;
        c.supports_ftol_abs  = true;     // solver_eps — absolute-style
        c.ladder_ftol        = OptimizerCaps::FtolKind::Abs;
        add(r, c);
    }
    {
        OptimizerCaps c;
        c.lib = "DLIB";  c.algo = "BFGS";
        c.is_gradient_based  = true;
        c.supports_maxeval   = true;
        c.supports_ftol_abs  = true;     // objective_delta_stop_strategy
        c.ladder_ftol        = OptimizerCaps::FtolKind::Abs;
        add(r, c);
    }

    /* CERES — GradientProblemSolver family (LBFGS / WOLFE / NLCG /
     * STEEPEST_DESCENT). All four are local, gradient-based, accept
     * function/parameter tolerance (relative) and a gradient-norm
     * tolerance via PARAM tol_gradinf. The Solver-family (LM, DOGLEG,
     * SUBSPACE_DOGLEG) is intentionally NOT registered: those algorithms
     * are degenerate over a scalar loss and require residual-mode loss
     * exposure plus autodiff to be useful — deferred to a later phase.
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
            c.ladder_ftol        = OptimizerCaps::FtolKind::Rel;
            add(r, c);
        }
    }

    /* NLopt — enumerate the catalogue. Every NLopt algorithm honours
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
        c.ladder_ftol       = OptimizerCaps::FtolKind::Rel;

        const std::string s = name;
        if (!s.empty() && s[0] == 'G')                    c.is_global = true;
        if (s.size() >= 2 && s[1] == 'D')                 c.is_gradient_based = true;
        if (s.find("MLSL")   != std::string::npos ||
            s.find("AUGLAG") != std::string::npos)        c.accepts_inner = true;
        if (s.find("CRS")    != std::string::npos ||
            s.find("ESCH")   != std::string::npos ||
            s.find("ISRES")  != std::string::npos ||
            s.find("MLSL")   != std::string::npos)        c.is_population = true;

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

void apply_ladder_ftol(OptimizerSpec& sp, double tol)
{
    const auto& c = optimizer_caps(sp);
    switch (c.ladder_ftol) {
        case OptimizerCaps::FtolKind::Rel:  sp.stop.ftol_rel = tol; break;
        case OptimizerCaps::FtolKind::Abs:  sp.stop.ftol_abs = tol; break;
        case OptimizerCaps::FtolKind::None: /* nothing useful to set */ break;
    }
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
