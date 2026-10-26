#include <tadah/hpo/analytical_gate.h>

#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/sims_grad.h>
#include <tadah/mlip/weight_provisioning.h>

#include <cmath>
#include <limits>
#include <sstream>

namespace tadah::hpo {

std::string analytical_gradient_blocker(const core::Context& ctx,
                                        const HPOTargets& trg,
                                        bool has_train_db)
{
    if (!has_train_db)
        return "there is no training DBFILE (it differentiates the "
               "least-squares fit of the weights)";
    if (ctx.contains("FIXINDEX"))
        return "FIXINDEX pins model weights (it assumes every weight is "
               "fitted)";
    if (ctx.get<bool>("STRESS"))
        return "STRESS is true (stress rows are not differentiated)";
    if (ctx.get<bool>("NORM"))
        return "NORM is true (it works on raw features)";
    /* It differentiates the least-squares fit of the training rows
     * (M_BLR, M_KRR implementation 1); implementation 2 solves (K +
     * LAMBDA sigma^2 I) w = t over its basis vectors instead, and reads no
     * OALGO (its fit refuses one but the default). */
    if (mlip::is_krr_implementation_2(ctx)) {
        const std::string kernel = ctx.get<std::string>("MODEL", 1);
        return "the model is MODEL M_KRR " + kernel + " 2, implementation "
               "2 (it differentiates the least-squares fit of the training "
               "rows, that of M_BLR and of MODEL M_KRR " + kernel + " 1, "
               "implementation 1, with OALGO 4; implementation 2 solves (K "
               "+ LAMBDA sigma^2 I) w = t over its basis vectors)";
    }
    if (const int oalgo = ctx.get<int>("OALGO", 0); oalgo != 4)
        return "OALGO is " + std::to_string(oalgo) +
               " (it needs OALGO 4, the Cholesky normal equations)";
    {
        const double lambda = ctx.get<double>("LAMBDA", 0);
        if (std::abs(lambda + 1.0) < std::numeric_limits<double>::min())
            return "LAMBDA is -1, the evidence approximation (it needs a "
                   "fixed LAMBDA >= 0)";
        if (lambda < 0.0) {
            std::ostringstream os;
            os << "LAMBDA is " << lambda << " (it needs a fixed LAMBDA >= 0)";
            return os.str();
        }
    }
    {
        const std::string basis = ctx.get<std::string>("MODEL", 1);
        if (basis != "BF_Linear" && basis != "Kern_Linear")
            return "the basis is " + basis +
                   " (it needs MODEL ... BF_Linear or Kern_Linear)";
    }
    if (trg.use_srmse)
        return "PC_SRMSE_GPA is set (stress metrics are not differentiated)";
    return {};
}

std::string analytical_loss_term_blocker(const HPOTargets& trg)
{
    for (std::size_t k = 0; k < trg.observables.size(); ++k) {
        const auto& obs = trg.observables[k];
        if (!obs) continue;
        const std::string term =
            "the PC_OBSERVABLE term " + observable_journal_label(*obs, k + 1);
        if (!loss_has_derivative_wrt_observed(obs->lossfn()))
            return term + " uses --loss " +
                   LossFunction::to_string(obs->lossfn().type()) +
                   ", which has no derivative with respect to the value";
        if (const std::string gap = obs->analytical_gradient_gap();
            !gap.empty())
            return term + ": " + gap;
    }
    return {};
}

}  // namespace tadah::hpo
