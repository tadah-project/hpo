#pragma once
/*
 * fit_switch_guard -- what a prediction computes, set for one scope.
 *
 * FORCE and STRESS in a trainer's context say what training fits. The same
 * switches decide what a prediction computes: the descriptor calculator
 * reads them from the trainer's context (force/stress descriptor parts),
 * M_Tadah_Base::predict from the context it is given (forces, stress). HPO
 * validation predicts what its metrics measure instead: forces whenever
 * PC_FRMSE is set, stresses whenever PC_SRMSE is set, whatever training
 * fitted. ScopedFitSwitches sets the switches for the validation prediction
 * and restores the trainer's own afterwards; validation_switches() is the
 * one rule for both call sites (HPO_Host::operator() and the analytical
 * gradient).
 */

#include <tadah/core/context.h>
#include <tadah/hpo/hpo_targets.h>

#include <optional>
#include <string>
#include <vector>

namespace tadah::hpo::detail {

/* FORCE and STRESS for a validation prediction: what the metrics measure. */
struct FitSwitches {
    bool force;
    bool stress;
};

inline FitSwitches validation_switches(const HPOTargets& trg)
{
    return {trg.use_frmse, trg.use_srmse};
}

/* Sets `key` to `value` in `ctx`, replacing any stored value. */
inline void set_fit_switch(core::Context& ctx, const char* key, bool value)
{
    ctx.remove(key);
    ctx.add(key, std::string(value ? "true" : "false"));
}

/* Sets FORCE and/or STRESS on `ctx` (a switch given as nullopt is left
 * alone) and restores the exact prior state when the scope ends -- the
 * stored values, or the absence of the key -- also when an exception
 * leaves the scope. The context must not be read by another thread while
 * the guard changes it: HPO's trainers are per thread (tls_trainer_cache_),
 * and a guard opens and closes outside any parallel region. */
class ScopedFitSwitches {
public:
    ScopedFitSwitches(core::Context& ctx, std::optional<bool> force,
                      std::optional<bool> stress)
        : ctx_(ctx)
    {
        if (force)  save_and_set("FORCE", *force);
        if (stress) save_and_set("STRESS", *stress);
    }

    ~ScopedFitSwitches()
    {
        for (auto it = saved_.rbegin(); it != saved_.rend(); ++it) {
            ctx_.remove(it->key);
            if (it->prior)
                for (const auto& v : *it->prior) ctx_.add(it->key, v);
        }
    }

    ScopedFitSwitches(const ScopedFitSwitches&)            = delete;
    ScopedFitSwitches& operator=(const ScopedFitSwitches&) = delete;

private:
    struct Saved {
        const char*                             key;
        std::optional<std::vector<std::string>> prior;
    };

    void save_and_set(const char* key, bool value)
    {
        Saved s{key, std::nullopt};
        /* has_stored, not contains(): contains() inserts the blueprint
         * default, so an absent key would come back stored. */
        if (ctx_.has_stored(key)) s.prior = ctx_(key);
        saved_.push_back(std::move(s));
        set_fit_switch(ctx_, key, value);
    }

    core::Context&     ctx_;
    std::vector<Saved> saved_;
};

}  // namespace tadah::hpo::detail
