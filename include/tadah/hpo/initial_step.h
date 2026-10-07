#pragma once
/*
 * initial_step -- the per-dimension initial-step plan for NLopt.
 *
 * STEP is a dimensionless fraction of each dimension's INTERNAL width
 * (identical meaning to the ANNEAL family's PARAM STEP_FRAC): dx_i =
 * frac_i * (high_i - low_i), with per-key / per-index overrides
 * (`STEP <KEY>[(indices)] <frac>`). The vector is what NLopt's
 * set_initial_step(vector) receives, so
 *   - BOBYQA / COBYLA turn it into their per-dim coordinate rescaling
 *     (a uniform scalar STEP used to disable that rescaling entirely),
 *   - NELDERMEAD / SBPLX use it as the first simplex's edges,
 *   - NEWUOA / PRAXIS collapse it to min_i |dx_i| (uniform in a unit box),
 *   - MMA / CCSAQ take it as the per-dim trust half-width,
 *   - MLSL / AUGLAG forward it to the INNER optimiser.
 *
 * Near-bound handling is per dimension (the old adapter shrank ONE
 * scalar to half the slack of the single tightest dimension, starving
 * every other dimension):
 *   - PerDimVector (NM/SBPLX): when the start is within |dx| of the
 *     upper bound but not of the lower one, dx is NEGATIVE so the first
 *     vertex points inward instead of being clamped/flipped by nldrmd
 *     (NLopt accepts a signed initial step);
 *   - BOBYQA alone (PerDimRescale): a dimension whose slack to the
 *     nearest bound is below |dx| gets dx = 0.75 * slack -- NLopt's own
 *     default near-bound rule -- so BOBYQA neither relocates the warm
 *     start (bobyqa.c: an interior start within rhobeg of a bound is moved
 *     to bound +/- rhobeg) nor rejects the box (width >= 2*rhobeg); a
 *     start exactly ON a bound is left alone (BOBYQA keeps it there).
 *     COBYLA treats the bounds as constraints, and the MinCollapsed
 *     algorithms (NEWUOA, PRAXIS) take one radius, min_i |dx_i|, that a
 *     shrink of one dimension would cut for every dimension; neither
 *     relocates the start, so their steps are not shrunk;
 *   - never zero (NLopt rejects a zero step).
 *
 * STEP unset: NLopt's x-dependent heuristic stays in force
 * (0.25*width, or 0.75*slack when the start is closer than that to a
 * bound); the adapter only reads it back for the journal.
 */

#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/log_index.h>
#include <tadah/hpo/optimizer_spec.h>

#include <cmath>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tadah::hpo {

struct StepPlan {
    bool                     user_set = false;   // STEP / overrides present
    std::string              source;             // one-line provenance
    std::vector<double>      dx;                 // internal units, signed
    std::vector<double>      frac;               // |dx| / internal width
    std::vector<std::string> note;               // per dim: "", "override", ...
};

/* Powell's rule for BOBYQA (XU-XL >= 2*RHOBEG) and NLopt's own default
 * (0.25*width) bound the useful range of a width fraction. */
inline constexpr double kStepFracMax = 0.5;

/* Resolve `STEP <KEY>[(indices)] <frac>` overrides against the search
 * space: returns one optional fraction per dimension. Throws on an
 * override that names a key that is not optimised or an index that is
 * not an OPTIM'd index of that key. Requires a bound space; with none
 * and no overrides returns all-empty. */
inline std::vector<std::optional<double>>
resolve_step_overrides(const OptimizerSpec& sp)
{
    const std::size_t n = sp.low.size();
    std::vector<std::optional<double>> out(n);
    if (sp.step_overrides.empty()) return out;
    if (!sp.space)
        throw std::runtime_error(
            "STEP <KEY> overrides require the OPTIM search space (internal error: "
            "spec has no bound SearchSpace)");
    for (const auto& ov : sp.step_overrides) {
        bool key_seen = false;
        std::vector<std::size_t> matched_idx;
        for (std::size_t i = 0; i < n && i < sp.space->dim(); ++i) {
            const DimSpec& d = sp.space->at(i);
            if (d.key != ov.key) continue;
            key_seen = true;
            if (!ov.indices.empty()) {
                bool hit = false;
                for (std::size_t ui : ov.indices)
                    if (ui == d.user_index) { hit = true; break; }
                if (!hit) continue;
            }
            out[i] = ov.frac;
            matched_idx.push_back(d.user_index);
        }
        if (!key_seen)
            throw std::runtime_error(
                "STEP override names OPTIM key '" + ov.key +
                "' which is not optimised (no OPTIM line for it)");
        for (std::size_t ui : ov.indices) {
            bool found = false;
            for (std::size_t m : matched_idx) if (m == ui) { found = true; break; }
            if (!found)
                throw std::runtime_error(
                    "STEP " + ov.key + "(" + std::to_string(ui) + "): index " +
                    std::to_string(ui) + " is not an OPTIM'd index of " + ov.key);
        }
    }
    return out;
}

/* Overrides without a block-wide STEP must cover every OPTIM'd dim.
 * Throws otherwise. Used by the startup validator (so a PRESTAGE/
 * MULTISTART run cannot burn its budget before the local stage throws)
 * and by make_step_plan. */
inline void require_step_coverage(const OptimizerSpec& sp)
{
    if (sp.step || sp.step_overrides.empty()) return;
    const auto ov = resolve_step_overrides(sp);
    for (std::size_t i = 0; i < ov.size(); ++i)
        if (!ov[i])
            throw std::runtime_error(
                "STEP overrides cover only some dimensions and no block-wide STEP "
                "is set; add `STEP <frac>` or override every OPTIM'd dimension");
}

/* Build the plan. `start_internal` may be empty (INNER specs: the local
 * starts are chosen by MLSL at run time) -- the near-bound rules are
 * then skipped. */
inline StepPlan make_step_plan(const OptimizerSpec&        sp,
                               const std::vector<double>&  start_internal,
                               const OptimizerCaps&        caps)
{
    const std::size_t n = sp.low.size();
    StepPlan plan;
    plan.dx.assign(n, 0.0);
    plan.frac.assign(n, 0.0);
    plan.note.assign(n, "");
    plan.user_set = sp.step.has_value() || !sp.step_overrides.empty();
    if (!plan.user_set) {
        plan.source = "STEP unset -- NLopt default per dim (0.25*width, or 0.75*distance "
                      "to the nearer bound when the start is closer than that)";
        return plan;
    }
    require_step_coverage(sp);
    const auto ov = resolve_step_overrides(sp);
    {
        std::ostringstream src;
        if (sp.step) src << "STEP " << *sp.step;
        else         src << "STEP overrides";
        src << " (fraction of transformed OPTIM width per dim)";
        if (!sp.step_overrides.empty()) {
            src << "; overrides:";
            for (const auto& o : sp.step_overrides) {
                src << " " << o.key;
                if (!o.indices.empty()) {
                    src << "(";
                    for (std::size_t k = 0; k < o.indices.size(); ++k)
                        src << (k ? "," : "") << o.indices[k];
                    src << ")";
                }
                src << " " << o.frac;
            }
        }
        plan.source = src.str();
    }
    const bool have_start = start_internal.size() == n;
    /* BOBYQA is the one algorithm that relocates an interior start lying
     * within rhobeg of a bound and refuses a box narrower than 2*rhobeg
     * (bobyqa.c ~3219-3257); COBYLA treats bounds as constraints,
     * NEWUOA/PRAXIS collapse the vector to min|dx| (so shrinking one dim
     * would shrink every dim's radius) and never relocate. */
    const bool bobyqa = caps.step_use == StepUse::PerDimRescale &&
                        sp.algo.find("BOBYQA") != std::string::npos;
    for (std::size_t i = 0; i < n; ++i) {
        const double width = sp.high[i] - sp.low[i];
        double frac  = ov[i] ? *ov[i] : *sp.step;
        std::string note = ov[i] ? "override" : "";
        if (bobyqa && frac >= kStepFracMax) {
            /* Powell's check is `width >= 2*rhobeg` evaluated in BOBYQA's
             * rescaled coordinates; at exactly 0.5 it fails by one ulp for
             * many width/step combinations. Stay strictly inside. */
            frac = kStepFracMax * (1.0 - 1.0e-9);
            note += (note.empty() ? "" : "; ");
            note += "capped just below width/2 (BOBYQA requires width >= 2*rhobeg)";
        }
        double dx = frac * width;
        if (have_start) {
            const double x = start_internal[i];
            const double slack_lo = x - sp.low[i];
            const double slack_hi = sp.high[i] - x;
            const double slack    = std::min(slack_lo, slack_hi);
            switch (caps.step_use) {
                case StepUse::PerDimVector:
                    if (slack_hi < dx && slack_lo >= dx) {
                        dx = -dx;
                        note += (note.empty() ? "" : "; ");
                        note += "inward (start within one step of the upper bound)";
                    }
                    break;
                case StepUse::PerDimRescale:
                    if (!bobyqa) break;
                    if (slack > 0.0 && slack < dx) {
                        dx = 0.75 * slack;
                        note += (note.empty() ? "" : "; ");
                        note += "near-bound: shrunk to 0.75*slack";
                    } else if (slack <= 0.0) {
                        note += (note.empty() ? "" : "; ");
                        note += "start on a bound (kept)";
                    }
                    break;
                default:
                    break;
            }
        }
        plan.dx[i]   = dx;
        plan.frac[i] = std::fabs(dx) / (width > 0.0 ? width : 1.0);
        plan.note[i] = note;
    }
    return plan;
}

/* Multi-line journal block: provenance, the algorithm's consumption mode
 * (registry text) and one row per dimension with the effective step in
 * internal units, as a width fraction, in real units at the start point
 * (through the search space when bound) and any per-dim note. */
inline std::string format_step_report(const StepPlan&             plan,
                                      const OptimizerSpec&        sp,
                                      const OptimizerCaps&        caps,
                                      const std::vector<double>&  start_internal)
{
    auto pad = [](std::string s, std::size_t w) {
        if (s.size() < w) s.resize(w, ' ');
        return s;
    };
    std::ostringstream os;
    os.setf(std::ios::scientific);
    os.precision(3);
    os << "Initial step (" << sp.lib << ":" << sp.algo << "): " << plan.source << "\n"
       << "  consumption: " << step_use_name(caps.step_use) << " -- " << caps.step_note << "\n"
       << "  " << pad("#", 4) << pad("parameter", 24) << pad("step (internal)", 18)
       << pad("step/width", 12) << pad("step (real @start)", 20) << "note";
    const bool have_start = start_internal.size() == plan.dx.size();
    for (std::size_t i = 0; i < plan.dx.size(); ++i) {
        const std::string lbl = i < sp.parameter_labels.size()
                                ? sp.parameter_labels[i] : logidx::dim_label(i);
        std::ostringstream row;
        row.setf(std::ios::scientific); row.precision(3);
        row << plan.dx[i];
        std::ostringstream fr; fr.setf(std::ios::fixed); fr.precision(4); fr << plan.frac[i];
        std::string real = "-";
        if (have_start) {
            std::ostringstream rs; rs.setf(std::ios::scientific); rs.precision(3);
            if (sp.space && i < sp.space->dim()) {
                const double d = sp.space->real_step_at(i, start_internal[i], plan.dx[i]);
                rs << d;
                if (sp.space->at(i).tf.kind == ScaleKind::Log) {
                    const double x0 = sp.space->to_real(i, start_internal[i]);
                    if (x0 > 0.0) {
                        std::ostringstream m; m.setf(std::ios::fixed); m.precision(3);
                        m << " (x" << (x0 + d) / x0 << ")";
                        rs << m.str();
                    }
                }
            } else {
                const bool lg = i < sp.log_scaled.size() && sp.log_scaled[i];
                rs << (lg ? std::exp(start_internal[i] + plan.dx[i]) - std::exp(start_internal[i])
                          : plan.dx[i]);
            }
            real = rs.str();
        }
        os << "\n  " << pad(std::to_string(i + 1), 4) << pad(lbl, 24) << pad(row.str(), 18)
           << pad(fr.str(), 12) << pad(real, 20) << plan.note[i];
    }
    return os.str();
}

}  // namespace tadah::hpo
