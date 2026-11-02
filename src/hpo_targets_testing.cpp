/* describe_parse(): the canonical text of what HPOTargets::read() set
 * (hpo_targets_testing.h describes the format and what it holds). */

#include <tadah/hpo/hpo_targets_testing.h>

#include <tadah/core/number_format.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/observables/cohesive_energy.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tadah::hpo::testing {

namespace {

std::string quote(const std::string& s)
{
    std::string out = "\"";
    for (const unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20 || c == 0x7f) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out + "\"";
}

std::string num(double v) { return core::format_roundtrip(v); }

std::string integer(std::uint64_t v) { return std::to_string(v); }

std::string signed_integer(long long v) { return std::to_string(v); }

std::string flag(bool b) { return b ? "true" : "false"; }

template <class T, class F>
std::string optional_of(const std::optional<T>& o, F f)
{
    return o ? f(*o) : std::string("none");
}

template <class T, class F>
std::string list_of(const std::vector<T>& v, F f)
{
    std::string out = "[";
    for (std::size_t i = 0; i < v.size(); ++i)
        out += (i ? ", " : "") + f(v[i]);
    return out + "]";
}

std::string strings(const std::vector<std::string>& v)
{
    return list_of(v, quote);
}

std::string numbers(const std::vector<double>& v) { return list_of(v, num); }

/* The keys of a map, in order (the maps of OptimizerSpec are unordered). */
template <class M>
std::vector<std::string> sorted_keys(const M& m)
{
    std::vector<std::string> keys;
    keys.reserve(m.size());
    for (const auto& kv : m) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    return keys;
}

/* FNV-1a (64 bits) of the lines, each followed by a newline. */
std::string fnv1a(const std::vector<std::string>& lines)
{
    std::uint64_t h = UINT64_C(0xcbf29ce484222325);
    const auto add = [&h](unsigned char c) {
        h ^= c;
        h *= UINT64_C(0x100000001b3);
    };
    for (const auto& l : lines) {
        for (const unsigned char c : l) add(c);
        add('\n');
    }
    char buf[24];
    std::snprintf(buf, sizeof buf, "%016llx",
                  static_cast<unsigned long long>(h));
    return buf;
}

const char* strategy_type_word(StrategyType t)
{
    switch (t) {
        case StrategyType::PLAIN:      return "PLAIN";
        case StrategyType::MULTISTART: return "MULTISTART";
        case StrategyType::PRESTAGE:   return "PRESTAGE";
    }
    return "?";
}

const char* stream_phase_word(StreamPhase p)
{
    switch (p) {
        case StreamPhase::Search:     return "Search";
        case StreamPhase::Refinement: return "Refinement";
    }
    return "?";
}

class Dump {
public:
    void put(const std::string& path, const std::string& value)
    {
        out_ += path;
        out_ += " = ";
        out_ += value;
        out_ += '\n';
    }
    std::string text() && { return std::move(out_); }

private:
    std::string out_;
};

void dump_loss(Dump& d, const std::string& p, const LossFunction& f)
{
    d.put(p + ".type", LossFunction::to_string(f.type()));
    d.put(p + ".params", numbers(f.params()));
}

void dump_caps(Dump& d, const std::string& p, const MetricSkipCaps& c)
{
    d.put(p + ".abs_above", optional_of(c.abs_above, num));
    d.put(p + ".rel_above", optional_of(c.rel_above, num));
}

void dump_assignment(Dump& d, const std::string& p,
                     const ValidationMetricAssignment& a)
{
    d.put(p + ".target", num(a.target));
    d.put(p + ".weight", num(a.weight));
    dump_caps(d, p + ".caps", a.caps);
    d.put(p + ".line", integer(a.line));
    d.put(p + ".text", quote(a.text));
}

void dump_metric(Dump& d, const std::string& p,
                 const ValidationMetricTargets& t)
{
    if (const auto& g = t.global_assignment())
        dump_assignment(d, p + ".global", *g);
    else
        d.put(p + ".global", "none");
    d.put(p + ".indexed.size", integer(t.indexed_assignments().size()));
    for (const auto& [index, a] : t.indexed_assignments())
        dump_assignment(d, p + ".indexed[" + integer(index) + "]", a);
}

void dump_init(Dump& d, const std::string& p, const InitStrategySpec& s)
{
    d.put(p + ".chain.size", integer(s.chain.size()));
    for (std::size_t i = 0; i < s.chain.size(); ++i) {
        const auto& e = s.chain[i];
        const std::string q = p + ".chain[" + integer(i) + "]";
        d.put(q + ".key", quote(e.key));
        d.put(q + ".method", quote(e.method));
        d.put(q + ".args", strings(e.args));
        d.put(q + ".file", quote(e.file));
        d.put(q + ".line", integer(e.line));
        d.put(q + ".line_text", quote(e.line_text));
    }
    d.put(p + ".k_total", optional_of(s.k_total, integer));
    /* A seed the constructor drew is not read()'s. */
    d.put(p + ".seed", s.seed_drawn ? std::string("none")
                                    : optional_of(s.seed, integer));
    d.put(p + ".warm_on_mismatch", quote(s.warm_on_mismatch));
    d.put(p + ".line", integer(s.line));
    d.put(p + ".line_text", quote(s.line_text));
}

void dump_spec(Dump& d, const std::string& p, const OptimizerSpec* sp)
{
    if (!sp) {
        d.put(p, "none");
        return;
    }
    const OptimizerSpec& s = *sp;
    d.put(p + ".strategy_type", strategy_type_word(s.strategy_type));
    d.put(p + ".lib", quote(s.lib));
    d.put(p + ".algo", quote(s.algo));
    d.put(p + ".vector_storage", optional_of(s.vector_storage, integer));
    d.put(p + ".pop", optional_of(s.pop, integer));
    d.put(p + ".num_threads", optional_of(s.num_threads, integer));
    /* A seed the constructor drew, or took from the STRATEGY-level SEED
     * for an exploration, is not read()'s. */
    d.put(p + ".seed", s.seed_drawn || s.seed_from_strategy
                           ? std::string("none")
                           : optional_of(s.seed, integer));
    d.put(p + ".stream_phase", stream_phase_word(s.stream_phase));
    d.put(p + ".seed_seeds_exploration", flag(s.seed_seeds_exploration));
    d.put(p + ".step", optional_of(s.step, num));
    d.put(p + ".step_overrides.size", integer(s.step_overrides.size()));
    for (std::size_t i = 0; i < s.step_overrides.size(); ++i) {
        const StepOverride& o = s.step_overrides[i];
        const std::string q = p + ".step_overrides[" + integer(i) + "]";
        d.put(q + ".key", quote(o.key));
        d.put(q + ".indices", list_of(o.indices, integer));
        d.put(q + ".frac", num(o.frac));
        d.put(q + ".line", integer(o.line));
        d.put(q + ".line_text", quote(o.line_text));
    }
    d.put(p + ".grad_strategy",
          s.grad_strategy ? grad_strategy_name(*s.grad_strategy) : "none");
    d.put(p + ".grad_strategy_resolved", flag(s.grad_strategy_resolved));
    d.put(p + ".stop.stopval", optional_of(s.stop.stopval, num));
    d.put(p + ".stop.ftol_rel", optional_of(s.stop.ftol_rel, num));
    d.put(p + ".stop.ftol_abs", optional_of(s.stop.ftol_abs, num));
    d.put(p + ".stop.xtol_rel", optional_of(s.stop.xtol_rel, num));
    d.put(p + ".stop.xtol_abs", optional_of(s.stop.xtol_abs, num));
    d.put(p + ".stop.maxtime", optional_of(s.stop.maxtime, num));
    d.put(p + ".stop.maxeval", optional_of(s.stop.maxeval, integer));
    {
        /* The PARAM fd_step NOISE AUTO derives in the constructor is not
         * read()'s. */
        std::vector<std::string> keys = sorted_keys(s.param);
        if (s.fd_step_from_noise_auto)
            keys.erase(std::remove(keys.begin(), keys.end(), "fd_step"),
                       keys.end());
        d.put(p + ".param.size", integer(keys.size()));
        for (const auto& k : keys)
            d.put(p + ".param[" + k + "]", num(s.param.at(k)));
    }
    d.put(p + ".param_raw.size", integer(s.param_raw.size()));
    for (const auto& k : sorted_keys(s.param_raw))
        d.put(p + ".param_raw[" + k + "]", quote(s.param_raw.at(k)));
    d.put(p + ".param_tokens.size", integer(s.param_tokens.size()));
    for (const auto& k : sorted_keys(s.param_tokens))
        d.put(p + ".param_tokens[" + k + "]", strings(s.param_tokens.at(k)));
    d.put(p + ".log_threshold", optional_of(s.log_threshold, num));
    dump_init(d, p + ".init_strategy_spec", s.init_strategy_spec);
    d.put(p + ".noise.autodetect", flag(s.noise.autodetect));
    d.put(p + ".noise.calibration_n", integer(s.noise.calibration_n));
    d.put(p + ".multistart.n_starts", integer(s.multistart.n_starts));
    d.put(p + ".multistart.budget_per_start",
          optional_of(s.multistart.budget_per_start, integer));
    d.put(p + ".stage_topk", integer(s.stage_topk));
    d.put(p + ".restart.after_stagnation",
          integer(s.restart.after_stagnation));
    d.put(p + ".restart.max", integer(s.restart.max));
    dump_init(d, p + ".restart.init_strategy", s.restart.init_strategy);
    d.put(p + ".restart.line", integer(s.restart.line));
    d.put(p + ".restart.line_text", quote(s.restart.line_text));
    d.put(p + ".source_file", quote(s.source_file));
    d.put(p + ".block_line", integer(s.block_line));
    d.put(p + ".block_line_text", quote(s.block_line_text));
    d.put(p + ".key_lines.size", integer(s.key_lines.size()));
    for (const auto& k : sorted_keys(s.key_lines)) {
        const KeyLine& kl = s.key_lines.at(k);
        const std::string q = p + ".key_lines[" + k + "]";
        d.put(q + ".number", integer(kl.number));
        d.put(q + ".text", quote(kl.text));
        d.put(q + ".block", quote(kl.block));
    }
    dump_spec(d, p + ".inner", s.inner.get());
    dump_spec(d, p + ".prestage", s.prestage.get());
}

void dump_sim(Dump& d, const std::string& p, const LammpsSim& s)
{
    d.put(p + ".line", integer(s.line));
    d.put(p + ".line_text", quote(s.line_text));
    d.put(p + ".is_failprobe", flag(s.is_failprobe));
    d.put(p + ".script", quote(s.script));
    d.put(p + ".script_lines.size", integer(s.script_lines.size()));
    d.put(p + ".script_lines.fnv1a", fnv1a(s.script_lines));
    d.put(p + ".invar", strings(s.invar));
    d.put(p + ".n_outvar", strings(s.n_outvar));
    d.put(p + ".n_varloss", strings(s.n_varloss));
    d.put(p + ".t_varloss", numbers(s.t_varloss));
    d.put(p + ".w_varloss", numbers(s.w_varloss));
    d.put(p + ".i_varloss", list_of(s.i_varloss, integer));
    d.put(p + ".lossFn.size", integer(s.lossFn.size()));
    for (std::size_t i = 0; i < s.lossFn.size(); ++i)
        dump_loss(d, p + ".lossFn[" + integer(i) + "]", s.lossFn[i]);
    d.put(p + ".maxtime", quote(s.maxtime));
    d.put(p + ".maxtime_seconds", optional_of(s.maxtime_seconds, num));
    d.put(p + ".maxtime_source", quote(s.maxtime_source));
    d.put(p + ".own_timer", flag(s.own_timer));
    d.put(p + ".failscore", num(s.failscore));
    d.put(p + ".hotvars.size", integer(s.hotvars.size()));
    for (std::size_t i = 0; i < s.hotvars.size(); ++i) {
        const auto& h = s.hotvars[i];
        const std::string q = p + ".hotvars[" + integer(i) + "]";
        d.put(q + ".name", quote(h.name));
        d.put(q + ".min", num(h.min));
        d.put(q + ".max", num(h.max));
        d.put(q + ".initial", optional_of(h.initial, num));
    }
    d.put(p + ".failvars.size", integer(s.failvars.size()));
    for (std::size_t i = 0; i < s.failvars.size(); ++i) {
        const auto& f = s.failvars[i];
        const std::string q = p + ".failvars[" + integer(i) + "]";
        d.put(q + ".name", quote(f.name));
        d.put(q + ".scale", num(f.scale));
        d.put(q + ".scale_given", flag(f.scale_given));
    }
    d.put(p + ".line_variables.size", integer(s.line_variables.size()));
    for (std::size_t i = 0; i < s.line_variables.size(); ++i) {
        const auto& v = s.line_variables[i];
        const std::string q = p + ".line_variables[" + integer(i) + "]";
        d.put(q + ".name", quote(v.name));
        d.put(q + ".option", quote(v.option));
    }
    d.put(p + ".reads_default_loss", flag(s.reads_default_loss));
    d.put(p + ".reads_default_failscore", flag(s.reads_default_failscore));
    d.put(p + ".reads_default_maxtime", flag(s.reads_default_maxtime));
}

void dump_observable(Dump& d, const std::string& p, const Observable& o)
{
    const auto* ce = dynamic_cast<const CohesiveEnergyObservable*>(&o);
    if (!ce)
        throw std::logic_error(
            "describe_parse: the PC_OBSERVABLE kind '" + o.kind() +
            "' has no description of its options");
    using Config = CohesiveEnergyObservable::Config;
    const Config& c = ce->config();
    d.put(p + ".kind", quote(o.kind()));
    d.put(p + ".species", quote(c.species));
    d.put(p + ".supercell",
          list_of(std::vector<long long>{c.nx, c.ny, c.nz}, signed_integer));
    d.put(p + ".target", num(c.target));
    d.put(p + ".weight", num(c.weight));
    d.put(p + ".a0_min", num(c.a0_min));
    d.put(p + ".a0_max", num(c.a0_max));
    d.put(p + ".a0_initial", optional_of(c.a0_initial, num));
    dump_loss(d, p + ".lossfn", c.lossfn);
    d.put(p + ".lossfn_given", flag(c.lossfn_given));
    d.put(p + ".relax_etol", num(c.relax_etol));
    d.put(p + ".relax_max_wall_time_s", num(c.relax_max_wall_time_s));
    d.put(p + ".relax_maxiter", integer(c.relax_maxiter));
    d.put(p + ".relax_maxeval", integer(c.relax_maxeval));
    d.put(p + ".eq_fmax_threshold", num(c.eq_fmax_threshold));
    d.put(p + ".relax_ptol", num(c.relax_ptol));
    d.put(p + ".unconverged_strategy",
          c.unconverged_strategy == Config::UnconvergedStrategy::Const
              ? "const"
              : "force-scaled");
    d.put(p + ".unconverged_value", num(c.unconverged_value));
}

}  // namespace

std::string describe_parse(const HPOTargets& t)
{
    Dump d;
    d.put("target_file", quote(t.target_file));

    /* The top-level keys. */
    d.put("FAIL_SCORE", num(t.FAIL_SCORE));
    d.put("QFLUSH_THRESHOLD", integer(t.QFLUSH_THRESHOLD));
    d.put("MAX_TIME", quote(t.MAX_TIME));
    d.put("MAX_TIME_seconds", optional_of(t.MAX_TIME_seconds, num));
    d.put("MAXTIME_line", integer(t.MAXTIME_line));
    d.put("PC_ISOLATION", quote(t.PC_ISOLATION));
    d.put("PC_ISOLATION_set", flag(t.PC_ISOLATION_set));
    d.put("PC_ISOLATION_line", integer(t.PC_ISOLATION_line));
    d.put("PC_WORKER_TIMEOUT", num(t.PC_WORKER_TIMEOUT));
    d.put("PC_WORKER_TIMEOUT_line", integer(t.PC_WORKER_TIMEOUT_line));
    d.put("PC_WORKER_TIMEOUT_text", quote(t.PC_WORKER_TIMEOUT_text));
    d.put("FAILSCORE_GRADING", flag(t.FAILSCORE_GRADING));
    d.put("BUMPGATE_STRICT", flag(t.BUMPGATE_STRICT));
    if (const auto& g = t.RHOGATE) {
        d.put("RHOGATE.tol", num(g->tol));
        d.put("RHOGATE.r_lo", optional_of(g->r_lo, num));
        d.put("RHOGATE.npts", integer(g->npts));
        d.put("RHOGATE.seed_warn", flag(g->seed_warn));
        d.put("RHOGATE.line", integer(g->line));
        d.put("RHOGATE.line_text", quote(g->line_text));
    } else {
        d.put("RHOGATE", "none");
    }
    if (const auto& g = t.RHOBARGATE) {
        d.put("RHOBARGATE.lattice", quote(g->lattice));
        d.put("RHOBARGATE.a0", num(g->a0));
        d.put("RHOBARGATE.f_lo", num(g->f_lo));
        d.put("RHOBARGATE.f_hi", num(g->f_hi));
        d.put("RHOBARGATE.npts", integer(g->npts));
        d.put("RHOBARGATE.seed_warn", flag(g->seed_warn));
        d.put("RHOBARGATE.line", integer(g->line));
        d.put("RHOBARGATE.line_text", quote(g->line_text));
    } else {
        d.put("RHOBARGATE", "none");
    }
    d.put("failprobe_idx", signed_integer(t.failprobe_idx));
    d.put("failprobe_svar", quote(t.failprobe_svar));
    d.put("failprobe_gate", num(t.failprobe_gate));
    d.put("failprobe_s0", num(t.failprobe_s0));
    d.put("failprobe_s0_given", flag(t.failprobe_s0_given));
    d.put("outputDigits", signed_integer(t.outputDigits));
    d.put("outputEvery", integer(t.outputEvery));
    d.put("bestOutputEvery", integer(t.bestOutputEvery));
    d.put("dumpPotEvery", integer(t.dumpPotEvery));
    d.put("dumpPotDir", quote(t.dumpPotDir));
    d.put("dumpPotLine.number", integer(t.dumpPotLine.number));
    d.put("dumpPotLine.text", quote(t.dumpPotLine.text));
    d.put("dumpBestPotEvery", integer(t.dumpBestPotEvery));
    d.put("dumpBestPotDir", quote(t.dumpBestPotDir));
    d.put("dumpBestPotLine.number", integer(t.dumpBestPotLine.number));
    d.put("dumpBestPotLine.text", quote(t.dumpBestPotLine.text));
    dump_loss(d, "globalLoss", t.globalLoss);

    /* The metric lines. */
    d.put("use_emae", flag(t.use_emae));
    d.put("emae", num(t.emae));
    d.put("w_emae", num(t.w_emae));
    dump_caps(d, "skip_emae", t.skip_emae);
    d.put("use_errmse", flag(t.use_errmse));
    d.put("errmse", num(t.errmse));
    d.put("w_errmse", num(t.w_errmse));
    dump_caps(d, "skip_errmse", t.skip_errmse);
    d.put("use_ersq", flag(t.use_ersq));
    d.put("ersq", num(t.ersq));
    d.put("w_ersq", num(t.w_ersq));
    dump_caps(d, "skip_ersq", t.skip_ersq);
    d.put("use_ermse", flag(t.use_ermse));
    dump_metric(d, "ermse_targets", t.ermse_targets);
    d.put("use_frmse", flag(t.use_frmse));
    dump_metric(d, "frmse_targets", t.frmse_targets);
    d.put("use_srmse", flag(t.use_srmse));
    dump_metric(d, "srmse_targets", t.srmse_targets);

    /* OPTIM_SCALE and the OPTIM lines. */
    d.put("scale_policy.normalise", flag(t.scale_policy_.normalise));
    d.put("scale_policy.set", flag(t.scale_policy_set_));
    d.put("optimap.size", integer(t.optimap.size()));
    for (const auto& [key, o] : t.optimap) {
        const std::string p = "optimap[" + key + "]";
        d.put(p + ".config", strings(o.config));
        d.put(p + ".size", integer(o.opt.size()));
        for (std::size_t i = 0; i < o.opt.size(); ++i) {
            if (!o.opt[i]) continue;
            const std::string q = p + "[" + integer(i) + "]";
            d.put(q + ".line", integer(o.line[i]));
            d.put(q + ".low", num(o.lowbound[i]));
            d.put(q + ".high", num(o.highbound[i]));
            d.put(q + ".scale", scale_kind_name(o.scale[i].kind));
            d.put(q + ".c", num(o.scale[i].c));
            d.put(q + ".tagged", flag(o.tagged[i]));
            d.put(q + ".origin", box_origin_name(o.origin[i]));
        }
    }

    /* The PC_LAMMPS, PC_FAILPROBE and PC_OBSERVABLE lines. */
    d.put("LS.size", integer(t.LS.size()));
    for (std::size_t i = 0; i < t.LS.size(); ++i)
        dump_sim(d, "LS[" + integer(i) + "]", t.LS[i]);
    d.put("observables.size", integer(t.observables.size()));
    for (std::size_t i = 0; i < t.observables.size(); ++i)
        dump_observable(d, "observables[" + integer(i) + "]",
                        *t.observables[i]);
    d.put("observable_lines.size", integer(t.observable_lines.size()));
    for (std::size_t i = 0; i < t.observable_lines.size(); ++i) {
        const std::string p = "observable_lines[" + integer(i) + "]";
        d.put(p + ".number", integer(t.observable_lines[i].number));
        d.put(p + ".text", quote(t.observable_lines[i].text));
    }

    /* The OPTIMIZER block. */
    dump_spec(d, "spec", &t.spec);

    /* The training configuration as the targets leave it. */
    const auto storage = t.run_context().storage_snapshot();
    d.put("context.size", integer(storage.size()));
    for (const auto& [key, values] : storage)
        d.put("context[" + key + "]", strings(values));

    return std::move(d).text();
}

}  // namespace tadah::hpo::testing
