/* hpotarget_handlers.cpp -- the handlers of the registry-driven parser of
 * the target file: one for each key of the registry, and for each line
 * option and form of the lines HPO reads (hpotarget_handlers_gen.h
 * declares them, as generated from the registry; blueprint/README.md, "The
 * handlers"). A key's handler reads the value of its line (and its
 * options, at the step Options of a line that has some) through the
 * readers of hpotarget_reads.h and sets what the line sets; the repeat
 * rule, the leftover check and the order of the steps are the parser's
 * (hpotarget_parser.cpp). The handlers of the keys of the optimiser group
 * also read the keys for `tadah refit` (dispatch_opt_attr), those of the
 * options of PC_LAMMPS and PC_FAILPROBE the options of LammpsSim, and that
 * of the form cohesive_energy a PC_OBSERVABLE line
 * (parse_pc_observable_line). */

#include "hpotarget_parser.h"

#include <tadah/core/periodic_table.h>
#include <tadah/hpo/init_strategy.h>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/observables/observable_parser.h>
#include <tadah/hpo/optimizer_block.h>
#include <tadah/hpo/strict_parse.h>
#include <tadah/mlip/hpotarget_optim.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace tadah {
namespace hpo {
namespace hpotarget_registry {

/* ------------------------------------------------------- readers ---- */

namespace {

std::string str(std::string_view s) { return std::string(s); }

std::string upper(std::string s)
{
    for (auto& c : s) c = static_cast<char>(std::toupper(
                          static_cast<unsigned char>(c)));
    return s;
}

bool same_word(std::string_view a, std::string_view b, bool any_case)
{
    if (!any_case) return a == b;
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::toupper(static_cast<unsigned char>(a[i])) !=
            std::toupper(static_cast<unsigned char>(b[i])))
            return false;
    return true;
}

/* Whether `tok` is a whole number (decimal digits) below `min`: the
 * refusal parse_uint() adds its note to. */
bool whole_below(const std::string& tok, std::uint64_t min)
{
    std::uint64_t v = 0;
    const char* last = tok.data() + tok.size();
    const auto r = std::from_chars(tok.data(), last, v);
    return !tok.empty() && r.ec == std::errc{} && r.ptr == last && v < min;
}

const Arg& arg0(const Key& k) { return k.args[0]; }
const Arg& arg0(const LineOption& o) { return o.args[0]; }

}  // namespace

std::string what_of(const Arg& a, std::string_view key)
{
    return fill(a.what, {{"key", key}});
}

std::uint64_t read_integer(const Arg& a, const std::string& tok,
                           std::string_view key)
{
    const std::uint64_t min = a.min.set ? a.min.whole_value : 0;
    const std::uint64_t max = a.max.set
                                  ? a.max.whole_value
                                  : std::numeric_limits<std::uint64_t>::max();
    try {
        return parse_uint(tok, what_of(a, key), min, max);
    } catch (const std::runtime_error&) {
        if (!a.msg.empty() && whole_below(tok, min))
            throw std::runtime_error(fill(a.msg, {{"tok", tok}, {"key", key}}));
        throw;
    }
}

double read_number(const Arg& a, const std::string& tok, std::string_view key)
{
    return parse_finite(tok, what_of(a, key));
}

double read_number_in_range(const Arg& a, const std::string& tok,
                            std::string_view key)
{
    const double v = read_number(a, tok, key);
    const bool low = a.min.set && (a.min_open ? !(v > a.min.value)
                                              : !(v >= a.min.value));
    const bool high = a.max.set && !(v <= a.max.value);
    if (low || high)
        throw std::runtime_error(fill(a.msg, {{"tok", tok}, {"key", key}}));
    return v;
}

double read_positive(const Arg& a, const std::string& tok,
                     std::string_view key)
{
    return parse_positive(tok, what_of(a, key),
                          a.noun.empty() ? std::string("number")
                                         : str(a.noun));
}

std::string read_word(const Arg& a, const std::string& tok,
                      std::string_view key)
{
    for (std::string_view w : a.words)
        if (same_word(w, tok, a.any_case)) return str(w);
    throw std::runtime_error(fill(a.msg, {{"tok", tok}, {"key", key}}));
}

const LineOption* find_option(std::string_view line, const std::string& word)
{
    for (const LineOption& o : kLineOptions) {
        if (!same_word(o.option, word, o.any_case)) continue;
        for (std::string_view l : o.lines)
            if (l == line) return &o;
    }
    return nullptr;
}

bool takes_option(const LineOption& o, std::string_view line)
{
    if (o.refused) return false;
    for (std::string_view l : o.lines)
        if (l == line) return true;
    return false;
}

/* ------------------------------------------------- the top level ---- */

namespace {

using hpotarget::Parser;

/* The settings a line reads when it gives none of its own. */
void scripts_read_settings(Parser& p, const LammpsSim& sim, bool loss)
{
    p.line_reads_setting("FAILSCORE", sim.reads_default_failscore);
    if (loss) p.line_reads_setting("LOSS", sim.reads_default_loss);
    p.line_reads_setting("PC_SCRIPT_TIMEOUT", sim.reads_default_maxtime);
}

/* The metric lines. */
constexpr KeyId kMetricKeys[] = {
    KeyId::ROOT_PC_EMAE, KeyId::ROOT_PC_ERMSE, KeyId::ROOT_PC_ErRMSE,
    KeyId::ROOT_PC_ERSQ, KeyId::ROOT_PC_SRMSE_GPA, KeyId::ROOT_PC_FRMSE};

/* The values of a metric line (PC_EMAE, PC_ERMSE, ...): an index set (the
 * dataset metrics only), the target and the weight. */
void metric_values(KeyRead& r)
{
    Parser& p = r.p();
    const std::string& key = r.word;
    const bool dataset_metric = r.key.id == KeyId::ROOT_PC_ERMSE ||
                                r.key.id == KeyId::ROOT_PC_FRMSE ||
                                r.key.id == KeyId::ROOT_PC_SRMSE_GPA;
    p.metric = {};
    r.iss >> std::ws;
    if (r.iss.peek() == '(') {
        if (!dataset_metric)
            throw std::runtime_error(key + ": indexed targets are supported "
                                     "only for PC_ERMSE, PC_FRMSE and "
                                     "PC_SRMSE_GPA");
        p.metric.indices = core::parse_indices_strict(
            mlip::read_index_selector(r.iss, key));
    }
    std::string target_token, weight_token;
    if (!(r.iss >> target_token >> weight_token))
        throw std::runtime_error(key + " requires <target> <weight>");
    p.metric.target = parse_metric_number(target_token, key + " <target>");
    p.metric.weight = parse_metric_number(weight_token, key + " <weight>");
    if (p.metric.weight < 0)
        throw std::runtime_error(key + " <weight> must be a number >= 0, "
                                 "got '" + weight_token + "'");
}

/* The options of a metric line and what the line sets. */
void metric_options(KeyRead& r)
{
    Parser& p = r.p();
    ParsedTarget& t = p.t;
    const std::string& key = r.word;
    MetricSkipCaps caps;
    MetricOptions m{key, r.iss, caps};
    std::string flag;
    while (r.iss >> flag) {
        /* An option of a metric line; one that another metric line takes
         * and this one does not (a cap of PC_ERSQ) is refused at the
         * option, before its value is read. */
        const LineOption* o = find_option(key, flag);
        if (!o) {
            bool sibling = false;
            for (const KeyId id : kMetricKeys)
                if (find_option(key_entry(id).name, flag)) sibling = true;
            if (!sibling)
                throw std::runtime_error(key + ": unknown flag '" + flag +
                                         "'");
            /* A cap rejects an evaluation whose metric is above it, and
             * R^2 is better when higher. */
            throw std::runtime_error(
                key + " takes no " + flag + ": a cap rejects an evaluation "
                "whose metric is above it, and a higher R^2 is a better fit, "
                "so the cap would reject the better fits");
        }
        OptionRead o_read{*o, flag};
        o_read.metric = &m;
        call(*o, o_read);
    }
    const auto& mv = p.metric;
    if (r.key.id == KeyId::ROOT_PC_ERMSE || r.key.id == KeyId::ROOT_PC_FRMSE ||
        r.key.id == KeyId::ROOT_PC_SRMSE_GPA) {
        auto& targets = r.key.id == KeyId::ROOT_PC_ERMSE   ? t.ermse_targets
                        : r.key.id == KeyId::ROOT_PC_FRMSE ? t.frmse_targets
                                                           : t.srmse_targets;
        targets.add(mv.indices, {mv.target, mv.weight, caps, p.nline, p.line});
        if (r.key.id == KeyId::ROOT_PC_ERMSE) t.use_ermse = true;
        else if (r.key.id == KeyId::ROOT_PC_FRMSE) t.use_frmse = true;
        else t.use_srmse = true;
        return;
    }
    p.metric_lines.emplace(key, std::make_pair(p.nline, p.line));
    if (r.key.id == KeyId::ROOT_PC_EMAE) {
        t.emae = mv.target; t.w_emae = mv.weight; t.use_emae = true;
        t.skip_emae = caps;
    } else if (r.key.id == KeyId::ROOT_PC_ErRMSE) {
        t.errmse = mv.target; t.w_errmse = mv.weight; t.use_errmse = true;
        t.skip_errmse = caps;
    } else {
        t.ersq = mv.target; t.w_ersq = mv.weight; t.use_ersq = true;
    }
}

void metric_line(KeyRead& r)
{
    if (r.step == LineStep::Value) metric_values(r);
    else metric_options(r);
}

/* The value of a gate line (RHOGATE, RHOBARGATE) and its options, which
 * stand in keyword pairs after it. */
const char* const kRhoGateGrammar =
    "expected RHOGATE <tol> [R_LO <r>] [NPTS <n>] [SEED ERROR|WARN]";
const char* const kRhoBarGateGrammar =
    "expected RHOBARGATE <bcc|fcc|hcp> <a0> <f_lo> <f_hi> [NPTS <n>] "
    "[SEED ERROR|WARN]";

void gate_options(KeyRead& r, const char* grammar, std::size_t first,
                  std::size_t& npts, bool& seed_warn,
                  std::optional<double>* r_lo)
{
    Parser& p = r.p();
    const std::string& key = r.word;
    const auto& toks = p.gate_tokens;
    std::vector<const LineOption*> seen;
    for (std::size_t i = first; i < toks.size(); i += 2) {
        const LineOption* o = find_option(key, toks[i]);
        if (!o)
            throw std::runtime_error(key + ": unexpected token '" + toks[i] +
                                     "'; " + grammar);
        const std::string kw = upper(toks[i]);
        if (std::find(seen.begin(), seen.end(), o) != seen.end())
            throw std::runtime_error(str(o->repeat_msg));
        seen.push_back(o);
        if (i + 1 >= toks.size())
            throw std::runtime_error(key + ": " + kw + " needs a value; " +
                                     grammar);
        GateOptions g{key, toks[i + 1], npts, seed_warn, r_lo};
        OptionRead o_read{*o, toks[i]};
        o_read.gate = &g;
        call(*o, o_read);
    }
}

}  // namespace

void Handlers::ROOT_FAILSCORE(KeyRead& r)
{
    if (r.step != LineStep::Value) return;
    r.p().t.FAIL_SCORE =
        read_positive(arg0(r.key), single_value(r.iss, r.word), r.word);
}

/* PC_SCRIPT_TIMEOUT, and MAXTIME at the top level, another name of the
 * key: a LAMMPS time spec (off, unlimited or a positive [[HH:]MM:]SS). */
void Handlers::ROOT_PC_SCRIPT_TIMEOUT(KeyRead& r)
{
    if (r.step != LineStep::Value) return;
    ParsedTarget& t = r.p().t;
    t.MAX_TIME = single_value(r.iss, r.word);
    t.MAX_TIME_seconds = parse_script_timeout(t.MAX_TIME, r.word);
}

void Handlers::ROOT_MAXTIME(KeyRead& r)
{
    if (r.step != LineStep::Value) return;
    ROOT_PC_SCRIPT_TIMEOUT(r);
    ParsedTarget& t = r.p().t;
    if (t.MAXTIME_line == 0) t.MAXTIME_line = r.p().nline;
}

void Handlers::ROOT_PC_ISOLATION(KeyRead& r)
{
    if (r.step != LineStep::Value) return;
    ParsedTarget& t = r.p().t;
    t.PC_ISOLATION =
        read_word(arg0(r.key), single_value(r.iss, r.word), r.word);
    t.PC_ISOLATION_set = true;
    t.PC_ISOLATION_line = r.p().nline;
}

void Handlers::ROOT_PC_WORKER_TIMEOUT(KeyRead& r)
{
    if (r.step != LineStep::Value) return;
    ParsedTarget& t = r.p().t;
    t.PC_WORKER_TIMEOUT = ParsedTarget::parse_pc_worker_timeout(
        single_value(r.iss, r.word), "PC_WORKER_TIMEOUT");
    t.PC_WORKER_TIMEOUT_line = r.p().nline;
    t.PC_WORKER_TIMEOUT_text = r.p().line;
}

void Handlers::ROOT_FAILSCORE_GRADING(KeyRead& r)
{
    if (r.step != LineStep::Value) return;
    const std::string v =
        read_word(arg0(r.key), single_value(r.iss, r.word), r.word);
    r.p().t.FAILSCORE_GRADING = v == "ON";
}

void Handlers::ROOT_BUMPGATE(KeyRead& r)
{
    if (r.step != LineStep::Value) return;
    const std::string v =
        read_word(arg0(r.key), single_value(r.iss, r.word), r.word);
    r.p().t.BUMPGATE_STRICT = v == "STRICT";
}

/* RHOGATE <tol> [R_LO <r>] [NPTS <n>] [SEED ERROR|WARN]. */
void Handlers::ROOT_RHOGATE(KeyRead& r)
{
    Parser& p = r.p();
    if (r.step == LineStep::Value) {
        p.gate_tokens.clear();
        for (std::string tok; r.iss >> tok;) p.gate_tokens.push_back(tok);
        if (p.gate_tokens.empty())
            throw std::runtime_error(std::string("RHOGATE: the tolerance is "
                                                 "missing; ") +
                                     kRhoGateGrammar);
        p.rhogate.emplace();
        p.rhogate->tol =
            read_number_in_range(arg0(r.key), p.gate_tokens[0], r.word);
        return;
    }
    RhoGateSpec& s = *p.rhogate;
    gate_options(r, kRhoGateGrammar, 1, s.npts, s.seed_warn, &s.r_lo);
    s.line = p.nline;
    s.line_text = p.line;
    p.t.RHOGATE = s;
    p.gate_lines.push_back({"RHOGATE", p.nline, p.line});
}

/* RHOBARGATE <lattice> <a0> <f_lo> <f_hi> [NPTS <n>] [SEED ERROR|WARN]. */
void Handlers::ROOT_RHOBARGATE(KeyRead& r)
{
    Parser& p = r.p();
    if (r.step == LineStep::Value) {
        p.gate_tokens.clear();
        for (std::string tok; r.iss >> tok;) p.gate_tokens.push_back(tok);
        const auto& toks = p.gate_tokens;
        if (toks.size() < 4)
            throw std::runtime_error(std::string("RHOBARGATE: the lattice, "
                                                 "a0, f_lo and f_hi are "
                                                 "required; ") +
                                     kRhoBarGateGrammar);
        p.rhobargate.emplace();
        RhoBarGateSpec& s = *p.rhobargate;
        s.lattice = read_word(r.key.args[0], toks[0], r.word);
        s.a0 = read_number(r.key.args[1], toks[1], r.word);
        s.f_lo = read_number(r.key.args[2], toks[2], r.word);
        s.f_hi = read_number(r.key.args[3], toks[3], r.word);
        if (!(s.a0 > 0.0))
            throw std::runtime_error("RHOBARGATE a0 must be > 0, got '" +
                                     toks[1] + "'");
        if (!(s.f_lo > 0.0) || !(s.f_lo < s.f_hi))
            throw std::runtime_error("RHOBARGATE needs 0 < f_lo < f_hi, got "
                                     "f_lo '" + toks[2] + "', f_hi '" +
                                     toks[3] + "'");
        return;
    }
    RhoBarGateSpec& s = *p.rhobargate;
    gate_options(r, kRhoBarGateGrammar, 4, s.npts, s.seed_warn, nullptr);
    s.line = p.nline;
    s.line_text = p.line;
    p.t.RHOBARGATE = s;
    p.gate_lines.push_back({"RHOBARGATE", p.nline, p.line});
}

void Handlers::ROOT_QFLUSH(KeyRead& r)
{
    if (r.step != LineStep::Value) return;
    r.p().t.QFLUSH_THRESHOLD =
        read_integer(arg0(r.key), single_value(r.iss, r.word), r.word);
}

/* OUTPUT and BOUTPUT are cadences (`iter % every`): 0 is refused with the
 * reason. */
void Handlers::ROOT_OUTPUT(KeyRead& r)
{
    if (r.step != LineStep::Value) return;
    r.p().t.outputEvery =
        read_integer(arg0(r.key), single_value(r.iss, r.word), r.word);
}

void Handlers::ROOT_BOUTPUT(KeyRead& r)
{
    if (r.step != LineStep::Value) return;
    r.p().t.bestOutputEvery =
        read_integer(arg0(r.key), single_value(r.iss, r.word), r.word);
}

void Handlers::ROOT_DIGITS(KeyRead& r)
{
    if (r.step != LineStep::Value) return;
    r.p().t.outputDigits = static_cast<int>(
        read_integer(arg0(r.key), single_value(r.iss, r.word), r.word));
}

namespace {
/* DUMP and BDUMP: <cadence> <directory>, the directory checked once the
 * file is read (check_archive_directories). */
void archive_line(KeyRead& r, std::size_t& every, std::string& dir,
                  ParsedTarget::ArchiveLine& at)
{
    Parser& p = r.p();
    std::string cadence, directory;
    if (!(r.iss >> cadence >> directory))
        throw std::runtime_error(r.word + " requires a cadence and a "
                                 "directory");
    expect_end(r.iss, r.word + " takes a cadence and a directory");
    every = read_integer(arg0(r.key), cadence, r.word);
    dir = directory;
    at = {p.nline, p.line};
}
}  // namespace

void Handlers::ROOT_DUMP(KeyRead& r)
{
    if (r.step != LineStep::Value) return;
    ParsedTarget& t = r.p().t;
    archive_line(r, t.dumpPotEvery, t.dumpPotDir, t.dumpPotLine);
}

void Handlers::ROOT_BDUMP(KeyRead& r)
{
    if (r.step != LineStep::Value) return;
    ParsedTarget& t = r.p().t;
    archive_line(r, t.dumpBestPotEvery, t.dumpBestPotDir, t.dumpBestPotLine);
}

void Handlers::ROOT_OPTIM_SCALE(KeyRead& r)
{
    if (r.step != LineStep::Value) return;
    ParsedTarget& t = r.p().t;
    const std::string v =
        read_word(arg0(r.key), single_value(r.iss, r.word), r.word);
    t.scale_policy_.normalise = v == "AUTO";
    t.scale_policy_set_ = true;
}

void Handlers::ROOT_PC_EMAE(KeyRead& r) { metric_line(r); }
void Handlers::ROOT_PC_ERMSE(KeyRead& r) { metric_line(r); }
void Handlers::ROOT_PC_ErRMSE(KeyRead& r) { metric_line(r); }
void Handlers::ROOT_PC_ERSQ(KeyRead& r) { metric_line(r); }
void Handlers::ROOT_PC_SRMSE_GPA(KeyRead& r) { metric_line(r); }
void Handlers::ROOT_PC_FRMSE(KeyRead& r) { metric_line(r); }

/* A PC_LAMMPS line: LammpsSim reads its options (the handlers below) and
 * its script. */
void Handlers::ROOT_PC_LAMMPS(KeyRead& r)
{
    if (r.step != LineStep::Value) return;
    Parser& p = r.p();
    ParsedTarget& t = p.t;
    std::string cmd;
    std::getline(r.iss, cmd);
    t.LS.emplace_back(cmd, t.MAX_TIME, t.FAIL_SCORE, t.globalLoss);
    LammpsSim& sim = t.LS.back();
    sim.line = p.nline;
    sim.line_text = p.line;
    sim.maxtime_source = p.maxtime_source(sim);
    scripts_read_settings(p, sim, /*loss=*/true);
}

/* A PC_FAILPROBE line: its --gate and --s0 first, then LammpsSim reads the
 * line; a probe grades failures, so it takes none of the options that
 * shape the objective. */
void Handlers::ROOT_PC_FAILPROBE(KeyRead& r)
{
    if (r.step != LineStep::Value) return;
    Parser& p = r.p();
    ParsedTarget& t = p.t;
    std::string cmd;
    std::getline(r.iss, cmd);
    std::istringstream ts(cmd);
    std::vector<std::string> toks;
    for (std::string tok; ts >> tok;) toks.push_back(tok);
    bool gate_seen = false;
    for (std::size_t i = 0; i < toks.size(); ++i) {
        const LineOption* o = find_option("PC_FAILPROBE", toks[i]);
        if (!o || (o->id != LineOptionId::PC_FAILPROBE_gate &&
                   o->id != LineOptionId::PC_FAILPROBE_s0))
            continue;
        ProbeOptions probe{toks, i, gate_seen, t.failprobe_s0_given,
                           t.failprobe_gate, t.failprobe_s0};
        OptionRead o_read{*o, toks[i]};
        o_read.probe = &probe;
        call(*o, o_read);
    }
    t.LS.emplace_back(cmd, t.MAX_TIME, t.FAIL_SCORE, t.globalLoss,
                      /*failprobe=*/true);
    LammpsSim& probe = t.LS.back();
    probe.line = p.nline;
    probe.line_text = p.line;
    probe.maxtime_source = p.maxtime_source(probe);
    scripts_read_settings(p, probe, /*loss=*/false);
    if (!probe.n_varloss.empty() || !probe.i_varloss.empty())
        throw std::runtime_error(
            "PC_FAILPROBE must not declare --varloss (it grades failures; it "
            "must not shape the feasible objective)");
    if (probe.n_outvar.empty())
        throw std::runtime_error(
            "PC_FAILPROBE requires --outvar <S> (the sanity scalar)");
    if (!probe.hotvars.empty() || !probe.failvars.empty())
        throw std::runtime_error(
            "PC_FAILPROBE must not declare --hotvar/--failvar");
    t.failprobe_idx = static_cast<int>(t.LS.size()) - 1;
    t.failprobe_svar = probe.n_outvar.front();
}

/* A PC_OBSERVABLE line: its form reads it; its loss function is that of
 * the LOSS line above it unless it gives --loss. */
void Handlers::ROOT_PC_OBSERVABLE(KeyRead& r)
{
    if (r.step != LineStep::Value) return;
    Parser& p = r.p();
    ParsedTarget& t = p.t;
    std::string cmd;
    std::getline(r.iss, cmd);
    t.observables.push_back(parse_pc_observable_line(cmd, t.globalLoss));
    t.observable_lines.push_back({p.nline, p.line});
    p.line_reads_setting("LOSS", t.observables.back()->reads_default_loss());
}

/* LOSS <type> [<param>]: the parameter is validated and defaulted by
 * LossFunction::resolve_params, as the --varloss override's. */
void Handlers::ROOT_LOSS(KeyRead& r)
{
    if (r.step != LineStep::Value) return;
    Parser& p = r.p();
    std::string type;
    if (!(r.iss >> type))
        throw std::runtime_error("LOSS requires a loss type");
    std::vector<double> params;
    for (std::string tok; r.iss >> tok;) {
        double v = 0.0;
        if (!parse_number(tok, v))
            throw std::runtime_error("LOSS " + type + ": parameter '" + tok +
                                     "' is not a number" +
                                     number_underflow_note(tok));
        params.push_back(v);
    }
    const auto id = tadah::LossFunction::from_name(type, "LOSS");
    p.t.globalLoss = tadah::LossFunction(id, params);
    p.loss_line = p.nline;
}

/* An OPTIM line: MLIP's OPTIM reader reads it (its forms and options), as
 * tadah analysis check_bounds reads it. */
void Handlers::ROOT_OPTIM(KeyRead& r)
{
    if (r.step != LineStep::Value) return;
    r.p().t.process_optim_config(r.iss, r.p().nline);
}

/* ------------------------------------------- the optimiser group ---- */

/* The keys of the optimiser group read their value into the optimiser
 * r.spec; the parser of the target file and `tadah refit`
 * (dispatch_opt_attr) call them. What follows the value is the caller's:
 * the leftover check of the line, or the next key of a refit block. */

void Handlers::OPTIMIZER_LIB(KeyRead& r)
{
    r.spec->lib = next_value(r.iss, r.word);
}

void Handlers::OPTIMIZER_ALGO(KeyRead& r)
{
    r.spec->algo = next_value(r.iss, r.word);
}

void Handlers::OPTIMIZER_MAXEVAL(KeyRead& r)
{
    r.spec->stop.maxeval =
        read_integer(arg0(r.key), next_value(r.iss, r.word), r.word);
}

void Handlers::OPTIMIZER_MAXTIME(KeyRead& r)
{
    r.spec->stop.maxtime =
        read_positive(arg0(r.key), next_value(r.iss, r.word), r.word);
}

void Handlers::OPTIMIZER_STOPVAL(KeyRead& r)
{
    r.spec->stop.stopval =
        read_number(arg0(r.key), next_value(r.iss, r.word), r.word);
}

void Handlers::OPTIMIZER_FTOL_REL(KeyRead& r)
{
    r.spec->stop.ftol_rel =
        read_number(arg0(r.key), next_value(r.iss, r.word), r.word);
}

void Handlers::OPTIMIZER_FTOL_ABS(KeyRead& r)
{
    r.spec->stop.ftol_abs =
        read_number(arg0(r.key), next_value(r.iss, r.word), r.word);
}

void Handlers::OPTIMIZER_XTOL_REL(KeyRead& r)
{
    r.spec->stop.xtol_rel =
        read_number(arg0(r.key), next_value(r.iss, r.word), r.word);
}

void Handlers::OPTIMIZER_XTOL_ABS(KeyRead& r)
{
    r.spec->stop.xtol_abs =
        read_number(arg0(r.key), next_value(r.iss, r.word), r.word);
}

void Handlers::OPTIMIZER_POPULATION(KeyRead& r)
{
    r.spec->pop = static_cast<unsigned>(
        read_integer(arg0(r.key), next_value(r.iss, r.word), r.word));
}

/* STEP <number>, the block-wide fraction, or STEP <KEY>[(indices)]
 * <number>, an override of some dimensions. A token shaped like a number
 * is the fraction (the number grammar judges it); the range is the
 * backend's (validate_optimizer_scaling). */
void Handlers::OPTIMIZER_STEP(KeyRead& r)
{
    OptimizerSpec& a = *r.spec;
    std::istringstream& iss = r.iss;
    const std::string first = next_value(iss, r.word);
    if (is_number_shaped(first)) {
        a.step = parse_finite(first, r.word);
        return;
    }
    StepOverride ov;
    /* KEY, KEY(3), KEY(1-4,7) or KEY (1,2) with a space before '(' */
    std::string spec = first;
    std::string rest_tok;
    if (spec.find('(') == std::string::npos) {
        std::streampos save = iss.tellg();
        if (iss >> rest_tok && !rest_tok.empty() && rest_tok.front() == '(') {
            spec += rest_tok;
        } else {
            iss.clear();
            iss.seekg(save);
        }
    }
    const std::size_t lp = spec.find('(');
    if (lp != std::string::npos) {
        const std::size_t rp = spec.find(')', lp);
        if (rp == std::string::npos || rp != spec.size() - 1)
            throw std::runtime_error("STEP override '" + spec +
                                     "': expected KEY(indices) <frac>");
        ov.key = spec.substr(0, lp);
        const std::string idx = spec.substr(lp + 1, rp - lp - 1);
        if (idx.empty())
            throw std::runtime_error("STEP override '" + spec +
                                     "': empty index list");
        /* The strict grammar of the OPTIM lines. */
        try {
            ov.indices = core::parse_indices_strict(idx);
            if (std::find(ov.indices.begin(), ov.indices.end(), 0) !=
                ov.indices.end())
                throw std::runtime_error("indices start at 1");
        } catch (const std::exception& e) {
            throw std::runtime_error("STEP override '" + spec +
                                     "': bad index list (" + e.what() + ")");
        }
    } else {
        ov.key = spec;
    }
    const std::string what = "STEP " + spec;
    ov.frac = parse_finite(next_value(iss, what), what);
    a.step_overrides.push_back(std::move(ov));
}

void Handlers::OPTIMIZER_SEED(KeyRead& r)
{
    r.spec->seed = read_integer(arg0(r.key), next_value(r.iss, r.word),
                                r.word);
}

void Handlers::OPTIMIZER_VECTOR_ARR(KeyRead& r)
{
    r.spec->vector_storage = static_cast<unsigned>(
        read_integer(arg0(r.key), next_value(r.iss, r.word), r.word));
}

/* THREADS is the width of the worker pool: at least one worker. */
void Handlers::OPTIMIZER_THREADS(KeyRead& r)
{
    r.spec->num_threads = static_cast<unsigned>(
        read_integer(arg0(r.key), next_value(r.iss, r.word), r.word));
}

void Handlers::OPTIMIZER_GRAD_STRATEGY(KeyRead& r)
{
    /* A line without a value is refused with the values the key takes. */
    std::string v;
    if (!(r.iss >> v))
        throw std::runtime_error(
            "GRAD_STRATEGY takes one value: NUMERICAL, ANALYTICAL or HYBRID");
    const std::string w = read_word(arg0(r.key), v, r.word);
    if (w == "NUMERICAL")
        r.spec->grad_strategy = GradStrategy::Numerical;
    else if (w == "ANALYTICAL")
        r.spec->grad_strategy = GradStrategy::Analytical;
    else if (w == "HYBRID")
        r.spec->grad_strategy = GradStrategy::Hybrid;
    else
        throw std::logic_error("GRAD_STRATEGY: the registry gives the word " +
                               w + ", which the parser does not know");
}

/* PARAM <name> <value>...: at most the values of the name's grammar
 * (param_takes_another_value), the rest left for the caller. */
void Handlers::OPTIMIZER_PARAM(KeyRead& r)
{
    OptimizerSpec& a = *r.spec;
    std::istringstream& iss = r.iss;
    std::string p;
    iss >> p;
    std::vector<std::string> vals;
    std::string ukey = upper(p);
    for (;;) {
        const std::streampos save = iss.tellg();
        std::string tok;
        if (!(iss >> tok)) { iss.clear(); break; }
        if (!param_takes_another_value(ukey, vals, tok)) {
            iss.clear();
            iss.seekg(save);
            break;
        }
        vals.push_back(std::move(tok));
    }
    if (p.empty() || vals.empty())
        throw std::runtime_error("PARAM requires a name and a value");
    const std::string& raw = vals.front();
    /* Numeric values land in `param`, a word in `param_raw`. A value
     * shaped like a number must be a finite one. */
    if (is_number_shaped(raw))
        a.param[p] = parse_finite(raw, "PARAM " + p);
    else
        a.param_raw[p] = raw;
    /* The numbers of INIT_MODE JITTER <fraction> | JITTER AUTO
     * <tolerance>, read with the number grammar here. */
    if (ukey == "INIT_MODE")
        for (std::size_t i = 1; i < vals.size(); ++i)
            if (upper(vals[i]) != "AUTO")
                (void)parse_finite(vals[i],
                                   i == 2 ? "PARAM " + p +
                                                " JITTER AUTO <tolerance>"
                                          : "PARAM " + p +
                                                " JITTER <fraction>");
    /* Extra value tokens are kept for the family registry to judge. */
    if (vals.size() > 1) a.param_tokens[p] = std::move(vals);
}

/* LOG_HP, a property of the search space, in OPTIMIZER or STRATEGY. */
void Handlers::OPTIMIZER_LOG_HP(KeyRead& r)
{
    Parser& p = r.p();
    const std::string tok = single_value(r.iss, r.word);
    p.t.spec.log_threshold = read_number_in_range(arg0(r.key), tok, r.word);
    r.target = &p.t.spec.log_threshold;
    r.what = "search space";
}

/* ------------------------------------------------------- STRATEGY ---- */

void Handlers::STRATEGY_TYPE(KeyRead& r)
{
    Parser& p = r.p();
    const std::string t =
        read_word(arg0(r.key), single_value(r.iss, r.word), r.word);
    if (t == "PLAIN")
        p.t.spec.strategy_type = hpo::StrategyType::PLAIN;
    else if (t == "MULTISTART")
        p.t.spec.strategy_type = hpo::StrategyType::MULTISTART;
    else if (t == "PRESTAGE")
        p.t.spec.strategy_type = hpo::StrategyType::PRESTAGE;
    else
        throw std::logic_error("STRATEGY TYPE: the registry gives the word " +
                               t + ", which the parser does not know");
    p.strategy_type_given = true;
    p.type_line = p.nline;
    p.type_text = p.line;
    r.target = &p.t.spec.strategy_type;
    r.what = "strategy";
}

/* N_STARTS 2 or more: one start is TYPE PLAIN. Its line: startup refuses
 * an INIT chain of fewer starting points there. */
void Handlers::STRATEGY_N_STARTS(KeyRead& r)
{
    Parser& p = r.p();
    p.t.spec.multistart.n_starts =
        read_integer(arg0(r.key), single_value(r.iss, r.word), r.word);
    p.n_starts_given = true;
    p.t.spec.key_lines[r.word] =
        KeyLine{p.nline, p.line, Parser::block_name(p.block())};
    r.target = &p.t.spec.strategy_type;
    r.what = "strategy";
}

void Handlers::STRATEGY_BUDGET_PER_START(KeyRead& r)
{
    Parser& p = r.p();
    p.t.spec.multistart.budget_per_start =
        read_integer(arg0(r.key), single_value(r.iss, r.word), r.word);
    r.target = &p.t.spec.strategy_type;
    r.what = "strategy";
}

void Handlers::STRATEGY_TOPK(KeyRead& r)
{
    Parser& p = r.p();
    p.t.spec.stage_topk =
        read_integer(arg0(r.key), single_value(r.iss, r.word), r.word);
    r.target = &p.t.spec.strategy_type;
    r.what = "strategy";
}

/* ----------------------------------------------------- INIT, RESTART ---- */

/* METHOD <name> <args>... (STRATEGY inside INIT and RESTART, its alias):
 * an entry of the chain, checked at its line (validate_init_entry). */
void Handlers::INIT_METHOD(KeyRead& r)
{
    Parser& p = r.p();
    InitStrategySpec::Entry e;
    e.file = p.fn;
    e.line = p.nline;
    e.line_text = p.line;
    e.key = r.word;
    r.iss >> e.method;
    if (e.method.empty())
        throw std::runtime_error(r.word + " requires a method name");
    for (std::string tok; r.iss >> tok;) e.args.push_back(tok);
    validate_init_entry(e);
    p.active_init->chain.push_back(std::move(e));
}

/* K_TOTAL: a cap of the points the METHOD lines give, 1 or more; whether
 * it caps anything is known at the end of the file. */
void Handlers::INIT_K_TOTAL(KeyRead& r)
{
    Parser& p = r.p();
    p.active_init->k_total =
        read_integer(arg0(r.key), single_value(r.iss, r.word), r.word);
    p.k_total_lines[p.active_init] = {p.nline, p.line};
    r.target = p.active_init;
    r.what = "starting points";
}

void Handlers::INIT_SEED(KeyRead& r)
{
    Parser& p = r.p();
    p.active_init->seed =
        read_integer(arg0(r.key), single_value(r.iss, r.word), r.word);
    r.target = p.active_init;
    r.what = "starting points";
}

void Handlers::INIT_WARM_ON_MISMATCH(KeyRead& r)
{
    Parser& p = r.p();
    p.active_init->warm_on_mismatch =
        read_word(arg0(r.key), single_value(r.iss, r.word), r.word);
    r.target = p.active_init;
    r.what = "starting points";
}

/* AFTER_STAGNATION and MAX, each 1 or more: 0 restarts nothing. */
void Handlers::RESTART_AFTER_STAGNATION(KeyRead& r)
{
    Parser& p = r.p();
    p.t.spec.restart.after_stagnation =
        read_integer(arg0(r.key), single_value(r.iss, r.word), r.word);
    r.target = &p.t.spec.restart;
    r.what = "restarts";
}

void Handlers::RESTART_MAX(KeyRead& r)
{
    Parser& p = r.p();
    p.t.spec.restart.max =
        read_integer(arg0(r.key), single_value(r.iss, r.word), r.word);
    r.target = &p.t.spec.restart;
    r.what = "restarts";
}

/* ----------------------------------------------------------- NOISE ---- */

/* AUTO alone means AUTO true. */
void Handlers::NOISE_AUTO(KeyRead& r)
{
    Parser& p = r.p();
    std::string v;
    r.iss >> v;
    expect_end(r.iss, "AUTO takes one value");
    p.t.spec.noise.autodetect =
        v.empty() ? true : parse_bool(v, what_of(arg0(r.key), r.word));
    r.target = &p.t.spec.noise;
    r.what = "noise settings";
}

/* CALIBRATION: N evaluations, 0 (off) or 2 or more. */
void Handlers::NOISE_CALIBRATION(KeyRead& r)
{
    Parser& p = r.p();
    const std::size_t v =
        read_integer(arg0(r.key), single_value(r.iss, r.word), r.word);
    if (v == 1)
        throw std::runtime_error(str(r.key.checks[0]));
    p.t.spec.noise.calibration_n = v;
    r.target = &p.t.spec.noise;
    r.what = "noise settings";
}

/* ------------------------------------------------- the gate options ---- */

void Handlers::RHOGATE_R_LO(OptionRead& r)
{
    GateOptions& g = *r.gate;
    *g.r_lo = read_number_in_range(arg0(r.option), g.value, g.key);
}

void Handlers::RHOGATE_NPTS(OptionRead& r)
{
    GateOptions& g = *r.gate;
    g.npts = read_integer(arg0(r.option), g.value, g.key);
}

void Handlers::RHOGATE_SEED(OptionRead& r)
{
    GateOptions& g = *r.gate;
    const std::string v = read_word(arg0(r.option), g.value, g.key);
    g.seed_warn = v == "WARN";
}

void Handlers::RHOBARGATE_NPTS(OptionRead& r) { RHOGATE_NPTS(r); }
void Handlers::RHOBARGATE_SEED(OptionRead& r) { RHOGATE_SEED(r); }

/* ----------------------------------------------- the metric options ---- */

namespace {
void metric_cap(OptionRead& r, std::optional<double>& cap)
{
    MetricOptions& m = *r.metric;
    std::string token;
    if (!(m.iss >> token))
        throw std::runtime_error(m.key + " " + r.word +
                                 " requires a positive number");
    const double value = read_positive(arg0(r.option), token, m.key);
    if (cap) throw std::runtime_error(m.key + ": repeated " + r.word);
    cap = value;
}
}  // namespace

void Handlers::PC_EMAE_skipabove(OptionRead& r)
{
    metric_cap(r, r.metric->caps.abs_above);
}

void Handlers::PC_EMAE_skiprel(OptionRead& r)
{
    metric_cap(r, r.metric->caps.rel_above);
}

/* ------------------------------ the options of PC_LAMMPS, PC_FAILPROBE ---- */

/* --script <file>, once. */
void Handlers::PC_LAMMPS_script(OptionRead& r)
{
    ScriptOptions& s = *r.script;
    s.once("--script");
    if (s.i + 1 >= s.tokens.size())
        throw std::runtime_error(s.key + " --script takes one value, the "
                                 "LAMMPS script file");
    s.sim.script = s.tokens[++s.i];
    ++s.i;
    s.option_read("--script takes one value");
}

/* --varloss <name> <target> <weight> [<loss> [<param>...]]: a loss term. */
void Handlers::PC_LAMMPS_varloss(OptionRead& r)
{
    ScriptOptions& s = *r.script;
    LammpsSim& sim = s.sim;
    const auto& tokens = s.tokens;
    const auto is_flag = [](const std::string& t) {
        return !t.empty() && t[0] == '-' && !is_number_shaped(t);
    };
    std::size_t j = s.i + 1;
    if (j >= tokens.size() || is_flag(tokens[j]))
        throw std::runtime_error(s.key + " --varloss takes <name> <target> "
                                 "<weight> (the name is missing)");
    const std::string vname = tokens[j++];
    if (j >= tokens.size() || is_flag(tokens[j]))
        throw std::runtime_error(s.key + " --varloss takes <name> <target> "
                                 "<weight> (" + vname + " has no target)");
    const std::string tgt_tok = tokens[j];
    const double tgt = parse_finite(tokens[j++],
                                    "--varloss " + vname + " <target>");
    if (j >= tokens.size() || is_flag(tokens[j]))
        throw std::runtime_error(s.key + " --varloss takes <name> <target> "
                                 "<weight> (" + vname + " has no weight)");
    /* Any finite weight; a negative one is accepted. */
    const double wgt = parse_finite(tokens[j++],
                                    "--varloss " + vname + " <weight>");
    /* The optional override <type> [<param>], resolved as the LOSS key's. */
    tadah::LossFunction my_loss = s.default_loss;
    if (j < tokens.size() && !is_flag(tokens[j])) {
        const std::string tname = tokens[j];
        auto type = tadah::LossFunction::from_name(tokens[j++],
                                                   "--varloss " + vname);
        std::vector<double> pars;
        while (j < tokens.size() && !is_flag(tokens[j]))
            pars.push_back(parse_finite(tokens[j++], "--varloss " + vname +
                                                         " " + tname +
                                                         " parameter"));
        my_loss = tadah::LossFunction(type, pars);
    } else {
        sim.reads_default_loss = true;
    }
    /* A target the loss function cannot measure is refused here. */
    if (const std::string why = my_loss.target_refusal(tgt); !why.empty())
        throw std::runtime_error("--varloss " + vname + " " + tgt_tok + ": " +
                                 why);
    if (std::find(sim.n_outvar.begin(), sim.n_outvar.end(), vname) ==
        sim.n_outvar.end())
        sim.n_outvar.push_back(vname);
    sim.n_varloss.push_back(vname);
    sim.i_varloss.push_back(static_cast<std::size_t>(std::distance(
        sim.n_outvar.begin(),
        std::find(sim.n_outvar.begin(), sim.n_outvar.end(), vname))));
    sim.t_varloss.push_back(tgt);
    sim.w_varloss.push_back(wgt);
    sim.lossFn.push_back(std::move(my_loss));
    s.i = j;
    /* Every value up to the next option was the loss override's. */
    s.option_read("--varloss takes <name> <target> <weight> "
                  "[<loss> [<param>...]]");
}

/* --invar <name> <value>: a LAMMPS variable on its command line. */
void Handlers::PC_LAMMPS_invar(OptionRead& r)
{
    ScriptOptions& s = *r.script;
    if (s.i + 2 >= s.tokens.size())
        throw std::runtime_error(s.key + " --invar takes two values, the "
                                 "name of a LAMMPS variable and its value");
    s.reject_reserved("--invar", s.tokens[s.i + 1]);
    s.define_variable("--invar", s.tokens[s.i + 1]);
    s.sim.invar.push_back("-var");
    s.sim.invar.push_back(s.tokens[++s.i]);
    s.sim.invar.push_back(s.tokens[++s.i]);
    ++s.i;
    s.option_read("--invar takes <name> <value>");
}

/* --outvar <name>: a LAMMPS variable the line reads back. */
void Handlers::PC_LAMMPS_outvar(OptionRead& r)
{
    ScriptOptions& s = *r.script;
    if (s.i + 1 >= s.tokens.size())
        throw std::runtime_error(s.key + " --outvar takes one value, the "
                                 "name of a LAMMPS variable");
    const std::string vname = s.tokens[++s.i];
    if (std::find(s.sim.n_outvar.begin(), s.sim.n_outvar.end(), vname) ==
        s.sim.n_outvar.end())
        s.sim.n_outvar.push_back(vname);
    ++s.i;
    s.option_read("--outvar takes one value");
}

/* --failscore <value>, once: on PC_LAMMPS the value of the line's
 * variables when its simulation fails (any finite number); on
 * PC_FAILPROBE a failure score (a positive finite number). */
void Handlers::PC_LAMMPS_failscore(OptionRead& r)
{
    ScriptOptions& s = *r.script;
    s.once("--failscore");
    if (s.i + 1 >= s.tokens.size())
        throw std::runtime_error(s.key + " --failscore takes one value, a "
                                 "number");
    s.sim.failscore =
        s.sim.is_failprobe
            ? parse_positive(s.tokens[++s.i], "PC_FAILPROBE --failscore",
                             "finite number")
            : parse_finite(s.tokens[++s.i], "--failscore");
    s.sim.reads_default_failscore = false;
    ++s.i;
    s.option_read("--failscore takes one value");
}

/* --maxtime <time>, once: the line's own script timeout. */
void Handlers::PC_LAMMPS_maxtime(OptionRead& r)
{
    ScriptOptions& s = *r.script;
    s.once("--maxtime");
    if (s.i + 1 >= s.tokens.size())
        throw std::runtime_error(s.key + " --maxtime takes one value, a time "
                                 "limit (off, unlimited or [[HH:]MM:]SS)");
    s.sim.maxtime = s.tokens[++s.i];
    (void)parse_script_timeout(s.sim.maxtime, "--maxtime");
    s.sim.reads_default_maxtime = false;
    ++s.i;
    s.option_read("--maxtime takes one value");
}

/* --hotvar <name> <min> <max> [<initial>]. */
void Handlers::PC_LAMMPS_hotvar(OptionRead& r)
{
    ScriptOptions& s = *r.script;
    const auto& tokens = s.tokens;
    const auto is_flag = [](const std::string& t) {
        return !t.empty() && t[0] == '-' && !is_number_shaped(t);
    };
    std::size_t j = s.i + 1;
    if (j + 2 >= tokens.size())
        throw std::runtime_error("--hotvar requires <name> <min> <max> "
                                 "[<initial>]");
    LammpsSim::HotVarDecl hv;
    hv.name = tokens[j++];
    if (hv.name.empty() || is_flag(hv.name))
        throw std::runtime_error("--hotvar: <name> must be a non-empty "
                                 "identifier");
    for (const auto& other : s.sim.hotvars)
        if (other.name == hv.name)
            throw std::runtime_error("--hotvar " + hv.name + " is declared "
                                     "twice on this line; each variable is "
                                     "declared once");
    hv.min = parse_finite(tokens[j++], "--hotvar " + hv.name + " <min>");
    hv.max = parse_finite(tokens[j++], "--hotvar " + hv.name + " <max>");
    if (!(hv.max > hv.min))
        throw std::runtime_error("--hotvar " + hv.name +
                                 ": <max> must exceed <min>");
    /* Optional <initial>: a token that is not an option is it. */
    if (j < tokens.size() && !is_flag(tokens[j]))
        hv.initial = parse_finite(tokens[j++],
                                  "--hotvar " + hv.name + " <initial>");
    s.sim.hotvars.push_back(std::move(hv));
    s.i = j;
    s.option_read("--hotvar takes <name> <min> <max> [<initial>]");
}

/* --failvar <name> [<scale>]: a violation measure the script sets. */
void Handlers::PC_LAMMPS_failvar(OptionRead& r)
{
    ScriptOptions& s = *r.script;
    const auto& tokens = s.tokens;
    const auto is_flag = [](const std::string& t) {
        return !t.empty() && t[0] == '-' && !is_number_shaped(t);
    };
    std::size_t j = s.i + 1;
    if (j >= tokens.size())
        throw std::runtime_error("--failvar requires <name> [<scale>]");
    LammpsSim::FailVarDecl fv;
    fv.name = tokens[j++];
    if (fv.name.empty() || is_flag(fv.name))
        throw std::runtime_error("--failvar: <name> must be a non-empty "
                                 "identifier");
    for (const auto& other : s.sim.failvars)
        if (other.name == fv.name)
            throw std::runtime_error("--failvar " + fv.name + " is declared "
                                     "twice on this line; each variable is "
                                     "declared once");
    if (j < tokens.size() && !is_flag(tokens[j])) {
        fv.scale = parse_positive(tokens[j++],
                                  "--failvar " + fv.name + " <scale>");
        fv.scale_given = true;
    }
    s.sim.failvars.push_back(std::move(fv));
    s.i = j;
    s.option_read("--failvar takes <name> [<scale>]");
}

/* --ncpu, a removed option, refused rather than passed on to LAMMPS. */
void Handlers::PC_LAMMPS_ncpu(OptionRead& r)
{
    throw std::runtime_error(
        r.script->key + " --ncpu: a removed option: it set the MPI ranks of "
        "one script in an MPI build of Tadah! 1.5.0; THREADS sets the number "
        "of workers that run the scripts, and PC_ISOLATION whether they run "
        "in worker processes or inside the tadah process");
}

namespace {
/* --gate <T> and --s0 <v0> of a PC_FAILPROBE line: read before LammpsSim
 * reads the line (ProbeOptions), each once; LammpsSim skips each with its
 * value, so a word after one is named with it (ScriptOptions). */
void probe_option(OptionRead& r, bool gate)
{
    if (ScriptOptions* s = r.script) {
        const std::string option = s->tokens[s->i];
        s->i += 2;
        s->option_read(option + " takes one value");
        return;
    }
    ProbeOptions& p = *r.probe;
    const std::string what = "PC_FAILPROBE " + r.word;
    bool& seen = gate ? p.gate_seen : p.s0_seen;
    if (seen)
        throw std::runtime_error(str(r.option.repeat_msg));
    seen = true;
    if (p.i + 1 >= p.tokens.size())
        throw std::runtime_error(what + " requires a value");
    if (gate)
        p.gate = read_number(arg0(r.option), p.tokens[p.i + 1], r.word);
    else
        p.s0 = read_positive(arg0(r.option), p.tokens[p.i + 1], r.word);
    ++p.i;
}
}  // namespace

void Handlers::PC_FAILPROBE_gate(OptionRead& r) { probe_option(r, true); }
void Handlers::PC_FAILPROBE_s0(OptionRead& r) { probe_option(r, false); }

/* ------------------------------- PC_OBSERVABLE cohesive_energy ---- */

namespace {
const std::string kCohesive = "cohesive_energy";

/* A finite number, the value of an option. */
double ce_number(const std::string& s, const std::string& option)
{
    double v = 0.0;
    if (!parse_number(s, v))
        throw std::runtime_error("PC_OBSERVABLE " + kCohesive + ": " + option +
                                 " takes a finite number (got '" + s + "')" +
                                 number_underflow_note(s));
    return v;
}

/* An integer from `min` to INT_MAX. */
int ce_integer(const std::string& s, const std::string& option, int min)
{
    return static_cast<int>(parse_uint(
        s, "PC_OBSERVABLE " + kCohesive + ": " + option,
        static_cast<std::uint64_t>(min),
        static_cast<std::uint64_t>(std::numeric_limits<int>::max())));
}

/* The value after the option at c.i. */
const std::string& ce_value(CohesiveOptions& c, const std::string& flag)
{
    if (c.i + 1 >= c.tokens.size())
        throw std::runtime_error("PC_OBSERVABLE cohesive_energy: '" + flag +
                                 "' requires a value");
    return c.tokens[++c.i];
}

/* Each option is given once: a second would replace the first's values. */
void ce_once(OptionRead& r)
{
    CohesiveOptions& c = *r.cohesive;
    if (std::find(c.seen.begin(), c.seen.end(), r.word) != c.seen.end())
        throw std::runtime_error(str(r.option.repeat_msg));
    c.seen.push_back(r.word);
}
}  // namespace

void Handlers::PC_OBSERVABLE_cohesive_energy_species(OptionRead& r)
{
    ce_once(r);
    CohesiveOptions& c = *r.cohesive;
    c.cfg.species = ce_value(c, r.word);
    /* An element symbol, in its case ("Ta"). */
    tadah::core::PeriodicTable::initialize();
    try {
        (void)tadah::core::PeriodicTable::find_by_symbol(c.cfg.species);
    } catch (const std::exception&) {
        throw std::runtime_error("PC_OBSERVABLE cohesive_energy: --species " +
                                 c.cfg.species +
                                 " is not the symbol of an element");
    }
    c.seen_species = true;
}

void Handlers::PC_OBSERVABLE_cohesive_energy_target(OptionRead& r)
{
    ce_once(r);
    CohesiveOptions& c = *r.cohesive;
    c.target_tok = ce_value(c, r.word);
    c.cfg.target = ce_number(c.target_tok, r.word);
    c.seen_target = true;
}

void Handlers::PC_OBSERVABLE_cohesive_energy_weight(OptionRead& r)
{
    ce_once(r);
    CohesiveOptions& c = *r.cohesive;
    const std::string& v = ce_value(c, r.word);
    c.cfg.weight = ce_number(v, r.word);
    /* A weight below 0 would reward a worse cohesive energy. */
    if (c.cfg.weight < 0.0)
        throw std::runtime_error("PC_OBSERVABLE cohesive_energy: --weight "
                                 "must be >= 0 (got " + v + ")");
    c.seen_weight = true;
}

void Handlers::PC_OBSERVABLE_cohesive_energy_hotvar(OptionRead& r)
{
    ce_once(r);
    CohesiveOptions& c = *r.cohesive;
    /* --hotvar a0 <lo> <hi> [<init>] */
    const std::string& name = ce_value(c, r.word);
    if (name != "a0")
        throw std::runtime_error("PC_OBSERVABLE cohesive_energy: --hotvar "
                                 "names a0, the lattice constant, the one "
                                 "variable of the term (got '" + name + "')");
    c.hotvar_lo_tok = ce_value(c, "--hotvar a0 <lo>");
    c.cfg.a0_min = ce_number(c.hotvar_lo_tok, "--hotvar a0 <lo>");
    c.hotvar_hi_tok = ce_value(c, "--hotvar a0 <hi>");
    c.cfg.a0_max = ce_number(c.hotvar_hi_tok, "--hotvar a0 <hi>");
    if (!(c.cfg.a0_max > c.cfg.a0_min))
        throw std::runtime_error("PC_OBSERVABLE cohesive_energy: --hotvar a0 "
                                 "<hi> must be greater than <lo> (got <lo> " +
                                 c.hotvar_lo_tok + ", <hi> " +
                                 c.hotvar_hi_tok + ")");
    /* Optional initial: a token that is not an option. */
    if (c.i + 1 < c.tokens.size() &&
        c.tokens[c.i + 1].substr(0, 2) != "--") {
        c.hotvar_init_tok = c.tokens[++c.i];
        c.cfg.a0_initial = ce_number(c.hotvar_init_tok, "--hotvar a0 <init>");
    }
    c.seen_hotvar = true;
}

void Handlers::PC_OBSERVABLE_cohesive_energy_lattice(OptionRead& r)
{
    ce_once(r);
    CohesiveOptions& c = *r.cohesive;
    const std::string& lattice = ce_value(c, r.word);
    if (lattice != "bcc")
        throw std::runtime_error("PC_OBSERVABLE cohesive_energy: --lattice "
                                 "takes bcc, the one lattice of the term "
                                 "(got '" + lattice + "')");
}

void Handlers::PC_OBSERVABLE_cohesive_energy_supercell(OptionRead& r)
{
    ce_once(r);
    CohesiveOptions& c = *r.cohesive;
    /* At least one cell along each axis. */
    c.cfg.nx = ce_integer(ce_value(c, r.word), "--supercell <nx>", 1);
    c.cfg.ny = ce_integer(ce_value(c, r.word), "--supercell <ny>", 1);
    c.cfg.nz = ce_integer(ce_value(c, r.word), "--supercell <nz>", 1);
}

void Handlers::PC_OBSERVABLE_cohesive_energy_loss(OptionRead& r)
{
    ce_once(r);
    CohesiveOptions& c = *r.cohesive;
    /* --loss <name> [<parameter>], as LOSS and --varloss take it. */
    const std::string& kind = ce_value(c, r.word);
    const auto type = tadah::LossFunction::from_name(
        kind, "PC_OBSERVABLE cohesive_energy: --loss");
    std::vector<double> params;
    while (c.i + 1 < c.tokens.size() &&
           c.tokens[c.i + 1].substr(0, 2) != "--")
        params.push_back(ce_number(c.tokens[++c.i],
                                   "--loss " + kind + " <param>"));
    c.cfg.lossfn = tadah::LossFunction(type, params);
    c.cfg.lossfn_given = true;
}

void Handlers::PC_OBSERVABLE_cohesive_energy_maxiter(OptionRead& r)
{
    ce_once(r);
    CohesiveOptions& c = *r.cohesive;
    c.cfg.relax_maxiter = static_cast<std::size_t>(
        ce_integer(ce_value(c, r.word), "--maxiter", 0));
}

void Handlers::PC_OBSERVABLE_cohesive_energy_maxeval(OptionRead& r)
{
    ce_once(r);
    CohesiveOptions& c = *r.cohesive;
    c.cfg.relax_maxeval = static_cast<std::size_t>(
        ce_integer(ce_value(c, r.word), "--maxeval", 0));
}

void Handlers::PC_OBSERVABLE_cohesive_energy_maxwalltime(OptionRead& r)
{
    ce_once(r);
    CohesiveOptions& c = *r.cohesive;
    c.cfg.relax_max_wall_time_s =
        ce_number(ce_value(c, r.word), "--max-wall-time");
}

void Handlers::PC_OBSERVABLE_cohesive_energy_etol(OptionRead& r)
{
    ce_once(r);
    CohesiveOptions& c = *r.cohesive;
    c.cfg.relax_etol = ce_number(ce_value(c, r.word), "--etol");
}

void Handlers::PC_OBSERVABLE_cohesive_energy_fmaxconverged(OptionRead& r)
{
    ce_once(r);
    CohesiveOptions& c = *r.cohesive;
    const std::string& v = ce_value(c, r.word);
    c.cfg.eq_fmax_threshold = ce_number(v, "--fmax-converged");
    if (!(c.cfg.eq_fmax_threshold > 0.0))
        throw std::runtime_error("PC_OBSERVABLE cohesive_energy: "
                                 "--fmax-converged must be > 0 (got " + v +
                                 ")");
}

void Handlers::PC_OBSERVABLE_cohesive_energy_ptol(OptionRead& r)
{
    ce_once(r);
    CohesiveOptions& c = *r.cohesive;
    const std::string& v = ce_value(c, r.word);
    c.cfg.relax_ptol = ce_number(v, "--ptol");
    if (!(c.cfg.relax_ptol > 0.0))
        throw std::runtime_error("PC_OBSERVABLE cohesive_energy: --ptol must "
                                 "be > 0 (got " + v + ")");
}

/* --unconverged const <V> | force-scaled <slope>. */
void Handlers::PC_OBSERVABLE_cohesive_energy_unconverged(OptionRead& r)
{
    ce_once(r);
    CohesiveOptions& c = *r.cohesive;
    using Strat = CohesiveEnergyObservable::Config::UnconvergedStrategy;
    const std::string& kind = ce_value(c, r.word);
    if (kind == "const")
        c.cfg.unconverged_strategy = Strat::Const;
    else if (kind == "force-scaled")
        c.cfg.unconverged_strategy = Strat::ForceScaled;
    else
        throw std::runtime_error("PC_OBSERVABLE cohesive_energy: "
                                 "--unconverged kind must be 'const' or "
                                 "'force-scaled' (got '" + kind + "')");
    const std::string v = ce_value(c, "--unconverged " + kind + " <value>");
    c.cfg.unconverged_value = ce_number(v, "--unconverged " + kind);
    /* A negative slope would move the value of an unconverged relaxation
     * towards the target, lowering its loss. */
    if (c.cfg.unconverged_strategy == Strat::ForceScaled &&
        c.cfg.unconverged_value < 0.0)
        throw std::runtime_error("PC_OBSERVABLE cohesive_energy: the slope "
                                 "of --unconverged force-scaled must be >= 0 "
                                 "(got " + v + ")");
}

/* PC_OBSERVABLE cohesive_energy --species <el> --target <E> --weight <w>
 * --hotvar a0 <lo> <hi> [<init>] [options]. */
void Handlers::PC_OBSERVABLE_cohesive_energy(FormRead& r)
{
    const auto& tokens = r.tokens;
    CohesiveEnergyObservable::Config cfg;
    cfg.nx = cfg.ny = cfg.nz = 4;
    /* The loss function of the LOSS line read above the line, as for a
     * --varloss term; --loss replaces it. */
    cfg.lossfn = r.default_loss;
    bool seen_species = false, seen_target = false, seen_weight = false;
    bool seen_hotvar = false;
    std::string target_tok, hotvar_lo_tok, hotvar_hi_tok, hotvar_init_tok;
    std::vector<std::string> seen;
    std::size_t i = 1;
    CohesiveOptions c{tokens, i, cfg, seen, seen_species, seen_target,
                      seen_weight, seen_hotvar, target_tok, hotvar_lo_tok,
                      hotvar_hi_tok, hotvar_init_tok};
    for (; i < tokens.size(); ++i) {
        const LineOption* o = find_option(r.form.id_text, tokens[i]);
        if (!o)
            throw std::runtime_error(
                "PC_OBSERVABLE cohesive_energy: unknown option '" + tokens[i] +
                "' (it takes --species, --target, --weight, --lattice, "
                "--supercell, --hotvar, --loss, --maxiter, --max-wall-time, "
                "--etol, --maxeval, --fmax-converged, --ptol and "
                "--unconverged)");
        OptionRead o_read{*o, tokens[i]};
        o_read.cohesive = &c;
        call(*o, o_read);
    }
    if (!seen_species)
        throw std::runtime_error("PC_OBSERVABLE cohesive_energy: --species "
                                 "is required");
    if (!seen_target)
        throw std::runtime_error("PC_OBSERVABLE cohesive_energy: --target is "
                                 "required");
    if (!seen_weight)
        throw std::runtime_error("PC_OBSERVABLE cohesive_energy: --weight is "
                                 "required");
    if (!seen_hotvar)
        throw std::runtime_error("PC_OBSERVABLE cohesive_energy: --hotvar a0 "
                                 "<lo> <hi> [<init>] is required");
    /* The start of a chain's first relaxation lies in the window its
     * relaxed a0 must stay in. */
    if (cfg.a0_initial &&
        (*cfg.a0_initial < cfg.a0_min || *cfg.a0_initial > cfg.a0_max) &&
        cfg.a0_max > cfg.a0_min)
        throw std::runtime_error("PC_OBSERVABLE cohesive_energy: --hotvar a0 "
                                 "<init> " + hotvar_init_tok +
                                 " is outside its window [" + hotvar_lo_tok +
                                 ", " + hotvar_hi_tok + "]");
    /* A target the loss function cannot measure. */
    if (const std::string why = cfg.lossfn.target_refusal(cfg.target);
        !why.empty())
        throw std::runtime_error("PC_OBSERVABLE cohesive_energy --target " +
                                 target_tok + ": " + why);
    r.observable = std::make_unique<CohesiveEnergyObservable>(std::move(cfg));
}

}  // namespace hpotarget_registry
}  // namespace hpo
}  // namespace tadah
