#include <tadah/hpo/check_summary.h>

#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/observables/observable.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/search_space.h>
#include <tadah/hpo/strategy_rules.h>   // strategy_type_name
#include <tadah/hpo/validation_metric.h>

#include <algorithm>
#include <sstream>
#include <utility>
#include <vector>

namespace tadah::hpo {

namespace {

/* A number as the target file writes it: six significant digits, no
 * trailing zeros (0.1, 4, 1e+09). */
std::string num(double v)
{
    std::ostringstream s;
    s << v;
    return s.str();
}

std::string padded(std::string s, std::size_t width)
{
    if (s.size() < width) s.resize(width, ' ');
    return s;
}

/* "<n> <word>" or "<n> <word>s". */
std::string count_of(std::size_t n, const std::string& word)
{
    return std::to_string(n) + " " + word + (n == 1 ? "" : "s");
}

/* The stops of one block, as the target file writes them. */
std::string stops_text(const StopCriteria& s)
{
    std::vector<std::string> parts;
    if (s.maxeval)  parts.push_back("MAXEVAL " + std::to_string(*s.maxeval));
    if (s.maxtime)  parts.push_back("MAXTIME " + num(*s.maxtime));
    if (s.stopval)  parts.push_back("STOPVAL " + num(*s.stopval));
    if (s.ftol_rel) parts.push_back("FTOL_REL " + num(*s.ftol_rel));
    if (s.ftol_abs) parts.push_back("FTOL_ABS " + num(*s.ftol_abs));
    if (s.xtol_rel) parts.push_back("XTOL_REL " + num(*s.xtol_rel));
    if (s.xtol_abs) parts.push_back("XTOL_ABS " + num(*s.xtol_abs));
    if (parts.empty()) return "stops: none set";
    std::string out = "stops: ";
    for (std::size_t i = 0; i < parts.size(); ++i)
        out += (i ? ", " : "") + parts[i];
    return out;
}

/* "<LIB>:<ALGO>; stops: ..." of one block. */
std::string search_text(const OptimizerSpec& s)
{
    return s.lib + ":" + s.algo + "; " + stops_text(s.stop);
}

/* A table of two columns: the first padded to its widest entry. */
void write_rows(std::ostream& out,
                const std::vector<std::pair<std::string, std::string>>& rows)
{
    std::size_t width = 0;
    for (const auto& r : rows) width = std::max(width, r.first.size());
    for (const auto& r : rows)
        out << "  " << padded(r.first, width) << "  " << r.second << "\n";
}

std::string target_weight(double target, double weight)
{
    return "target " + num(target) + ", weight " + num(weight);
}

/* ", loss <NAME> [<parameter>]": the loss function of a term. */
std::string loss_text(const LossFunction& fn)
{
    std::string out = ", loss " + LossFunction::to_string(fn.type());
    for (double p : fn.params()) out += " " + num(p);
    return out;
}

void write_search_space(std::ostream& out, const HPOTargets& trg)
{
    const SearchSpacePtr space = trg.space();
    const std::size_t n = space ? space->dim() : 0;
    out << "Search parameters: " << n << "\n";
    std::vector<std::pair<std::string, std::string>> rows;
    for (std::size_t i = 0; i < n; ++i) {
        const DimSpec& d = space->at(i);
        std::string box = "[" + num(d.lo) + ", " + num(d.hi) + "]  " +
                          space->describe_dim(i);
        if (d.origin != BoxOrigin::Explicit)
            box += "  (" + std::string(box_origin_name(d.origin)) + ")";
        rows.emplace_back(i < trg.spec.parameter_labels.size()
                              ? trg.spec.parameter_labels[i] : d.label,
                          box);
    }
    write_rows(out, rows);
}

void write_optimiser(std::ostream& out, const OptimizerSpec& sp)
{
    out << "Optimiser: STRATEGY TYPE "
        << strategy_type_name(sp.strategy_type) << "\n";
    std::vector<std::pair<std::string, std::string>> rows;
    /* In the order the run runs them: the PRESTAGE exploration first. */
    if (sp.strategy_type == StrategyType::PRESTAGE && sp.prestage) {
        rows.emplace_back("EXPLORE", search_text(*sp.prestage));
        if (sp.prestage->inner)
            rows.emplace_back("EXPLORE INNER", search_text(*sp.prestage->inner));
    }
    std::string strategy = search_text(sp);
    if (sp.strategy_type == StrategyType::MULTISTART) {
        strategy += "; N_STARTS " + std::to_string(sp.multistart.n_starts);
        if (sp.multistart.budget_per_start)
            strategy += ", BUDGET_PER_START " +
                        std::to_string(*sp.multistart.budget_per_start);
    } else if (sp.strategy_type == StrategyType::PRESTAGE) {
        strategy += "; TOPK " + std::to_string(sp.stage_topk);
    }
    rows.emplace_back("STRATEGY", strategy);
    if (sp.inner) rows.emplace_back("INNER", search_text(*sp.inner));
    if (sp.restart.active())
        rows.emplace_back("RESTART",
                          "AFTER_STAGNATION " +
                              std::to_string(sp.restart.after_stagnation) +
                              ", MAX " + std::to_string(sp.restart.max));
    write_rows(out, rows);
}

/* The rows of PC_ERMSE or PC_FRMSE: one for a global target, one per
 * validation dataset for indexed ones. */
void add_metric_rows(std::vector<std::pair<std::string, std::string>>& rows,
                     const std::string& key,
                     const ValidationMetricTargets& targets,
                     const std::string& loss)
{
    if (!targets.enabled()) return;
    if (!targets.indexed()) {
        const auto& a = targets.global();
        rows.emplace_back(key, target_weight(a.target, a.weight) + loss);
        return;
    }
    const auto& datasets = targets.datasets();
    const auto& files = targets.files();
    for (std::size_t i = 0; i < datasets.size(); ++i)
        rows.emplace_back(key + " (" + std::to_string(i + 1) + ")" +
                              (i < files.size() ? " " + files[i] : ""),
                          target_weight(datasets[i].target,
                                        datasets[i].weight) + loss);
}

/* Every term with its target, weight and loss function: the metric lines
 * take the last LOSS line's, a --varloss term its own or its line's (the
 * LOSS above the line), a PC_OBSERVABLE term its --loss or the LOSS above
 * its line. */
void write_loss(std::ostream& out, const HPOTargets& trg)
{
    out << "Loss terms:\n";
    const std::string metric_loss = loss_text(trg.globalLoss);
    std::vector<std::pair<std::string, std::string>> rows;
    if (trg.use_emae)
        rows.emplace_back("PC_EMAE",
                          target_weight(trg.emae, trg.w_emae) + metric_loss);
    add_metric_rows(rows, "PC_ERMSE", trg.ermse_targets, metric_loss);
    if (trg.use_errmse)
        rows.emplace_back("PC_ErRMSE", target_weight(trg.errmse, trg.w_errmse) +
                                           metric_loss);
    if (trg.use_ersq)
        rows.emplace_back("PC_ERSQ",
                          target_weight(trg.ersq, trg.w_ersq) + metric_loss);
    add_metric_rows(rows, "PC_FRMSE", trg.frmse_targets, metric_loss);
    add_metric_rows(rows, "PC_SRMSE_GPA", trg.srmse_targets, metric_loss);
    for (const auto& sim : trg.LS)
        for (std::size_t i = 0; i < sim.n_varloss.size(); ++i)
            rows.emplace_back("PC_LAMMPS " + sim.script + ": " +
                                  sim.n_varloss[i],
                              target_weight(sim.t_varloss[i],
                                            sim.w_varloss[i]) +
                                  loss_text(sim.lossFn[i]));
    for (std::size_t k = 0; k < trg.observables.size(); ++k)
        if (const auto& obs = trg.observables[k])
            rows.emplace_back("PC_OBSERVABLE " +
                                  observable_journal_label(*obs, k + 1),
                              target_weight(obs->target(), obs->weight()) +
                                  loss_text(obs->lossfn()));
    /* The labels differ in length too much to align: each row as
     * "<term>  <target, weight>". */
    for (const auto& r : rows)
        out << "  " << r.first << "  " << r.second << "\n";
}

/* ", with 2 warnings and 1 hint" for @p notes; "" for none. */
std::string notes_count(const std::vector<JournalNote>& notes)
{
    std::size_t warnings = 0, hints = 0;
    for (const auto& n : notes) (n.tag == "HINT" ? hints : warnings) += 1;
    std::vector<std::string> parts;
    if (warnings) parts.push_back(count_of(warnings, "warning"));
    if (hints)    parts.push_back(count_of(hints, "hint"));
    if (parts.empty()) return "";
    return ", with " + parts[0] + (parts.size() > 1 ? " and " + parts[1] : "");
}

}  // namespace

void write_check_notes(std::ostream& out,
                       const std::vector<JournalNote>& notes)
{
    if (notes.empty()) return;
    out << "Warnings and hints, which a run writes to its journal "
           "(hpo_run.log):\n";
    const std::string under(11, ' ');  // "  " + the tag column + "  "
    for (const auto& n : notes) {
        out << "  " << padded(n.tag, 7) << "  ";
        std::size_t start = 0;
        for (;;) {
            const std::size_t nl = n.text.find('\n', start);
            out << n.text.substr(start, nl == std::string::npos
                                            ? std::string::npos
                                            : nl - start) << "\n";
            if (nl == std::string::npos) break;
            start = nl + 1;
            out << under;
        }
    }
}

void write_check_summary(std::ostream& out, const std::string& target_file,
                         const HPOTargets& trg, std::size_t n_train,
                         std::size_t n_val,
                         const std::vector<JournalNote>& notes)
{
    out << "tadah hpo --check: " << target_file
        << " passed the startup checks" << notes_count(notes) << ".\n\n";
    write_search_space(out, trg);
    out << "\n";
    write_optimiser(out, trg.spec);
    out << "\n";
    write_loss(out, trg);
    out << "\nData: " << count_of(n_train, "training structure") << ", "
        << count_of(n_val, "validation structure") << "\n\n";
    if (!notes.empty()) {
        write_check_notes(out, notes);
        out << "\n";
    }
    out << "No candidate was evaluated and no file was written.\n";
    out.flush();
}

}  // namespace tadah::hpo
