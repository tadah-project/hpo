#include <tadah/hpo/validation_metric.h>
#include <tadah/hpo/strict_parse.h>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tadah::hpo {

bool metric_value_is_finite(double value) {
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    static_assert(std::numeric_limits<double>::is_iec559);
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

double parse_metric_number(const std::string& token, const std::string& what) {
    double value = 0.0;
    if (!parse_number(token, value))
        throw std::runtime_error(
            (what.empty() ? std::string("Expected a finite number")
                          : what + " must be a finite number") +
            ", got '" + token + "'" + number_underflow_note(token));
    return value;
}

void ValidationMetricTargets::add(const std::vector<std::size_t>& indices,
                                   const ValidationMetricAssignment& a) {
    if (!metric_value_is_finite(a.target) || !metric_value_is_finite(a.weight) || a.weight < 0)
        throw std::runtime_error(key_ + " requires a finite target and nonnegative finite weight");
    for (const auto cap : {a.caps.abs_above, a.caps.rel_above})
        if (cap && (!metric_value_is_finite(*cap) || *cap <= 0))
            throw std::runtime_error(key_ + " skip thresholds must be finite and positive");
    if (indices.empty()) {
        if (global_)
            throw std::runtime_error(key_ + ": repeated declaration (first declared on line "
                + std::to_string(global_->line) + ")");
        if (enabled()) throw std::runtime_error(key_ + ": cannot mix global and indexed forms");
        global_ = a;
    } else {
        if (global_) throw std::runtime_error(key_ + ": cannot mix global and indexed forms");
        auto pending = indexed_;
        for (auto i : indices) {
            if (i == 0) throw std::runtime_error(key_ + ": validation indices start at 1");
            const auto [it, inserted] = pending.emplace(i, a);
            if (!inserted)
                throw std::runtime_error(key_ + ": duplicate assignment for dataset " + std::to_string(i)
                    + " (first declared on line " + std::to_string(it->second.line) + ")");
        }
        indexed_ = std::move(pending);
    }
    resolved_.clear();
    files_.clear();
}

void ValidationMetricTargets::resolve(const std::vector<std::string>& files) {
    if (!enabled()) return;
    if (files.empty()) throw std::runtime_error(key_ + " requires validation DBFILE entries");
    if (indexed() && indexed_.rbegin()->first > files.size()) {
        const auto& [i, a] = *indexed_.rbegin();
        throw std::runtime_error(key_ + ": dataset " + std::to_string(i) + " on line "
            + std::to_string(a.line) + " is outside validation DBFILE range 1-" + std::to_string(files.size()));
    }
    std::vector<ValidationMetricAssignment> resolved;
    std::string missing;
    for (std::size_t i = 0; i < files.size(); ++i) {
        if (global_) resolved.push_back(*global_);
        else if (const auto it = indexed_.find(i + 1); it != indexed_.end()) resolved.push_back(it->second);
        else missing += "\n  dataset " + std::to_string(i + 1) + " (" + files[i] + ")";
    }
    if (!missing.empty()) throw std::runtime_error(key_ + ": missing target and weight for:" + missing);
    resolved_ = std::move(resolved);
    files_ = files;
}

bool ValidationMetricTargets::has_relative_cap() const {
    if (global_ && global_->caps.rel_above) return true;
    for (const auto& kv : indexed_)
        if (kv.second.caps.rel_above) return true;
    return false;
}
const ValidationMetricAssignment& ValidationMetricTargets::global() const {
    if (!global_) throw std::logic_error(key_ + ": no global assignment");
    return *global_;
}
const std::vector<ValidationMetricAssignment>& ValidationMetricTargets::datasets() const {
    if (resolved_.empty()) throw std::logic_error(key_ + ": validation targets have not been resolved");
    return resolved_;
}
void ValidationMetricTargets::check_size(std::size_t n) const {
    if (!enabled() || n == 0 || (indexed() && datasets().size() != n)
        || (!resolved_.empty() && resolved_.size() != n))
        throw std::logic_error(key_ + ": validation metric/assignment count mismatch");
}
double ValidationMetricTargets::loss(const LossFunction& fn, double average,
                                      const std::vector<double>& measured,
                                      std::vector<double>& per_dataset) const {
    check_size(measured.size());
    per_dataset.resize(measured.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < measured.size(); ++i) {
        if (!is_measured(i)) {  // no data: no loss term
            per_dataset[i] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        const auto& a = indexed() ? resolved_[i] : *global_;
        per_dataset[i] = fn.loss(measured[i], a.target, a.weight);
        sum += per_dataset[i];
    }
    return indexed() ? sum / measured.size() : fn.loss(average, global_->target, global_->weight);
}
std::vector<double> ValidationMetricTargets::slopes(const LossFunction& fn, double average,
                                                    const std::vector<double>& measured) const {
    check_size(measured.size());
    std::vector<double> out(measured.size(), 0.0);
    // A global target is the loss of the average over the measured datasets.
    std::size_t n_measured = 0;
    for (std::size_t i = 0; i < measured.size(); ++i) n_measured += is_measured(i);
    for (std::size_t i = 0; i < measured.size(); ++i) {
        if (!is_measured(i)) continue;  // no data: slope 0
        const auto& a = indexed() ? resolved_[i] : *global_;
        out[i] = indexed()
            ? fn.loss_prime_observed(measured[i], a.target, a.weight) / measured.size()
            : fn.loss_prime_observed(average, a.target, a.weight) / n_measured;
    }
    return out;
}

void ValidationMetricTargets::set_measured(std::vector<bool> measured) {
    if (!measured.empty() && !resolved_.empty() && measured.size() != resolved_.size())
        throw std::logic_error(key_ + ": measured-dataset count mismatch");
    measured_ = std::move(measured);
}

std::vector<std::string> ValidationMetricTargets::descriptions() const {
    std::vector<std::string> out;
    if (!enabled()) return out;
    auto format = [&](const std::string& label, const ValidationMetricAssignment& a) {
        std::ostringstream s;
        s << label << " target=" << std::setprecision(12) << a.target << " weight=" << a.weight;
        if (a.caps.abs_above) s << " --skip-above " << *a.caps.abs_above;
        if (a.caps.rel_above) s << " --skip-rel " << *a.caps.rel_above;
        out.push_back(s.str());
    };
    if (global_) format("global (loss of average metric)", *global_);
    else {
        out.push_back("indexed (average of dataset losses)");
        for (const auto& [i, a] : indexed_)
            format("[" + std::to_string(i) + "]"
                 + (i <= files_.size() ? " " + files_[i - 1] : ""), a);
    }
    return out;
}

std::optional<MetricSkipTracker::Trigger> MetricSkipTracker::check(
        const std::string& key, double value, const MetricSkipCaps& caps,
        std::vector<std::pair<std::string, double>>* deferred) {
    if (!metric_value_is_finite(value)) return std::nullopt;
    std::lock_guard<std::mutex> lock(mutex_);
    auto [it, inserted] = best_.emplace(key, std::numeric_limits<double>::infinity());
    const double best = it->second;
    const bool abs = caps.abs_above && value > *caps.abs_above;
    const bool rel = caps.rel_above && metric_value_is_finite(best) && value > *caps.rel_above * best;
    if (abs || rel) {
        const double cap = abs ? *caps.abs_above : *caps.rel_above * best;
        // A zero best makes every positive value a relative violation.
        const double excess = cap > 0 ? value / cap - 1 : std::numeric_limits<double>::max();
        return Trigger{abs, rel, best, excess, ++skips_};
    }
    if (value < best) {
        if (deferred) deferred->emplace_back(key, value);
        else it->second = value;
    }
    return std::nullopt;
}

void MetricSkipTracker::note_best(const std::string& key, double value) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto [it, inserted] = best_.emplace(key, std::numeric_limits<double>::infinity());
    (void)inserted;
    if (value < it->second) it->second = value;
}
} // namespace tadah::hpo
