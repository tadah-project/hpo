#pragma once

#include <tadah/hpo/loss_function.h>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace tadah::hpo {

struct MetricSkipCaps {
    std::optional<double> abs_above;
    std::optional<double> rel_above;
};

// Works in the Release build too (-ffast-math may fold std::isfinite).
bool metric_value_is_finite(double value);
/* A finite number of the number grammar every key reads (strict_parse.h,
 * parse_number: decimal, a leading '+' allowed, no hexadecimal). Otherwise
 * throws "<what> must be a finite number, got '<tok>'" ("Expected a finite
 * number, got '<tok>'" when `what` is empty), with parse_finite's
 * too-small note. */
double parse_metric_number(const std::string& token,
                           const std::string& what = "");

struct ValidationMetricAssignment {
    double target = 0.0;
    double weight = 0.0;
    MetricSkipCaps caps;
    std::size_t line = 0;
};

/** One global target or a complete, disjoint set of one-based assignments.
 * resolve() binds file identities before host evaluations. Indexed callers
 * must resolve first; the global form is also usable by standalone gradient
 * clients with no validation filenames. */
class ValidationMetricTargets {
public:
    explicit ValidationMetricTargets(std::string key) : key_(std::move(key)) {}
    bool enabled() const { return global_.has_value() || !indexed_.empty(); }
    bool indexed() const { return !indexed_.empty(); }
    void add(const std::vector<std::size_t>& indices,
             const ValidationMetricAssignment& assignment);
    void resolve(const std::vector<std::string>& files);
    const ValidationMetricAssignment& global() const;
    const std::vector<ValidationMetricAssignment>& datasets() const;
    const std::vector<std::string>& files() const { return files_; }
    // Returns one contribution; per_dataset contains unaveraged losses.
    double loss(const LossFunction& fn, double average,
                const std::vector<double>& measured,
                std::vector<double>& per_dataset) const;
    // d(contribution)/d(metric_i), including the 1/N dataset average.
    std::vector<double> slopes(const LossFunction& fn, double average,
                               const std::vector<double>& measured) const;
    // Consistent configuration descriptions for journal and pot headers.
    std::vector<std::string> descriptions() const;
    /** The datasets the metric measures (MK, 2026-10-06): one without data
     *  for it (every structure with force weight 0, for PC_FRMSE) is not
     *  measured; its value is nan, its loss term and slope are 0, and a
     *  global target is the loss of the average over the measured
     *  datasets. Empty (the default): every dataset. */
    void set_measured(std::vector<bool> measured);
    const std::vector<bool>& measured() const { return measured_; }
    /** Whether a --skip-rel cap is set, on the global target or on any
     *  dataset's (resolved or not). */
    bool has_relative_cap() const;

private:
    std::string key_;
    std::optional<ValidationMetricAssignment> global_;
    std::map<std::size_t, ValidationMetricAssignment> indexed_;
    std::vector<ValidationMetricAssignment> resolved_;
    std::vector<std::string> files_;
    std::vector<bool> measured_;
    bool is_measured(std::size_t i) const { return measured_.empty() || measured_[i]; }
    void check_size(std::size_t size) const;
};

/** Thread-shared, run-owned pre-filter histories. No process-global slots.
 * Copies of a host share its tracker; new hosts start with empty histories. */
class MetricSkipTracker {
public:
    struct Trigger {
        bool absolute = false;
        bool relative = false;
        double best = 0.0;
        double excess = 0.0;
        std::size_t count = 0;
    };
    std::optional<Trigger> check(const std::string& key, double value,
                                 const MetricSkipCaps& caps);
private:
    std::mutex mutex_;
    std::map<std::string, double> best_;
    std::size_t skips_ = 0;
};
} // namespace tadah::hpo
