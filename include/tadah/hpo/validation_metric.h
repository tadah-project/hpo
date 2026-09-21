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
double parse_metric_number(const std::string& token);

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

private:
    std::string key_;
    std::optional<ValidationMetricAssignment> global_;
    std::map<std::size_t, ValidationMetricAssignment> indexed_;
    std::vector<ValidationMetricAssignment> resolved_;
    std::vector<std::string> files_;
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
