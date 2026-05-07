#include <tadah/hpo/progress_tracker.h>

#include <cmath>
#include <sstream>

namespace tadah {
namespace hpo {

void ProgressTracker::start_stage(const StageConfig& cfg)
{
    cfg_ = cfg;
    if (cfg_.window_size == 0) cfg_.window_size = 50;
    reset_stage();
}

void ProgressTracker::reset_stage()
{
    hist_.clear();
    best_              = std::numeric_limits<double>::infinity();
    best_window_start_ = std::numeric_limits<double>::infinity();
    n_obs_             = 0;
    sum_wall_ms_       = 0.0;
    opt_converged_     = false;
    opt_maxeval_       = false;
}

void ProgressTracker::observe(double loss, double wall_ms)
{
    ++n_obs_;
    sum_wall_ms_ += wall_ms;
    if (loss < best_) best_ = loss;
    hist_.emplace_back(loss, wall_ms);

    /* Slide the window. We track the loss observed at the front of the
     * window (i.e. what the caller's best-so-far was W evals ago). The
     * rate is then drop / window_size, which is sensitive to monotone
     * progress. Taking the min over the window — which we used to do —
     * collapses to current best when loss is monotone, falsely
     * reporting zero rate even while the optimiser is still improving. */
    while (hist_.size() > cfg_.window_size) hist_.pop_front();

    /* When the window is short the rate is undefined; once it has at
     * least 2 entries the front gives us the W-old reference. */
    if (!hist_.empty()) best_window_start_ = hist_.front().first;
    else                best_window_start_ = std::numeric_limits<double>::infinity();

    /* Hard maxeval safety cap. */
    if (cfg_.max_evals > 0 && n_obs_ >= cfg_.max_evals) opt_maxeval_ = true;
}

void ProgressTracker::mark_optimizer_converged() { opt_converged_ = true; }
void ProgressTracker::mark_optimizer_maxeval_hit() { opt_maxeval_   = true; }

double ProgressTracker::rate_per_eval() const
{
    if (hist_.size() < 2) return std::numeric_limits<double>::infinity();
    const double a = best_window_start_;
    const double b = best_;
    if (!std::isfinite(a) || !std::isfinite(b)) return 0.0;
    const double denom = std::max(std::fabs(a), 1e-300);
    const double drop  = std::max(0.0, a - b);
    return drop / denom / static_cast<double>(hist_.size());
}

double ProgressTracker::rate_per_second() const
{
    if (sum_wall_ms_ <= 0.0) return 0.0;
    const double a = best_window_start_;
    const double b = best_;
    if (!std::isfinite(a) || !std::isfinite(b)) return 0.0;
    const double denom = std::max(std::fabs(a), 1e-300);
    const double drop  = std::max(0.0, a - b);
    return drop / denom / (sum_wall_ms_ / 1000.0);
}

bool ProgressTracker::is_stuck() const
{
    /* Condition 1: not premature. */
    if (n_obs_ < std::max<std::size_t>(cfg_.min_evals, cfg_.window_size))
        return false;

    /* Condition 2: rate floor. */
    const double thresh = std::max(cfg_.rate_factor * cfg_.tol, 0.0);
    if (rate_per_eval() >= thresh) return false;

    /* Condition 3: optimiser-side termination signal. The user's
     * safety preference: don't pull the rug from under a slow but
     * still-progressing optimiser unless the optimiser itself has
     * stopped reporting fresh news. */
    if (!(opt_converged_ || opt_maxeval_)) return false;

    return true;
}

std::string ProgressTracker::stuck_reason() const
{
    std::ostringstream os;
    if (n_obs_ < std::max<std::size_t>(cfg_.min_evals, cfg_.window_size)) {
        os << "n_obs=" << n_obs_ << " < window=" << cfg_.window_size;
        return os.str();
    }
    os.setf(std::ios::scientific);
    os.precision(3);
    const double r = rate_per_eval();
    const double thresh = cfg_.rate_factor * cfg_.tol;
    os << "rate=" << r << "/eval";
    if (r < thresh) os << " < threshold=" << thresh;
    else            os << " >= threshold=" << thresh;
    if (opt_converged_) os << ", optimizer-converged";
    if (opt_maxeval_)   os << ", optimizer-maxeval";
    return os.str();
}

}   // namespace hpo
}   // namespace tadah
