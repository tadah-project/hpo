#include <tadah/hpo/tolerance_ladder.h>

#include <algorithm>
#include <cmath>

namespace tadah {
namespace hpo {

namespace {

/* Default ladder generator: start at 1e-3, descend by 0.1x per rung,
 * stop when we cross the floor.  */
std::vector<double> default_ladder(double floor, std::size_t max_rungs)
{
    std::vector<double> out;
    out.reserve(max_rungs);
    double cur = 1.0e-3;
    for (std::size_t i = 0; i < max_rungs; ++i) {
        if (cur < floor) break;
        out.push_back(cur);
        cur *= 0.1;
    }
    /* Always include at least one rung so loops downstream don't
     * trip on empty ladders. */
    if (out.empty()) out.push_back(std::max(floor, 1.0e-3));
    return out;
}

}  // namespace

ToleranceLadder ToleranceLadder::from_explicit(std::vector<double> glo,
                                               std::vector<double> loc)
{
    ToleranceLadder t;
    t.glo_ = std::move(glo);
    t.loc_ = std::move(loc);
    /* Sort descending -- rung k+1 must be tighter than rung k. */
    std::sort(t.glo_.begin(), t.glo_.end(), std::greater<double>{});
    std::sort(t.loc_.begin(), t.loc_.end(), std::greater<double>{});
    return t;
}

ToleranceLadder ToleranceLadder::build_default(
        std::optional<double> noise_rel_spread, std::size_t default_rungs)
{
    const double floor =
        noise_rel_spread ? std::max(1.0e-9, 10.0 * *noise_rel_spread)
                         : 1.0e-9;
    ToleranceLadder t;
    t.glo_ = default_ladder(floor, default_rungs);
    t.loc_ = default_ladder(floor, default_rungs + 2);   // 2 extra fine rungs
    return t;
}

double ToleranceLadder::glo(int round) const
{
    if (glo_.empty()) return 1.0e-3;
    const int idx = std::clamp(round - 1, 0, static_cast<int>(glo_.size()) - 1);
    return glo_[static_cast<std::size_t>(idx)];
}

double ToleranceLadder::loc(int round, int step) const
{
    if (loc_.empty()) return 1.0e-3;
    /* step is 1-indexed within the cumulative ladder for round k. */
    const int s = std::clamp(step - 1, 0, static_cast<int>(loc_.size()) - 1);
    /* Round-cap: round k visits up to k local rungs. */
    const int cap = std::clamp(round - 1, 0, static_cast<int>(loc_.size()) - 1);
    const int idx = std::min(s, cap);
    return loc_[static_cast<std::size_t>(idx)];
}

int ToleranceLadder::loc_steps_for(int round) const
{
    if (loc_.empty()) return 0;
    return std::min(static_cast<int>(loc_.size()), std::max(round, 1));
}

void ToleranceLadder::trim_below(double floor)
{
    auto trim = [floor](std::vector<double>& v) {
        v.erase(std::remove_if(v.begin(), v.end(),
                               [floor](double x) { return x < floor; }),
                v.end());
        if (v.empty()) v.push_back(floor);
    };
    trim(glo_);
    trim(loc_);
}

}   // namespace hpo
}   // namespace tadah
