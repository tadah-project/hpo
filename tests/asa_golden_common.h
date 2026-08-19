#pragma once
/* Shared deterministic problem for the ASA golden-neutrality gate:
 * the SAME cost function and the SAME explicit LCG drive both the
 * featured build (asa_testing::run_raw_featured) and the pristine
 * all-gates-off build (run_golden_pristine, defined in
 * test_asa_golden_pristine_tu.cpp). The cost function records every
 * evaluation so the two trajectories can be compared bitwise. */

#include <cstddef>
#include <vector>

namespace asa_golden {

struct Trajectory {
    std::vector<double> flat_x;  /* dim-strided evaluation points */
    std::vector<double> f;       /* returned costs, in call order  */
};

inline Trajectory& traj()
{
    static Trajectory t;
    return t;
}

inline unsigned long& lcg_state()
{
    static unsigned long s = 1;
    return s;
}

inline void reset(unsigned long seed)
{
    traj().flat_x.clear();
    traj().f.clear();
    lcg_state() = seed;
}

/* Ingber's myrand() LCG constants (asa_usr.c): portable, identical
 * draws for both builds. Returns in [0, 1). */
inline double golden_rng()
{
    auto& s = lcg_state();
    s = (25173UL * s + 13849UL) % 65536UL;
    return static_cast<double>(s) / 65536.0;
}

/* Smooth double-well quartic with a linear tilt; every state valid. */
inline double golden_cost(const double* x, long dim, int* flag)
{
    double f = 0.0;
    for (long i = 0; i < dim; ++i)
        f += 0.25 * (x[i] * x[i] - 2.0) * (x[i] * x[i] - 2.0) + 0.1 * x[i];
    auto& t = traj();
    t.flat_x.insert(t.flat_x.end(), x, x + dim);
    t.f.push_back(f);
    *flag = 1;
    return f;
}

struct GoldenResult {
    int    exit_status = 0;
    double final_cost  = 0.0;
    long   n_accepted  = 0;
    long   n_generated = 0;
    std::vector<double> best_x;
};

}  // namespace asa_golden

/* Runs the PRISTINE (all-algorithmic-gates-off, tadah_asaref_-renamed)
 * asa() at the canonical default options over the callbacks above.
 * Defined in test_asa_golden_pristine_tu.cpp. */
asa_golden::GoldenResult run_golden_pristine(std::size_t dim,
                                             const double* lo,
                                             const double* hi,
                                             const double* x0,
                                             long limit_generated);
