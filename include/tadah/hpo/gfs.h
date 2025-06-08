/******************************************************************************
 * parameter_space_divider.h
 *
 * Divide an axis-aligned hyper-rectangular parameter space into an EXACT,
 * user-requested number of chunks (5–100 is the intended range, but any
 * positive value is accepted).  The class keeps the API of the previous
 * draft yet replaces the “same splits on every axis” rule by a greedy,
 * per-axis split allocator that guarantees
 *
 *      Π splits[i]  ==  target_chunks    (exact)
 *
 * while trying to keep the resulting hyper-rectangles as close as possible
 * in hyper-volume.  Optional log–scaling is still applied automatically on
 * axes whose ratio high/low ≥ 1000 and low > 0.
 *
 * No dynamic allocation except small std::vector, no 3rd-party deps besides
 * dlib, header-only, C++17.
 *
 * Edinburgh University – ELM / GPT-4
 ******************************************************************************/

#ifndef ELM_PARAMETER_SPACE_DIVIDER_H_INCLUDED
#define ELM_PARAMETER_SPACE_DIVIDER_H_INCLUDED

#include <dlib/matrix.h>
#include <dlib/rand.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <optional>
#include <stdexcept>
#include <vector>

namespace elm
{
class ParameterSpaceDivider
{
public:
    using Vec = dlib::matrix<double,0,1>;       // column vector

    /* ------------------------------------------------------------------ */
    /*  Construction                                                      */
    /* ------------------------------------------------------------------ */
    ParameterSpaceDivider(const Vec& low,
                          const Vec& high,
                          std::size_t target_chunks,
                          bool  enable_log_scaling = false,
                          std::optional<unsigned long> seed = std::nullopt)
        : dim_{static_cast<std::size_t>(low.nr())},
          rnd_{}
    {
        if (dim_ == 0)
            throw std::invalid_argument("ParameterSpaceDivider: zero dimension");
        if (high.nr() != static_cast<long>(dim_))
            throw std::invalid_argument("low/high dimension mismatch");
        if (target_chunks == 0)
            throw std::invalid_argument("target_chunks must be > 0");

        /* ---- RNG ----------------------------------------------------- */
        if (seed) rnd_.set_seed(dlib::cast_to_string(*seed));
        else      rnd_.set_seed(dlib::cast_to_string(std::time(nullptr)));

        /* ---- decide per-axis scaling & transformed ranges ------------ */
        use_log_ .resize(dim_);
        min_t_  .resize(dim_);
        width_t_.resize(dim_);
        splits_ .assign(dim_, 1);        // start with 1 chunk along each axis
        range_t_.resize(dim_);

        const double RATIO_THRESHOLD = 1000.0;
        for (std::size_t i = 0; i < dim_; ++i)
        {
            if (high(i) <= low(i))
                throw std::invalid_argument("low(i) must be < high(i)");

            const bool log_axis =
                enable_log_scaling && low(i) > 0.0 &&
                (high(i) / low(i)) >= RATIO_THRESHOLD;

            use_log_[i] = log_axis;

            min_t_[i]   = log_axis ? std::log(low(i)) : low(i);
            const double max_t = log_axis ? std::log(high(i)) : high(i);

            range_t_[i] = max_t - min_t_[i];
        }

        /* ---- greedy assignment of splits[] so that Π splits == target --- */
        std::size_t product = 1;
        while (product < target_chunks)
        {
            /* pick axis with widest current cell */
            std::size_t best_axis = 0;
            double      best_width = current_cell_width(0);

            for (std::size_t i = 1; i < dim_; ++i)
            {
                double w = current_cell_width(i);
                if (w > best_width)
                {
                    best_width = w;
                    best_axis = i;
                }
            }

            ++splits_[best_axis];
            // product *= 1;    // product will be recomputed below for speed
            product = 1;
            for (std::size_t s: splits_) product *= s;

            if (product > target_chunks)
            {
                /* we just exceeded; undo, and break (exact target impossible) */
                --splits_[best_axis];
                break;
            }
        }

        /* If product < target_chunks we still need to reach it exactly.
         * Increment axes round-robin until we hit the target.  This second
         * pass only fires when the greedy loop couldn’t match the exact
         * product due to small remaining factorisation (rare, small).        */
        std::size_t axis = 0;
        while (product < target_chunks)
        {
            ++splits_[axis];
            product = 1;
            for (std::size_t s: splits_) product *= s;
            axis = (axis + 1) % dim_;
        }
        if (product != target_chunks)
            throw std::logic_error("Internal error: could not reach target_chunks");

        num_chunks_ = product;

        /* ---- pre-compute width_t_  ---------------------------------- */
        for (std::size_t i = 0; i < dim_; ++i)
            width_t_[i] = range_t_[i] / static_cast<double>(splits_[i]);

        /* ---- multipliers for row-major linearisation ---------------- */
        mult_.resize(dim_);
        mult_[0] = 1;
        for (std::size_t i = 1; i < dim_; ++i)
            mult_[i] = mult_[i-1] * splits_[i-1];
    }

    /* ------------------------------------------------------------------ */
    /*  Basic queries                                                     */
    /* ------------------------------------------------------------------ */
    std::size_t dimensionality() const noexcept { return dim_; }
    std::size_t num_chunks()      const noexcept { return num_chunks_; }

    /* inclusive-inclusive random chunk */
    std::size_t random_chunk(std::size_t inclusive_min,
                             std::size_t inclusive_max)
    {
        if (inclusive_min > inclusive_max || inclusive_max >= num_chunks_)
            throw std::out_of_range("random_chunk(): invalid range");

        const std::size_t span = inclusive_max - inclusive_min + 1;
        return inclusive_min +
               static_cast<std::size_t>(rnd_.get_random_double() * span);
    }

    /* ------------------------------------------------------------------ */
    /*  Random point in chunk                                             */
    /* ------------------------------------------------------------------ */
    Vec random_point_in_chunk(std::size_t idx)
    {
        auto coord = decompose_index(idx);
        Vec  out(dim_);

        for (std::size_t i = 0; i < dim_; ++i)
        {
            double cell_min_t = min_t_[i] +
                                static_cast<double>(coord[i]) * width_t_[i];
            double v_t = cell_min_t + rnd_.get_random_double() * width_t_[i];

            out(i) = use_log_[i] ? std::exp(v_t) : v_t;
        }
        return out;
    }

    /* ------------------------------------------------------------------ */
    /*  Membership test                                                   */
    /* ------------------------------------------------------------------ */
    bool point_in_chunk(const Vec& p, std::size_t idx) const
    {
        if (static_cast<std::size_t>(p.nr()) != dim_)
            throw std::invalid_argument("point dimensionality mismatch");

        auto coord = decompose_index(idx);

        for (std::size_t i = 0; i < dim_; ++i)
        {
            double pt_t = use_log_[i] ? std::log(p(i)) : p(i);

            double cell_min = min_t_[i] +
                              static_cast<double>(coord[i]) * width_t_[i];
            double cell_max = cell_min + width_t_[i];

            bool last = (coord[i] == splits_[i]-1);
            if ( last )
            {
                if (pt_t < cell_min || pt_t > cell_max + 1e-12)
                    return false;
            }
            else
            {
                if (pt_t < cell_min || pt_t >= cell_max)
                    return false;
            }
        }
        return true;
    }

    /* ------------------------------------------------------------------ */
    /*  Centre-to-centre distance                                         */
    /* ------------------------------------------------------------------ */
    double distance_between_chunks(std::size_t idx1,
                                   std::size_t idx2) const
    {
        Vec c1 = centre_of_chunk(idx1);
        Vec c2 = centre_of_chunk(idx2);

        double sum = 0.0;
        for (std::size_t i = 0; i < dim_; ++i)
        {
            double d = c1(i) - c2(i);
            sum += d*d;
        }
        return std::sqrt(sum);
    }

private:
    /* ------------------------------------------------------------------ */
    /*  Helpers                                                           */
    /* ------------------------------------------------------------------ */
    double current_cell_width(std::size_t axis) const
    {
        return range_t_[axis] / static_cast<double>(splits_[axis]);
    }

    std::vector<std::size_t> decompose_index(std::size_t idx) const
    {
        if (idx >= num_chunks_)
            throw std::out_of_range("chunk index out of range");

        std::vector<std::size_t> coord(dim_);
        for (std::size_t i = dim_; i-- > 0; )
        {
            coord[i] = idx / mult_[i];
            idx      = idx % mult_[i];
        }
        return coord;
    }

    Vec centre_of_chunk(std::size_t idx) const
    {
        auto c = decompose_index(idx);
        Vec centre(dim_);
        for (std::size_t i = 0; i < dim_; ++i)
        {
            double centre_t = min_t_[i] +
                              (static_cast<double>(c[i]) + 0.5) * width_t_[i];
            centre(i) = use_log_[i] ? std::exp(centre_t) : centre_t;
        }
        return centre;
    }

    /* ------------------------------------------------------------------ */
    /*  Data members                                                      */
    /* ------------------------------------------------------------------ */
    const std::size_t dim_;
    std::size_t       num_chunks_;

    /* per-axis data */
    std::vector<bool>   use_log_;
    std::vector<double> min_t_;
    std::vector<double> width_t_;
    std::vector<double> range_t_;        /* transformed total range */
    std::vector<std::size_t> splits_;    /* #chunks along each axis */
    std::vector<std::size_t> mult_;      /* row-major multipliers   */

    /* RNG (not thread-safe by design) */
    dlib::rand rnd_;
};

} // namespace elm

#endif  /* ELM_PARAMETER_SPACE_DIVIDER_H_INCLUDED */


// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef TADAH_DLIB_GLOBAL_FuNCTION_SEARCH_Hh_
#define TADAH_DLIB_GLOBAL_FuNCTION_SEARCH_Hh_

#include <tadah/hpo/gfs_abstract.h>
#include <vector>
#include <dlib/matrix.h>
#include <mutex>
#include <dlib/rand.h>
#include <dlib/global_optimization/upper_bound_function.h>
#include <dlib/global_optimization/global_function_search_abstract.h>
#include <dlib/global_optimization/global_function_search.h>
#include <dlib/test_for_odr_violations.h>

namespace tadah
{

#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>

class HypercubeSeparator {
public:
  // default ctor: creates an “empty” separator (d_=0)
  HypercubeSeparator() = default;

  // main ctor: builds a separator for [lowbound,highbound], seperation S
  HypercubeSeparator(
      const dlib::matrix<double,0,1>& lowbound,
      const dlib::matrix<double,0,1>& highbound,
      int S)
  {
    if(S < 1) 
      throw std::invalid_argument("S must be ≥ 1");
    d_ = int(lowbound.size());
    if(d_ == 0 || int(highbound.size()) != d_)
      throw std::invalid_argument("bad bounds");

    lowbound_  = lowbound;
    highbound_ = highbound;

        // compute midpoint once
    midpoint_ = dlib::matrix<double>(d_, 1);
    for(int i=0;i<d_;++i)
      midpoint_(i) = 0.5*(lowbound(i) + highbound(i));

    // set γ⁻¹ = S  =>  γ = 1/S
    gamma_inv_ = double(S);
    max_S_     = S;

    // precompute inv_widths[i] = 1/(γ*width_i) = S/width_i
    inv_widths_ = dlib::matrix<double>(d_, 1);
    for(int i = 0; i < d_; ++i) {
      double w = highbound(i) - lowbound(i);
      if(w <= 0.0)
        throw std::invalid_argument("bad bounds on dim " + std::to_string(i));
      inv_widths_(i) = gamma_inv_ / w;
    }

    count_ = std::floor(gamma_inv_);
    std::time_t now = std::time(nullptr);
    rnd.set_seed(dlib::cast_to_string(now));
  }

    // default copy/move
  HypercubeSeparator(const HypercubeSeparator&) = default;
  HypercubeSeparator(HypercubeSeparator&&) noexcept = default;
  HypercubeSeparator& operator=(const HypercubeSeparator&) = default;
  HypercubeSeparator& operator=(HypercubeSeparator&&) noexcept = default;

  bool separated(
    const dlib::matrix<double,0,1>& a,
    const dlib::matrix<double,0,1>& b,
  int M) const
{
  check_initialized();
  if (M < 0 || M > max_S_)
    throw std::invalid_argument(
      "separated(): M out of valid range [0," + std::to_string(max_S_)+"]");
  if (int(a.size()) != d_ || int(b.size()) != d_)
    throw std::invalid_argument("separated(): dimension mismatch");

  double max_norm_inf = 0.0;
  for(int i = 0; i < d_; ++i) {
    double diff   = std::abs(a(i) - b(i));
    double normed = diff * inv_widths_(i);
    if (normed > max_norm_inf)
      max_norm_inf = normed;
  }

  if (M == 0) {
    // same‐cube test: max_norm_inf < 1
    return (max_norm_inf < 1.0);
  }
  // separation test: max_norm_inf >= M
  return (max_norm_inf >= double(M+1));
}

  int dimension() const { return d_; }

  int max_separation() const {
    if(d_==0) throw std::logic_error("uninitialized");
    return max_S_;
  }

// draw a random integer in [min_M, max_separation()]
  int random_separation(int min_M ) {
    check_initialized();
    if (min_M < 0 || min_M > max_S_)
      throw std::invalid_argument(
        "random_separation(): min_M out of valid range [0," + std::to_string(max_S_)+"]");

    long long res = rnd.get_integer_in_range(min_M, max_S_);
    return res;
  }

  int random_shell(int M) {
    check_initialized();
    if (M < 0 || 2*M > max_S_)
      throw std::invalid_argument(
        "random_shell(): M out of valid range [0," + std::to_string(max_S_)+"]");

    int res = rnd.get_integer_in_range(M, max_S_/2);

    return res;

  }

  bool in_cube_shell(
    const dlib::matrix<double,0,1>& centre,
    const dlib::matrix<double,0,1>& q,
    int M) const
  {
    check_initialized();
    if(int(centre.size())!=d_ || int(q.size())!=d_)
      throw std::invalid_argument("dimension mismatch");
    if(M < 0 || M > max_S_)
      throw std::invalid_argument("M out of [0," + std::to_string(max_S_)+"]");

    // compute cube‐indices and L∞‐index‐distance
    int max_delta = 0;
    for(int i = 0; i < d_; ++i){
      double c_norm = (centre(i) - lowbound_(i)) * inv_widths_(i);
      double q_norm = (q     (i) - lowbound_(i)) * inv_widths_(i);
      int ci = int(std::floor(c_norm));
      int qi = int(std::floor(q_norm));
      max_delta = std::max(max_delta, std::abs(qi - ci));
    }

    return max_delta == M;
  }

  bool in_cube_shell(
    const dlib::matrix<double,0,1>& q,
    int M) const
  {
    return in_cube_shell(midpoint_, q, M);
  }

  // Sample a uniform random point in the "M‐th shell" around 'centre'
  // i.e. in one of the cubes with L∞ index‐distance exactly M from centre's cube.
dlib::matrix<double,0,1> random_point_in_cube_shell(
    const dlib::matrix<double,0,1>& centre,
    int M) 
  {
    check_initialized();
    if(M < 0 || M > max_S_) 
      throw std::invalid_argument(
        "M out of range [0,"+std::to_string(max_S_)+"]");
    if(int(centre.size())!=d_) 
      throw std::invalid_argument("dimension mismatch");

    // 1) find the centre‐cube indices c_i = floor((centre_i - low_i)*inv_widths_i)
    std::vector<int> cidx(d_);
    for(int i=0;i<d_;++i){
      double norm = (centre(i) - lowbound_(i)) * inv_widths_(i);
      cidx[i] = int(std::floor(norm));
    }

    // 2) rejection‐sample an index‐offset delta in [-M..M]^d
    //    s.t. max_i |delta_i| == M and 0 <= cidx[i]+delta_i < count_.

    std::vector<int> qidx(d_);
    for(;;){
      // draw a candidate delta
      int max_abs = 0;
      for(int i=0;i<d_;++i){
        int di = rnd.get_integer_in_range(-M,M);
        qidx[i] = cidx[i] + di;
        max_abs = std::max(max_abs, std::abs(di));
      }
      if(max_abs != M) 
        continue;              // not on the M‐shell

      // check bounds
      bool ok = true;
      for(int i=0;i<d_;++i){
        if(qidx[i] < 0 || qidx[i] >= int(count_)){
          ok = false; break;
        }
      }
      if(ok) break;           // found a valid cube‐index
    }

    // 3) sample uniformly inside that cube
    dlib::matrix<double,0,1> q(d_);
    for(int i=0;i<d_;++i){
      // cube side‐length = (high-low) / count_ = (high-low) * gamma
      double w = highbound_(i) - lowbound_(i);
      double cube_w = w / double(count_);
      double base   = lowbound_(i) + qidx[i] * cube_w;
      q(i) = base + rnd.get_random_double() * cube_w;
    }
    return q;
  }

  // Overload: use the box‐midpoint as centre
dlib::matrix<double,0,1> random_point_in_cube_shell(int M) {
    return random_point_in_cube_shell(midpoint_, M);
  }

private:
  void check_initialized() const {
    if(d_ == 0) throw std::logic_error("uninitialized separator");
  }
  int d_ = 0;
  double gamma_inv_ = 0;    // = 1/gamma = (n/alpha)^(1/d)
  int    max_S_     = 0;    // = floor(gamma_inv_),max separation between cubes
  double count_     = 0.0;  // ≈ number of cubes per dim = γ_inv_

  dlib::rand rnd;

  dlib::matrix<double,0,1> inv_widths_; // 1 / (γ * (high[i]-low[i]))
  dlib::matrix<double,0,1> lowbound_;
  dlib::matrix<double,0,1> midpoint_;
  dlib::matrix<double,0,1> highbound_;
};

  using namespace dlib;

        struct ofer
        {
            size_t request_id = 0;   // unique id for this eval request
            matrix<double,0,1> x;   // function x to evaluate 
                                    //
            int m_shell = 1; // current M for explore region

            // trust region specific stuff
            bool was_trust_region_generated_request = false;
            bool was_explore_region_generated_request = false;
            // bool run_bob = false; // if true, run BOB on this request
            double predicted_improvement = std::numeric_limits<double>::quiet_NaN();
            double anchor_objective_value = std::numeric_limits<double>::quiet_NaN(); // objective value at center of TR step

            bool operator==(const ofer& item) const { return request_id == item.request_id; }
        };

        struct fi
        {
            fi() = delete;
            fi(const fi&) = delete;
            fi& operator=(const fi&) = delete;

            fi(
                const function_spec& spec,
                size_t function_idx, 
                const std::shared_ptr<std::mutex>& m
            ) : 
                spec(spec), function_idx(function_idx), m(m)
            {
                best_x = zeros_matrix(spec.lower);
            }

            upper_bound_function build_upper_bound_with_all_function_evals (
            ) const;

            static double find_nn (
                const std::vector<function_evaluation>& evals,
                const matrix<double,0,1>& x
            );


            function_spec spec;
            size_t function_idx = 0;
            std::shared_ptr<std::mutex> m;
            upper_bound_function ub;
            std::vector<ofer> outstanding_evals;
            matrix<double,0,1> best_x; 
            std::vector<matrix<double,0,1>> best_xs; 
            std::vector<double> bov; 
            double best_objective_value = -std::numeric_limits<double>::infinity(); 
            double radius = 0;
            std::vector<double> radii; // radii for explore region
            std::vector<size_t> count_to_explore_region_update;
            size_t current_m_shell = 0;
            std::vector<std::vector<function_evaluation>> m_samples;

        };


    class fer
    {
    public:

        fer() = delete;
        fer(const fer&) = delete;
        fer& operator=(const fer&) = delete;


        fer(fer&& item);
        fer& operator=(fer&& item);

        ~fer();

        size_t function_idx (
        ) const;

        const matrix<double,0,1>& x (
        ) const;
        matrix<double,0,1>& x (
        ) ;

        bool has_been_evaluated (
        ) const;

        void set (
            double y
        );

        void swap(fer& item);

    private:

        friend class gfs;

        explicit fer(
            const ofer& req,
            const std::shared_ptr<fi>& info
        ) : req(req), info(info) {}

        bool m_has_been_evaluated = false;
        ofer req;
        std::shared_ptr<fi> info;
    };


    class gfs
    {
    public:

        gfs() = default;

        explicit gfs(
            const function_spec& function
        ); 

        explicit gfs(
            const std::vector<function_spec>& functions_
        );

        gfs(
            const std::vector<function_spec>& functions_,
            const std::vector<std::vector<function_evaluation>>& initial_function_evals,
            const double relative_noise_magnitude = 0.001
        ); 

        gfs(const gfs&) = delete;
        gfs& operator=(const gfs& item) = delete;

        gfs(gfs&& item) = default;
        gfs& operator=(gfs&& item) = default;

        void init_hypercube_separators (
            int sep
        );
        void init_chunks (
            size_t splits
        );

        size_t num_functions(
        ) const;

        void set_seed (
            time_t seed
        );

        void get_function_evaluations (
            std::vector<function_spec>& specs,
            std::vector<std::vector<function_evaluation>>& function_evals
        ) const;

        void get_best_function_eval (
            matrix<double,0,1>& x,
            double& y,
            size_t& function_idx
        ) const;

        fer get_next_x (
        ); 

        double get_pure_random_search_probability (
        ) const; 

        void set_pure_random_search_probability (
            double prob
        );

        double get_solver_epsilon (
        ) const; 

        void set_solver_epsilon (
            double eps
        );

        void set_explore_epsilon (
            double eps
        );

        double get_relative_noise_magnitude (
        ) const; 

        void set_relative_noise_magnitude (
            double value
        );

        size_t get_monte_carlo_upper_bound_sample_num (
        ) const; 

        void set_monte_carlo_upper_bound_sample_num (
            size_t num
        );

        void force_trust_region_steps(size_t M);
        size_t get_force_trust_region_steps() const;
        void force_trust_region_until_converged();

        bool do_trust_region_step = true;
        bool do_trust_region_step2 = true;

        // std::vector<HypercubeSeparator> hypercube_separators;
        std::vector<elm::ParameterSpaceDivider> pchunks;

    private:
  using Cand = std::pair<double, dlib::matrix<double,0,1> const*>;

  // 1) define a comparator type
  struct CandCompare {
    bool operator()(Cand const& a, Cand const& b) const {
      return a.first > b.first; 
    }
  };

  // 2) declare your priority_queue with that comparator
  std::priority_queue<
      Cand,
      std::vector<Cand>,
      CandCompare
    > min_heap{CandCompare()};

        size_t m_counter = 0;

        double run_explore_probability = 0.90;
        double run_explore_probability_lowbound = 0.5;
        double run_explore_probability_highbound = 0.95;

        size_t count_to_radius_reset = 0;
        bool get_new_explore_region = false;

        std::shared_ptr<fi> best_function(
        ) const;

        std::shared_ptr<fi> best_function(
            size_t& idx
        ) const;

        bool has_outstanding_trust_region_request (
        ) const;
        bool has_outstanding_explore_region_request (
        ) const;


        dlib::rand rnd;
        double pure_random_search_probability = 0.02;
        double min_trust_region_epsilon = 0;
        double min_trust_region_epsilon_lowbound = 0;
        double min_trust_region_epsilon_highbound = 0;

        double min_explore_region_epsilon = 0;
        double min_explore_region_epsilon_lowbound = 0;
        double min_explore_region_epsilon_highbound = 0;

        double relative_noise_magnitude = 0.001;
        size_t num_random_samples = 5000;

        size_t next_request_id = 1;

        size_t delayed_trust_region_count = 0;

        std::vector<std::shared_ptr<fi>> functions;
        std::shared_ptr<std::mutex> m;

        size_t forced_trust_region_count = 0;
        bool force_trust_converge    = false;
    };

// ----------------------------------------------------------------------------------------

}

#endif // TADAH_DLIB_GLOBAL_FuNCTION_SEARCH_Hh_
