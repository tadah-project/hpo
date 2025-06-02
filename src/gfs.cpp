
#include <tadah/hpo/gfs.h>
#include <dlib/global_optimization/upper_bound_function.h>
#include <dlib/optimization.h>


namespace tadah
{
          upper_bound_function fi::build_upper_bound_with_all_function_evals (
        ) const
        {
            upper_bound_function tmp(ub);

            // we are going to add the outstanding evals into this and assume the
            // outstanding evals are going to take y values equal to their nearest
            // neighbor complete evals.
            for (auto& eval : outstanding_evals)
            {
                function_evaluation e;
                e.x = eval.x;
                e.y = find_nn(ub.get_points(), eval.x);
                tmp.add(e);
            }

            return tmp;
        }

        double fi::find_nn (
            const std::vector<function_evaluation>& evals,
            const matrix<double,0,1>& x
        )
        {
            double best_y = 0;
            double best_dist = std::numeric_limits<double>::infinity();
            for (auto& v : evals)
            {
                double dist = length_squared(v.x-x);
                if (dist < best_dist)
                {
                    best_dist = dist;
                    best_y = v.y;
                }
            }
            return best_y;
        }

         void fit_quadratic_to_points_mse(
            const matrix<double>& X,
            const matrix<double,0,1>& Y,
            matrix<double>& H,
            matrix<double,0,1>& g,
            double& c
        )
        {
            DLIB_CASSERT(X.size() > 0);
            DLIB_CASSERT(X.nc() == Y.size());
            DLIB_CASSERT(X.nc() >= (X.nr()+1)*(X.nr()+2)/2);

            const long dims = X.nr();
            const long M = X.nc();

            matrix<double> W((X.nr()+1)*(X.nr()+2)/2, M);

            set_subm(W, 0,0, dims, M) = X;
            set_subm(W, dims,0, 1, M) = 1;
            for (long c = 0; c < X.nc(); ++c)
            {
                long wr = dims+1;
                for (long r = 0; r < X.nr(); ++r)
                {
                    for (long r2 = r; r2 < X.nr(); ++r2)
                    {
                        W(wr,c) = X(r,c)*X(r2,c);
                        if (r2 == r)
                            W(wr,c) *= 0.5;
                        ++wr;
                    }
                }
            }

            matrix<double,0,1> z = pinv(trans(W))*Y;

            c = z(dims);
            g = rowm(z, range(0,dims-1));

            H.set_size(dims,dims);

            long wr = dims+1;
            for (long r = 0; r < X.nr(); ++r)
            {
                for (long r2 = r; r2 < X.nr(); ++r2)
                {
                    H(r,r2) = H(r2,r) = z(wr++);
                }
            }
        }

    // ----------------------------------------------------------------------------------------

        void fit_quadratic_to_points(
            const matrix<double>& X,
            const matrix<double,0,1>& Y,
            matrix<double>& H,
            matrix<double,0,1>& g,
            double& c
        )
        /*!
            requires
                - X.size() > 0
                - X.nc() == Y.size()
                - X.nr()+1 <= X.nc()      
            ensures
                - This function finds a quadratic function, Q(x), that interpolates the
                  given set of points.  If there aren't enough points to uniquely define
                  Q(x) then the Q(x) that fits the given points with the minimum Frobenius
                  norm hessian matrix is selected. 
                - To be precise:
                    - Let: Q(x) == 0.5*trans(x)*H*x + trans(x)*g + c
                    - Then this function finds H, g, and c that minimizes the following:
                        sum(squared(H))
                      such that:
                        Q(colm(X,i)) == Y(i),  for all valid i
                    - If there are more points than necessary to constrain Q then the Q
                      that best interpolates the function in the mean squared sense is
                      found.
        !*/
        {
            DLIB_CASSERT(X.size() > 0);
            DLIB_CASSERT(X.nc() == Y.size());
            DLIB_CASSERT(X.nr()+1 <= X.nc());


            if (X.nc() >= (X.nr()+1)*(X.nr()+2)/2)
            {
                fit_quadratic_to_points_mse(X,Y,H,g,c);
                return;
            }


            const long dims = X.nr();
            const long M = X.nc();

            /*
                Our implementation uses the equations 3.9 - 3.12 from the paper:
                The NEWUOA software for unconstrained optimization without derivatives
                By M.J.D. Powell, 40th Workshop on Large Scale Nonlinear Optimization (Erice, Italy, 2004)
            */

            matrix<double> W(M + dims + 1, M + dims + 1);

            set_subm(W, 0, 0, M, M) = 0.5*squared(tmp(trans(X)*X));
            set_subm(W, 0, M, M, 1) = 1;
            set_subm(W, M, 0, 1, M) = 1;
            set_subm(W, M, M, dims+1, dims+1) = 0;
            set_subm(W, 0, M+1, X.nc(), X.nr()) = trans(X);
            set_subm(W, M+1, 0, X.nr(), X.nc()) = X;


            const matrix<double,0,1> r = join_cols(Y, zeros_matrix<double>(dims+1,1));

            //matrix<double,0,1> z = pinv(W)*r;
            lu_decomposition<decltype(W)> lu(W);
            matrix<double,0,1> z = lu.solve(r);
            //if (lu.is_singular()) std::cout << "WARNING, THE W MATRIX IS SINGULAR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;

            matrix<double,0,1> lambda = rowm(z, range(0,M-1));

            c = z(M);
            g = rowm(z, range(M+1,z.size()-1));
            H = X*diagm(lambda)*trans(X);
        }

                struct quad_interp_result
        {
            quad_interp_result() = default;

            template <typename EXP>
            quad_interp_result(
                const matrix_exp<EXP>& best_x,
                double predicted_improvement
            ) : best_x(best_x), predicted_improvement(predicted_improvement)  {}

            matrix<double,0,1> best_x;
            double predicted_improvement = std::numeric_limits<double>::quiet_NaN();
        };

    // ----------------------------------------------------------------------------------------

        quad_interp_result find_max_quadraticly_interpolated_vector (
            const matrix<double,0,1>& anchor,
            const double radius,
            const std::vector<matrix<double,0,1>>& x,
            const std::vector<double>& y,
            const matrix<double,0,1>& lower,
            const matrix<double,0,1>& upper
        )
        {
            DLIB_CASSERT(x.size() == y.size());
            DLIB_CASSERT(x.size() > 0);
            for (size_t i = 0; i < x.size(); ++i)
                DLIB_CASSERT(anchor.size() == x[i].size());

            const long x_size = static_cast<long>(x.size());
            DLIB_CASSERT(anchor.size()+1 <= x_size && x_size <= (anchor.size()+1)*(anchor.size()+2)/2);


            matrix<double> X(anchor.size(), x.size());
            matrix<double,0,1> Y(x.size());
            for (size_t i = 0; i < x.size(); ++i)
            {
                set_colm(X,i) = x[i] - anchor;
                Y(i) = y[i];
            }

            matrix<double> H;
            matrix<double,0,1> g;
            double c;

            fit_quadratic_to_points(X, Y, H, g, c);

            matrix<double,0,1> p;

            solve_trust_region_subproblem_bounded(-H,-g, radius, p,  0.001, 500, lower-anchor, upper-anchor);

            // ensure we never move more than radius from the anchor.  This might happen if the
            // trust region subproblem isn't solved accurately enough.
            if (length(p) >= radius)
                p *= radius/length(p);


            double predicted_improvement = 0.5*trans(p)*H*p + trans(p)*g;
            return quad_interp_result{clamp(anchor+p,lower,upper), predicted_improvement};
        }

          quad_interp_result pick_next_sample_using_trust_region (
            const std::vector<function_evaluation>& samples,
            double& radius,
            const matrix<double,0,1>& lower,
            const matrix<double,0,1>& upper,
            const std::vector<bool>& is_integer_variable,
            HypercubeSeparator& hc,
            int mShell = -1
        )
        {
            DLIB_CASSERT(samples.size() > 0);
            // We don't use the QP to optimize integer variables.  Instead, we just fix them at
            // their best observed value and use the QP to optimize the real variables.  So the
            // number of dimensions, as far as the QP is concerned, is the number of non-integer
            // variables.
            size_t dims = 0;
            for (auto is_int : is_integer_variable)
            {
                if (!is_int)
                    ++dims;
            }

            DLIB_CASSERT(samples.size() >= dims+1);

            // Use enough points to fill out a quadratic model or the max available if we don't
            // have quite enough.
            const long N = std::min(samples.size(), (dims+1)*(dims+2)/2); 


            // first find the best sample;
            double best_val = -1e300;
            matrix<double,0,1> best_x;
            for (auto& v : samples)
            {
              if (v.y > best_val)
              {
                best_val = v.y;
                best_x = v.x;
              }
            }

            double best_val2 = -1e300;
            matrix<double,0,1> best_x2;
            if (mShell>=0) {
              std::cout << "Finding new trust region in the mShell:= " << mShell << std::endl;
              std::cout << "BEST: " << std::setprecision(15) << best_val << " at:\n" << std::setprecision(5) << trans(best_x) << std::endl;
              bool found = false;
              for (auto& v : samples)
              {
                if (v.y > best_val2 && v.y < best_val && hc.in_cube_shell(v.x, mShell))
                {
                  best_val2 = v.y;
                  best_x2 = v.x;
                  found = true;
                }
              }
              if (found) {
                best_x = best_x2;
                std::cout << "r=" << radius << " NEW:  " << std::setprecision(15) << best_val2 << " mShell: " << mShell << " at:\n" << std::setprecision(5) << trans(best_x) << std::endl;
              }
              else {
                std::cout << "No new point found in Mth shell." << std::endl;
                best_x = hc.random_point_in_cube_shell(mShell);
                std::cout << "NO NEW point found, return random point in mShell: " << hc.in_cube_shell(best_x,mShell) << std::endl;
              }
            }

            // if there are only integer variables then there isn't really anything to do.  So just
            // return the best_x and say there is no improvement.
            if (dims == 0)
                return quad_interp_result(best_x, 0);

            matrix<long,0,1> active_dims(dims);
            long j = 0;
            for (size_t i = 0; i < is_integer_variable.size(); ++i)
            {
                if (!is_integer_variable[i])
                    active_dims(j++) = i;
            }

            // now find the N-1 nearest neighbors of best_x
            std::vector<std::pair<double,size_t>> distances;
            for (size_t i = 0; i < samples.size(); ++i)
                distances.emplace_back(length(best_x-samples[i].x), i);
            std::sort(distances.begin(), distances.end());
            distances.resize(N);

            std::vector<matrix<double,0,1>> x;
            std::vector<double> y;
            for (auto& idx : distances)
            {
                x.emplace_back(rowm(samples[idx.second].x, active_dims));
                y.emplace_back(samples[idx.second].y);
            }

            if (radius == 0)
            {
                for (auto& idx : distances) {
                    double radius_tmp = std::max(radius, length(rowm(best_x-samples[idx.second].x, active_dims)) );
                    if (radius_tmp > radius)
                        radius = radius_tmp;
                }
                // Shrink the radius a little so we are always going to be making the sampling of
                // points near the best current point smaller.
                radius *= 0.95;
            }


            auto tmp = find_max_quadraticly_interpolated_vector(rowm(best_x,active_dims), radius, x, y, rowm(lower,active_dims), rowm(upper,active_dims));

            // stick the integer variables back into the solution
            for (long i = 0; i < active_dims.size(); ++i)
                best_x(active_dims(i)) = tmp.best_x(i);

            tmp.best_x = best_x;
            return tmp;
        }

        matrix<double,0,1> make_random_vector(
            dlib::rand& rnd,
            const matrix<double,0,1>& lower,
            const matrix<double,0,1>& upper,
            const std::vector<bool>& is_integer_variable
        )
        {
            matrix<double,0,1> temp(lower.size());
            for (long i = 0; i < temp.size(); ++i)
            {
                temp(i) = rnd.get_double_in_range(lower(i), upper(i));
                if (is_integer_variable[i])
                    temp(i) = std::round(temp(i));
            }
            return temp;
        }

        
        struct max_upper_bound_function 
        {
            max_upper_bound_function() = default;

            template <typename EXP>
            max_upper_bound_function(
                const matrix_exp<EXP>& x,
                double predicted_improvement,
                double upper_bound 
            ) : x(x), predicted_improvement(predicted_improvement), upper_bound(upper_bound)  {}

            matrix<double,0,1> x;
            double predicted_improvement = 0;
            double upper_bound = 0;
        };

    // ------------------------------------------------------------------------------------

        max_upper_bound_function pick_next_sample_as_max_upper_bound (
            dlib::rand& rnd,
            const upper_bound_function& ub,
            const matrix<double,0,1>& lower,
            const matrix<double,0,1>& upper,
            const std::vector<bool>& is_integer_variable,
            const size_t num_random_samples 
        )
        {
            DLIB_CASSERT(ub.num_points() > 0);



            // now do a simple random search to find the maximum upper bound
            double best_ub_so_far = -std::numeric_limits<double>::infinity();
            matrix<double,0,1> vtemp(lower.size()), v;
            for (size_t rounds = 0; rounds < num_random_samples; ++rounds)
            {
                vtemp = make_random_vector(rnd, lower, upper, is_integer_variable);

                double bound = ub(vtemp);
                if (bound > best_ub_so_far)
                {
                    best_ub_so_far = bound;
                    v = vtemp;
                }
            }

            double max_value = -std::numeric_limits<double>::infinity();
            for (auto& v : ub.get_points())
                max_value = std::max(max_value, v.y);

            return max_upper_bound_function(v, best_ub_so_far - max_value, best_ub_so_far);
        }

    fer::fer(
        fer&& item
    )
    {
        m_has_been_evaluated = item.m_has_been_evaluated;
        req = item.req;
        info = item.info;
        item.info.reset();

        item.m_has_been_evaluated = true;
    }

// ----------------------------------------------------------------------------------------
    
    fer& fer::
    operator=(
        fer&& item
    )
    {
        fer(std::move(item)).swap(*this);
        return *this;
    }

// ----------------------------------------------------------------------------------------

    void fer::
    swap(
        fer& item
    )
    {
        std::swap(m_has_been_evaluated, item.m_has_been_evaluated);
        std::swap(req, item.req);
        std::swap(info, item.info);
    }

// ----------------------------------------------------------------------------------------

    size_t fer::
    function_idx (
    ) const
    {
        return info->function_idx;
    }


    const matrix<double,0,1>& fer::
    x (
    ) const
    {
        return req.x;
    }
    matrix<double,0,1>& fer::
    x (
    ) 
    {
        return req.x;
    }

// ----------------------------------------------------------------------------------------

    bool fer::
    has_been_evaluated (
    ) const
    {
        return m_has_been_evaluated;
    }

// ----------------------------------------------------------------------------------------

    fer::
    ~fer()
    {
        if (!m_has_been_evaluated)
        {
            std::lock_guard<std::mutex> lock(*info->m);

            // remove the evaluation request from the outstanding list.
            auto i = std::find(info->outstanding_evals.begin(), info->outstanding_evals.end(), req);
            info->outstanding_evals.erase(i);
        }
    }

// ----------------------------------------------------------------------------------------

    void fer::
    set (
        double y
    )
    {
        DLIB_CASSERT(has_been_evaluated() == false);
        std::lock_guard<std::mutex> lock(*info->m);

        m_has_been_evaluated = true;


        // move the evaluation from outstanding to complete
        auto i = std::find(info->outstanding_evals.begin(), info->outstanding_evals.end(), req);
        //DLIB_CASSERT(i != info->outstanding_evals.end());
        info->outstanding_evals.erase(i);

        info->ub.add(function_evaluation(req.x,y));


        // Now do trust region radius maintenance and keep track of the best objective
        // values and all that.
        if (req.was_trust_region_generated_request)
        {
            // Adjust trust region radius based on how good this evaluation
            // was.
            double measured_improvement = y-req.anchor_objective_value;
            double rho = measured_improvement/std::abs(req.predicted_improvement);
            //std::cout << "rho: "<< rho << std::endl;
            //std::cout << "radius: "<< info->radius << std::endl;
            if (rho < 0.25)
                info->radius *= 0.5;
            else if (rho > 0.75)
                info->radius *= 2;
        }

        if (req.was_explore_region_generated_request)
        {
            // Adjust explore region radius based on how good this evaluation
            // was.
            double measured_improvement = y-req.anchor_objective_value;
            double rho = measured_improvement/std::abs(req.predicted_improvement);
            // std::cout << "rho: "<< rho << std::endl;
            //std::cout << "radius: "<< info->radius << std::endl;
            if (rho < 0.25)
                info->radii[req.current_m_shell] *= 0.5;
            else if (rho > 0.75)
                info->radii[req.current_m_shell] *= 2;
        }

        if (y > info->best_objective_value)
        {
            if (!req.was_trust_region_generated_request && length(req.x - info->best_x) > info->radius*1.001)
            {
                //std::cout << "reset radius because of big move, " << length(req.x - info->best_x) << "  radius was " << info->radius << std::endl;
                // reset trust region radius since we made a big move.  Doing this will
                // cause the radius to be reset to the size of the local region.
                info->radius = 0;
            }
            if (!req.was_explore_region_generated_request && length(req.x - info->best_xs[req.current_m_shell]) > info->radii[req.current_m_shell]*1.001)
            {
                //std::cout << "reset radius because of big move, " << length(req.x - info->best_x) << "  radius was " << info->radius << std::endl;
                // reset trust region radius since we made a big move.  Doing this will
                // cause the radius to be reset to the size of the local region.
                // info->radii[req.current_m_shell] = 0;
                // info->best_xs[req.current_m_shell] = req.x;
            }
            info->best_objective_value = y;
            info->best_x = req.x;

        }

        if (req.was_explore_region_generated_request) {
          if (y > info->bov[req.current_m_shell]) {
            info->bov[req.current_m_shell] = y;
            info->best_xs[req.current_m_shell] = req.x;
          }
        }


    }

    gfs::
    gfs(
        const function_spec& function
    ) : gfs(std::vector<function_spec>(1,function)) {}

// ----------------------------------------------------------------------------------------

    gfs::
    gfs(
        const std::vector<function_spec>& functions_
    )
    {
        DLIB_CASSERT(functions_.size() > 0);
        m = std::make_shared<std::mutex>();
        functions.reserve(functions_.size());
        for (size_t i = 0; i < functions_.size(); ++i)
            functions.emplace_back(std::make_shared<fi>(functions_[i],i,m));
    }

// ----------------------------------------------------------------------------------------

    gfs::
    gfs(
        const std::vector<function_spec>& functions_,
        const std::vector<std::vector<function_evaluation>>& initial_function_evals,
        const double relative_noise_magnitude_
    ) : 
        gfs(functions_) 
    {
        DLIB_CASSERT(functions_.size() > 0);
        DLIB_CASSERT(functions_.size() == initial_function_evals.size());
        DLIB_CASSERT(relative_noise_magnitude_ >= 0);
        relative_noise_magnitude = relative_noise_magnitude_;
        for (size_t i = 0; i < initial_function_evals.size(); ++i)
        {
            functions[i]->ub = upper_bound_function(initial_function_evals[i], relative_noise_magnitude);

            if (initial_function_evals[i].size() != 0)
            {
                auto best = max_scoring_element(initial_function_evals[i], [](const function_evaluation& e) { return e.y; }).first;
                functions[i]->best_objective_value = best.y;
                functions[i]->best_x = best.x;
            }

        }
    }

// ----------------------------------------------------------------------------------------
    void gfs::init_hypercube_separators (
            int sep)
    {
      hypercube_separators.resize(functions.size());
      for (size_t i = 0; i < functions.size(); ++i)
      {
        auto &lower = functions[i]->spec.lower;
        auto &upper = functions[i]->spec.upper;
        hypercube_separators[i] = 
            HypercubeSeparator(lower, upper, sep);

        auto& info = functions[i];
        info->best_xs.resize(sep+2);
        for (size_t j = 0; j < info->best_xs.size(); ++j)
          info->best_xs[j] = zeros_matrix(lower);
        for (size_t j = 0; j < info->radii.size(); ++j)
          info->radii[j] = 0;
        info->bov.resize(sep+2);
        for (size_t j = 0; j < info->bov.size(); ++j)
          info->bov[j] = -std::numeric_limits<double>::infinity();
        info->radii.resize(sep+2);
        for (size_t j = 0; j < info->radii.size(); ++j)
          info->radii[j] = 0;

        info->count_to_explore_region_update.resize(sep+2,0);
      }
    }

    size_t gfs::
    num_functions(
    ) const 
    { 
        return functions.size();
    }

// ----------------------------------------------------------------------------------------

    void gfs::
    set_seed (
        time_t seed
    )
    {
        rnd = dlib::rand(seed);
    }

// ----------------------------------------------------------------------------------------

    void gfs::
    get_function_evaluations (
        std::vector<function_spec>& specs,
        std::vector<std::vector<function_evaluation>>& function_evals
    ) const
    {
        std::lock_guard<std::mutex> lock(*m);
        specs.clear();
        function_evals.clear();
        for (size_t i = 0; i < functions.size(); ++i)
        {
            specs.emplace_back(functions[i]->spec);
            function_evals.emplace_back(functions[i]->ub.get_points());
        }
    }

// ----------------------------------------------------------------------------------------

    void gfs::
    get_best_function_eval (
        matrix<double,0,1>& x,
        double& y,
        size_t& function_idx
    ) const
    {
        DLIB_CASSERT(num_functions() != 0);

        std::lock_guard<std::mutex> lock(*m);

        // find the largest value
        auto& info = *best_function(function_idx);
        y = info.best_objective_value;
        x = info.best_x;
    }

// ----------------------------------------------------------------------------------------

    fer gfs::
    get_next_x (
    ) 
    {
        DLIB_CASSERT(num_functions() != 0);

        using namespace gopt_impl;

        std::lock_guard<std::mutex> lock(*m);

        // the first thing we do is make sure each function has at least max(3,dimensionality of function) evaluations
        for (auto& info : functions)
        {
            const long dims = info->spec.lower.size();
            // If this is the very beginning of the optimization process
            if (info->ub.num_points()+info->outstanding_evals.size() < 1)
            {
                ofer new_req;
                new_req.request_id = next_request_id++;
                // Pick the point right in the center of the bounds to evaluate first since
                // people will commonly center the bound on a location they think is good.
                // So might as well try there first.
                new_req.x = (info->spec.lower + info->spec.upper)/2.0;
                for (long i = 0; i < new_req.x.size(); ++i)
                {
                    if (info->spec.is_integer_variable[i])
                        new_req.x(i) = std::round(new_req.x(i));
                }
                info->outstanding_evals.emplace_back(new_req);
                return fer(new_req,info);
            }
            else if (info->ub.num_points() < std::max<long>(3,dims))
            {
                ofer new_req;
                new_req.request_id = next_request_id++;
                new_req.x = make_random_vector(rnd, info->spec.lower, info->spec.upper, info->spec.is_integer_variable);
                info->outstanding_evals.emplace_back(new_req);
                return fer(new_req,info);
            }

        }

          // if (run_explore_probability < run_explore_probability_highbound)
          //   run_explore_probability *= 1.001;
          auto binfo = best_function();
          double best_objective_value = binfo->best_objective_value;

          // iterate over binfo->bov and find the largest value
        std::cout << "Best objective value: " << best_objective_value << std::endl;
        double best_bov = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < binfo->bov.size(); ++i)
        {
          if (binfo->bov[i] > best_bov)
          {
            best_bov = binfo->bov[i];
          }
        }

        if (best_bov > -std::numeric_limits<double>::infinity() && best_objective_value > best_bov)
        {
        run_explore_probability = 0.1*(1 - best_objective_value/best_bov);
        std::cout << "Best BOV: " << best_bov << " p: " << run_explore_probability <<std::endl;
        }


          if (rnd.get_random_double() < run_explore_probability && !has_outstanding_explore_region_request())
        {

          auto info = best_function();
          size_t &current_m_shell = info->current_m_shell;
          auto &cteru = info->count_to_explore_region_update[current_m_shell];
          double &radius = info->radii[current_m_shell];
          const long dims = info->spec.lower.size();

          // TODO postpone this until we have enough points to do explore region step
          if (info->ub.num_points() > (dims+1))
          {
            cteru++;
            auto tmp = pick_next_sample_using_trust_region(info->ub.get_points(),
                radius, info->spec.lower, info->spec.upper, info->spec.is_integer_variable, hypercube_separators[info->function_idx],current_m_shell);
            // std::cout << "Picking from the explore region. p=" << run_explore_probability << " Pred: "<< tmp.predicted_improvement << " r=" << info->radii[current_m_shell];
            if (tmp.predicted_improvement > min_explore_region_epsilon)
            {
              // std::cout << " OK." << std::endl;
              ofer new_req;
              new_req.request_id = next_request_id++;
              new_req.x = tmp.best_x;
              new_req.was_explore_region_generated_request = true;
              new_req.current_m_shell = current_m_shell;
              new_req.anchor_objective_value = info->bov[current_m_shell];
              new_req.predicted_improvement = tmp.predicted_improvement;
              info->outstanding_evals.emplace_back(new_req);

              if (cteru > 1) {
                // we are doing well lets stay in this region
                cteru--;
              }
              return fer(new_req, info);
            }

            if (cteru > 2) {
              // hiccup, lets try to explore some other region
              radius = 0; // reset previous explore region radius
              cteru = 0; // reset counter
              current_m_shell = hypercube_separators[info->function_idx].random_separation(0);
              current_m_shell /= 2; // shell not M hypercubes
            }
          }
        } 
          else if (do_trust_region_step && !has_outstanding_trust_region_request() 
            && !has_outstanding_explore_region_request())
        {
          // find the currently best performing function, we will do a trust region
          // step on it.
          auto info = best_function();
          const long dims = info->spec.lower.size();
          // if we have enough points to do a trust region step
          if (info->ub.num_points() > dims+1)
          {
            count_to_radius_reset++;
            auto tmp = pick_next_sample_using_trust_region(info->ub.get_points(),
                info->radius, info->spec.lower, info->spec.upper, 
                info->spec.is_integer_variable, hypercube_separators[info->function_idx]);
            // std::cout << " improvement: "<< tmp.predicted_improvement;
            // std::cout << "Picking sample from the trust region." << " Pred: "<< tmp.predicted_improvement << " r=" << info->radius;
            if (tmp.predicted_improvement > min_trust_region_epsilon)
            {
              // std::cout << " OK." << std::endl;
              do_trust_region_step = false;
              ofer new_req;
              new_req.request_id = next_request_id++;
              new_req.x = tmp.best_x;
              new_req.was_trust_region_generated_request = true;
              new_req.anchor_objective_value = info->best_objective_value;
              new_req.predicted_improvement = tmp.predicted_improvement;
              info->outstanding_evals.emplace_back(new_req);
              count_to_radius_reset--;
              return fer(new_req, info);
            }
            if (count_to_radius_reset > 2) 
            {
              // we are not doing well, lets try to reset
              count_to_radius_reset = 0;
              info->radius = 0;
            }
          }
        }

        do_trust_region_step = true;



        if (rnd.get_random_double() >= pure_random_search_probability)
        {
          // std::cout << "Using upper bound function to pick next sample. " << std::endl;
            // pick a point at random to sample according to the upper bound
            double best_upper_bound = -std::numeric_limits<double>::infinity();
            std::shared_ptr<fi> best_funct;
            matrix<double,0,1> next_sample;
            // so figure out if any function has a good upper bound and if so pick the
            // function with the largest upper bound for evaluation.
            for (auto& info : functions)
            {
                auto tmp = pick_next_sample_as_max_upper_bound(rnd,
                    info->build_upper_bound_with_all_function_evals(), info->spec.lower, info->spec.upper,
                    info->spec.is_integer_variable,  num_random_samples);

              // std::cout << " improvement: " << tmp.predicted_improvement;
                if (tmp.predicted_improvement > 0 && tmp.upper_bound > best_upper_bound) 
                {
                  // std::cout << " OK";
                    best_upper_bound = tmp.upper_bound;
                    next_sample = std::move(tmp.x);
                    best_funct = info;
                }
            // std::cout << std::endl;
            }

            // if we found a good function to evaluate then return that. 
            if (best_funct)
            {
                ofer new_req;
                new_req.request_id = next_request_id++;
                new_req.x = std::move(next_sample);
                best_funct->outstanding_evals.emplace_back(new_req);
                return fer(new_req, best_funct);
            }
        }


        // pick entirely at random
        std::cout << "Picking next sample at random." << std::endl;
        size_t function_idx = rnd.get_integer(functions.size());
        auto info = functions[function_idx];
        ofer new_req;
        new_req.request_id = next_request_id++;
        new_req.x = make_random_vector(rnd, info->spec.lower, info->spec.upper, info->spec.is_integer_variable);
        info->outstanding_evals.emplace_back(new_req);
        return fer(new_req, info);

    }

// ----------------------------------------------------------------------------------------

    double gfs::
    get_pure_random_search_probability (
    ) const 
    { 
        return pure_random_search_probability; 
    }

// ----------------------------------------------------------------------------------------

    void gfs::
    set_pure_random_search_probability (
        double prob
    ) 
    {
        DLIB_CASSERT(0 <= prob && prob <= 1);
        pure_random_search_probability = prob;
    }

// ----------------------------------------------------------------------------------------

    double gfs::
    get_solver_epsilon (
    ) const 
    { 
        return min_trust_region_epsilon; 
    }

// ----------------------------------------------------------------------------------------

    void gfs::
    set_solver_epsilon (
        double eps
    )
    {
        DLIB_CASSERT(0 <= eps);

        std::cout << "Setting solver epsilon to " << eps << std::endl;
        min_trust_region_epsilon_lowbound = eps;
        min_trust_region_epsilon = eps;
        min_trust_region_epsilon_highbound = eps*1000;

    }

    void gfs::
    set_explore_epsilon (
        double eps
    )
    {
        DLIB_CASSERT(0 <= eps);

        std::cout << "Setting explore epsilon to " << eps << std::endl;
        min_explore_region_epsilon_lowbound = eps;
        min_explore_region_epsilon = eps;
        min_explore_region_epsilon_highbound = eps*1000;
    }

// ----------------------------------------------------------------------------------------

    double gfs::
    get_relative_noise_magnitude (
    ) const 
    { 
        return relative_noise_magnitude; 
    }

// ----------------------------------------------------------------------------------------

    void gfs::
    set_relative_noise_magnitude (
        double value
    )
    {
        DLIB_CASSERT(0 <= value);
        relative_noise_magnitude = value;
        if (m)
        {
            std::lock_guard<std::mutex> lock(*m);
            // recreate all the upper bound functions with the new relative noise magnitude
            for (auto& f : functions)
                f->ub = upper_bound_function(f->ub.get_points(), relative_noise_magnitude);
        }
    }

// ----------------------------------------------------------------------------------------

    size_t gfs::
    get_monte_carlo_upper_bound_sample_num (
    ) const 
    { 
        return num_random_samples; 
    }

// ----------------------------------------------------------------------------------------

    void gfs::
    set_monte_carlo_upper_bound_sample_num (
        size_t num
    )
    {
        num_random_samples = num;
    }

// ----------------------------------------------------------------------------------------

    std::shared_ptr<fi> gfs::
    best_function(
    ) const
    {
        size_t idx = 0;
        return best_function(idx);
    }

// ----------------------------------------------------------------------------------------

    std::shared_ptr<fi> gfs::
    best_function(
        size_t& idx
    ) const
    {
        auto compare = [](const std::shared_ptr<fi>& a, const std::shared_ptr<fi>& b) 
            { return a->best_objective_value < b->best_objective_value; };

        auto i = std::max_element(functions.begin(), functions.end(), compare);

        idx = std::distance(functions.begin(),i);
        return *i;
    }

// ----------------------------------------------------------------------------------------

    bool gfs::
    has_outstanding_trust_region_request (
    ) const 
    {
        for (auto& f : functions)
        {
            for (auto& i : f->outstanding_evals)
            {
                if (i.was_trust_region_generated_request)
                    return true;
            }
        }
        return false;
    }

    bool gfs::
    has_outstanding_explore_region_request (
    ) const 
    {
        for (auto& f : functions)
        {
            for (auto& i : f->outstanding_evals)
            {
                if (i.was_explore_region_generated_request)
                    return true;
            }
        }
        return false;
    }

// ----------------------------------------------------------------------------------------

    void gfs::force_trust_region_steps(size_t M) 
    { 
      forced_trust_region_count = M; 
    }

    size_t gfs::get_force_trust_region_steps() const 
    { 
      return forced_trust_region_count; 
    }

    void gfs::force_trust_region_until_converged()
    {
      force_trust_converge = true;
    }
}
