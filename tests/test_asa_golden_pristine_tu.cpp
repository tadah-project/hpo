/* Pristine-side runner of the ASA golden-neutrality gate.
 *
 * This TU sees the PRISTINE USER_DEFINES layout (all algorithmic gates
 * off) and calls the symbol-renamed tadah_asaref_asa() from
 * libtadah_asa_pristine. The featured TU (src/algorithms_asa.cpp) sees
 * a different USER_DEFINES layout -- formally an ODR wrinkle, benign
 * here because the type never crosses a TU boundary and the two builds
 * export disjoint symbols. */

#include "asa_golden_common.h"

extern "C" {
#include <asa/tadah_asa_pristine_config.h>
#include <asa/asa_usr_asa.h>
}

#include <cstring>
#include <vector>

extern "C" double tadah_asa_golden_pristine_cost(
    double* x, double* /*lo*/, double* /*hi*/, double* /*tang*/,
    double* /*curv*/, ALLOC_INT* dim, int* /*type*/, int* cost_flag,
    int* /*exit_code*/, USER_DEFINES* /*o*/)
{
    int fl = 1;
    const double f =
        asa_golden::golden_cost(x, static_cast<long>(*dim), &fl);
    *cost_flag = fl ? 1 : 0;
    return f;
}

extern "C" double tadah_asa_golden_pristine_rng(LONG_INT* /*seed*/)
{
    return asa_golden::golden_rng();
}

asa_golden::GoldenResult run_golden_pristine(std::size_t dim,
                                             const double* lo_in,
                                             const double* hi_in,
                                             const double* x0,
                                             long limit_generated)
{
    std::vector<double> x(x0, x0 + dim), lo(lo_in, lo_in + dim),
        hi(hi_in, hi_in + dim);
    std::vector<double> tang(dim, 0.0), curv(1, 0.0); /* Curvature_0=1 */
    std::vector<int> ptype(dim, REAL_TYPE);
    ALLOC_INT D = static_cast<ALLOC_INT>(dim);
    int valid = TRUE, exit_status = 0;
    LONG_INT rand_seed = 696969;
    static char out_name[] = "asa_out.golden_pristine";

    USER_DEFINES o;
    std::memset(&o, 0, sizeof o);
    /* canonical defaults -- must mirror AsaOptions{} scalar defaults */
    o.Limit_Acceptances               = 10000;
    o.Limit_Generated                 = limit_generated;
    o.Limit_Invalid_Generated_States  = 1000;
    o.Accepted_To_Generated_Ratio     = 1.0e-6;
    o.Cost_Precision                  = 1.0e-18;
    o.Maximum_Cost_Repeat             = 5;
    o.Number_Cost_Samples             = 5;
    o.Temperature_Ratio_Scale         = 1.0e-5;
    o.Cost_Parameter_Scale_Ratio      = 1.0;
    o.Temperature_Anneal_Scale        = 100.0;
    o.Include_Integer_Parameters      = FALSE;
    o.User_Initial_Parameters         = TRUE;
    o.Sequential_Parameters           = -1;
    o.Initial_Parameter_Temperature   = 1.0;
    o.Acceptance_Frequency_Modulus    = 100;
    o.Generated_Frequency_Modulus     = 10000;
    o.Reanneal_Cost                   = 1;
    o.Reanneal_Parameters             = TRUE;
    o.Delta_X                         = 0.001;
    o.User_Tangents                   = FALSE;
    o.Curvature_0                     = 1;
    /* asa_test_asa_options rejects a NULL Asa_Data_Ptr */
    static int dummy_user_data = 0;
    o.Asa_Data_Dim_Ptr                = 1;
    o.Asa_Data_Ptr                    = &dummy_user_data;
    o.Asa_Out_File                    = out_name;
    o.Immediate_Exit                  = FALSE;
    o.Asa_Recursive_Level             = 0;

    asa_golden::GoldenResult r;
    /* `asa` is macro-renamed to tadah_asaref_asa by the pristine config */
    r.final_cost =
        asa(&tadah_asa_golden_pristine_cost, &tadah_asa_golden_pristine_rng,
            &rand_seed, x.data(), lo.data(), hi.data(), tang.data(),
            curv.data(), &D, ptype.data(), &valid, &exit_status, &o);
    r.exit_status = exit_status;
    r.n_accepted  = static_cast<long>(o.N_Accepted);
    r.n_generated = static_cast<long>(o.N_Generated);
    r.best_x      = x;
    return r;
}
