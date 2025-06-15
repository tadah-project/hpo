#pragma once
#include <unordered_map>
#include <optional>
#include <vector>
#include <string>
#include <iostream>
#include <cstdint>
#include <memory>
#include <limits>
#include <cmath>

namespace tadah::hpo {

/* ---------------- basic containers -------------------------------- */
struct StopCriteria {
    std::optional<double> stopval, ftol_rel, ftol_abs,
                           xtol_rel, xtol_abs, maxtime;
    std::optional<uint64_t> maxeval;
};

using ParamMap = std::unordered_map<std::string,double>;

/* ---------------- Optimiser description --------------------------- */
struct OptimiserSpec {
    std::string  lib   = "TADAH";   // DLIB | NLOPT | TADAH
    std::string  algo  = "RANDOM";
    StopCriteria stop;
    ParamMap     param;             // algorithm-specific knobs
    std::unique_ptr<OptimiserSpec> local;   // one sub-optimiser

    /* -------- bounds, integer mask, log-scale --------------------- */
    std::vector<double> low, high;          // filled by HPOTargets
    std::vector<bool>   is_int;
    std::vector<bool>   log_scaled;         // same size as low/high
    std::optional<double> log_threshold;    // OFF = std::nullopt
    std::vector<std::vector<double>> optim_init;
    std::map<std::string, Opti> optimap;  // used by HPO_Host

    /* call once after low/high are known */
    void analyse_logscale()
    {
        const double thr = log_threshold.value_or(std::numeric_limits<double>::infinity());
        log_scaled.resize(low.size(),false);
        for (size_t i=0;i<low.size();++i)
        {
            if ( low[i]>0 && high[i]/low[i] >= thr )
            {
                log_scaled[i]=true;
                low [i] = std::log(low [i]);
                high[i] = std::log(high[i]);
            }
        }
    }
    /* Helper to go in/out of transformed space */
    template<typename Vec>
    Vec in(const Vec& x)  const
    {   Vec y=x;  for(size_t i=0;i<y.size();++i) if(log_scaled[i]) y(i)=std::log(y(i)); return y; }
    template<typename Vec>
    Vec out(const Vec& x) const
    {   Vec y=x;  for(size_t i=0;i<y.size();++i) if(log_scaled[i]) y(i)=std::exp(y(i)); return y; }

    /* ------------- capability print-outs -------------------------- */
    static void print_caps(const std::string& lib,
                           const std::string& algo);
};
}   // namespace
