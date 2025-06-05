#pragma once
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace tadah {

/* --------------------------------------------------------------------- *
 *  LossFunction
 * --------------------------------------------------------------------- */
class LossFunction
{
public:
    /* ----------------------------------------------------------------- *
     *  Supported losses
     * ----------------------------------------------------------------- */
    enum class Type
    {
        L1, L2,
        LOG_COSH,
        MAPE,
        HUBER,
        RMSLE,
        TUKEY,
        SQL2         // squared L2 (identical to L2 for loss, kept for clarity)
    };

    /* ----------------------------------------------------------------- *
     *  Construction
     * ----------------------------------------------------------------- */
    explicit LossFunction(Type t = Type::L2,
                          const std::vector<double>& params = {})
    : type_(t)
    , params_(params.empty()? default_params(t) : params)
    {}

    /* from / to string ------------------------------------------------- */
    static Type                 from_string(const std::string&);
    static std::string          to_string  (Type);
    static std::vector<double>  default_params(Type);

    /* accessors -------------------------------------------------------- */
    Type                         type ()   const noexcept { return type_;   }
    const std::vector<double>&   params()  const noexcept { return params_; }

    /* ----------------------------------------------------------------- *
     *  Evaluate     L(y, ŷ ; w )
     * ----------------------------------------------------------------- */
    double loss(double y, double ypred, double w = 1.0,
                double eps = 1e-12) const;

private:
    Type                type_;
    std::vector<double> params_;

    /* helper (used by LOG_COSH) --------------------------------------- */
    static double stable_log_cosh(double r) noexcept;
};

}   // namespace tadah

