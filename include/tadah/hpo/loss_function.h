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
     *  Evaluate     L(y, y ; w )
     * ----------------------------------------------------------------- */
    double loss(double y, double ypred, double w = 1.0,
                double eps = 1e-12) const;

    /* ----------------------------------------------------------------- *
     *  Derivative wrt ypred:  dL/d(ypred)
     *
     *  Used by HPO analytical / hybrid gradients (Phase 2) to chain into
     *  d(metric)/d(theta). Sign convention matches the loss definition:
     *  with r = y - ypred, dL/dypred = (dL/dr) * (-1).
     *
     *  Subgradient at non-differentiable points (r=0 for L1/MAPE,
     *  |r|=delta for HUBER, |r|=c for TUKEY) returns 0 (kink) or the
     *  smooth-side limit (HUBER/TUKEY); finite, FD-consistent.
     * ----------------------------------------------------------------- */
    double loss_prime(double y, double ypred, double w = 1.0,
                      double eps = 1e-12) const;

    /* --- meta information ------------------------------------------- */
    bool is_squared() const noexcept;    // true  -> collect in RMS bucket

private:
    Type                type_;
    std::vector<double> params_;

    /* helper (used by LOG_COSH) --------------------------------------- */
    static double stable_log_cosh(double r) noexcept;
};

}   // namespace tadah

