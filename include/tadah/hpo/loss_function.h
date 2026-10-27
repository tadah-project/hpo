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
    /* Every construction path (root LOSS, PC_LAMMPS --varloss,
     * PC_OBSERVABLE --loss) goes through resolve_params(), so the
     * parameter defaults and their validation live in one place. */
    explicit LossFunction(Type t = Type::L2,
                          const std::vector<double>& params = {})
    : type_(t)
    , params_(resolve_params(t, params))
    {}

    /* from / to string ------------------------------------------------- */
    static Type                 from_string(const std::string&);
    static std::string          to_string  (Type);
    /* Defaults used when a parameterised loss is given no parameter:
     * HUBER delta = 50, TUKEY c = 4.685. Empty for the other types. */
    static std::vector<double>  default_params(Type);
    /* Returns the parameter list a LossFunction of type t will hold:
     * default_params(t) when `params` is empty, otherwise `params` after
     * validation. HUBER and TUKEY take at most one parameter, which must
     * be finite and > 0 (a zero Huber delta makes the loss identically
     * zero); the other types take none. Throws std::runtime_error. */
    static std::vector<double>  resolve_params(Type t,
                                               const std::vector<double>& params);

    /* Why this loss function cannot measure a value against `target`,
     * "" when it can: MAPE divides by the target, so a target of 0 is
     * refused; RMSLE takes ln(target + 1e-12), so a target at or below
     * -1e-12 is refused. Every site that pairs a target with a loss
     * function asks (a --varloss term, a PC_OBSERVABLE term, the metric
     * lines under LOSS), and refuses the line with the reason. */
    std::string target_refusal(double target) const;

    /* accessors -------------------------------------------------------- */
    Type                         type ()   const noexcept { return type_;   }
    const std::vector<double>&   params()  const noexcept { return params_; }

    /* ----------------------------------------------------------------- *
     *  Evaluate     L(y, ypred; w): y the measured value and ypred the
     *  target (HPO calls loss(measured, target, weight)). MAPE is
     *  w |y - ypred| / |ypred|, relative to the target, which is not 0
     *  (target_refusal).
     * ----------------------------------------------------------------- */
    double loss(double y, double ypred, double w = 1.0,
                double eps = 1e-12) const;

    /* ----------------------------------------------------------------- *
     *  Derivative wrt ypred:  dL/d(ypred)
     *
     *  Used by HPO analytical / hybrid gradients to chain into
     *  d(metric)/d(theta). Sign convention matches the loss definition:
     *  with r = y - ypred, dL/dypred = (dL/dr) * (-1).
     *
     *  Subgradient at non-differentiable points (r=0 for L1/MAPE,
     *  |r|=delta for HUBER, |r|=c for TUKEY) returns 0 (kink) or the
     *  smooth-side limit (HUBER/TUKEY); finite, FD-consistent.
     * ----------------------------------------------------------------- */
    double loss_prime(double y, double ypred, double w = 1.0,
                      double eps = 1e-12) const;

    // Derivative with respect to the FIRST argument of loss(). HPO uses
    // loss(measured, target, weight); MAPE is not symmetric in its inputs.
    double loss_prime_observed(double measured, double target, double w = 1.0,
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
