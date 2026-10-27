#include <tadah/hpo/loss_function.h>

#include <algorithm>    // std::transform
#include <sstream>

namespace tadah {

/* ----------  mapping strings <-> enum  ---------------------------------- */
LossFunction::Type LossFunction::from_string(const std::string& s)
{
    std::string u(s);                                      // upper-case copy
    std::transform(u.begin(), u.end(), u.begin(), ::toupper);

    if(u=="L1")        return Type::L1;
    if(u=="L2")        return Type::L2;
    if(u=="LOG_COSH")  return Type::LOG_COSH;
    if(u=="MAPE")      return Type::MAPE;
    if(u=="HUBER")     return Type::HUBER;
    if(u=="RMSLE")     return Type::RMSLE;
    if(u=="TUKEY")     return Type::TUKEY;
    throw std::runtime_error("unknown loss function '" + s + "'; the loss "
                             "functions are L1, L2, LOG_COSH, MAPE, HUBER, "
                             "RMSLE and TUKEY, in any case");
}

LossFunction::Type LossFunction::from_name(const std::string& s,
                                           const std::string& where)
{
    try {
        return from_string(s);
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(where + ": " + e.what());
    }
}

std::string LossFunction::to_string(Type t)
{
    switch(t){
      case Type::L1:        return "L1";
      case Type::L2:        return "L2";
      case Type::LOG_COSH:  return "LOG_COSH";
      case Type::MAPE:      return "MAPE";
      case Type::HUBER:     return "HUBER";
      case Type::RMSLE:     return "RMSLE";
      case Type::TUKEY:     return "TUKEY";
    }
    return "<invalid>";
}

/* ----------  default parameter lists --------------------------------- */
std::vector<double> LossFunction::default_params(Type t)
{
    switch(t){
      case Type::HUBER : return {50.0};      // delta
      case Type::TUKEY : return {4.685};     // c
      default : return {};
    }
}

std::vector<double> LossFunction::resolve_params(Type t,
                                                 const std::vector<double>& params)
{
    if (params.empty()) return default_params(t);
    const std::string name = to_string(t);
    const std::size_t n_expected = default_params(t).size();
    if (n_expected == 0)
        throw std::runtime_error(
            "loss " + name + " takes no parameter (got " +
            std::to_string(params.size()) + ")");
    if (params.size() > n_expected)
        throw std::runtime_error(
            "loss " + name + " takes at most " + std::to_string(n_expected) +
            " parameter (got " + std::to_string(params.size()) + ")");
    for (double p : params) {
        if (!std::isfinite(p) || !(p > 0.0)) {
            std::ostringstream oss;
            oss << "loss " << name << " parameter ("
                << (t == Type::HUBER ? "delta" : "c")
                << ") must be finite and > 0 (got " << p
                << "); omit it to use the default " << default_params(t)[0];
            throw std::runtime_error(oss.str());
        }
    }
    return params;
}

/* ----------  numerically safe log(cosh) ------------------------------ */
double LossFunction::stable_log_cosh(double r) noexcept
{
    constexpr double LOG_2 = 0.693147180559945309417232121458176568;
    double a = std::abs(r);
    return a + std::log1p(std::exp(-2.0*a)) - LOG_2;
}

/* --------------------------------------------------------------------- *
 *  Evaluate
 * --------------------------------------------------------------------- */
double LossFunction::loss(double y, double ypred, double w,
                          double eps) const
{
    if (w == 0.0) return 0.0;
    double r     = y - ypred;
    double abs_r = std::abs(r);

    switch(type_)
    {
        case Type::L1 :       return abs_r * w;
        case Type::L2 :       return abs_r*abs_r * w;

        case Type::LOG_COSH : return stable_log_cosh(r) * w;

        /* Relative to the target (ypred), not the measured value: a
         * target of 0 is refused where it is given (target_refusal). */
        case Type::MAPE :     return abs_r / std::abs(ypred) * w;

        case Type::HUBER : {
            const double delta = params_[0];   // resolve_params() guarantees one
            return abs_r <= delta ? 0.5*r*r*w
                                  : delta*(abs_r-0.5*delta)*w;
        }

        case Type::RMSLE : {
            double ly  = std::log(y     + eps);
            double lyp = std::log(ypred + eps);
            return (ly-lyp)*(ly-lyp) * w;
        }

        case Type::TUKEY : {
            const double c = params_[0];       // resolve_params() guarantees one
            if(abs_r <= c){
                double t = r/c;
                return (1-std::pow(1-t*t,3))*w;
                //return (c*c/6.0)*(1-std::pow(1-t*t,3))*w;
            }
            return w;
            //return (c*c/6.0)*w;
        }
    }
    return 0.0;     // should not get here
}

double LossFunction::loss_prime(double y, double ypred, double w,
                                double eps) const
{
    if (w == 0.0) return 0.0;
    /* dL/d(ypred). With r = y - ypred, dr/d(ypred) = -1, so for losses
     * that depend on r we return -(dL/dr). Subgradient kinks return 0
     * for L1/MAPE; HUBER/TUKEY return the smooth-side limit at the
     * switch which is what FD with non-trivial step sees in expectation. */
    const double r     = y - ypred;
    const double abs_r = std::abs(r);
    auto sgn = [](double v) -> double {
        return (v > 0) ? 1.0 : (v < 0) ? -1.0 : 0.0;
    };

    switch(type_)
    {
        case Type::L1 :
            return -w * sgn(r);

        case Type::L2 :
            return -2.0 * w * r;

        case Type::LOG_COSH :
            return -w * std::tanh(r);

        case Type::MAPE : {
            /* L = w |y - t| / |t|, t = ypred:
             * dL/dt = -w sgn(r) / |t| - w |r| sgn(t) / t^2. */
            const double t = std::abs(ypred);
            const double sign_t = (ypred > 0) - (ypred < 0);
            return -w * sgn(r) / t - w * abs_r * sign_t / (t * t);
        }

        case Type::HUBER : {
            const double delta = params_[0];
            if (abs_r <= delta) return -w * r;
            return -w * delta * sgn(r);
        }

        case Type::RMSLE : {
            const double ly  = std::log(y     + eps);
            const double lyp = std::log(ypred + eps);
            return -2.0 * w * (ly - lyp) / (ypred + eps);
        }

        case Type::TUKEY : {
            const double c = params_[0];
            if (abs_r > c) return 0.0;
            const double t = r / c;
            const double one_minus_t2 = 1.0 - t * t;
            // L = (1 - (1-t^2)^3) * w  (per loss() body)
            // dL/dt = 3 * (1-t^2)^2 * 2t * w = 6t (1-t^2)^2 w
            // dt/dr = 1/c, dr/dypred = -1
            return -w * 6.0 * t * one_minus_t2 * one_minus_t2 / c;
        }
    }
    return 0.0;
}

double LossFunction::loss_prime_observed(double measured, double target,
                                         double w, double eps) const
{
    if (w == 0.0) return 0.0;
    if (type_ == Type::MAPE) {
        /* L = w |m - t| / |t|: dL/dm = w sign(m - t) / |t|. */
        const double r = measured - target;
        const double sign_r = (r > 0) - (r < 0);
        return w * sign_r / std::abs(target);
    }
    if (type_ == Type::RMSLE)
        return 2 * w * (std::log(measured + eps) - std::log(target + eps))
             / (measured + eps);
    return -loss_prime(measured, target, w, eps);
}

std::string LossFunction::target_refusal(double target) const
{
    if (type_ == Type::MAPE && target == 0.0)
        return "MAPE divides by the target, and the target is 0";
    if (type_ == Type::RMSLE && !(target > -1e-12)) {
        std::ostringstream oss;
        oss << "RMSLE takes ln(target + 1e-12), and the target " << target
            << " is at or below -1e-12";
        return oss.str();
    }
    return {};
}

bool LossFunction::is_squared() const noexcept
{
    return type_==Type::L2 || type_==Type::RMSLE;
}

}   // namespace tadah
