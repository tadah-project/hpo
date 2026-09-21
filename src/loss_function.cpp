#include <tadah/hpo/loss_function.h>

#include <algorithm>    // std::transform

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
    throw std::runtime_error("LossFunction::from_string: unknown type '"+s+"'");
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

        case Type::MAPE :     return abs_r / (std::abs(y)+eps) * w;

        case Type::HUBER : {
            double delta = params_.empty()? 1.0 : params_[0];
            return abs_r <= delta ? 0.5*r*r*w
                                  : delta*(abs_r-0.5*delta)*w;
        }

        case Type::RMSLE : {
            double ly  = std::log(y     + eps);
            double lyp = std::log(ypred + eps);
            return (ly-lyp)*(ly-lyp) * w;
        }

        case Type::TUKEY : {
            double c = params_.empty()? 4.685 : params_[0];
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
            const double denom = std::abs(y) + eps;
            return -w * sgn(r) / denom;
        }

        case Type::HUBER : {
            const double delta = params_.empty() ? 1.0 : params_[0];
            if (abs_r <= delta) return -w * r;
            return -w * delta * sgn(r);
        }

        case Type::RMSLE : {
            const double ly  = std::log(y     + eps);
            const double lyp = std::log(ypred + eps);
            return -2.0 * w * (ly - lyp) / (ypred + eps);
        }

        case Type::TUKEY : {
            const double c = params_.empty() ? 4.685 : params_[0];
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
        const double r = measured - target;
        const double denom = std::abs(measured) + eps;
        const double sign_r = (r > 0) - (r < 0);
        const double sign_y = (measured > 0) - (measured < 0);
        return w * (sign_r / denom - std::abs(r) * sign_y / (denom * denom));
    }
    if (type_ == Type::RMSLE)
        return 2 * w * (std::log(measured + eps) - std::log(target + eps))
             / (measured + eps);
    return -loss_prime(measured, target, w, eps);
}

bool LossFunction::is_squared() const noexcept
{
    return type_==Type::L2 || type_==Type::RMSLE;
}

}   // namespace tadah
