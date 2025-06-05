#include <tadah/hpo/loss_function.h>

#include <algorithm>    // std::transform

namespace tadah {

/* ----------  mapping strings ↔ enum  ---------------------------------- */
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
    if(u=="SQL2")      return Type::SQL2;
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
      case Type::SQL2:      return "SQL2";
    }
    return "<invalid>";
}

/* ----------  default parameter lists --------------------------------- */
std::vector<double> LossFunction::default_params(Type t)
{
    switch(t){
      case Type::HUBER : return {50.0};      // δ
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
    double r     = y - ypred;
    double abs_r = std::abs(r);

    switch(type_)
    {
        case Type::L1 :       return abs_r * w;
        case Type::L2 :       return abs_r*abs_r * w;
        case Type::SQL2:      return abs_r*abs_r * w;

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
                return (c*c/6.0)*(1-std::pow(1-t*t,3))*w;
            }
            return (c*c/6.0)*w;
        }
    }
    return 0.0;     // should not get here
}

}   // namespace tadah

