#include <tadah/hpo/optimizer_spec.h>
#include <iomanip>
#include <nlopt.hpp>
#include <dlib/optimization.h>

using namespace tadah::hpo;

/* One, easily extendable, registry ---------------------------------*/
static void print_tadah_caps(std::ostream& os, const std::string& algo)
{
    if (algo=="RANDOM")
        os<<"   * No parameters.\n";
    else if(algo=="GRID")
        os<<"   * PARAM SPACING <double>   (default 20 steps each dim)\n";
    else if(algo=="ANNEAL")
        os<<"   * Classic Kirkpatrick/Metropolis SA (refs: Kirkpatrick 1983, Metropolis 1953)\n"
            "   * MAXEVAL / MAXTIME / STOPVAL honoured (FTOL_REL/FTOL_ABS not honoured)\n"
            "   * STEP <s>                  uniform sigma override (overrides STEP_FRAC)\n"
            "   * SEED <int>                RNG seed (default: random_device)\n"
            "   * PARAM T_INIT     <double> initial temperature (default: auto-calibrated)\n"
            "   * PARAM T_MIN      <double> cooling floor; stop when T < T_MIN (default 1e-12)\n"
            "   * PARAM RT         <double> geometric cooling factor in (0,1) (default 0.85)\n"
            "   * PARAM L          <int>    Markov-chain length per temperature (default 50)\n"
            "   * PARAM STEP_FRAC  <double> sigma_i = STEP_FRAC*(high_i-low_i) (default 0.10)\n"
            "   * PARAM T0_PROBES  <int>    random samples used to calibrate T0 (default 20)\n"
            "   * PARAM T0_PACCEPT <double> target accept prob for T0 in (0,1) (default 0.80)\n";
    else
        os<<"   * (unknown tadah built-in algo)\n";
}

static void print_dlib_caps(std::ostream& os, const std::string& algo)
{
    if(algo=="GFS")
        os<<"   * SOLVER_EPS   (double, default 1e-7)\n"
            "   * REL_NOISE    (double, default 1e-12)\n";
    else if(algo=="BFGS")
        os<<"   * MAXEVAL      (iteration limit)\n"
            "   * FTOL_ABS     (objective-delta stop, default 1e-7)\n"
            "   * PARAM fd_step <double>      (approx-derivative step, default 1e-6)\n"
            "   * PARAM tol_gradinf <double>  (gradient-inf-norm stop; alternative to FTOL_ABS)\n";
    else if(algo=="BOBYQA")
        os<<"   * RHO_BEGIN / RHO_END\n";
    else
        os<<"   * (use Dlib docs for details)\n";
}

static void print_nlopt_caps(std::ostream& os,const std::string& algo)
{
  nlopt_algorithm a = nlopt_algorithm_from_string(algo.c_str());
    if (a==NLOPT_NUM_ALGORITHMS) { os<<"   * (unknown)\n"; return; }

    os<<"   * any NLopt stopping criterion\n"
        "   * PARAM <name> <value> – any nlopt::opt::set_param key\n";
    if (algo[0]=='G')        os<<"   * global optimizer\n";
    if (algo[1]=='N') os<<"   * derivative-free\n";
}

void OptimizerSpec::print_caps(const std::string& lib,
                               const std::string& algo)
{
    std::cout<<"Options for "<<lib<<":"<<algo<<"\n";
    if      (lib=="TADAH") print_tadah_caps(std::cout,algo);
    else if (lib=="DLIB")  print_dlib_caps (std::cout,algo);
    else if (lib=="NLOPT") print_nlopt_caps(std::cout,algo);
    else std::cout<<"   * (no info)\n";
}
