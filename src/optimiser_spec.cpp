#include <tadah/hpo/optimiser_spec.h>
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
    else
        os<<"   * (unknown tadah built-in algo)\n";
}

static void print_dlib_caps(std::ostream& os, const std::string& algo)
{
    if(algo=="GFS")
        os<<"   * SOLVER_EPS   (double, default 1e-7)\n"
            "   * REL_NOISE    (double, default 1e-12)\n";
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
    if (algo[0]=='G')        os<<"   * global optimiser\n";
    if (algo[1]=='N') os<<"   * derivative-free\n";
}

void OptimiserSpec::print_caps(const std::string& lib,
                               const std::string& algo)
{
    std::cout<<"Options for "<<lib<<":"<<algo<<"\n";
    if      (lib=="TADAH") print_tadah_caps(std::cout,algo);
    else if (lib=="DLIB")  print_dlib_caps (std::cout,algo);
    else if (lib=="NLOPT") print_nlopt_caps(std::cout,algo);
    else std::cout<<"   * (no info)\n";
}
