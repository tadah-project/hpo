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
        os<<"   * Classic (Kirkpatrick/Metropolis 1983/1953) by default;\n"
            "     opt-in Corana adaptive variant via PARAM CORANA 1\n"
            "     (Corana et al. 1987; Brommer 2003).\n"
            "   * MAXEVAL / MAXTIME / STOPVAL honoured; FTOL_REL/FTOL_ABS enable the\n"
            "     Corana sec.-4 eps-NEPS plateau stop (end-of-rung value within eps of\n"
            "     the last NEPS rung values AND of the running best)\n"
            "   * STEP <s>                  uniform sigma override (overrides STEP_FRAC)\n"
            "   * SEED <int>                RNG seed (default: random_device)\n"
            "   * PARAM T_HOT      <double> start/hottest temperature (default: auto-\n"
            "                               calibrated from FAILSCORE-robust local probes\n"
            "                               around the seed point; renamed from T_INIT)\n"
            "   * PARAM T_MIN      <double> cooling floor; stop/re-anneal when T < T_MIN\n"
            "                               (default: auto = T0 * 1e-6)\n"
            "   * PARAM RT         <double> geometric cooling factor in (0,1) (default 0.90)\n"
            "   * PARAM L          <int>    Markov-chain length per T (CLASSIC only, default 50)\n"
            "   * PARAM STEP_FRAC  <double> sigma_i = STEP_FRAC*(high_i-low_i) (default 0.10)\n"
            "   * PARAM T0_PROBES  <int>    local probes used to calibrate T0 (default 20)\n"
            "   * PARAM T0_PACCEPT <double> target accept prob for T0 in (0,1) (default 0.05;\n"
            "                               targets the small-dE tail -- on FAILSCORE-cliff\n"
            "                               landscapes 0.8-style targets give a T0 that\n"
            "                               never cools within realistic budgets)\n"
            "   * PARAM NEPS       <int>    plateau window in rungs for the FTOL stop (default 4)\n"
            "   * PARAM CORANA     <0|1>    enable Corana adaptive variant (default 0=CLASSIC)\n"
            "   * PARAM NS         <int>    Corana step cycles per adjust cycle (default 5)\n"
            "   * PARAM NT         <int>    Corana step-adjust cycles per T (default 5)\n"
            "   * PARAM C          <double> Corana step-adjust gain (default 2.0)\n"
            "   * PARAM V_FLOOR_FRAC <double> Corana V[i] floor as fraction of width (1e-3)\n"
            "   * PARAM RESET_TO_BEST <0|1> Brommer reset-to-best after each cool\n"
            "                               (default 1 under CORANA, 0 otherwise)\n"
            "   * PARAM REANNEAL   <0|1>    warm re-anneal from REANNEAL_SCALE^n * T0 when\n"
            "                               T < T_MIN with budget left (default 1)\n"
            "   * PARAM REANNEAL_SCALE <double> re-anneal start-T factor in (0,1) (0.25)\n"
            "   * THREADS <n>               shared fan-out pool width; sizes the pool that\n"
            "                               runs the per-eval pc_lammps scripts + MLIP\n"
            "                               train/predict (default 1; ANNEAL still drives\n"
            "                               ONE chain, only the fan-out is parallel)\n"
            "   * Dead knobs are hard errors: XTOL_*, VECTOR_ARR, GRAD_STRATEGY, unknown\n"
            "     PARAM keys; POPULATION -> use TADAH:ANNEAL_PA\n";
    else if(algo=="ANNEAL_PT")
        os<<"   * Parallel tempering (replica exchange) over the ANNEAL stepper\n"
            "     (classic by default, PARAM CORANA 1 for the adaptive variant).\n"
            "     K replicas at fixed temperatures T_COLD..T_HOT (slot 0 coldest);\n"
            "     adjacent pairs attempt a state swap every SWAP_EVERY epochs\n"
            "     (deterministic even/odd pairing). Same SEED => same result for\n"
            "     any THREADS.\n"
            "   * THREADS <n>               shared worker-pool width: replica evals + pc_lammps\n"
            "                               fan-out + MLIP train/predict (default 1)\n"
            "   * SEED / STEP / MAXEVAL / MAXTIME / STOPVAL as in ANNEAL;\n"
            "     FTOL_REL/FTOL_ABS arm the plateau stop on the coldest slot\n"
            "   * PARAM REPLICAS    <int>   ladder size K (default max(THREADS,2))\n"
            "   * PARAM T_HOT       <double> hottest slot (default: auto-calibrated)\n"
            "   * PARAM T_COLD      <double> coldest slot (default T_HOT/LADDER_SPAN)\n"
            "   * PARAM LADDER_SPAN <double> T_HOT/T_COLD ratio (default 1e3)\n"
            "   * PARAM SWAP_EVERY  <int>   epochs between exchange attempts (1)\n"
            "   * PARAM RT          <double> ladder cooling per epoch in (0,1];\n"
            "                               1.0 (default) = static ladder (pure PT);\n"
            "                               <1 = PT->SA hybrid, stops at T_MIN\n"
            "   * PARAM INIT_MODE   <0|1>   0: chain0=seed, rest sigma-jittered (default)\n"
            "                               1: chains>0 uniform in the box\n"
            "   * PARAM RESET_TO_BEST <0|1> snap COLDEST slot to global best each\n"
            "                               epoch (default 0 in PT)\n"
            "   * All ANNEAL stepper PARAMs (CORANA/NS/NT/C/L/STEP_FRAC/...) apply\n"
            "     per replica; V adapts per temperature slot (swaps move states)\n"
            "   * Per-pair swap acceptance is logged -- tune LADDER_SPAN/REPLICAS\n"
            "     for 20-40% per pair\n";
    else if(algo=="ANNEAL_PA")
        os<<"   * Population annealing over the ANNEAL stepper (classic by\n"
            "     default, PARAM CORANA 1 for the adaptive variant). R walkers\n"
            "     anneal from T_HOT to T_MIN; each temperature step reweights by\n"
            "     exp(-dbeta*f), resamples (systematic; FAILSCORE walkers die),\n"
            "     then every walker runs SWEEPS rungs at the new T. Same SEED =>\n"
            "     same result for any THREADS.\n"
            "   * THREADS <n>               shared worker-pool width: walker evals + pc_lammps\n"
            "                               fan-out + MLIP train/predict (default 1)\n"
            "   * POPULATION <R>            population size (default 4*THREADS, min 16)\n"
            "   * SEED / STEP / MAXEVAL / MAXTIME / STOPVAL as in ANNEAL;\n"
            "     FTOL_REL/FTOL_ABS arm the plateau stop on the population best\n"
            "   * PARAM SWEEPS      <int>   rungs per walker per T-step (default 1)\n"
            "   * PARAM ESS_TARGET  <double> adaptive-dbeta target in [0,1);\n"
            "                               ESS/N_finite held at target (default 0.5);\n"
            "                               0 = pure geometric RT schedule\n"
            "   * PARAM T_HOT / T_MIN / RT  as in ANNEAL (RT also caps the adaptive\n"
            "                               per-step cooling rate)\n"
            "   * PARAM INIT_MODE   <0|1>   0: walker0=seed, rest sigma-jittered\n"
            "                               (default); 1: walkers>0 uniform in box\n"
            "   * All ANNEAL stepper PARAMs apply per walker; V is walker state\n"
            "     (children inherit the parent's step calibration on resample)\n"
            "   * Per-step journal line: T, ESS, unique parents, culled, finite\n"
            "     fraction, population best\n";
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
        "   * PARAM <name> <value> - any nlopt::opt::set_param key\n";
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
