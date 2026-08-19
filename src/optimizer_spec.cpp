#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/asa_options.h>
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
            "   * STEP <s>                  explicit proposal scale (mutually exclusive with\n"
            "                               PARAM STEP_FRAC): CLASSIC Gaussian sigma;\n"
            "                               CORANA requested initial U(-V,V) half-width V\n"
            "                               (floor/cap-clamped); PT/PA INIT_MODE 0 reuses the\n"
            "                               resulting per-dimension scale as Gaussian sigma\n"
            "   * SEED <int>                RNG seed (default: random_device)\n"
            "   * PARAM T_HOT      <double> start/hottest temperature (default: auto-\n"
            "                               calibrated from FAILSCORE-robust local probes\n"
            "                               around the seed point; renamed from T_INIT)\n"
            "   * PARAM T_MIN      <double> cooling floor; stop/re-anneal when T < T_MIN\n"
            "                               (default: auto = T0 * 1e-6)\n"
            "   * PARAM RT         <double> geometric cooling factor in (0,1) (default 0.90)\n"
            "   * PARAM L          <int>    Markov-chain length per T (CLASSIC only, default 50)\n"
            "   * PARAM STEP_FRAC  <double> scale_i/width_i (default 0.10): Gaussian sigma\n"
            "                               for CLASSIC; initial half-width V for CORANA\n"
            "   * PARAM BOUNDARY_MODE <0|1> 0: exact reflection (default); 1: redraw the\n"
            "                               changed coordinate until it is in bounds; also\n"
            "                               governs PT/PA Gaussian initialization jitter.\n"
            "                               Mode 1 is NOT a symmetric proposal and no\n"
            "                               Hastings correction is applied, so sampling\n"
            "                               near an OPTIM bound is biased away from the\n"
            "                               edge; the reported optimum is unaffected.\n"
            "                               Mode 0 is an exact symmetric fold -- prefer it\n"
            "                               unless you have a reason not to.\n"
            "   * PARAM T0_PROBES  <int>    local probes used to calibrate T0 (default 20);\n"
            "                               CLASSIC uses full-vector Gaussian probes;\n"
            "                               CORANA uses one-coordinate uniform probes\n"
            "   * PARAM T0_PACCEPT <double> acceptance probability assigned to the median\n"
            "                               uphill probe delta (default 0.05; tune per workload);\n"
            "                               if none is uphill, median absolute delta is used\n"
            "     T_HOT is mutually exclusive with T0_PROBES/T0_PACCEPT.\n"
            "   * PARAM NEPS       <int>    plateau window in rungs for the FTOL stop (default 4)\n"
            "   * PARAM CORANA     <0|1>    enable Corana adaptive variant (default 0=CLASSIC)\n"
            "   * PARAM NS         <int>    Corana-only step cycles per adjust cycle (default 5)\n"
            "   * PARAM NT         <int>    Corana-only adjust cycles per T (default 5)\n"
            "   * PARAM C          <double> Corana-only step-adjust gain (default 2.0)\n"
            "   * PARAM V_FLOOR_FRAC <double> Corana-only V[i] floor fraction in [0,1]\n"
            "                               (default 1e-3; 0 disables the floor)\n"
            "   * PARAM RESET_TO_BEST <0|1> snap the chain to its best after each cool\n"
            "                               (default 1 under CORANA, 0 under CLASSIC)\n"
            "   * PARAM REANNEAL   <0|1>    warm re-anneal from REANNEAL_SCALE^n * T0 when\n"
            "                               T < T_MIN with budget left (default 1)\n"
            "   * PARAM REANNEAL_SCALE <double> re-anneal start-T factor in (0,1) (0.25);\n"
            "                               next start must be >= T_MIN/RT (two rungs)\n"
            "     REANNEAL_SCALE is invalid when REANNEAL=0.\n"
            "   * THREADS <n>               shared fan-out pool width; sizes the pool that\n"
            "                               runs the per-eval pc_lammps scripts + MLIP\n"
            "                               train/predict (default 1; ANNEAL still drives\n"
            "                               ONE chain, only the fan-out is parallel)\n"
            "   * Dead knobs are hard errors: XTOL_*, VECTOR_ARR, GRAD_STRATEGY, unknown\n"
            "     PARAM keys; POPULATION -> use TADAH:ANNEAL_PA\n";
    else if(algo=="ANNEAL_PT")
        os<<"   * Parallel tempering (replica exchange) over the ANNEAL stepper\n"
            "     (classic by default, PARAM CORANA 1 for the adaptive variant).\n"
            "     K replicas on a power-shaped log T_COLD..T_HOT ladder (slot 0 coldest);\n"
            "     LADDER_POWER=1 is geometric; optional adaptation preserves endpoints;\n"
            "     the ladder is static by default and cools as a whole when RT < 1;\n"
            "     adjacent pairs attempt a state swap every SWAP_EVERY epochs\n"
            "     (deterministic even/odd pairing). With REPLICAS fixed, a deterministic\n"
            "     objective, and no timing-dependent stop, the same SEED gives the same\n"
            "     evaluated set for any THREADS.\n"
            "   * THREADS <n>               shared worker-pool width: replica evals + pc_lammps\n"
            "                               fan-out + MLIP train/predict (default 1)\n"
            "   * SEED / STEP / MAXEVAL / MAXTIME / STOPVAL as in ANNEAL;\n"
            "     FTOL_REL/FTOL_ABS arm the plateau stop on the coldest slot\n"
            "   * PARAM REPLICAS    <int>   ladder size K (default max(THREADS,2));\n"
            "                               under MPI, must be >= rank count\n"
            "   * PARAM T_HOT       <double> hottest slot (default: auto-calibrated)\n"
            "   * PARAM T_COLD      <double> coldest slot (default T_HOT/LADDER_SPAN)\n"
            "   * PARAM LADDER_SPAN <double> T_HOT/T_COLD ratio (default 1e3)\n"
            "     T_COLD and LADDER_SPAN are mutually exclusive; T_MIN is invalid for RT=1.\n"
            "   * PARAM SWAP_EVERY  <int>   epochs between exchange attempts (1)\n"
            "   * PARAM SWAP_WINDOW <int>   epochs per windowed per-pair acceptance table\n"
            "                               (default 20; 0 = cumulative table only)\n"
            "   * PARAM LADDER_POWER <double> initial normalized log-ladder power (>0; 1.0);\n"
            "                               <1 places slots toward T_HOT, >1 toward T_COLD\n"
            "   * PARAM LADDER_ADAPT <0|1>  adapt interior temperatures (default 0)\n"
            "   * PARAM LADDER_ADAPT_EVERY <int> attempts required at every pair/update (20)\n"
            "   * PARAM LADDER_ADAPT_GAIN <double> initial stochastic gain in (0,1] (1.0)\n"
            "   * PARAM LADDER_ADAPT_DECAY <double> diminishing exponent in (0.5,1] (0.6)\n"
            "   * PARAM LADDER_ADAPT_UPDATES <int> maximum updates; 0 = ongoing (default)\n"
            "     Fixed endpoints/REPLICAS control absolute acceptance; adaptation equalizes\n"
            "     pair rates. Adapt controls are invalid when LADDER_ADAPT=0; LADDER_POWER is\n"
            "     independent. Updates go to hpo_temperature_ladder.tsv. STEP_ADAPT_EVERY=L\n"
            "     is recommended while temperatures adapt.\n"
            "   * PARAM RT          <double> ladder cooling per epoch in (0,1];\n"
            "                               1.0 (default) = static ladder (pure PT);\n"
            "                               <1 = PT->SA hybrid, stops at T_MIN\n"
            "   * PARAM STEP_AUTO   <0|1>   CLASSIC automatic diagonal calibration (default 1);\n"
            "                               0 restores fixed STEP_FRAC behavior; explicit STEP\n"
            "                               disables calibration\n"
            "   * PARAM STEP_AUTO_SWEEPS <int> minimum shuffled coordinate sweeps (default 12;\n"
            "                               unconverged coordinates continue to at most 2x)\n"
            "   * PARAM STEP_ADAPT_EVERY <int> production scalar+diagonal RAM apply cadence per slot;\n"
            "                               0 freezes the calibrated kernel (default), L is a\n"
            "                               recommended continuing-adaptation cadence\n"
            "   * PARAM INIT_MODE   <0|1>   0: automatic CLASSIC starts all slots at the valid\n"
            "                               warm seed; fixed CLASSIC/Corana jitter higher slots\n"
            "                               1: chains>0 uniform in the box\n"
            "   * PARAM RESET_TO_BEST <0|1> snap COLDEST slot to global best each\n"
            "                               epoch (default 0 in PT)\n"
            "   * STEP_FRAC/BOUNDARY_MODE and T0 controls are shared with ANNEAL;\n"
            "     L is CLASSIC-only, while NS/NT/C/V_FLOOR_FRAC are CORANA-only.\n"
            "     V adapts per temperature slot (swaps move states)\n"
            "   * Automatic warm-up uses fixed 4-proposal coordinate batches, then\n"
            "     100-200 full-vector pilots per slot; all count against MAXEVAL/MAXTIME\n"
            "   * Learned calibration scales/statuses are written once to\n"
            "     hpo_step_scales.tsv; adaptive production scales are logged per SWAP_WINDOW\n"
            "   * Per-pair swap acceptance is logged; 20-40% is a literature-derived\n"
            "     starting heuristic, not a workload-independent target\n";
    else if(algo=="ANNEAL_PA")
        os<<"   * Population annealing over the ANNEAL stepper (classic by\n"
            "     default, PARAM CORANA 1 for the adaptive variant). R walkers\n"
            "     anneal from T_HOT to T_MIN; each temperature step reweights by\n"
            "     exp(-dbeta*f), resamples (systematic; FAILSCORE walkers receive zero\n"
            "     weight when at least one feasible walker exists; all-fail skips resampling\n"
            "     and continues geometric cooling), then every walker runs\n"
            "     SWEEPS rungs at the new T. With POPULATION fixed, a deterministic\n"
            "     objective, and no timing-dependent stop, the same SEED gives the same\n"
            "     evaluated set for any THREADS.\n"
            "   * THREADS <n>               shared worker-pool width: walker evals + pc_lammps\n"
            "                               fan-out + MLIP train/predict (default 1)\n"
            "   * POPULATION <R>            population size (default 4*THREADS, min 16);\n"
            "                               under MPI, must be >= rank count\n"
            "   * SEED / STEP / MAXEVAL / MAXTIME / STOPVAL as in ANNEAL;\n"
            "     FTOL_REL/FTOL_ABS arm the plateau stop on the population best\n"
            "   * PARAM SWEEPS      <int>   rungs per walker per T-step (default 1)\n"
            "   * PARAM ESS_TARGET  <double> adaptive-dbeta target in [0,1);\n"
            "                               ESS/N_feasible held at target (default 0.5);\n"
            "                               0 = pure geometric RT schedule\n"
            "   * PARAM T_HOT / T_MIN / RT  as in ANNEAL (RT also caps the adaptive\n"
            "                               per-step cooling rate)\n"
            "   * PARAM INIT_MODE   <0|1>   0: walker0=seed; others use Gaussian jitter with\n"
            "                               sigma=CLASSIC scale or initial CORANA V (default)\n"
            "                               1: walkers>0 uniform in box\n"
            "   * STEP_FRAC/BOUNDARY_MODE and T0 controls are shared with ANNEAL;\n"
            "     L is CLASSIC-only, NS/NT/C/V_FLOOR_FRAC are CORANA-only, and\n"
            "     RESET_TO_BEST is not defined for ANNEAL_PA. V is walker state\n"
            "     (children inherit the parent's step calibration on resample)\n"
            "   * Per-step journal line: T, ESS, unique parents, culled, finite\n"
            "     field (feasible fraction), population best\n";
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

static void print_asa_caps(std::ostream& os, const std::string& algo)
{
    if (algo != "ASA") { os << "   * (the ASA library has one algorithm: ALGO ASA)\n"; return; }
    os << "   * Adaptive Simulated Annealing (Lester Ingber, ASA-30.53,\n"
          "     vendored under external/asa, BSD-3-Clause)\n"
          "   * single-instance: one ASA run per process at a time\n"
          "   * stops: MAXEVAL / MAXTIME / STOPVAL plus ASA's own\n"
          "     Limit_* and plateau options below; FTOL/XTOL rejected\n"
          "   * SEED seeds the proposal stream; INIT seed auto-sets\n"
          "     User_Initial_Parameters\n"
          "   * PARAM keys use the original ASA option spellings\n"
          "     (case-sensitive); per-dim options accept Name[i]\n";
    print_asa_param_table(os);
}

void OptimizerSpec::print_caps(const std::string& lib,
                               const std::string& algo)
{
    std::cout<<"Options for "<<lib<<":"<<algo<<"\n";
    if      (lib=="TADAH") print_tadah_caps(std::cout,algo);
    else if (lib=="DLIB")  print_dlib_caps (std::cout,algo);
    else if (lib=="NLOPT") print_nlopt_caps(std::cout,algo);
    else if (lib=="ASA")   print_asa_caps  (std::cout,algo);
    else std::cout<<"   * (no info)\n";
}
