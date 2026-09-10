#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/asa_options.h>
#include <tadah/hpo/anneal_keys_gen.h>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <vector>
#include <nlopt.hpp>
#include <dlib/optimization.h>

using namespace tadah::hpo;

/* ANNEAL-family key listing, generated from the registry
 * (blueprint/anneal_keys.toml -> anneal_keys_gen.h): names, defaults,
 * enum tokens, grouping, and importance order all come from the same
 * tables Knobs::parse validates against, so this text cannot drift. */
static void print_anneal_family_caps(std::ostream& os, unsigned kind_mask)
{
    namespace reg = tadah::hpo::anneal_registry;
    static const char* const group_order[] = {
        "budget", "temperature", "ladder", "swaps", "steps", "init",
        "stopping"};
    for (const char* group : group_order) {
        std::vector<const reg::AnnealKeyInfo*> keys;
        for (const auto& k : reg::kKeys)
            if ((k.kinds & kind_mask) && std::string_view(group) == k.group)
                keys.push_back(&k);
        if (keys.empty()) continue;
        std::sort(keys.begin(), keys.end(),
                  [](const reg::AnnealKeyInfo* a, const reg::AnnealKeyInfo* b) {
                      return a->rank < b->rank;
                  });
        os << "   * " << group << ":\n";
        for (const auto* k : keys) {
            bool has = false;
            double def = 0.0;
            if (kind_mask == reg::KIND_SERIAL) {
                has = k->has_def_serial; def = k->def_serial;
            } else if (kind_mask == reg::KIND_PT) {
                has = k->has_def_pt; def = k->def_pt;
            } else {
                has = k->has_def_pa; def = k->def_pa;
            }
            std::ostringstream head;
            head << "PARAM " << k->name;
            if (k->type == reg::AnnealKeyType::Enum) {
                head << " <";
                for (std::size_t i = 0; i < k->enum_count; ++i)
                    head << (i ? "|" : "") << k->enum_values[i];
                head << ">";
                if (has)
                    head << " (default "
                         << k->enum_values[static_cast<std::size_t>(def)]
                         << ")";
            } else if (has && k->accepts_auto && def == k->auto_value) {
                /* 1.5.0: print the TOKEN the user has to type. These keys
                 * carry their AUTO sentinel as the numeric default, so the
                 * raw value would advertise "(default -1)" for a key that
                 * rejects -1. */
                head << " (default AUTO)";
            } else if (has && k->accepts_off && def == k->off_value) {
                head << " (default OFF)";
            } else if (has) {
                head << " (default " << def << ")";
            } else {
                head << " (default auto)";
            }
            if (k->type != reg::AnnealKeyType::Enum &&
                (k->accepts_auto || k->accepts_off)) {
                head << " [accepts ";
                if (k->accepts_auto) head << "AUTO";
                if (k->accepts_auto && k->accepts_off) head << "/";
                if (k->accepts_off) head << "OFF";
                head << "]";
            }
            os << "     " << head.str() << "\n"
               << "        " << k->doc << "\n";
        }
    }
}

/* One, easily extendable, registry ---------------------------------*/
static void print_tadah_caps(std::ostream& os, const std::string& algo)
{
    if (algo=="RANDOM")
        os<<"   * No parameters.\n";
    else if(algo=="GRID")
        os<<"   * PARAM GRID_POINTS <int>  (default 20 points per dimension;\n"
            "     spacing = width/(n-1); total cost = n^dim). STEP is rejected.\n";
    else if(algo=="ANNEAL")
    {
        os<<"   * Serial simulated annealing: classic whole-vector Gaussian kernel\n"
            "     (Kirkpatrick et al. 1983; Metropolis et al. 1953) by default;\n"
            "     PARAM CORANA ON selects the per-dimension adaptive kernel\n"
            "     (Corana et al. 1987). Geometric cooling T <- T*RT; automatic T0\n"
            "     calibration unless PARAM T_INIT is set; CHAIN_LENGTH/NT default\n"
            "     AUTO (MAXEVAL budget spread over the cooling schedule).\n"
            "   * Stops: MAXEVAL / MAXTIME / STOPVAL; FTOL_REL/FTOL_ABS (> 0) arm the\n"
            "     Corana eps-NEPS plateau; T < T_MIN ends a schedule (REANNEAL\n"
            "     restarts from the best until the budget runs out).\n"
            "   * SEED <int>  RNG seed (default: random_device). THREADS <n> sizes the\n"
            "     shared per-eval fan-out pool (the chain itself is serial).\n"
            "   * Dead knobs are hard errors: STEP, XTOL_*, VECTOR_ARR, GRAD_STRATEGY,\n"
            "     unknown PARAM keys; POPULATION -> use TADAH:ANNEAL_PA.\n";
        print_anneal_family_caps(os, tadah::hpo::anneal_registry::KIND_SERIAL);
    }
    else if(algo=="ANNEAL_PT")
    {
        os<<"   * Parallel tempering (replica exchange): K replicas on a static\n"
            "     (optionally self-adapting) log temperature ladder, slot 1 coldest.\n"
            "     A replica runs ONE sweep per epoch (CLASSIC: one whole-vector\n"
            "     proposal; CORANA: dim single-coordinate moves), then adjacent pairs\n"
            "     attempt a state swap every SWAP_EVERY epochs with deterministic\n"
            "     even/odd (DEO) pairing. With REPLICAS fixed, a deterministic\n"
            "     objective, and no timing-dependent stop, the same SEED gives the\n"
            "     same evaluated set for any THREADS.\n"
            "   * Stops: MAXEVAL / MAXTIME / STOPVAL; FTOL_REL/FTOL_ABS (> 0) arm the\n"
            "     best-stall stop (global best unimproved over NEPS diagnostics\n"
            "     windows). Cooling schedules belong to serial ANNEAL: RT/T_MIN are\n"
            "     rejected here.\n"
            "   * THREADS <n>  shared worker-pool width (replica evals + pc_lammps\n"
            "     fan-out + MLIP train/predict); also the REPLICAS default.\n"
            "   * Diagnostics: per-pair swap tables + step reports every SWAP_WINDOW\n"
            "     epochs; ladder updates in hpo_temperature_ladder.tsv; calibrated\n"
            "     step scales in hpo_step_scales.tsv. Under CORANA, the\n"
            "     per-dimension acceptance the stepper adapts on is summarised per\n"
            "     slot in the journal and written in full to\n"
            "     hpo_corana_acceptance.tsv (per SLOT/temperature, not per walker).\n";
        print_anneal_family_caps(os, tadah::hpo::anneal_registry::KIND_PT);
    }
    else if(algo=="ANNEAL_PA")
    {
        os<<"   * Population annealing over the ANNEAL stepper (classic by\n"
            "     default, PARAM CORANA ON for the adaptive variant). R walkers\n"
            "     anneal from T_HOT to T_MIN; each temperature step reweights by\n"
            "     exp(-dbeta*f), resamples (systematic; FAILSCORE walkers receive zero\n"
            "     weight when at least one feasible walker exists; all-fail skips\n"
            "     resampling and continues geometric cooling), then every walker runs\n"
            "     SWEEPS rungs at the new T. Same SEED => same evaluated set for any\n"
            "     THREADS.\n"
            "   * POPULATION <R> population size (default 4*THREADS, min 16).\n"
            "   * V is walker state (children inherit the parent's step calibration\n"
            "     on resample). Per-step journal line: T, ESS, unique parents,\n"
            "     culled, finite field (feasible fraction), population best.\n";
        print_anneal_family_caps(os, tadah::hpo::anneal_registry::KIND_PA);
    }
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
    else
        os<<"   * (unknown DLIB algo; registered: GFS, BFGS)\n";
}

static void print_nlopt_caps(std::ostream& os,const std::string& algo)
{
  nlopt_algorithm a = nlopt_algorithm_from_string(algo.c_str());
    if (a==NLOPT_NUM_ALGORITHMS) { os<<"   * (unknown)\n"; return; }

    os<<"   * any NLopt stopping criterion\n"
        "   * PARAM <name> <value> - any nlopt::opt::set_param key\n";
    if (algo[0]=='G')        os<<"   * global optimizer\n";
    if (algo[1]=='N') os<<"   * derivative-free\n";
    /* Initial-step policy from the registry (single source of truth). */
    try {
        const auto& c = optimizer_caps("NLOPT", algo);
        if (step_use_consumes(c.step_use))
            os<<"   * STEP <double>  initial step; consumption: "
              << step_use_name(c.step_use) << "\n"
              << "     " << c.step_note << "\n";
        else
            os<<"   * STEP is rejected: " << c.step_note << "\n";
        if (c.ignores_bounds)
            os<<"   * WARNING: " << c.bounds_note << "\n";
    } catch (const std::exception&) { /* unregistered: nothing to add */ }
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
