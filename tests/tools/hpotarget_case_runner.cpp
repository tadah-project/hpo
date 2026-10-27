/* hpotarget_case_runner -- runs target files through the parser of
 * `tadah hpo` and reports what it did, for the generated checks of the
 * target-file registry (tests/check_hpotarget_generated.py).
 *
 *   hpotarget_case_runner CASES RESULTS
 *
 * CASES is a TOML file of [[case]] tables:
 *   id          a name for the case
 *   mode        "parse"    HPOTargets reads the file;
 *               "startup"  and validate_hpo_setup checks it;
 *               "check"    hpo_run with CHECK true (`tadah hpo --check`)
 *                          runs on it, which also reads the environment
 *                          overrides of the run;
 *               "defaults" HPOTargets reads the file, and the values of
 *                          the registry entries that have a default are
 *                          reported (`values`, below)
 *   file        the target file
 *   config      the training configuration (the --config file)
 *   validation  optional: the validation configuration ("check" only)
 *   has_train_db, has_validation   optional booleans for validate_hpo_setup
 *   env         optional table of environment variables set for the case
 *   fixtures    optional array of { path, text }: files written first
 *
 * Each case runs in a fresh directory, its working directory while it
 * runs, as in test_hpotarget_registry_conformance.cpp. RESULTS is a JSON
 * array with one object per case: id, parse_error, startup_error,
 * check_error (an empty string when the step passed or did not run) and,
 * for "defaults", values: the registry id of each entry with a default ->
 * the values the code gives it, as text.
 *
 * The values of "defaults": the table `defaults_of` below maps each id to
 * the setting of the parsed HPOTargets or OptimizerSpec, or to the code's
 * own default where no parse sets it (AsaOptions, FdSettings, the Ceres
 * options, the TADAH:GRID grid, the loss parameters). THREADS and
 * GRAD_STRATEGY have no single default in the code: the spec leaves them
 * unset and each use takes value_or(1) and value_or(NUMERICAL); the table
 * repeats those. The cohesive_energy options are those of a
 * default-constructed CohesiveEnergyObservable::Config, which the parser
 * starts from. Exit status: 0, or 2 for a usage or input error. */

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/asa_options.h>
#include <tadah/hpo/backend_rules.h>
#include <tadah/hpo/fd_stencil.h>
#include <tadah/hpo/hpo.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/observables/cohesive_energy.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/startup_validation.h>
#include <tadah/hpo/strategy_rules.h>
#ifdef TADAH_USE_CERES
#include <tadah/hpo/ceres_adapter.h>
#endif

#include <toml.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using tadah::hpo::HPOTargets;
using tadah::hpo::OptimizerSpec;

/* ---- JSON output ------------------------------------------------------ */

std::string json_string(const std::string& s)
{
    std::string out = "\"";
    for (const unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out + "\"";
}

std::string num(double v)
{
    std::ostringstream os;
    os.precision(17);
    os << v;
    return os.str();
}

/* ---- the cases -------------------------------------------------------- */

struct Fixture { std::string path, text; };

struct Case {
    std::string id, mode, file, config, validation;
    bool has_train_db = true, has_validation = true;
    std::map<std::string, std::string> env;
    std::vector<Fixture> fixtures;
};

struct Result {
    std::string parse_error, startup_error, check_error;
    std::map<std::string, std::vector<std::string>> values;
};

std::string str(const toml::value& t, const std::string& k,
                const std::string& dflt = "")
{
    if (!t.contains(k)) return dflt;
    return t.at(k).as_string().str;
}

bool flag(const toml::value& t, const std::string& k, bool dflt)
{
    return t.contains(k) ? t.at(k).as_boolean() : dflt;
}

std::vector<Case> read_cases(const std::string& path)
{
    const toml::value doc = toml::parse(path);
    std::vector<Case> out;
    if (!doc.contains("case")) return out;
    for (const auto& t : doc.at("case").as_array()) {
        Case c;
        c.id = str(t, "id");
        c.mode = str(t, "mode", "parse");
        c.file = str(t, "file");
        c.config = str(t, "config");
        c.validation = str(t, "validation");
        c.has_train_db = flag(t, "has_train_db", true);
        c.has_validation = flag(t, "has_validation", true);
        if (t.contains("env"))
            for (const auto& [k, v] : t.at("env").as_table())
                c.env[k] = v.as_string().str;
        if (t.contains("fixtures"))
            for (const auto& f : t.at("fixtures").as_array())
                c.fixtures.push_back({str(f, "path"), str(f, "text")});
        if (c.mode != "parse" && c.mode != "startup" && c.mode != "check" &&
            c.mode != "defaults")
            throw std::runtime_error("case " + c.id + ": mode '" + c.mode +
                                     "' is not parse, startup, check or "
                                     "defaults");
        out.push_back(std::move(c));
    }
    return out;
}

/* A fresh directory, the working directory while it lives. */
class ScratchCwd {
public:
    ScratchCwd()
    {
        static unsigned long n = 0;
        dir_ = fs::temp_directory_path() /
               ("tadah_hpotarget_cases_" + std::to_string(::getpid()) + "_" +
                std::to_string(n++) + "_" +
                std::to_string(std::chrono::steady_clock::now()
                                   .time_since_epoch().count()));
        fs::create_directories(dir_);
        old_ = fs::current_path();
        fs::current_path(dir_);
    }
    ~ScratchCwd()
    {
        std::error_code ec;
        fs::current_path(old_, ec);
        fs::remove_all(dir_, ec);
    }
    ScratchCwd(const ScratchCwd&) = delete;
    ScratchCwd& operator=(const ScratchCwd&) = delete;

private:
    fs::path dir_, old_;
};

/* The environment variables of a case, restored when it ends. */
class ScopedEnv {
public:
    explicit ScopedEnv(const std::map<std::string, std::string>& env)
    {
        for (const auto& [k, v] : env) {
            const char* old = std::getenv(k.c_str());
            saved_.emplace_back(k, old ? std::optional<std::string>(old)
                                       : std::nullopt);
            ::setenv(k.c_str(), v.c_str(), 1);
        }
    }
    ~ScopedEnv()
    {
        for (const auto& [k, v] : saved_) {
            if (v) ::setenv(k.c_str(), v->c_str(), 1);
            else   ::unsetenv(k.c_str());
        }
    }
    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
    std::vector<std::pair<std::string, std::optional<std::string>>> saved_;
};

void write_file(const fs::path& p, const std::string& text)
{
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    std::ofstream os(p);
    os << text;
}

tadah::core::Context parse_config(const std::string& text)
{
    std::istringstream in(text);
    return tadah::core::CoreUserConfigParser().parse(in);
}

/* ---- the defaults ----------------------------------------------------- */

/* The values the code gives each registry entry with a default, for the
 * targets `trg` read from a file that does not set it: id -> values (more
 * than one where the default depends on the case, as RATIO and FRAC of
 * OPTIM AUTO). */
using Values = std::map<std::string, std::vector<std::string>>;

Values defaults_of(const HPOTargets& trg)
{
    using namespace tadah::hpo;
    Values v;
    const auto add = [&v](const std::string& id, const std::string& value) {
        v[id].push_back(value);
    };
    const OptimizerSpec& sp = trg.spec;

    /* The top-level keys. */
    add("ROOT.FAILSCORE", num(trg.FAIL_SCORE));
    add("ROOT.PC_SCRIPT_TIMEOUT", trg.MAX_TIME);
    add("ROOT.MAXTIME", trg.MAX_TIME);
    add("ROOT.PC_WORKER_TIMEOUT", num(trg.PC_WORKER_TIMEOUT));
    add("ROOT.FAILSCORE_GRADING", trg.FAILSCORE_GRADING ? "ON" : "OFF");
    add("ROOT.BUMPGATE", trg.BUMPGATE_STRICT ? "STRICT" : "CANDIDATE");
    add("ROOT.QFLUSH", std::to_string(trg.QFLUSH_THRESHOLD));
    add("ROOT.OUTPUT", std::to_string(trg.outputEvery));
    add("ROOT.BOUTPUT", std::to_string(trg.bestOutputEvery));
    add("ROOT.DIGITS", std::to_string(trg.outputDigits));
    if (trg.space())
        add("ROOT.OPTIM_SCALE", trg.space()->normalised() ? "AUTO" : "NONE");
    add("ROOT.LOSS", tadah::LossFunction::to_string(trg.globalLoss.type()));

    /* The STRATEGY-level optimiser and its blocks. THREADS and
     * GRAD_STRATEGY: value_or at their uses (the file comment). */
    add("OPTIMIZER.THREADS", std::to_string(sp.num_threads.value_or(1)));
    add("OPTIMIZER.GRAD_STRATEGY",
        grad_strategy_name(sp.grad_strategy.value_or(GradStrategy::Numerical)));
    add("STRATEGY.TYPE", strategy_type_name(sp.strategy_type));
    add("STRATEGY.TOPK", std::to_string(sp.stage_topk));
    add("INIT.WARM_ON_MISMATCH", sp.init_strategy_spec.warm_on_mismatch);
    add("NOISE.AUTO", sp.noise.autodetect ? "true" : "false");
    add("NOISE.CALIBRATION", std::to_string(sp.noise.calibration_n));

    /* The AUTO boxes the file holds: the factor of a geometric box,
     * high/x0, and the fraction of a linear one, (high - x0)/|x0|. */
    if (trg.space())
        for (std::size_t i = 0; i < trg.space()->dim(); ++i) {
            const auto& d = trg.space()->at(i);
            if (!(d.x0 == d.x0) || d.x0 == 0.0) continue;
            std::string f;
            if (d.origin == BoxOrigin::AutoGeometric)
                f = num(d.hi / d.x0);
            else if (d.origin == BoxOrigin::AutoLinear)
                f = num((d.hi - d.x0) / std::abs(d.x0));
            else
                continue;
            add("OPTIM.AUTO.RATIO", f);
            add("OPTIM.AUTO.FRAC", f);
        }

    /* The options of the lines the file holds. */
    if (trg.RHOGATE) {
        add("RHOGATE.NPTS", std::to_string(trg.RHOGATE->npts));
        add("RHOGATE.SEED", trg.RHOGATE->seed_warn ? "WARN" : "ERROR");
    }
    if (trg.RHOBARGATE) {
        add("RHOBARGATE.NPTS", std::to_string(trg.RHOBARGATE->npts));
        add("RHOBARGATE.SEED", trg.RHOBARGATE->seed_warn ? "WARN" : "ERROR");
    }
    for (const auto& ls : trg.LS) {
        if (ls.is_failprobe) add("PC_FAILPROBE.--s0", num(trg.failprobe_s0));
        for (const auto& fv : ls.failvars)
            add("PC_LAMMPS.--failvar", num(fv.scale));
    }
    {
        const CohesiveEnergyObservable::Config c{};
        const std::string p = "PC_OBSERVABLE.cohesive_energy.";
        add(p + "--supercell", std::to_string(c.nx) + " " +
                                   std::to_string(c.ny) + " " +
                                   std::to_string(c.nz));
        add(p + "--loss", tadah::LossFunction::to_string(c.lossfn.type()));
        add(p + "--maxiter", std::to_string(c.relax_maxiter));
        add(p + "--maxeval", std::to_string(c.relax_maxeval));
        add(p + "--max-wall-time", num(c.relax_max_wall_time_s));
        add(p + "--etol", num(c.relax_etol));
        add(p + "--fmax-converged", num(c.eq_fmax_threshold));
        add(p + "--unconverged",
            std::string(c.unconverged_strategy ==
                                CohesiveEnergyObservable::Config::
                                    UnconvergedStrategy::Const
                            ? "const "
                            : "force-scaled ") +
                num(c.unconverged_value));
    }

    /* The PARAM keys, from the code that applies them to a spec without
     * PARAM lines. */
    {
        OptimizerSpec grid;
        grid.lib = "TADAH";
        grid.algo = "GRID";
        add("PARAM.TADAH:GRID.GRID_POINTS",
            std::to_string(validate_grid_spec(grid)));
    }
#ifdef TADAH_USE_CERES
    {
        OptimizerSpec ceres;
        ceres.lib = "CERES";
        ceres.algo = "STEEPEST_DESCENT";
        const auto o = tadah::hpo::detail::make_ceres_options(ceres,
                                          ::ceres::STEEPEST_DESCENT,
                                          ::ceres::ARMIJO);
        add("PARAM.CERES.tol_gradinf", num(o.gradient_tolerance));
        add("PARAM.CERES.max_iter", std::to_string(o.max_num_iterations));
        add("PARAM.CERES.max_ls_iter",
            std::to_string(o.max_num_line_search_step_size_iterations));
        add("OPTIMIZER.FTOL_REL", num(o.function_tolerance));
        add("OPTIMIZER.XTOL_REL", num(o.parameter_tolerance));
    }
#endif
    {
        const FdSettings fd = fd_settings_of(OptimizerSpec{});
        add("PARAM.FD.fd_step", num(fd.fd_step));
        add("PARAM.FD.diff_method",
            fd.scheme == FdScheme::Central ? "0" : "1");
    }
    {
        OptimizerSpec asa;
        asa.lib = "ASA";
        asa.algo = "ASA";
        const AsaOptions o = parse_asa_options(asa, 1);
        const std::string p = "PARAM.ASA:ASA.";
        add(p + "Limit_Acceptances", std::to_string(o.limit_acceptances));
        add(p + "Limit_Generated", std::to_string(o.limit_generated));
        add(p + "Limit_Invalid_Generated_States",
            std::to_string(o.limit_invalid_generated_states));
        add(p + "Accepted_To_Generated_Ratio",
            num(o.accepted_to_generated_ratio));
        add(p + "Cost_Precision", num(o.cost_precision));
        add(p + "Maximum_Cost_Repeat", std::to_string(o.maximum_cost_repeat));
        add(p + "Number_Cost_Samples", std::to_string(o.number_cost_samples));
        add(p + "Temperature_Ratio_Scale", num(o.temperature_ratio_scale));
        add(p + "Cost_Parameter_Scale_Ratio",
            num(o.cost_parameter_scale_ratio));
        add(p + "Temperature_Anneal_Scale", num(o.temperature_anneal_scale));
        add(p + "Sequential_Parameters",
            std::to_string(o.sequential_parameters));
        add(p + "Initial_Parameter_Temperature",
            num(o.initial_parameter_temperature));
        add(p + "Acceptance_Frequency_Modulus",
            std::to_string(o.acceptance_frequency_modulus));
        add(p + "Generated_Frequency_Modulus",
            std::to_string(o.generated_frequency_modulus));
        add(p + "Reanneal_Cost", std::to_string(o.reanneal_cost));
        add(p + "Reanneal_Parameters", std::to_string(o.reanneal_parameters));
        add(p + "Delta_X", num(o.delta_x));
        add(p + "User_Tangents", std::to_string(o.user_tangents));
        add(p + "Curvature_0", std::to_string(o.curvature_0));
        add(p + "User_Quench_Cost_Scale", num(o.user_quench_cost_scale));
        if (!o.user_quench_param_scale.empty())
            add(p + "User_Quench_Param_Scale",
                num(o.user_quench_param_scale.front()));
        if (!o.user_temperature_ratio.empty())
            add(p + "User_Temperature_Ratio",
                num(o.user_temperature_ratio.front()));
        add(p + "Multi_Number", std::to_string(o.multi_number));
        add(p + "Multi_Specify", std::to_string(o.multi_specify));
        if (!o.multi_grid.empty())
            add(p + "Multi_Grid", num(o.multi_grid.front()));
        add(p + "Queue_Size", std::to_string(o.queue_size));
        if (!o.queue_resolution.empty())
            add(p + "Queue_Resolution", num(o.queue_resolution.front()));
        add(p + "Gener_Block", std::to_string(o.gener_block));
        add(p + "Gener_Mov_Avr", std::to_string(o.gener_mov_avr));
    }

    /* The parameters of the loss functions. */
    add("LOSS.HUBER.delta",
        num(tadah::LossFunction::default_params(
                tadah::LossFunction::Type::HUBER).at(0)));
    add("LOSS.TUKEY.c",
        num(tadah::LossFunction::default_params(
                tadah::LossFunction::Type::TUKEY).at(0)));
    return v;
}

/* ---- running a case --------------------------------------------------- */

Result run(const Case& c)
{
    Result r;
    tadah::core::PeriodicTable::initialize();
    ScratchCwd cwd;
    for (const auto& f : c.fixtures) write_file(f.path, f.text);
    write_file("target.hpo", c.file);
    ScopedEnv scoped(c.env);
    if (c.mode == "check") {
        try {
            tadah::core::Context ctx = parse_config(c.config);
            ctx.add("HPOTARGET", "target.hpo");
            if (!c.validation.empty()) {
                write_file("config.val", c.validation);
                ctx.add("VALIDATION", "config.val");
            }
            ctx.add("CHECK", "true");
            std::ostringstream sink;
            std::streambuf* old = std::cout.rdbuf(sink.rdbuf());
            try {
                tadah::hpo::hpo_run(ctx);
            } catch (...) {
                std::cout.rdbuf(old);
                throw;
            }
            std::cout.rdbuf(old);
        } catch (const std::exception& e) {
            r.check_error = e.what();
            if (r.check_error.empty()) r.check_error = "(empty message)";
        }
        return r;
    }
    tadah::core::Context ctx;
    try {
        ctx = parse_config(c.config);
    } catch (const std::exception& e) {
        throw std::runtime_error("case " + c.id + ": the configuration does "
                                 "not parse: " + e.what());
    }
    std::optional<HPOTargets> trg;
    try {
        trg.emplace(ctx, "target.hpo");
    } catch (const std::exception& e) {
        r.parse_error = e.what();
        if (r.parse_error.empty()) r.parse_error = "(empty message)";
        return r;
    }
    if (c.mode == "defaults") {
        r.values = defaults_of(*trg);
        return r;
    }
    if (c.mode != "startup") return r;
    try {
        tadah::hpo::validate_hpo_setup(ctx, *trg, c.has_train_db,
                                       c.has_validation);
    } catch (const std::exception& e) {
        r.startup_error = e.what();
        if (r.startup_error.empty()) r.startup_error = "(empty message)";
    }
    return r;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "usage: hpotarget_case_runner CASES RESULTS\n";
        return 2;
    }
    std::vector<Case> cases;
    try {
        cases = read_cases(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "hpotarget_case_runner: " << argv[1] << ": " << e.what()
                  << "\n";
        return 2;
    }
    std::ostringstream out;
    out << "[\n";
    for (std::size_t i = 0; i < cases.size(); ++i) {
        Result r;
        try {
            r = run(cases[i]);
        } catch (const std::exception& e) {
            std::cerr << "hpotarget_case_runner: " << e.what() << "\n";
            return 2;
        }
        out << "{\"id\": " << json_string(cases[i].id)
            << ", \"parse_error\": " << json_string(r.parse_error)
            << ", \"startup_error\": " << json_string(r.startup_error)
            << ", \"check_error\": " << json_string(r.check_error)
            << ", \"values\": {";
        bool first = true;
        for (const auto& [k, vs] : r.values) {
            out << (first ? "" : ", ") << json_string(k) << ": [";
            for (std::size_t j = 0; j < vs.size(); ++j)
                out << (j ? ", " : "") << json_string(vs[j]);
            out << "]";
            first = false;
        }
        out << "}}" << (i + 1 < cases.size() ? ",\n" : "\n");
    }
    out << "]\n";
    std::ofstream os(argv[2]);
    os << out.str();
    return os ? 0 : 2;
}
