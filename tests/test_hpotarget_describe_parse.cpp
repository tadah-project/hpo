// test_hpotarget_describe_parse.cpp
//
// tadah::hpo::testing::describe_parse() (hpo_targets_testing.h): the
// canonical text of what the parse of a target file set, which the
// comparison of two parsers of the target file and the corpus of target
// files compare. Pins:
//  - each setting a line of the target file gives is in the text, at its
//    path and in its canonical form, and is not in the text of the same
//    file without the line (a top-level key, a line option, an OPTIM box,
//    an optimiser key, a key of each block, the INNER and PRESTAGE specs,
//    the lines each setting is read from);
//  - two constructions of the targets of one file give one text: a seed
//    the constructor draws (an INIT block or an optimiser without SEED),
//    the seed an exploration takes from the STRATEGY-level SEED and the
//    PARAM fd_step NOISE AUTO derives are shown as read() leaves them;
//  - strings are quoted with escapes; the training configuration the
//    targets read is in the text;
//  - a PC_OBSERVABLE kind it has no description for is refused
//    (std::logic_error naming the kind), not left out;
//  - the LAMMPS command line of a PC_LAMMPS line is given without its
//    first argument, the LAMMPS library the build links, so one file gives
//    one text in every build.
// Before the fix: the header hpo_targets_testing.h does not exist and the
// one case of this file fails.

#include "catch2/catch.hpp"

#if __has_include(<tadah/hpo/hpo_targets_testing.h>)

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/hpo_targets_testing.h>
#include <tadah/hpo/observables/observable.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace {

namespace fs = std::filesystem;
using tadah::core::Context;
using tadah::hpo::HPOTargets;
using tadah::hpo::testing::describe_parse;

/* The training configurations of the registry's contexts default and
 * main_ream (blueprint/hpotarget.toml). */
const char* const kDefaultConfig =
    "ATOM H\nWATOM 1.0\nDBFILE dummy.db\nINIT2B true\nINITMB false\n"
    "TYPE2B D2_BP 2 2 H H\nCGRID2B 1.0 2.0\nSGRID2B 0.5 1.0\n"
    "RCTYPE2B Cut_Cos\nRCUT2B 5.0\nVERBOSE 0\nMODEL M_BLR BF_Linear\n"
    "OALGO 4\nLAMBDA 1e-3\n";
const char* const kReamConfig =
    "ATOM Ta\nWATOM 1.0\nDBFILE dummy.db\nINIT2B true\nINITMB true\n"
    "TYPE2B D2_Blip5 4 4 Ta Ta\nCGRID2B 2.5 3.2 3.9 4.5\n"
    "SGRID2B 3.0 3.0 3.0 3.0\nTYPEMB DM_REAM 3 3 3 3 5 5 0 0 Ta Ta\n"
    "CGRIDMB 2.8 3.6 4.4\nSGRIDMB 3.0 3.0 3.0\nAMPGRIDMB 0.5 0.3 0.1\n"
    "RHOFORM SIGNED\nCEMBFUNC -0.2 0.3 0.8 1.3 1.8\n"
    "SEMBFUNC 2.0 2.0 2.0 2.0 2.0\nRCTYPE2B Cut_Dummy\nRCTYPEMB Cut_Dummy\n"
    "RCUT2B 5.5\nRCUTMB 5.5\nVERBOSE 0\nMODEL M_BLR BF_Linear\nOALGO 4\n"
    "LAMBDA 1e-3\n";

/* A fresh working directory with the LAMMPS scripts the PC_LAMMPS and
 * PC_FAILPROBE lines name, removed on exit. */
class Scratch {
public:
    Scratch()
    {
        static unsigned n = 0;
        dir_ = fs::temp_directory_path() /
               ("tadah_describe_parse_" + std::to_string(::getpid()) + "_" +
                std::to_string(n++) + "_" +
                std::to_string(std::chrono::steady_clock::now()
                                   .time_since_epoch().count()));
        fs::create_directories(dir_);
        old_ = fs::current_path();
        fs::current_path(dir_);
        std::ofstream("lines_pc.in") << "variable E equal 1.0\n"
                                        "variable S equal 0.0\n"
                                        "variable V equal 0.0\n";
    }
    ~Scratch()
    {
        std::error_code ec;
        fs::current_path(old_, ec);
        fs::remove_all(dir_, ec);
    }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;

private:
    fs::path dir_, old_;
};

Context config_of(const std::string& text)
{
    std::istringstream in(text);
    return tadah::core::CoreUserConfigParser().parse(in);
}

/* The description of the parse of the target file `file`. */
std::string describe(const std::string& file,
                     const std::string& config = kDefaultConfig)
{
    tadah::core::PeriodicTable::initialize();
    Scratch s;
    std::ofstream("target.hpo") << file;
    Context ctx = config_of(config);
    HPOTargets trg(ctx, "target.hpo");
    return describe_parse(trg);
}

/* The registry's root wrapper with `top` at its slot and `strategy` after
 * the optimiser keys of its STRATEGY block. */
std::string root(const std::string& top, const std::string& strategy = "")
{
    return top + "OPTIM CGRID2B (1) 0.5 1.5\nPC_ERMSE 0 1\nOPTIMIZER\n"
                 "STRATEGY\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 10\n" +
           strategy + "ENDSTRATEGY\nENDOPTIMIZER\n";
}

bool has_line(const std::string& text, const std::string& line)
{
    return ("\n" + text).find("\n" + line + "\n") != std::string::npos;
}

struct Setting {
    const char* what;
    std::string file;
    std::vector<std::string> lines;
    const char* config = kDefaultConfig;
};

/* An observable of a kind describe_parse() has no description for. */
class UnknownObservable final : public tadah::hpo::Observable {
public:
    std::string kind() const override { return "test_kind"; }
    std::string column_tag() const override { return "T"; }
    std::string analytical_gradient_gap() const override { return ""; }
    tadah::hpo::ObservableEvalResult evaluate(
        tadah::mlip::M_Tadah_Base&, tadah::mlip::DC_Base&,
        const tadah::core::Context&) override
    {
        return {};
    }
    double target() const override { return 0.0; }
    double weight() const override { return 1.0; }
    const tadah::LossFunction& lossfn() const override { return loss_; }

private:
    tadah::LossFunction loss_;
};

}  // namespace

TEST_CASE("describe_parse: each setting of a target file is in the "
          "description of its parse", "[hpotarget_describe]")
{
    const std::string pc = "PC_LAMMPS --script lines_pc.in --varloss E -5.0 1.0\n";
    const std::vector<Setting> settings = {
        {"FAILSCORE", root("FAILSCORE 1000\n"), {"FAIL_SCORE = 1000"}},
        {"QFLUSH", root("QFLUSH 7\n"), {"QFLUSH_THRESHOLD = 7"}},
        {"PC_SCRIPT_TIMEOUT and a PC_LAMMPS line",
         root("PC_SCRIPT_TIMEOUT 600\n" + pc),
         {"MAX_TIME = \"600\"", "MAX_TIME_seconds = 600", "LS.size = 1",
          "LS[0].line = 2", "LS[0].script = \"lines_pc.in\"",
          "LS[0].script_lines.size = 4", "LS[0].n_varloss = [\"E\"]",
          "LS[0].t_varloss = [-5]", "LS[0].w_varloss = [1]",
          "LS[0].maxtime = \"600\"",
          "LS[0].maxtime_source = \"PC_SCRIPT_TIMEOUT on line 1\"",
          "LS[0].is_failprobe = false"}},
        {"MAXTIME at the top level", root("MAXTIME 600\n" + pc),
         {"MAXTIME_line = 1",
          "LS[0].maxtime_source = \"MAXTIME on line 1\""}},
        {"the options of a PC_LAMMPS line",
         root("FAILSCORE 1e9\nFAILSCORE_GRADING ON\n"
              "PC_LAMMPS --script lines_pc.in --varloss E -5.0 1.0 HUBER 2 "
              "--outvar S --invar b 1 --maxtime 1:30 --failscore -3 "
              "--hotvar a 1 2 1.5 --failvar V 2 -nocite\n"),
         {"FAILSCORE_GRADING = true", "LS[0].n_outvar = [\"E\", \"S\"]",
          "LS[0].lossFn[0].type = HUBER", "LS[0].lossFn[0].params = [2]",
          "LS[0].maxtime = \"1:30\"", "LS[0].maxtime_seconds = 90",
          "LS[0].maxtime_source = \"its --maxtime\"",
          "LS[0].failscore = -3", "LS[0].hotvars.size = 1",
          "LS[0].hotvars[0].name = \"a\"", "LS[0].hotvars[0].min = 1",
          "LS[0].hotvars[0].max = 2", "LS[0].hotvars[0].initial = 1.5",
          "LS[0].failvars[0].name = \"V\"", "LS[0].failvars[0].scale = 2",
          "LS[0].failvars[0].scale_given = true",
          "LS[0].reads_default_failscore = false",
          "LS[0].reads_default_maxtime = false"}},
        {"PC_ISOLATION", root("PC_ISOLATION THREAD\n"),
         {"PC_ISOLATION = \"THREAD\"", "PC_ISOLATION_set = true",
          "PC_ISOLATION_line = 1"}},
        {"PC_WORKER_TIMEOUT, a tab in the line",
         root("PC_WORKER_TIMEOUT\t30\n"),
         {"PC_WORKER_TIMEOUT = 30", "PC_WORKER_TIMEOUT_line = 1",
          "PC_WORKER_TIMEOUT_text = \"PC_WORKER_TIMEOUT\\t30\""}},
        {"BUMPGATE", root("BUMPGATE STRICT\n"), {"BUMPGATE_STRICT = true"}},
        {"RHOGATE",
         "RHOGATE 0 R_LO 1.7168 NPTS 2001 SEED WARN\n"
         "OPTIM AMPGRIDMB (3) -0.5 0.5\nPC_ERMSE 0 1\nOPTIMIZER\nSTRATEGY\n"
         "LIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 10\nENDSTRATEGY\nENDOPTIMIZER\n",
         {"RHOGATE.tol = 0", "RHOGATE.r_lo = 1.7168", "RHOGATE.npts = 2001",
          "RHOGATE.seed_warn = true", "RHOGATE.line = 1",
          "RHOGATE.line_text = \"RHOGATE 0 R_LO 1.7168 NPTS 2001 SEED WARN\""},
         kReamConfig},
        {"RHOBARGATE",
         "RHOBARGATE BCC 3.304 0.9 1.1 npts 501 seed warn\n"
         "OPTIM AMPGRIDMB (3) -0.5 0.5\nPC_ERMSE 0 1\nOPTIMIZER\nSTRATEGY\n"
         "LIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 10\nENDSTRATEGY\nENDOPTIMIZER\n",
         {"RHOBARGATE.lattice = \"bcc\"", "RHOBARGATE.a0 = 3.304",
          "RHOBARGATE.f_lo = 0.9", "RHOBARGATE.f_hi = 1.1",
          "RHOBARGATE.npts = 501", "RHOBARGATE.seed_warn = true",
          "RHOBARGATE.line = 1"},
         kReamConfig},
        {"PC_FAILPROBE",
         root("PC_FAILPROBE --script lines_pc.in --outvar S --gate 5 "
              "--s0 2\n"),
         {"failprobe_idx = 0", "failprobe_svar = \"S\"", "failprobe_gate = 5",
          "failprobe_s0 = 2", "failprobe_s0_given = true",
          "LS[0].is_failprobe = true"}},
        {"OUTPUT, BOUTPUT, DIGITS", root("OUTPUT 5\nBOUTPUT 3\nDIGITS 9\n"),
         {"outputEvery = 5", "bestOutputEvery = 3", "outputDigits = 9"}},
        {"DUMP, a quote in the directory", root("DUMP 2 pots\"a\n"),
         {"dumpPotEvery = 2", "dumpPotDir = \"pots\\\"a\"",
          "dumpPotLine.number = 1",
          "dumpPotLine.text = \"DUMP 2 pots\\\"a\""}},
        {"BDUMP", root("BDUMP 3 bests\n"),
         {"dumpBestPotEvery = 3", "dumpBestPotDir = \"bests\"",
          "dumpBestPotLine.number = 1"}},
        {"LOSS", root("LOSS HUBER 2\n"),
         {"globalLoss.type = HUBER", "globalLoss.params = [2]"}},
        {"PC_EMAE with caps", root("PC_EMAE 0.5 2 --skip-above 10 --skip-rel 3\n"),
         {"use_emae = true", "emae = 0.5", "w_emae = 2",
          "skip_emae.abs_above = 10", "skip_emae.rel_above = 3"}},
        {"PC_ErRMSE", root("PC_ErRMSE 1 2 --skip-above 4\n"),
         {"use_errmse = true", "errmse = 1", "w_errmse = 2",
          "skip_errmse.abs_above = 4"}},
        {"PC_ERSQ", root("PC_ERSQ 0.9 1\n"),
         {"use_ersq = true", "ersq = 0.9", "w_ersq = 1"}},
        {"PC_FRMSE", root("PC_FRMSE 0.1 3\n"),
         {"use_frmse = true", "frmse_targets.global.target = 0.1",
          "frmse_targets.global.weight = 3", "frmse_targets.global.line = 1",
          "frmse_targets.global.text = \"PC_FRMSE 0.1 3\""}},
        {"PC_SRMSE_GPA of one dataset", root("PC_SRMSE_GPA (2) 0.5 1\n"),
         {"use_srmse = true", "srmse_targets.indexed.size = 1",
          "srmse_targets.indexed[2].target = 0.5",
          "srmse_targets.indexed[2].weight = 1"}},
        {"OPTIM_SCALE", root("OPTIM_SCALE NONE\n"),
         {"scale_policy.normalise = false", "scale_policy.set = true"}},
        {"an OPTIM line with a shape", root("OPTIM SGRID2B (2) 0.2 2.0 LOG\n"),
         {"optimap.size = 2", "optimap[SGRID2B].config = [\"0.5\", \"1\"]",
          "optimap[SGRID2B].size = 2", "optimap[SGRID2B][1].line = 1",
          "optimap[SGRID2B][1].low = 0.2", "optimap[SGRID2B][1].high = 2",
          "optimap[SGRID2B][1].scale = LOG", "optimap[SGRID2B][1].tagged = true",
          "optimap[SGRID2B][1].origin = OPTIM"}},
        {"an OPTIM AUTO line", root("OPTIM CGRID2B (2) AUTO\n"),
         {"optimap[CGRID2B][1].origin = AUTO FRAC",
          "optimap[CGRID2B][1].low = 1", "optimap[CGRID2B][1].high = 3",
          "optimap[CGRID2B][1].line = 1"}},
        {"a PC_OBSERVABLE line",
         root("PC_OBSERVABLE cohesive_energy --species H --target 2.0 "
              "--weight 1 --hotvar a0 1.5 2.5 2.0 --supercell 2 2 3 --loss L1 "
              "--maxiter 10 --maxeval 20 --unconverged force-scaled 10\n"),
         {"observables.size = 1", "observables[0].kind = \"cohesive_energy\"",
          "observables[0].species = \"H\"", "observables[0].supercell = [2, 2, 3]",
          "observables[0].target = 2", "observables[0].weight = 1",
          "observables[0].a0_min = 1.5", "observables[0].a0_max = 2.5",
          "observables[0].a0_initial = 2", "observables[0].lossfn.type = L1",
          "observables[0].lossfn_given = true",
          "observables[0].relax_maxiter = 10", "observables[0].relax_maxeval = 20",
          "observables[0].unconverged_strategy = force-scaled",
          "observables[0].unconverged_value = 10",
          "observable_lines.size = 1", "observable_lines[0].number = 1"}},
        {"optimiser keys",
         root("", "SEED 42\nTHREADS 2\nSTEP 0.1\nSTEP CGRID2B (1) 0.2\n"
                  "STOPVAL -1\nFTOL_REL 1e-3\nFTOL_ABS 1e-4\nXTOL_REL 1e-5\n"
                  "XTOL_ABS 1e-6\nMAXTIME 100\nPOPULATION 20\nVECTOR_ARR 5\n"
                  "PARAM xyz 1.5\nPARAM abc word\nLOG_HP 10\n"),
         {"spec.seed = 42", "spec.num_threads = 2", "spec.step = 0.1",
          "spec.step_overrides.size = 1",
          "spec.step_overrides[0].key = \"CGRID2B\"",
          "spec.step_overrides[0].indices = [1]",
          "spec.step_overrides[0].frac = 0.2",
          "spec.step_overrides[0].line = 11",
          "spec.stop.stopval = -1", "spec.stop.ftol_rel = 0.001",
          "spec.stop.ftol_abs = 0.0001", "spec.stop.xtol_rel = 1e-05",
          "spec.stop.xtol_abs = 1e-06", "spec.stop.maxtime = 100",
          "spec.pop = 20", "spec.vector_storage = 5",
          "spec.param[xyz] = 1.5", "spec.param_raw[abc] = \"word\"",
          "spec.log_threshold = 10", "spec.key_lines[STEP].number = 10"}},
        {"GRAD_STRATEGY", root("", "GRAD_STRATEGY HYBRID\n"),
         {"spec.grad_strategy = HYBRID",
          "spec.key_lines[GRAD_STRATEGY].number = 8",
          "spec.key_lines[GRAD_STRATEGY].block = \"STRATEGY\""}},
        {"STRATEGY TYPE MULTISTART",
         root("", "TYPE MULTISTART\nN_STARTS 3\nBUDGET_PER_START 4\n"),
         {"spec.strategy_type = MULTISTART", "spec.multistart.n_starts = 3",
          "spec.multistart.budget_per_start = 4",
          "spec.key_lines[N_STARTS].number = 9"}},
        {"STRATEGY TYPE PRESTAGE",
         "OPTIM CGRID2B (1) 0.5 1.5\nPC_ERMSE 0 1\nOPTIMIZER\nSTRATEGY\n"
         "TYPE PRESTAGE\nTOPK 2\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 10\n"
         "EXPLORE\nLIB TADAH\nALGO RANDOM\nMAXEVAL 5\nSEED 3\nENDEXPLORE\n"
         "ENDSTRATEGY\nENDOPTIMIZER\n",
         {"spec.strategy_type = PRESTAGE", "spec.stage_topk = 2",
          "spec.prestage.lib = \"TADAH\"", "spec.prestage.algo = \"RANDOM\"",
          "spec.prestage.seed = 3", "spec.prestage.stop.maxeval = 5",
          "spec.prestage.block_line = 15",
          "spec.prestage.block_line_text = \"ENDEXPLORE\""}},
        {"an INNER block",
         "OPTIM CGRID2B (1) 0.5 1.5\nPC_ERMSE 0 1\nOPTIMIZER\nSTRATEGY\n"
         "LIB NLOPT\nALGO LN_AUGLAG\nMAXEVAL 10\nINNER\nLIB NLOPT\n"
         "ALGO LN_BOBYQA\nMAXEVAL 5\nENDINNER\nENDSTRATEGY\nENDOPTIMIZER\n",
         {"spec.inner.lib = \"NLOPT\"", "spec.inner.algo = \"LN_BOBYQA\"",
          "spec.inner.stop.maxeval = 5", "spec.inner.block_line = 12",
          "spec.inner.inner = none"}},
        {"an INIT block",
         "OPTIM CGRID2B (1) 0.5 1.5\nPC_ERMSE 0 1\nOPTIMIZER\nINIT\n"
         "METHOD RANDOM 3\nK_TOTAL 2\nSEED 5\nWARM_ON_MISMATCH error\n"
         "ENDINIT\nSTRATEGY\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 10\n"
         "ENDSTRATEGY\nENDOPTIMIZER\n",
         {"spec.init_strategy_spec.chain.size = 1",
          "spec.init_strategy_spec.chain[0].key = \"METHOD\"",
          "spec.init_strategy_spec.chain[0].method = \"RANDOM\"",
          "spec.init_strategy_spec.chain[0].args = [\"3\"]",
          "spec.init_strategy_spec.chain[0].file = \"target.hpo\"",
          "spec.init_strategy_spec.chain[0].line = 5",
          "spec.init_strategy_spec.k_total = 2",
          "spec.init_strategy_spec.seed = 5",
          "spec.init_strategy_spec.warm_on_mismatch = \"error\"",
          "spec.init_strategy_spec.line = 4",
          "spec.init_strategy_spec.line_text = \"INIT\""}},
        {"a NOISE block",
         "OPTIM CGRID2B (1) 0.5 1.5\nPC_ERMSE 0 1\nOPTIMIZER\nNOISE\n"
         "AUTO false\nCALIBRATION 3\nENDNOISE\nSTRATEGY\nLIB NLOPT\n"
         "ALGO LD_LBFGS\nMAXEVAL 10\nENDSTRATEGY\nENDOPTIMIZER\n",
         {"spec.noise.calibration_n = 3"}},
        {"a RESTART block",
         root("", "RESTART\nAFTER_STAGNATION 20\nMAX 2\nMETHOD RANDOM 2\n"
                  "ENDRESTART\n"),
         {"spec.restart.after_stagnation = 20", "spec.restart.max = 2",
          "spec.restart.init_strategy.chain.size = 1",
          "spec.restart.init_strategy.chain[0].method = \"RANDOM\"",
          "spec.restart.line = 8", "spec.restart.line_text = \"RESTART\""}},
        {"PC_OBSERVABLE resolves an unset GRAD_STRATEGY",
         "OPTIM CGRID2B (1) 0.5 1.5\nPC_OBSERVABLE cohesive_energy --species H "
         "--target 2.0 --weight 1 --hotvar a0 1.5 2.5\nOPTIMIZER\nSTRATEGY\n"
         "LIB NLOPT\nALGO LD_LBFGS\nMAXEVAL 10\nENDSTRATEGY\nENDOPTIMIZER\n",
         {"spec.grad_strategy = ANALYTICAL",
          "spec.grad_strategy_resolved = true"}},
    };
    /* The file of each configuration without the lines of a setting: the
     * root wrapper, and the registry's main_ream wrapper. */
    const std::string without_default = describe(root(""));
    const std::string without_ream = describe(
        "OPTIM AMPGRIDMB (3) -0.5 0.5\nPC_ERMSE 0 1\nOPTIMIZER\nSTRATEGY\n"
        "LIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 10\nENDSTRATEGY\nENDOPTIMIZER\n",
        kReamConfig);
    for (const Setting& s : settings) {
        INFO(s.what << ":\n" << s.file);
        const std::string& without =
            s.config == kDefaultConfig ? without_default : without_ream;
        const std::string with = describe(s.file, s.config);
        for (const std::string& line : s.lines) {
            INFO("expected line: " << line << "\nthe description:\n" << with);
            CHECK(has_line(with, line));
            CHECK_FALSE(has_line(without, line));
        }
    }
}

TEST_CASE("describe_parse: the description of the root wrapper is in its "
          "canonical form", "[hpotarget_describe]")
{
    const std::string d = describe(root(""));
    INFO(d);
    for (const char* line : {
             "target_file = \"target.hpo\"",
             "FAIL_SCORE = 1.7976931348623157e+308",
             "MAX_TIME = \"off\"", "MAX_TIME_seconds = none",
             "RHOGATE = none", "RHOBARGATE = none", "failprobe_idx = -1",
             "failprobe_gate = inf", "LS.size = 0", "observables.size = 0",
             "use_ermse = true", "ermse_targets.global.target = 0",
             "ermse_targets.global.weight = 1",
             "ermse_targets.global.caps.abs_above = none",
             "optimap.size = 1", "optimap[CGRID2B].config = [\"1\", \"2\"]",
             "optimap[CGRID2B][0].line = 1", "optimap[CGRID2B][0].low = 0.5",
             "optimap[CGRID2B][0].high = 1.5", "optimap[CGRID2B][0].scale = LIN",
             "optimap[CGRID2B][0].c = 1", "optimap[CGRID2B][0].tagged = false",
             "spec.strategy_type = PLAIN", "spec.lib = \"NLOPT\"",
             "spec.algo = \"LN_SBPLX\"", "spec.seed = none",
             "spec.stop.maxeval = 10", "spec.param.size = 0",
             "spec.source_file = \"target.hpo\"", "spec.block_line = 9",
             "spec.block_line_text = \"ENDOPTIMIZER\"",
             "spec.key_lines.size = 2", "spec.key_lines[ALGO].number = 6",
             "spec.key_lines[ALGO].text = \"ALGO LN_SBPLX\"",
             "spec.key_lines[LIB].block = \"STRATEGY\"",
             "spec.inner = none", "spec.prestage = none",
             "context[RCUT2B] = [\"5.0\"]",
             "context[TYPE2B] = [\"D2_BP\", \"2\", \"2\", \"H\", \"H\"]"}) {
        INFO("expected line: " << line);
        CHECK(has_line(d, line));
    }
    /* One line per setting: `<path> = <value>`. */
    std::istringstream in(d);
    std::string line;
    std::size_t n = 0;
    while (std::getline(in, line)) {
        ++n;
        INFO("line " << n << ": " << line);
        CHECK(line.find(" = ") != std::string::npos);
    }
    CHECK(n > 100);
}

TEST_CASE("describe_parse: two constructions of the targets of one file give "
          "one description", "[hpotarget_describe]")
{
    /* The INIT block has no SEED and TADAH:RANDOM draws random numbers: the
     * constructor draws a seed for each. The exploration of the second file
     * takes the STRATEGY-level SEED. NOISE AUTO derives PARAM fd_step for
     * the gradient-based LD_LBFGS. */
    const std::string drawn =
        "OPTIM CGRID2B (1) 0.5 1.5\nPC_ERMSE 0 1\nOPTIMIZER\nINIT\n"
        "METHOD RANDOM 2\nENDINIT\nSTRATEGY\nLIB TADAH\nALGO RANDOM\n"
        "MAXEVAL 10\nENDSTRATEGY\nENDOPTIMIZER\n";
    const std::string taken =
        "OPTIM CGRID2B (1) 0.5 1.5\nPC_ERMSE 0 1\nOPTIMIZER\nSTRATEGY\n"
        "TYPE PRESTAGE\nSEED 7\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 10\n"
        "EXPLORE\nLIB TADAH\nALGO RANDOM\nMAXEVAL 5\nENDEXPLORE\n"
        "ENDSTRATEGY\nENDOPTIMIZER\n";
    const std::string noise =
        "OPTIM CGRID2B (1) 0.5 1.5\nPC_ERMSE 0 1\nOPTIMIZER\nNOISE\nAUTO true\n"
        "ENDNOISE\nSTRATEGY\nLIB NLOPT\nALGO LD_LBFGS\nMAXEVAL 10\n"
        "ENDSTRATEGY\nENDOPTIMIZER\n";
    for (const std::string& file : {root(""), drawn, taken, noise}) {
        INFO(file);
        const std::string a = describe(file);
        const std::string b = describe(file);
        CHECK(a == b);
    }
    const std::string d = describe(drawn);
    CHECK(has_line(d, "spec.seed = none"));
    CHECK(has_line(d, "spec.init_strategy_spec.seed = none"));
    const std::string t = describe(taken);
    CHECK(has_line(t, "spec.seed = 7"));
    CHECK(has_line(t, "spec.prestage.seed = none"));
    const std::string n = describe(noise);
    CHECK(has_line(n, "spec.noise.autodetect = true"));
    CHECK(has_line(n, "spec.param.size = 0"));
}

/* Pins: the LAMMPS command line of a PC_LAMMPS line (LammpsSim::invar,
 * whose first argument is the LAMMPS library of the build,
 * TADAH_LAMMPS_LIB) is in the text from its second argument on
 * (`LS[i].invar[1:]`), and the library is not.
 * Before the fix: fails, the text holds `LS[0].invar` with the library. */
TEST_CASE("describe_parse: the LAMMPS command line of a PC_LAMMPS line "
          "leaves out the library of the build", "[hpotarget_describe]")
{
    tadah::core::PeriodicTable::initialize();
    Scratch s;
    std::ofstream("target.hpo")
        << root("PC_LAMMPS --script lines_pc.in --varloss E -5.0 1.0 "
                "--invar b 1\n");
    Context ctx = config_of(kDefaultConfig);
    HPOTargets trg(ctx, "target.hpo");
    REQUIRE(trg.LS.size() == 1);
    REQUIRE(trg.LS[0].invar.size() > 1);
    const std::string lib = trg.LS[0].invar[0];
    const std::string d = describe_parse(trg);
    CHECK(d.find(lib) == std::string::npos);
    std::string args;
    for (std::size_t i = 1; i < trg.LS[0].invar.size(); ++i)
        args += (args.empty() ? "\"" : ", \"") + trg.LS[0].invar[i] + "\"";
    CHECK(has_line(d, "LS[0].invar[1:] = [" + args + "]"));
}

TEST_CASE("describe_parse: a PC_OBSERVABLE kind without a description is "
          "refused", "[hpotarget_describe]")
{
    tadah::core::PeriodicTable::initialize();
    Scratch s;
    std::ofstream("target.hpo") << root("");
    Context ctx = config_of(kDefaultConfig);
    HPOTargets trg(ctx, "target.hpo");
    trg.observables.push_back(std::make_unique<UnknownObservable>());
    REQUIRE_THROWS_AS(describe_parse(trg), std::logic_error);
    REQUIRE_THROWS_WITH(describe_parse(trg),
                        Catch::Contains("PC_OBSERVABLE kind 'test_kind'"));
}

#else

TEST_CASE("describe_parse: the header hpo_targets_testing.h exists",
          "[hpotarget_describe]")
{
    FAIL("tadah/hpo/hpo_targets_testing.h does not exist: describe_parse() "
         "is not implemented");
}

#endif
