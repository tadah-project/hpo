// test_analytical_gradient_audit.cpp
//
// The analytical gradient of the tadah hpo loss against central
// differences of the loss, through HPO_Host as GRAD_STRATEGY HYBRID and
// ANALYTICAL call it: host(p) for the loss, host.analytical_gradient(p)
// after host(p) for the gradient and the parameters it hands to finite
// differences (needs_fd).
//
// For every two-body descriptor that reads WATOM, D2_mJoin and none,
// beside the many-body model of each test case, the search space holds
// index 1 of every parameter of the training configuration that the closed
// forms of a descriptor take (WATOM, the grids, the cutoffs, the embedding
// grids, RCUTENV, the component paths of DESCRIPTOR blocks). Each setup is
// run with FORCE false and true, STRESS false and true, and the loss terms
// PC_ERMSE, and PC_ERMSE with PC_FRMSE (LOSS L2, M_BLR BF_Linear, OALGO 4,
// LAMBDA 1e-3), on structures with pairs inside both cutoffs; with
// FORCE true and both loss terms also on structures whose pairs lie inside
// the two-body cutoff only, and inside the many-body cutoff only. The
// energies and forces of the structures are those of a reference pair
// potential (reference_pair), which the descriptors fit closely.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/analytical_gate.h>
#include <tadah/hpo/dloss_dtheta.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/runners.h>
#include <tadah/models/descriptors/dm/embed_spec.h>

#include <dlib/matrix.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using tadah::core::Context;
using ParamVec = dlib::matrix<double, 0, 1>;
using Host = tadah::hpo::HPO_Host<ParamVec, ::LammpsRunner>;

std::string num(double v)
{
    std::ostringstream o;
    o.precision(17);
    o << v;
    return o.str();
}

/* A temporary working directory, the cwd while the object lives (the host
 * writes its result files there). */
struct ScratchDir {
    fs::path dir, original_cwd;
    ScratchDir()
    {
        dir = fs::temp_directory_path() /
              ("tadah_gradient_audit_" + std::to_string(::getpid()) + "_" +
               std::to_string(std::chrono::steady_clock::now()
                                  .time_since_epoch().count()));
        fs::create_directories(dir);
        original_cwd = fs::current_path();
        fs::current_path(dir);
    }
    ~ScratchDir()
    {
        std::error_code ec;
        fs::current_path(original_cwd);
        fs::remove_all(dir, ec);
    }
};

/* ---- the descriptors ------------------------------------------------- */

/* The two-body descriptor `name` ("none": INIT2B false) with the cutoff
 * Cut_Cos `rc`: grids whose functions reach the pairs of the structures
 * below. The knots of D2_Knot5 end at the cutoff, which tadah hpo takes
 * from the largest knot. */
std::string two_body(const std::string& name, double rc)
{
    const std::string cut = "RCTYPE2B Cut_Cos\nRCUT2B " + num(rc) + "\n";
    const std::string grid = "CGRID2B 2.8 3.6\nSGRID2B 1.0 1.1\n";
    if (name == "none") return "INIT2B false\n";
    if (name == "D2_Blip" || name == "D2_Blip5" || name == "D2_BP")
        return "INIT2B true\nTYPE2B " + name + " 2 2 Ta Ta\n" + grid + cut;
    if (name == "D2_Knot5")
        return "INIT2B true\nTYPE2B D2_Knot5 2 2 Ta Ta\nCGRID2B " +
               num(0.75 * rc) + " " + num(rc) + "\nKDEG2B 5 5\n" + cut;
    if (name == "D2_LJ") return "INIT2B true\nTYPE2B D2_LJ Ta Ta\n" + cut;
    if (name == "D2_MIE")
        return "INIT2B true\nTYPE2B D2_MIE 12 6 Ta Ta\n" + cut;
    if (name == "D2_ZBL")
        return "INIT2B true\nTYPE2B D2_ZBL 0.5292 0.88534 0.23 0.23 Ta Ta\n" +
               cut;
    if (name == "D2_mJoin")
        return "INIT2B true\nTYPE2B D2_mJoin\nTYPE2B D2_BP 1 1 Ta Ta\n"
               "TYPE2B D2_Blip 1 1 Ta Ta\nRCTYPE2B Cut_Cos Cut_Cos\nRCUT2B " +
               num(rc) + " " + num(rc) + "\n" + grid;
    FAIL("unknown two-body descriptor " << name);
    return {};
}

/* A DESCRIPTOR block `id` (a DM_Embed component) of the density basis
 * `density` and the embedding basis `embedding`, cutoff Cut_Cos `rc`. */
std::string block(const std::string& id, const std::string& density,
                  const std::string& embedding, double rc)
{
    return "DESCRIPTOR " + id + " DM_Embed\nPAIRS Ta Ta\nDENSITY\n" +
           density + "CUTOFF\nTYPE Cut_Cos\nRADIUS " + num(rc) +
           "\nENDCUTOFF\nENDDENSITY\nEMBEDDING\n" + embedding +
           "REFERENCE ZERO\nENDEMBEDDING\nENDDESCRIPTOR\n";
}

const std::string kReamLine = "TYPEMB DM_REAM 3 3 3 3 5 5 0 0 Ta Ta\n";
const std::string kReamGrids =
    "CGRIDMB 2.8 3.6 4.4\nSGRIDMB 3.0 3.0 3.0\nAMPGRIDMB 0.5 0.3 0.1\n"
    "CEMBFUNC -0.2 0.3 0.8 1.3 1.8\nSEMBFUNC 2.0 2.0 2.0 2.0 2.0\n";

/* The many-body model `name` ("none": INITMB false) with the cutoff
 * Cut_Cos `rc`: DM_REAM in the density form after its name, "DESCRIPTOR"
 * one DESCRIPTOR block (DM_Embed), "DM_mJoin lines" DM_mBlip and DM_REAM
 * joined, "DM_mJoin blocks" two DESCRIPTOR blocks. */
std::string many_body(const std::string& name, double rc)
{
    const std::string cut = "RCTYPEMB Cut_Cos\nRCUTMB " + num(rc) + "\n";
    const std::string grid = "CGRIDMB 2.8 3.6\nSGRIDMB 1.0 1.1\n";
    const std::string emb = "CEMBFUNC 1.0 2.0\nSEMBFUNC 1.0 1.0\n";
    const std::string blip5 =
        "BASIS Blip5\nFORM SIGNED\nCENTRES 2.8 3.6 4.4\nETA 3 3 3\n"
        "AMPLITUDES 0.5 0.3 0.1\n";
    const std::string gauss = "BASIS Gaussian\nCENTRES 0 0.5 1\nETA 2 2 2\n";
    if (name == "none") return "INITMB false\n";
    if (name == "DM_Blip")
        return "INITMB true\nTYPEMB DM_Blip 0 2 2 Ta Ta\n" + grid + cut;
    if (name == "DM_EAD")
        return "INITMB true\nTYPEMB DM_EAD 1 2 2 Ta Ta\n" + grid + cut;
    if (name == "DM_mBlip")
        return "INITMB true\nTYPEMB DM_mBlip 0 1 1 2 2 Ta Ta\n"
               "CGRIDMB 0.0\nSGRIDMB 0.2\n" + emb + cut;
    if (name == "DM_mBlipEnv")
        return "INITMB true\nTYPEMB DM_mBlipEnv 0 1 1 2 2 Ta Ta\n"
               "CGRIDMB 0.0\nSGRIDMB 0.2\n" + emb + "RCUTENV 4.0\n" + cut;
    if (name == "DM_mRLR" || name == "DM_mSQRT")
        return "INITMB true\nTYPEMB " + name + " 0 2 2 2 2 Ta Ta\n" + grid +
               emb + cut;
    if (name.rfind("DM_REAM ", 0) == 0)
        return "INITMB true\n" + kReamLine + kReamGrids + "RHOFORM " +
               name.substr(8) + "\n" + cut;
    if (name == "DM_PS_SQ")
        return "INITMB true\nTYPEMB DM_PS_SQ 1 2 2 Ta Ta\n" + grid + cut;
    if (name == "DESCRIPTOR")
        return "INITMB true\n" + block("ta", blip5, gauss, rc);
    if (name == "DM_mJoin lines")
        return "INITMB true\nTYPEMB DM_mJoin\n"
               "TYPEMB DM_mBlip 0 1 1 2 2 Ta Ta\n" + kReamLine +
               "RCTYPEMB Cut_Cos Cut_Cos\nRCUTMB " + num(rc) + " " + num(rc) +
               "\nCGRIDMB 0.0 2.8 3.6 4.4\nSGRIDMB 0.2 3.0 3.0 3.0\n"
               "AMPGRIDMB 0.5 0.3 0.1\n"
               "CEMBFUNC 1.0 2.0 -0.2 0.3 0.8 1.3 1.8\n"
               "SEMBFUNC 1.0 1.0 2.0 2.0 2.0 2.0 2.0\nRHOFORM SIGNED\n";
    if (name == "DM_mJoin blocks")
        return "INITMB true\n" + block("ta", blip5, gauss, rc) +
               block("tb",
                     "BASIS Blip\nFORM SIGNED\nCENTRES 3.0 3.8\nETA 1.5 1.5\n"
                     "AMPLITUDES 0.4 0.2\n",
                     "BASIS Blip\nCENTRES 0.2 0.9\nETA 1 1\n", rc);
    FAIL("unknown many-body model " << name);
    return {};
}

/* ---- the structures ------------------------------------------------- */

/* A structure of Ta atoms in an orthorhombic cell (`sides`, A) at
 * `positions`. */
struct Cell {
    std::array<double, 3> sides;
    std::vector<std::array<double, 3>> positions;
};

/* The reference pair potential the energies and forces of the structures
 * come from, a Morse potential (0.6 eV, 1.4 /A, 2.86 A) tapered to 0 at
 * 4.5 A by (cos(pi r / 4.5) + 1) / 2: the descriptors fit it closely, so
 * the loss is a misfit of a few meV per atom, not of the arbitrary targets
 * a model cannot reach, and its round-off stays small beside its
 * derivatives. Returns phi(r) and phi'(r). */
std::pair<double, double> reference_pair(double r)
{
    const double rc = 4.5, d = 0.6, a = 1.4, r0 = 2.86;
    if (r >= rc) return {0.0, 0.0};
    const double pi = std::acos(-1.0);
    const double e = std::exp(-a * (r - r0));
    const double phi = d * ((1.0 - e) * (1.0 - e) - 1.0);
    const double dphi = 2.0 * d * a * (1.0 - e) * e;
    const double fc = 0.5 * (std::cos(pi * r / rc) + 1.0);
    const double dfc = -0.5 * pi / rc * std::sin(pi * r / rc);
    return {phi * fc, dphi * fc + phi * dfc};
}

/* The dataset file `path` of `cells` (eweight fweight sweight 1, stresses
 * 0), the energy and the forces of each from reference_pair() over every
 * periodic image within its range. */
void write_db(const fs::path& path, const std::vector<Cell>& cells)
{
    std::ofstream os(path);
    os.precision(17);
    for (std::size_t c = 0; c < cells.size(); ++c) {
        const Cell& cell = cells[c];
        const std::size_t n = cell.positions.size();
        std::array<int, 3> images{};
        for (int k = 0; k < 3; ++k)
            images[k] = static_cast<int>(std::ceil(4.5 / cell.sides[k]));
        double energy = 0.0;
        std::vector<std::array<double, 3>> force(n, {0.0, 0.0, 0.0});
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                for (int a = -images[0]; a <= images[0]; ++a)
                    for (int b = -images[1]; b <= images[1]; ++b)
                        for (int g = -images[2]; g <= images[2]; ++g) {
                            if (i == j && a == 0 && b == 0 && g == 0)
                                continue;
                            const std::array<double, 3> shift = {
                                a * cell.sides[0], b * cell.sides[1],
                                g * cell.sides[2]};
                            std::array<double, 3> d{};
                            double r2 = 0.0;
                            for (int k = 0; k < 3; ++k) {
                                d[k] = cell.positions[i][k] -
                                       cell.positions[j][k] - shift[k];
                                r2 += d[k] * d[k];
                            }
                            const double r = std::sqrt(r2);
                            const auto [phi, dphi] = reference_pair(r);
                            energy += 0.5 * phi;
                            for (int k = 0; k < 3; ++k)
                                force[i][k] -= dphi * d[k] / r;
                        }
        os << "cell " << c + 1 << "\n1.0 1.0 1.0\n" << energy << " 0.0\n"
           << cell.sides[0] << " 0 0\n0 " << cell.sides[1] << " 0\n0 0 "
           << cell.sides[2] << "\n0 0 0\n0 0 0\n0 0 0\n";
        for (std::size_t i = 0; i < n; ++i)
            os << "Ta " << cell.positions[i][0] << " " << cell.positions[i][1]
               << " " << cell.positions[i][2] << " " << force[i][0] << " "
               << force[i][1] << " " << force[i][2] << "\n";
        os << "\n";
    }
}

/* Ta atoms at the corners of a triangle with sides `a`, `b`, `c` (A) in a
 * box of 20 A, out of the plane of the axes. */
Cell triangle(double a, double b, double c)
{
    const double x = (a * a + b * b - c * c) / (2.0 * a);
    const double y = std::sqrt(b * b - x * x);
    return {{20.0, 20.0, 20.0},
            {{5.0, 5.0, 5.0},
             {5.0 + a * 0.8, 5.0 + a * 0.6, 5.0},
             {5.0 + x * 0.8 - y * 0.36, 5.0 + x * 0.6 + y * 0.48,
              5.0 + y * 0.8}}};
}

/* Ta atoms at distance `r` along (0.6, 0.48, 0.64) in a box of 20 A. */
Cell pair_cell(double r)
{
    const double n = std::sqrt(0.6 * 0.6 + 0.48 * 0.48 + 0.64 * 0.64);
    return {{20.0, 20.0, 20.0},
            {{5.0, 5.0, 5.0},
             {5.0 + r * 0.6 / n, 5.0 + r * 0.48 / n, 5.0 + r * 0.64 / n}}};
}

/* The data of a setup: the training and the validation structures and the
 * cutoffs of the two-body and the many-body descriptor. */
struct Data {
    const char* name;
    std::vector<Cell> train, val;
    double rc2, rcm;
};

/* A two-atom bcc Ta cell of side `a`, the second atom moved by `d` off
 * the centre. */
Cell bcc2(double a, std::array<double, 3> d)
{
    return {{a, a, a},
            {{0.0, 0.0, 0.0},
             {0.5 * a + d[0], 0.5 * a + d[1], 0.5 * a + d[2]}}};
}

/* Bcc Ta cells off their sites, dimers and trimers: pairs at many
 * distances inside both cutoffs (4.5 A), more structures than the
 * descriptors have entries, so that the fits with energy rows only are
 * well posed. */
Data pairs_inside_both()
{
    const Cell c4{{3.31, 3.31, 6.62},
                  {{0.04, 0.0, 0.02},
                   {1.66, 1.61, 1.7},
                   {0.0, 0.07, 3.27},
                   {1.6, 1.68, 4.99}}};
    const Cell c4b{{3.27, 3.27, 6.54},
                   {{0.0, 0.03, 0.0},
                    {1.6, 1.68, 1.61},
                    {0.06, 0.0, 3.3},
                    {1.66, 1.6, 4.86}}};
    return {"pairs inside both cutoffs (4.5 A)",
            {bcc2(3.2, {0.07, -0.04, 0.02}), bcc2(3.25, {-0.08, 0.05, 0.0}),
             bcc2(3.3, {0.07, -0.04, 0.01}), bcc2(3.35, {0.02, 0.09, -0.05}),
             bcc2(3.4, {-0.06, 0.0, 0.08}), c4, c4b, pair_cell(2.6),
             pair_cell(2.9), pair_cell(3.3), pair_cell(3.8), pair_cell(4.2),
             triangle(2.7, 3.1, 3.9), triangle(2.8, 3.4, 4.2),
             triangle(3.0, 3.0, 3.0), triangle(2.6, 3.6, 4.0)},
            {bcc2(3.32, {-0.05, 0.05, 0.03}), bcc2(3.22, {0.04, 0.06, -0.07}),
             triangle(2.8, 3.3, 4.1), pair_cell(3.1)},
            4.5,
            4.5};
}

/* Dimers and trimers whose pairs lie between 3 A and 4.5 A: inside the
 * cutoff of one descriptor (4.5 A) and beyond that of the other (3 A). */
Data pairs_between(const char* name, double rc2, double rcm)
{
    return {name,
            {pair_cell(3.2), pair_cell(3.5), pair_cell(3.7), pair_cell(3.9),
             pair_cell(4.1), pair_cell(4.3), triangle(3.4, 3.8, 4.2),
             triangle(3.2, 3.6, 4.4), triangle(3.3, 3.5, 3.9),
             triangle(3.6, 4.0, 4.4)},
            {pair_cell(3.6), triangle(3.3, 3.9, 4.3)},
            rc2,
            rcm};
}

/* ---- the setup ------------------------------------------------------- */

/* A search parameter: its OPTIM key (a component path in lower case) and
 * its value in the training configuration (index 1). */
struct Parameter {
    std::string key;
    double value;
};

/* The search parameters of `c` that the closed forms of a descriptor take:
 * index 1 of each key the configuration sets, and of each component path.
 * D2_Knot5's RCUT2B is not one: tadah hpo takes its cutoff from the
 * largest knot, whatever the OPTIM line gives. */
std::vector<Parameter> parameters(const Context& c, bool knot5)
{
    std::vector<Parameter> out;
    for (const char* key :
         {"WATOM", "CGRID2B", "SGRID2B", "RCUT2B", "CGRIDMB", "SGRIDMB",
          "RCUTMB", "AMPGRIDMB", "CEMBFUNC", "SEMBFUNC", "RCUTENV"}) {
        if (!c.has_stored(key)) continue;
        if (knot5 && std::string(key) == "RCUT2B") continue;
        out.push_back({key, c.get<double>(key, 0)});
    }
    for (const auto& b : tadah::models::embed_bindings(c)) {
        std::string path = b.path;
        std::transform(path.begin(), path.end(), path.begin(),
                       [](unsigned char ch) { return std::tolower(ch); });
        out.push_back({path, b.values.at(0)});
    }
    return out;
}

/* One run of the audit: the training configuration's settings. */
struct Run {
    bool force, stress, frmse;
};

/* The loss and its gradient at the values of the configuration, the
 * parameters on needs_fd and the parameters the ANALYTICAL startup check
 * lists, for `d2` beside `dm` on `data` with the settings of `run`. */
struct Outcome {
    std::vector<std::string> labels;   // "WATOM(1)", in the host's order
    std::vector<double> gradient;      // analytical, 0 on needs_fd
    std::vector<double> d1, d2;        // central differences, h and h/2
    std::vector<double> loss_scale;    // the largest |loss| evaluated
    std::vector<double> steps;         // h
    std::set<std::string> needs_fd, without_derivative;
};

Outcome audit(const std::string& d2, const std::string& dm, const Data& data,
              const Run& run)
{
    Host::reset_globals_for_test();
    Host::reset_best_state_for_fresh_run();
    ScratchDir sd;
    write_db("train.db", data.train);
    write_db("val.db", data.val);
    std::ofstream("config.val") << "DBFILE val.db\n";

    std::istringstream in(
        "DBFILE train.db\nATOM Ta\nWATOM 1.3\nMODEL M_BLR BF_Linear\n"
        "VERBOSE 0\nBIAS false\nNORM false\nOALGO 4\nLAMBDA 1e-3\n"
        "FORCE " + std::string(run.force ? "true" : "false") +
        "\nSTRESS " + std::string(run.stress ? "true" : "false") + "\n" +
        two_body(d2, data.rc2) + many_body(dm, data.rcm) +
        "HPOTARGET config.hpo\nVALIDATION config.val\n");
    Context ctx = tadah::core::CoreUserConfigParser().parse(in);

    const std::vector<Parameter> params = parameters(ctx, d2 == "D2_Knot5");
    {
        std::ofstream os("config.hpo");
        os.precision(17);
        os << "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\n"
              "GRAD_STRATEGY HYBRID\nENDOPTIMIZER\nLOSS L2\n"
              "FAILSCORE 1e16\nPC_ERMSE 0 1\n";
        if (run.frmse) os << "PC_FRMSE 0 1\n";
        for (const Parameter& p : params) {
            const double d = 0.05 * std::max(1.0, std::abs(p.value));
            os << "OPTIM " << p.key << " (1) " << p.value - d << " "
               << p.value + d << "\n";
        }
    }
    tadah::hpo::HPOTargets trg(ctx, "config.hpo");
    Host host(ctx, trg, "config.val");

    const std::vector<tadah::hpo::ParamRef> prefs =
        tadah::hpo::make_param_ref_table(trg);
    ParamVec p(static_cast<long>(prefs.size()));
    Outcome out;
    for (const auto& r : prefs) {
        const auto& opti = trg.optimap.at(r.theta_name);
        p(static_cast<long>(r.flat_idx)) =
            tadah::core::to_double(opti.config.at(r.token));
        out.labels.push_back(r.theta_name + "(" + std::to_string(r.arg_pos) +
                             ")");
    }

    const double f0 = host(p);
    REQUIRE(std::isfinite(f0));
    ParamVec g;
    std::vector<int> needs_fd;
    REQUIRE(host.analytical_gradient(p, g, needs_fd));
    for (int i : needs_fd)
        out.needs_fd.insert(out.labels.at(static_cast<std::size_t>(i)));
    if (!run.stress) {
        for (const auto& w : tadah::hpo::parameters_without_analytical_derivative(
                 host.master_context(), trg, host.train_stdb(),
                 host.val_stdb()))
            out.without_derivative.insert(w.key + "(" +
                                          std::to_string(w.index) + ")");
    }
    for (std::size_t i = 0; i < prefs.size(); ++i) {
        const long k = static_cast<long>(prefs[i].flat_idx);
        out.gradient.push_back(g(k));
        const double h = 1e-3 * std::max(1.0, std::abs(p(k)));
        auto at = [&](double step) {
            ParamVec q = p;
            q(k) += step;
            return host(q);
        };
        double largest = std::abs(f0);
        std::vector<double> f;
        for (double s : {h, -h, 0.5 * h, -0.5 * h}) {
            f.push_back(at(s));
            largest = std::max(largest, std::abs(f.back()));
        }
        out.d1.push_back((f[0] - f[1]) / (2.0 * h));
        out.d2.push_back((f[2] - f[3]) / h);
        out.loss_scale.push_back(largest);
        out.steps.push_back(h);
    }
    host.flush_remaining();
    Host::reset_globals_for_test();
    Host::reset_best_state_for_fresh_run();
    return out;
}

/* Whether the analytical derivative `a` of the loss is its central
 * difference (d1 with the step h, d2 with h/2, the loss at most `loss`).
 *
 * Tolerance. The central difference D(h) of a loss with three continuous
 * derivatives errs by c h^2 + O(h^4), so D(h) - D(h/2) is three times the
 * error of D(h/2), and R = D(h/2) + (D(h/2) - D(h))/3 removes the h^2 term.
 * The bound is 2|D(h) - D(h/2)|, many times the remainder of R, plus the
 * round-off of the loss divided by the step, plus 1e-4 of |R|. Every
 * evaluation fits the coefficients by least squares, whose round-off is
 * allowed 1e4 ulp of the loss; the step, 1e-3 of the value (at least
 * 1e-3), keeps that term near 2e-9 of the loss per unit of the parameter.
 * The analytical gradient solves the normal equations of the fit once
 * more, and its round-off grows with their condition number (the nearly
 * collinear embedding columns of DM_REAM make it some 1e-5 of the
 * derivative): 1e-4 of |R|, which also covers the O(h^4) term of R. A
 * gradient without the part of a descriptor (the WATOM derivative of a
 * two-body descriptor beside DM_REAM) misses by far more. */
bool derivative_matches(double a, double d1, double d2, double loss,
                        double h, double& bound, double& r)
{
    r = d2 + (d2 - d1) / 3.0;
    bound = 2.0 * std::abs(d1 - d2) +
            1e4 * std::numeric_limits<double>::epsilon() * loss / h +
            1e-4 * std::abs(r);
    return std::abs(a - r) <= bound;
}

const std::vector<std::string>& two_body_descriptors()
{
    static const std::vector<std::string> v = {
        "D2_Blip", "D2_Blip5", "D2_BP",    "D2_Knot5", "D2_LJ",
        "D2_MIE",  "D2_ZBL",   "D2_mJoin", "none"};
    return v;
}

/* The audit of the many-body model `dm` beside every two-body descriptor:
 *   - STRESS true: every parameter on needs_fd (the analytical gradient
 *     does not differentiate stress rows);
 *   - STRESS false: a parameter not on needs_fd has an analytical
 *     derivative equal to the central difference of the loss
 *     (derivative_matches), and the parameters the ANALYTICAL startup
 *     check lists (parameters_without_analytical_derivative) are exactly
 *     those on needs_fd: ANALYTICAL refuses what HYBRID takes by finite
 *     differences.
 * Some derivatives are compared. */
void audit_beside_every_two_body(const std::string& dm)
{
    tadah::core::PeriodicTable::initialize();
    std::size_t compared = 0;
    for (const std::string& d2 : two_body_descriptors()) {
        if (d2 == "none" && dm == "none") continue;
        std::vector<std::pair<Data, Run>> runs;
        for (const bool stress : {false, true})
            for (const bool force : {false, true})
                for (const bool frmse : {false, true})
                    runs.push_back({pairs_inside_both(),
                                    Run{force, stress, frmse}});
        runs.push_back({pairs_between("pairs inside the two-body cutoff only "
                                      "(4.5 A; many-body 3 A)",
                                      4.5, 3.0),
                        Run{true, false, true}});
        runs.push_back({pairs_between("pairs inside the many-body cutoff only "
                                      "(4.5 A; two-body 3 A)",
                                      3.0, 4.5),
                        Run{true, false, true}});
        for (const auto& [data, run] : runs) {
            INFO(d2 << " + " << dm << ", " << data.name << ", FORCE "
                    << run.force << ", STRESS " << run.stress
                    << (run.frmse ? ", PC_ERMSE and PC_FRMSE" : ", PC_ERMSE"));
            const Outcome o = audit(d2, dm, data, run);
            std::string fd_list, listed;
            for (const auto& l : o.needs_fd) fd_list += " " + l;
            for (const auto& l : o.without_derivative) listed += " " + l;
            INFO("needs_fd:" << fd_list << "; ANALYTICAL refuses:" << listed);
            if (run.stress) {
                CHECK(o.needs_fd.size() == o.labels.size());
                continue;
            }
            CHECK(o.needs_fd == o.without_derivative);
            for (std::size_t i = 0; i < o.labels.size(); ++i) {
                if (o.needs_fd.count(o.labels[i])) continue;
                ++compared;
                double bound = 0.0, r = 0.0;
                const bool ok =
                    derivative_matches(o.gradient[i], o.d1[i], o.d2[i],
                                       o.loss_scale[i], o.steps[i], bound, r);
                INFO(o.labels[i] << ": analytical " << o.gradient[i]
                                 << ", central difference " << r << " (h: "
                                 << o.d1[i] << ", h/2: " << o.d2[i]
                                 << "), bound " << bound);
                CHECK(ok);
            }
        }
    }
    INFO("analytical derivatives compared: " << compared);
    CHECK(compared > 0);
}

}  // namespace

/* Pins: the analytical gradient of the loss (audit_beside_every_two_body):
 * every parameter on needs_fd under STRESS true; under STRESS false, each
 * parameter not on needs_fd differentiated as the central difference of
 * the loss, and needs_fd the parameters the ANALYTICAL startup check
 * refuses; for each two-body descriptor that reads WATOM, D2_mJoin and
 * none beside the many-body model of the test case.
 * Before the fix: every test case fails but the first and DM_mRLR. The
 * WATOM derivative of a two-body descriptor beside DM_REAM (each form), a
 * DESCRIPTOR block, DM_mJoin (lines or blocks) and DM_PS_SQ was not on
 * needs_fd and lacked the part of the many-body descriptor (D2_Knot5 and
 * DM_REAM NONNEG, FORCE false, PC_ERMSE: -1.7e-3 against a central
 * difference of -354; on the structures whose pairs lie inside the
 * many-body cutoff only, 0), and so did that of D2_mJoin beside DM_Blip,
 * DM_EAD and DM_mSQRT (FORCE false, PC_ERMSE alone), and beside DM_mBlip
 * and DM_mBlipEnv on the structures inside the two-body cutoff only, the
 * part of D2_mJoin missing. Beside DM_mRLR that part changes the loss by
 * less than the bound (it scales entries the fit absorbs but for its
 * ridge); the walker's rows show it (MLIP test_dphi_dtheta_rows_fd.cpp).
 * The first test case passes as well (a pin). Run time: about 60 s for the
 * fourteen test cases. */
TEST_CASE("analytical gradient audit: no many-body descriptor beside each "
          "two-body descriptor",
          "[HPO][analytical_gradient][audit][watom]")
{
    audit_beside_every_two_body("none");
}

TEST_CASE("analytical gradient audit: DM_Blip beside each two-body "
          "descriptor",
          "[HPO][analytical_gradient][audit][watom]")
{
    audit_beside_every_two_body("DM_Blip");
}

TEST_CASE("analytical gradient audit: DM_EAD beside each two-body "
          "descriptor",
          "[HPO][analytical_gradient][audit][watom]")
{
    audit_beside_every_two_body("DM_EAD");
}

TEST_CASE("analytical gradient audit: DM_mBlip beside each two-body "
          "descriptor",
          "[HPO][analytical_gradient][audit][watom]")
{
    audit_beside_every_two_body("DM_mBlip");
}

TEST_CASE("analytical gradient audit: DM_mBlipEnv beside each two-body "
          "descriptor",
          "[HPO][analytical_gradient][audit][watom]")
{
    audit_beside_every_two_body("DM_mBlipEnv");
}

TEST_CASE("analytical gradient audit: DM_mRLR beside each two-body "
          "descriptor",
          "[HPO][analytical_gradient][audit][watom]")
{
    audit_beside_every_two_body("DM_mRLR");
}

TEST_CASE("analytical gradient audit: DM_mSQRT beside each two-body "
          "descriptor",
          "[HPO][analytical_gradient][audit][watom]")
{
    audit_beside_every_two_body("DM_mSQRT");
}

TEST_CASE("analytical gradient audit: DM_REAM RHOFORM SIGNED beside each "
          "two-body descriptor",
          "[HPO][analytical_gradient][audit][watom]")
{
    audit_beside_every_two_body("DM_REAM SIGNED");
}

TEST_CASE("analytical gradient audit: DM_REAM RHOFORM NONNEG beside each "
          "two-body descriptor",
          "[HPO][analytical_gradient][audit][watom]")
{
    audit_beside_every_two_body("DM_REAM NONNEG");
}

TEST_CASE("analytical gradient audit: DM_REAM RHOFORM MONOTONE beside each "
          "two-body descriptor",
          "[HPO][analytical_gradient][audit][watom]")
{
    audit_beside_every_two_body("DM_REAM MONOTONE");
}

TEST_CASE("analytical gradient audit: DM_PS_SQ beside each two-body "
          "descriptor",
          "[HPO][analytical_gradient][audit][watom]")
{
    audit_beside_every_two_body("DM_PS_SQ");
}

TEST_CASE("analytical gradient audit: a DESCRIPTOR block beside each "
          "two-body descriptor",
          "[HPO][analytical_gradient][audit][watom]")
{
    audit_beside_every_two_body("DESCRIPTOR");
}

TEST_CASE("analytical gradient audit: DM_mJoin of DM_mBlip and DM_REAM "
          "beside each two-body descriptor",
          "[HPO][analytical_gradient][audit][watom]")
{
    audit_beside_every_two_body("DM_mJoin lines");
}

TEST_CASE("analytical gradient audit: DM_mJoin of two DESCRIPTOR blocks "
          "beside each two-body descriptor",
          "[HPO][analytical_gradient][audit][watom]")
{
    audit_beside_every_two_body("DM_mJoin blocks");
}
