// test_nn_exact_e2e.cpp
//
// Exact neighbour lists through the HPO host:
//
//   * A validation-only host with a fixed potential (every weight pinned by
//     FIXINDEX/FIXWEIGHT, OPTIM FIXWEIGHT (1); no training DBFILE) gives the
//     same PC_ERMSE loss, and the same PC_FRMSE loss, on thin, triclinic,
//     sheared and unwrapped validation cells as on supercells of the same
//     crystals, built here with a test-local loop along the cell rows
//     (wrapped positions, every cell height >= the cutoff). The two metrics
//     are evaluated separately: the energy RMSE (meV/atom) would otherwise
//     hide the force RMSE (eV/A) at the tolerance. Each structure is a DBFILE
//     of its own: PC_FRMSE pools the force components of a DBFILE, so
//     structures replicated by different factors would change its weights,
//     while a single structure's RMSE does not change when every one of its
//     atoms is repeated. The reference energies of a supercell are scaled
//     here (extensive), its forces copied. All cells together and each cell
//     alone (to name a failing one) are compared.
//     A second SECTION evaluates an interior RCUT2B of an OPTIM RCUT2B box:
//     the lists are built once at the box top (detail::nn_safe_rcutmax) and
//     the descriptors filter them at the candidate's cutoff.
//   * The one-time lists cover every cutoff of the OPTIM box: a candidate
//     on the box top (OPTIM RCUT2B box [4, 5], evaluated at 5) has the loss
//     of the same host with the box [4, 6].
//   * A candidate whose cutoff lies beyond the OPTIM box (box [4, 5],
//     evaluated at 5.5, as an optimiser that ignores the bounds proposes)
//     is not evaluated (box_penalty.h): its
//     loss is the box's score of its distance, 0.5 of the box's width, no
//     list is built for it, and the journal's DIAGNOSTICS count it ("Out-of-
//     box points"). So with a trained host (a training DBFILE, every weight
//     fitted by OALGO 4 with a ridge, three OPTIM dimensions): the
//     candidate is not trained, and the analytical gradient at it hands
//     every parameter to finite differences (it has no fit to
//     differentiate). A candidate far beyond the box (RCUT2B 100 A on 6.2 A
//     cells, more than 10^4 image cells) is scored the same way: NNFinder is
//     not asked for its lists. And so in a run that reads no dataset (no
//     training DBFILE, no --validation; the loss from PC_OBSERVABLE).
//   * Every loss compared is a real one (finite, positive, below 1e10): a
//     failed evaluation returns FAIL_SCORE (DBL_MAX without a FAILSCORE
//     line), and two failures would compare equal.
//
// Before the exact-image fix the original cells lose periodic images (the
// image bound ignored the atoms' offsets inside the cell, sheared
// representations were searched in their given basis, unwrapped atoms were
// searched from their raw positions). Before the box fix a candidate beyond
// the box was evaluated with neighbour lists rebuilt at its cutoff (counted
// as "Neighbour lists rebuilt"), and RCUT2B 100 failed on NNFinder's
// refusal of its lists.
//
// Hidden [nn_e2e][.] like the other HPO_Host end-to-end groups: operator()
// fills the thread-local trainer cache and the HPOLog / HPO_Host statics;
// the group is one CTest entry (hpo_nn_e2e, tests/CMakeLists.txt):
//   ./test_hpo "[nn_e2e]"

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/element.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/box_penalty.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/startup_validation.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/structure.h>

#include <dlib/matrix.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using tadah::core::Context;
using tadah::core::PeriodicTable;
using tadah::hpo::HPOLog;
using tadah::hpo::HPOTargets;
using tadah::hpo::validate_hpo_setup;
using tadah::mlip::Atom;
using tadah::mlip::Structure;
using ParamVec = dlib::matrix<double, 0, 1>;
using Host     = tadah::hpo::HPO_Host<ParamVec, ::LammpsRunner>;

/* @p stem, this process id and a steady-clock count: distinct across the
 * parallel ctest processes. */
std::string unique_name(const std::string& stem) {
    return stem + std::to_string(::getpid()) + "_" +
           std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count());
}

/* A temporary working directory, the cwd while the object lives (the host
 * writes its per-evaluation files and best_pot.tadah there). */
struct ScratchDir {
    fs::path dir, original_cwd;
    ScratchDir() {
        dir = fs::temp_directory_path() / unique_name("tadah_nn_e2e_");
        fs::create_directories(dir);
        original_cwd = fs::current_path();
        fs::current_path(dir);
    }
    ~ScratchDir() {
        std::error_code ec;
        fs::current_path(original_cwd);
        fs::remove_all(dir, ec);
    }
};

/* hpo_run.log in a temporary file, read back after finish(). */
struct Journal {
    fs::path path;
    Journal() : path(fs::temp_directory_path() / (unique_name("tadah_nn_e2e_log_") + ".log")) {
        if (HPOLog::instance().is_open()) HPOLog::instance().finish("closed");
        HPOLog::instance().reset_for_test();
        HPOLog::instance().open(path.string(), "test", "test_cwd", 0);
    }
    ~Journal() {
        if (HPOLog::instance().is_open()) HPOLog::instance().finish("closed");
        HPOLog::instance().reset_for_test();
        std::error_code ec;
        fs::remove(path, ec);
    }
    std::string finish() {
        HPOLog::instance().finish("TEST");
        std::ifstream is(path);
        std::ostringstream oss;
        oss << is.rdbuf();
        return oss.str();
    }
};

/* ---------------------------------------------------------------------------
 * Cells and supercells, all in the test's own arithmetic (no production
 * transformation or neighbour code). Row k of Structure::cell is the lattice
 * vector a_k, a position is r = f0 a0 + f1 a1 + f2 a2.
 * ------------------------------------------------------------------------- */
using Rows = std::array<std::array<double, 3>, 3>;
using Frac = std::array<double, 3>;
using Offset = std::array<int, 3>;

/* r = f . A */
std::array<double, 3> cartesian(const Rows& a, const Frac& f) {
    std::array<double, 3> r{};
    for (int c = 0; c < 3; ++c)
        r[c] = f[0] * a[0][c] + f[1] * a[1][c] + f[2] * a[2][c];
    return r;
}

/* Deterministic reference forces of atom @p i of cell @p cell_id: arbitrary
 * but distinct values, so the force RMSE measures every predicted
 * component. */
std::array<double, 3> reference_force(int cell_id, std::size_t i) {
    std::array<double, 3> f{};
    for (int k = 0; k < 3; ++k)
        f[k] = 0.3 * std::sin(1.7 * static_cast<double>(i) +
                              0.9 * static_cast<double>(k) +
                              0.37 * static_cast<double>(cell_id));
    return f;
}

/* A cell with its rows, the fractional positions of its atoms plus integer
 * lattice offsets (unwrapped atoms; none when @p offsets is empty), a
 * reference energy of -5 eV per atom and reference_force() forces. */
Structure make_cell(const std::string& label, int cell_id, const Rows& a,
                    const std::vector<Frac>& fracs,
                    const std::vector<Offset>& offsets = {}) {
    REQUIRE((offsets.empty() || offsets.size() == fracs.size()));
    Structure st;
    st.label = label;
    st.cell.load(a[0][0], a[0][1], a[0][2],
                 a[1][0], a[1][1], a[1][2],
                 a[2][0], a[2][1], a[2][2]);
    const tadah::core::Element ta = PeriodicTable::find_by_symbol("Ta");
    for (std::size_t i = 0; i < fracs.size(); ++i) {
        Frac f = fracs[i];
        if (!offsets.empty())
            for (int k = 0; k < 3; ++k) f[k] += offsets[i][k];
        const auto r = cartesian(a, f);
        const auto F = reference_force(cell_id, i);
        st.add_atom(Atom(ta, r[0], r[1], r[2], F[0], F[1], F[2]));
    }
    st.energy = -5.0 * static_cast<double>(st.natoms());
    return st;
}

Rows rows_of(const Structure& st) {
    Rows a{};
    for (int k = 0; k < 3; ++k)
        for (int c = 0; c < 3; ++c) a[k][c] = st.cell(k, c);
    return a;
}

/* @p st with every atom moved into the cell, f_k in [0, 1): fractional
 * coordinates by the adjugate of the row matrix (the test's own inverse). */
Structure wrapped(const Structure& st) {
    const Rows a = rows_of(st);
    const double det =
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
        a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
        a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    REQUIRE(det != 0.0);
    /* f = r . A^-1, A^-1 = adj(A) / det; column k of A^-1 is b*_k. */
    double inv[3][3];
    inv[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) / det;
    inv[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) / det;
    inv[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) / det;
    inv[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) / det;
    inv[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) / det;
    inv[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) / det;
    inv[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) / det;
    inv[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) / det;
    inv[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) / det;
    Structure out = st;
    for (auto& at : out.atoms) {
        Frac f{};
        for (int k = 0; k < 3; ++k)
            f[k] = at.position[0] * inv[0][k] + at.position[1] * inv[1][k] +
                   at.position[2] * inv[2][k];
        for (int k = 0; k < 3; ++k) {
            f[k] -= std::floor(f[k]);
            if (f[k] >= 1.0) f[k] = 0.0;   // -tiny rounds up to 1.0
        }
        const auto r = cartesian(a, f);
        for (int c = 0; c < 3; ++c) at.position[c] = r[c];
    }
    return out;
}

/* Supercell along the rows: row k scaled by n_k; copy (x, y, z) of atom i at
 * r_i + x a0 + y a1 + z a2 with the forces of atom i; the reference energy
 * scaled by the number of copies. */
Structure supercell(const Structure& st, int n0, int n1, int n2) {
    const Rows a = rows_of(st);
    Structure out = st;
    out.label = st.label + " supercell";
    out.atoms.clear();
    out.nn.reset();
    const int n[3] = {n0, n1, n2};
    for (int k = 0; k < 3; ++k)
        for (int c = 0; c < 3; ++c) out.cell(k, c) = a[k][c] * n[k];
    for (int x = 0; x < n0; ++x)
        for (int y = 0; y < n1; ++y)
            for (int z = 0; z < n2; ++z)
                for (const Atom& at : st.atoms) {
                    Atom copy = at;
                    for (int c = 0; c < 3; ++c)
                        copy.position[c] = at.position[c] + x * a[0][c] +
                                           y * a[1][c] + z * a[2][c];
                    out.atoms.push_back(copy);
                }
    out.energy = st.energy * static_cast<double>(n0 * n1 * n2);
    return out;
}

/* One validation case: the original cell (host A) and its thick, wrapped
 * supercell (host B). */
struct Case {
    std::string name;
    Structure original;
    Structure reference;
};

/* The four cells of the design, at a cutoff of 10 A:
 *   - 2-atom bcc a = 3.41: every height 3.41 < 10; the body-centre
 *     neighbours at (-2.5, 0.5, 0.5) a need the image -3 a0;
 *   - a triangular (hexagonal) cell a = 4, c = 3.6 with 3 atoms at general
 *     positions: in-plane heights 3.46, a triclinic representation;
 *   - the sheared representation (4,0,0),(12,4,0),(0,0,4) of a simple
 *     cubic lattice, 2 atoms: its a0 height is 1.26 while the lattice's is
 *     4 (a lattice-reduction case);
 *   - the triangular cell again with its atoms displaced by different
 *     lattice vectors (unwrapped positions, some of them negative).
 * The supercells make every height >= 10: 3x3x3, 3x3x3, 8x3x3 and the
 * triangular cell's 3x3x3 built from the wrapped positions. */
std::vector<Case> make_cases() {
    const double s3 = std::sqrt(3.0);
    const Rows bcc_rows = {{{3.41, 0.0, 0.0}, {0.0, 3.41, 0.0}, {0.0, 0.0, 3.41}}};
    const Rows tri_rows = {{{4.0, 0.0, 0.0}, {2.0, 2.0 * s3, 0.0}, {0.0, 0.0, 3.6}}};
    const Rows shear_rows = {{{4.0, 0.0, 0.0}, {12.0, 4.0, 0.0}, {0.0, 0.0, 4.0}}};
    const std::vector<Frac> tri_fracs = {
        {0.02, 0.05, 0.10}, {0.36, 0.68, 0.55}, {0.71, 0.29, 0.83}};

    std::vector<Case> cases;
    {
        Structure st = make_cell("bcc a=3.41", 1, bcc_rows,
                                 {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}});
        cases.push_back({"bcc a=3.41, 2 atoms", st, supercell(st, 3, 3, 3)});
    }
    {
        Structure st = make_cell("triangular", 2, tri_rows, tri_fracs);
        cases.push_back({"triangular a=4 c=3.6, 3 atoms", st, supercell(st, 3, 3, 3)});
    }
    {
        Structure st = make_cell("sheared", 3, shear_rows,
                                 {{0.1, 0.2, 0.3}, {0.6, 0.45, 0.8}});
        cases.push_back({"sheared (4,0,0),(12,4,0),(0,0,4), 2 atoms", st,
                         supercell(st, 8, 3, 3)});
    }
    {
        Structure st = make_cell("triangular unwrapped", 4, tri_rows, tri_fracs,
                                 {{2, -3, 1}, {-1, 4, -2}, {5, 0, -3}});
        cases.push_back({"triangular with lattice offsets (unwrapped)", st,
                         supercell(wrapped(st), 3, 3, 3)});
    }
    return cases;
}

/* 2x2x2 bcc a = 3.1 (16 atoms), shifted into the cell and rattled
 * deterministically (@p phase varies the rattle): every height 6.2 and the
 * atoms wrapped, so its lists at 5, 5.5 and 6 hold every pair within them
 * (no image question here), and it has the shells 5.14 and 5.37 between the
 * list cutoff 5 of an OPTIM RCUT2B box [4, 5] and a candidate's 5.5. */
Structure rattled_bcc(const std::string& label, int cell_id, double phase) {
    const double a = 3.1;
    const Rows rows = {{{2 * a, 0.0, 0.0}, {0.0, 2 * a, 0.0}, {0.0, 0.0, 2 * a}}};
    std::vector<Frac> fracs;
    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
            for (int z = 0; z < 2; ++z)
                for (double h : {0.0, 0.25}) {
                    const std::size_t i = fracs.size();
                    Frac f{};
                    const int n[3] = {x, y, z};
                    for (int k = 0; k < 3; ++k)
                        f[k] = 0.5 * n[k] + h + 0.02 +
                               0.004 * std::sin(2.3 * static_cast<double>(i) +
                                                1.1 * static_cast<double>(k) +
                                                phase);
                    fracs.push_back(f);
                }
    Structure st = make_cell(label, cell_id, rows, fracs);
    REQUIRE(st.natoms() == 16u);
    return st;
}

/* The pinned weights of the fixed potential. */
const std::array<double, 5> kFixedWeights = {0.8, -0.35, 0.2, -0.12, 0.07};

/* The potential keys every host here shares: D2_BP (5 Gaussians, Cut_Cos)
 * + DM_Dummy with a linear model (BF_Linear), no bias, raw features. FORCE
 * true for PC_FRMSE. */
Context potential_context(double rcut2b, const std::string& cgrid,
                          const std::string& sgrid, double rcutmb) {
    Context c;
    c.add("ATOM",     "Ta");
    c.add("WATOM",    1);
    c.add("INIT2B",   "true");
    c.add("INITMB",   "false");
    c.add("TYPE2B",   "D2_BP");
    c.add("TYPE2B",   static_cast<int>(5));
    c.add("TYPE2B",   static_cast<int>(5));
    c.add("TYPE2B",   "Ta");
    c.add("TYPE2B",   "Ta");
    c.add("CGRID2B",  cgrid);
    c.add("SGRID2B",  sgrid);
    c.add("RCTYPE2B", "Cut_Cos");
    c.add("RCUT2B",   rcut2b);
    c.add("TYPEMB",   "DM_Dummy");
    c.add("TYPEMB",   "Ta");
    c.add("TYPEMB",   "Ta");
    c.add("RCTYPEMB", "Cut_Cos");
    c.add("RCUTMB",   rcutmb);
    c.add("DIMER",    "false");
    c.add("DIMER",    0);
    c.add("DIMER",    "false");
    c.add("FORCE",    "true");
    c.add("STRESS",   "false");
    c.add("ESIGMA", 1.0);
    c.add("FSIGMA", 0.1);
    c.add("FSIGMA", 0.0);
    c.add("SSIGMA", 0.5);
    c.add("SSIGMA", 0.0);
    c.add("BIAS",     "false");
    c.add("NORM",     "false");
    c.add("VERBOSE",  0);
    c.add("MODEL",    "M_BLR");
    c.add("MODEL",    "BF_Linear");
    return c;
}

/* The fixed potential: every weight pinned (FIXINDEX 1-5), so it is not
 * trained. */
void pin_weights(Context& c) {
    c.add("LAMBDA",   0);
    c.add("OALGO",    1);
    c.add("FIXINDEX", "1-5");
    for (double w : kFixedWeights) c.add("FIXWEIGHT", w);
}

/* D2_BP (5 Gaussians, Cut_Cos) + DM_Dummy, every weight pinned: the
 * potential is fixed, the loss is a function of the validation data only.
 * FORCE true for PC_FRMSE. */
Context make_context(double rcut2b, const std::string& cgrid,
                     const std::string& sgrid, const fs::path& hpo_cfg,
                     const fs::path& val_cfg, double rcutmb) {
    Context c = potential_context(rcut2b, cgrid, sgrid, rcutmb);
    pin_weights(c);
    c.add("VALIDATION", val_cfg.string());
    c.add("HPOTARGET",  hpo_cfg.string());
    return c;
}

/* A derivative-free optimiser block: the cases call the objective directly
 * and never run the optimiser. */
constexpr const char* kOptimizerBlock =
    "OPTIMIZER\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 1\nENDOPTIMIZER\n";

/* Writes each structure to a DBFILE of its own (dump_to_file: 18
 * significant digits, lossless) and the --validation file listing them. */
void write_validation(const fs::path& val_cfg, const std::string& tag,
                      const std::vector<Structure>& sts) {
    std::ofstream cfg(val_cfg);
    REQUIRE(cfg.good());
    for (std::size_t k = 0; k < sts.size(); ++k) {
        const fs::path db = val_cfg.parent_path() /
                            (tag + "_" + std::to_string(k + 1) + ".db");
        {
            std::ofstream os(db);
            REQUIRE(os.good());
            sts[k].dump_to_file(os);
        }
        cfg << "DBFILE " << db.string() << "\n";
    }
}

/* The fixed pair potential of a case: D2_BP with these grids and cutoff,
 * DM_Dummy at rcutmb. */
struct Potential {
    double rcut2b;
    std::string cgrid, sgrid;
    double rcutmb;
};

/* The validation metrics, one per evaluation: the energy RMSE (meV/atom)
 * and the force RMSE (eV/A) differ by orders of magnitude, so a loss that
 * summed them would hide the forces at the tolerance below. */
const std::vector<std::pair<std::string, std::string>> kMetrics = {
    {"energy RMSE", "PC_ERMSE 0 1\n"},
    {"force RMSE",  "PC_FRMSE 0 1\n"}};

/* A loss an evaluation computed: finite, positive and far below the
 * FAIL_SCORE a failed evaluation returns (DBL_MAX: the configs here have
 * no FAILSCORE line, and HPO_Host::operator() turns any throw into it), so
 * that two failed evaluations cannot pass as equal losses. */
void require_real_loss(double loss) {
    REQUIRE(std::isfinite(loss));
    REQUIRE(loss > 0.0);
    REQUIRE(loss < 1e10);
}

/* The loss of one evaluation of a fresh validation-only host at the point
 * @p p0 of the single OPTIM dimension @p optim_line, with the validation
 * metric @p metric_line. With @p journal set, hpo_run.log is open for the
 * host and its text is returned there. */
double validation_loss(const fs::path& dir, const std::string& tag,
                       const std::vector<Structure>& sts,
                       const std::string& metric_line,
                       const std::string& optim_line, double p0,
                       const Potential& pot, std::string* journal = nullptr) {
    const fs::path val_cfg = dir / (tag + ".val");
    const fs::path hpo_cfg = dir / (tag + ".hpo");
    write_validation(val_cfg, tag, sts);
    {
        std::ofstream os(hpo_cfg);
        REQUIRE(os.good());
        os << kOptimizerBlock << "LOSS L2\n" << metric_line << optim_line;
    }

    Host::reset_globals_for_test();
    Context ctx = make_context(pot.rcut2b, pot.cgrid, pot.sgrid, hpo_cfg,
                               val_cfg, pot.rcutmb);
    std::optional<Journal> j;
    if (journal) j.emplace();
    HPOTargets trg(ctx, hpo_cfg.string());
    REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, false, true));
    double loss = 0.0;
    {
        Host host(ctx, trg, val_cfg.string());
        REQUIRE(host.train_size() == 0u);
        REQUIRE(host.val_size() == sts.size());
        ParamVec p(1);
        p(0) = p0;
        REQUIRE_NOTHROW(loss = host(p));
        host.flush_remaining();
    }
    if (journal) *journal = j->finish();
    Host::reset_globals_for_test();
    return loss;
}

/* For each metric: the loss on every original cell (one DBFILE each)
 * equals the loss on their supercells, and so does each cell's alone (the
 * failing cell is named). */
void check_supercell_losses(const fs::path& dir, const std::string& tag,
                            const std::vector<Case>& cases,
                            const std::string& optim_line, double p0,
                            const Potential& pot) {
    std::vector<Structure> originals, references;
    for (const Case& c : cases) {
        originals.push_back(c.original);
        references.push_back(c.reference);
    }
    for (std::size_t m = 0; m < kMetrics.size(); ++m) {
        const std::string& metric = kMetrics[m].first;
        const std::string& line = kMetrics[m].second;
        const std::string t = tag + std::to_string(m);
        {
            CAPTURE(metric);
            const double la = validation_loss(dir, t + "a", originals, line,
                                              optim_line, p0, pot);
            const double lb = validation_loss(dir, t + "b", references, line,
                                              optim_line, p0, pot);
            CAPTURE(la, lb);
            require_real_loss(la);
            require_real_loss(lb);
            CHECK(la == Approx(lb).epsilon(1e-12));
        }
        for (std::size_t k = 0; k < cases.size(); ++k) {
            CAPTURE(metric, cases[k].name);
            const std::string tk = t + "_" + std::to_string(k);
            const double la = validation_loss(dir, tk + "a", {cases[k].original},
                                              line, optim_line, p0, pot);
            const double lb = validation_loss(dir, tk + "b", {cases[k].reference},
                                              line, optim_line, p0, pot);
            CAPTURE(la, lb);
            require_real_loss(la);
            require_real_loss(lb);
            CHECK(la == Approx(lb).epsilon(1e-12));
        }
    }
}

/* The DIAGNOSTICS section of a journal: from its header to the next
 * section header. */
std::string diagnostics_section(const std::string& journal) {
    const auto at = journal.find("-- DIAGNOSTICS");
    if (at == std::string::npos) return {};
    const auto end = journal.find("\n-- ", at + 1);
    return journal.substr(at, end == std::string::npos ? std::string::npos
                                                       : end - at);
}

/* The first line of @p text holding @p needle ("" when none does). */
std::string line_with(const std::string& text, const std::string& needle) {
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line))
        if (line.find(needle) != std::string::npos) return line;
    return {};
}

/* The trained hosts' potential: RCUT2B 4.5 inside the OPTIM boxes, RCUTMB 4
 * below them (the lists are built at the RCUT2B box top), and Gaussians out
 * to 5.3, so the shells between 5 and 5.5 would enter the fit at a
 * candidate RCUT2B of 5.5 if it were trained. */
const Potential kTrainedPot{4.5, "2.5 3.2 3.9 4.6 5.3", "1.0 1.0 1.0 1.0 1.0",
                            4.0};

/* The OPTIM lines of a trained host, RCUT2B in the box [4, @p rcut_top].
 * The parameter vector follows HPOTargets::optimap, ordered by key: p(0) is
 * the fifth Gaussian's centre (CGRID2B), p(1) RCUT2B and p(2) the fifth
 * Gaussian's width (SGRID2B). Neither grid is a knot channel, so the lists
 * are built at the RCUT2B box top alone (detail::nn_safe_rcutmax). */
std::string trained_optim_lines(double rcut_top) {
    std::ostringstream os;
    os << "OPTIM CGRID2B (5) 4.8 5.4\n"
       << "OPTIM RCUT2B (1) 4.0 " << rcut_top << "\n"
       << "OPTIM SGRID2B (5) 0.5 2.0\n";
    return os.str();
}

/* A host with training data (one DBFILE, every weight fitted by the normal
 * equations with a ridge: OALGO 4, LAMBDA 1e-3) and a validation DBFILE,
 * PC_ERMSE and PC_FRMSE: the setup the analytical gradient covers
 * (analytical_gate.h). Both structures are rattled_bcc() cells with
 * reference forces of their own. @p body runs on the host while hpo_run.log
 * is open; the journal's text is returned. */
template <class Body>
std::string with_trained_host(const fs::path& dir, const std::string& tag,
                              double rcut_top, Body&& body) {
    const fs::path train_db = dir / (tag + "_train.db");
    const fs::path val_cfg = dir / (tag + ".val");
    const fs::path hpo_cfg = dir / (tag + ".hpo");
    {
        std::ofstream os(train_db);
        REQUIRE(os.good());
        rattled_bcc("bcc a=3.1 2x2x2 training", 7, 0.8).dump_to_file(os);
    }
    write_validation(val_cfg, tag,
                     {rattled_bcc("bcc a=3.1 2x2x2 validation", 8, 1.9)});
    {
        std::ofstream os(hpo_cfg);
        REQUIRE(os.good());
        os << "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\n"
              "GRAD_STRATEGY HYBRID\nENDOPTIMIZER\n"
           << "LOSS L2\nPC_ERMSE 0 1\nPC_FRMSE 0 1\n"
           << trained_optim_lines(rcut_top);
    }

    Host::reset_globals_for_test();
    Context ctx = potential_context(kTrainedPot.rcut2b, kTrainedPot.cgrid,
                                    kTrainedPot.sgrid, kTrainedPot.rcutmb);
    ctx.add("LAMBDA",     1e-3);
    ctx.add("OALGO",      4);
    ctx.add("DBFILE",     train_db.string());
    ctx.add("VALIDATION", val_cfg.string());
    ctx.add("HPOTARGET",  hpo_cfg.string());
    Journal j;
    HPOTargets trg(ctx, hpo_cfg.string());
    REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, true, true));
    {
        Host host(ctx, trg, val_cfg.string());
        REQUIRE(host.train_size() == 1u);
        REQUIRE(host.val_size() == 1u);
        body(host);
        host.flush_remaining();
    }
    const std::string journal = j.finish();
    Host::reset_globals_for_test();
    return journal;
}

/* The trained hosts' parameter vector (trained_optim_lines order). */
ParamVec trained_point(double cgrid5, double rcut2b, double sgrid5) {
    ParamVec p(3);
    p(0) = cgrid5;
    p(1) = rcut2b;
    p(2) = sgrid5;
    return p;
}

}  // namespace

TEST_CASE("nn e2e: thin triclinic and unwrapped validation cells give the "
          "loss of their supercells", "[nn_e2e][.]")
{
    PeriodicTable::initialize();
    ScratchDir sd;
    const std::vector<Case> cases = make_cases();

    /* Gaussians out to the 10 A cutoff, so the loss sees every shell. */
    const Potential pot{10.0, "1.5 3.5 5.5 7.5 9.0", "0.4 0.4 0.4 0.4 0.4", 5.0};

    SECTION("a fixed cutoff") {
        /* FIXWEIGHT (1) evaluated at its pinned value. */
        check_supercell_losses(sd.dir, "fix", cases,
                               "OPTIM FIXWEIGHT (1) -1.0 1.0\n", 0.8, pot);
    }

    SECTION("an interior RCUT2B of an OPTIM RCUT2B box") {
        /* The lists are built once at the box top, 10, and the descriptors
         * filter them at the candidate's 9.3. */
        check_supercell_losses(sd.dir, "rc", cases,
                               "OPTIM RCUT2B (1) 8.0 10.0\n", 9.3, pot);
    }
}

TEST_CASE("nn e2e: the one-time lists cover every cutoff of the OPTIM box, "
          "and a candidate cutoff beyond the box is not evaluated",
          "[nn_e2e][.]")
{
    PeriodicTable::initialize();
    ScratchDir sd;

    /* Every height 6.2 >= 6: its lists at 5 and at 6 hold every pair within
     * them, and the shells 5.14 and 5.37 lie between the list cutoff 5 of
     * the box [4, 5] and the candidate's 5.5. */
    const Structure st = rattled_bcc("bcc a=3.1 2x2x2 rattled", 5, 0.0);

    /* RCUT2B 4.5 lies inside both boxes; RCUTMB 4 below them, so the lists
     * are built at the box top. */
    const Potential pot{4.5, "2.5 3.2 3.9 4.6 5.3", "1.0 1.0 1.0 1.0 1.0", 4.0};

    /* Pins: at the box top 5 the loss of the box [4, 5] equals that of the
     * box [4, 6], and neither run has an out-of-box point. At 5.5 the box
     * [4, 5] scores out_of_box_score(DBL_MAX, 0.5) -- its distance is half
     * the box's width -- without a "Neighbour lists rebuilt" line, with
     * DIAGNOSTICS "Out-of-box points : 1" and the WARNING "eval 1: RCUT2B[1]
     * = 5.5 lies outside its OPTIM box [4, 5]: it is not evaluated".
     * Before the fix: at 5.5 the box [4, 5] gave the loss of the box [4, 6]
     * (evaluated on lists rebuilt at 5.5, "Neighbour lists rebuilt : 1"). */
    for (std::size_t m = 0; m < kMetrics.size(); ++m) {
        const std::string& metric = kMetrics[m].first;
        const std::string& line = kMetrics[m].second;
        CAPTURE(metric);

        std::string journal_top;
        const double loss_top = validation_loss(
            sd.dir, "top" + std::to_string(m), {st}, line,
            "OPTIM RCUT2B (1) 4.0 5.0\n", 5.0, pot, &journal_top);
        const double loss_top_long = validation_loss(
            sd.dir, "toplong" + std::to_string(m), {st}, line,
            "OPTIM RCUT2B (1) 4.0 6.0\n", 5.0, pot);
        CAPTURE(loss_top, loss_top_long);
        require_real_loss(loss_top);
        require_real_loss(loss_top_long);
        CHECK(loss_top == Approx(loss_top_long).epsilon(1e-12));
        CHECK(journal_top.find("Out-of-box points") == std::string::npos);

        std::string journal_out;
        const double loss_out = validation_loss(
            sd.dir, "out" + std::to_string(m), {st}, line,
            "OPTIM RCUT2B (1) 4.0 5.0\n", 5.5, pot, &journal_out);
        CAPTURE(loss_out);
        CHECK(loss_out ==
              tadah::hpo::out_of_box_score(
                  std::numeric_limits<double>::max(), 0.5));
        INFO(journal_out);
        const std::string diag = diagnostics_section(journal_out);
        REQUIRE_FALSE(diag.empty());
        CHECK(diag.find("Neighbour lists rebuilt") == std::string::npos);
        CHECK(line_with(diag, "Out-of-box points").find(": 1 ") !=
              std::string::npos);
        CHECK(journal_out.find("= 5.5 lies outside its OPTIM box [4, 5]: it "
                               "is not evaluated") != std::string::npos);
    }
}

TEST_CASE("nn e2e: a trained candidate beyond the OPTIM box is neither "
          "trained nor differentiated", "[nn_e2e][.]")
{
    PeriodicTable::initialize();
    ScratchDir sd;

    /* The fifth Gaussian at 5.1 with width 1.2 (inside their boxes) and
     * RCUT2B 5.5 (0.5 of the width beyond the box [4, 5]) or 100 (95 of it:
     * on the 6.2 A cells NNFinder would need (2 N + 1)^3 > 10^4 image cells
     * and refuse the lists). */
    const ParamVec inside = trained_point(5.1, 4.5, 1.2);
    struct Leg { const char* tag; double rcut2b, distance; };
    for (const Leg& leg : {Leg{"out", 5.5, 0.5}, Leg{"far", 100.0, 95.0}}) {
        DYNAMIC_SECTION("RCUT2B " << leg.rcut2b) {
            const ParamVec p = trained_point(5.1, leg.rcut2b, 1.2);
            double loss = 0.0;
            bool ok = false;
            ParamVec g;
            std::vector<int> needs_fd;
            std::string gap;
            const std::string journal = with_trained_host(
                sd.dir, leg.tag, 5.0, [&](Host& host) {
                    /* An optimiser's first steps: the cached trainer is
                     * trained at a point inside the box and differentiated
                     * analytically. */
                    double loss_inside = 0.0;
                    REQUIRE_NOTHROW(loss_inside = host(inside));
                    require_real_loss(loss_inside);
                    ParamVec g_inside;
                    std::vector<int> fd_inside;
                    REQUIRE(host.analytical_gradient(inside, g_inside,
                                                     fd_inside));
                    REQUIRE(fd_inside.empty());

                    REQUIRE_NOTHROW(loss = host(p));
                    REQUIRE_NOTHROW(ok = host.analytical_gradient(p, g,
                                                                   needs_fd));
                    gap = host.analytical_gradient_gap();
                });
            /* Pins: the box's score of the distance, no training (the
             * gradient has no fit to differentiate: every parameter is
             * handed to finite differences, which GradientProvider replaces
             * by the score's slope at such a point), no list asked of
             * NNFinder, DIAGNOSTICS "Out-of-box points : 1".
             * Before the fix: RCUT2B 5.5 was trained, validated and
             * differentiated on lists rebuilt at 5.5 (a real loss, no
             * parameter handed to finite differences); RCUT2B 100 failed
             * (DBL_MAX) on NNFinder's refusal, "the cell is too small for
             * the cutoff 100 A". */
            CAPTURE(loss);
            CHECK(loss == tadah::hpo::out_of_box_score(
                              std::numeric_limits<double>::max(),
                              leg.distance));
            REQUIRE(ok);
            std::vector<int> fd = needs_fd;
            std::sort(fd.begin(), fd.end());
            CHECK(fd == std::vector<int>{0, 1, 2});
            CHECK(gap.find("outside the OPTIM box") != std::string::npos);
            INFO(journal);
            CHECK(journal.find("the cell is too small for the cutoff") ==
                  std::string::npos);
            const std::string diag = diagnostics_section(journal);
            REQUIRE_FALSE(diag.empty());
            CHECK(diag.find("Neighbour lists rebuilt") == std::string::npos);
            CHECK(line_with(diag, "Out-of-box points").find(": 1 ") !=
                  std::string::npos);
        }
    }
}

TEST_CASE("nn e2e: a run that reads no dataset does not evaluate a "
          "candidate beyond the OPTIM box either", "[nn_e2e][.]")
{
    PeriodicTable::initialize();
    ScratchDir sd;

    /* No training DBFILE, no --validation, every weight pinned: the loss is
     * the cohesive energy of a bcc crystal the observable relaxes with its
     * own neighbour lists. OPTIM RCUT2B box [4, 5], evaluated at its top 5
     * and at 5.5. */
    const fs::path hpo_cfg = sd.dir / "observable.hpo";
    {
        /* GRAD_STRATEGY HYBRID: an unset one resolves to ANALYTICAL with
         * PC_OBSERVABLE, which needs training data (validate_hpo_setup). */
        std::ofstream os(hpo_cfg);
        REQUIRE(os.good());
        os << "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\n"
              "GRAD_STRATEGY HYBRID\nENDOPTIMIZER\n"
           << "LOSS L2\nFAILSCORE 1e8\n"
           << "PC_OBSERVABLE cohesive_energy --species Ta --target 8.1 "
              "--weight 1 --supercell 2 2 2 --maxiter 50 --maxeval 200 "
              "--hotvar a0 2.7 3.9 3.3\n"
           << "OPTIM RCUT2B (1) 4.0 5.0\n";
    }

    Host::reset_globals_for_test();
    Context ctx = potential_context(kTrainedPot.rcut2b, kTrainedPot.cgrid,
                                    kTrainedPot.sgrid, kTrainedPot.rcutmb);
    pin_weights(ctx);
    ctx.add("HPOTARGET", hpo_cfg.string());
    Journal j;
    HPOTargets trg(ctx, hpo_cfg.string());
    REQUIRE_NOTHROW(validate_hpo_setup(ctx, trg, false, false));
    double model_cutoff = 0.0;
    double loss_top = 0.0, loss_out = 0.0;
    {
        Host host(ctx, trg, /*val_cfg=*/"");
        REQUIRE(host.train_size() == 0u);
        REQUIRE(host.val_size() == 0u);
        ParamVec p(1);
        p(0) = 5.0;
        REQUIRE_NOTHROW(loss_top = host(p));
        const auto* tr = Host::tls_trainer_for_test();
        REQUIRE(tr != nullptr);
        model_cutoff = tr->dc.neighbour_cutoff();
        p(0) = 5.5;
        REQUIRE_NOTHROW(loss_out = host(p));
        host.flush_remaining();
    }
    const std::string journal = j.finish();
    Host::reset_globals_for_test();

    /* Pins: the box top is evaluated (a loss below FAILSCORE 1e8, the
     * model's cutoff 5); 5.5 is scored out_of_box_score(1e8, 0.5), with
     * DIAGNOSTICS "Out-of-box points : 1" and no rebuild line.
     * Before the fix: 5.5 was evaluated (a model cutoff of 5.5, its own
     * loss). */
    CAPTURE(loss_top, loss_out);
    CHECK(model_cutoff == 5.0);
    CHECK(std::isfinite(loss_top));
    CHECK(loss_top < 0.999e8);
    CHECK(loss_out == tadah::hpo::out_of_box_score(1e8, 0.5));
    INFO(journal);
    const std::string diag = diagnostics_section(journal);
    REQUIRE_FALSE(diag.empty());
    CHECK(diag.find("Neighbour lists rebuilt") == std::string::npos);
    CHECK(line_with(diag, "Out-of-box points").find(": 1 ") !=
          std::string::npos);
}
