// test_nn_exact_e2e.cpp
//
// Exact neighbour lists through the HPO host (workstream C,
// DESIGN-nnfinder-exact-images.md, section 5.3 and the C handover):
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
//   * A candidate whose cutoff exceeds the one-time lists (OPTIM RCUT2B box
//     [4, 5], evaluated at 5.5, e.g. proposed by an optimiser that ignores
//     the bounds) is evaluated with lists that cover it: its loss equals that
//     of the same host with the box [4, 6], and the journal's DIAGNOSTICS
//     section counts the rebuild ("Neighbour lists rebuilt").
//
// Before the exact-image fix the original cells lose periodic images (the
// image bound ignored the atoms' offsets inside the cell, sheared
// representations were searched in their given basis, unwrapped atoms were
// searched from their raw positions), and a candidate beyond the lists was
// scored with the short lists without a word.
//
// Hidden [nn_e2e][.] like the other HPO_Host end-to-end groups: operator()
// fills the thread-local trainer cache and the HPOLog / HPO_Host statics;
// the group is one CTest entry (hpo_nn_e2e, tests/CMakeLists.txt):
//   ./test_hpo "[nn_e2e]"

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/element.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/startup_validation.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/structure.h>

#include <dlib/matrix.h>

#include <unistd.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

/* A trainer built for this host: the thread-local trainer cache may still
 * hold another host's model. */
struct ForceRebuildGuard {
    bool restore;
    ForceRebuildGuard() {
        restore = std::getenv("TADAH_HPO_FORCE_REBUILD") == nullptr;
        if (restore) ::setenv("TADAH_HPO_FORCE_REBUILD", "1", 1);
    }
    ~ForceRebuildGuard() {
        if (restore) ::unsetenv("TADAH_HPO_FORCE_REBUILD");
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

/* D2_BP (5 Gaussians, Cut_Cos) + DM_Dummy, every weight pinned: the
 * potential is fixed, the loss is a function of the validation data only.
 * FORCE true for PC_FRMSE. */
Context make_context(double rcut2b, const std::string& cgrid,
                     const std::string& sgrid, const fs::path& hpo_cfg,
                     const fs::path& val_cfg, double rcutmb) {
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
    c.add("EWEIGHT",  1.0);
    c.add("FWEIGHT",  1.0);
    c.add("SWEIGHT",  1.0);
    c.add("BIAS",     "false");
    c.add("NORM",     "false");
    c.add("VERBOSE",  0);
    c.add("MODEL",    "M_BLR");
    c.add("MODEL",    "BF_Linear");
    c.add("LAMBDA",   0);
    c.add("OALGO",    1);
    c.add("FIXINDEX", "1-5");
    for (double w : {0.8, -0.35, 0.2, -0.12, 0.07}) c.add("FIXWEIGHT", w);
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
    ForceRebuildGuard rebuild;
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
            REQUIRE(std::isfinite(lb));
            REQUIRE(lb > 0.0);
            CHECK(la == Approx(lb).epsilon(1e-12));
        }
        for (std::size_t k = 0; k < cases.size(); ++k) {
            CAPTURE(metric, cases[k].name);
            const std::string tk = t + "_" + std::to_string(k);
            const double la = validation_loss(dir, tk + "a", {cases[k].original},
                                              line, optim_line, p0, pot);
            const double lb = validation_loss(dir, tk + "b", {cases[k].reference},
                                              line, optim_line, p0, pot);
            REQUIRE(lb > 0.0);
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

TEST_CASE("nn e2e: a candidate cutoff beyond the neighbour lists is "
          "evaluated with rebuilt lists", "[nn_e2e][.]")
{
    PeriodicTable::initialize();
    ScratchDir sd;

    /* 2x2x2 bcc a = 3.1 (16 atoms), shifted into the cell and rattled
     * deterministically: every height 6.2 >= 6 and the atoms wrapped, so its
     * lists at 5 and at 6 hold every pair within them (no image question
     * here), and it has the shells 5.14 and 5.37 between the list cutoff 5
     * of the box [4, 5] and the candidate's 5.5. */
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
                                                1.1 * static_cast<double>(k));
                    fracs.push_back(f);
                }
    const Structure st = make_cell("bcc a=3.1 2x2x2 rattled", 5, rows, fracs);
    REQUIRE(st.natoms() == 16u);

    /* RCUT2B 4.5 lies inside both boxes; RCUTMB 4 below them, so the lists
     * are built at the box top. */
    const Potential pot{4.5, "2.5 3.2 3.9 4.6 5.3", "1.0 1.0 1.0 1.0 1.0", 4.0};
    const double candidate = 5.5;

    for (std::size_t m = 0; m < kMetrics.size(); ++m) {
        const std::string& metric = kMetrics[m].first;
        const std::string& line = kMetrics[m].second;
        CAPTURE(metric);

        /* Box [4, 5]: lists at 5; the candidate at 5.5 is beyond them (an
         * optimiser that ignores the bounds proposed it). */
        std::string journal_a;
        const double loss_a = validation_loss(
            sd.dir, "short" + std::to_string(m), {st}, line,
            "OPTIM RCUT2B (1) 4.0 5.0\n", candidate, pot, &journal_a);

        /* Box [4, 6]: lists at 6 cover the candidate. */
        std::string journal_b;
        const double loss_b = validation_loss(
            sd.dir, "long" + std::to_string(m), {st}, line,
            "OPTIM RCUT2B (1) 4.0 6.0\n", candidate, pot, &journal_b);

        REQUIRE(std::isfinite(loss_b));
        REQUIRE(loss_b > 0.0);
        CHECK(loss_a == Approx(loss_b).epsilon(1e-12));

        /* The rebuild is counted in DIAGNOSTICS: one evaluation of one
         * structure in one DBFILE, so the count is 1 whatever it counts. */
        {
            INFO(journal_a);
            const std::string diag_a = diagnostics_section(journal_a);
            REQUIRE_FALSE(diag_a.empty());
            const std::string rebuilt = line_with(diag_a, "Neighbour lists rebuilt");
            CAPTURE(rebuilt);
            CHECK_FALSE(rebuilt.empty());
            const auto colon = rebuilt.find(':');
            CHECK(colon != std::string::npos);
            long count = -1;
            if (colon != std::string::npos) {
                std::istringstream count_in(rebuilt.substr(colon + 1));
                count_in >> count;
            }
            CHECK(count == 1);
        }

        /* A candidate inside the lists needs no rebuild. */
        {
            INFO(journal_b);
            const std::string diag_b = diagnostics_section(journal_b);
            REQUIRE_FALSE(diag_b.empty());
            CHECK(diag_b.find("Neighbour lists rebuilt") == std::string::npos);
        }
    }
}
