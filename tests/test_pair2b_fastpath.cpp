// test_pair2b_fastpath.cpp
//
// EAMFAST collapse fast paths against the generic kernels, driven end-to-end
// through LammpsRunner.
//
// Pure-2B (phi-only) collapse, compute_2b_fast vs compute_2b, on a rattled
// bcc Kr cell (D2_Knot5, 21 knots, Cut_Dummy, linear, INITMB false -- the Kr
// HPO champion):
//   - static energy/force-norm/pressure parity (EAMFAST true vs false),
//   - box/relax trajectory equivalence (relaxed a0),
//   - adjoint capture under the fast path (pure-2B falls back to the
//     generic kernel, no error, snapshot filled).
//
// Certified tables (the Ta D2_LJ potential of the Tadah! 1.6.0 quickstart,
// and D2_LJ + DM_REAM for compute_2b_mb_half_eamfast vs compute_2b_mb_half):
//   - energy, per-atom force and virial parity on a rattled bcc Ta cell
//     that holds pairs in the top table segment just below the Cut_Dummy
//     cutoff, to margins derived from the tables' certificates,
//   - the same with one extra atom 0.1 A from another, far below the
//     certified table range, where the fast kernels evaluate analytically,
//   - an extra atom 2.0 A from another, on a density feature narrower than
//     the rho table spacing and below the rho table's certified range: the
//     EAM kernel must take rho and rho' from eam_density() there,
//   - the load-time log line (read from a -log file) states the certified
//     range, the relative tolerance and the analytic clause,
//   - a potential whose phi or rho table has no certified range runs the
//     generic kernel (bitwise) and its log line says why.

#include "catch2/catch.hpp"

#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/runners.h>
#include <tadah/tadah.lammps/eam_collapse.h>
#include <tadah/tadah.lammps/lammps_tadah.h>

#include <atom.h>
#include <domain.h>
#include <force.h>
#include <pair.h>
#include <pair_tadah.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

/* Kr pure-2B champion (ROUND6): physical scale, mechanically stable bcc. */
const char* KR_POT_BODY =
    "ATOM              Kr\n"
    "BIAS              false\n"
    "CGRID2B           5.654180679071199478 5.411252065569847503 3.217331521801811256 3.930824515846677247 2.537779659971619584 2.250748637725905610 3.737568302103229101 2.292155276555360999 4.166118987621620029 3.895659538414396827 3.882828370167631160 5.104012407403072160 1.892622625426633620 1.861528429044043031 6.197172052531381148 6.089092122840964905 6.744489232873557505 7.244744465733268690 7.327761008518089092 8.274486017059761167 8.116941074816644175\n"
    "FORCE             false\n"
    "INIT2B            true\n"
    "INITMB            false\n"
    "LAMBDA            0\n"
    "MODEL             M_BLR   BF_Linear\n"
    "NORM              false\n"
    "RCTYPE2B          Cut_Dummy\n"
    "RCUT2B            8.274486017059761167\n"
    "STRESS            false\n"
    "TYPE2B            D2_Knot5   21   0   Kr   Kr\n"
    "WATOM             1.000000000000000000\n"
    "WEIGHTS           0.002022341035452596 -0.234834468777806615 -0.176955931611558204 -0.101273818977953803 -0.640091273014962159 -0.371627635837921522 -0.122730970028676681 -0.639591836790447621 0.889938055528681660 -0.501552854167752749 -0.306677826118867403 0.142464681149275907 -0.541937682987483349 -0.085416359679197792 0.085834571424044798 0.000135043203549349 0.000000000000000000 -0.011937057559490772 0.000000000000000000 0.000000000000000000 0.000000690371680869\n";

/* Pot files land in the shared ctest cwd, and each TEST_CASE runs as its
 * own concurrent process under `ctest -j` -- the name must be unique per
 * process or parallel cases clobber each other's pots mid-parse. */
std::string write_pot(bool eamfast) {
    const std::string path = "pair2b_" + std::to_string(::getpid())
                           + (eamfast ? "_on.tadah" : "_off.tadah");
    std::ofstream ofs(path);
    ofs << KR_POT_BODY;
    ofs << "EAMFAST           " << (eamfast ? "true" : "false") << "\n";
    return path;
}

/* 5x5x5 bcc Kr (250 atoms): box 20.5 A > 2*rcut = 16.55 A, then a
 * deterministic rattle so forces are non-zero. */
std::vector<std::string> rattled_bcc_kr(double amp) {
    return {
        "clear",
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        "atom_modify map array",
        "lattice bcc 4.1024",
        "region box block 0 5 0 5 0 5 units lattice",
        "create_box 1 box",
        "create_atoms 1 region box",
        "mass 1 83.798",
        "pair_style tadah",
        "pair_coeff * * EXTERN Kr",
        "neighbor 2.0 bin",
        "neigh_modify delay 0 every 1 check yes",
        "displace_atoms all random " + std::to_string(amp) + " "
            + std::to_string(amp) + " " + std::to_string(amp)
            + " 777 units box",
        "variable e equal pe",
        "variable fn equal fnorm",
        "variable fm equal fmax",
        "variable p equal press",
        "run 0 post no",
    };
}

std::vector<std::string> boxrelax_bcc_kr() {
    return {
        "clear",
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        "atom_modify map array",
        "lattice bcc 4.0",
        "region box block 0 5 0 5 0 5 units lattice",
        "create_box 1 box",
        "create_atoms 1 region box",
        "mass 1 83.798",
        "pair_style tadah",
        "pair_coeff * * EXTERN Kr",
        "neighbor 2.0 bin",
        "neigh_modify delay 0 every 1 check yes",
        "fix 1 all box/relax iso 0.0 vmax 0.001",
        "min_style cg",
        "min_modify dmax 0.01 line quadratic",
        "minimize 0 1e-10 2000 20000",
        "unfix 1",
        "variable a0 equal lx/5.0",
    };
}

double getd(LammpsRunner& runner, const LammpsRunner::LMPPtr& lmp,
            const char* name) {
    return std::stod(runner.get_variable(lmp, name));
}

/* The Ta D2_LJ potential that Tadah! 1.6.0 shipped as the docs quickstart
 * potential (Tadah.MLIP/docs/source/quickstart/pot.tadah of that release;
 * Tadah.LAMMPS/tests/tests_data/pots/ta_lj_cutdummy.tadah): D2_LJ Ta,
 * Cut_Dummy, RCUT2B 6.5, M_KRR Kern_Linear, INITMB false. */
const char* TA_LJ_BODY =
    "ATOM     Ta\n"
    "BIAS     false\n"
    "DIMER     false     0     false\n"
    "INIT2B     true\n"
    "INITMB     false\n"
    "LAMBDA     0\n"
    "MODEL     M_KRR     Kern_Linear     1\n"
    "NORM     false\n"
    "OALGO     1\n"
    "RCTYPE2B     Cut_Dummy\n"
    "RCUT2B     6.5\n"
    "TYPE2B     D2_LJ     Ta     Ta\n"
    "UNCERTAINTY     false\n"
    "WATOM     73\n"
    "WEIGHTS     14.204213349318765225     2604.396965191186154698\n";

/* D2_LJ (the pair term of TA_LJ_BODY, WATOM 73 folded into the LJ weights) +
 * DM_REAM Blip density and embedding (grids of MODELS/tests/test_dm_ream.cpp):
 * the EAM-form linear model that takes compute_2b_mb_half_eamfast. Same
 * potential as Tadah.LAMMPS/tests/tests_data/pots/ta_lj_ream.tadah. */
const char* TA_LJ_REAM_BODY =
    "ATOM              Ta\n"
    "AMPGRIDMB         0.40 0.25 0.12 0.05\n"
    "BIAS              false\n"
    "CEMBFUNC          0.0 0.4 0.8 1.2 1.6\n"
    "CGRIDMB           1.0 2.0 3.0 4.0\n"
    "DIMER             false 0 false\n"
    "INIT2B            true\n"
    "INITMB            true\n"
    "LAMBDA            0\n"
    "MODEL             M_BLR BF_Linear\n"
    "NORM              false\n"
    "RCTYPE2B          Cut_Dummy\n"
    "RCTYPEMB          Cut_Cos\n"
    "RCUT2B            6.5\n"
    "RCUTMB            6.0\n"
    "RHOFORM           SIGNED\n"
    "SEMBFUNC          1.5 1.5 1.5 1.5 1.5\n"
    "SGRIDMB           1.0 1.0 1.0 1.0\n"
    "TYPE2B            D2_LJ Ta Ta\n"
    "TYPEMB            DM_REAM 0 4 4 4 5 5 0 0 Ta Ta\n"
    "WATOM             1\n"
    "WEIGHTS           1036.907574500269861425 190120.978458956589292954 "
    "-1.2 -0.9 -0.6 -0.3 0.1\n";

/* TA_LJ_REAM_BODY with a fifth density Blip, 0.002 wide (eta 2000) and
 * centred at NARROW_RHO_R = 2.0, below the rho table's certified range: the
 * table (spacing 7.3e-4) cannot follow it, so there the fast EAM kernel is
 * right only if it takes rho and rho' from eam_density(). */
const char* TA_LJ_REAM_NARROW_RHO_BODY =
    "ATOM              Ta\n"
    "AMPGRIDMB         0.40 0.25 0.12 0.05 0.3\n"
    "BIAS              false\n"
    "CEMBFUNC          0.0 0.4 0.8 1.2 1.6\n"
    "CGRIDMB           1.0 2.0 3.0 4.0 2.0\n"
    "DIMER             false 0 false\n"
    "INIT2B            true\n"
    "INITMB            true\n"
    "LAMBDA            0\n"
    "MODEL             M_BLR BF_Linear\n"
    "NORM              false\n"
    "RCTYPE2B          Cut_Dummy\n"
    "RCTYPEMB          Cut_Cos\n"
    "RCUT2B            6.5\n"
    "RCUTMB            6.0\n"
    "RHOFORM           SIGNED\n"
    "SEMBFUNC          1.5 1.5 1.5 1.5 1.5\n"
    "SGRIDMB           1.0 1.0 1.0 1.0 2000\n"
    "TYPE2B            D2_LJ Ta Ta\n"
    "TYPEMB            DM_REAM 0 5 5 5 5 5 0 0 Ta Ta\n"
    "WATOM             1\n"
    "WEIGHTS           1036.907574500269861425 190120.978458956589292954 "
    "-1.2 -0.9 -0.6 -0.3 0.1\n";
constexpr double NARROW_RHO_R = 2.0;

/* Pure 2B: one D2_Blip bump 0.002 wide inside the top phi table segment
 * (Tadah.LAMMPS tests_data/pots/ta_blip_uncertifiable.tadah): the phi table
 * misses the collapse tolerance in its top segment. */
const char* TA_BLIP_UNCERTIFIABLE_BODY =
    "ATOM              Ta\n"
    "BIAS              false\n"
    "CGRID2B           6.4996\n"
    "DIMER             false 0 false\n"
    "INIT2B            true\n"
    "INITMB            false\n"
    "LAMBDA            0\n"
    "MODEL             M_BLR BF_Linear\n"
    "NORM              false\n"
    "RCTYPE2B          Cut_Dummy\n"
    "RCUT2B            6.5\n"
    "SGRID2B           2000\n"
    "TYPE2B            D2_Blip 1 1 Ta Ta\n"
    "WATOM             1\n"
    "WEIGHTS           1.0\n";

/* EAM: TA_LJ_REAM_BODY with a density Blip 0.002 wide inside the top rho
 * table segment (Cut_Dummy at 6.0; Tadah.LAMMPS tests_data/pots/
 * ta_lj_ream_rho_uncertifiable.tadah): the rho table misses the collapse
 * tolerance in its top segment, the phi table is certified. */
const char* TA_LJ_REAM_RHO_UNCERTIFIABLE_BODY =
    "ATOM              Ta\n"
    "AMPGRIDMB         0.40 0.25 0.12 0.05 0.3\n"
    "BIAS              false\n"
    "CEMBFUNC          0.0 0.4 0.8 1.2 1.6\n"
    "CGRIDMB           1.0 2.0 3.0 4.0 5.9996\n"
    "DIMER             false 0 false\n"
    "INIT2B            true\n"
    "INITMB            true\n"
    "LAMBDA            0\n"
    "MODEL             M_BLR BF_Linear\n"
    "NORM              false\n"
    "RCTYPE2B          Cut_Dummy\n"
    "RCTYPEMB          Cut_Dummy\n"
    "RCUT2B            6.5\n"
    "RCUTMB            6.0\n"
    "RHOFORM           SIGNED\n"
    "SEMBFUNC          1.5 1.5 1.5 1.5 1.5\n"
    "SGRIDMB           1.0 1.0 1.0 1.0 2000\n"
    "TYPE2B            D2_LJ Ta Ta\n"
    "TYPEMB            DM_REAM 0 5 5 5 5 5 0 0 Ta Ta\n"
    "WATOM             1\n"
    "WEIGHTS           1036.907574500269861425 190120.978458956589292954 "
    "-1.2 -0.9 -0.6 -0.3 0.1\n";

/* Distance of the extra atom from atom 1 in the close-pair scripts: far
 * below any certified table range for these potentials. */
constexpr double CLOSE_PAIR_R = 0.1;

std::string unique_path(const std::string& stem, const std::string& ext) {
    return stem + "_" + std::to_string(::getpid()) + ext;
}

/* Removes a scratch file however the test case ends (a failed REQUIRE
 * throws past a trailing std::remove). */
struct RemoveOnExit {
    std::string path;
    ~RemoveOnExit() { std::remove(path.c_str()); }
};

std::string write_pot_body(const char* body, bool eamfast,
                           const std::string& stem) {
    const std::string path = unique_path(
        stem + (eamfast ? "_on" : "_off"), ".tadah");
    std::ofstream ofs(path);
    ofs << body;
    ofs << "EAMFAST           " << (eamfast ? "true" : "false") << "\n";
    return path;
}

/* 5x5x5 bcc Ta, deterministic rattle; with close_pair an extra atom is
 * created pair_r (default CLOSE_PAIR_R) along x from (rattled) atom 1. */
std::vector<std::string> rattled_bcc_ta(double a0, bool close_pair,
                                        double pair_r = CLOSE_PAIR_R) {
    std::vector<std::string> s = {
        "clear",
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        "atom_modify map array",
        "lattice bcc " + std::to_string(a0),
        "region box block 0 5 0 5 0 5 units lattice",
        "create_box 1 box",
        "create_atoms 1 region box",
        "mass 1 180.95",
        "pair_style tadah",
        "pair_coeff * * EXTERN Ta",
        "neighbor 2.0 bin",
        "neigh_modify delay 0 every 1 check yes",
        "displace_atoms all random 0.08 0.08 0.08 4981 units box",
    };
    if (close_pair) {
        std::ostringstream dx;
        dx.precision(17);
        dx << pair_r;
        s.push_back("variable cx equal x[1]+" + dx.str());
        s.push_back("variable cy equal y[1]");
        s.push_back("variable cz equal z[1]");
        s.push_back("create_atoms 1 single ${cx} ${cy} ${cz} units box remap yes");
    }
    s.push_back("variable e equal pe");
    s.push_back("variable p equal press");
    s.push_back("thermo_style custom step pe press");
    s.push_back("run 0 post no");
    return s;
}

/* What the certificates allow the fast kernels to differ from the generic
 * ones by (see certificate_bound()). */
struct Bound {
    double e = 0.0;                     // total energy
    std::vector<double> f;              // each force component, per atom (by tag)
    double virial = 0.0;                // each pair virial component
};

/* rho and rho' at a probe distance: the table and eam_density(). */
struct RhoProbe {
    double r = 0.0;
    double table_v = 0.0, table_d = 0.0;
    double exact_v = 0.0, exact_d = 0.0;
};

struct Static {
    bool fast = false;
    double e = 0.0, p = 0.0;
    std::vector<double> x;              // 3 per atom, ordered by tag
    std::vector<double> f;              // 3 per atom, ordered by tag
    std::array<double, 3> box{};        // orthogonal box lengths
    std::array<double, 6> virial{};     // pair virial (serial run)
    double fmax = 0.0;                  // max |force component|
    std::vector<double> fp;             // EAM F'(rho_bar) per atom (fast EAM)
    tadah::TableCertificate phi_cert, rho_cert;   // fast runs
    Bound bound;                        // fast runs
    std::vector<RhoProbe> rho_probe;    // fast EAM runs
};

/* Minimum-image distance between atoms a and b (by tag - 1). */
double pair_distance(const Static& s, std::size_t a, std::size_t b) {
    double r2 = 0.0;
    for (int k = 0; k < 3; ++k) {
        double d = s.x[3 * a + k] - s.x[3 * b + k];
        d -= s.box[k] * std::round(d / s.box[k]);
        r2 += d * d;
    }
    return std::sqrt(r2);
}

/* Number of atom pairs (minimum image) with r_a <= r < r_b. */
int pairs_within(const Static& s, double r_a, double r_b) {
    const std::size_t n = s.x.size() / 3;
    int count = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j) {
            const double r = pair_distance(s, i, j);
            if (r >= r_a && r < r_b) ++count;
        }
    return count;
}

/* Largest |node value| (or |node derivative|) of `tab` over the range the
 * certificate's floor covers for the certified segments,
 * [min(r_lo, rc/2), rc]. */
double floor_max(const tadah::EamRadialTable& tab,
                 const tadah::TableCertificate& c, bool derivative) {
    const double r_from = std::min(c.r_lo, 0.5 * tab.rmax);
    double m = 0.0;
    for (int k = 1; k < tab.nr; ++k)
        if (k * tab.dr >= r_from * (1.0 - 1e-12))
            m = std::max(m, std::fabs(derivative ? tab.d[k] : tab.v[k]));
    return m;
}

/* The differences the certificates allow between the fast and the generic
 * kernel on this configuration. Inside a table's certified range each pair
 * term differs by at most max_rel * M (the certificate's error bound times
 * its floor, M <= the largest node value over the floor's range); below it
 * the fast kernel evaluates the analytic term, so the pair contributes no
 * table error. For EAM a density error dr_i = sum_j e_rho on atom i also
 * moves the embedding: F by |F'_i| dr_i (+ F''_i dr_i^2 / 2) and F' by
 * |F''_i| dr_i, which enters every pair force through rho'_ij. Summing
 * these over the actual pairs gives the energy, per-atom force and virial
 * bounds; the factor 2 covers the floor's maximum between nodes (the
 * bound takes it at the nodes), F'' taken by central difference and
 * higher orders in dr_i. */
Bound certificate_bound(const tadah::LammpsTadah& lt, const Static& s,
                        const std::vector<double>& rho_bar) {
    const tadah::EamCollapseTables& T = lt.eam_tables;
    const std::size_t n = s.x.size() / 3;
    const double e_pv = T.phi_cert.max_rel_val * floor_max(T.phi, T.phi_cert, false);
    const double e_pd = T.phi_cert.max_rel_der * floor_max(T.phi, T.phi_cert, true);
    double e_rv = 0.0, e_rd = 0.0;
    std::vector<double> fp(n, 0.0), fpp(n, 0.0);
    if (T.has_mb) {
        e_rv = T.rho_cert.max_rel_val * floor_max(T.rho, T.rho_cert, false);
        e_rd = T.rho_cert.max_rel_der * floor_max(T.rho, T.rho_cert, true);
        for (std::size_t i = 0; i < n; ++i) {
            fp[i] = s.fp[i];
            const double h = 1e-6 * std::max(1.0, std::fabs(rho_bar[i]));
            double F, fp_hi, fp_lo;
            lt.S.dmb->eam_embed(T.w, rho_bar[i] + h, F, fp_hi);
            lt.S.dmb->eam_embed(T.w, rho_bar[i] - h, F, fp_lo);
            fpp[i] = (fp_hi - fp_lo) / (2.0 * h);
        }
    }
    const double rc = std::max(T.phi.rmax, T.has_mb ? T.rho.rmax : 0.0);

    Bound b;
    b.f.assign(n, 0.0);
    std::vector<double> drho(n, 0.0);   // density error bound per atom
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j) {
            const double r = pair_distance(s, i, j);
            if (r >= rc) continue;
            const bool cp = r >= T.phi_cert.r_lo && r < T.phi.rmax;
            const bool cr = T.has_mb && r >= T.rho_cert.r_lo && r < T.rho.rmax;
            if (cp) b.e += e_pv;
            if (cr) {
                b.e += (std::fabs(fp[i]) + std::fabs(fp[j])) * e_rv;
                drho[i] += e_rv;
                drho[j] += e_rv;
            }
        }
    for (std::size_t i = 0; i < n; ++i)
        b.e += 0.5 * std::fabs(fpp[i]) * drho[i] * drho[i];
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j) {
            const double r = pair_distance(s, i, j);
            if (r >= rc) continue;
            const bool cp = r >= T.phi_cert.r_lo && r < T.phi.rmax;
            const bool cr = T.has_mb && r >= T.rho_cert.r_lo && r < T.rho.rmax;
            double e = cp ? e_pd : 0.0;
            if (cr) e += std::fabs(fp[i] + fp[j]) * e_rd;
            if (T.has_mb && r < T.rho.rmax) {
                double rv, rd;
                lt.S.dmb->eam_density(T.Z, T.Z, r, rv, rd);
                e += (std::fabs(fpp[i]) * drho[i] + std::fabs(fpp[j]) * drho[j])
                     * std::fabs(rd);
            }
            b.f[i] += e;
            b.f[j] += e;
            b.virial += r * e;
        }
    b.e *= 2.0;
    b.virial *= 2.0;
    for (double& v : b.f) v *= 2.0;
    return b;
}

/* One static evaluation of `script` with the potential `body`; the LAMMPS
 * log goes to `log_path` ("none" for no log). The instance is destroyed
 * before returning, so the log file is complete. A fast run also records
 * the certificates, the bound they give on this configuration and, for
 * EAM, rho and rho' (table and eam_density()) at each `rho_probe_r`. */
Static run_static(LammpsRunner& runner, const char* body, bool eamfast,
                  const std::vector<std::string>& script,
                  const std::string& log_path, const std::string& stem,
                  const std::vector<double>& rho_probe_r = {}) {
    const std::string pot_path = write_pot_body(body, eamfast, stem);
    const RemoveOnExit pot_guard{pot_path};
    tadah::core::Context pot_ctx;
    REQUIRE_NOTHROW(pot_ctx =
        tadah::core::CoreUserConfigParser().parse(pot_path));

    Static out;
    {
        auto lmp = runner.create_instance(
            {"lmp", "-log", log_path, "-screen", "none"});
        REQUIRE(lmp != nullptr);
        REQUIRE(runner.run_script(lmp, script, pot_ctx));

        auto* pt = dynamic_cast<LAMMPS_NS::PairTadah*>(lmp->force->pair);
        REQUIRE(pt != nullptr);
        const tadah::LammpsTadah& lt = *pt->lammps_tadah();
        out.fast = lt.eam_fast;
        out.e = getd(runner, lmp, "e");
        out.p = getd(runner, lmp, "p");

        const int nlocal = lmp->atom->nlocal;
        const auto natoms = static_cast<int>(lmp->atom->natoms);
        REQUIRE(nlocal == natoms);      // serial
        const auto n = static_cast<std::size_t>(natoms);
        out.f.assign(3 * n, 0.0);
        out.x.assign(3 * n, 0.0);
        std::vector<double> rho_bar(n, 0.0);
        const bool eam = out.fast && lt.eam_tables.has_mb;
        if (eam) out.fp.assign(n, 0.0);
        for (int k = 0; k < 3; ++k) out.box[k] = lmp->domain->prd[k];
        for (int i = 0; i < nlocal; ++i) {
            const auto t = static_cast<std::size_t>(lmp->atom->tag[i] - 1);
            for (int k = 0; k < 3; ++k) {
                out.x[3 * t + k] = lmp->atom->x[i][k];
                out.f[3 * t + k] = lmp->atom->f[i][k];
                out.fmax = std::max(out.fmax, std::fabs(lmp->atom->f[i][k]));
            }
            if (eam) {
                out.fp[t] = lt.eam_fp[i];
                rho_bar[t] = lt.eam_rho_bar[i];
            }
        }
        for (int k = 0; k < 6; ++k) out.virial[k] = lmp->force->pair->virial[k];

        if (out.fast) {
            out.phi_cert = lt.eam_tables.phi_cert;
            out.rho_cert = lt.eam_tables.rho_cert;
            out.bound = certificate_bound(lt, out, rho_bar);
        }
        if (eam)
            for (double r : rho_probe_r) {
                RhoProbe q;
                q.r = r;
                lt.eam_tables.rho.eval(r, q.table_v, q.table_d);
                lt.S.dmb->eam_density(lt.Z[0], lt.Z[0], r, q.exact_v,
                                      q.exact_d);
                out.rho_probe.push_back(q);
            }
    }
    return out;
}

std::string read_file(const std::string& path) {
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

/* The single "Tadah! ... fast path ON ..." line of a LAMMPS log. */
std::string collapse_log_line(const std::string& log) {
    std::istringstream in(log);
    std::string line, found;
    int n = 0;
    while (std::getline(in, line)) {
        if (line.find("Tadah!") != std::string::npos &&
            line.find("fast path") != std::string::npos) {
            found = line;
            ++n;
        }
    }
    REQUIRE(n == 1);
    return found;
}

/* All numbers after "analytic below r=" in a collapse log line. */
std::vector<double> analytic_below(const std::string& line) {
    std::vector<double> out;
    const std::regex re(R"(r=([0-9.eE+-]+))");
    const auto pos = line.find("analytic below");
    REQUIRE(pos != std::string::npos);
    const std::string tail = line.substr(pos);
    for (std::sregex_iterator it(tail.begin(), tail.end(), re), end;
         it != end; ++it)
        out.push_back(std::stod((*it)[1].str()));
    return out;
}

/* Energy, per-atom force and pair-virial parity within the bounds the fast
 * run's certificates give on this configuration (certificate_bound()); the
 * 1e-12 relative part covers rounding, which dominates for the huge
 * close-pair terms, whose fast-path value is the analytic one. */
void require_parity(const Static& fast, const Static& gen) {
    REQUIRE(fast.fast);
    REQUIRE(gen.f.size() == fast.f.size());
    REQUIRE(fast.bound.f.size() == fast.f.size() / 3);
    CAPTURE(fast.e, gen.e, fast.fmax, gen.fmax, fast.bound.e,
            fast.bound.virial);
    REQUIRE(fast.e == Approx(gen.e).epsilon(1e-12).margin(fast.bound.e));
    double worst = 0.0, worst_bound = 0.0;
    for (std::size_t k = 0; k < fast.f.size(); ++k) {
        const double d = std::fabs(fast.f[k] - gen.f[k]);
        if (d > worst) { worst = d; worst_bound = fast.bound.f[k / 3]; }
    }
    CAPTURE(worst, worst_bound);
    for (std::size_t k = 0; k < fast.f.size(); ++k) {
        CAPTURE(k);
        REQUIRE(fast.f[k] == Approx(gen.f[k]).epsilon(1e-12)
                                 .margin(fast.bound.f[k / 3]));
    }
    for (int k = 0; k < 6; ++k) {
        CAPTURE(k);
        REQUIRE(fast.virial[k] == Approx(gen.virial[k]).epsilon(1e-12)
                                      .margin(fast.bound.virial));
    }
}

}  // namespace

TEST_CASE("pair2b fast path: static + box/relax parity vs generic kernel",
          "[pair_tadah][pair2b][eamfast]") {
    /* Building the potential resolves "Kr"; required when this case runs
     * first in its process (ctest runs one case per process). */
    tadah::core::PeriodicTable::initialize();
    LammpsRunner runner;

    struct Obs { double e, fn, fm, p, a0; bool fast; };
    Obs obs[2];

    for (int mode = 0; mode < 2; ++mode) {
        const bool eamfast = (mode == 0);
        const std::string pot_path = write_pot(eamfast);
        tadah::core::Context pot_ctx;
        REQUIRE_NOTHROW(pot_ctx =
            tadah::core::CoreUserConfigParser().parse(pot_path));

        auto lmp = runner.create_instance(
            {"lmp", "-log", "none", "-screen", "none"});
        REQUIRE(lmp != nullptr);

        REQUIRE(runner.run_script(lmp, rattled_bcc_kr(0.08), pot_ctx));

        auto* pt = dynamic_cast<LAMMPS_NS::PairTadah*>(lmp->force->pair);
        REQUIRE(pt != nullptr);
        obs[mode].fast = pt->lammps_tadah()->eam_fast;

        obs[mode].e  = getd(runner, lmp, "e");
        obs[mode].fn = getd(runner, lmp, "fn");
        obs[mode].fm = getd(runner, lmp, "fm");
        obs[mode].p  = getd(runner, lmp, "p");

        REQUIRE(runner.run_script(lmp, boxrelax_bcc_kr(), pot_ctx));
        obs[mode].a0 = getd(runner, lmp, "a0");

        std::remove(pot_path.c_str());
    }

    /* The gate must actually differ between the two pots. */
    REQUIRE(obs[0].fast);
    REQUIRE_FALSE(obs[1].fast);

    /* Static parity: table is node-exact, mid-node Hermite error ~1e-9. */
    REQUIRE(obs[0].e  == Approx(obs[1].e).epsilon(1e-9));
    REQUIRE(obs[0].fn == Approx(obs[1].fn).epsilon(1e-7).margin(1e-8));
    REQUIRE(obs[0].fm == Approx(obs[1].fm).epsilon(1e-7).margin(1e-8));
    REQUIRE(obs[0].p  == Approx(obs[1].p).epsilon(1e-6).margin(1e-6));

    /* Same relaxed lattice constant through a full box/relax trajectory. */
    REQUIRE(obs[0].a0 == Approx(obs[1].a0).margin(1e-7));
}

TEST_CASE("pair2b fast path: adjoint capture falls back to the generic kernel",
          "[pair_tadah][pair2b][eamfast][adjoint]") {
    tadah::core::PeriodicTable::initialize();
    LammpsRunner runner;
    const std::string pot_path = write_pot(/*eamfast=*/true);
    tadah::core::Context pot_ctx;
    REQUIRE_NOTHROW(pot_ctx =
        tadah::core::CoreUserConfigParser().parse(pot_path));

    auto lmp = runner.create_instance(
        {"lmp", "-log", "none", "-screen", "none"});
    REQUIRE(lmp != nullptr);
    REQUIRE(runner.run_script(lmp, rattled_bcc_kr(0.05), pot_ctx));

    auto* pt = dynamic_cast<LAMMPS_NS::PairTadah*>(lmp->force->pair);
    REQUIRE(pt != nullptr);
    auto* lt = pt->lammps_tadah();
    REQUIRE(lt->eam_fast);

    /* Toggle capture and recompute: pure-2B must NOT error (the MB fast
     * path would) and must fill the snapshot via the generic kernel. */
    lt->capture_adjoint_state = true;
    lt->adjoint_snapshot.box_dofs = 1;
    REQUIRE_NOTHROW(lmp->force->pair->compute(1, 1));
    REQUIRE(lt->adjoint_snapshot.valid);
    REQUIRE(lt->adjoint_snapshot.n_desc == lt->S.descriptor_size());
    lt->capture_adjoint_state = false;

    std::remove(pot_path.c_str());
}

TEST_CASE("pair2b fast path: the Ta D2_LJ potential matches the generic "
          "kernel, the top table segment included, and logs its certificate",
          "[pair_tadah][pair2b][eamfast][certificate]") {
    tadah::core::PeriodicTable::initialize();
    LammpsRunner runner;
    const std::string log_path = unique_path("pair2b_lj_fast", ".log");
    const RemoveOnExit log_guard{log_path};

    const Static fast = run_static(runner, TA_LJ_BODY, true,
                                   rattled_bcc_ta(3.3, false), log_path,
                                   "pair2b_lj");
    const Static gen = run_static(runner, TA_LJ_BODY, false,
                                  rattled_bcc_ta(3.3, false), "none",
                                  "pair2b_lj");
    REQUIRE(fast.fast);
    REQUIRE_FALSE(gen.fast);

    /* The load-time line states the certified range, the tolerance and the
     * analytic clause (it used to be "parity dphi=3.071e+42"). */
    const std::string line = collapse_log_line(read_file(log_path));
    INFO(line);
    REQUIRE(line.find("Tadah! pair-collapse fast path ON (2b-only, nr=8192): "
                      "table certified for r in [") != std::string::npos);
    REQUIRE(line.find(", 6.5] (rel tol 1e-9: energy ") != std::string::npos);
    REQUIRE(line.find(", force ") != std::string::npos);
    REQUIRE(line.find("parity") == std::string::npos);
    REQUIRE(analytic_below(line).size() == 1);

    /* The cell holds pairs in the top table segment [rc - dr, rc), where
     * Cut_Dummy's jump at rc used to pull the table towards 0. */
    REQUIRE(pairs_within(fast, 6.5 - 6.5 / 8191.0, 6.5) > 0);
    require_parity(fast, gen);
}

TEST_CASE("pair2b fast path: a pair far below the certified table range "
          "matches the generic kernel",
          "[pair_tadah][pair2b][eamfast][certificate]") {
    tadah::core::PeriodicTable::initialize();
    LammpsRunner runner;
    const std::string log_path = unique_path("pair2b_lj_close", ".log");
    const RemoveOnExit log_guard{log_path};

    /* One extra atom CLOSE_PAIR_R from atom 1 of the 3.5 A lattice:
     * that pair term is ~1e17, and the table is off there. The fast kernel
     * evaluates it analytically, so it matches to rounding. */
    const Static fast = run_static(runner, TA_LJ_BODY, true,
                                   rattled_bcc_ta(3.5, true), log_path,
                                   "pair2b_lj_close");
    const Static gen = run_static(runner, TA_LJ_BODY, false,
                                  rattled_bcc_ta(3.5, true), "none",
                                  "pair2b_lj_close");
    REQUIRE(fast.fast);
    REQUIRE_FALSE(gen.fast);
    REQUIRE(pairs_within(fast, 0.0, 1.01 * CLOSE_PAIR_R) == 1);
    REQUIRE(gen.e > 1e15);
    require_parity(fast, gen);

    const std::string line = collapse_log_line(read_file(log_path));
    INFO(line);
    const auto r_lo = analytic_below(line);
    REQUIRE(r_lo.size() == 1);
    REQUIRE(CLOSE_PAIR_R < r_lo[0]);
}

TEST_CASE("EAM fast path: compute_2b_mb_half_eamfast matches the generic "
          "kernel for energy, forces and virial, and logs its certificates",
          "[pair_tadah][eam][eamfast][certificate]") {
    tadah::core::PeriodicTable::initialize();
    LammpsRunner runner;
    const std::string log_path = unique_path("eam_ream_fast", ".log");
    const RemoveOnExit log_guard{log_path};

    const Static fast = run_static(runner, TA_LJ_REAM_BODY, true,
                                   rattled_bcc_ta(3.3, false), log_path,
                                   "eam_ream");
    const Static gen = run_static(runner, TA_LJ_REAM_BODY, false,
                                  rattled_bcc_ta(3.3, false), "none",
                                  "eam_ream");
    REQUIRE(fast.fast);
    REQUIRE_FALSE(gen.fast);

    /* The embedding must be live, or the density half of the kernel is not
     * exercised: the EAM energy differs from the pair-only sum. */
    const Static lj_only = run_static(runner, TA_LJ_BODY, false,
                                      rattled_bcc_ta(3.3, false), "none",
                                      "eam_ream_lj");
    REQUIRE(std::fabs(gen.e - lj_only.e) > 1.0);

    REQUIRE(pairs_within(fast, 6.5 - 6.5 / 8191.0, 6.5) > 0);
    require_parity(fast, gen);

    const std::string line = collapse_log_line(read_file(log_path));
    INFO(line);
    REQUIRE(line.find("Tadah! EAM-collapse fast path ON (nr=8192): "
                      "phi table certified for r in [") != std::string::npos);
    REQUIRE(line.find("rho table certified for r in [") != std::string::npos);
    REQUIRE(line.find("rel tol 1e-9") != std::string::npos);
    REQUIRE(line.find("parity") == std::string::npos);
    REQUIRE(analytic_below(line).size() == 2);   // phi, rho
}

TEST_CASE("EAM fast path: a pair far below both certified table ranges "
          "matches the generic kernel",
          "[pair_tadah][eam][eamfast][certificate]") {
    tadah::core::PeriodicTable::initialize();
    LammpsRunner runner;
    const std::string log_path = unique_path("eam_ream_close", ".log");
    const RemoveOnExit log_guard{log_path};

    const Static fast = run_static(runner, TA_LJ_REAM_BODY, true,
                                   rattled_bcc_ta(3.5, true), log_path,
                                   "eam_ream_close");
    const Static gen = run_static(runner, TA_LJ_REAM_BODY, false,
                                  rattled_bcc_ta(3.5, true), "none",
                                  "eam_ream_close");
    REQUIRE(fast.fast);
    REQUIRE_FALSE(gen.fast);
    REQUIRE(pairs_within(fast, 0.0, 1.01 * CLOSE_PAIR_R) == 1);
    REQUIRE(gen.e > 1e15);
    require_parity(fast, gen);

    const std::string line = collapse_log_line(read_file(log_path));
    INFO(line);
    const auto r_lo = analytic_below(line);     // phi, rho
    REQUIRE(r_lo.size() == 2);
    REQUIRE(CLOSE_PAIR_R < r_lo[0]);
    REQUIRE(CLOSE_PAIR_R < r_lo[1]);
}

TEST_CASE("EAM fast path: below the rho table's certified range the kernel "
          "takes rho and rho' from eam_density()",
          "[pair_tadah][eam][eamfast][certificate]") {
    tadah::core::PeriodicTable::initialize();
    LammpsRunner runner;
    const std::string log_path = unique_path("eam_narrow_rho", ".log");
    const RemoveOnExit log_guard{log_path};

    /* One extra atom NARROW_RHO_R from atom 1: that pair sits on the
     * narrow density Blip. */
    const Static fast = run_static(runner, TA_LJ_REAM_NARROW_RHO_BODY, true,
                                   rattled_bcc_ta(3.5, true, NARROW_RHO_R),
                                   log_path, "eam_narrow_rho",
                                   {NARROW_RHO_R});
    const Static gen = run_static(runner, TA_LJ_REAM_NARROW_RHO_BODY, false,
                                  rattled_bcc_ta(3.5, true, NARROW_RHO_R),
                                  "none", "eam_narrow_rho");
    REQUIRE(fast.fast);
    REQUIRE_FALSE(gen.fast);
    const std::size_t n = fast.x.size() / 3;
    REQUIRE(pair_distance(fast, 0, n - 1) ==
            Approx(NARROW_RHO_R).epsilon(1e-12));   // atom 1, the extra one

    /* The premise. The pair lies below the rho table's certified range,
     * where the table is far off: */
    const std::string line = collapse_log_line(read_file(log_path));
    INFO(line);
    const auto r_lo = analytic_below(line);          // phi, rho
    REQUIRE(r_lo.size() == 2);
    REQUIRE(r_lo[1] == Approx(fast.rho_cert.r_lo).epsilon(1e-5));
    REQUIRE(NARROW_RHO_R < fast.rho_cert.r_lo);
    REQUIRE(fast.rho_probe.size() == 1);
    const RhoProbe& q = fast.rho_probe[0];
    const double d_rho = std::fabs(q.table_v - q.exact_v);
    const double d_rhop = std::fabs(q.table_d - q.exact_d);
    CAPTURE(q.table_v, q.exact_v, q.table_d, q.exact_d);
    REQUIRE(d_rho > 1e-2);
    REQUIRE(d_rhop > 10.0);
    /* ... and the embedding is live on the pair, so a kernel that took
     * rho (pass 1) or rho' (pass 2) from the table would miss the energy
     * and the pair's forces by far more than the certified bounds. */
    const double fp_pair = std::fabs(fast.fp[0] + fast.fp[n - 1]);
    CAPTURE(fast.fp[0], fast.fp[n - 1], fast.bound.e, fast.bound.f[0]);
    REQUIRE(fp_pair * d_rho > 1e3 * fast.bound.e);
    REQUIRE(fp_pair * d_rhop > 1e3 * fast.bound.f[0]);

    require_parity(fast, gen);
}

TEST_CASE("EAMFAST: a potential whose phi or rho table has no certified "
          "range runs the generic kernel and says why",
          "[pair_tadah][eamfast][certificate]") {
    tadah::core::PeriodicTable::initialize();
    LammpsRunner runner;
    struct Case {
        const char* body;
        const char* stem;
        const char* why;
    };
    for (const Case& c :
         {Case{TA_BLIP_UNCERTIFIABLE_BODY, "unc_blip",
               "phi table misses rel tol 1e-9 already in its top segment, "
               "just below the cutoff r=6.5"},
          Case{TA_LJ_REAM_RHO_UNCERTIFIABLE_BODY, "unc_ream",
               "rho table misses rel tol 1e-9 already in its top segment, "
               "just below the cutoff r=6"}}) {
        CAPTURE(c.stem);
        const std::string log_path = unique_path(c.stem, ".log");
        const RemoveOnExit log_guard{log_path};
        const Static on = run_static(runner, c.body, true,
                                     rattled_bcc_ta(3.3, false), log_path,
                                     c.stem);
        const Static off = run_static(runner, c.body, false,
                                      rattled_bcc_ta(3.3, false), "none",
                                      c.stem);
        REQUIRE_FALSE(on.fast);
        REQUIRE_FALSE(off.fast);

        const std::string line = collapse_log_line(read_file(log_path));
        REQUIRE(line == std::string("  Tadah! collapse fast path "
                                    "unavailable: ") + c.why +
                            "; using the generic kernel");

        /* Both runs take the generic kernel: the same numbers, bit for
         * bit. */
        REQUIRE(on.e == off.e);
        REQUIRE(on.f == off.f);
        REQUIRE(on.virial == off.virial);
    }
}
