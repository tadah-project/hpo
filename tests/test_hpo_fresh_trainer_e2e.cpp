// test_hpo_fresh_trainer_e2e.cpp
//
// Every evaluation of HPO_Host trains a model built from that evaluation's
// candidate context: the -c configuration with the candidate's OPTIM
// values. A thread used to keep the Trainer of its first evaluation and to
// rewrite only the OPTIM keys of its context for the next ones
// (Trainer::reinit), so whatever the model, the kernel or the design
// matrix read when they were built kept the first evaluation's value: the
// EWEIGHT/FWEIGHT/SWEIGHT row weights and the kernel's MPARAMS were not
// applied after a thread's first evaluation (while every potential
// recorded the candidate's values), OALGO 3 appended a SIGMA per fit to
// the reused context (the weight covariance is now WCOV and WNOISE, which
// the potential gets from the model), an M_KRR -c potential with WEIGHTS could not be
// trained at the first evaluation, and a second host on the same thread
// (a second `TASK hpo` of a task file) evaluated the first host's model.
//
// The reference of each case is an independent Trainer built from the
// candidate context and trained on the data as the host prepares it
// (update_candidate_context, apply_load_transforms, one-time neighbour
// lists at nn_safe_rcutmax): the arithmetic is the host's, so the weights
// agree to rounding.
//
// Data (the pattern of test_validation_follows_metric_e2e.cpp): 2x2x2 bcc
// Ta cells of 16 atoms, rattled, with made-up energies, forces and
// stresses; D2_BP with 3 Gaussians, Cut_Cos at 4.5 A, no many-body term,
// no bias (3 weights).
//
// Hidden [trainer_e2e][.] like the other HPO_Host end-to-end groups: the
// group is one CTest entry (hpo_trainer_e2e, tests/CMakeLists.txt):
//   ./test_hpo "[trainer_e2e]"
// Each "Before the fix" below says how the case fails run alone. Run as
// the group, in one process, the cases failed more widely before the fix:
// a case's first evaluation also trained the model of the previous case
// (the thread's trainer of the previous host), e.g. a Kern_Linear model
// for the MPARAMS case.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/candidate_context.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/rcut_derivation.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/startup_validation.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/load_transforms.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>
#include <tadah/models/functions/kernels/kern_rbf.h>

#include <dlib/matrix.h>

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;
using tadah::core::Context;
using tadah::core::PeriodicTable;
using tadah::hpo::HPOTargets;
using tadah::hpo::validate_hpo_setup;
using tadah::mlip::Atom;
using tadah::mlip::NNFinder;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using ParamVec = dlib::matrix<double, 0, 1>;
using Host     = tadah::hpo::HPO_Host<ParamVec, ::LammpsRunner>;

constexpr double kFailScore = 1e9;

std::string unique_name(const std::string& stem) {
    return stem + std::to_string(::getpid()) + "_" +
           std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count());
}

/* A temporary working directory, the cwd while the object lives. */
struct ScratchDir {
    fs::path dir, original_cwd;
    ScratchDir() {
        dir = fs::temp_directory_path() / unique_name("tadah_trainer_e2e_");
        fs::create_directories(dir);
        original_cwd = fs::current_path();
        fs::current_path(dir);
    }
    ~ScratchDir() {
        std::error_code ec;
        fs::current_path(original_cwd, ec);
        fs::remove_all(dir, ec);
    }
};

bool same_bits(double a, double b) {
    return std::memcmp(&a, &b, sizeof a) == 0;
}

double rel_diff(const std::vector<double>& a, const std::vector<double>& b) {
    REQUIRE(a.size() == b.size());
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        num = std::max(num, std::fabs(a[i] - b[i]));
        den = std::max(den, std::fabs(b[i]));
    }
    return den > 0.0 ? num / den : num;
}

/* ---- data ------------------------------------------------------------ */

double ref_force(int id, std::size_t i, int k) {
    return 0.3 * std::sin(1.7 * static_cast<double>(i) +
                          0.9 * static_cast<double>(k) +
                          0.37 * static_cast<double>(id));
}

double ref_stress(int id, int x, int y) {
    return 0.02 * std::cos(0.7 * static_cast<double>(id) +
                           1.3 * static_cast<double>(x + y) +
                           0.4 * static_cast<double>(x * y));
}

/* A 2x2x2 conventional bcc cell of lattice constant @p a, 16 atoms
 * rattled by up to 0.006 of the edge (@p phase varies the rattle). */
Structure bcc_cell(int id, double a, double phase) {
    const double L = 2.0 * a;
    Structure st;
    st.label = "rattled bcc 2x2x2 cell " + std::to_string(id);
    st.cell.load(L, 0.0, 0.0, 0.0, L, 0.0, 0.0, 0.0, L);
    const tadah::core::Element ta = PeriodicTable::find_by_symbol("Ta");
    std::size_t i = 0;
    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
            for (int z = 0; z < 2; ++z)
                for (double h : {0.0, 0.25}) {
                    const int n[3] = {x, y, z};
                    double r[3];
                    for (int k = 0; k < 3; ++k) {
                        const double f =
                            0.5 * n[k] + h + 0.02 +
                            0.006 * std::sin(2.3 * static_cast<double>(i) +
                                             1.1 * static_cast<double>(k) +
                                             phase);
                        r[k] = f * L;
                    }
                    st.add_atom(Atom(ta, r[0], r[1], r[2],
                                     ref_force(id, i, 0), ref_force(id, i, 1),
                                     ref_force(id, i, 2)));
                    ++i;
                }
    st.energy = (-5.0 + 2.0 * (a - 3.15) * (a - 3.15) +
                 0.01 * std::sin(static_cast<double>(id))) *
                static_cast<double>(st.natoms());
    for (int x = 0; x < 3; ++x)
        for (int y = 0; y < 3; ++y) st.stress(x, y) = ref_stress(id, x, y);
    return st;
}

void write_db(const fs::path& path, const std::vector<Structure>& sts) {
    std::ofstream os(path);
    REQUIRE(os.good());
    for (const Structure& st : sts) st.dump_to_file(os);
    REQUIRE(os.good());
}

/* The training set, the validation set and its --validation file, in the
 * working directory. */
struct DataFiles {
    fs::path train_db, val_db, val_cfg, hpo_cfg;
    explicit DataFiles(const fs::path& dir)
        : train_db(dir / "train.db"), val_db(dir / "val.db"),
          val_cfg(dir / "config.val"), hpo_cfg(dir / "config.hpo") {
        write_db(train_db, {bcc_cell(1, 2.95, 0.3), bcc_cell(2, 3.05, 0.9),
                            bcc_cell(3, 3.15, 1.4), bcc_cell(4, 3.25, 2.2),
                            bcc_cell(5, 3.35, 2.9)});
        write_db(val_db, {bcc_cell(11, 3.10, 1.9), bcc_cell(12, 3.20, 0.6)});
        std::ofstream os(val_cfg);
        os << "DBFILE " << val_db.string() << "\n";
    }
    void write_target(const std::string& text) const {
        std::ofstream os(hpo_cfg);
        os << "OPTIMIZER\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 1\nENDOPTIMIZER\n"
           << "LOSS L2\nFAILSCORE 1e9\nDIGITS 15\n" << text;
    }
};

/* ---- the -c configurations -------------------------------------------- */

const std::vector<std::string> kBLR = {"M_BLR", "BF_Linear"};
const std::vector<std::string> kKRR = {"M_KRR", "Kern_Linear", "1"};

void set_key(Context& c, const std::string& key, const std::string& value) {
    c.remove(key);
    c.add(key, value);
}

/* D2_BP (3 Gaussians, Cut_Cos 4.5 A) on Ta, FORCE @p force, STRESS
 * @p stress, OALGO 4 with LAMBDA 1e-3, the training DBFILE, the
 * --validation file and the hpotarget. */
Context master_context(const std::vector<std::string>& model,
                       const DataFiles& f, bool force, bool stress) {
    Context c;
    c.add("ATOM",     "Ta");
    c.add("WATOM",    1);
    c.add("INIT2B",   "true");
    c.add("INITMB",   "false");
    c.add("TYPE2B",   "D2_BP 3 3 Ta Ta");
    c.add("CGRID2B",  "2.6 3.3 4.0");
    c.add("SGRID2B",  "1.0 1.0 1.0");
    c.add("RCTYPE2B", "Cut_Cos");
    c.add("RCUT2B",   4.5);
    c.add("DIMER",    "false 0 false");
    c.add("FORCE",    force ? "true" : "false");
    c.add("STRESS",   stress ? "true" : "false");
    c.add("EWEIGHT",  1.0);
    c.add("FWEIGHT",  1.0);
    c.add("SWEIGHT",  1.0);
    c.add("BIAS",     "false");
    c.add("NORM",     "false");
    c.add("VERBOSE",  0);
    for (const std::string& tok : model) c.add("MODEL", tok);
    c.add("LAMBDA",   1e-3);
    c.add("OALGO",    4);
    c.add("DBFILE",     f.train_db.string());
    c.add("VALIDATION", f.val_cfg.string());
    c.add("HPOTARGET",  f.hpo_cfg.string());
    return c;
}

/* ---- the independent reference --------------------------------------- */

/* A Trainer built from the candidate context of @p p and trained on the
 * training set as the host prepares it; its weights and potential. */
struct Reference {
    std::vector<double> weights;
    Context pot;
};

Reference reference_fit(const Context& master, const fs::path& hpo_cfg,
                        const ParamVec& p) {
    Context m = master;
    HPOTargets trg(m, hpo_cfg.string());
    Context cand = m;
    tadah::hpo::detail::update_candidate_context(cand, trg, p, false, false);
    StructureDB db(m);
    (void)tadah::mlip::apply_load_transforms(db, m);
    NNFinder().calc(db, tadah::hpo::detail::nn_safe_rcutmax(m, trg));
    tadah::mlip::memory::MLIPWorkspaceManager ws;
    tadah::mlip::Trainer tr(cand, ws);
    const StructureDB& cdb = db;
    tr.train(cdb);
    Reference out;
    const auto& w = tr.model->get_weights();
    out.weights.assign(w.begin(), w.end());
    out.pot = tr.model->get_param_file();
    return out;
}

/* The slot's trained weights and potential after an evaluation. */
std::vector<double> slot_weights() {
    const auto* tr = Host::tls_trainer_for_test();
    REQUIRE(tr != nullptr);
    const auto& w = tr->model->get_weights();
    return std::vector<double>(w.begin(), w.end());
}

Context slot_potential() {
    auto* tr = const_cast<tadah::mlip::Trainer*>(Host::tls_trainer_for_test());
    REQUIRE(tr != nullptr);
    return tr->model->get_param_file();
}

ParamVec point(std::initializer_list<double> v) {
    ParamVec p(static_cast<long>(v.size()));
    long i = 0;
    for (double x : v) p(i++) = x;
    return p;
}

/* A host on @p ctx, validated like hpo_run() validates it. */
struct HostRun {
    HPOTargets trg;
    Host host;
    HostRun(Context& ctx, const DataFiles& f)
        : trg(ctx, f.hpo_cfg.string()),
          host((validate_hpo_setup(ctx, trg, true, true), ctx), trg,
               f.val_cfg.string()) {}
};

}  // namespace

/* Pins: with FORCE and STRESS true and OPTIM EWEIGHT (1), FWEIGHT (1) and
 * SWEIGHT (1), each of three evaluations of one host on one thread, at
 * (1, 1, 1), (5, 300, 0.2) and (0.3, 20, 700), trains weights equal to
 * those of an independent Trainer of its own candidate (1e-12 relative),
 * for M_BLR BF_Linear and M_KRR Kern_Linear; the three weight vectors
 * differ (the row weights change the fit).
 * Before the fix: evaluations 2 and 3 trained the weights of evaluation 1
 * (the design matrix of the reused Trainer kept the row weights it was
 * built with), and the potential recorded the candidate's weights keys. */
TEST_CASE("Each evaluation applies its EWEIGHT, FWEIGHT and SWEIGHT",
          "[trainer_e2e][.]") {
    PeriodicTable::initialize();
    for (const auto* model : {&kBLR, &kKRR}) {
        DYNAMIC_SECTION((*model)[0] << " " << (*model)[1]) {
            Host::reset_globals_for_test();
            ScratchDir sd;
            DataFiles f(sd.dir);
            f.write_target("PC_ERMSE 0 1\n"
                           "OPTIM EWEIGHT (1) 0.1 10\n"
                           "OPTIM FWEIGHT (1) 0.1 1000\n"
                           "OPTIM SWEIGHT (1) 0.1 1000\n");
            Context ctx = master_context(*model, f, true, true);
            HostRun run(ctx, f);
            std::vector<std::vector<double>> seen;
            for (const ParamVec& p : {point({1.0, 1.0, 1.0}),
                                      point({5.0, 300.0, 0.2}),
                                      point({0.3, 20.0, 700.0})}) {
                INFO("candidate " << p(0) << " " << p(1) << " " << p(2));
                double loss = 0.0;
                REQUIRE_NOTHROW(loss = run.host(p));
                CHECK(loss < kFailScore);
                const std::vector<double> w = slot_weights();
                const Reference ref = reference_fit(ctx, f.hpo_cfg, p);
                INFO("weights " << w[0] << " " << w[1] << " " << w[2]);
                CHECK(rel_diff(w, ref.weights) < 1e-12);
                seen.push_back(w);
            }
            CHECK(rel_diff(seen[0], seen[1]) > 1e-6);
            CHECK(rel_diff(seen[1], seen[2]) > 1e-6);
            Host::reset_globals_for_test();
        }
    }
}

/* Pins: M_KRR Kern_RBF (SBASIS 4) with OPTIM MPARAMS (1), evaluated at
 * sigma 1.0 and then 0.3 by one host on one thread: after the second
 * evaluation the model's kernel has sigma 0.3, its weights equal an
 * independent Trainer's at sigma 0.3 (1e-12 relative), and its potential
 * records MPARAMS 0.3 and the reference's BASIS and WEIGHTS.
 * Before the fix: the kernel kept sigma 1.0 from the first evaluation, so
 * the second fit and its loss were those of sigma 1.0 while the potential
 * recorded MPARAMS 0.3 (it predicted otherwise than the loss reported). */
TEST_CASE("Each evaluation builds the kernel of its MPARAMS",
          "[trainer_e2e][.]") {
    PeriodicTable::initialize();
    Host::reset_globals_for_test();
    ScratchDir sd;
    DataFiles f(sd.dir);
    f.write_target("PC_ERMSE 0 1\nOPTIM MPARAMS (1) 0.3 3.0\n");
    Context ctx = master_context({"M_KRR", "Kern_RBF", "1"}, f, false, false);
    ctx.add("MPARAMS", 1.0);
    ctx.add("SBASIS", 4);
    set_key(ctx, "OALGO", "1");
    set_key(ctx, "LAMBDA", "0");
    HostRun run(ctx, f);
    for (double sigma : {1.0, 0.3}) {
        INFO("sigma " << sigma);
        const ParamVec p = point({sigma});
        double loss = 0.0;
        REQUIRE_NOTHROW(loss = run.host(p));
        CHECK(loss < kFailScore);
        const auto* tr = Host::tls_trainer_for_test();
        REQUIRE(tr != nullptr);
        const auto* rbf = dynamic_cast<const tadah::models::Kern_RBF*>(tr->fb);
        REQUIRE(rbf != nullptr);
        CHECK(same_bits(rbf->sigma, sigma));
        const Reference ref = reference_fit(ctx, f.hpo_cfg, p);
        CHECK(rel_diff(slot_weights(), ref.weights) < 1e-12);
        const Context pot = slot_potential();
        CHECK(same_bits(pot.get<double>("MPARAMS"), sigma));
        REQUIRE(pot.size("BASIS") == ref.pot.size("BASIS"));
        std::vector<double> b(pot.size("BASIS")), rb(ref.pot.size("BASIS"));
        Context(pot).get("BASIS", b);
        Context(ref.pot).get("BASIS", rb);
        CHECK(rel_diff(b, rb) < 1e-12);
    }
    Host::reset_globals_for_test();
}

/* Pins: OALGO 3 with OPTIM LAMBDA (1), three evaluations of one host on
 * one thread (LAMBDA 1e-4, 1e-3, 1e-2): each potential holds one weight
 * covariance, WCOV of 3*3+1 = 10 values and a WNOISE, equal (1e-12
 * relative) to those of an independent Trainer of its candidate, and no
 * SIGMA.
 * Before the fix (WCOV and WNOISE): the potentials had SIGMA (10 values,
 * the transposed and unscaled matrix; this case checked it) and no WCOV.
 * Before fix/hpo-trainer-reuse: 10, 20 and 30 values of SIGMA: each fit
 * appended its SIGMA to the reused context (best_pot.tadah then carried
 * several, and `tadah predict --error` on it read past its covariance
 * matrix). */
TEST_CASE("Each OALGO 3 evaluation writes the SIGMA of its own fit",
          "[trainer_e2e][.]") {
    PeriodicTable::initialize();
    Host::reset_globals_for_test();
    ScratchDir sd;
    DataFiles f(sd.dir);
    f.write_target("PC_ERMSE 0 1\nOPTIM LAMBDA (1) 1e-6 1e-1 LOG\n");
    Context ctx = master_context(kBLR, f, false, false);
    set_key(ctx, "OALGO", "3");
    HostRun run(ctx, f);
    for (double lambda : {1e-4, 1e-3, 1e-2}) {
        INFO("LAMBDA " << lambda);
        const ParamVec p = point({lambda});
        double loss = 0.0;
        REQUIRE_NOTHROW(loss = run.host(p));
        CHECK(loss < kFailScore);
        const Context pot = slot_potential();
        const Reference ref = reference_fit(ctx, f.hpo_cfg, p);
        CHECK_FALSE(pot.has_stored("SIGMA"));
        REQUIRE(ref.pot.size("WCOV") == 10);
        REQUIRE(pot.size("WCOV") == 10);
        std::vector<double> s(10), rs(10);
        Context(pot).get("WCOV", s);
        Context(ref.pot).get("WCOV", rs);
        CHECK(rel_diff(s, rs) < 1e-12);
        REQUIRE(ref.pot.size("WNOISE") == pot.size("WNOISE"));
        REQUIRE(pot.size("WNOISE") >= 2);
        for (size_t k = 1; k < pot.size("WNOISE"); k += 2)
            CHECK(rel_diff({pot.get<double>("WNOISE", k)},
                           {ref.pot.get<double>("WNOISE", k)}) < 1e-12);
    }
    Host::reset_globals_for_test();
}

/* Pins: two hosts on one thread, A (LAMBDA 1e-4, FWEIGHT 1) and then B
 * (LAMBDA 1e-1, FWEIGHT 100; neither searched), both FORCE true with OPTIM
 * CGRID2B (3), PC_ERMSE and PC_FRMSE, each evaluated once at CGRID2B[3] =
 * 3.95 (as two `TASK hpo` blocks of one task file run): B's loss is, bit
 * for bit, the loss B gives on a thread that evaluated nothing before, and
 * B's model was trained with B's LAMBDA and FWEIGHT.
 * Before the fix: B evaluated A's model (the thread's Trainer of A with
 * B's CGRID2B): its loss was A's and its context held LAMBDA 1e-4 and
 * FWEIGHT 1. */
TEST_CASE("A second host on a thread evaluates its own configuration",
          "[trainer_e2e][.]") {
    PeriodicTable::initialize();
    ScratchDir sd;
    DataFiles f(sd.dir);
    f.write_target("PC_ERMSE 0 1\nPC_FRMSE 0 1\nOPTIM CGRID2B (3) 3.7 4.3\n");
    Context a = master_context(kBLR, f, true, false);
    set_key(a, "LAMBDA", "1e-4");
    Context b = master_context(kBLR, f, true, false);
    set_key(b, "LAMBDA", "1e-1");
    set_key(b, "FWEIGHT", "100");
    const ParamVec p = point({3.95});

    double b_alone = 0.0;
    std::thread fresh([&] {
        Host::reset_globals_for_test();
        Context cb = b;
        HostRun run(cb, f);
        b_alone = run.host(p);
    });
    fresh.join();
    REQUIRE(b_alone < kFailScore);

    double loss_a = 0.0, loss_b = 0.0;
    {
        Host::reset_globals_for_test();
        HostRun run(a, f);
        loss_a = run.host(p);
    }
    {
        Host::reset_globals_for_test();
        HostRun run(b, f);
        loss_b = run.host(p);
        const auto* tr = Host::tls_trainer_for_test();
        REQUIRE(tr != nullptr);
        CHECK(tr->context.stored_value("LAMBDA").value_or("") == "1e-1");
        CHECK(tr->context.stored_value("FWEIGHT").value_or("") == "100");
    }
    INFO("A " << loss_a << ", B after A " << loss_b << ", B alone " << b_alone);
    CHECK(same_bits(loss_b, b_alone));
    CHECK_FALSE(same_bits(loss_a, b_alone));
    Host::reset_globals_for_test();
}

/* Pins: an M_KRR Kern_Linear -c potential with WEIGHTS (a potential used as
 * the starting point) is trained at the first evaluation: the loss is
 * below FAILSCORE and the weights are those of an independent Trainer of
 * the candidate (1e-12 relative), not the -c WEIGHTS.
 * Before the fix: the first evaluation failed (FAILSCORE): the model built
 * from the WEIGHTS counted as trained, and train() threw "This object is
 * already trained!". */
TEST_CASE("An M_KRR -c potential with WEIGHTS is trained at the first "
          "evaluation",
          "[trainer_e2e][.]") {
    PeriodicTable::initialize();
    Host::reset_globals_for_test();
    ScratchDir sd;
    DataFiles f(sd.dir);
    f.write_target("PC_ERMSE 0 1\nOPTIM CGRID2B (3) 3.7 4.3\n");
    Context ctx = master_context(kKRR, f, false, false);
    ctx.add("WEIGHTS", "0.5 -0.25 0.125");
    HostRun run(ctx, f);
    const ParamVec p = point({3.95});
    double loss = 0.0;
    REQUIRE_NOTHROW(loss = run.host(p));
    CHECK(loss < kFailScore);
    const Reference ref = reference_fit(ctx, f.hpo_cfg, p);
    CHECK(rel_diff(slot_weights(), ref.weights) < 1e-12);
    Host::reset_globals_for_test();
}

/* Pins: HPO_Host::analytical_gradient of a host B on a thread whose
 * trainer slot holds another host's trained model (host A, the same
 * configuration, evaluated at the same point) routes every parameter to
 * finite differences, as when the slot is empty; after B has evaluated
 * the point, its gradient is analytical (no parameter needs finite
 * differences). The setup is one the analytical gradient covers (OALGO 4,
 * fixed LAMBDA, BF_Linear, STRESS false, every weight fitted).
 * Before the fix: B's first gradient differentiated A's model (no
 * parameter routed to finite differences). */
TEST_CASE("A host does not differentiate another host's model",
          "[trainer_e2e][.]") {
    PeriodicTable::initialize();
    ScratchDir sd;
    DataFiles f(sd.dir);
    f.write_target("PC_ERMSE 0 1\nOPTIM CGRID2B (3) 3.7 4.3\n");
    Context a = master_context(kBLR, f, false, false);
    Context b = master_context(kBLR, f, false, false);
    const ParamVec p = point({3.95});
    {
        Host::reset_globals_for_test();
        HostRun run(a, f);
        REQUIRE(run.host(p) < kFailScore);
        ParamVec g;
        std::vector<int> needs_fd;
        REQUIRE(run.host.analytical_gradient(p, g, needs_fd));
        REQUIRE(needs_fd.empty());   // the setup has an analytical gradient
    }
    Host::reset_globals_for_test();
    HostRun run(b, f);
    ParamVec g;
    std::vector<int> needs_fd;
    REQUIRE(run.host.analytical_gradient(p, g, needs_fd));
    CHECK(needs_fd.size() == 1u);
    REQUIRE(run.host(p) < kFailScore);
    needs_fd.clear();
    REQUIRE(run.host.analytical_gradient(p, g, needs_fd));
    CHECK(needs_fd.empty());
    Host::reset_globals_for_test();
}
