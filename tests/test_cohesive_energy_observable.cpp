// test_cohesive_energy_observable.cpp
//
// Coverage for CohesiveEnergyObservable.
//
// What these tests check:
//   1. Construction validation (bad args throw cleanly).
//   2. Hot-var seeding: default → midpoint, explicit → that value,
//      out-of-bounds → clamped.
//   3. Closed-form dO/dW matches W-FD to high precision when the
//      relaxer is bypassed (maxiter=0). This is the analytical contract
//      we ship under PC_OBSERVABLE — at fixed atomic state the formula
//      `dE_coh/dW = −(dE_bulk/dW − N · dE_iso/dW)/N` must hold byte-for-
//      byte (modulo float noise).
//   4. Hot-var updates: with relax enabled, a0_current changes after
//      evaluate.
//   5. Neighbour lists: the E_coh of a relaxation whose cell
//      changed equals a fresh neighbour-list evaluation of the relaxed bulk
//      (the relaxer's cached list must carry the image displacements of
//      the current cell); and --supercell 1 1 1 gives the E_coh of 4 4 4.
//   6. --fmax-converged is the force tolerance of the atoms; --ptol, in
//      GPa, the pressure tolerance of the cell: it loosens and tightens its
//      relaxation.
//   7. --unconverged force-scaled reports the cohesive energy of the cell
//      a relaxation that does not converge ends at, moved away from the
//      target by the slope times the strain force of the cell there.
//   8. The relaxed value is that of the crystal at zero pressure: dO/dW is
//      its derivative, and the value does not depend on the start a0 (the
//      Ta potential data/ta/ta_md_blip.tadah against LAMMPS, also at a
//      --ptol far below the default, and the DM_REAM potential
//      data/ta/ta2_refit_seed.tadah, whose energy is noisy).

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/element.h>
#include <tadah/core/periodic_table.h>
#include <tadah/core/registry.h>
#include <tadah/hpo/observables/cohesive_energy.h>
#include <tadah/hpo/observables/observable_parser.h>
#include <tadah/mlip/cell_builder.h>
#include <tadah/mlip/descriptors_calc.h>
#include <tadah/mlip/design_matrix/functions/dm_function_base.h>
#include <tadah/mlip/models/m_tadah_base.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_transformations.h>
#include <tadah/mlip/trainer.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/models/dc_selector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <memory>
#include <string>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using tadah::LossFunction;
using tadah::core::Context;
using tadah::core::PeriodicTable;
using tadah::hpo::CohesiveEnergyObservable;
using tadah::hpo::ObservableEvalResult;
using tadah::mlip::NNFinder;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::Trainer;
using tadah::mlip::memory::MLIPWorkspaceManager;

Context make_ta_context(double rcut, const std::vector<double>& cgrid,
                        const std::vector<double>& sgrid)
{
    Context c;
    c.add("ATOM",     "Ta");
    c.add("WATOM",    1);
    c.add("INIT2B",   "true");
    c.add("INITMB",   "false");
    c.add("TYPE2B",   "D2_BP");
    c.add("TYPE2B",   static_cast<int>(cgrid.size()));
    c.add("TYPE2B",   static_cast<int>(sgrid.size()));
    c.add("TYPE2B",   "Ta"); c.add("TYPE2B", "Ta");
    c.add("TYPEMB",   "DM_Dummy");
    c.add("TYPEMB",   "Ta"); c.add("TYPEMB", "Ta");
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < cgrid.size(); ++i) {
            if (i) ss << " "; ss << cgrid[i];
        }
        c.add("CGRID2B", ss.str());
    }
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < sgrid.size(); ++i) {
            if (i) ss << " "; ss << sgrid[i];
        }
        c.add("SGRID2B", ss.str());
    }
    c.add("RCTYPE2B", "Cut_Cos");
    c.add("RCTYPEMB", "Cut_Cos");
    c.add("RCUT2B",   rcut);
    c.add("RCUTMB",   rcut);
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
    c.add("LAMBDA",   1e-3);
    c.add("OALGO",    4);
    return c;
}

StructureDB make_ta_train_db() {
    StructureDB db;
    const std::vector<double> a0s = {3.20, 3.30, 3.40};
    db.dbidx = {0, a0s.size()};
    db.structures.reserve(a0s.size());
    for (std::size_t i = 0; i < a0s.size(); ++i) {
        auto st = tadah::mlip::cells::bcc("Ta", a0s[i]);
        tadah::mlip::StructureTransformation::replicate(st, 2, 2, 2);
        st.energy = -0.5 * static_cast<double>(i + 1);
        db.structures.push_back(st);
    }
    return db;
}

/* n x n x n copies of @p st along its cell rows, built here (independent of
 * StructureTransformation::replicate): rows scaled by n, copy (x, y, z) of
 * atom i at r_i + x a0 + y a1 + z a2. */
Structure supercell_along_rows(const Structure& st, int n) {
    Structure out = st;
    out.atoms.clear();
    out.nn.reset();
    for (int k = 0; k < 3; ++k)
        for (int c = 0; c < 3; ++c) out.cell(k, c) = st.cell(k, c) * n;
    for (int x = 0; x < n; ++x)
        for (int y = 0; y < n; ++y)
            for (int z = 0; z < n; ++z)
                for (auto at : st.atoms) {
                    for (int c = 0; c < 3; ++c)
                        at.position[c] += x * st.cell(0, c) +
                                          y * st.cell(1, c) + z * st.cell(2, c);
                    out.atoms.push_back(at);
                }
    return out;
}

/* 2x2x2 bcc cells, a = 3.0 ... 3.6, labelled with E/atom = -8 + 5 (a - 3.3)^2:
 * a D2_BP model fitted to them has a bcc energy minimum inside the hot-var
 * range [2.7, 3.9] (near a = 3.34 for the grids used below), so an Iso
 * relaxation from a0 = 3.27 moves the cell by about 2%. */
StructureDB make_parabola_train_db() {
    StructureDB db;
    const std::vector<double> a0s = {3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6};
    db.dbidx = {0, a0s.size()};
    for (double a : a0s) {
        Structure st = supercell_along_rows(tadah::mlip::cells::bcc("Ta", a), 2);
        st.energy = static_cast<double>(st.natoms()) *
                    (-8.0 + 5.0 * (a - 3.3) * (a - 3.3));
        db.structures.push_back(st);
    }
    return db;
}

/* |dE/deps| / (3V) of @p st, the magnitude of its pressure in eV/A^3, by a
 * central difference of the energy in the isotropic strain eps (cell and
 * positions scaled by 1 +- 1e-5), with neighbour lists built afresh at the
 * cutoff of @p ctx. */
double pressure_of(tadah::mlip::M_Tadah_Base& model, tadah::mlip::DC_Base& dc,
                   const Context& ctx, const Structure& st) {
    const double h = 1.0e-5;
    auto energy = [&](double s) {
        Structure x = st;
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j) x.cell(i, j) = s * st.cell(i, j);
        for (auto& at : x.atoms)
            for (int k = 0; k < 3; ++k) at.position[k] *= s;
        x.nn.reset();
        NNFinder().calc(x, ctx);
        return model.epredict(dc.calc(x));
    };
    return std::abs(energy(1.0 + h) - energy(1.0 - h)) / (2.0 * h) /
           (3.0 * st.get_volume());
}

/* A Ta potential of data/ta (ta_md_blip.tadah by default), loaded as
 * tadah predict loads a potential (its descriptors, DescriptorsCalc and the
 * model of its MODEL line), with FORCE true, which the relaxation's forces
 * need (tadah hpo sets it for PC_OBSERVABLE). */
class TaMdPotential {
public:
    explicit TaMdPotential(const std::string& file = "ta_md_blip.tadah")
        : ctx_(load(file)),
          dcs_(ctx_),
          dc_(ctx_, *dcs_.d2b, *dcs_.dmb, *dcs_.c2b, *dcs_.cmb),
          fb_(tadah::core::factory<tadah::mlip::DM_Function_Base,
                                   const Context&>(
              ctx_.get<std::string>("MODEL", 1), ctx_)),
          model_(tadah::core::factory<tadah::mlip::M_Tadah_Base,
                                      tadah::mlip::DM_Function_Base&,
                                      Context&>(
              ctx_.get<std::string>("MODEL", 0), *fb_, ctx_)) {}

    tadah::mlip::M_Tadah_Base& model() { return *model_; }
    tadah::mlip::DC_Base& dc() { return dc_; }
    const Context& context() const { return ctx_; }

private:
    static Context load(const std::string& file) {
        Context c = tadah::core::CoreUserConfigParser().parse(
            std::string(TADAH_HPO_TA_DATA_DIR) + "/" + file);
        c.remove("FORCE");
        c.add("FORCE", "true");
        return c;
    }

    Context ctx_;
    tadah::models::DC_Selector dcs_;
    tadah::mlip::DescriptorsCalc<> dc_;
    std::unique_ptr<tadah::mlip::DM_Function_Base> fb_;
    std::unique_ptr<tadah::mlip::M_Tadah_Base> model_;
};

/* The cohesive energy -(E_bulk - N E_iso) / N of the n x n x n bcc cell of
 * lattice constant @p a0 and of the isolated atom of @p species, with
 * neighbour lists built afresh at the cutoff of @p ctx. */
double ecoh_at(tadah::mlip::M_Tadah_Base& model, tadah::mlip::DC_Base& dc,
               const Context& ctx, const std::string& species, double a0,
               int n) {
    Structure bulk = tadah::mlip::cells::bcc(species, a0);
    tadah::mlip::StructureTransformation::replicate(bulk, n, n, n);
    Structure iso = tadah::mlip::cells::isolated_atom(species);
    NNFinder().calc(bulk, ctx);
    NNFinder().calc(iso, ctx);
    const double N = static_cast<double>(bulk.natoms());
    return -(model.epredict(dc.calc(bulk)) - N * model.epredict(dc.calc(iso))) /
           N;
}

/* E_coh = -(E_bulk - N E_iso) / N of the observable's relaxed bulk and
 * isolated atom, with neighbour lists built afresh here at the model's
 * cutoff. */
double fresh_ecoh(Trainer& host, const ObservableEvalResult& r) {
    REQUIRE(r.energy_terms.size() == 2u);
    Structure bulk = r.energy_terms[0].structure;
    Structure iso  = r.energy_terms[1].structure;
    REQUIRE(iso.natoms() == 1u);
    bulk.nn.reset();
    iso.nn.reset();
    NNFinder().calc(bulk, host.context);
    NNFinder().calc(iso, host.context);
    const double E_bulk = host.model->epredict(host.dc.calc(bulk));
    const double E_iso  = host.model->epredict(host.dc.calc(iso));
    const double N = static_cast<double>(bulk.natoms());
    return -(E_bulk - N * E_iso) / N;
}

}  // namespace

TEST_CASE("CohesiveEnergyObservable constructor validates",
          "[hpo][observable][cohesive_energy][ctor]")
{
    PeriodicTable::initialize();

    auto valid = []() {
        CohesiveEnergyObservable::Config cfg;
        cfg.species = "Ta";
        cfg.nx = cfg.ny = cfg.nz = 2;
        cfg.target = 8.1;
        cfg.weight = 1000.0;
        cfg.a0_min = 2.7;
        cfg.a0_max = 3.9;
        cfg.a0_initial = 3.30;
        return cfg;
    };

    CHECK_NOTHROW(CohesiveEnergyObservable{valid()});

    {
        auto c = valid(); c.species = "";
        CHECK_THROWS_AS(CohesiveEnergyObservable{c}, std::invalid_argument);
    }
    {
        auto c = valid(); c.nx = 0;
        CHECK_THROWS_AS(CohesiveEnergyObservable{c}, std::invalid_argument);
    }
    {
        auto c = valid(); c.a0_min = 3.5; c.a0_max = 3.5;
        CHECK_THROWS_AS(CohesiveEnergyObservable{c}, std::invalid_argument);
    }
    {
        auto c = valid(); c.weight = -1.0;
        CHECK_THROWS_AS(CohesiveEnergyObservable{c}, std::invalid_argument);
    }
}

TEST_CASE("CohesiveEnergyObservable seeds hot-var correctly",
          "[hpo][observable][cohesive_energy][hotvar]")
{
    PeriodicTable::initialize();

    CohesiveEnergyObservable::Config cfg;
    cfg.species = "Ta";
    cfg.nx = cfg.ny = cfg.nz = 2;
    cfg.target = 8.1; cfg.weight = 1000.0;
    cfg.a0_min = 2.7; cfg.a0_max = 3.9;

    /* No explicit initial → midpoint. */
    {
        CohesiveEnergyObservable obs(cfg);
        CHECK(std::abs(obs.start_a0() - 3.3) < 1e-12);
    }
    /* Explicit initial inside bounds. */
    {
        auto c = cfg; c.a0_initial = 3.50;
        CohesiveEnergyObservable obs(c);
        CHECK(std::abs(obs.start_a0() - 3.50) < 1e-12);
    }
    /* Pins: an initial outside the window is refused (the parser refuses
     * the line).
     * Before the fix: it was moved onto the nearer bound, 3.9 and 2.7,
     * without a word. */
    {
        auto c = cfg; c.a0_initial = 4.5;   /* > a0_max */
        CHECK_THROWS_AS(CohesiveEnergyObservable{c}, std::invalid_argument);
    }
    {
        auto c = cfg; c.a0_initial = 2.0;   /* < a0_min */
        CHECK_THROWS_AS(CohesiveEnergyObservable{c}, std::invalid_argument);
    }
}

TEST_CASE("CohesiveEnergyObservable dO/dW matches W-FD without relaxation",
          "[hpo][observable][cohesive_energy][w_grad]")
{
    PeriodicTable::initialize();

    MLIPWorkspaceManager ws;
    const double rcut = 4.5;
    Context ctx = make_ta_context(rcut, {2.5, 3.5}, {2.0, 3.0});
    Trainer host(ctx, ws);
    auto train_db = make_ta_train_db();
    host.train(train_db);

    CohesiveEnergyObservable::Config cfg;
    cfg.species = "Ta";
    cfg.nx = cfg.ny = cfg.nz = 2;
    cfg.target = 8.1;
    cfg.weight = 1000.0;
    cfg.a0_min = 2.7;
    cfg.a0_max = 3.9;
    cfg.a0_initial = 3.30;
    cfg.relax_maxiter = 0;        /* skip relax — fixed cell → clean FD parity */
    /* A pressure tolerance, in GPa, that the cell meets, so the
     * iteration-0 check accepts it as it is (at the default, 1e-4 GPa,
     * the cell of maxiter 0 is unconverged and gives no dO/dW). */
    cfg.relax_ptol = 1.0e9;

    CohesiveEnergyObservable obs(cfg);
    ObservableEvalResult r0 = obs.evaluate(*host.model, host.dc, host.context);

    REQUIRE(r0.dO_dW.size() == host.model->get_weights().size());
    REQUIRE(!r0.needs_adjoint);

    const tadah::core::t_type w0 = host.model->get_weights();
    const double h = 1e-5;

    /* FD reference: re-evaluate at W ± h·e_c. Each call uses the SAME
     * hot-var seed (we re-seed by re-constructing obs each iter to keep
     * the test independent of the relaxer's warmstart cache). */
    double max_rel = 0.0;
    for (std::size_t c = 0; c < w0.size(); ++c) {
        tadah::core::t_type wp = w0;  wp(c) += h;
        host.model->set_weights(wp);
        CohesiveEnergyObservable obs_p(cfg);
        ObservableEvalResult rp = obs_p.evaluate(*host.model, host.dc, host.context);

        tadah::core::t_type wm = w0;  wm(c) -= h;
        host.model->set_weights(wm);
        CohesiveEnergyObservable obs_m(cfg);
        ObservableEvalResult rm = obs_m.evaluate(*host.model, host.dc, host.context);

        const double fd = (rp.value - rm.value) / (2.0 * h);
        const double anal = r0.dO_dW(c);
        const double scale = std::max({std::abs(fd), std::abs(anal), 1e-9});
        const double rel = std::abs(fd - anal) / scale;
        max_rel = std::max(max_rel, rel);
        INFO("c=" << c << "  anal=" << anal << "  fd=" << fd
             << "  rel=" << rel);
        CHECK(rel < 1e-5);
    }
    INFO("max rel = " << max_rel);

    host.model->set_weights(w0);
}

TEST_CASE("CohesiveEnergyObservable updates hot-var after relaxation",
          "[hpo][observable][cohesive_energy][hotvar][relax]")
{
    PeriodicTable::initialize();

    MLIPWorkspaceManager ws;
    const double rcut = 4.5;
    Context ctx = make_ta_context(rcut, {2.5, 3.5}, {2.0, 3.0});
    Trainer host(ctx, ws);
    auto train_db = make_ta_train_db();
    host.train(train_db);

    CohesiveEnergyObservable::Config cfg;
    cfg.species = "Ta";
    cfg.nx = cfg.ny = cfg.nz = 2;
    cfg.target = 8.1;
    cfg.weight = 1000.0;
    cfg.a0_min = 2.7;
    cfg.a0_max = 3.9;
    cfg.a0_initial = 3.10;        /* start away from any plausible minimum */
    cfg.eq_fmax_threshold = 1.0e-2;
    cfg.relax_maxiter = 500;

    CohesiveEnergyObservable obs(cfg);
    const double a0_before = obs.start_a0();
    ObservableEvalResult r = obs.evaluate(*host.model, host.dc, host.context);

    REQUIRE(r.hotvar_updated_value.has_value());
    INFO("a0 before = " << a0_before
         << "  a0 after = " << *r.hotvar_updated_value
         << "  Ecoh = " << r.value);

    /* Pins: a0 changed under relaxation (synthetic potential, but the
     * relaxer's Iso DOF still moves it): the a0 handed on differs from the
     * start 3.10 and stays inside the window, and the observable's start
     * is still 3.10. The observable keeps none of it: with the relaxation
     * capped at 2 iterations, where the start matters (from the a0 the
     * first evaluation handed on, it ends at another a0), a second
     * evaluation from the start gives the first one's value and a0, bit
     * for bit (the host keeps the relaxed a0 per chain).
     * Before the fix: the observable kept the relaxed a0 as its own start
     * (current_a0() was the relaxed value), one value for every chain: the
     * second evaluation started from the first one's a0 and ended at the
     * a0 of the evaluation from it. (The converged relaxation alone could
     * not tell: from its relaxed a0 it stops at once, at the same a0.) */
    CHECK(*r.hotvar_updated_value != a0_before);
    CHECK(*r.hotvar_updated_value >= cfg.a0_min);
    CHECK(*r.hotvar_updated_value <= cfg.a0_max);
    CHECK(obs.start_a0() == a0_before);

    auto bits = [](double x) {
        std::uint64_t b = 0;
        std::memcpy(&b, &x, sizeof b);
        return b;
    };
    CohesiveEnergyObservable::Config capped_cfg = cfg;
    capped_cfg.relax_maxiter = 2;
    CohesiveEnergyObservable capped(capped_cfg);
    const ObservableEvalResult first =
        capped.evaluate(*host.model, host.dc, host.context);
    const ObservableEvalResult again =
        capped.evaluate(*host.model, host.dc, host.context);
    REQUIRE(first.hotvar_updated_value.has_value());
    REQUIRE(again.hotvar_updated_value.has_value());
    const ObservableEvalResult from_first = capped.evaluate(
        *host.model, host.dc, host.context, first.hotvar_updated_value);
    REQUIRE(from_first.hotvar_updated_value.has_value());
    INFO(std::hexfloat << "first: value " << first.value << ", a0 "
                       << *first.hotvar_updated_value << "; again: value "
                       << again.value << ", a0 "
                       << *again.hotvar_updated_value
                       << "; from the first a0: a0 "
                       << *from_first.hotvar_updated_value);
    /* Precondition: the start of the capped relaxation changes its a0. */
    REQUIRE(bits(*from_first.hotvar_updated_value) !=
            bits(*first.hotvar_updated_value));
    CHECK(bits(again.value) == bits(first.value));
    CHECK(bits(*again.hotvar_updated_value) ==
          bits(*first.hotvar_updated_value));
}

/* The relaxed bulk's neighbour list is the relaxer's cached one (built at
 * rcut + skin and reused while it covers the cutoff). Its image
 * displacements must be those of the relaxed cell: before the exact-image
 * fix a cache hit kept the displacements of the cell the list was built
 * for, so after an Iso step the images sat where the old cell put them --
 * a perfect bcc lattice felt spurious forces (the relaxation stalls on
 * them) and E_coh was computed from misplaced images.
 *
 * relax_etol makes the observable report the E_coh it computes for the bulk
 * the relaxation ends at, whether it stops on the force criterion (exact
 * lists: every atomic force ~1e-15 on a perfect lattice, below the
 * eq_fmax_threshold of 1e-4, and the pressure below relax_ptol, 1e-4 GPa
 * by default) or on the energy criterion; that value must be the relaxed
 * bulk's energy. (The relaxation stopped at a force tolerance of its own,
 * 1e-4, while a lenient eq_fmax_threshold, 1e3, judged it; the
 * --fmax-converged tolerance does both for the atoms.) */
TEST_CASE("CohesiveEnergyObservable: relaxed E_coh equals a fresh "
          "neighbour-list evaluation of the relaxed bulk",
          "[hpo][observable][cohesive_energy][nn]")
{
    PeriodicTable::initialize();

    MLIPWorkspaceManager ws;
    Context ctx = make_ta_context(4.5, {2.4, 3.0, 3.6}, {2.0, 2.0, 2.0});
    Trainer host(ctx, ws);
    auto train_db = make_parabola_train_db();
    host.train(train_db);

    CohesiveEnergyObservable::Config cfg;
    cfg.species = "Ta";
    cfg.nx = cfg.ny = cfg.nz = 2;
    cfg.target = 8.0;
    cfg.weight = 1.0;
    cfg.a0_min = 2.7;
    cfg.a0_max = 3.9;
    cfg.a0_initial = 3.27;           /* about 2% below the model's minimum */
    cfg.eq_fmax_threshold = 1.0e-4;  /* the atoms; the cell relaxes to relax_ptol: several Iso steps */
    cfg.relax_etol = 1.0e-12;
    cfg.relax_maxiter = 500;

    CohesiveEnergyObservable obs(cfg);
    const ObservableEvalResult r = obs.evaluate(*host.model, host.dc, host.context);

    REQUIRE(r.hotvar_updated_value.has_value());
    const double relaxed_a0 = *r.hotvar_updated_value;
    INFO("a0 " << *cfg.a0_initial << " -> " << relaxed_a0
         << ", E_coh " << r.value);
    /* Precondition: the relaxation changed the cell (accepted Iso steps
     * after the list was built), so the cached list's cell is not the
     * relaxed one. Each Iso step changes the scale by at most 0.1%. */
    REQUIRE(std::abs(relaxed_a0 / *cfg.a0_initial - 1.0) > 5.0e-3);
    /* The observable's own value, not the unconverged penalty. */
    REQUIRE_FALSE(r.value_is_constant);
    REQUIRE(std::isfinite(r.value));
    REQUIRE(r.energy_terms.size() == 2u);
    REQUIRE(r.energy_terms[0].structure.cell(0, 0) ==
            Approx(2.0 * relaxed_a0).epsilon(1e-12));

    const double ecoh = fresh_ecoh(host, r);
    INFO("fresh-list E_coh " << ecoh);
    CHECK(std::abs(r.value - ecoh) <= 1.0e-10 * std::max(1.0, std::abs(ecoh)));
}

/* A guard, expected to pass before and after the exact-image fix: the E_coh
 * of the 2-atom bcc cell (--supercell 1 1 1) equals that of 4 4 4 at the
 * same a0, without relaxation (relax_maxiter 0; an eq_fmax_threshold, the
 * --fmax-converged force tolerance, that no force reaches, and a
 * relax_ptol, the --ptol pressure tolerance, that no pressure reaches, so
 * the iteration-0 check accepts the cell as it is).
 * At rcut 6 and a0 3.3 (rcut/a0 = 1.82) a list built at the model's cutoff
 * would miss images of the 2-atom cell before the fix, but the relaxer
 * builds its list at rcut + skin (2 A), wide enough that no pair inside
 * rcut is lost. */
TEST_CASE("CohesiveEnergyObservable: supercell 1 1 1 and 4 4 4 give the "
          "same E_coh",
          "[hpo][observable][cohesive_energy][nn][supercell]")
{
    PeriodicTable::initialize();

    MLIPWorkspaceManager ws;
    Context ctx = make_ta_context(6.0, {2.4, 3.0, 3.6, 4.5}, {2.0, 2.0, 2.0, 2.0});
    Trainer host(ctx, ws);
    auto train_db = make_parabola_train_db();
    host.train(train_db);

    auto ecoh = [&](int n) {
        CohesiveEnergyObservable::Config cfg;
        cfg.species = "Ta";
        cfg.nx = cfg.ny = cfg.nz = n;
        cfg.target = 8.0;
        cfg.weight = 1.0;
        cfg.a0_min = 2.7;
        cfg.a0_max = 3.9;
        cfg.a0_initial = 3.3;
        cfg.relax_maxiter = 0;
        cfg.eq_fmax_threshold = 1.0e6;
        cfg.relax_ptol = 1.0e9;
        CohesiveEnergyObservable obs(cfg);
        const ObservableEvalResult r =
            obs.evaluate(*host.model, host.dc, host.context);
        CAPTURE(n, r.value);
        REQUIRE_FALSE(r.value_is_constant);
        REQUIRE(std::isfinite(r.value));
        REQUIRE(r.energy_terms.size() == 2u);
        REQUIRE(r.energy_terms[0].structure.natoms() ==
                static_cast<std::size_t>(2 * n * n * n));
        REQUIRE(r.hotvar_updated_value.has_value());
        CHECK(*r.hotvar_updated_value == 3.3);   /* not relaxed */
        return r.value;
    };

    const double e1 = ecoh(1);
    const double e4 = ecoh(4);
    INFO("E_coh 1x1x1 " << e1 << ", 4x4x4 " << e4);
    CHECK(std::abs(e1 - e4) <= 1.0e-10 * std::max(1.0, std::abs(e4)));
}

/* Pins: the hot-var a0 that an evaluation hands on (hotvar_updated_value)
 * is the a0 it started from times the Iso
 * relaxation's scale factor (NativeRelaxer::Result::scale_factor), the
 * quantity the early-reject predicate already uses, and not the relaxed
 * cell(0,0) divided by nx. An evaluation that does not move the cell
 * (relax_maxiter 0, an eq_fmax_threshold, the --fmax-converged force
 * tolerance, that no force reaches, and a relax_ptol, the --ptol pressure
 * tolerance, that no pressure reaches, so the iteration-0 check accepts
 * the cell with scale factor 1) therefore hands
 * on its a0 bit for bit, whatever the supercell.
 *
 * a0 = 3.3 (0x1.a666666666666p+1) with --supercell 3 3 3: replicate()
 * stores cell(0,0) = fl(3 x 3.3) = 0x1.3ccccccccccccp+3, and that divided
 * by 3 is 0x1.a666666666665p+1 = 3.2999999999999994, one ulp below 3.3.
 * A power-of-two supercell (1, 2, 4, 8) multiplies and divides exactly,
 * which is why the 1 1 1 and 4 4 4 case above passes either way; for
 * a0 = 3.3 the supercells 3 and 6 are the ones up to 8 that change it.
 * The expected values are written as literals, not computed here, so the
 * test does no arithmetic that -ffast-math could rewrite.
 *
 * Before the fix: new_a0 was bulk.cell(0,0) / nx, so the observable's a0
 * (which it kept then) and hotvar_updated_value were 3.2999999999999994
 * (0x1.a666666666665p+1)
 * instead of 3.3 (0x1.a666666666666p+1). */
TEST_CASE("CohesiveEnergyObservable: an evaluation that does not move the "
          "cell hands on its a0 bit for bit with --supercell 3 3 3",
          "[hpo][observable][cohesive_energy][hotvar][supercell]")
{
    PeriodicTable::initialize();

    MLIPWorkspaceManager ws;
    Context ctx = make_ta_context(4.5, {2.5, 3.5}, {2.0, 3.0});
    Trainer host(ctx, ws);
    auto train_db = make_ta_train_db();
    host.train(train_db);

    CohesiveEnergyObservable::Config cfg;
    cfg.species = "Ta";
    cfg.nx = cfg.ny = cfg.nz = 3;
    cfg.target = 8.0;
    cfg.weight = 1.0;
    cfg.a0_min = 2.7;
    cfg.a0_max = 3.9;
    cfg.a0_initial = 3.3;
    cfg.relax_maxiter = 0;
    cfg.eq_fmax_threshold = 1.0e6;
    cfg.relax_ptol = 1.0e9;

    CohesiveEnergyObservable obs(cfg);
    REQUIRE(obs.start_a0() == 0x1.a666666666666p+1);   /* 3.3 */
    const ObservableEvalResult r =
        obs.evaluate(*host.model, host.dc, host.context);

    /* The observable's own value: the relaxation accepted the cell at
     * iteration 0, so the relaxed bulk is the bulk as built. */
    REQUIRE_FALSE(r.value_is_constant);
    REQUIRE(r.energy_terms.size() == 2u);
    const Structure& bulk = r.energy_terms[0].structure;
    REQUIRE(bulk.natoms() == 54u);
    /* Fixture check: the cell is fl(3 x 3.3), the value whose third is
     * one ulp below 3.3. */
    REQUIRE(bulk.cell(0, 0) == 0x1.3ccccccccccccp+3);

    REQUIRE(r.hotvar_updated_value.has_value());
    INFO("hotvar_updated_value " << std::hexfloat << *r.hotvar_updated_value
         << ", expected 0x1.a666666666666p+1 (3.3)");
    CHECK(*r.hotvar_updated_value == 0x1.a666666666666p+1);
}

/* Pins: --fmax-converged is the force tolerance of the atoms and of the
 * convergence test, and the cell relaxes to the pressure tolerance
 * whatever it is. From a0 = 3.1, about 7% below the model's minimum, with
 * --fmax-converged 100, the default 0.1 and 1e-4, each relaxation
 * converges (the value is not the --unconverged one) and all three hand
 * on the same a0, bit for bit, away from 3.1: the atomic forces of the
 * bcc cell vanish by symmetry, so no force tolerance ends the relaxation
 * of its cell (expected change: the option bounded the pressure as well,
 * and these three a0 differed). With --fmax-converged 1e-300, below the
 * atomic forces the bcc cell has from rounding, no relaxation meets it:
 * the term reports the --unconverged value (--maxiter 200 --maxeval 2000
 * bound the relaxation that cannot converge).
 * Before the fix: --fmax-converged bounded the atomic forces and the
 * pressure |dE/deps| / (3V) alike: with 100, which the start's pressure
 * meets, the relaxation stopped at once and handed on 3.1 bit for bit;
 * 0.1 and 1e-4 ended at two other a0. The 1e-300 check passes as well (a
 * pin: the code compared --fmax-converged with the largest atomic force;
 * without the check the case passed with the option dropped, the
 * relaxer's ftol fixed at 0.1 and the comparison removed, where 1e-300
 * gives the relaxed value). */
TEST_CASE("CohesiveEnergyObservable: --fmax-converged is the force tolerance "
          "of the atoms",
          "[hpo][observable][cohesive_energy][fmax_converged]")
{
    PeriodicTable::initialize();

    MLIPWorkspaceManager ws;
    Context ctx = make_ta_context(4.5, {2.4, 3.0, 3.6}, {2.0, 2.0, 2.0});
    Trainer host(ctx, ws);
    auto train_db = make_parabola_train_db();
    host.train(train_db);

    auto relaxed_a0 = [&](const std::string& fmax) {
        auto obs = tadah::hpo::parse_pc_observable_line(
            "cohesive_energy --species Ta --target 8 --weight 1 "
            "--supercell 2 2 2 --hotvar a0 2.7 3.9 3.1 --maxiter 2000 "
            "--maxeval 20000" +
            (fmax.empty() ? std::string() : " --fmax-converged " + fmax));
        const ObservableEvalResult r =
            obs->evaluate(*host.model, host.dc, host.context);
        INFO("--fmax-converged " << fmax << ": value " << r.value);
        REQUIRE_FALSE(r.value_is_constant);
        REQUIRE(r.hotvar_updated_value.has_value());
        return *r.hotvar_updated_value;
    };
    const double loose = relaxed_a0("100");
    const double standard = relaxed_a0("");
    const double tight = relaxed_a0("1e-4");
    INFO(std::hexfloat << "a0: 100 -> " << loose << ", 0.1 -> " << standard
                       << ", 1e-4 -> " << tight);
    CHECK(standard != 3.1);
    CHECK(loose == standard);
    CHECK(tight == standard);

    auto unreachable = tadah::hpo::parse_pc_observable_line(
        "cohesive_energy --species Ta --target 8 --weight 1 "
        "--supercell 2 2 2 --hotvar a0 2.7 3.9 3.1 --maxiter 200 "
        "--maxeval 2000 --fmax-converged 1e-300");
    const ObservableEvalResult r =
        unreachable->evaluate(*host.model, host.dc, host.context);
    INFO("--fmax-converged 1e-300: value " << r.value);
    CHECK(r.value_is_constant);
    CHECK(r.value == 100.0);
}

/* Pins: --ptol is the pressure tolerance of the relaxation of the cell, in
 * GPa. From a0 = 3.1, about 7% below the model's minimum (a pressure of
 * about 20 GPa): with --ptol 100, which the start meets, the relaxation
 * stops at once and hands on 3.1 bit for bit; with --ptol 1 it relaxes the
 * cell to a pressure below 1 GPa; with the default, 1e-4 GPa, it relaxes
 * further, to another a0, at a pressure below 1e-4 GPa. The pressure is
 * that of the relaxed bulk, by a central difference of its energy in the
 * isotropic strain. Each relaxation converges (the value is not the
 * --unconverged one).
 * Before the fix: each line with --ptol was refused ("PC_OBSERVABLE
 * cohesive_energy: unknown option '--ptol'"). */
TEST_CASE("CohesiveEnergyObservable: --ptol is the pressure tolerance of the "
          "relaxation of the cell",
          "[hpo][observable][cohesive_energy][ptol]")
{
    PeriodicTable::initialize();

    MLIPWorkspaceManager ws;
    Context ctx = make_ta_context(4.5, {2.4, 3.0, 3.6}, {2.0, 2.0, 2.0});
    Trainer host(ctx, ws);
    auto train_db = make_parabola_train_db();
    host.train(train_db);

    const double gpa = 6.241509074e-3;  /* 1 GPa in eV/A^3 */
    /* The a0 handed on, and the pressure of the relaxed bulk in GPa. */
    auto relax = [&](const std::string& ptol) {
        auto obs = tadah::hpo::parse_pc_observable_line(
            "cohesive_energy --species Ta --target 8 --weight 1 "
            "--supercell 2 2 2 --hotvar a0 2.7 3.9 3.1 --maxiter 2000 "
            "--maxeval 20000" +
            (ptol.empty() ? std::string() : " --ptol " + ptol));
        const ObservableEvalResult r =
            obs->evaluate(*host.model, host.dc, host.context);
        REQUIRE(r.hotvar_updated_value.has_value());
        REQUIRE(r.energy_terms.size() == 2u);
        const double p = pressure_of(*host.model, host.dc, host.context,
                                     r.energy_terms[0].structure) / gpa;
        INFO(std::setprecision(17) << "--ptol " << ptol << ": value "
             << r.value << ", a0 " << *r.hotvar_updated_value
             << ", pressure " << p << " GPa");
        CHECK_FALSE(r.value_is_constant);
        return std::make_pair(*r.hotvar_updated_value, p);
    };
    const auto [start_a0, start_p] = relax("100");
    const auto [one_a0, one_p] = relax("1");
    const auto [default_a0, default_p] = relax("");
    INFO(std::setprecision(17) << "a0: --ptol 100 -> " << start_a0 << " ("
         << start_p << " GPa), 1 -> " << one_a0 << " (" << one_p
         << " GPa), default -> " << default_a0 << " (" << default_p
         << " GPa)");
    REQUIRE(start_p > 1.0);
    CHECK(start_a0 == 3.1);
    CHECK(one_a0 != 3.1);
    CHECK(one_p < 1.0);
    CHECK(default_a0 != one_a0);
    CHECK(default_p < 1.0e-4);
}

/* Pins: the analytical dO/dW of the relaxed cohesive energy is the
 * derivative of the value the observable reports. With every relaxation
 * option at its default, from a0 = 3.1, about 7% below the model's
 * minimum (a pressure of about 20 GPa, above 0.1 eV/A^3), the central
 * difference of the reported value in each weight (step 1e-4, each
 * evaluation relaxing from 3.1) equals dO/dW of the evaluation at the
 * weights to 1e-5 of the largest |dO/dW| (5e-7 here; with --ptol 1e-2,
 * 7e-5). dO/dW is taken with the relaxed cell held fixed; at zero
 * pressure the change of the cell with the weights adds nothing to the
 * derivative of the energy.
 * Before the fix: passes as well (a pin: the relaxation stopped at the
 * first cell whose pressure was below 0.1 eV/A^3, after steps of the
 * largest strain a step takes, 0.1 %, so the cell it ended at did not
 * change with the weights, and the value was that of a fixed cell). */
TEST_CASE("CohesiveEnergyObservable dO/dW matches W-FD of the relaxed value",
          "[hpo][observable][cohesive_energy][w_grad][relax]")
{
    PeriodicTable::initialize();

    MLIPWorkspaceManager ws;
    Context ctx = make_ta_context(4.5, {2.4, 3.0, 3.6}, {2.0, 2.0, 2.0});
    Trainer host(ctx, ws);
    auto train_db = make_parabola_train_db();
    host.train(train_db);

    const std::string line =
        "cohesive_energy --species Ta --target 8 --weight 1 "
        "--supercell 2 2 2 --hotvar a0 2.7 3.9 3.1";
    auto evaluate = [&]() {
        auto obs = tadah::hpo::parse_pc_observable_line(line);
        return obs->evaluate(*host.model, host.dc, host.context);
    };
    const ObservableEvalResult r0 = evaluate();
    REQUIRE_FALSE(r0.value_is_constant);
    REQUIRE(r0.hotvar_updated_value.has_value());
    /* Precondition: the relaxation moved the cell. */
    REQUIRE(*r0.hotvar_updated_value != 3.1);
    const tadah::core::t_type w0 = host.model->get_weights();
    REQUIRE(r0.dO_dW.size() == w0.size());

    double largest = 0.0;
    for (std::size_t c = 0; c < w0.size(); ++c)
        largest = std::max(largest, std::abs(r0.dO_dW(c)));
    REQUIRE(largest > 0.0);

    const double h = 1e-4;
    double worst = 0.0;
    for (std::size_t c = 0; c < w0.size(); ++c) {
        tadah::core::t_type wp = w0;  wp(c) += h;
        host.model->set_weights(wp);
        const double vp = evaluate().value;
        tadah::core::t_type wm = w0;  wm(c) -= h;
        host.model->set_weights(wm);
        const double vm = evaluate().value;
        const double fd = (vp - vm) / (2.0 * h);
        const double diff = std::abs(fd - r0.dO_dW(c)) / largest;
        worst = std::max(worst, diff);
        INFO(std::setprecision(17) << "weight " << c << ": dO/dW "
             << r0.dO_dW(c) << ", central difference " << fd
             << ", difference / largest |dO/dW| " << diff);
        CHECK(diff <= 1e-5);
    }
    host.model->set_weights(w0);
    INFO("largest difference / largest |dO/dW|: " << worst);
}

/* Pins: --unconverged force-scaled reports the cohesive energy of the cell
 * the relaxation ends at, moved away from the target by slope x fmax, fmax
 * the larger of the largest atomic force and the strain force of the
 * cell, |dE/deps| / (3V), at the end (the measures the relaxation stops
 * at). In the bcc cell the atomic forces vanish by symmetry, so the strain
 * force decides. A relaxation from a0 = 3.1, about 7% below the model's
 * minimum, capped at one iteration (--maxiter 1; the --unconverged const
 * value shows that it does not converge), ends at a cell whose cohesive
 * energy, computed here at the a0 it hands on, is below 8 and above 7:
 * with --target 8 the value is that energy less slope x the strain force
 * of the cell, computed here by the relaxer's central difference, with
 * --target 7 that energy plus it, and the distance from the energy doubles
 * with the slope (1000, 2000).
 * Before the fix: the value was the target plus slope x fmax, whatever
 * the cohesive energy of the cell (8 + 1000 x fmax with --target 8). */
TEST_CASE("CohesiveEnergyObservable: --unconverged force-scaled moves the "
          "cohesive energy of the final cell away from the target",
          "[hpo][observable][cohesive_energy][unconverged]")
{
    PeriodicTable::initialize();

    MLIPWorkspaceManager ws;
    Context ctx = make_ta_context(4.5, {2.4, 3.0, 3.6}, {2.0, 2.0, 2.0});
    Trainer host(ctx, ws);
    auto train_db = make_parabola_train_db();
    host.train(train_db);

    auto evaluate = [&](const std::string& target,
                        const std::string& unconverged) {
        auto obs = tadah::hpo::parse_pc_observable_line(
            "cohesive_energy --species Ta --target " + target +
            " --weight 1 --supercell 2 2 2 --hotvar a0 2.7 3.9 3.1 "
            "--maxiter 1 " + unconverged);
        return obs->evaluate(*host.model, host.dc, host.context);
    };

    /* Precondition: the capped relaxation does not converge. */
    const ObservableEvalResult constant =
        evaluate("8", "--unconverged const 100");
    REQUIRE(constant.value_is_constant);
    REQUIRE(constant.value == 100.0);
    REQUIRE(constant.hotvar_updated_value.has_value());

    const ObservableEvalResult below =
        evaluate("8", "--unconverged force-scaled 1000");
    const ObservableEvalResult below2 =
        evaluate("8", "--unconverged force-scaled 2000");
    const ObservableEvalResult above =
        evaluate("7", "--unconverged force-scaled 1000");
    REQUIRE_FALSE(below.value_is_constant);
    REQUIRE(below.hotvar_updated_value.has_value());
    REQUIRE(*below.hotvar_updated_value == *constant.hotvar_updated_value);

    /* The cell the relaxation ends at: the 2x2x2 bcc cell of a0 = 3.1
     * scaled to the a0 handed on; its cohesive energy and its strain force
     * by the relaxer's central difference (step 1e-5 in the strain). */
    const double a0_end = *below.hotvar_updated_value;
    Structure end = tadah::mlip::cells::bcc("Ta", 3.1);
    tadah::mlip::StructureTransformation::replicate(end, 2, 2, 2);
    const double scale = a0_end / 3.1;
    REQUIRE(scale != 1.0);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) end.cell(i, j) *= scale;
    for (auto& at : end.atoms)
        for (int k = 0; k < 3; ++k) at.position[k] *= scale;
    const double strain_force =
        pressure_of(*host.model, host.dc, host.context, end);
    const double ecoh =
        ecoh_at(*host.model, host.dc, host.context, "Ta", a0_end, 2);
    INFO(std::setprecision(17) << "a0 at the end " << a0_end
         << ", cohesive energy " << ecoh << " eV/atom, strain force "
         << strain_force << " eV/A^3; value " << below.value
         << " (target 8, slope 1000), " << below2.value
         << " (target 8, slope 2000), " << above.value
         << " (target 7, slope 1000)");
    REQUIRE(strain_force > 1.0e-4);
    REQUIRE(ecoh < 8.0);
    REQUIRE(ecoh > 7.0);
    CHECK(below.value == Approx(ecoh - 1000.0 * strain_force).epsilon(1e-6));
    CHECK(above.value == Approx(ecoh + 1000.0 * strain_force).epsilon(1e-6));
    CHECK(below2.value - ecoh ==
          Approx(2.0 * (below.value - ecoh)).epsilon(1e-9));
}

/* Pins: a relaxation that stops near zero pressure without converging
 * reports, under --unconverged force-scaled, about the cohesive energy of
 * its cell, not about the target. With data/ta/ta_md_blip.tadah (8.126
 * eV/atom relaxed) the 2-atom cell from a0 = 3.25, capped at --maxiter 17,
 * does not converge (the --unconverged const value) and ends at a
 * pressure between 1e-6 and 1e-2 eV/A^3 (3.1e-5). With --unconverged
 * force-scaled 1000 the value is the cohesive energy of that cell,
 * computed here at the a0 it hands on, plus 1000 x its strain force with
 * --target 5, and less it with --target 11, each to 1e-6 eV/atom (the
 * central differences here and in the relaxer, with lists built at other
 * cutoffs, differ by about 1e-10 eV/A^3): the loss of the term is at
 * least that of the cell's cohesive energy.
 * Before the fix: the values were the target plus 1000 x fmax, 6.4987
 * and 12.4987 (the relaxation then stopped at 1.5e-3 eV/A^3): with
 * --target 5 a loss below that of the cell's 8.126 eV/atom; a relaxation
 * stopped nearer zero pressure reported about the target itself. */
TEST_CASE("CohesiveEnergyObservable: --unconverged force-scaled near zero "
          "pressure reports about the cohesive energy of the cell",
          "[hpo][observable][cohesive_energy][unconverged][ta_md]")
{
    PeriodicTable::initialize();
    TaMdPotential pot;

    auto evaluate = [&](const std::string& target,
                        const std::string& unconverged) {
        auto obs = tadah::hpo::parse_pc_observable_line(
            "cohesive_energy --species Ta --target " + target +
            " --weight 1 --supercell 1 1 1 --hotvar a0 3.1 3.5 3.25 "
            "--maxiter 17 " + unconverged);
        return obs->evaluate(pot.model(), pot.dc(), pot.context());
    };
    /* Precondition: the capped relaxation does not converge. */
    const ObservableEvalResult constant =
        evaluate("5", "--unconverged const 100");
    REQUIRE(constant.value_is_constant);
    REQUIRE(constant.hotvar_updated_value.has_value());

    const ObservableEvalResult low =
        evaluate("5", "--unconverged force-scaled 1000");
    const ObservableEvalResult high =
        evaluate("11", "--unconverged force-scaled 1000");
    REQUIRE_FALSE(low.value_is_constant);
    REQUIRE(low.hotvar_updated_value.has_value());
    const double a0_end = *low.hotvar_updated_value;
    REQUIRE(a0_end == *constant.hotvar_updated_value);

    Structure end = tadah::mlip::cells::bcc("Ta", a0_end);
    const double strain_force =
        pressure_of(pot.model(), pot.dc(), pot.context(), end);
    const double ecoh =
        ecoh_at(pot.model(), pot.dc(), pot.context(), "Ta", a0_end, 1);
    INFO(std::setprecision(17) << "a0 at the end " << a0_end
         << ", cohesive energy " << ecoh << " eV/atom, strain force "
         << strain_force << " eV/A^3; value " << low.value
         << " (target 5), " << high.value << " (target 11)");
    REQUIRE(strain_force > 1.0e-6);
    REQUIRE(strain_force < 1.0e-2);
    CHECK(std::abs(low.value - (ecoh + 1000.0 * strain_force)) <= 1.0e-6);
    CHECK(std::abs(high.value - (ecoh - 1000.0 * strain_force)) <= 1.0e-6);
}

/* Pins: PC_OBSERVABLE cohesive_energy reports the cohesive energy of the
 * crystal relaxed at zero pressure, whatever lattice constant its
 * relaxation starts from. With the potential data/ta/ta_md_blip.tadah and
 * a line that sets only the required options (--hotvar a0 3.1 3.5
 * <start>; every relaxation option at its default), the starts 3.25,
 * 3.32011 and 3.38 A each give the value LAMMPS gives for the crystal
 * relaxed at zero pressure, 8.1263245850672 eV/atom (the file's header),
 * to 1e-6 eV/atom, and hand on its lattice constant, 3.305677525578418 A,
 * to 1e-5 A; the pressure of the relaxed bulk, by a central difference of
 * its energy in the isotropic strain, is below 1e-4 GPa (6.241509074e-7
 * eV/A^3). Each value is the observable's own (not the --unconverged
 * one).
 * Before the fix: the relaxation stopped at the first cell whose pressure
 * was below 0.1 eV/A^3 (16 GPa), the --fmax-converged default: from 3.25
 * after one step, at a0 3.25325 A (8.0857026 eV/atom, 0.0958 eV/A^3);
 * from 3.32011 at once (8.1231170 eV/atom, 0.0270 eV/A^3); from 3.38 at
 * 3.3631338 A (8.0770966 eV/atom, 0.0981 eV/A^3). */
TEST_CASE("CohesiveEnergyObservable: the relaxed cohesive energy does not "
          "depend on the start a0",
          "[hpo][observable][cohesive_energy][relax][ta_md]")
{
    PeriodicTable::initialize();
    TaMdPotential pot;

    const double lammps_ecoh = 8.1263245850672;
    const double lammps_a0 = 3.305677525578418;
    for (const char* start : {"3.25", "3.32011", "3.38"}) {
        auto obs = tadah::hpo::parse_pc_observable_line(
            std::string("cohesive_energy --species Ta --target 8.11398 "
                        "--weight 2.5e5 --hotvar a0 3.1 3.5 ") + start);
        const ObservableEvalResult r =
            obs->evaluate(pot.model(), pot.dc(), pot.context());
        REQUIRE(r.hotvar_updated_value.has_value());
        REQUIRE(r.energy_terms.size() == 2u);
        const double p = pressure_of(pot.model(), pot.dc(), pot.context(),
                                     r.energy_terms[0].structure);
        INFO(std::setprecision(17) << "start " << start << ": Ecoh "
             << r.value << " eV/atom, a0 handed on "
             << *r.hotvar_updated_value << " A, pressure of the relaxed bulk "
             << p << " eV/A^3");
        CHECK_FALSE(r.value_is_constant);
        CHECK(std::abs(r.value - lammps_ecoh) <= 1.0e-6);
        CHECK(std::abs(*r.hotvar_updated_value - lammps_a0) <= 1.0e-5);
        CHECK(p < 6.241509074e-7);
    }
}

/* Pins: --ptol reaches a tolerance whose energy is far below the rounding
 * error of the energy. With data/ta/ta_md_blip.tadah and --ptol 1e-7 (GPa,
 * 6.2e-10 eV/A^3, which leaves the energy about 3e-18 eV/atom above its
 * minimum, while an ulp of the energy of the 4x4x4 cell is 2e-15 eV/atom),
 * the starts 3.25, 3.32011 and 3.38 A of the 2-atom cell (--supercell 1 1 1)
 * and the start 3.25 A of the default 4x4x4 cell each give the observable's
 * own value, LAMMPS's 8.1263245850672 eV/atom to 1e-9, and hand on LAMMPS's
 * lattice constant, 3.305677525578418 A, to 2e-9 A (the default --ptol,
 * 1e-4 GPa, ends about 6e-8 A from it), at a pressure of the relaxed bulk,
 * by a central difference of its energy, below 1e-7 GPa.
 * Before the fix: the starts 3.25 and 3.38 of the 2-atom cell and 3.25 of
 * the 4x4x4 cell reported the --unconverged value, 100 (3.32011 converged):
 * the line search on the energy that moved the cell accepted steps that
 * left it where it was until the 5000 evaluations of --maxeval ran out
 * (about 45 s for the 4x4x4 cell). */
TEST_CASE("CohesiveEnergyObservable: --ptol far below the default relaxes "
          "the crystal to it",
          "[hpo][observable][cohesive_energy][relax][ta_md][ptol]")
{
    PeriodicTable::initialize();
    TaMdPotential pot;

    const double lammps_ecoh = 8.1263245850672;
    const double lammps_a0 = 3.305677525578418;
    const double ptol = 1.0e-7 * 6.241509074e-3;  /* 1e-7 GPa in eV/A^3 */
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"3.25", "1 1 1"}, {"3.32011", "1 1 1"}, {"3.38", "1 1 1"},
        {"3.25", "4 4 4"}};
    for (const auto& [start, supercell] : cases) {
        auto obs = tadah::hpo::parse_pc_observable_line(
            "cohesive_energy --species Ta --target 8.11398 --weight 2.5e5 "
            "--hotvar a0 3.1 3.5 " + start + " --supercell " + supercell +
            " --ptol 1e-7");
        const ObservableEvalResult r =
            obs->evaluate(pot.model(), pot.dc(), pot.context());
        REQUIRE(r.hotvar_updated_value.has_value());
        INFO(std::setprecision(17) << "start " << start << ", supercell "
             << supercell << ": value " << r.value << ", a0 handed on "
             << *r.hotvar_updated_value);
        CHECK_FALSE(r.value_is_constant);
        CHECK(std::abs(r.value - lammps_ecoh) <= 1.0e-9);
        CHECK(std::abs(*r.hotvar_updated_value - lammps_a0) <= 2.0e-9);
        if (r.energy_terms.size() == 2u) {
            const double p = pressure_of(pot.model(), pot.dc(), pot.context(),
                                         r.energy_terms[0].structure);
            INFO(std::setprecision(17) << "pressure of the relaxed bulk " << p
                 << " eV/A^3");
            CHECK(p < ptol);
        }
    }
}

/* Pins: the relaxed cohesive energy does not depend on the start a0 for a
 * potential whose energy is noisy. The energy of the 4x4x4 cell of the
 * DM_REAM potential data/ta/ta2_refit_seed.tadah near its minimum changes
 * by a few 1e-9 eV, up and down, between strains 1e-7 apart, more than
 * the energy a pressure of 1e-4 GPa leaves above the minimum (about 4e-10
 * eV in that cell); the central difference of the pressure resolves it
 * (a few 1e-9 eV over 2e-5 x 3V is about 2e-8 eV/A^3). With every
 * relaxation option at
 * its default (--ptol 1e-4 GPa), the starts 3.14123, 3.29123, 3.304 and
 * 3.25123 A each give the observable's own value, the four within 1e-9
 * eV/atom of each other, at a pressure of the relaxed bulk, by a central
 * difference of its energy, below 1e-4 GPa.
 * Before the fix: 3.14123, 3.29123 and 3.304 reported the --unconverged
 * value, 100: the line search on the energy that moved the cell failed,
 * every trial step above the noisy energy, at a pressure just above
 * --ptol (6.9e-7, 8.1e-7 and 1.8e-6 eV/A^3 against 6.2e-7). */
TEST_CASE("CohesiveEnergyObservable: the relaxed cohesive energy of a "
          "potential with a noisy energy does not depend on the start a0",
          "[hpo][observable][cohesive_energy][relax][ream]")
{
    PeriodicTable::initialize();
    TaMdPotential pot("ta2_refit_seed.tadah");

    std::vector<double> values;
    for (const char* start : {"3.14123", "3.29123", "3.304", "3.25123"}) {
        auto obs = tadah::hpo::parse_pc_observable_line(
            std::string("cohesive_energy --species Ta --target 8.1 "
                        "--weight 800 --hotvar a0 3.1 3.5 ") + start);
        const ObservableEvalResult r =
            obs->evaluate(pot.model(), pot.dc(), pot.context());
        REQUIRE(r.hotvar_updated_value.has_value());
        INFO(std::setprecision(17) << "start " << start << ": value "
             << r.value << ", a0 handed on " << *r.hotvar_updated_value);
        CHECK_FALSE(r.value_is_constant);
        if (r.energy_terms.size() == 2u) {
            const double p = pressure_of(pot.model(), pot.dc(), pot.context(),
                                         r.energy_terms[0].structure);
            INFO(std::setprecision(17) << "pressure of the relaxed bulk " << p
                 << " eV/A^3");
            CHECK(p < 6.241509074e-7);
            values.push_back(r.value);
        }
    }
    REQUIRE(values.size() == 4u);
    const auto [lo, hi] = std::minmax_element(values.begin(), values.end());
    INFO(std::setprecision(17) << "values from " << *lo << " to " << *hi);
    CHECK(*hi - *lo <= 1.0e-9);
}
