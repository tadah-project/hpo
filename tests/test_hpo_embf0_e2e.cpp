// test_hpo_embf0_e2e.cpp
//
// EMBF0 through the HPO host (tadah hpo / NFP): a DM_REAM model whose only
// search dimensions are embedding centres/widths (CEMBFUNC, SEMBFUNC) of
// bumps that cover rho = 0, every weight pinned (FIXINDEX all, no training
// DBFILE). The embedding reference g_m(0) must be recomputed for EVERY
// candidate from the values actually evaluated -- on the fresh-trainer path
// and on the cached-trainer update_context + reinit path (checked at
// rho > 0 against a DM_REAM built from the candidate's potential) -- so that:
//   - EMBF0 ZERO (default): the isolated atom predicts exactly 0.0;
//   - EMBF0 FREE: it predicts the raw F(0) of the candidate;
//   - the per-evaluation potential carries the EMBF0 line and the candidate's
//     grid values, and LAMMPS built from it gives F(0) = 0 on the EAMFAST
//     embedding path too.
//
// Hidden `[.]` like the other HPO_Host e2e cases. Run explicitly with:
//   ./test_hpo "[embf0_e2e]"

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/element.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/startup_validation.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/tadah.lammps/lammps_tadah.h>

#include <dlib/matrix.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using ParamVec = dlib::matrix<double, 0, 1>;
using Host     = tadah::hpo::HPO_Host<ParamVec, ::LammpsRunner>;

struct ScratchDir {
  fs::path dir, original_cwd;
  ScratchDir() {
    dir = fs::temp_directory_path() /
          ("tadah_hpo_embf0_" +
           std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count()));
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

void write_ta_db(const fs::path &path) {
  std::ofstream os(path);
  os << "bcc_ta\n1.0 1.0 1.0\n-8.1 0.0\n"
     << "3.304 0.0 0.0\n0.0 3.304 0.0\n0.0 0.0 6.608\n"
     << "0.0 0.0 0.0\n0.0 0.0 0.0\n0.0 0.0 0.0\n"
     << "Ta 0.0 0.0 0.0  0.0 0.0 0.0\n"
     << "Ta 1.652 1.652 1.652  0.0 0.0 0.0\n"
     << "Ta 0.0 0.0 3.304  0.0 0.0 0.0\n"
     << "Ta 1.652 1.652 4.956  0.0 0.0 0.0\n\n";
}

/* D2_Blip5 (4) + DM_REAM Blip5 (3 density, 5 embedding bumps; the first four
 * cover rho = 0). All 9 weights pinned. */
tadah::core::Context make_ctx(const fs::path &hpo_cfg, const char *embf0) {
  tadah::core::Context c;
  c.add("ATOM", "Ta");
  c.add("WATOM", 1);
  c.add("INIT2B", "true");
  c.add("INITMB", "true");
  c.add("TYPE2B", "D2_Blip5");
  c.add("TYPE2B", 4);
  c.add("TYPE2B", 4);
  c.add("TYPE2B", "Ta");
  c.add("TYPE2B", "Ta");
  c.add("CGRID2B", "2.5 3.2 3.9 4.5");
  c.add("SGRID2B", "3.0 3.0 3.0 3.0");
  c.add("TYPEMB", "DM_REAM");
  c.add("TYPEMB", 3);
  c.add("TYPEMB", 3); c.add("TYPEMB", 3); c.add("TYPEMB", 3);
  c.add("TYPEMB", 5); c.add("TYPEMB", 5);
  c.add("TYPEMB", 0); c.add("TYPEMB", 0);
  c.add("TYPEMB", "Ta");
  c.add("TYPEMB", "Ta");
  c.add("CGRIDMB", "2.8 3.6 4.4");
  c.add("SGRIDMB", "3.0 3.0 3.0");
  c.add("AMPGRIDMB", "0.5 0.3 0.1");
  c.add("RHOFORM", "SIGNED");
  c.add("CEMBFUNC", "-0.2 0.3 0.8 1.3 1.8");
  c.add("SEMBFUNC", "2.0 2.0 2.0 2.0 2.0");
  c.add("RCTYPE2B", "Cut_Dummy");
  c.add("RCTYPEMB", "Cut_Dummy");
  c.add("RCUT2B", 5.5);
  c.add("RCUTMB", 5.5);
  c.add("DIMER", "false");
  c.add("DIMER", 0);
  c.add("DIMER", "false");
  c.add("FORCE", "false");
  c.add("STRESS", "false");
  c.add("EWEIGHT", 1.0);
  c.add("FWEIGHT", 1.0);
  c.add("SWEIGHT", 1.0);
  c.add("BIAS", "false");
  c.add("NORM", "false");
  c.add("VERBOSE", 0);
  c.add("MODEL", "M_BLR");
  c.add("MODEL", "BF_Linear");
  c.add("LAMBDA", 0);
  c.add("OALGO", 1);
  c.add("FIXINDEX", "1-9");
  for (double w : {0.8, -0.3, 0.1, -0.05, -2.0, 1.5, -0.7, 0.4, 0.2})
    c.add("FIXWEIGHT", w);
  if (embf0) c.add("EMBF0", embf0);
  c.add("HPOTARGET", hpo_cfg.string());
  return c;
}

double isolated_atom_energy(const tadah::mlip::Trainer *tr) {
  tadah::mlip::Structure st;
  st.cell.load(40.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 40.0);
  st.add_atom(tadah::mlip::Atom(tadah::core::Element("Ta"), 20.0, 20.0, 20.0,
                                0.0, 0.0, 0.0));
  tadah::mlip::NNFinder finder;
  finder.calc(st, 5.5);
  auto *t = const_cast<tadah::mlip::Trainer *>(tr);
  auto d = t->dc.calc(st);
  return t->model->epredict(d);
}

void run_embf0_case(const char *embf0) {
  tadah::core::PeriodicTable::initialize();
  Host::reset_globals_for_test();
  ForceRebuildGuard rebuild;
  ScratchDir sd;

  const fs::path val_db = sd.dir / "val.db";
  const fs::path val_cfg = sd.dir / "config.val";
  const fs::path hpo_cfg = sd.dir / "config.hpo";
  write_ta_db(val_db);
  { std::ofstream os(val_cfg); os << "DBFILE " << val_db.string() << "\n"; }
  {
    std::ofstream os(hpo_cfg);
    os << "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\nENDOPTIMIZER\n"
       << "LOSS L2\nPC_EMAE 0 1\n"
       << "OPTIM CEMBFUNC (1,2) -0.5 0.6\n"
       << "OPTIM SEMBFUNC (1,2) 1.5 2.5\n";
  }

  tadah::core::Context ctx = make_ctx(hpo_cfg, embf0);
  tadah::hpo::HPOTargets trg(ctx, hpo_cfg.string());
  REQUIRE_NOTHROW(tadah::hpo::validate_hpo_setup(ctx, trg, false, true));
  Host host(ctx, trg, val_cfg.string());

  // Search-space order is by key name, then token: CEMBFUNC 1,2; SEMBFUNC 1,2.
  const std::vector<std::vector<double>> points = {
      {-0.2, 0.3, 2.0, 2.0}, {0.25, -0.4, 1.7, 2.3}, {-0.45, 0.55, 2.4, 1.6}};
  bool first = true;
  for (const auto &pt : points) {
    if (!first) ::unsetenv("TADAH_HPO_FORCE_REBUILD");  // cached trainer path
    first = false;
    ParamVec p(4);
    for (int i = 0; i < 4; ++i) p(i) = pt[i];
    double loss = 0.0;
    REQUIRE_NOTHROW(loss = host(p));
    CHECK(std::isfinite(loss));

    const auto *tr = Host::tls_trainer_for_test();
    REQUIRE(tr != nullptr);
    auto pot = const_cast<tadah::mlip::Trainer *>(tr)->get_param_file();
    const auto mode = pot.stored_value("EMBF0");
    REQUIRE(mode.has_value());
    REQUIRE(*mode == (embf0 ? std::string(embf0) : std::string("ZERO")));
    // The candidate's values are what the model evaluates.
    std::vector<double> ce(pot.size("CEMBFUNC")), se(pot.size("SEMBFUNC"));
    pot.get("CEMBFUNC", ce);
    pot.get("SEMBFUNC", se);
    REQUIRE(ce[0] == pt[0]);
    REQUIRE(ce[1] == pt[1]);
    REQUIRE(se[0] == pt[2]);
    REQUIRE(se[1] == pt[3]);

    const double e_iso = isolated_atom_energy(tr);
    const auto raw = tr->DCS.dmb->raw_embedding_f0(tr->model->get_weights());
    REQUIRE(raw.has_value());
    REQUIRE(*raw != 0.0);
    // LAMMPS built from this candidate's potential (the in-hpo EXTERN pot).
    tadah::LammpsTadah lt(pot);
    tadah::core::t_type w;
    w.resize(lt.c.size("WEIGHTS"));
    lt.c.get<tadah::core::t_type>("WEIGHTS", w);
    double F = 0.0, Fp = 0.0;
    lt.S.dmb->eam_embed(w, 0.0, F, Fp);
    // rho > 0 (rho == 0 short-circuits, so it cannot reveal a stale g0):
    // the trainer's DM_REAM -- fresh or updated by reinit -- must hold g0 of
    // THIS candidate: bitwise the DM_REAM built from the candidate's own
    // potential, and (ZERO) its FREE twin minus the raw F(0).
    const bool zero = !embf0 || std::string(embf0) == "ZERO";
    {
      const double rb = 0.7;
      double Ft = 0.0, Fpt = 0.0, Fl = 0.0, Fpl = 0.0, Ff = 0.0, Fpf = 0.0;
      tr->DCS.dmb->eam_embed(tr->model->get_weights(), rb, Ft, Fpt);
      lt.S.dmb->eam_embed(w, rb, Fl, Fpl);
      REQUIRE(Ft == Fl);
      REQUIRE(Fpt == Fpl);
      tadah::core::Context pot_free = pot;
      pot_free.remove("EMBF0");
      pot_free.add("EMBF0", std::string("FREE"));
      tadah::LammpsTadah lf(pot_free);
      lf.S.dmb->eam_embed(w, rb, Ff, Fpf);
      REQUIRE(Fpf == Fpl);
      if (zero)
        REQUIRE(std::fabs(Fl - (Ff - *raw)) <=
                1e-12 * (1.0 + std::fabs(Ff) + std::fabs(*raw)));
      else
        REQUIRE(Fl == Ff);
    }
    if (zero) {
      REQUIRE(e_iso == 0.0);
      REQUIRE(F == 0.0);
    } else {
      REQUIRE(e_iso == Approx(*raw).epsilon(1e-12).margin(1e-14));
      REQUIRE(F == Approx(*raw).epsilon(1e-12).margin(1e-14));
    }
  }
  Host::reset_globals_for_test();
}

}  // namespace

TEST_CASE("EMBF0 ZERO (default): isolated atom exactly 0 for every candidate",
          "[embf0_e2e][.]") {
  run_embf0_case(nullptr);
}

TEST_CASE("EMBF0 ZERO explicit: isolated atom exactly 0 for every candidate",
          "[embf0_e2e][.]") {
  run_embf0_case("ZERO");
}

TEST_CASE("EMBF0 FREE: isolated atom is the candidate's raw F(0)",
          "[embf0_e2e][.]") {
  run_embf0_case("FREE");
}
