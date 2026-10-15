#include "catch2/catch.hpp"
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/candidate_context.h>
#include <tadah/hpo/bump_support_gate.h>
#include <tadah/hpo/rho_gates.h>
#include <tadah/hpo/init_strategy.h>
#include <tadah/models/descriptors/dm/embed_spec.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace {
using namespace tadah;
core::Context config() {
  core::PeriodicTable::initialize();
  std::istringstream in(
    "VERBOSE 0\nATOM Ta\nMODEL M_BLR BF_Linear\nINIT2B false\nINITMB true\n"
    "DESCRIPTOR ta DM_Embed\nPAIRS Ta Ta\nDENSITY\nBASIS Blip5\nFORM NONNEG\n"
    "CENTRES 1 2\nWIDTHS 0.5 0.75\nAMPLITUDES 0.4 0.2\n"
    "CUTOFF\nTYPE Cut_Poly\nRADIUS 6\nENDCUTOFF\nENDDENSITY\n"
    "EMBEDDING\nBASIS Gaussian\nCENTRES 0 1\nWIDTHS 0.5 0.75\n"
    "ENVELOPE\nTYPE Smoothstep2\nCUTOFF 2\nENDENVELOPE\n"
    "ENDEMBEDDING\nENDDESCRIPTOR\n");
  return core::CoreUserConfigParser().parse(in);
}
struct File {
  std::filesystem::path path;
  File(const std::string &s,bool optimizer=true):path(std::filesystem::temp_directory_path()/
    ("tadah-embed-optim-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
    std::ofstream f(path); f<<s;if(optimizer)f<<"OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 2\nENDOPTIMIZER\n";
  }
  ~File() { std::error_code e; std::filesystem::remove(path,e); }
};
struct Point { std::vector<double> v; double operator()(std::size_t i) const{return v.at(i);} };
}

TEST_CASE("DM_Embed named OPTIM paths retain selected coordinates and independent candidates",
          "[unified-dm][HPO]") {
  auto c=config();
  File f("OPTIM ta.embedding.widths (1-2) AUTO RATIO 1.2\n"
         "OPTIM ta.density.cutoff.radius (1) 5.5 6.5\n"
         "OPTIM ta.embedding.envelope.cutoff (1) 1 30\n");
  hpo::HPOTargets t(c,f.path.string());
  REQUIRE(t.space()->dim()==4);
  REQUIRE(hpo::detail::nn_safe_rcutmax(c,t)==6.5);
  auto a=c,b=c;
  // optimap order is density radius, embedding envelope, embedding widths.
  hpo::detail::update_candidate_context(a,t,Point{{6.1,2.3,0.55,0.8}},false,false);
  hpo::detail::update_candidate_context(b,t,Point{{6.2,2.4,0.6,0.9}},false,false);
  REQUIRE(models::embed_binding(a,"ta.embedding.widths").values[0]==0.55);
  REQUIRE(models::embed_binding(b,"ta.embedding.widths").values[0]==0.6);
  REQUIRE(models::embed_binding(c,"ta.embedding.widths").values[0]==0.5);
  REQUIRE(a.get<double>("RCUTMAX")==6.1);
}

TEST_CASE("DM_Embed OPTIM refuses inactive aliases discrete fields and repeated dimensions",
          "[unified-dm][HPO]") {
  for(const auto &line:{
    "OPTIM ta.density.eta (1) 1 2\n",
    "OPTIM ta.density.basis (1) 0 1\n",
    "OPTIM ta.embedding.widths (1) -1 1\n",
    "OPTIM ta.embedding.widths (1) 0.4 0.6\nOPTIM TA.EMBEDDING.WIDTHS (1) 0.4 0.7\n"}) {
    auto c=config(); File f(line); REQUIRE_THROWS(hpo::HPOTargets(c,f.path.string()));
  }
}

TEST_CASE("DM_Embed NONE support gate uses physical width coordinates and strict corners",
          "[unified-dm][HPO][support]") {
  auto c=config();auto blocks=c.blocks();
  for(auto &line:blocks[0].lines)if(line=="TYPE Cut_Poly")line="TYPE NONE";
  c.replace_blocks(blocks);
  File f("OPTIM ta.density.centres (2) 1 8\n");hpo::HPOTargets targets(c,f.path.string());
  auto table=hpo::detail::build_bump_gate_table(c,targets);REQUIRE(table.size()==1);
  REQUIRE(hpo::detail::bump_gate_violation(table,Point{{2}})==0);
  REQUIRE(hpo::detail::bump_gate_violation(table,Point{{7}})>0);
  targets.BUMPGATE_STRICT=true;REQUIRE_THROWS(hpo::detail::build_bump_gate_table(c,targets));
  File radius("OPTIM ta.density.cutoff.radius (1) 5 7\n");
  REQUIRE_THROWS(hpo::HPOTargets(c,radius.path.string()));
}

TEST_CASE("DM_Embed density gates read the local component form and density",
          "[unified-dm][HPO][density-gates]") {
  auto c=config();
  File f("OPTIM ta.density.amplitudes (1) 0.2 0.6\n"
         "RHOBARGATE bcc 3.3 0.9 1.1 NPTS 11\n");
  hpo::HPOTargets targets(c,f.path.string());
  REQUIRE_NOTHROW(hpo::detail::validate_density_gates(c,targets));
  const auto gates=hpo::detail::build_density_gates(c,targets);
  REQUIRE(gates.rhobar.has_value());
  const auto result=hpo::detail::eval_rhobar_gate(*gates.rhobar,c,true);
  REQUIRE_FALSE(result.abstain);REQUIRE_FALSE(result.fail);
}

TEST_CASE("DM_Embed WARM reads saved blocks in selected width coordinates", "[unified-dm][HPO][warm]") {
  auto c=config(),saved=c;models::embed_binding(saved,"ta.embedding.widths").write(saved,0,0.61);
  std::ostringstream text;saved.write_potential(text,18);File seed(text.str(),false);
  File f("OPTIM ta.embedding.widths (1) 0.4 0.7\n");hpo::HPOTargets targets(c,f.path.string());
  hpo::InitStrategySpec spec;spec.warm_on_mismatch="error";spec.chain.push_back({"WARM",{seed.path.string()}});
  auto strategy=hpo::build_init_strategy(spec);hpo::InitContext init{targets.spec,targets,0,&c};
  const auto points=strategy->generate(init,1);REQUIRE(points.size()==1);REQUIRE(points[0][0]==0.61);
}

/* Pins: OPTIM on a legacy many-body key (CGRIDMB, SGRIDMB, AMPGRIDMB,
 * CEMBFUNC, SEMBFUNC, RCUTMB) of a potential whose many-body term is a
 * DM_Embed component, which has no such key, is refused naming the key and
 * the component path that holds the parameter (the refit's starter target
 * names them; an hpotarget of a DM_REAM refit names the legacy keys).
 * Before the fix the refusal was 'Key not found in the Context object:
 * CGRIDMB'. */
TEST_CASE("DM_Embed OPTIM of a legacy many-body key names the component path",
          "[unified-dm][HPO]") {
  auto c=config();
  const std::vector<std::pair<std::string,std::string>> keys{
    {"CGRIDMB","TA.DENSITY.CENTRES"},{"SGRIDMB","TA.DENSITY.WIDTHS"},
    {"AMPGRIDMB","TA.DENSITY.AMPLITUDES"},{"CEMBFUNC","TA.EMBEDDING.CENTRES"},
    {"SEMBFUNC","TA.EMBEDDING.WIDTHS"},{"RCUTMB","TA.DENSITY.CUTOFF.RADIUS"}};
  for(const auto &[key,path]:keys) {
    CAPTURE(key);
    File f("OPTIM "+key+" (1) 0.5 1.5\n");
    REQUIRE_THROWS_WITH(hpo::HPOTargets(c,f.path.string()),
                        Catch::Contains("OPTIM "+key) && Catch::Contains(path) &&
                        Catch::Contains("DM_Embed"));
  }
}
