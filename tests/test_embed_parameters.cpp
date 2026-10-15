#include "catch2/catch.hpp"
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/candidate_context.h>
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
  File(const std::string &s):path(std::filesystem::temp_directory_path()/
    ("tadah-embed-optim-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
    std::ofstream f(path); f<<s<<"OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 2\nENDOPTIMIZER\n";
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
