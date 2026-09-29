// test_hpo_embf0_journal.cpp
//
// EMBF0 lines of the hpo_run.log journal (HPOLog in isolation, no host):
//   - POTENTIAL SETUP: the "Embedding F(0)" row of a DM_REAM configuration
//     (mode read without inserting the default) and the raw F(0) row of the
//     -c potential (derive_raw_embedding_f0 result); nothing for a potential
//     without a DM_REAM block;
//   - RESULT: the one-line start (eval 1) / best (eval N) summary printed by
//     finish() once armed; out-of-order best records keep the lowest loss;
//     nothing when not armed.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/startup_validation.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {

using tadah::core::Context;
using tadah::hpo::HPOLog;
using tadah::hpo::RawEmbeddingF0;

struct JournalFile {
  std::filesystem::path path;
  JournalFile()
  : path(std::filesystem::temp_directory_path() /
         ("tadah_embf0_journal_" +
          std::to_string(
              std::chrono::steady_clock::now().time_since_epoch().count()) +
          ".log")) {
    /* open() cannot reopen a journal another test left open. */
    if (HPOLog::instance().is_open()) HPOLog::instance().finish("closed");
    HPOLog::instance().reset_for_test();
    HPOLog::instance().open(path.string(), "test", "test_cwd", 0);
  }
  ~JournalFile() {
    HPOLog::instance().reset_for_test();
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
  std::string read() const {
    std::ifstream is(path);
    std::ostringstream oss;
    oss << is.rdbuf();
    return oss.str();
  }
};

/* The keys write_potential_section reads, plus a DM_REAM TYPEMB. */
Context make_ream_ctx(const char* embf0) {
  Context c;
  c.add("ATOM", "Ta");
  c.add("INIT2B", "false");
  c.add("INITMB", "true");
  c.add("TYPEMB", "DM_REAM");
  c.add("TYPEMB", 0);
  c.add("TYPEMB", 2); c.add("TYPEMB", 2); c.add("TYPEMB", 2);
  c.add("TYPEMB", 3); c.add("TYPEMB", 3);
  c.add("TYPEMB", 0); c.add("TYPEMB", 0);
  c.add("TYPEMB", "Ta");
  c.add("TYPEMB", "Ta");
  c.add("RCUTMB", 4.8);
  c.add("RHOFORM", "SIGNED");
  if (embf0) c.add("EMBF0", embf0);
  return c;
}

void log_best(std::size_t eval, double loss) {
  HPOLog::instance().log_new_best(eval, loss, 0.0, false, 0.0, false,
                                  0.0, false, 0.0, false, 0.0, false, 1.0);
}

}  // namespace

TEST_CASE("EMBF0 journal: POTENTIAL SETUP rows", "[hpo_log][embf0]") {
  auto& log = HPOLog::instance();

  SECTION("key absent: ZERO by default, the key is not inserted") {
    JournalFile jf;
    Context ctx = make_ream_ctx(nullptr);
    RawEmbeddingF0 seed;
    seed.kind = RawEmbeddingF0::Kind::Value;
    seed.value = -0.0123;
    seed.source = "FIXWEIGHT";
    log.write_potential_section(ctx, 4, 0, "Ta", &seed);
    log.finish("TEST");
    const std::string txt = jf.read();
    CHECK(txt.find("  Embedding F(0)        : EMBF0 ZERO (key absent -> "
                   "default): DM_REAM embedding zero-referenced, "
                   "isolated-atom embedding energy 0 by construction") !=
          std::string::npos);
    CHECK(txt.find("  Raw F(0) of -c pot    : -1.230000e-02 eV = sum w g(0), "
                   "FIXWEIGHT of the -c potential (removed per atom by "
                   "EMBF0 ZERO)") != std::string::npos);
    CHECK_FALSE(ctx.has_stored("EMBF0"));
  }
  SECTION("explicit ZERO drops the default note") {
    JournalFile jf;
    Context ctx = make_ream_ctx("ZERO");
    log.write_potential_section(ctx, 4, 0, "Ta");
    log.finish("TEST");
    const std::string txt = jf.read();
    CHECK(txt.find(": EMBF0 ZERO: DM_REAM embedding zero-referenced") !=
          std::string::npos);
    CHECK(txt.find("key absent") == std::string::npos);
    CHECK(txt.find("Raw F(0)") == std::string::npos);  // no seed passed
  }
  SECTION("FREE, weights fitted per evaluation") {
    JournalFile jf;
    Context ctx = make_ream_ctx("FREE");
    RawEmbeddingF0 seed;
    seed.kind = RawEmbeddingF0::Kind::PerEvaluation;
    log.write_potential_section(ctx, 4, 0, "Ta", &seed);
    log.finish("TEST");
    const std::string txt = jf.read();
    CHECK(txt.find(": EMBF0 FREE: raw embedding (tadah <= 1.5.0), "
                   "F(0) = sum w g(0)") != std::string::npos);
    CHECK(txt.find("  Raw F(0) of -c pot    : determined per evaluation") !=
          std::string::npos);
  }
  SECTION("not derivable: n/a with the reason") {
    JournalFile jf;
    Context ctx = make_ream_ctx(nullptr);
    RawEmbeddingF0 seed;
    seed.reason = "the model is not linear in the descriptor";
    log.write_potential_section(ctx, 4, 0, "Ta", &seed);
    log.finish("TEST");
    CHECK(jf.read().find("  Raw F(0) of -c pot    : n/a (the model is not "
                         "linear in the descriptor)") != std::string::npos);
  }
  SECTION("no DM_REAM block: no rows") {
    JournalFile jf;
    Context ctx;
    ctx.add("INIT2B", "true");
    ctx.add("INITMB", "false");
    ctx.add("RCUT2B", 5.0);
    RawEmbeddingF0 seed;
    log.write_potential_section(ctx, 4, 0, "Ta", &seed);
    log.finish("TEST");
    const std::string txt = jf.read();
    CHECK(txt.find("Embedding F(0)") == std::string::npos);
    CHECK(txt.find("Raw F(0)") == std::string::npos);
  }
}

TEST_CASE("EMBF0 journal: RESULT line", "[hpo_log][embf0]") {
  auto& log = HPOLog::instance();

  SECTION("ZERO: start (eval 1) and the final best") {
    JournalFile jf;
    log.arm_embf0(/*zero_mode=*/true, /*derivable=*/true);
    log.record_embf0_start(1, -0.0123);
    log_best(1, 0.9);
    log.record_embf0_best(1, 0.9, -0.0123);
    log_best(7, 0.5);
    /* Journal calls of concurrent evaluations can arrive out of order: a
     * superseded best (higher loss) recorded late must not replace eval 7. */
    log.record_embf0_best(7, 0.5, -0.02);
    log_best(5, 0.7);
    log.record_embf0_best(5, 0.7, 1.0);
    log.finish("TEST");
    CHECK(jf.read().find(
              "  Embedding F(0)          : EMBF0 ZERO (removed per atom): "
              "start (eval 1) -1.230000e-02 eV, best (eval 7) "
              "-2.000000e-02 eV") != std::string::npos);
  }
  SECTION("FREE: the isolated-atom embedding energy; missing values n/a") {
    JournalFile jf;
    log.arm_embf0(/*zero_mode=*/false, /*derivable=*/true);
    log.finish("TEST");
    CHECK(jf.read().find(
              "  Embedding F(0)          : EMBF0 FREE (isolated-atom "
              "embedding energy): start n/a, best n/a") != std::string::npos);
  }
  SECTION("a best recorded in another session is not reported") {
    JournalFile jf;
    log.arm_embf0(true, true);
    log.record_embf0_best(3, 0.4, 0.25);
    log_best(3, 0.4);
    log_best(9, 0.1);  // e.g. a checkpointed best: no value this session
    log.finish("TEST");
    CHECK(jf.read().find("start n/a, best n/a") != std::string::npos);
  }
  SECTION("model not linear in the descriptor") {
    JournalFile jf;
    log.arm_embf0(true, /*derivable=*/false);
    log.finish("TEST");
    CHECK(jf.read().find("EMBF0 ZERO (removed per atom): n/a (the model is "
                         "not linear in the descriptor)") !=
          std::string::npos);
  }
  SECTION("not armed (no DM_REAM block): no line") {
    JournalFile jf;
    log.record_embf0_start(1, 0.5);
    log_best(1, 0.5);
    log.finish("TEST");
    CHECK(jf.read().find("Embedding F(0)") == std::string::npos);
  }
}
