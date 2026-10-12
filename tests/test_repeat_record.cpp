/* The run's record of its evaluations and what it changes (repeat_points.h,
 * dlib_adapter.h, init_strategies.cpp): the pieces with an interface of
 * their own.
 *
 *  - detail::bfgs_box_search ends a search that an iteration without an
 *    evaluation took back to a point where it stood (BfgsBoxEnd::Revisit,
 *    bfgs_box_search (e)), after the stop strategy's own test;
 *  - HPO_Host::recorded() answers for the point an evaluation of the asked
 *    point would make (the snap onto the box, the RHOFORM projection), and
 *    starts empty at every run;
 *  - a RESTART round's RANDOM and LHS methods draw a stream of their own
 *    (InitStream::Restart), SOBOL its block.
 *
 * The searches that use them end to end are in test_repeat_points.cpp
 * (stub hosts) and test_hpo_termination.cpp (hpo_run on HPO_Host). */

#include "catch2/catch.hpp"

#include "rhoform_test_support.h"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/dlib_adapter.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/init_strategy.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/seed_stream.h>

#include <dlib/matrix.h>
#include <dlib/optimization.h>

#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using Vec  = dlib::matrix<double, 0, 1>;
using Host = tadah::hpo::HPO_Host<Vec, ::LammpsRunner>;
using tadah::hpo::HPOLog;
using tadah::hpo::OptimizerSpec;

struct NoRunner {};   // GradientProvider's Runner is only a type tag

bool same_bits(const Vec& a, const Vec& b)
{
    if (a.size() != b.size()) return false;
    for (long i = 0; i < a.size(); ++i)
        if (std::memcmp(&a(i), &b(i), sizeof(double)) != 0) return false;
    return true;
}

/* An open journal, closed and removed with the object. */
struct OpenJournal {
    fs::path path = fs::temp_directory_path() /
                    ("tadah_repeat_record_" + std::to_string(::getpid()) +
                     "_" +
                     std::to_string(std::chrono::steady_clock::now()
                                        .time_since_epoch().count()) +
                     ".log");
    OpenJournal() {
        HPOLog::instance().reset_for_test();
        HPOLog::instance().open(path.string(), "test", "/", 0);
    }
    ~OpenJournal() {
        HPOLog::instance().finish();
        HPOLog::instance().reset_for_test();
        std::error_code ec;
        fs::remove(path, ec);
    }
};

/* test_repeat_points.cpp's plateau of the loss's rounding: 23207.610197404942
 * + 0.0034262213715514594 |u - 0.89349363830995211| on [0, 1], with a host
 * whose every evaluation is a function of its point. */
struct PlateauV {
    std::size_t n = 0;
    static double loss(double u) {
        return 23207.610197404942 +
               0.0034262213715514594 * std::fabs(u - 0.89349363830995211);
    }
    double operator()(const Vec& x) { ++n; return loss(x(0)); }
    std::string repeat_reuse_blocker() const { return ""; }
    std::size_t g_iter_value() const { return n; }
};

OptimizerSpec plateau_spec()
{
    OptimizerSpec sp;
    sp.lib = "DLIB";
    sp.algo = "BFGS";
    sp.low = {0.0};
    sp.high = {1.0};
    sp.is_int.assign(1, false);
    sp.log_scaled.assign(1, false);
    sp.analyse_logscale();
    sp.parameter_labels = {"A"};
    sp.optim_init = {{0.96678801528256708}};
    sp.param["fd_step"] = 1e-6;
    sp.param_raw["fd_kernel"] = "inhouse";
    sp.failscore = std::numeric_limits<double>::max();
    return sp;
}

/* The objective-delta stop (FTOL_ABS 0) that records the point, loss and
 * gradient of every test and ends the search at test `cap` + 1. */
struct RecordingStop {
    dlib::objective_delta_stop_strategy inner{0.0};
    std::size_t cap;
    std::vector<Vec>* xs;
    std::vector<double>* fs;
    std::vector<Vec>* gs;
    bool should_continue_search(const Vec& x, double f, const Vec& g)
    {
        xs->push_back(x);
        fs->push_back(f);
        gs->push_back(g);
        if (xs->size() > cap) return false;
        return inner.should_continue_search(x, f, g);
    }
};

/* One bfgs_box_search of the plateau through DLIB:BFGS's objective (its
 * record, NUMERICAL finite differences) from u = 0.96678801528256708. */
struct PlateauSearch {
    PlateauV host;
    OptimizerSpec sp = plateau_spec();
    std::vector<Vec> xs, gs;
    std::vector<double> fs;
    Vec x;
    double f = 0.0;
    unsigned long evaluations = 0;
    tadah::hpo::detail::BfgsBoxResult result;

    explicit PlateauSearch(std::size_t cap)
    {
        OpenJournal j;
        tadah::hpo::detail::BfgsObjective<Vec, NoRunner, PlateauV> objective(
            sp, host, 0);
        Vec lo(1), hi(1);
        lo = 0.0;
        hi = 1.0;
        x.set_size(1);
        x = sp.optim_init.front()[0];
        result = tadah::hpo::detail::bfgs_box_search(
            dlib::bfgs_search_strategy(), RecordingStop{dlib::objective_delta_stop_strategy(0.0), cap, &xs, &fs, &gs},
            [&](const Vec& v) { return objective.loss(v); },
            [&](const Vec& v) { return objective.gradient(v); }, x, lo, hi, f,
            [](const Vec&) {}, [&objective] { return objective.evaluations(); });
        evaluations = objective.evaluations();
    }
};

constexpr std::size_t kTestCap = 100000;

}  // namespace

/* Pins: bfgs_box_search with an evaluation count (BfgsObjective's), on the
 * plateau under FTOL_ABS 0: an iteration moves the point without an
 * evaluation and the next stop test passes, so the search ends
 * BfgsBoxEnd::Revisit there after 111 evaluations, at a point where it
 * stood at an earlier test, with that test's loss and gradient, bit for
 * bit; `revisit_from` is the point of the test before. A stop strategy that
 * ends the search at that same test ends it first: BfgsBoxEnd::Stop at the
 * same point (dlib's own test comes before the rule). With plain functions
 * (the overload without an evaluation count: every call an evaluation) the
 * rule never fires.
 * Before the fix: does not compile (the evaluation count and
 * BfgsBoxEnd::Revisit are new); the loop went on from that point without an
 * evaluation until the cap of tests. */
TEST_CASE("Run record: DLIB:BFGS's loop ends an iteration that went back to "
          "a point without an evaluation",
          "[HPO][repeat_points][dlib_bfgs][revisit]")
{
    using tadah::hpo::detail::BfgsBoxEnd;
    const PlateauSearch a(kTestCap);
    REQUIRE(a.result.end == BfgsBoxEnd::Revisit);
    CHECK(a.evaluations == 111u);
    REQUIRE(a.xs.size() >= 3u);
    CHECK(same_bits(a.x, a.xs.back()));
    CHECK(same_bits(a.result.revisit_from, a.xs[a.xs.size() - 2]));
    CHECK_FALSE(same_bits(a.result.revisit_from, a.x));
    std::size_t earlier = a.xs.size();
    for (std::size_t k = 0; k + 1 < a.xs.size(); ++k)
        if (same_bits(a.xs[k], a.x)) earlier = k;
    REQUIRE(earlier < a.xs.size() - 1);
    CHECK(std::memcmp(&a.fs[earlier], &a.f, sizeof a.f) == 0);
    CHECK(same_bits(a.gs[earlier], a.result.gradient));

    /* The stop test at which the rule fired ends the search itself. */
    const PlateauSearch b(a.xs.size() - 1);
    CHECK(b.result.end == BfgsBoxEnd::Stop);
    CHECK(same_bits(b.x, a.x));
    CHECK(b.evaluations == a.evaluations);

    /* Plain functions: every call is an evaluation. */
    const double h = 1e-6;
    std::vector<Vec> xs, gs;
    std::vector<double> fs;
    Vec x(1), lo(1), hi(1);
    x = 0.96678801528256708;
    lo = 0.0;
    hi = 1.0;
    double f = 0.0;
    const auto r = tadah::hpo::detail::bfgs_box_search(
        dlib::bfgs_search_strategy(), RecordingStop{dlib::objective_delta_stop_strategy(0.0), 2000, &xs, &fs, &gs},
        [](const Vec& v) { return PlateauV::loss(v(0)); },
        [h](const Vec& v) {
            Vec g(1);
            g(0) = (PlateauV::loss(v(0) + h) - PlateauV::loss(v(0) - h)) /
                   (2 * h);
            return g;
        },
        x, lo, hi, f, [](const Vec&) {});
    CHECK(r.end != BfgsBoxEnd::Revisit);
}

namespace {

/* Four-atom bcc Ta (test_hpo_termination.cpp's fixture). */
void write_ta_db(const fs::path& path, double offset_x, double energy)
{
    std::ofstream os(path);
    os << "test_bcc_ta\n1.0 1.0 1.0\n" << energy << " 0.0\n"
       << "3.301 0.0 0.0\n0.0 3.301 0.0\n0.0 0.0 6.602\n"
       << "0.0 0.0 0.0\n0.0 0.0 0.0\n0.0 0.0 0.0\n"
       << "Ta " << (0.0 + offset_x) << " 0.0 0.0  0.0 0.0 0.0\n"
       << "Ta " << (1.6505 + offset_x) << " 1.6505 1.6505  0.0 0.0 0.0\n"
       << "Ta " << (0.0 + offset_x) << " 0.0 3.301  0.0 0.0 0.0\n"
       << "Ta " << (1.6505 + offset_x) << " 1.6505 4.9515  0.0 0.0 0.0\n\n";
}

tadah::core::Context ta_context(const fs::path& dbfile, const fs::path& hpo,
                                const fs::path& val)
{
    tadah::core::Context c;
    c.add("DBFILE", dbfile.string());
    c.add("ATOM", "Ta");
    c.add("WATOM", 1);
    c.add("INIT2B", "true");
    c.add("INITMB", "false");
    c.add("TYPE2B", "D2_BP");
    c.add("TYPE2B", static_cast<int>(2));
    c.add("TYPE2B", static_cast<int>(2));
    c.add("TYPE2B", "Ta");
    c.add("TYPE2B", "Ta");
    c.add("CGRID2B", "0.5 1.4");
    c.add("SGRID2B", "3.0 5.0");
    c.add("RCTYPE2B", "Cut_Cos");
    c.add("RCUT2B", 5.0);
    c.add("TYPEMB", "DM_Dummy");
    c.add("TYPEMB", "Ta");
    c.add("TYPEMB", "Ta");
    c.add("RCTYPEMB", "Cut_Cos");
    c.add("RCUTMB", 5.0);
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
    c.add("LAMBDA", 1e-3);
    c.add("OALGO", 4);
    c.add("HPOTARGET", hpo.string());
    c.add("VALIDATION", val.string());
    return c;
}

Vec ta_point(double a, double b, double c)
{
    Vec v(3);
    v(0) = a;
    v(1) = b;
    v(2) = c;
    return v;
}

}  // namespace

/* Pins: HPO_Host::recorded on the Ta fixture (OPTIM CGRID2B (1) in [0.1,
 * 0.9], CGRID2B (2) in [1, 2], SGRID2B (2) in [4, 6]; PC_ERMSE): empty at
 * first; after evaluations of P = (0.5, 1.4, 5) and B = (0.9, 1.4, 5) it
 * gives P's loss and evaluation number for P, nothing for a point one ulp
 * from P inside the box, and B's loss, number and point for B and for B
 * with CGRID2B (1) one ulp above its upper bound 0.9 (an evaluation of that
 * point is one of B: the snap onto the box), but nothing for 0.9 + 1e-6
 * (evaluated where it is); prepare_best_publication, which starts every run,
 * empties it.
 * Before the fix: does not compile (HPO_Host::recorded is new; each search
 * kept a record of its own). */
TEST_CASE("Run record: HPO_Host answers for the point an evaluation would "
          "make", "[HPO][repeat_points][run_record]")
{
    tadah::core::PeriodicTable::initialize();
    HPOLog::instance().reset_for_test();
    Host::reset_globals_for_test();
    rhoform_test::ScratchDir sd("tadah_repeat_record_");
    const fs::path train = sd.dir / "train.db", valdb = sd.dir / "val.db";
    const fs::path val = sd.dir / "config.val", hpo = sd.dir / "config.hpo";
    write_ta_db(train, 0.0, -0.001);
    write_ta_db(valdb, 0.07, -0.04);
    std::ofstream(val) << "DBFILE " << valdb.string() << "\n";
    std::ofstream(hpo) << "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 5\n"
                          "ENDOPTIMIZER\nLOSS L2\nFAILSCORE 1e9\nPC_ERMSE 0 1\n"
                          "OPTIM CGRID2B (1) 0.1 0.9\n"
                          "OPTIM CGRID2B (2) 1.0 2.0\n"
                          "OPTIM SGRID2B (2) 4.0 6.0\n";
    tadah::core::Context ctx = ta_context(train, hpo, val);
    tadah::hpo::HPOTargets trg(ctx, hpo.string());
    {
        Host host(ctx, trg, val.string());
        host.prepare_best_publication();
        const Vec P = ta_point(0.5, 1.4, 5.0), B = ta_point(0.9, 1.4, 5.0);
        CHECK_FALSE(host.recorded(P));

        const double lp = host(P);
        const std::size_t ep = Host::g_iter_value();
        const double lb = host(B);
        const std::size_t eb = Host::g_iter_value();

        const auto rp = host.recorded(P);
        REQUIRE(rp);
        CHECK(std::memcmp(&rp->loss, &lp, sizeof lp) == 0);
        CHECK(rp->eval == std::optional<std::size_t>(ep));
        CHECK(same_bits(rp->point, P));
        CHECK_FALSE(host.recorded(ta_point(std::nextafter(0.5, 1.0), 1.4, 5.0)));

        for (const double above : {0.9, std::nextafter(0.9, 1.0)}) {
            CAPTURE(above);
            const auto rb = host.recorded(ta_point(above, 1.4, 5.0));
            REQUIRE(rb);
            CHECK(std::memcmp(&rb->loss, &lb, sizeof lb) == 0);
            CHECK(rb->eval == std::optional<std::size_t>(eb));
            CHECK(same_bits(rb->point, B));
        }
        CHECK_FALSE(host.recorded(ta_point(0.9 + 1e-6, 1.4, 5.0)));

        host.prepare_best_publication();
        CHECK_FALSE(host.recorded(P));
        CHECK_FALSE(host.recorded(B));
        host.flush_remaining();
    }
    Host::reset_globals_for_test();
    HPOLog::instance().reset_for_test();
}

/* Pins: under RHOFORM NONNEG an AMPGRIDMB coordinate below 0 is evaluated
 * at 0 (the projection), so HPO_Host::recorded answers for any negative one
 * with the evaluation of 0: after an evaluation of AMPGRIDMB (3) = -1e-6 (the
 * DM_REAM fixture, OPTIM AMPGRIDMB (3) in [0, 0.5]) it gives that loss for
 * -1e-6, -0.3 and 0, the recorded point 0, and nothing for 0.25.
 * Before the fix: does not compile (HPO_Host::recorded is new). */
TEST_CASE("Run record: HPO_Host answers for a point the RHOFORM projection "
          "evaluates elsewhere", "[HPO][repeat_points][run_record][rhoform]")
{
    tadah::core::PeriodicTable::initialize();
    rhoform_test::ScratchDir sd("tadah_repeat_record_ream_");
    rhoform_test::Journal j;
    Host::reset_globals_for_test();
    const fs::path val_db = sd.dir / "val.db", val = sd.dir / "config.val";
    const fs::path hpo = sd.dir / "config.hpo";
    rhoform_test::write_ta_db(val_db);
    std::ofstream(val) << "DBFILE " << val_db.string() << "\n";
    std::ofstream(hpo) << "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\n"
                          "ENDOPTIMIZER\nLOSS L2\nPC_EMAE 0 1\nFAILSCORE 1e9\n"
                          "OPTIM AMPGRIDMB (3) 0 0.5\n";
    tadah::core::Context ctx = rhoform_test::make_ream_ctx("NONNEG",
                                                          "0.5 0.3 0.0");
    ctx.add("HPOTARGET", hpo.string());
    tadah::hpo::HPOTargets trg(ctx, hpo.string());
    {
        Host host(ctx, trg, val.string());
        host.prepare_best_publication();
        Vec p(1);
        p(0) = -1e-6;
        const double l = host(p);
        const std::size_t e = Host::g_iter_value();
        for (const double a : {-1e-6, -0.3, 0.0}) {
            CAPTURE(a);
            Vec q(1);
            q(0) = a;
            const auto r = host.recorded(q);
            REQUIRE(r);
            CHECK(std::memcmp(&r->loss, &l, sizeof l) == 0);
            CHECK(r->eval == std::optional<std::size_t>(e));
            CHECK(r->point(0) == 0.0);
        }
        Vec in(1);
        in(0) = 0.25;
        CHECK_FALSE(host.recorded(in));
        host.flush_remaining();
    }
    Host::reset_globals_for_test();
}

namespace {

/* An HPOTargets without OPTIM keys (the init strategies' stand-in). */
struct BareTargets {
    rhoform_test::TempFile file{"OPTIMIZER\nLIB DLIB\nALGO BFGS\nENDOPTIMIZER\n",
                                "tadah_repeat_record_trg_"};
    tadah::core::Context ctx;
    tadah::hpo::HPOTargets trg{ctx, file.path.string()};
};

OptimizerSpec unit_box(std::size_t n)
{
    OptimizerSpec sp;
    sp.low.assign(n, 0.0);
    sp.high.assign(n, 1.0);
    sp.is_int.assign(n, false);
    sp.log_scaled.assign(n, false);
    return sp;
}

std::vector<std::vector<double>> draw(const std::string& method,
                                      const std::string& k,
                                      const OptimizerSpec& sp,
                                      const tadah::hpo::HPOTargets& trg,
                                      std::uint64_t seed,
                                      tadah::hpo::InitStream stream)
{
    tadah::hpo::InitStrategySpec ispec;
    ispec.chain.push_back({method, {k}});
    const auto strat = tadah::hpo::build_init_strategy(ispec);
    tadah::hpo::InitContext ic{sp, trg, seed, nullptr, stream};
    return strat->generate(ic, static_cast<std::size_t>(std::stoul(k)));
}

}  // namespace

/* Pins: for one seed (5), the RANDOM and LHS methods of a RESTART round
 * (InitStream::Restart) draw other points than an INIT block's
 * (InitStream::Init), the same ones on every draw: RANDOM's are the uniform
 * numbers of seed_stream(5, seed_streams::kRestartPoints), an INIT block's
 * those of std::mt19937_64(5), as they always were; SOBOL, which draws no
 * random numbers, gives block 5 in both. So a round whose RESTART SEED + r
 * is an INIT block's SEED does not start where that block started.
 * draws_from_seed is true for RANDOM, LHS, SOBOL and a chain holding one,
 * false for CONFIG.
 * Before the fix: does not compile (InitStream and draws_from_seed are new);
 * a round with RESTART SEED + r = 5 drew std::mt19937_64(5), the INIT
 * block's points. */
TEST_CASE("Run record: a RESTART round's RANDOM and LHS draw a stream of "
          "their own", "[HPO][init_strategy][restart][seed]")
{
    using tadah::hpo::InitStream;
    BareTargets bt;
    const OptimizerSpec sp = unit_box(2);
    for (const char* method : {"RANDOM", "LHS"}) {
        CAPTURE(method);
        const auto init = draw(method, "3", sp, bt.trg, 5, InitStream::Init);
        const auto restart = draw(method, "3", sp, bt.trg, 5, InitStream::Restart);
        REQUIRE(restart.size() == 3u);
        CHECK(restart != init);
        CHECK(draw(method, "3", sp, bt.trg, 5, InitStream::Restart) == restart);
        CHECK(draw(method, "3", sp, bt.trg, 5, InitStream::Init) == init);
    }
    {
        std::mt19937_64 rng = tadah::hpo::seed_stream(
            5, tadah::hpo::seed_streams::kRestartPoints);
        std::mt19937_64 old(5);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        const auto restart = draw("RANDOM", "1", sp, bt.trg, 5, InitStream::Restart);
        const auto init = draw("RANDOM", "1", sp, bt.trg, 5, InitStream::Init);
        REQUIRE(restart.size() == 1u);
        const double r0 = u(rng), r1 = u(rng);
        CHECK(restart[0] == std::vector<double>{r0, r1});
        const double i0 = u(old), i1 = u(old);
        CHECK(init[0] == std::vector<double>{i0, i1});
    }
    CHECK(draw("SOBOL", "3", sp, bt.trg, 5, InitStream::Restart) ==
          draw("SOBOL", "3", sp, bt.trg, 5, InitStream::Init));

    using Methods =
        std::vector<std::pair<std::string, std::vector<std::string>>>;
    const auto strategy = [](const Methods& m) {
        tadah::hpo::InitStrategySpec ispec;
        for (const auto& [name, args] : m) ispec.chain.push_back({name, args});
        return tadah::hpo::build_init_strategy(ispec);
    };
    CHECK_FALSE(strategy({{"CONFIG", {}}})->draws_from_seed());
    CHECK(strategy({{"RANDOM", {"1"}}})->draws_from_seed());
    CHECK(strategy({{"LHS", {"2"}}})->draws_from_seed());
    CHECK(strategy({{"SOBOL", {"2"}}})->draws_from_seed());
    CHECK(strategy({{"CONFIG", {}}, {"RANDOM", {"2"}}})->draws_from_seed());
}
