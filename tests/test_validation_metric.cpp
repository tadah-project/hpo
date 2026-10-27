#include "catch2/catch.hpp"
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/pot_header_writer.h>
#include <tadah/hpo/validation_metric.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <thread>

using namespace tadah::hpo;
using tadah::LossFunction;

namespace {
struct ConfigFile {
    std::filesystem::path path;
    explicit ConfigFile(const std::string& lines) {
        static std::size_t seq = 0;
        path = std::filesystem::temp_directory_path() / ("pc_dataset_"
             + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
             + "_" + std::to_string(++seq) + ".hpo");
        std::ofstream(path) << "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\nENDOPTIMIZER\n" << lines;
    }
    ~ConfigFile() { std::error_code ec; std::filesystem::remove(path, ec); }
};
}

TEST_CASE("PC dataset selectors resolve independently in validation order", "[pc_dataset][parser]") {
    ConfigFile f("PC_ERMSE (4, 2) 3 0 --skip-rel 2\n"
                 "PC_ERMSE (1-3:2) +1e1 0.25 --skip-above 20 # comment\n"
                 "PC_FRMSE 0.05 2\n");
    tadah::core::Context ctx;
    HPOTargets t(ctx, f.path.string());
    REQUIRE(t.use_ermse);
    REQUIRE(t.use_frmse);
    t.resolve_validation_targets({"a", "b", "c", "d"});
    const auto& e = t.ermse_targets.datasets();
    REQUIRE(e.size() == 4);
    CHECK(e[0].target == 10);
    CHECK(e[1].target == 3);
    CHECK(e[2].weight == 0.25);
    CHECK(e[3].weight == 0);
    CHECK(e[0].caps.abs_above == 20);
    CHECK(e[1].caps.rel_above == 2);
    CHECK(t.frmse_targets.global().target == Approx(0.05));
    CHECK(t.frmse_targets.datasets()[3].weight == 2);
    CHECK(t.ermse_targets.descriptions()[1].find("[1] a target=10") != std::string::npos);
}

TEST_CASE("PC dataset declarations reject ambiguous or invalid input", "[pc_dataset][parser]") {
    for (const auto* metric : {"PC_ERMSE", "PC_FRMSE", "PC_SRMSE_GPA"}) {
        const std::string key = metric;
        for (const auto& body : std::vector<std::string>{
            "", "0", "(1) 0", "() 0 1", "(1, ) 0 1", "(1x,2) 0 1",
            "(1-2:0) 0 1", "(3-1) 0 1", "(1-) 0 1", "(0) 0 1", "(-1) 0 1",
            "((1)) 0 1", "(1 0 1", "(1)) 0 1", "(1)0 1", "(1)(2) 0 1",
            "(1,1) 0 1", "(1-3,2) 0 1", "(1) 0 -1", "nan 1", "0 inf",
            "1e999 1", "0 1x", "0 1 extra", "0 1 --skip-rel 0", "0 1 --skip-above inf",
            "0 1 --skip-above 2 --skip-above 3", "(1) 0 1 --skip-rel",
            "0 1\n" + key + " 0 1", "0 1\n" + key + " (1) 0 1",
            "(1) 0 1\n" + key + " 0 1", "(1-2) 0 1\n" + key + " (2) 0 1"}) {
            INFO(key << " " << body);
            ConfigFile f(key + " " + body + "\n");
            tadah::core::Context ctx;
            CHECK_THROWS(HPOTargets(ctx, f.path.string()));
        }
    }
    ConfigFile unsupported("PC_EMAE (1) 0 1\n");
    tadah::core::Context ctx;
    CHECK_THROWS_WITH(HPOTargets(ctx, unsupported.path.string()),
                     Catch::Matchers::Contains("only for PC_ERMSE, PC_FRMSE and PC_SRMSE_GPA"));
}

/* A repeated top-level PC_EMAE, PC_ErRMSE or PC_ERSQ line is refused at
 * its line, naming the line of the first, as a repeated PC_ERMSE, PC_FRMSE
 * or PC_SRMSE_GPA line is ("Error on line 7: PC_EMAE: repeated declaration
 * (first declared on line 6)"); a line of each metric once is read. Before
 * the fix: the later line replaced the first without a word (the last
 * target, weight and caps won). */
TEST_CASE("A repeated PC_EMAE, PC_ErRMSE or PC_ERSQ line is refused",
          "[pc_dataset][parser]") {
    for (const auto* metric : {"PC_EMAE", "PC_ErRMSE", "PC_ERSQ"}) {
        const std::string key = metric;
        INFO(key);
        ConfigFile f(key + " 0 1\n" + key + " 0 2 --skip-above 5\n");
        tadah::core::Context ctx;
        CHECK_THROWS_WITH(HPOTargets(ctx, f.path.string()),
                          Catch::Matchers::Contains(
                              "Error on line 7: " + key +
                              ": repeated declaration (first declared on "
                              "line 6)"));
    }
    ConfigFile once("PC_EMAE 0 1\nPC_ErRMSE 0 2\nPC_ERSQ 0 3\nPC_SRMSE_GPA 0 4\n");
    tadah::core::Context ctx;
    HPOTargets t(ctx, once.path.string());
    CHECK(t.use_emae);
    CHECK(t.w_emae == 1);
    CHECK(t.w_errmse == 2);
    CHECK(t.w_ersq == 3);
    CHECK(t.use_srmse);
    CHECK(t.srmse_targets.global().weight == 4);
}

/* Pins: PC_ERSQ refuses --skip-above and --skip-rel at its line: a cap
 * rejects an evaluation whose metric is above it, and R^2 is better when
 * higher, so the cap would reject the better fits ("Error on line 6:
 * PC_ERSQ takes no --skip-above: a cap rejects an evaluation whose metric
 * is above it, and a higher R^2 is a better fit, so the cap would reject
 * the better fits"; with both options the first is named). The other
 * metric lines keep their caps, and PC_ERSQ without a cap is read. A
 * repeated PC_ERSQ line is refused as repeated whatever its options (the
 * case above).
 * Before the fix: the caps were read, and an evaluation with R^2 0.95
 * under --skip-above 0.9 was rejected as a failure. */
TEST_CASE("PC_ERSQ refuses --skip-above and --skip-rel at its line",
          "[pc_dataset][parser]") {
    const std::string why =
        ": a cap rejects an evaluation whose metric is above it, and a higher "
        "R^2 is a better fit, so the cap would reject the better fits";
    for (const auto& [line, option] :
         std::vector<std::pair<std::string, std::string>>{
             {"PC_ERSQ 1 1 --skip-above 0.9", "--skip-above"},
             {"PC_ERSQ 1 1 --skip-rel 2", "--skip-rel"},
             {"PC_ERSQ 1 1 --skip-rel 2 --skip-above 0.9", "--skip-rel"}}) {
        INFO(line);
        ConfigFile f(line + "\n");
        tadah::core::Context ctx;
        CHECK_THROWS_WITH(HPOTargets(ctx, f.path.string()),
                          "Error on line 6: PC_ERSQ takes no " + option + why +
                              "\nLine: " + line);
    }
    ConfigFile ok("PC_ERSQ 1 1\nPC_ERMSE 0 1 --skip-above 5 --skip-rel 2\n"
                  "PC_EMAE 0 1 --skip-above 5\n");
    tadah::core::Context ctx;
    HPOTargets t(ctx, ok.path.string());
    CHECK(t.use_ersq);
    CHECK_FALSE(t.skip_ersq.abs_above);
    CHECK_FALSE(t.skip_ersq.rel_above);
    CHECK(t.skip_emae.abs_above == 5);
}

/* A second global PC_ERMSE, PC_FRMSE or PC_SRMSE_GPA line is refused at its
 * line, naming the line of the first, as a repeated PC_EMAE is ("Error on
 * line 7:
 * PC_ERMSE: repeated declaration (first declared on line 6)"); a global
 * line with indexed ones, in either order, is refused as a mix of the two
 * forms. Lines with other indices are read.
 * Before the fix: "PC_ERMSE: repeated global declaration or mixed
 * global/indexed forms", without the first line. */
TEST_CASE("A repeated global PC_ERMSE or PC_FRMSE line is refused naming "
          "the first", "[pc_dataset][parser]") {
    for (const auto* metric : {"PC_ERMSE", "PC_FRMSE", "PC_SRMSE_GPA"}) {
        const std::string key = metric;
        INFO(key);
        tadah::core::Context ctx;
        ConfigFile twice(key + " 0 1\n" + key + " 0 2\n");
        CHECK_THROWS_WITH(HPOTargets(ctx, twice.path.string()),
                          Catch::Matchers::StartsWith(
                              "Error on line 7: " + key +
                              ": repeated declaration (first declared on "
                              "line 6)"));
        for (const std::string body :
             {key + " (1) 0 1\n" + key + " 0 2\n",
              key + " 0 1\n" + key + " (1) 0 2\n"}) {
            ConfigFile mixed(body);
            CHECK_THROWS_WITH(HPOTargets(ctx, mixed.path.string()),
                              Catch::Matchers::StartsWith(
                                  "Error on line 7: " + key +
                                  ": cannot mix global and indexed forms"));
        }
        ConfigFile indexed(key + " (1) 0 1\n" + key + " (2) 0 2\n");
        CHECK_NOTHROW(HPOTargets(ctx, indexed.path.string()));
    }
}

TEST_CASE("PC dataset coverage errors name the missing dataset and metric", "[pc_dataset][parser]") {
    for (const auto* key : {"PC_ERMSE", "PC_FRMSE", "PC_SRMSE_GPA"}) {
        ConfigFile f(std::string(key) + " (1,3) 0 1\n");
        tadah::core::Context ctx;
        HPOTargets t(ctx, f.path.string());
        CHECK_THROWS_WITH(t.resolve_validation_targets({"one.db", "two.db", "three.db"}),
            Catch::Matchers::Contains(std::string(key) + ": missing target and weight for:")
            && Catch::Matchers::Contains("dataset 2 (two.db)"));
        CHECK_THROWS_WITH(t.resolve_validation_targets({"one.db"}),
                          Catch::Matchers::Contains("outside validation DBFILE range"));
        CHECK_THROWS(t.resolve_validation_targets({}));
    }
    ConfigFile f("PC_ERMSE 0 1\nPC_FRMSE (1) 0 1\n");
    tadah::core::Context ctx;
    HPOTargets t(ctx, f.path.string());
    CHECK_THROWS_WITH(t.resolve_validation_targets({"a", "b"}), Catch::Matchers::Contains("PC_FRMSE"));
}

TEST_CASE("OPTIM uses strict selectors without losing valid syntax", "[pc_dataset][parser]") {
    tadah::core::Context ctx;
    ctx.add("WATOM", "1 2 3");
    ConfigFile valid("OPTIM WATOM (1, 2-3) 0.1 4\nPC_ERMSE 0 1\n");
    CHECK_NOTHROW(HPOTargets(ctx, valid.path.string()));
    for (const auto* selector : {"(0)", "(1,x)", "(1,1)", "(1-2:0)", "(1)(2)", "((1))"}) {
        ConfigFile bad(std::string("OPTIM WATOM ") + selector + " 0.1 4\nPC_ERMSE 0 1\n");
        CHECK_THROWS(HPOTargets(ctx, bad.path.string()));
    }
}

TEST_CASE("PC dataset loss averages weighted losses and preserves legacy global folding", "[pc_dataset][loss]") {
    LossFunction fn;
    ValidationMetricTargets indexed("PC_ERMSE"), global("PC_ERMSE");
    indexed.add({1}, {1, 2, {}, 1});
    indexed.add({2}, {3, 0.5, {}, 2});
    indexed.add({3}, {5, 0, {}, 3});
    indexed.resolve({"a", "b", "c"});
    global.add({}, {1, 2, {}, 1});
    const std::vector<double> values{3, 7, 11};
    std::vector<double> loss;
    CHECK(indexed.loss(fn, 7, values, loss) == Approx(16.0 / 3));
    CHECK(loss == std::vector<double>{8, 8, 0});
    CHECK(global.loss(fn, 7, values, loss) == Approx(72));
    CHECK(loss == std::vector<double>{8, 72, 200});
    CHECK_THROWS(indexed.loss(fn, 4, {3, 5}, loss));
    ValidationMetricTargets single("PC_ERMSE");
    single.add({1}, {1, 2, {}, 1});
    single.resolve({"a"});
    CHECK(single.loss(fn, 3, {3}, loss) == global.loss(fn, 3, {3}, loss));
    ValidationMetricTargets same("PC_ERMSE");
    same.add({1,2,3}, {1, 2, {}, 1});
    same.resolve({"a", "b", "c"});
    CHECK(same.loss(fn, 7, values, loss) == Approx(280.0 / 3));
}

TEST_CASE("Observed loss derivatives and dataset slopes match finite differences", "[pc_dataset][gradient]") {
    for (const auto* name : {"L1", "L2", "LOG_COSH", "MAPE", "HUBER", "RMSLE", "TUKEY"}) {
        INFO(name);
        LossFunction fn(LossFunction::from_string(name));
        for (const double weight : {0.0, 0.5, 3.0}) {
            const double x = 1.7, target = 0.8, h = 1e-6;
            const double fd = (fn.loss(x+h, target, weight) - fn.loss(x-h, target, weight)) / (2*h);
            CHECK(fn.loss_prime_observed(x, target, weight) == Approx(fd).margin(1e-8));
        }
        for (const bool indexed : {false, true}) {
            ValidationMetricTargets t("PC_FRMSE");
            if (indexed) {
                t.add({1}, {0.8, 2, {}, 1});
                t.add({2}, {0.4, 0.5, {}, 2});
            } else t.add({}, {0.8, 2, {}, 1});
            t.resolve({"a", "b"});
            std::vector<double> values{1.7, 2.4}, scratch;
            const auto slopes = t.slopes(fn, 2.05, values);
            for (std::size_t i = 0; i < 2; ++i) {
                const double h = 1e-6;
                auto plus = values, minus = values;
                plus[i] += h; minus[i] -= h;
                const double fd = (t.loss(fn, (plus[0]+plus[1])/2, plus, scratch)
                                 - t.loss(fn, (minus[0]+minus[1])/2, minus, scratch)) / (2*h);
                CHECK(slopes[i] == Approx(fd).margin(1e-8));
            }
        }
    }
}

TEST_CASE("PC dataset filters have independent thread safe run histories", "[pc_dataset][filters]") {
    MetricSkipTracker tracker;
    MetricSkipCaps caps;
    caps.rel_above = 2;
    CHECK_FALSE(tracker.check("ermse[1]", 1, caps));
    CHECK_FALSE(tracker.check("ermse[2]", 100, caps));
    CHECK_FALSE(tracker.check("ermse[2]", 150, caps));
    const auto hit = tracker.check("ermse[1]", 3, caps);
    REQUIRE(hit);
    CHECK(hit->relative);
    CHECK(hit->best == 1);
    CHECK(hit->excess == Approx(0.5));
    MetricSkipTracker fresh;
    CHECK_FALSE(fresh.check("ermse[1]", 3, caps));
    caps.abs_above = 2;
    CHECK(tracker.check("frmse[1]", 3, caps)->absolute);
    CHECK_FALSE(tracker.check("zero", 0, caps));
    REQUIRE(tracker.check("zero", 1, caps));
    CHECK(metric_value_is_finite(tracker.check("zero", 1, caps)->excess));
    std::vector<std::thread> threads;
    for (int i = 1; i <= 8; ++i)
        threads.emplace_back([&, i] { tracker.check("concurrent", i, {}); });
    for (auto& t : threads) t.join();
    CHECK(tracker.check("concurrent", 3, caps)->best == 1);
}

TEST_CASE("PC finite number validation survives release fast math", "[pc_dataset][parser]") {
    CHECK_FALSE(metric_value_is_finite(std::numeric_limits<double>::infinity()));
    CHECK_FALSE(metric_value_is_finite(std::numeric_limits<double>::quiet_NaN()));
    CHECK(metric_value_is_finite(std::numeric_limits<double>::max()));
    for (const auto* token : {"nan", "inf", "-inf", "1e999", "1junk", "+-1", "",
                              "0x1p4", "0X10", "+0x1"})
        CHECK_THROWS(parse_metric_number(token));
    CHECK(parse_metric_number("+1e2") == 100);
}
