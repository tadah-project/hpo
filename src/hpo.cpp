#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/algorithms.h>
#include <tadah/hpo/nlopt_adapter.h>
#include <tadah/hpo/dlib_adapter.h>
#include <tadah/hpo/hpo.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/core/context.h>
#include <tadah/core/blueprint_parser.h>
#include <tadah/mlip/structure_db.h>

#include <filesystem>
#include <set>
#include <unistd.h>   // getpid()
// #include <dlib/global_optimization/find_max_global.h>
#include <dlib/optimization.h>
#include <dlib/global_optimization.h>

#include <nlopt.hpp>

#include <random>
#include <variant>

namespace tadah::hpo {
using R = LammpsRunner;
using V = dlib::matrix<double,0,1>;
int hpo_run(core::Context& ctx)
{
    HPOTargets trg(ctx, ctx.get<std::string>("HPOTARGET"));
    std::string val = ctx.get<std::string>("VALIDATION");

    /* ── Open run journal ──────────────────────────────────────────── */
    auto& log = HPOLog::instance();
    {
        std::string version = "unknown";
        try { version = core::BlueprintParser::getBlueprintTadah().version; }
        catch (...) {}
        std::string cwd = std::filesystem::current_path().string();
        log.open("hpo_run.log", version, cwd, static_cast<int>(getpid()));
    }

    /* ── Build param labels for the search space section ────────────── */
    std::vector<std::string> param_labels;
    for (const auto& kv : trg.optimap) {
        for (std::size_t i = 0; i < kv.second.opt.size(); ++i) {
            if (!kv.second.opt[i]) continue;
            std::string lbl = kv.first;
            if (kv.second.opt.size() > 1) lbl += "[" + std::to_string(i) + "]";
            param_labels.push_back(lbl);
        }
    }

    /* ── Build element string ────────────────────────────────────────── */
    std::string elem_str;
    try {
        auto elems = mlip::StructureDB::find_unique_elements(ctx);
        for (const auto& e : elems) {
            if (!elem_str.empty()) elem_str += ", ";
            elem_str += e.symbol;
        }
    } catch (...) { elem_str = "(could not determine)"; }

    /* ── Write setup sections ───────────────────────────────────────── */
    log.write_files_section(ctx.get<std::string>("HPOTARGET"), val, trg);

    log.write_optimizer_section(trg.spec, param_labels);

    /* Build HPO_Host (loads train + val structures) */
    HPO_Host<V,R> host(ctx, trg, val);
    log.write_potential_section(ctx, host.train_size(), host.val_size(), elem_str);
    log.write_loss_section(trg);
    log.start_progress();

    const OptimizerSpec& sp = trg.spec;

    /* quick info */
    std::cout<<"[HPO] "<<sp.lib<<":"<<sp.algo<<"   dim="<<sp.low.size()<<"\n";

    /* ── Run optimiser ──────────────────────────────────────────────── */
    std::string result_str = "UNKNOWN";
    bool used_nlopt = (sp.lib == "NLOPT");
    try {
        if      (sp.lib=="TADAH" && sp.algo=="RANDOM")
            run_random<V,R>(sp, host);
        else if (sp.lib=="TADAH" && sp.algo=="GRID")
            run_grid<V,R>  (sp, host);
        else if (sp.lib=="DLIB" && sp.algo=="GFS")
            run_dlib_global(sp, host);
        else if (sp.lib=="NLOPT")
            run_nlopt<V,R>(sp, host);   // run_nlopt calls log.finish() itself
        else
            throw std::runtime_error("unknown optimizer "+sp.lib+":"+sp.algo);

        if (!used_nlopt) {
            host.flush_remaining();
            result_str = "COMPLETED — all evaluations done";
            log.finish(result_str);
        }
    } catch (const std::exception& e) {
        result_str = std::string("STOPPED with error: ") + e.what();
        log.warning(std::string("Optimiser threw: ") + e.what());
        if (!used_nlopt) {
            host.flush_remaining();
            log.finish(result_str);  // nlopt adapter already called finish()
        }
        throw;   // re-throw so HPOEngine can report it
    }

    return 0;
}
} // namespace tadah::hpo
