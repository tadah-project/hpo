#pragma once
/*
 * PotHeaderWriter -- writes commented (#) metadata headers into potential files.
 *
 * Two levels of detail:
 *   * write_best_header()    -- full, self-contained header for best_pot.tadah
 *                              (enough information to reproduce the HPO run)
 *   * write_archive_header() -- compact header for periodic pot/best_pot archives
 *
 * All output lines are prefixed with '#' so TADAH parsers skip them.
 * Target line width <= 80 characters where practical.
 */

#include <tadah/core/context.h>
#include <tadah/mlip/load_transforms.h>
#include <tadah/mlip/structure_db.h>
#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

namespace tadah::hpo {

// Forward declarations
class  HPOTargets;
struct OptimizerSpec;
template<class P> struct EvalResult;

class PotHeaderWriter
{
public:
    /* ------------------------------------------------------------------ *
     *  Full header for best_pot.tadah -- self-contained run summary
     * ------------------------------------------------------------------ */
    template<class ParamVec>
    static void write_best_header(
        std::ostream&                       os,
        const EvalResult<ParamVec>&         result,
        const HPOTargets&                   trg,
        const core::Context&                master_ctx,
        const core::Context&                val_ctx,
        const mlip::StructureDB&            train_stdb,
        const mlip::StructureDB&            val_stdb,
        const std::vector<std::string>&     param_labels,
        const mlip::LoadTransformReport&    report,
        std::size_t                         cw,
        int                                 prec);

    /** Persist the hot variables produced by this exact evaluation. Unlike
     * the process-global warm-start cache, this snapshot cannot be changed by
     * a later concurrent candidate before the file is rendered. */
    template<class ParamVec>
    static void write_candidate_hot(
        std::ostream&                       os,
        const EvalResult<ParamVec>&         result);

    /* ------------------------------------------------------------------ *
     *  Compact header for periodic pot archives (pot_N.tadah)
     * ------------------------------------------------------------------ */
    template<class ParamVec>
    static void write_archive_header(
        std::ostream&                os,
        const EvalResult<ParamVec>&  result,
        std::size_t                  cw,
        int                         prec);

    /* ------------------------------------------------------------------ *
     *  Compact header for best pot archives (best_pot_N.tadah)
     * ------------------------------------------------------------------ */
    template<class ParamVec>
    static void write_best_archive_header(
        std::ostream&                os,
        const EvalResult<ParamVec>&  result,
        const HPOTargets&            trg,
        std::size_t                  cw,
        int                         prec);

    /* ------------------------------------------------------------------ *
     *  The potential below the header (every file above)
     * ------------------------------------------------------------------ */
    /** Writes the potential @p pot, the context of an evaluation, after its
     * header: pot.write_potential(os, sig_digits) of a copy without the
     * settings of the run that mlip::remove_run_inputs removes (VERBOSE,
     * NUMERIC, OUTFILE, AUDIT, AUDIT_SAMPLE, HEALTH_LOG, HPOTARGET,
     * VALIDATION); every other key is written. The context itself keeps
     * them, so the PC_LAMMPS potential built from it is built at the run's
     * VERBOSE. best_pot.tadah, the durable best and the BDUMP and DUMP
     * archives are written through it. */
    static void write_potential_body(std::ostream&         os,
                                     const core::Context&  pot,
                                     int                   sig_digits);

private:
    static constexpr int W  = 78;   // target line width (inside "# ")
    static constexpr int KW = 24;   // key column width

    /* ---- tiny formatting helpers ------------------------------------ */
    static void rule   (std::ostream& os);                    // # ====...
    static void thin   (std::ostream& os);                    // # ----...
    static void blank  (std::ostream& os);                    // #
    static void heading(std::ostream& os, const std::string& title);
    static void kv     (std::ostream& os,
                        const std::string& key,
                        const std::string& val);
    static void line   (std::ostream& os, const std::string& text);
    static void wrap   (std::ostream& os,
                        const std::string& prefix,
                        const std::vector<std::string>& items,
                        int max_width = W);

    static std::string fmt_double(double v, int prec = 4);
    static std::string fmt_timestamp();
    static std::string file_timestamp(const std::string& path);

    /* ---- section writers -------------------------------------------- */
    static void write_banner       (std::ostream& os, std::size_t best_iter,
                                    std::size_t eval_iter, double loss);
    static void write_data_section (std::ostream& os,
                                    const mlip::StructureDB& train,
                                    const mlip::StructureDB& val,
                                    const core::Context& train_ctx,
                                    const core::Context& val_ctx);
    static void write_load_transforms_section(std::ostream& os,
                                              const mlip::LoadTransformReport& report);
    static void write_optim_section(std::ostream& os,
                                    const HPOTargets& trg,
                                    const std::vector<std::string>& labels);
    static void write_loss_section (std::ostream& os,
                                    const HPOTargets& trg);
    template<class ParamVec>
    static void write_result_table (std::ostream& os,
                                    const EvalResult<ParamVec>& e,
                                    const HPOTargets& trg,
                                    std::size_t cw, int prec);
};

}   // namespace tadah::hpo
