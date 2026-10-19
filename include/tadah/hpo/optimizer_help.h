#pragma once
/*
 * optimizer_help -- the text of `tadah explain ALGO` and `tadah explain
 * <LIB>:<ALGO>`: the optimisers a `tadah hpo` target's LIB and ALGO can
 * name, from the caps registry (optimizer_caps.h), and one optimiser's ALGO
 * help (OptimizerSpec::print_caps). tadah sets explain_optimiser as MLIP's
 * explain hook (mlip::engines::set_explain_hook), so MLIP's explain command
 * reaches it without depending on HPO.
 */

#include <ostream>
#include <string>

namespace tadah::hpo {

struct OptimizerCaps;

/* What a registered backend is, its row of the table after the ALGO name:
 * global or local; the local search it runs (MLSL, AUGLAG: NLopt's default
 * one, or an INNER block), else gradient-based or derivative-free; then
 * whether it needs a stop and whether it ignores the OPTIM box ("global,
 * derivative-free, needs a stop"). */
std::string optimiser_row_text(const OptimizerCaps& c);

/* The table of the optimisers: a short preface, then for each library
 * (optimizer_libraries(), sorted) a "LIB <lib>" line and one row per
 * registered backend, sorted, with what it is -- global or local,
 * gradient-based or derivative-free (MLSL and AUGLAG: the local search they
 * run), whether it needs a stop or ignores the OPTIM box
 * (optimiser_row_text); then the refused
 * backends and how to see one. `lib` not empty: that library's section
 * alone. */
void print_optimiser_table(std::ostream& out, const std::string& lib = "");

/* What `tadah explain <name>` prints for an HPO optimiser name, read in any
 * letter case: ALGO or LIB -- the table (print_optimiser_table); LIB:ALGO,
 * or an ALGO alone -- the ALGO help of every registered or refused backend
 * of that name (OptimizerSpec::print_caps; ASA is ASA:ASA); a LIB alone --
 * its section of the table. Writes to `out` and returns true for such a
 * name; returns false, writing nothing, for any other name; throws
 * std::runtime_error "tadah explain: " + require_registered_backend()'s
 * text for a LIB:ALGO that names no backend ("tadah explain: unknown
 * optimiser NLOPT:LD_FOO: LIB NLOPT has no ALGO LD_FOO; ..."). */
bool explain_optimiser(const std::string& name, std::ostream& out);

}  // namespace tadah::hpo
