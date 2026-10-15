#pragma once
/*
 * Strategy rules -- which STRATEGY TYPE can drive which optimiser, and
 * which STRATEGY keys each TYPE takes. One set of free functions, the
 * single source of truth for:
 *   - HPOTargets::read(): require_driver() when an OPTIMIZER or EXPLORE
 *     block ends, require_inner_driver() when an INNER block ends, and
 *     require_strategy_key() and require_prestage_block() when a STRATEGY
 *     block ends;
 *   - validate_hpo_setup(): validate_run_policy(), at startup, before the
 *     host is built or any data is loaded;
 *   - MultiStartPolicy::validate() and StagedPolicy::validate(), which the
 *     policies call again when they run, and run_inner(), which applies
 *     require_inner_backend() to every optimiser it runs (a safety net for
 *     specs built outside HPOTargets).
 * Every message is built here, so all of them give the same text.
 */

#include <tadah/hpo/optimizer_spec.h>

#include <string>

namespace tadah::hpo {

struct OptimizerCaps;

/* The phase an optimiser drives, for require_driver(). */
enum class DriverPhase {
    Run,       // the STRATEGY-level optimiser: the PLAIN run, every
               // MULTISTART start, the PRESTAGE refinement
    Explore,   // the PRESTAGE exploration (the EXPLORE block)
};

/* LIB and ALGO name the optimiser of a phase; OptimizerSpec leaves both
 * "NOT SET" until the configuration sets them. Throws std::runtime_error
 * when either is missing, naming the block where they belong, and when
 * they name a backend the caps registry refuses or does not have
 * (require_registered_backend(), optimizer_caps.h), naming the phase as
 * validate_run_policy() does: "the STRATEGY-level optimiser:
 * NLOPT:LN_NEWUOA_BOUND is refused: ...", "the PRESTAGE refinement:
 * unknown optimiser NLOPT:LD_FOO: LIB NLOPT has no ALGO LD_FOO; ..." (a
 * REFINE block's optimiser), "the EXPLORE block: ...". HPOTargets::read()
 * applies it where the block closes, so the error names that line. */
void require_driver(const OptimizerSpec& phase_spec, DriverPhase phase);

/* The same rule for an INNER block, the local optimiser NLopt's MLSL and
 * AUGLAG run: `enclosing` names the block it sits in (STRATEGY, EXPLORE or
 * REFINE): "the INNER block inside STRATEGY: NLOPT:LN_NEWUOA_BOUND is
 * refused: ...", "... unknown optimiser ...". HPOTargets::read() applies
 * it when the INNER block closes. */
void require_inner_driver(const OptimizerSpec& inner_spec,
                          const std::string& enclosing);

/* An INNER block is the local search that NLopt's MLSL and AUGLAG run
 * inside their own run (build_nlopt hands the INNER ALGO to nlopt::opt):
 * MLSL starts it at the points it samples in the OPTIM box (mlsl.c), AUGLAG,
 * which HPO sets no constraint, once at its own starting point (auglag.c).
 * Throws std::runtime_error when
 *   - `parent` is an MLSL or AUGLAG without a default local optimiser
 *     (OptimizerCaps::default_inner: AUGLAG, AUGLAG_EQ, G_MLSL, G_MLSL_LDS)
 *     and has no INNER block: "NLOPT:AUGLAG needs an INNER block (the
 *     STRATEGY-level optimiser): NLopt has no default local optimiser for
 *     it; ..." -- NLopt refused it only once the run had started;
 * and, when `parent` has one, unless
 *   - the INNER optimiser is not refused (reject_refused_backend(),
 *     optimizer_caps.h: "the INNER block of the STRATEGY-level optimiser:
 *     NLOPT:LN_NEWUOA_BOUND is refused: ..."),
 *   - the parent runs one (OptimizerCaps::accepts_inner),
 *   - the INNER optimiser is an NLopt algorithm (LIB NLOPT),
 *   - it does not need an INNER block of its own (INNER blocks do not
 *     nest), and
 *   - it is a local optimiser (not OptimizerCaps::is_global; owner
 *     decision, MK, 2026-10-15: no nested global search). Every search that
 *     does not start at the point it is given (OptimizerCaps::starts_at_x0:
 *     StoGO, DIRECT, AGS) is global, and its refusal says where it starts
 *     instead: "the INNER block of the PRESTAGE refinement: NLOPT:AUGLAG
 *     cannot run NLOPT:GD_STOGO as its INNER search: an INNER search is a
 *     local search, which NLOPT:AUGLAG starts at its own starting point,
 *     and NLOPT:GD_STOGO is a global optimiser that does not start at the
 *     point it is given (StoGO evaluates ...; NLopt never reads that
 *     point). Use a local NLopt algorithm in the INNER block, which starts
 *     at the point it is given (NLOPT:LD_AUGLAG, ..., every one), or run
 *     NLOPT:GD_STOGO as the exploration (EXPLORE block) or once, as
 *     STRATEGY TYPE PLAIN." (under MLSL "... which NLOPT:G_MLSL_LDS starts
 *     at the points it samples in the OPTIM box, and NLOPT:GN_CRS2_LM is a
 *     global optimiser. ...").
 * `phase` names the optimiser whose INNER block it is ("the EXPLORE
 * block"); run_inner() passes "" for a spec built at run time. The rule
 * holds for every INNER block of every phase and STRATEGY TYPE:
 * validate_run_policy() applies it at the INNER block's line, the policies'
 * validate() and run_inner() again. */
void require_inner_backend(const OptimizerSpec& parent,
                           const std::string& phase);

/* True when a backend can run once from each of several starting points:
 * a local optimiser, or a global one that consumes several seeds in one
 * run (DLIB:GFS). A global optimiser that consumes a single starting point
 * cannot. MULTISTART and the PRESTAGE refinement over TOPK > 1 candidates
 * both need this. */
bool runs_from_each_start(const OptimizerCaps& caps);

/* STRATEGY TYPE MULTISTART: `spec` names an optimiser, and one that
 * runs_from_each_start() accepts, for any N_STARTS. A multi-seed backend
 * (OptimizerCaps::accepts_multi_seed: DLIB:GFS) runs every start in one
 * optimiser (plan_multistart(), phase_specs.h), so BUDGET_PER_START, each
 * start's budget, has no effect there and is rejected: "BUDGET_PER_START
 * has no effect with DLIB:GFS, which runs every start in one optimiser;
 * use MAXEVAL". */
void require_multistart_backend(const OptimizerSpec& spec);

/* STRATEGY TYPE PRESTAGE: an EXPLORE block that names an optimiser. */
void require_prestage_explore(const OptimizerSpec& spec);

/* STRATEGY TYPE PRESTAGE: `spec` names a refinement optimiser; one whose
 * first call is at the point it is given, for any TOPK: the refinement
 * starts from the exploration's candidates, which a search that does not
 * start there (OptimizerCaps::starts_at_x0: NLopt's StoGO, DIRECT and AGS)
 * never uses -- "STRATEGY TYPE PRESTAGE cannot refine with
 * NLOPT:GN_DIRECT_L: it does not start at the point it is given (DIRECT
 * evaluates the centre of the OPTIM box first and then divides the whole
 * box; NLopt never reads that point), so the refinement would never use
 * the exploration's candidates, for any TOPK. Use a local optimiser, which
 * starts at each candidate (CERES:LBFGS, ..., every one), or run
 * NLOPT:GN_DIRECT_L as the exploration (EXPLORE block) or once, as
 * STRATEGY TYPE PLAIN." (checked before the TOPK rule); and with a
 * configured TOPK > 1 one that runs_from_each_start() accepts -- whatever
 * number of candidates the exploration will harvest. An AUGLAG whose INNER
 * search is such a search is require_inner_backend()'s: an INNER search is
 * local. */
void require_prestage_refinement(const OptimizerSpec& spec);

/* The STRATEGY-level phase of `s` as validate_run_policy() names it: "the
 * PRESTAGE refinement" under STRATEGY TYPE PRESTAGE, "the STRATEGY-level
 * optimiser" under PLAIN and MULTISTART. */
std::string run_phase_name(const OptimizerSpec& s);

/* Every rule of spec.strategy_type: require_driver() for PLAIN,
 * require_multistart_backend() for MULTISTART, require_prestage_explore()
 * and require_prestage_refinement() for PRESTAGE; then
 * require_inner_backend() for the INNER block of the STRATEGY-level
 * optimiser and, under PRESTAGE, for the INNER block of the EXPLORE
 * block. A refusal of a
 * parsed spec names the line of the block it is about
 * (OptimizerSpec::block_line, with_block_line): "Error on line N: ...
 * \nLine: ENDOPTIMIZER" (or the INNER block's ENDINNER, the EXPLORE
 * block's ENDEXPLORE). */
void validate_run_policy(const OptimizerSpec& spec);

/* The refusals of validate_run_policy() alone, with its text: throws
 * require_registered_backend()'s error (optimizer_caps.h) for the first
 * refused or unknown backend of the EXPLORE block (under PRESTAGE), the
 * STRATEGY-level optimiser, its INNER block and the EXPLORE block's INNER
 * block (a block without LIB or ALGO is require_driver()'s), named as
 * validate_run_policy() names them ("the EXPLORE block", "the PRESTAGE
 * refinement", "the INNER block of the STRATEGY-level optimiser", ...). It
 * looks up no caps, so validate_phase_backends() (backend_rules.h) runs it
 * before strategy_phase_specs() (phase_specs.h), which needs the caps of
 * the STRATEGY-level backend to plan a MULTISTART from INIT points or a
 * PRESTAGE refinement of TOPK > 1. */
void reject_refused_backends(const OptimizerSpec& spec);

/* STRATEGY keys that belong to one TYPE: N_STARTS, PARALLEL and
 * BUDGET_PER_START to MULTISTART, TOPK to PRESTAGE. Under another TYPE
 * such a key has no effect, so require_strategy_key() throws, naming the
 * TYPE the key needs. `type_given` says whether the block had a TYPE line
 * (PLAIN is the default). Keys of no single TYPE are always accepted. */
bool is_strategy_type_key(const std::string& key);
void require_strategy_key(const std::string& key, StrategyType type,
                          bool type_given);

/* The EXPLORE and REFINE blocks configure the phases of TYPE PRESTAGE,
 * which must be declared before them; a later TYPE line of the same
 * STRATEGY block could still change it. Throws unless `type`, the TYPE the
 * block ends with, is PRESTAGE. `block` is "EXPLORE" or "REFINE". */
void require_prestage_block(const std::string& block, StrategyType type);

/* "PLAIN", "MULTISTART" or "PRESTAGE". */
const char* strategy_type_name(StrategyType type);

}  // namespace tadah::hpo
