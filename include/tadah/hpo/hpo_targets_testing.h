#pragma once
/* hpo_targets_testing.h -- test seam of the target-file parser (the
 * registry-driven parser of src/hpotarget_parser.cpp, which HPOTargets
 * reads its target file with). Not part of the public API.
 *
 * describe_parse() writes what the parse of a target file gave as a
 * canonical text, so two parses can be compared as text: the expected
 * results of the frozen corpus of target files
 * (tests/test_hpotarget_corpus.cpp). One line per setting,
 * `<path> = <value>`, in a fixed order:
 *
 *   - a string is quoted, with \" \\ \n \r \t and \u00XX escapes;
 *   - a number is the shortest text that reads back as the same double
 *     (core::format_roundtrip: "1000", "0.25", "1e-20", "inf", "nan");
 *     a count or an index is an integer;
 *   - a flag is true or false, a value never set `none`, an enumerator its
 *     word (NUMERICAL, PRESTAGE, ...);
 *   - a list of plain values stands on one line, `[a, b]`; a list of
 *     records gives its `.size` and then each record, `<path>[i].<field>`;
 *     a map gives its `.size` and its entries in key order,
 *     `<path>[<key>]`;
 *   - a LAMMPS script is given by its number of lines and the FNV-1a hash
 *     of its lines (`.size`, `.fnv1a`), not line by line, and the LAMMPS
 *     command line of a PC_LAMMPS line from its second argument on
 *     (`.invar[1:]`): the first is the LAMMPS library the build links.
 *
 * The dump holds every member of ParsedTarget the parse sets, with the
 * OptimizerSpec of the run and its INNER and PRESTAGE specs, each PC_LAMMPS
 * and PC_FAILPROBE line, each PC_OBSERVABLE term with its options, the
 * boxes of the OPTIM lines (optimap) and the training configuration the
 * targets read (context[<KEY>]: the parse inserts the defaults of the keys
 * it reads). It leaves out what the constructor derives once the parse has
 * returned and what does not depend on the file: the search space and the
 * internal bounds, the starting points of the INIT and RESTART chains, the
 * copies of FAILSCORE and FAILSCORE_GRADING in the specs, the training
 * record and NUMERIC. A seed the constructor draws (an INIT block or an
 * optimiser without SEED) and the seed an exploration takes from the
 * STRATEGY-level SEED are shown as the parse leaves them (`none`), and so is
 * the PARAM fd_step that NOISE AUTO derives (left out of `param`), so the
 * dump of one file does not change from one construction to the next.
 * The environment override TADAH_OPTIM_SCALE, which the constructor
 * applies, shows in scale_policy.
 *
 * Throws std::logic_error for a PC_OBSERVABLE kind it does not know, whose
 * options it could not show.
 *
 * drawn_seeds() gives the seeds the constructor drew for the run (each
 * OptimizerSpec of the dump, the run's and its INNER and PRESTAGE specs,
 * and each INIT chain of theirs, RESTART's included, whose seed_drawn is
 * set), which the lines it prints name and which differ from one
 * construction to the next. */

#include <tadah/hpo/hpo_targets.h>

#include <cstdint>
#include <string>
#include <vector>

namespace tadah::hpo {
namespace testing {

std::string describe_parse(const ParsedTarget& targets);
std::vector<std::uint64_t> drawn_seeds(const ParsedTarget& targets);

}  // namespace testing
}  // namespace tadah::hpo
