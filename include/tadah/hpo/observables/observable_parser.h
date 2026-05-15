#ifndef TADAH_HPO_OBSERVABLE_PARSER_H
#define TADAH_HPO_OBSERVABLE_PARSER_H

#include <tadah/hpo/observables/observable.h>

#include <memory>
#include <string>

namespace tadah {
namespace hpo {

/** \brief Parse one PC_OBSERVABLE config line into an Observable.
 *
 *  Input is the substring AFTER the `PC_OBSERVABLE` keyword on the
 *  `.hpotarget` line. The first token is the observable kind; remaining
 *  tokens are kind-specific `--flag value` pairs.
 *
 *  Supported kinds in MVP:
 *    - `cohesive_energy` -- required: `--species`, `--target`, `--weight`,
 *      `--hotvar a0 <lo> <hi> [<init>]`. Optional: `--lattice bcc`
 *      (default), `--supercell nx ny nz` (default 4 4 4), `--loss
 *      L2|L1|HUBER|...` (default L2).
 *
 *  Throws `std::runtime_error` with a human-readable message on any
 *  parse failure (unknown kind, missing required flag, malformed value).
 */
std::unique_ptr<Observable> parse_pc_observable_line(const std::string& cmd);

}  // namespace hpo
}  // namespace tadah

#endif  // TADAH_HPO_OBSERVABLE_PARSER_H
