#include <tadah/hpo/observables/observable_parser.h>

#include "hpotarget_reads.h"

#include <tadah/core/number_format.h>
#include <tadah/core/periodic_table.h>

#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/observables/cohesive_energy.h>
#include <tadah/hpo/strict_parse.h>         // parse_number, parse_uint

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tadah {
namespace hpo {

namespace {

/* Tokenise on whitespace (no shell-style quoting; PC_OBSERVABLE values
 * are simple identifiers / numbers). */
std::vector<std::string> ws_tokens(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream iss(s);
    std::string t;
    while (iss >> t) out.push_back(t);
    return out;
}

}  // anonymous namespace

std::unique_ptr<Observable> parse_pc_observable_line(
    const std::string& cmd, const tadah::LossFunction& default_loss)
{
    auto tokens = ws_tokens(cmd);
    if (tokens.empty())
        throw std::runtime_error("PC_OBSERVABLE: empty line");

    /* The kind is the word of a form of PC_OBSERVABLE in the registry,
     * which its handler reads (hpotarget_handlers.cpp). */
    namespace R = hpotarget_registry;
    const std::string& kind = tokens[0];
    for (const R::Form& f : R::kForms) {
        if (f.line != "PC_OBSERVABLE" || f.word != kind) continue;
        R::FormRead read(f, tokens, default_loss);
        R::call(f, read);
        return std::move(read.observable);
    }
    throw std::runtime_error("PC_OBSERVABLE: unknown observable kind '" + kind + "'");
}

}  // namespace hpo
}  // namespace tadah
