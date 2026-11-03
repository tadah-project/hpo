#pragma once
/* hpotarget_reads.h -- what a handler of the registry-driven parser of the
 * target file reads (hpotarget_handlers_gen.h declares the handlers,
 * hpotarget_handlers.cpp defines them): the line of a key (KeyRead), an
 * option of a line (OptionRead) and the line of a form (FormRead), with
 * the readers of a value through strict_parse.h that the generated tables
 * of the registry describe (an argument's kind, what, min, max, noun,
 * words and msg). Private to HPO: the parser (hpotarget_parser.cpp), the
 * reader of the optimiser keys (dispatch_opt_attr, optimizer_block.cpp),
 * the reader of the options of a PC_LAMMPS or PC_FAILPROBE line
 * (LammpsSim::parse_tokens) and the reader of a PC_OBSERVABLE line
 * (parse_pc_observable_line) call the handlers through it. */

#include <tadah/hpo/hpotarget_handlers_gen.h>
#include <tadah/hpo/hpotarget_registry_gen.h>
#include <tadah/hpo/observables/cohesive_energy.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace tadah {
class LossFunction;
namespace hpo {

class LammpsSim;
class Observable;
struct OptimizerSpec;
struct MetricSkipCaps;
struct RhoGateSpec;
struct RhoBarGateSpec;

namespace hpotarget {
class Parser;
}

namespace hpotarget_registry {

/* The entry of a key, of a line option and of a form, by its id. */
inline const Key& key_entry(KeyId id)
{
    return kKeys[static_cast<std::size_t>(id)];
}
inline const LineOption& option_entry(LineOptionId id)
{
    return kLineOptions[static_cast<std::size_t>(id)];
}
inline const Form& form_entry(FormId id)
{
    return kForms[static_cast<std::size_t>(id)];
}

/* The line of a key. The parser calls the key's handler for each step of
 * the line that reads it (`step`: Value, then Options for a line with
 * options); dispatch_opt_attr() calls the handler of a key of the
 * optimiser group with no parser, for `tadah refit` too. */
struct KeyRead {
    KeyRead(const Key& k, const std::string& w, std::istringstream& in)
        : key(k), word(w), iss(in) {}

    const Key& key;
    const std::string& word;       // the key as the line writes it
    std::istringstream& iss;       // the line after the key
    LineStep step = LineStep::Value;
    hpotarget::Parser* parser = nullptr;
    OptimizerSpec* spec = nullptr;  // the optimiser a key of the group sets
    /* Set by the handler of a key of a block: what the line sets once
     * (its repeat rule), and what that is, for the refusal of a second
     * line in another block that sets it. */
    const void* target = nullptr;
    std::string_view what;

    /* The parser; a logic_error for a key read without one. */
    hpotarget::Parser& p() const;
};

/* The options of a PC_LAMMPS or PC_FAILPROBE line, which
 * LammpsSim::parse_tokens reads: the tokens of the line, the position of
 * the option, and the rules of the line (an option given once, a LAMMPS
 * variable defined once, the reserved variable maxtime, what a word after
 * the option's values is refused as). */
struct ScriptOptions {
    LammpsSim& sim;
    const std::string& key;                  // PC_LAMMPS or PC_FAILPROBE
    const std::vector<std::string>& tokens;
    std::size_t& i;                          // the option, then past it
    const tadah::LossFunction& default_loss;
    std::function<void(const std::string&)> once;
    std::function<void(const std::string&, const std::string&)> define_variable;
    std::function<void(const std::string&, const std::string&)> reject_reserved;
    std::function<void(const std::string&)> option_read;
};

/* The options of a gate line (RHOGATE, RHOBARGATE): the value of the
 * option, the line's NPTS and SEED and RHOGATE's R_LO. */
struct GateOptions {
    const std::string& key;
    const std::string& value;
    std::size_t& npts;
    bool& seed_warn;
    std::optional<double>* r_lo;             // RHOGATE only
};

/* The options of a metric line (--skip-above, --skip-rel). */
struct MetricOptions {
    const std::string& key;
    std::istringstream& iss;
    MetricSkipCaps& caps;
};

/* The options --gate and --s0 of a PC_FAILPROBE line, which the line reads
 * before LammpsSim reads the others: the tokens and the position. */
struct ProbeOptions {
    const std::vector<std::string>& tokens;
    std::size_t& i;
    bool& gate_seen;
    bool& s0_seen;
    double& gate;
    double& s0;
};

/* The options of a PC_OBSERVABLE cohesive_energy line. */
struct CohesiveOptions {
    const std::vector<std::string>& tokens;
    std::size_t& i;
    CohesiveEnergyObservable::Config& cfg;
    std::vector<std::string>& seen;
    bool& seen_species;
    bool& seen_target;
    bool& seen_weight;
    bool& seen_hotvar;
    std::string& target_tok;
    std::string& hotvar_lo_tok;
    std::string& hotvar_hi_tok;
    std::string& hotvar_init_tok;
};

/* One option of a line: the handler reads it with the state of its
 * line, one of the pointers below. */
struct OptionRead {
    OptionRead(const LineOption& o, const std::string& w)
        : option(o), word(w) {}

    const LineOption& option;
    const std::string& word;                 // the option as written
    ScriptOptions* script = nullptr;
    GateOptions* gate = nullptr;
    MetricOptions* metric = nullptr;
    ProbeOptions* probe = nullptr;
    CohesiveOptions* cohesive = nullptr;
};

/* The line of a form of PC_OBSERVABLE: its tokens (the kind first) and the
 * loss function of the LOSS line above it; the handler sets the
 * observable. */
struct FormRead {
    FormRead(const Form& f, const std::vector<std::string>& t,
             const tadah::LossFunction& loss)
        : form(f), tokens(t), default_loss(loss) {}

    const Form& form;
    const std::vector<std::string>& tokens;
    const tadah::LossFunction& default_loss;
    std::unique_ptr<Observable> observable;
};

/* The handler of an option or a form, whose entry has one (not one of the
 * OPTIM line, which MLIP reads). */
inline void call(const LineOption& o, OptionRead& r)
{
    kOptionHandlers[static_cast<std::size_t>(o.id)](r);
}
inline void call(const Form& f, FormRead& r)
{
    kFormHandlers[static_cast<std::size_t>(f.id)](r);
}
inline void call(const Key& k, KeyRead& r)
{
    kKeyHandlers[static_cast<std::size_t>(k.id)](r);
}

/* ---- the readers of a value ------------------------------------------ */

/* {what} of an argument, filled with the key as written. */
std::string what_of(const Arg& a, std::string_view key);

/* An integer argument: parse_uint() with the argument's what, min and max
 * (no max: any 64-bit value); a whole number below min is refused with
 * the argument's msg when it has one (the note of parse_uint: why such a
 * number is refused). */
std::uint64_t read_integer(const Arg& a, const std::string& tok,
                           std::string_view key);
/* A number argument: parse_finite() with the argument's what. */
double read_number(const Arg& a, const std::string& tok,
                   std::string_view key);
/* A number argument within its min (open or not) and max: read_number(),
 * then the argument's msg for a value outside. */
double read_number_in_range(const Arg& a, const std::string& tok,
                            std::string_view key);
/* A positive argument: parse_positive() with the argument's what and
 * noun. */
double read_positive(const Arg& a, const std::string& tok,
                     std::string_view key);
/* A word argument: one of its words, as the registry writes it (in any
 * case when the argument is case any); the argument's msg otherwise. */
std::string read_word(const Arg& a, const std::string& tok,
                      std::string_view key);
/* The word `word` of the options of the line or form `line` (in any case
 * for an option that is case any); nullptr for none. */
const LineOption* find_option(std::string_view line, const std::string& word);
/* The words a refused option is refused for (an option the registry
 * keeps to refuse it, such as --ncpu, is not taken). */
bool takes_option(const LineOption& o, std::string_view line);

}  // namespace hpotarget_registry
}  // namespace hpo
}  // namespace tadah
