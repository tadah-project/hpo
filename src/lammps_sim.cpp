#include <tadah/core/logger.h>
#include <tadah/hpo/hpo_mpi.h>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/strict_parse.h>
#include <iostream>

#include <cstring>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <unordered_set>

namespace tadah::hpo { bool try_parse_double(const std::string& s, double& value); }

namespace {

using tadah::hpo::try_parse_double;

inline std::string to_lower_ascii(std::string s)
{
    for (char &c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

inline std::string trim_ascii(std::string s)
{
    const auto beg = s.find_first_not_of(" \t\r\n");
    if (beg == std::string::npos) return "";
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(beg, end - beg + 1);
}

std::vector<std::string> split_lammps_words(const std::string& line)
{
    std::vector<std::string> out;
    std::string cur;
    bool in_single = false;
    bool in_double = false;

    for (char c : line) {
        if (!in_double && c == '\'') {
            in_single = !in_single;
            continue;
        }
        if (!in_single && c == '"') {
            in_double = !in_double;
            continue;
        }

        if (!in_single && !in_double && std::isspace(static_cast<unsigned char>(c))) {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
            continue;
        }

        cur.push_back(c);
    }

    if (!cur.empty()) out.push_back(cur);
    return out;
}

std::optional<std::string>
try_get_variable_value(const LammpsRunner::LMPPtr& lmp,
                       const std::string& name)
{
    if (!lmp || !lmp->input || !lmp->input->variable) return std::nullopt;
    try {
        const int idx = lmp->input->variable->find(name.c_str());
        if (idx < 0) return std::nullopt;
        char* raw = lmp->input->variable->retrieve(name.c_str());
        if (!raw) return std::nullopt;
        return std::string(raw);
    } catch (...) {
        // Validation should not force evaluation of equal-style variables
        // whose formulas depend on runtime simulation state (e.g. lx,
        // pe/atoms, computes after run 0, etc.). Treat those as
        // unavailable for static substitution and fall back to the
        // literal-scan result or placeholder handling.
        return std::nullopt;
    }
}

bool substitute_known_variables(const LammpsRunner::LMPPtr& lmp,
                                const std::string& in,
                                std::string& out)
{
    out.clear();
    out.reserve(in.size());

    for (std::size_t i = 0; i < in.size(); ++i) {
        if (in[i] != '$') {
            out.push_back(in[i]);
            continue;
        }

        if (i + 1 >= in.size()) {
            out.push_back(in[i]);
            continue;
        }

        if (in[i + 1] == '(') {
            out.push_back(in[i]);
            continue;
        }

        if (in[i + 1] == '{') {
            const std::size_t close = in.find('}', i + 2);
            if (close == std::string::npos) return false;
            const std::string name = in.substr(i + 2, close - (i + 2));
            auto value = try_get_variable_value(lmp, name);
            if (!value.has_value()) return false;
            out += *value;
            i = close;
            continue;
        }

        if (std::isalnum(static_cast<unsigned char>(in[i + 1])) || in[i + 1] == '_') {
            const std::string name(1, in[i + 1]);
            auto value = try_get_variable_value(lmp, name);
            if (!value.has_value()) return false;
            out += *value;
            ++i;
            continue;
        }

        out.push_back(in[i]);
    }

    return true;
}

bool evaluate_condition(const LammpsRunner::LMPPtr& lmp,
                        const std::string& condition,
                        bool& value)
{
    const auto tokens = split_lammps_words(condition);
    if (tokens.size() == 3 &&
        (tokens[1] == "==" || tokens[1] == "!=")) {
        double lhs = 0.0;
        double rhs = 0.0;
        if (!try_parse_double(tokens[0], lhs) || !try_parse_double(tokens[2], rhs)) {
            value = (tokens[0] == tokens[2]);
            if (tokens[1] == "!=") value = !value;
            return true;
        }
    }

    if (!lmp || !lmp->input || !lmp->input->variable) return false;

    try {
        lmp->input->one("variable __tadah_validator_cond equal " + condition);
        const int idx = lmp->input->variable->find("__tadah_validator_cond");
        if (idx < 0) return false;
        value = (lmp->input->variable->compute_equal(idx) != 0.0);
        lmp->input->one("variable __tadah_validator_cond delete");
        return true;
    } catch (...) {
        try { lmp->input->one("variable __tadah_validator_cond delete"); } catch (...) {}
        return false;
    }
}

void ensure_placeholder_zero(const LammpsRunner::LMPPtr& lmp)
{
    if (!lmp || !lmp->input || !lmp->input->variable) return;
    if (lmp->input->variable->find("__tadah_placeholder_zero") >= 0) return;
    try {
        lmp->input->one("variable __tadah_placeholder_zero equal 0.0");
    } catch (...) {
    }
}

std::string placeholder_variable_command(const std::string& name,
                                         const std::string& style)
{
    if (style == "equal") {
        return "variable " + name + " equal 0.0";
    }
    if (style == "string") {
        return "variable " + name + " string 0";
    }
    if (style == "format") {
        return "variable " + name + " format __tadah_placeholder_zero %.0f";
    }
    return {};
}
} // anonymous

namespace tadah::hpo {

bool try_parse_double(const std::string& s, double& value)
{
    try {
        std::size_t pos = 0;
        value = std::stod(s, &pos);
        return pos == s.size();
    } catch (...) {
        return false;
    }
}

/* An option token starts with '-' and is not shaped like a number
 * (strict_parse.h, is_number_shaped): a negative target, weight, bound or
 * loss parameter is a value; --outvar, -var and the other options are not.
 * A malformed number that starts with "-" and a digit or '.' ("-3x",
 * "-0x1p3") or a signed inf/nan is a value too, which the number grammar
 * rejects, rather than an option passed on to LAMMPS. */
static bool is_option_token(const std::string& s)
{
    return !s.empty() && s[0] == '-' && !is_number_shaped(s);
}

LammpsSim::LammpsSim() {}
LammpsSim::LammpsSim(std::string& cmd_,
                     const std::string& MAX_TIME,
                     double FAIL_SCORE,
                     const tadah::LossFunction& defaultLoss,
                     bool failprobe)
  : cmd(cmd_)
  , defLoss(defaultLoss)
  , maxtime(MAX_TIME)
  , failscore(FAIL_SCORE)
{
  is_failprobe = failprobe;
  invar.push_back(TADAH_LAMMPS_LIB);
  tokenise();
  parse_tokens();
  /* The root PC_SCRIPT_TIMEOUT (checked at its own line) or this line's
   * --maxtime (checked in parse_tokens). */
  maxtime_seconds = parse_script_timeout(maxtime, "PC_SCRIPT_TIMEOUT");

  // set default values if not specified by user
  auto it_screen = std::find(invar.begin(), invar.end(), "-screen");
  if (it_screen == invar.end()) {
    invar.push_back("-screen");
    invar.push_back("none");
  }
  auto it_log = std::find(invar.begin(), invar.end(), "-log");
  if (it_log == invar.end()) {
    invar.push_back("-log");
    invar.push_back("none");
  }
  /* The script timeout, as `timer timeout ${maxtime}` reads it
   * (prep_script_lines). parse_tokens() rejects a -var maxtime of the
   * user's, which would replace it unchecked. */
  invar.push_back("-var");
  invar.push_back("maxtime");
  invar.push_back(maxtime+" every 1");

  if (!script.length())
    throw std::runtime_error("LammpsSim object must have a script specified.\n");

  /* A PC_FAILPROBE sim is forbidden from declaring --varloss (it grades
   * failures; it must not shape the feasible objective) -- the >=1
   * requirement applies to ordinary PC_LAMMPS sims only. */
  if (!n_varloss.size() && !is_failprobe)
    throw std::runtime_error("LammpsSim: --varloss variable is not defined.\n");



  prep_script_lines();

  if (script_lines.empty()) {
    throw std::runtime_error("LammpsSim: script_lines is empty.\n");
  }
  // Validate PC_LAMMPS contract
  // every --varloss/--outvar must be defined in the script and must be equal-style.
  validate_requested_variables_defined_equal();


 // prepare containers
  if (n_varloss.size() != t_varloss.size() || w_varloss.size() != n_varloss.size()) {
    throw std::runtime_error("LammpsSim: n_varloss, w_varloss and v_varloss must have the same size.\n");
  }
  // v_outvar.resize(n_outvar.size());
  // prepare i_varloss
  // i_varloss is used to map n_varloss to n_outvar
  // it is a vector of indices that correspond to the indices of n_outvar
  // for (const auto& var : n_varloss) {
  //   auto it = std::find(n_outvar.begin(), n_outvar.end(), var);
  //   if (it != n_outvar.end()) {
  //     i_varloss.push_back(std::distance(n_outvar.begin(), it));
  //   } else {
  //     continue; // skip if not found
  //   }
  // }
  // check if i_varloss is the same size as n_varloss
  if (i_varloss.size() != n_varloss.size()) {
    throw std::runtime_error("LammpsSim: i_varloss must have the same size as n_varloss.\n");
  }

  if (lossFn.size()!=n_varloss.size())
    throw std::runtime_error("lossTypeId size mismatch in LammpsSim");
}
void LammpsSim::tokenise() {
  std::stringstream ss(cmd);
  std::string token;
  while (ss >> token) {
    if (token[0]=='#') break;
    tokens.push_back(token);
  }
}
void LammpsSim::parse_tokens() {

  /* A token that is not an option is a value: of the LAMMPS option before
   * it (`-var x 1 2`, `-pk omp 4 neigh no`: LAMMPS decides how many it
   * takes), or of an HPO option, which reads exactly its own values. A
   * bare token after an HPO option's values (`--maxtime 1:30 00`, a typo
   * for 1:30:00) used to reach LAMMPS as a command-line argument, and every
   * run failed on it; it is an error naming the option. */
  std::string hpo_values;       /* "<option> takes ...", after its values */
  bool lammps_values = false;   /* after a LAMMPS option or its values */
  auto hpo_option_read = [&](const std::string& takes) {
    hpo_values = takes;
    lammps_values = false;
  };
  /* The variable maxtime carries the script timeout (the constructor
   * passes -var maxtime "<spec> every 1" for `timer timeout ${maxtime}`);
   * a user's definition would replace the checked, reported value. */
  auto reject_reserved = [](const std::string& option, const std::string& name) {
    if (name == "maxtime")
      throw std::runtime_error(
          option + " maxtime: the variable maxtime is reserved for the "
          "script timeout HPO injects (timer timeout ${maxtime}); set "
          "--maxtime on this line or PC_SCRIPT_TIMEOUT instead");
  };

  size_t i = 0;
  while (i < tokens.size()) {
    if (tokens[i] == "--script") { // LAMMPS script file
      if (i + 1 < tokens.size()) {
        script = tokens[++i];
        i++;
        hpo_option_read("--script takes a single value");
      } else {
        throw std::runtime_error("Invalid number of arguments for --script");
      }
    }
    // Tadah! specific keywords
    else if (tokens[i] == "--varloss")
    {
      auto is_flag = is_option_token;

      std::size_t j=i+1;

      if(j>=tokens.size() || is_flag(tokens[j]))
        throw std::runtime_error("--varloss: missing <name>");
      std::string vname = tokens[j++];

      if(j>=tokens.size() || is_flag(tokens[j]))
        throw std::runtime_error("--varloss "+vname+": missing <target>");
      const double tgt = parse_finite(tokens[j++], "--varloss "+vname+" <target>");

      if(j>=tokens.size() || is_flag(tokens[j]))
        throw std::runtime_error("--varloss "+vname+": missing <weight>");
      /* Any finite weight; a negative one is accepted. */
      const double wgt = parse_finite(tokens[j++], "--varloss "+vname+" <weight>");

      /* optional override: <type> [<param>]. Defaults and validation of
       * the parameter come from LossFunction::resolve_params, shared with
       * the root LOSS key. ---------------------------------------------- */
      tadah::LossFunction myLoss = defLoss;
      if(j<tokens.size() && !is_flag(tokens[j])){
        const std::string tname = tokens[j];
        auto t = tadah::LossFunction::from_string(tokens[j++]);
        std::vector<double> pars;
        while(j<tokens.size() && !is_flag(tokens[j]))
          pars.push_back(parse_finite(tokens[j++],
                                      "--varloss "+vname+" "+tname+" parameter"));
        myLoss = tadah::LossFunction(t, pars);
      }

      /* store ----------------------------------------------------------- */
      if (std::find(n_outvar.begin(), n_outvar.end(), vname) == n_outvar.end()) {
        n_outvar .push_back(vname);
      }
      n_varloss.push_back(vname);
      // i_varloss.push_back(n_outvar.size()-1);
      i_varloss.push_back(std::distance(n_outvar.begin(), std::find(n_outvar.begin(), n_outvar.end(), vname)));
      t_varloss.push_back(tgt);
      w_varloss.push_back(wgt);
      lossFn   .push_back(std::move(myLoss));

      i = j;
      /* Every value up to the next option was the loss override's. */
      hpo_option_read("--varloss takes <name> <target> <weight> "
                      "[<loss> [<param>...]]");
    }
    else if (tokens[i] == "--invar") {
      if (i + 2 < tokens.size()) {
        reject_reserved("--invar", tokens[i + 1]);
        invar.push_back("-var");
        invar.push_back(tokens[++i].c_str());
        invar.push_back(tokens[++i].c_str());
        i++;
        hpo_option_read("--invar takes <name> <value>");
      } else {
        throw std::runtime_error("Invalid number of arguments for --invar");
      }
    }
    else if (tokens[i] == "--outvar") {
      if (i + 1 < tokens.size()) {
        std::string vname = tokens[++i];
        // Only add to n_outvar if not already present
        if (std::find(n_outvar.begin(), n_outvar.end(), vname) == n_outvar.end()) {
          n_outvar.push_back(vname);
        }
        i++;
        hpo_option_read("--outvar takes a single value");
      } else {
        throw std::runtime_error("Invalid number of arguments for --outvar");
      }
    }
    else if (tokens[i] == "--failscore") {
      if (i + 1 < tokens.size()) {
        failscore = parse_finite(tokens[++i], "--failscore");
        i++;
        hpo_option_read("--failscore takes a single value");
      } else {
        throw std::runtime_error("Invalid number of arguments for --failscore");
      }
    }
    else if (tokens[i] == "--maxtime") {
      if (i + 1 < tokens.size()) {
        maxtime = tokens[++i];
        (void)parse_script_timeout(maxtime, "--maxtime");
        i++;
        hpo_option_read("--maxtime takes a single value");
      } else {
        throw std::runtime_error("Invalid number of arguments for --maxtime");
      }
    }
    else if (tokens[i] == "--hotvar") {
      /* --hotvar <name> <min> <max> [<initial>] */
      auto is_flag = is_option_token;
      std::size_t j = i + 1;
      if (j+2 >= tokens.size())
        throw std::runtime_error("--hotvar requires <name> <min> <max> [<initial>]");
      HotVarDecl hv;
      hv.name = tokens[j++];
      if (hv.name.empty() || is_flag(hv.name))
        throw std::runtime_error("--hotvar: <name> must be a non-empty identifier");
      hv.min = parse_finite(tokens[j++], "--hotvar " + hv.name + " <min>");
      hv.max = parse_finite(tokens[j++], "--hotvar " + hv.name + " <max>");
      if (!(hv.max > hv.min))
        throw std::runtime_error("--hotvar " + hv.name + ": <max> must exceed <min>");
      /* Optional <initial>: a token that is not an option is it, and must
       * be a number -- it used to be passed on to LAMMPS when it was not. */
      if (j < tokens.size() && !is_flag(tokens[j]))
        hv.initial = parse_finite(tokens[j++], "--hotvar " + hv.name + " <initial>");
      hotvars.push_back(std::move(hv));
      i = j;
      hpo_option_read("--hotvar takes <name> <min> <max> [<initial>]");
    }
    else if (tokens[i] == "--failvar") {
      /* --failvar <name> [<scale>]  (graded-FAILSCORE Tier 2; see
       * INVESTIGATIONS/SMOOTH_FAILSCORE_analysis.md 4.8). Convention: the
       * LAMMPS variable <name> is a nonnegative violation measure -- 0 means
       * "this sim vouches for itself", > 0 means "failed, this badly" (in
       * natural units; <scale> maps to the dimensionless band-B measure).
       * When it reads > 0, the sim is classified band B, its outvars are
       * excluded from the fit, and scale*value grades the failure loss. */
      auto is_flag = is_option_token;
      std::size_t j = i + 1;
      if (j >= tokens.size())
        throw std::runtime_error("--failvar requires <name> [<scale>]");
      FailVarDecl fv;
      fv.name = tokens[j++];
      if (fv.name.empty() || is_flag(fv.name))
        throw std::runtime_error("--failvar: <name> must be a non-empty identifier");
      /* Optional <scale>: a token that is not an option is it, and must be
       * a number -- it used to be passed on to LAMMPS when it was not. */
      if (j < tokens.size() && !is_flag(tokens[j])) {
        fv.scale = parse_positive(tokens[j++],
                                  "--failvar " + fv.name + " <scale>");
      }
      failvars.push_back(std::move(fv));
      i = j;
      hpo_option_read("--failvar takes <name> [<scale>]");
    }
    else if (tokens[i] == "--ncpu") {
      /* Rejected, not ignored: an unrecognised option would otherwise be
       * forwarded to LAMMPS as a command-line argument below. */
      throw std::runtime_error(
          "--ncpu was removed in 1.6.0 (legacy MPI build); parallelism over "
          "PC_LAMMPS scripts is THREADS / PC_ISOLATION. Delete --ncpu from "
          "this line.");
    }
    else if (is_option_token(tokens[i])) {
      /* A LAMMPS command-line option: it and the values that follow it go
       * to LAMMPS. -var (-v) maxtime is reserved. */
      if ((tokens[i] == "-var" || tokens[i] == "-v") && i + 1 < tokens.size())
        reject_reserved(tokens[i], tokens[i + 1]);
      invar.push_back(tokens[i++]);
      hpo_values.clear();
      lammps_values = true;
    }
    else if (lammps_values) {
      invar.push_back(tokens[i++]);   /* a value of the LAMMPS option */
    }
    else if (!hpo_values.empty()) {
      throw std::runtime_error(hpo_values + " (unexpected '" + tokens[i] +
                               "')");
    }
    else {
      throw std::runtime_error(
          "unexpected '" + tokens[i] + "': a PC_LAMMPS token is an option or "
          "a value of one (HPO options start with --, LAMMPS command-line "
          "options with -)");
    }
  }
}


/* ----------------------------------------------------------------------
   remove comments while respecting quoting; update quote state flags
------------------------------------------------------------------------- */
std::string LammpsSim::stripComments(const std::string& in,
                                       bool& in_single,
                                       bool& in_double,
                                       bool& in_triple) const
{
    std::string out;
    const std::size_t n = in.size();

    for (std::size_t i = 0; i < n; ++i)
    {
        char c = in[i];

        /* --- triple quotes ------------------------------------------------ */
        if (!in_single && !in_double)
        {
            if (!in_triple && c == '"' && i + 2 < n &&
                in[i + 1] == '"' && in[i + 2] == '"')
            {
                in_triple = true;
                out.append("\"\"\"");
                i += 2;
                continue;
            }
            if (in_triple && c == '"' && i + 2 < n &&
                in[i + 1] == '"' && in[i + 2] == '"')
            {
                in_triple = false;
                out.append("\"\"\"");
                i += 2;
                continue;
            }
        }

        /* --- inside triple quotes: copy verbatim ------------------------- */
        if (in_triple)
        {
            out.push_back(c);
            continue;
        }

        /* --- single / double --------------------------------------------- */
        if (!in_double && c == '\'' )
        {
            in_single = !in_single;
            out.push_back(c);
            continue;
        }
        if (!in_single && c == '"' )
        {
            in_double = !in_double;
            out.push_back(c);
            continue;
        }

        /* --- comment outside quotes -------------------------------------- */
        if (!in_single && !in_double && c == '#')
        {
            break;                      // discard rest of physical line
        }

        out.push_back(c);
    }
    return out;
}

/* ----------------------------------------------------------------------
   read the complete contents of 'filename' into a std::string
   throws std::runtime_error if the file cannot be opened or read
------------------------------------------------------------------------- */
inline std::string read_entire_file(const std::string& filename)
{
    std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary);
    if (!file)
        throw std::runtime_error("read_entire_file: cannot open \"" + filename + "\"");

    std::ostringstream buffer;
    buffer << file.rdbuf();            // read *everything*
    if (!file.good() && !file.eof())   // eof is fine; any other error is not
        throw std::runtime_error("read_entire_file: error while reading \"" + filename + "\"");

    return buffer.str();
}
/* ----------------------------------------------------------------------
   create script_lines from raw script
------------------------------------------------------------------------- */
void LammpsSim::prep_script_lines()
{
  std::string buffer = expand_includes(script);
  script_lines.clear();

  std::istringstream iss(buffer);
  std::string        line;            // current physical line
  std::string        cmd;             // command under construction

  /* quote states that may span several physical lines */
  bool in_single  = false;
  bool in_double  = false;
  bool in_triple  = false;

  while (std::getline(iss, line))
  {
    /* drop trailing CR in CRLF files */
    if (!line.empty() && line.back() == '\r') line.pop_back();

    /* 1. comment stripping (updates quote state flags) */
    std::string clean =
      stripComments(line, in_single, in_double, in_triple);

    /* 2. decide whether this line ends with an ampersand continuation  */
    bool continuation = false;


        if (!in_triple)
        {
            std::size_t last_print = clean.find_last_not_of(" \t");
            if (last_print != std::string::npos && clean[last_print] == '&')
            {
                continuation = true;

                // erase everything from the '&' to the end of line
                clean.erase(last_print);

                // make sure the part we keep ends with *exactly* one space
                // (so words do not stick together after concatenation)
                if (clean.empty() || !std::isspace(static_cast<unsigned char>(clean.back())))
                    clean.push_back(' ');
                else
                {
                    // collapse possible multiple blanks to one
                    while (clean.size() >= 2 &&
                           std::isspace(static_cast<unsigned char>(clean[clean.size()-2])))
                        clean.pop_back();
                    clean.back() = ' ';
                }
            }
        }

    /* 3. build up the logical command string ------------------------- */
    if (cmd.empty())
      cmd = std::move(clean);
    else
      cmd += clean;            // keep leading blanks of next line

    /* keep newline inside triple quotes so that user-supplied text   */
    /* keeps its original layout                                       */
    if (in_triple)
    {
      cmd.push_back('\n');
      continue;                 // command is definitely not finished
    }

    if (continuation)
    {
      /* no newline -> exactly LAMMPS semantics */
      continue;                 // wait for next physical line
    }

    /* 4. command finished ------------------------------------------- */
    cmd.push_back('\n');

    /* reject blank lines (may happen after stripping) */
    if (cmd.find_first_not_of(" \t\n") != std::string::npos)
      script_lines.push_back(cmd);

    cmd.clear();
  }

  /* file ended but last command not yet pushed */
  if (!cmd.empty() &&
      cmd.find_first_not_of(" \t\n") != std::string::npos)
    script_lines.push_back(cmd);

  /* Per-sim wall-clock enforcement. LAMMPS's native `timer timeout` is the
   * only mechanism that stops a running minimize cleanly (checked every
   * minimizer iteration; ends the run with TIMEOUT status, not an error),
   * and `clear` rebuilds the Timer, so the command must be re-issued after
   * every clear. ${maxtime} expands to "<seconds> every 1" (the -var
   * registered in the ctor). Enforcement is opt-in: MAXTIME/--maxtime
   * default to "off"; a script that issues its own `timer` command keeps
   * full control and gets no injection.
   *
   * Scripts run through Input::file(), which never checks the timer
   * BETWEEN commands, and `clear` destroys a FIRED Timer -- so on its own
   * the re-armed timer would let a timed-out section roll silently into
   * the next section with a fresh budget (and the end-of-script
   * is_timeout() check would see a healthy timer: a timed-out sim
   * misreported as OK). The `timeremain` guard injected BEFORE each
   * mid-script clear aborts the run at the section boundary instead: after
   * a fire, `$(timeremain)` is exactly 0.0 and the sentinel executes as an
   * unknown command, failing Input::file() with a message the runner
   * classifies as a soft-timeout stop. This preserves the per-section
   * timeout semantics and cost profile of the old per-line executor. The
   * final section needs no guard: with no clear after it, the fired timer
   * survives to the end-of-script is_timeout() check.
   *
   * Only a limit in seconds is injected: "off" and "unlimited" (LAMMPS'
   * two spellings of no limit; maxtime_seconds is empty) inject nothing.
   * An injected `timer timeout unlimited` leaves $(timeremain) at 0.0, so
   * the guard aborted every multi-section script as timed out. */
  if (maxtime_seconds && !script_lines.empty()) {
    bool user_timer = false;
    for (const auto& l : script_lines) {
      std::istringstream ls(l);
      std::string tok;
      if ((ls >> tok) && to_lower_ascii(tok) == "timer") { user_timer = true; break; }
    }
    if (!user_timer) {
      const std::string timer_line = "timer timeout ${maxtime}\n";
      const std::string guard_line =
          std::string("if \"$(timeremain) == 0.0\" then \"") +
          tadah::hpo::PC_TIMEOUT_ABORT_SENTINEL + "\"\n";
      auto first_tok_is_clear = [](const std::string& l) {
        std::istringstream ls(l);
        std::string tok;
        return (ls >> tok) && to_lower_ascii(tok) == "clear";
      };
      auto first_tok_creates_box = [](const std::string& l) {
        std::istringstream ls(l);
        std::string tok;
        if (!(ls >> tok)) return false;
        tok = to_lower_ascii(tok);
        return tok == "create_box" || tok == "read_data" ||
               tok == "read_restart";
      };
      std::vector<std::string> out;
      out.reserve(2 * script_lines.size() + 4);
      bool armed = false;   /* a timer line precedes the current position */
      /* `$(timeremain)` cannot be evaluated before a simulation box
       * exists, so the guard is placed only after sections that created
       * one (create_box/read_data/read_restart since the last clear).
       * This is exact, not a heuristic: a boxless section has no
       * run/minimize, so its timer can never fire and the guard would be
       * a no-op there anyway. Scripts opening with variable-setup lines
       * followed by a defensive `clear` (vac_bcc_P, eos_deriv_v2, ...)
       * would otherwise die on the guard itself. */
      bool section_has_box = false;
      if (!first_tok_is_clear(script_lines.front())) {
        out.push_back(timer_line);
        armed = true;
      }
      for (const auto& l : script_lines) {
        if (first_tok_is_clear(l)) {
          if (armed && section_has_box) out.push_back(guard_line);
          out.push_back(l);
          out.push_back(timer_line);
          armed = true;
          section_has_box = false;
        } else {
          if (first_tok_creates_box(l)) section_has_box = true;
          out.push_back(l);
        }
      }
      script_lines.swap(out);
    }
  }
}
const tadah::hpo::ScriptFile& LammpsSim::script_file() const
{
    /* File-local mutex (not a member) so LammpsSim stays copyable/movable
     * for HPOTargets::LS vector growth. Contention is negligible: one
     * render per sim per process lifetime. */
    static std::mutex script_file_mutex;
    std::lock_guard<std::mutex> lock(script_file_mutex);
    if (!script_file_)
        script_file_ = std::make_shared<tadah::hpo::ScriptFile>(script_lines);
    return *script_file_;
}

std::unordered_map<std::string, LammpsSim::VarDef>
LammpsSim::index_defined_variables() const
{
    std::unordered_map<std::string, VarDef> defs;

    for (std::size_t li = 0; li < script_lines.size(); ++li) {
        std::istringstream iss(script_lines[li]);

        std::string cmd, name, style;
        if (!(iss >> cmd)) continue;
        cmd = to_lower_ascii(cmd);

        if (cmd != "variable") continue;

        if (!(iss >> name))  continue;   // malformed; let LAMMPS fail later if run
        if (!(iss >> style)) continue;   // malformed; same

        style = to_lower_ascii(style);

        // Track deletions too (LAMMPS: "variable X delete")
        if (style == "delete") {
            defs.erase(name);
        } else {
            defs[name] = VarDef{style, li + 1}; // 1-based logical command index
        }
    }

    LammpsRunner runner;
    /* Serial builds: comm 0 (LAMMPS STUBS). TADAH_HPO_MPI builds: a
     * literal 0 would be a NULL MPI_Comm -- use MPI_COMM_SELF. */
    auto lmp = runner.create_instance(hpo_lammps_comm(), invar);
    if (!lmp || !lmp->input || !lmp->input->variable) return defs;

    std::unordered_map<std::string, std::size_t> labels;
    for (std::size_t li = 0; li < script_lines.size(); ++li) {
        const auto tokens = split_lammps_words(trim_ascii(script_lines[li]));
        if (tokens.size() >= 2 && to_lower_ascii(tokens[0]) == "label") {
            labels[tokens[1]] = li;
        }
    }

    auto define_variable = [&](const std::string& resolved_line, std::size_t logical_cmd) {
        const auto tokens = split_lammps_words(trim_ascii(resolved_line));
        if (tokens.size() < 3 || to_lower_ascii(tokens[0]) != "variable") return;

        const std::string name = tokens[1];
        const std::string style = to_lower_ascii(tokens[2]);

        if (style == "delete") {
            try { lmp->input->one(resolved_line); } catch (...) {}
            defs.erase(name);
            return;
        }

        bool defined_in_lammps = false;
        if (resolved_line.find("$(") == std::string::npos) {
            try {
                lmp->input->one(resolved_line);
                defined_in_lammps = true;
            } catch (...) {
                defined_in_lammps = false;
            }
        }

        if (!defined_in_lammps) {
            ensure_placeholder_zero(lmp);
            const std::string placeholder = placeholder_variable_command(name, style);
            if (!placeholder.empty()) {
                try {
                    lmp->input->one(placeholder);
                    defined_in_lammps = true;
                } catch (...) {
                    defined_in_lammps = false;
                }
            }
        }

        defs[name] = VarDef{style, logical_cmd};
    };

    std::size_t pc = 0;
    std::size_t steps = 0;
    const std::size_t max_steps = std::max<std::size_t>(script_lines.size() * 8, 64);

    while (pc < script_lines.size() && steps++ < max_steps) {
        const std::string raw = trim_ascii(script_lines[pc]);
        const auto raw_tokens = split_lammps_words(raw);
        if (raw_tokens.empty()) {
            ++pc;
            continue;
        }

        const std::string cmd = to_lower_ascii(raw_tokens[0]);

        if (cmd == "label") {
            ++pc;
            continue;
        }

        if (cmd == "clear") {
            try { lmp->input->one("clear"); } catch (...) {}
            ++pc;
            continue;
        }

        if (cmd == "jump") {
            if (raw_tokens.size() >= 3 && raw_tokens[1] == "SELF") {
                auto it = labels.find(raw_tokens[2]);
                if (it != labels.end()) {
                    pc = it->second + 1;
                    continue;
                }
            }
            ++pc;
            continue;
        }

        if (cmd == "if") {
            std::string substituted;
            if (substitute_known_variables(lmp, raw, substituted)) {
                const auto tokens = split_lammps_words(substituted);
                if (tokens.size() >= 4 && to_lower_ascii(tokens[2]) == "then") {
                    bool cond_value = false;
                    if (evaluate_condition(lmp, tokens[1], cond_value)) {
                        std::string chosen;
                        if (cond_value) chosen = tokens[3];
                        else if (tokens.size() >= 6 && to_lower_ascii(tokens[4]) == "else") chosen = tokens[5];

                        if (!chosen.empty()) {
                            const auto chosen_tokens = split_lammps_words(chosen);
                            if (!chosen_tokens.empty()) {
                                const std::string chosen_cmd = to_lower_ascii(chosen_tokens[0]);
                                if (chosen_cmd == "jump" && chosen_tokens.size() >= 3 && chosen_tokens[1] == "SELF") {
                                    auto it = labels.find(chosen_tokens[2]);
                                    if (it != labels.end()) {
                                        pc = it->second + 1;
                                        continue;
                                    }
                                }
                                if (chosen_cmd == "variable") {
                                    define_variable(chosen, pc + 1);
                                } else if (chosen_cmd == "clear") {
                                    try { lmp->input->one("clear"); } catch (...) {}
                                }
                            }
                        }
                    }
                }
            }

            ++pc;
            continue;
        }

        if (cmd == "variable") {
            std::string substituted;
            if (substitute_known_variables(lmp, raw, substituted)) {
                define_variable(substituted, pc + 1);
            }
            ++pc;
            continue;
        }

        ++pc;
    }

    return defs;
}

void LammpsSim::validate_requested_variables_defined_equal() const
{
    const auto defs = index_defined_variables();

    // Make membership check for varloss to improve error messages for outvar-only requests
    std::unordered_set<std::string> varloss_set(n_varloss.begin(), n_varloss.end());

    struct Problem {
        std::string name;
        std::string origin;   // "--varloss" or "--outvar"
        std::string kind;     // "missing" or "wrongstyle"
        std::string style;    // for wrongstyle
        std::size_t where;    // logical command index where it was defined (if any)
    };

    std::vector<Problem> problems;

    auto require_equal = [&](const std::string &v, const std::string &origin)
    {
        auto it = defs.find(v);
        if (it == defs.end()) {
            problems.push_back(Problem{v, origin, "missing", "", 0});
            return;
        }
        if (it->second.style != "equal") {
            problems.push_back(Problem{v, origin, "wrongstyle", it->second.style, it->second.logical_cmd});
            return;
        }
    };

    // Docs: --varloss vars must be defined & equal-style
    for (const auto &v : n_varloss)
        require_equal(v, "--varloss");

    // Docs: --outvar vars must be defined & equal-style as well (you requested this requirement)
    for (const auto &v : n_outvar) {
        if (varloss_set.find(v) != varloss_set.end()) continue; // already checked as varloss
        require_equal(v, "--outvar");
    }

    // Graded-FAILSCORE: --failvar names get the same static guarantee. A
    // typo'd failvar would otherwise read back empty at runtime, parse as
    // "no violation" and silently disable failure detection for that sim.
    {
        std::unordered_set<std::string> outvar_set(n_outvar.begin(), n_outvar.end());
        for (const auto &fv : failvars) {
            if (varloss_set.count(fv.name) || outvar_set.count(fv.name)) continue;
            require_equal(fv.name, "--failvar");
        }
    }

    if (!problems.empty()) {
        std::ostringstream oss;
        oss << "PC_LAMMPS validation failed for script '" << script << "'.\n"
            << "Tadah requires every --varloss/--outvar/--failvar variable to be defined in the LAMMPS script (or included files)\n"
            << "as an *equal-style* variable:\n"
            << "  variable <name> equal ...\n\n";

        oss << "Problems found:\n";
        for (const auto &p : problems) {
            if (p.kind == "missing") {
                oss << "  - " << p.name << " requested by " << p.origin << " but not defined\n";
            } else { // wrongstyle
                oss << "  - " << p.name << " requested by " << p.origin
                    << " but defined as style '" << p.style << "'"
                    << " (logical command #" << p.where << " in expanded script)\n";
            }
        }

        throw std::runtime_error(oss.str());
    }
}


namespace fs = std::filesystem;

// -----------------------------------------------------------------
//  Recursive worker that actually performs the include expansion
// -----------------------------------------------------------------
std::string LammpsSim::expand_includes_rec(const fs::path &file,
                                           std::unordered_set<std::string> &stack)
{
    const fs::path abs = fs::absolute(file);

    if (!fs::exists(abs))
        throw std::runtime_error("LAMMPS include file not found: " + abs.string());

    /* Detect circular includes ("A includes B includes A ...")             */
    if (!stack.insert(abs.string()).second)
        throw std::runtime_error("Circular include detected at: " + abs.string());

    std::string          text   = read_entire_file(abs);
    std::ostringstream   out;
    std::istringstream   iss(text);
    std::string          line;

    while (std::getline(iss, line))
  {
    /* strip trailing CR put there by Windows text files */
    if (!line.empty() && line.back() == '\r') line.pop_back();

    /* find first non-blank character */
    const std::size_t first = line.find_first_not_of(" \t");
    if (first == std::string::npos)        /* completely blank line */
    {
      out << '\n';
      continue;
    }

    /* test for   include  <filename>   */
    // if (line.compare(first, 7, "include") == 0 &&
    //     std::isspace(static_cast<unsigned char>(line[first + 7])))
    // {
    //     std::size_t fname_start = line.find_first_not_of(" \t", first + 7);
    //     if (fname_start == std::string::npos)
    //         throw std::runtime_error("Missing file name in include statement ("
    //                                  + abs.string() + ')');
    //
    //     std::size_t fname_end = line.find_first_of(" \t#\r\n", fname_start);
    //     std::string inc_name  = line.substr(fname_start,
    //                                         fname_end == std::string::npos
    //                                             ? std::string::npos
    //                                             : fname_end - fname_start);
    //
    //     fs::path inc_path = abs.parent_path() / inc_name;
    //     out << expand_includes_rec(inc_path, stack);   // recursive step
    //     continue;                                      // nothing of the original line survives
    // }
    /* test for   include  <filename>   (case-insensitive) */
    if (line.size() >= first + 7) {
      std::string head = line.substr(first, 7);
      head = to_lower_ascii(head);

      if (head == "include" &&
        (line.size() > first + 7) &&
        std::isspace(static_cast<unsigned char>(line[first + 7])))
      {
        std::size_t fname_start = line.find_first_not_of(" \t", first + 7);
        if (fname_start == std::string::npos)
          throw std::runtime_error("Missing file name in include statement ("
                                   + abs.string() + ')');

        std::size_t fname_end = line.find_first_of(" \t#\r\n", fname_start);
        std::string inc_name  = line.substr(fname_start,
                                            fname_end == std::string::npos
                                            ? std::string::npos
                                            : fname_end - fname_start);

        fs::path inc_path = abs.parent_path() / inc_name;
        out << expand_includes_rec(inc_path, stack);   // recursive step
        continue;                                      // nothing of the original line survives
      }
    }


    /* ordinary line - keep verbatim */
    out << line << '\n';
  }

    /* done with this file - pop it from the recursion guard */
    stack.erase(abs.string());

    return out.str();
}

// -----------------------------------------------------------------
//  Public one-liner wrapper that callers can use
// -----------------------------------------------------------------
std::string LammpsSim::expand_includes(const fs::path &file)
{
    std::unordered_set<std::string> recursion_guard;
    return expand_includes_rec(file, recursion_guard);
}


} // namespace tadah::hpo
