#include <tadah/core/logger.h>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/hpo_targets.h>
#include <iostream>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <unordered_set>

namespace {


/* provide an empty vector for loss types that need no parameters          */
/* Huber[0]=δ , Tukey[0]=c                                                 */
inline std::vector<double> defaultParams(int id,double dHuber,double tC)
{
    switch(id){
      case 4: return {dHuber};
      case 6: return {tC};
      default: return {};
    }
}

inline std::string to_lower_ascii(std::string s)
{
    for (char &c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
} // anonymous

namespace tadah::hpo {

LammpsSim::LammpsSim() {}
LammpsSim::LammpsSim(std::string& cmd_,
                     const std::string& MAX_TIME,
                     double FAIL_SCORE,
                     const tadah::LossFunction& defaultLoss)
  : cmd(cmd_)
  , defLoss(defaultLoss)
  , maxtime(MAX_TIME)
  , failscore(FAIL_SCORE)
{
  invar.push_back(TADAH_LAMMPS_LIB);
  tokenise();
  parse_tokens();

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
  auto it_maxtime = std::find(invar.begin(), invar.end(), "maxtime");
  if (it_maxtime == invar.end()) {
    invar.push_back("-var");
    invar.push_back("maxtime");
    invar.push_back(maxtime+" every 1");
  }

  if (!script.length())
    throw std::runtime_error("LammpsSim object must have a script specified.\n");

  if (!n_varloss.size())
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

  size_t i = 0;
  while (i < tokens.size()) {
    if (tokens[i] == "--script") { // LAMMPS script file
      if (i + 1 < tokens.size()) {
        script = tokens[++i];
        i++;
      } else {
        throw std::runtime_error("Invalid number of arguments for --script");
      }
    }
    // Tadah! specific keywords
    else if (tokens[i] == "--varloss")
    {
      auto is_flag = [](const std::string& s){ return s.rfind("-",0)==0; };

      std::size_t j=i+1;

      if(j>=tokens.size() || is_flag(tokens[j]))
        throw std::runtime_error("--varloss: missing <name>");
      std::string vname = tokens[j++];

      if(j>=tokens.size() || is_flag(tokens[j]))
        throw std::runtime_error("--varloss "+vname+": missing <target>");
      double tgt = std::stod(tokens[j++]);

      if(j>=tokens.size() || is_flag(tokens[j]))
        throw std::runtime_error("--varloss "+vname+": missing <weight>");
      double wgt = std::stod(tokens[j++]);

      /* optional override ---------------------------------------------- */
      tadah::LossFunction myLoss = defLoss;
      if(j<tokens.size() && !is_flag(tokens[j])){
        auto t = tadah::LossFunction::from_string(tokens[j++]);
        std::vector<double> pars;
        while(j<tokens.size() && !is_flag(tokens[j]))
          pars.push_back(std::stod(tokens[j++]));
        myLoss = tadah::LossFunction(t, pars);
      }

      /* store ----------------------------------------------------------- */
      n_outvar .push_back(vname);
      n_varloss.push_back(vname);
      i_varloss.push_back(n_outvar.size()-1);
      t_varloss.push_back(tgt);
      w_varloss.push_back(wgt);
      lossFn   .push_back(std::move(myLoss));

      i = j;
    }
    else if (tokens[i] == "--invar") {
      if (i + 2 < tokens.size()) {
        invar.push_back("-var");
        invar.push_back(tokens[++i].c_str());
        invar.push_back(tokens[++i].c_str());
        i++;
      } else {
        throw std::runtime_error("Invalid number of arguments for --invar");
      }
    }
    else if (tokens[i] == "--outvar") {
      if (i + 1 < tokens.size()) {
        n_outvar.push_back(tokens[++i]);
        i++;
      } else {
        throw std::runtime_error("Invalid number of arguments for --outvar");
      }
    }
    else if (tokens[i] == "--failscore") {
      if (i + 1 < tokens.size()) {
        failscore = std::stod(tokens[++i]);
        i++;
      } else {
        throw std::runtime_error("Invalid number of arguments for --failscore");
      }
    }
    else if (tokens[i] == "--maxtime") {
      if (i + 1 < tokens.size()) {
        maxtime = tokens[++i];
        i++;
      } else {
        throw std::runtime_error("Invalid number of arguments for --maxtime");
      }
    }
    else if (tokens[i] == "--ncpu") {
      if (i + 1 < tokens.size()) {
        ncpu = std::stoi(tokens[++i]);
        i++;
      } else {
        throw std::runtime_error("Invalid number of arguments for -ncpu");
      }
    }
    else {
      // Otherwise we just push it to invar, as it might be a LAMMPS command line argument
      invar.push_back(tokens[i++]);
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

                // erase everything from the ‘&’ to the end of line
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

    if (!problems.empty()) {
        std::ostringstream oss;
        oss << "PC_LAMMPS validation failed for script '" << script << "'.\n"
            << "Tadah requires every --varloss/--outvar variable to be defined in the LAMMPS script (or included files)\n"
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

    /* Detect circular includes (“A includes B includes A …”)             */
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


    /* ordinary line – keep verbatim */
    out << line << '\n';
  }

    /* done with this file – pop it from the recursion guard */
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


