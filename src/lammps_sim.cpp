#include <tadah/hpo/lammps_sim.h>
#include <iostream>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

LammpsSim::LammpsSim() {}
LammpsSim::LammpsSim(std::string &cmd, std::string MAX_TIME, double FAIL_SCORE):
  cmd(cmd), maxtime(MAX_TIME), failscore(FAIL_SCORE)
{
  invar.push_back(TADAH_LAMMPS_LIB); // Ensure this is a string literal
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

  invar.push_back("-var");
  invar.push_back("maxtime");
  invar.push_back(maxtime+" every 1");

  if (!script.length())
    throw std::runtime_error("LammpsSim object must have a script specified.\n");

  if (!n_varloss.size())
    throw std::runtime_error("LammpsSim: --varloss variable is not defined.\n");


  v_varloss.resize(n_varloss.size());
  v_outvar.resize(outvar.size());

  prep_script_lines();

  if (script_lines.empty()) {
    throw std::runtime_error("LammpsSim: script_lines is empty.\n");
  }

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
    else if (tokens[i] == "--varloss") {
      if (i + 2 < tokens.size()) {
        n_varloss.push_back(tokens[++i]);
        w_varloss.push_back(std::stod(tokens[++i]));
        i++;
      } else {
        throw std::runtime_error("Invalid number of arguments for --varloss");
      }
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
        outvar.push_back(tokens[++i]);
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
  std::string buffer = read_entire_file(script);
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

