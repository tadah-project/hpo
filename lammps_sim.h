#ifndef LAMMPS_SIM_H
#define LAMMPS_SIM_H

#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <mpi.h>

/**
 * @class LammpsSim
 * 
 * @brief Represents a LAMMPS simulation and provides functionality
 * for serialisation and MPI communication.
 */
class LammpsSim {
  private:
    std::string cmd; /**< Command line configuration */
    std::vector<std::string> tokens; /**< Tokenised command line */
    std::vector<std::string> outvar; /**< LAMMPS output variables */

  public:
    std::vector<const char*> invar; /**< LAMMPS input variables */
    std::string script; /**< Name of the LAMMPS script file */
    std::vector<std::string> varloss; /**< Variables related to the loss function */

    int ncpu = 1; /**< Number of CPUs to be used */
    double weight; /**< Weight parameter for simulation */
    int lossfunc; /**< Identifier for the loss function used */
    std::string maxtime; /**< Maximum allowed time for simulation execution */
    double failscore; /**< Score given when a simulation fails */

    LammpsSim() {};

    /**
     * @brief Constructor for LammpsSim class.
     * 
     * @param cmd The command line configuration.
     * @param LOSS_FUNC Default loss function identifier.
     * @param MAX_TIME Maximum time allowed for the simulation.
     * @param FAIL_SCORE Default fail score value.
     */
    LammpsSim(std::string &cmd, int LOSS_FUNC, std::string MAX_TIME, double FAIL_SCORE):
      cmd(cmd), lossfunc(LOSS_FUNC), maxtime(MAX_TIME), failscore(FAIL_SCORE)
    {
      tokenise();
      parse_tokens();

      if (std::find(invar.begin(), invar.end(), "-log") == invar.end() || 
          std::find(invar.begin(), invar.end(), "-l") == invar.end()) {
        invar.push_back("-log");
        invar.push_back("none");
      }
      if (std::find(invar.begin(), invar.end(), "-screen") == invar.end() || 
          std::find(invar.begin(), invar.end(), "-sc") == invar.end()) {
        invar.push_back("-screen");
        invar.push_back("none");
      }

      if (!script.length())
        throw std::runtime_error("LammpsSim object must have a script specified!\n");
    }

    /**
     * @brief Tokenises the command line configuration.
     */
    void tokenise() {
      std::stringstream ss(cmd);
      std::string token;
      while (ss >> token) {
        tokens.push_back(token);
      }
    }

    /**
     * @brief Parses the tokenised command line to set class parameters.
     */
    void parse_tokens() {
      for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == "-s" || tokens[i] == "--script") {
          if (i + 1 < tokens.size()) {
            script = tokens[i+1];
          }
          i++;
        }
        else if (tokens[i] == "-v" || tokens[i] == "--varloss") {
          if (i + 1 < tokens.size()) {
            varloss.push_back(tokens[i+1]);
          }
          i++;
        }
        else if (tokens[i] == "-i" || tokens[i] == "--invar") {
          if (i + 2 < tokens.size()) {
            invar.push_back("-var");
            invar.push_back(tokens[i+1].c_str());
            invar.push_back(tokens[i+2].c_str());
          }
          i+=2; // Skip the next 2 tokens
        }
        else if (tokens[i] == "-o" || tokens[i] == "--outvar") {
          if (i + 1 < tokens.size()) {
            outvar.push_back(tokens[i+1]);
          }
          i++;
        }
        else if (tokens[i] == "-w" || tokens[i] == "--weight") {
          if (i + 1 < tokens.size()) {
            weight = std::stod(tokens[i+1]);
          }
          i++;
        }
        else if (tokens[i] == "-f" || tokens[i] == "--failscore") {
          if (i + 1 < tokens.size()) {
            failscore = std::stod(tokens[i+1]);
          }
          i++;
        }
        else if (tokens[i] == "-l" || tokens[i] == "--lossfunc") {
          if (i + 1 < tokens.size()) {
            lossfunc = std::stoi(tokens[i+1]);
          }
          i++;
        }
        else if (tokens[i] == "-m" || tokens[i] == "--maxtime") {
          if (i + 1 < tokens.size()) {
            int tsec = std::stoi(tokens[i+1]);
            int s,m,h;
            h=tsec/3600;
            m=(tsec-h*3600)/60;
            s=tsec-h*3600-m*60;
            maxtime=std::to_string(h) + ":" + std::to_string(m)
              + ":" + std::to_string(s) + " every 1";
          }
          i++;
        }
        else if (tokens[i] == "-n" || tokens[i] == "--ncpu") {
          if (i + 1 < tokens.size()) {
            ncpu = std::stoi(tokens[i+1]);
          }
          i++;
        }
        else {
          throw std::runtime_error("Unknown LAMMPS config:\n" + cmd);
        }
      }
    }

    /**
     * @brief Compares two LammpsSim objects based on the number of CPUs.
     * 
     * @param lhs Left-hand side LammpsSim object.
     * @param rhs Right-hand side LammpsSim object.
     * @return True if lhs uses more CPUs than rhs, otherwise False.
     */
    static bool greater(const LammpsSim& lhs, const LammpsSim& rhs) {
      return lhs.ncpu > rhs.ncpu;
    }

    /**
     * @brief Serialises the LammpsSim object for MPI communication.
     * 
     * @param buffer Vector to store serialised data.
     */
    void serialise(std::vector<char>& buffer) const {
      serialise_string(cmd, buffer);
      serialise_string(script, buffer);
      serialise_string(maxtime, buffer);

      serialise_vector_string(varloss, buffer);
      serialise_vector_string(outvar, buffer);

      buffer.insert(buffer.end(), reinterpret_cast<const char*>(&ncpu), reinterpret_cast<const char*>(&ncpu) + sizeof(ncpu));
      buffer.insert(buffer.end(), reinterpret_cast<const char*>(&weight), reinterpret_cast<const char*>(&weight) + sizeof(weight));
      buffer.insert(buffer.end(), reinterpret_cast<const char*>(&lossfunc), reinterpret_cast<const char*>(&lossfunc) + sizeof(lossfunc));
      buffer.insert(buffer.end(), reinterpret_cast<const char*>(&failscore), reinterpret_cast<const char*>(&failscore) + sizeof(failscore));
    }

    /**
     * @brief Serialises a string for MPI communication.
     * 
     * @param str String to serialise.
     * @param buffer Vector to store serialised data.
     */
    static void serialise_string(const std::string& str, std::vector<char>& buffer) {
      size_t len = str.size();
      buffer.insert(buffer.end(), reinterpret_cast<const char*>(&len), reinterpret_cast<const char*>(&len) + sizeof(len));
      buffer.insert(buffer.end(), str.begin(), str.end());
    }

    /**
     * @brief Serialises a vector of strings for MPI communication.
     * 
     * @param vec Vector of strings to serialise.
     * @param buffer Vector to store serialised data.
     */
    static void serialise_vector_string(const std::vector<std::string>& vec, std::vector<char>& buffer) {
      size_t len = vec.size();
      buffer.insert(buffer.end(), reinterpret_cast<const char*>(&len), reinterpret_cast<const char*>(&len) + sizeof(len));
      for (const auto& str : vec) {
        serialise_string(str, buffer);
      }
    }

    /**
     * @brief Deserialises the LammpsSim object from MPI communication buffer.
     * 
     * @param buffer Vector containing serialised data.
     * @param offset Current position in the buffer.
     */
    void deserialise(const std::vector<char>& buffer, size_t& offset) {
      cmd = deserialise_string(buffer, offset);
      script = deserialise_string(buffer, offset);
      maxtime = deserialise_string(buffer, offset);

      varloss = deserialise_vector_string(buffer, offset);
      outvar = deserialise_vector_string(buffer, offset);

      ncpu = *reinterpret_cast<const int*>(&buffer[offset]);
      offset += sizeof(ncpu);
      weight = *reinterpret_cast<const double*>(&buffer[offset]);
      offset += sizeof(weight);
      lossfunc = *reinterpret_cast<const int*>(&buffer[offset]);
      offset += sizeof(lossfunc);
      failscore = *reinterpret_cast<const double*>(&buffer[offset]);
      offset += sizeof(failscore);
    }

    /**
     * @brief Deserialises a string from MPI communication buffer.
     * 
     * @param buffer Vector containing serialised data.
     * @param offset Current position in the buffer.
     * @return Deserialised string.
     */
    static std::string deserialise_string(const std::vector<char>& buffer, size_t& offset) {
      size_t len = *reinterpret_cast<const size_t*>(&buffer[offset]);
      offset += sizeof(len);
      std::string str(&buffer[offset], len);
      offset += len;
      return str;
    }

    /**
     * @brief Deserialises a vector of strings from MPI communication buffer.
     * 
     * @param buffer Vector containing serialised data.
     * @param offset Current position in the buffer.
     * @return Deserialised vector of strings.
     */
    static std::vector<std::string> deserialise_vector_string(const std::vector<char>& buffer, size_t& offset) {
      size_t len = *reinterpret_cast<const size_t*>(&buffer[offset]);
      offset += sizeof(len);
      std::vector<std::string> vec(len);
      for (size_t i = 0; i < len; ++i) {
        vec[i] = deserialise_string(buffer, offset);
      }
      return vec;
    }

    /**
     * @brief Serialises a vector of LammpsSim objects for MPI communication.
     * 
     * @param sims Vector of LammpsSim objects.
     * @param buffer Vector to store serialised data.
     */
    static void serialise_vector_LammpsSim(const std::vector<LammpsSim>& sims, std::vector<char>& buffer) {
      size_t len = sims.size();
      buffer.insert(buffer.end(), reinterpret_cast<const char*>(&len), reinterpret_cast<const char*>(&len) + sizeof(len));
      for (const auto& sim : sims) {
        sim.serialise(buffer);
      }
    }

    /**
     * @brief Deserialises a vector of LammpsSim objects from MPI communication buffer.
     * 
     * @param buffer Vector containing serialised data.
     * @param sims Vector to store deserialised LammpsSim objects.
     */
    static void deserialise_vector_LammpsSim(const std::vector<char>& buffer, std::vector<LammpsSim>& sims) {
      size_t offset = 0;
      size_t len = *reinterpret_cast<const size_t*>(&buffer[offset]);
      offset += sizeof(len);
      sims.resize(len);
      for (size_t i = 0; i < len; ++i) {
        sims[i].deserialise(buffer, offset);
      }
    }
};
#endif // LAMMPS_SIM_H

