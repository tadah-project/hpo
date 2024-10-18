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
 * @brief Represents a LAMMPS simulation.
 *
 * This class encapsulates the setup and execution information required 
 * to run a LAMMPS simulation using Tadah! It also holds parameters 
 * specific to the simulation for data processing with HPO (Hyperparameter Optimization).
 */
class LammpsSim {
  private:
    std::string cmd; ///< Command line extracted from the HPO configuration file.
    std::vector<std::string> tokens; ///< Tokenized command line.
    std::vector<std::string> outvar; ///< LAMMPS output variables.
    // TODO: Define MPI_Comm if needed for MPI communication.

  public:
    std::vector<const char*> invar; ///< LAMMPS input variables.

    /**
     * @brief Name of the LAMMPS script file.
     *
     * The script file must be specified for the simulation to run.
     */
    std::string script;

    /**
     * @brief Variables related to the loss function.
     *
     * At least one variable must be defined to work with HPO.
     */
    std::vector<std::string> varloss;

    int ncpu = 1; ///< Number of CPUs to be used, applicable in MPI environments only.
    double weight; ///< Weight parameter for simulation.
    int lossfunc; ///< Identifier for the loss function used.
    std::string maxtime; ///< Maximum allowed time for simulation execution.
    double failscore; ///< Score given when a simulation fails.

    /**
     * @brief Constructor for LammpsSim class.
     * 
     * Initializes simulation parameters and tokenizes the command.
     * 
     * @param cmd Command line configuration.
     * @param LOSS_FUNC Default loss function identifier.
     * @param MAX_TIME Maximum time allowed for the simulation.
     * @param FAIL_SCORE Default fail score value.
     */
    LammpsSim(std::string &cmd, int LOSS_FUNC, std::string MAX_TIME, double FAIL_SCORE):
      cmd(cmd), lossfunc(LOSS_FUNC), maxtime(MAX_TIME), failscore(FAIL_SCORE)
  {
    // Ensure LAMMPS library integration.
    invar.push_back(TADAH_LAMMPS_LIB); // Assume TADAH_LAMMPS_LIB is defined appropriately.
    tokenise();
    parse_tokens();

    // Ensure required log and screen outputs are disabled by default.
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
     * @brief Tokenizes the command string.
     */
    void tokenise() {
      std::stringstream ss(cmd);
      std::string token;
      while (ss >> token) {
        tokens.push_back(token);
      }
    }

    /**
     * @brief Parses the tokenized command string to set class parameters.
     * 
     * Processes each token and updates class variables based on command-line options.
     */
    void parse_tokens() {
      for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == "-s" || tokens[i] == "--script") {
          if (i + 1 < tokens.size()) {
            script = tokens[++i];
          }
        } else if (tokens[i] == "-v" || tokens[i] == "--varloss") {
          if (i + 1 < tokens.size()) {
            varloss.push_back(tokens[++i]);
          }
        } else if (tokens[i] == "-i" || tokens[i] == "--invar") {
          if (i + 2 < tokens.size()) {
            invar.push_back("-var");
            invar.push_back(tokens[++i].c_str());
            invar.push_back(tokens[++i].c_str());
          }
          i++; // skip additional parameter
        } else if (tokens[i] == "-o" || tokens[i] == "--outvar") {
          if (i + 1 < tokens.size()) {
            outvar.push_back(tokens[++i]);
          }
        } else if (tokens[i] == "-w" || tokens[i] == "--weight") {
          if (i + 1 < tokens.size()) {
            weight = std::stod(tokens[++i]);
          }
        } else if (tokens[i] == "-f" || tokens[i] == "--failscore") {
          if (i + 1 < tokens.size()) {
            failscore = std::stod(tokens[++i]);
          }
        } else if (tokens[i] == "-l" || tokens[i] == "--lossfunc") {
          if (i + 1 < tokens.size()) {
            lossfunc = std::stoi(tokens[++i]);
          }
        } else if (tokens[i] == "-m" || tokens[i] == "--maxtime") {
          if (i + 1 < tokens.size()) {
            int tsec = std::stoi(tokens[++i]);
            int s, m, h;
            h = tsec / 3600;
            m = (tsec - h * 3600) / 60;
            s = tsec - h * 3600 - m * 60;
            maxtime = std::to_string(h) + ":" + std::to_string(m)
              + ":" + std::to_string(s) + " every 1";
          }
        } else if (tokens[i] == "-n" || tokens[i] == "--ncpu") {
          if (i + 1 < tokens.size()) {
            ncpu = std::stoi(tokens[++i]);
          }
        } else {
          throw std::runtime_error("Unknown LAMMPS config option:\n" + cmd);
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
     * @brief Serializes the LammpsSim object for MPI communication.
     * 
     * @param buffer Vector to store serialized data.
     */
    void serialize(std::vector<char>& buffer) const {
      // Serialize each element
      serialize_string(cmd, buffer);
      serialize_string(script, buffer);
      serialize_string(maxtime, buffer);

      serialize_vector_string(varloss, buffer);
      serialize_vector_string(outvar, buffer);

      // Serialize primitive types
      buffer.insert(buffer.end(), reinterpret_cast<const char*>(&ncpu), reinterpret_cast<const char*>(&ncpu) + sizeof(ncpu));
      buffer.insert(buffer.end(), reinterpret_cast<const char*>(&weight), reinterpret_cast<const char*>(&weight) + sizeof(weight));
      buffer.insert(buffer.end(), reinterpret_cast<const char*>(&lossfunc), reinterpret_cast<const char*>(&lossfunc) + sizeof(lossfunc));
      buffer.insert(buffer.end(), reinterpret_cast<const char*>(&failscore), reinterpret_cast<const char*>(&failscore) + sizeof(failscore));
    }

    static void serialize_string(const std::string& str, std::vector<char>& buffer) {
      size_t len = str.size();
      buffer.insert(buffer.end(), reinterpret_cast<const char*>(&len), reinterpret_cast<const char*>(&len) + sizeof(len));
      buffer.insert(buffer.end(), str.begin(), str.end());
    }

    static void serialize_vector_string(const std::vector<std::string>& vec, std::vector<char>& buffer) {
      size_t len = vec.size();
      buffer.insert(buffer.end(), reinterpret_cast<const char*>(&len), reinterpret_cast<const char*>(&len) + sizeof(len));
      for (const auto& str : vec) {
        serialize_string(str, buffer);
      }
    }

    /**
     * @brief Deserializes the LammpsSim object from MPI communication buffer.
     * 
     * @param buffer Vector containing serialized data.
     */
    void deserialize(const std::vector<char>& buffer) {
      size_t offset = 0;
      cmd = deserialize_string(buffer, offset);
      script = deserialize_string(buffer, offset);
      maxtime = deserialize_string(buffer, offset);

      varloss = deserialize_vector_string(buffer, offset);
      outvar = deserialize_vector_string(buffer, offset);

      ncpu = *reinterpret_cast<const int*>(&buffer[offset]);
      offset += sizeof(ncpu);
      weight = *reinterpret_cast<const double*>(&buffer[offset]);
      offset += sizeof(weight);
      lossfunc = *reinterpret_cast<const int*>(&buffer[offset]);
      offset += sizeof(lossfunc);
      failscore = *reinterpret_cast<const double*>(&buffer[offset]);
      offset += sizeof(failscore);
    }

    static std::string deserialize_string(const std::vector<char>& buffer, size_t& offset) {
      size_t len = *reinterpret_cast<const size_t*>(&buffer[offset]);
      offset += sizeof(len);
      std::string str(&buffer[offset], len);
      offset += len;
      return str;
    }

    static std::vector<std::string> deserialize_vector_string(const std::vector<char>& buffer, size_t& offset) {
      size_t len = *reinterpret_cast<const size_t*>(&buffer[offset]);
      offset += sizeof(len);
      std::vector<std::string> vec(len);
      for (size_t i = 0; i < len; ++i) {
        vec[i] = deserialize_string(buffer, offset);
      }
      return vec;
    }
};
#endif // LAMMPS_SIM_H
