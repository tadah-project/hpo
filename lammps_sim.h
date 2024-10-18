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

    LammpsSim();

    /**
     * @brief Constructor for LammpsSim class.
     * 
     * @param cmd The command line configuration.
     * @param LOSS_FUNC Default loss function identifier.
     * @param MAX_TIME Maximum time allowed for the simulation.
     * @param FAIL_SCORE Default fail score value.
     */
    LammpsSim(std::string &cmd, int LOSS_FUNC, std::string MAX_TIME, double FAIL_SCORE);

    /**
     * @brief Tokenises the command line configuration.
     */
    void tokenise();

    /**
     * @brief Parses the tokenised command line to set class parameters.
     */
    void parse_tokens();

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
    void serialise(std::vector<char>& buffer) const;

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
    void deserialise(const std::vector<char>& buffer, size_t& offset);

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
