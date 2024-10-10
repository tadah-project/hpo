#ifndef LAMMPS_RUNNER_H
#define LAMMPS_RUNNER_H
#include <iostream>
#include <memory>
#ifdef TADAH_ENABLE_MPI
#include <mpi.h>
#endif


#include TADAH_LMPA
#include TADAH_LMPB
#include TADAH_LMPC

/** Simple Lammps runner class
 *
 * When this file is being compiled from outside of tadah library
 * the TADAH_LAMMPS_LIB macro must be provided.
 * In this case simply the name of the lammps library
 *
 * e.g. -DTADAH_LAMMPS_LIB=\"liblammps_single\"
 *
 * Note escape characters used to escape "".
 */
class LammpsRunner {
    public:
        typedef LAMMPS_NS::LAMMPS RI;

    public:
        LammpsRunner()
        {}

        template <typename ...Args>
        RI* instance(MPI_Comm &sub_comm,
                std::string logfile="none",Args...args) {
            const char *lmpargv[] {TADAH_LAMMPS_LIB,"-log",logfile.c_str(),args...};    // use append
            int lmpargc = sizeof(lmpargv)/sizeof(const char *);

            // create LAMMPS instance
            return new RI(lmpargc,
                    (char **)lmpargv, sub_comm);
        }

        template <typename ...Args>
        RI* run(MPI_Comm &sub_comm, std::string script,
                std::string logfile="log.lammps", Args...args) {

            RI *lmp=nullptr;
            try {
                lmp = instance(sub_comm,logfile,args...);
                lmp->input->file(script.c_str());
            }
            catch (std::exception& e) {
                std::cout << e.what() << std::endl;
                delete lmp;
                return nullptr;
            }
            return lmp;
        }

        std::string get_variable(RI*lmp, std::string varname) {
            return lmp->input->variable->retrieve(varname.c_str());
        }
};
#endif
