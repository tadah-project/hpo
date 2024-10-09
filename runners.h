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
        //typedef std::unique_ptr<RI> uLMP;  // TODO implement with unique ptr

    private:
        template<typename T, typename... Args>
            std::unique_ptr<T> make_unique(Args&&... args) {
                return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
            }

        //template <typename ...Args>
        //    std::unique_ptr<RI> instance2(
        //            std::string logfile="none",Args...args) {
        //            //std::string logfile="log.lammps",Args...args) {
        //        const char *lmpargv[] {TADAH_LAMMPS_LIB,"-log","none",args...};    // use append
        //        int lmpargc = sizeof(lmpargv)/sizeof(const char *);

        //        // create LAMMPS instance
        //        return make_unique<RI>(lmpargc,
        //                (char **)lmpargv, MPI_COMM_WORLD);
        //    }
    public:
        LammpsRunner()
        {}

        template <typename ...Args>
        RI* instance(
                std::string logfile="none",Args...args) {
                //std::string logfile="log.lammps",Args...args) {
            // custom argument vector for LAMMPS library
           // const char *lmpargv[] {"liblammps","-log",
           //     logfile.c_str(),args...};    // use append
            const char *lmpargv[] {TADAH_LAMMPS_LIB,"-log",logfile.c_str(),args...};    // use append
            int lmpargc = sizeof(lmpargv)/sizeof(const char *);

            // create LAMMPS instance
            return new RI(lmpargc,
                    (char **)lmpargv, MPI_COMM_WORLD);
        }

        template <typename ...Args>
        RI* run(std::string script,
                std::string logfile="log.lammps", Args...args) {

                std::cout << "DEBUG 1" << std::endl;
            RI *lmp=nullptr;
                std::cout << "DEBUG 2" << std::endl;
            try {
                std::cout << "DEBUG 3" << std::endl;
                lmp = instance(logfile,args...);
                std::cout << "DEBUG 4" << std::endl;
                lmp->input->file(script.c_str());
                std::cout << "DEBUG 5" << std::endl;
            }
            catch (std::exception& e) {
                std::cout << "DEBUG 6" << std::endl;
                std::cout << e.what() << std::endl;
                std::cout << "DEBUG 7" << std::endl;
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
