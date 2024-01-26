#ifndef LAMMPS_SCRIPTS_H
#define LAMMPS_SCRIPTS_H

#include <cmrc/cmrc.hpp>
#include <fstream>
#include <unistd.h>
// Gain access to embedded lammps script files
CMRC_DECLARE(hpo);
class LammpsScripts {
    private:
        // allow access to embedded lammps scripts
        cmrc::embedded_filesystem hpo_fs;

        // List of lammps scripts, each string will point
        // to a temporary file which contains relavant scripts
        std::map<std::string, std::string> scripts;

        /* Open file at temporary location at attach stream to it. */
        std::string open_temp(std::string path, std::ofstream& f) {
            path += "/XXXXXX";
            std::vector<char> dst_path(path.begin(), path.end());
            dst_path.push_back('\0');

            int fd = mkstemp(&dst_path[0]);
            if(fd != -1) {
                path.assign(dst_path.begin(), dst_path.end() - 1);
                f.open(path.c_str(),
                        std::ios_base::trunc | std::ios_base::out);
                close(fd);
            }
            return path;
        }

        /* Copy embedded lammps scripts to temp location and fill scripts map.
        */
        void prep_lammps_script() {
            for (auto const &de: hpo_fs.iterate_directory("hpo")) {
                auto const script = hpo_fs.open("hpo/"+de.filename());
                std::ofstream temp_stream;
                const std::string temp_script_path = open_temp("/tmp", temp_stream);
                std::copy(script.begin(), script.end(), std::ostreambuf_iterator<char>(temp_stream));
                temp_stream.close();
                scripts.insert(std::make_pair(de.filename(),temp_script_path));
            }
        }

    public:
        LammpsScripts():
            hpo_fs(cmrc::hpo::get_filesystem())
    {
        // Prepare scripts for lammps
        prep_lammps_script();
    }
        template <typename R>
        typename R::RI* run_lata(R &runner,std::string P, std::string structure,
                double lata_guess, std::string element,
                std::string log="log.lata", std::string MAX_TIME="10") {
            auto lmp = runner.run(
                    scripts["tadah.lata"], log.c_str(),
                    "-var","lata_guess",std::to_string(lata_guess).c_str(),
                    "-var","P",P.c_str(),
                    "-var", "tout",MAX_TIME.c_str(),
                    "-var", "structure", structure.c_str(),
                    "-var", "element", element.c_str());
            return lmp;
        }
        template <typename R>
        typename R::RI*  run_uvacf(R &runner, std::string P, double lata_guess,
                std::string log="log.uvacf", std::string MAX_TIME="10") {
            auto lmp = runner.run(
                    scripts["tadah.unrelaxed_vacancy_at_p"], log,
                    "-var", "ao",std::to_string(lata_guess).c_str(),
                    "-var", "P",P.c_str(),
                    "-var", "tout",MAX_TIME.c_str());
            return lmp;
        }
        template <typename R>
        typename R::RI* run_usurface(R &runner, int hkl, double lata,
                std::string structure ,std::string element,
                std::string log="log.usurface", std::string MAX_TIME="10") {
            auto lmp = runner.run(
                     scripts["tadah.usurface"], log,
                    "-var","lata",std::to_string(lata).c_str(),
                    "-var","surface",std::to_string(hkl).c_str(),
                    "-var", "structure", structure.c_str(),
                    "-var", "element", element.c_str(),
                    "-var", "tout",MAX_TIME.c_str());
            return lmp;
        }
        template <typename R>
        typename R::RI* run_surface(R &runner, int hkl, double lata,
                std::string structure ,std::string element,
                std::string log="log.surface", std::string MAX_TIME="10") {
            auto lmp = runner.run(
                     scripts["tadah.surface"], log,
                    "-var","lata",std::to_string(lata).c_str(),
                    "-var","surface",std::to_string(hkl).c_str(),
                    "-var", "structure", structure.c_str(),
                    "-var", "element", element.c_str(),
                    "-var", "tout",MAX_TIME.c_str());
            return lmp;
        }
        template <typename R>
        typename R::RI* run_envol(R &runner,std::string lata,
                std::string structure,
                std::string element,
                std::string log="log.lata", std::string MAX_TIME="10") {
            auto lmp = runner.run(
                    scripts["tadah.envol"], log.c_str(),
                    "-var","lata", lata.c_str(),
                    "-var", "tout",MAX_TIME.c_str(),
                    "-var", "structure", structure.c_str(),
                    "-var", "element", element.c_str());
            return lmp;
        }

       // int run_elastic(int thread, std::string log="log.elastic") {
       //     int status = lmp_runner.run(
       //             thread,scripts["tadah.elastic"], log,
       //             "-var", "tout",trg.MAX_TIME.c_str());
       //     return status;
       // }
};
#endif
