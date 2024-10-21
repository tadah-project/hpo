#ifndef LAMMPS_SCORES_H
#define LAMMPS_SCORES_H

#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/runners.h>
#include <tadah/mlip/analytics/analytics.h>
#include <tadah/core/utils.h>
#include <cmrc/cmrc.hpp>

#include <unistd.h>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <map>

class GLF2 {
  std::<double> w;
}
class GLF {
    private:
        HPOTargets &trg;

        size_t count=0;
        int prec=4;
        int width=prec+8;

        double tscore = 0.0;

        // Return loss bounded by MAX_SCORE^N
        double loss(double target, double val, double weight) {
            double s = std::pow(std::abs(val-target),trg.POW)*weight;
            return s;
            //return s>pow(trg.MAX_SCORE,trg.POW) ? pow(trg.MAX_SCORE,trg.POW) : s;
        }

        int score_lata(std::stringstream &outscores_temp,
                std::stringstream &outprops_temp, TrgData &pdata) {
            if (trg.use_lata) {
                for (auto const &v : trg.data.m1) {
                    std::string P=v.first;
                    double s;
                    if (pdata.is_fail(P)) {
                        s=pow(trg.MAX_SCORE,trg.POW);
                    }
                    else {
                        s= loss(trg.data.get_lata(P),
                                pdata.get_lata(P),
                                trg.data.get_wlata(P));
                    }
                    tscore += s;
                    outscores_temp << std::setw(width) << s;
                    outprops_temp << std::setw(width)
                        << pdata.get_lata(P);
                }
            }

            if (trg.use_ecoh) {
                for (auto const &v : trg.data.m1) {
                    std::string P=v.first;
                    double s;
                    if (pdata.is_fail(P)) {
                        s=pow(trg.MAX_SCORE,trg.POW);
                    }
                    else {
                        s= loss(trg.data.get_ecoh(P),
                                pdata.get_ecoh(P),
                                trg.data.get_wecoh(P));
                    }
                    tscore += s;
                    outscores_temp << std::setw(width) << s;
                    outprops_temp << std::setw(width)
                        << pdata.get_ecoh(P);
                }
            }

            return 0;
        }
        int score_uvacf(std::stringstream &outscores_temp,
                std::stringstream &outprops_temp, TrgData &pdata) {
            if (trg.use_uEvf) {
                for (auto const &v : trg.data.m2) {
                    std::string P=v.first;
                    if (trg.data.get_uEvf(P)!=0) {
                        double s;
                        // Unrelaxed so can't fail!
                        //if (pdata.is_fail(P)) {
                        //    s=pow(trg.MAX_SCORE,trg.POW);//*trg.uvacf.get_wE_vf(P);
                        //}
                        //else {
                        //    s= loss(trg.data.get_uEvf(P),
                        //            pdata.get_uEvf(P),
                        //            trg.data.get_uwEvf(P));
                        //}
                        s= loss(trg.data.get_uEvf(P),
                                pdata.get_uEvf(P),
                                trg.data.get_uwEvf(P));
                        tscore += s;
                        outscores_temp << std::setw(width) << s;
                        outprops_temp << std::setw(width)
                            << pdata.get_uEvf(P);
                    }
                }
            }
            if (trg.use_uHvf) {
                for (auto const &v : trg.data.m2) {
                    std::string P=v.first;
                    if (trg.data.get_uHvf(P)!=0) {
                        double s;
                        // Unrelaxed so can't fail!
                        //if (pdata.is_fail(P)) {
                        //    s=pow(trg.MAX_SCORE,trg.POW);//*trg.uvacf.get_H_vf(P);
                        //}
                        //else {
                        //    s= loss(trg.data.get_uHvf(P),
                        //            pdata.get_uHvf(P),
                        //            trg.data.get_uwHvf(P));
                        //}
                        s= loss(trg.data.get_uHvf(P),
                                pdata.get_uHvf(P),
                                trg.data.get_uwHvf(P));
                        tscore += s;
                        outscores_temp << std::setw(width) << s;
                        outprops_temp << std::setw(width)
                            << pdata.get_uHvf(P);
                    }
                }
            }
            return 0;
        }
        int score_usurface(std::stringstream &outscores_temp,
                std::stringstream &outprops_temp, TrgData &pdata) {
            if (trg.use_usurface) {
                for (auto const &v : trg.data.m4) {
                    int hkl=v.first;
                    double s = loss(trg.data.get_usurf(hkl),
                            pdata.get_usurf(hkl),
                            trg.data.get_uwsurf(hkl));
                    tscore += s;
                    outscores_temp << std::setw(width) << s;
                    outprops_temp << std::setw(width)
                        << pdata.get_usurf(hkl);
                }
            }
            return 0;
        }
        int score_surface(std::stringstream &outscores_temp,
                std::stringstream &outprops_temp, TrgData &pdata) {
            if (trg.use_surface) {
                for (auto const &v : trg.data.m5) {
                    int hkl=v.first;
                    double s = loss(trg.data.get_surf(hkl),
                            pdata.get_surf(hkl),
                            trg.data.get_wsurf(hkl));
                    tscore += s;
                    outscores_temp << std::setw(width) << s;
                    outprops_temp << std::setw(width)
                        << pdata.get_surf(hkl);
                }
            }
            return 0;
        }
        int score_envol(std::stringstream &outscores_temp,
                std::stringstream &outprops_temp, TrgData &pdata) {
            if (trg.use_envol) {
                for (auto const &v : trg.data.m7) {
                    std::string a =v.first;
                    double s = loss(trg.data.get_envol(a),
                            pdata.get_envol(a),
                            trg.data.get_wenvol(a));
                    tscore += s;
                    outscores_temp << std::setw(width) << s;
                    outprops_temp << std::setw(width)
                        << pdata.get_envol(a);
                }
            }
            return 0;
        }
        //        void elastic_const(std::map<int,std::tuple<double,double>> &elastic) {
        //
        //            int thread = omp_get_thread_num();
        //            int status = run_elastic(thread);
        //            if (status!=-1) {
        //                for (auto &ec: trg.elastic) {
        //                    int ij = ec.first;
        //                    double target = std::get<0>(trg.elastic[ij]);
        //                    double weight = std::get<1>(trg.elastic[ij]);
        //                    double val = stod(lmp_runner.get_variable(thread,
        //                                "C"+std::to_string(ij)+"all"));
        //                    double s = loss(target,val,weight);
        //                    elastic[ij] = std::make_tuple(val,s);
        //                    tscore += s;
        //                }
        //            }
        //            else {
        //                for (auto &ec: trg.elastic) {
        //                    //    int ij = ec.first;
        //                    //    elastic[ij] = std::make_tuple(0,trg.MAX_SCORE);
        //                    tscore += trg.MAX_SCORE;
        //                }
        //                fail=true;
        //            }
        //            lmp_runner.delete_lmp_thread(thread);
        //        }

    public:
        GLF(HPOTargets &trg):
            trg(trg)
    {

        std::ofstream outparams("params.dat");
        std::ofstream outprops("properties.dat");
        outprops.setf(std::ios::left);
        outprops << std::setw(width) << " ID";

        std::ofstream outscores("scores.dat");
        outscores.setf(std::ios::left);
        outscores << std::setw(width) << "ID";
        if (trg.use_ermse) outscores << std::setw(width) << "ERMSE";
        if (trg.use_ermse) outprops << std::setw(width) << "ERMSE";
        if (trg.use_srmse) outscores << std::setw(width) << "SRMSE";
        if (trg.use_srmse) outprops << std::setw(width) << "SRMSE";

        if (trg.use_lata) {
            for (auto &v : trg.data.m1) {
                const std::string P = v.first;
                char buff[10];
                snprintf(buff, sizeof(buff), "%s", P.c_str());
                std::string Pstr = buff;
                outscores << std::setw(width) << "Lat "+Pstr;
                outprops << std::setw(width) << "Lat "+Pstr;
            }
        }
        if (trg.use_ecoh) {
            for (auto &v : trg.data.m1) {
                const std::string P = v.first;
                char buff[10];
                snprintf(buff, sizeof(buff), "%s", P.c_str());
                std::string Pstr = buff;
                outscores << std::setw(width) << "E_c "+Pstr;
                outprops << std::setw(width) << "E_c "+Pstr;
            }
        }
        if (trg.use_uEvf) {
            for (auto &v : trg.data.m2) {
                const std::string P = v.first;
                char buff[10];
                snprintf(buff, sizeof(buff), "%s", P.c_str());
                std::string Pstr = buff;
                if (trg.data.get_uEvf(P)!=0) {
                    outscores << std::setw(width) << "uEvf "+Pstr;
                    outprops << std::setw(width) << "uEvf "+Pstr;
                }
            }
        }
        if (trg.use_uHvf) {
            for (auto &v : trg.data.m2) {
                const std::string P = v.first;
                char buff[10];
                snprintf(buff, sizeof(buff), "%s", P.c_str());
                std::string Pstr = buff;
                if (trg.data.get_uHvf(P)!=0) {
                    outscores << std::setw(width) << "uHvf "+Pstr;
                    outprops << std::setw(width) << "uHvf "+Pstr;
                }
            }
        }
        if (trg.use_usurface) {
            for (auto &surf : trg.data.m4) {
                outscores << std::setw(width)
                    << "uE_"+std::to_string(std::get<0>(surf));
                outprops << std::setw(width)
                    << "uE_"+std::to_string(std::get<0>(surf));
            }
        }
        if (trg.use_surface) {
            for (auto &surf : trg.data.m5) {
                outscores << std::setw(width)
                    << "E_"+std::to_string(std::get<0>(surf));
                outprops << std::setw(width)
                    << "E_"+std::to_string(std::get<0>(surf));
            }
        }
        if (trg.use_envol) {
            for (auto &envol : trg.data.m7) {
                outscores << std::setw(width)
                    << "E_"+std::get<0>(envol);
                outprops << std::setw(width)
                    << "E_"+std::get<0>(envol);
            }
        }
        //if (trg.use_elastic) {
        //    for (auto &ec : trg.elastic) {
        //        int ij = ec.first;
        //        outprops << std::setw(width)
        //            << "C_"+std::to_string(ij);
        //        outscores << std::setw(width)
        //            << "C_"+std::to_string(ij);
        //    }
        //}

        outscores<< std::setw(width) << "Total Score" << std::endl;
        outprops << std::endl;
    }

        template <typename V>
            double total_loss(const Analytics &a,
                    V &p, TrgData &pdata, bool fail) {

                tscore = 0.0;

                // Open scores and properties dump
                std::stringstream outscores_temp;
                outscores_temp.setf(std::ios::scientific);
                outscores_temp.setf(std::ios::left);
                outscores_temp.precision(prec);
                std::stringstream outprops_temp;
                outprops_temp.setf(std::ios::scientific);
                outprops_temp.setf(std::ios::left);
                outprops_temp.precision(prec);

                outscores_temp << std::setw(width) << count;
                outprops_temp << std::setw(width) << count;

                // Energy/Stress RMSE
                if (trg.use_ermse) {
                    double ermse = a.calc_e_rmse().mean(); // eV/atom
                    ermse *= 1000; // meV/atom;
                    double ermse_score = loss(trg.data.get_ermse(),ermse,trg.data.get_wermse());
                    tscore += ermse_score;
                    outscores_temp << std::setw(width) << ermse_score;
                    outprops_temp << std::setw(width) << ermse;
                }
                if (trg.use_srmse) {    // TODO
                    double srmse = a.calc_s_rmse().mean();
                    double srmse_score = loss(trg.data.get_srmse(),srmse,trg.data.get_wsrmse());
                    tscore += srmse_score;
                    outscores_temp << std::setw(width) << srmse_score;
                    outprops_temp << std::setw(width) << srmse;
                }

                // Cohesieve energy and lattice parameter is also calculated for surfaces
                if (trg.use_lata | trg.use_ecoh | trg.use_surface) {
                    score_lata(outscores_temp,outprops_temp,pdata);
                }

                // Vacancy formation energy/enthalpy at selected pressures
                if (trg.use_uEvf || trg.use_uHvf) {
                    score_uvacf(outscores_temp,outprops_temp,pdata);
                }

                // Surface energy
                if (trg.use_usurface) {
                    score_usurface(outscores_temp,outprops_temp,pdata);
                }
                if (trg.use_surface) {
                    score_surface(outscores_temp,outprops_temp,pdata);
                }
                if (trg.use_envol) {
                    score_envol(outscores_temp,outprops_temp,pdata);
                }

                // {
                //     // Elastic Constants
                //     if (trg.use_elastic) {
                //         elastic_const(elastic);
                //     }
                // }

                outscores_temp << std::setw(width) << tscore;

                // TODO consider printing even when failed
                if (!fail) {
                    std::ofstream outscores("scores.dat", std::ios_base::app);
                    std::ofstream outprops("properties.dat", std::ios_base::app);
                    outscores << outscores_temp.str() << std::endl;
                    outprops << outprops_temp.str() << std::endl;
                    // Open params dump
                    std::ofstream outparams("params.dat", std::ios_base::app);
                    outparams << std::setw(width) << count << std::setprecision(6)
                        << " " << p << std::endl;
                    count++;
                }

                return tscore;

            }
};
#endif
