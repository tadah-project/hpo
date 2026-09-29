// SIGNED goldens of the compact fitter: the fitter before RHOFORM (HPO
// 3ce1001) on the inputs of test_compact_fitter.cpp "compact fitter RHOFORM:
// SIGNED reproduces the numbers of the fitter before RHOFORM", printed as C99
// hexfloats by a driver built with the flags of the HPO library (-O3, no
// -march=native, no -ffast-math) against the 3ce1001 sources. The FEAT-02
// fitter reproduced every value bit for bit when these were generated.
// Frozen reference data; do not edit.
#ifndef TADAH_HPO_TESTS_COMPACT_FITTER_SIGNED_GOLDENS_H
#define TADAH_HPO_TESTS_COMPACT_FITTER_SIGNED_GOLDENS_H

#include <vector>

namespace compact_signed_golden {

// Blip, value solve: c (5), curve_rmse, (eval, deriv, deriv2) at 1.3 / 2.75 /
// 4.1, reduced_mse, its gradient (mu..., eta...)
const std::vector<double> blip = {
    0x1.4989b6fe74337p-4, -0x1.3ea1b8cb1c51dp-3, -0x1.0681fa5a90873p-3,
    -0x1.0e11b4c7b0f0dp-3, -0x1.ee07deaba59f9p-3, 0x1.68c3e01f7ff76p-5,
    0x1.4110a77d8ca74p-5, -0x1.7822e2f1e1df4p-6, -0x1.491b3ca6ee7bap-2,
    -0x1.af32e3b4802bep-3, -0x1.674710b5a910cp-4, 0x1.4f6423dfd18f4p-2,
    -0x1.c2aadb7dd0ac6p-3, -0x1.2431a6eff8ff1p-4, -0x1.24264d7c971b8p-2,
    0x1.799b3367b1c87p-9, 0x1.1e37c76a9af94p-9, 0x1.5dc59109af654p-10,
    -0x1.9e711a5e063d2p-10, 0x1.34dc904b96433p-9, -0x1.04be6896a4dbep-7,
    0x1.b72079ebfa70ep-11, 0x1.29f93c2b33e1ep-12, -0x1.3f62ed3e88507p-12,
    0x1.a08dd08622ac9p-12, 0x1.9fd65351ceadep-10};

// Blip5: the same layout (4 bumps)
const std::vector<double> blip5 = {
    0x1.3d13723eb4279p-3, -0x1.2460bb5d6e9a2p-2, 0x1.f26120e0fadbfp-6,
    -0x1.3af7ab99da6b3p-2, 0x1.595a18d6a98e1p-5, 0x1.29ebeb0d2d86ap-4,
    -0x1.04f96c9f6d4d8p-3, -0x1.6b23afbe8abdep-2, -0x1.af48c02f2633fp-3,
    -0x1.ee693f6c51b76p-6, 0x1.7e4d94994452p-2, -0x1.05a0ee3f3b8afp-2,
    -0x1.4b92e489a1386p-3, 0x1.44806e8ef86dp-5, 0x1.6105e9e1dc23dp-9,
    0x1.6cdbf15c66fa5p-12, 0x1.37007050eb6aep-13, -0x1.26709b3edf304p-13,
    -0x1.80a7169c82a3ep-8, 0x1.9c9f164eae634p-11, 0x1.7fa6b374bfcb9p-10,
    0x1.7f467ff56a9a6p-13, 0x1.bdf8f550978ccp-8};

// Knot5L: the same layout (4 knots; gradient over mu only)
const std::vector<double> knot5 = {
    0x1.6aa03538f999bp+0, -0x1.996da65a7c6d6p-3, 0x1.3580588726861p-5,
    -0x1.1698b5474f00ep-9, 0x1.570ab1b37b1d7p-3, 0x1.5ed639c8cf3p-5,
    -0x1.5872d10fed88p-4, 0x1.cbfb1e9e5393cp+0, -0x1.c0e777624dad7p-3,
    0x1.3443bdecd2e7cp-3, 0x1.0efb0a1a57684p-1, -0x1.7696f8a7b3badp-7,
    0x1.4e748295bbe6cp-5, -0x1.ddcb03b155928p-4, 0x1.5a2f1eec705f6p-5,
    -0x1.17ddc4b459cf9p-9, -0x1.b890644f00c7cp-9, -0x1.187be189eedfp-8,
    -0x1.2ca8b13bd005ep-8};

// Blip, curve objective (curve_deriv_weight 1): curve_rmse, c (5), d2_rmse,
// combined_curve_score
const std::vector<double> stacked = {
    0x1.822da61bffd97p-1, -0x1.ee520c1d8e8fep-4, -0x1.fddd356360187p-4,
    -0x1.e404edc19c3cep-4, -0x1.744fd7f508661p-4, -0x1.c0d46b5e7f5bp-5,
    0x1.2731e8c0bccf9p-3, 0x1.822da61bffd97p-1};

// make_curve_cost (Blip5, 6 samples, 3 bumps): residuals, then the c / mu /
// eta Jacobian blocks
const std::vector<double> curve_cost = {
    0x1.ec20b362fb2ep-8, 0x1.709b4e4aa7aa8p-3, -0x1.8a689df4ce31ep-2,
    -0x1.2909ea36bee06p-3, 0x1.9b35b3a14df16p-2, 0x1.b4a88c4e21488p-2,
    0x1.80a60eef7fc21p-1, 0x1.a247683135429p-6, 0x0p+0, 0x1.3824311ee6eacp+0,
    0x1.a4a8688546901p-2, 0x0p+0, 0x1.ce1e443f09f2dp-1, 0x1.52d75aab0c6fbp+0,
    0x1.01e9d720faf97p-5, 0x1.79ec0129a033p-3, 0x1.888018bbf4017p-1,
    0x1.0b001ac14d2dep-1, 0x1.71e7395915b21p-6, 0x1.b6539fe80e9dfp-3,
    0x1.298d47d2eebafp+0, 0x1.044f44ddb598ap-12, 0x1.25a4391027797p-7,
    0x1.12801a1ad9518p-2, -0x1.4c9dec455b3c5p-3, 0x1.2898e44c9f209p-4, -0x0p+0,
    0x1.0ae1cbf799cbep-5, 0x1.1611cd670d07bp-1, -0x0p+0, 0x1.f96cfce1141e1p-3,
    0x1.a2e5d718aa1d3p-2, -0x1.1e85c62cc986fp-5, 0x1.ac0d9b0aa3433p-4,
    -0x1.ea4028603aafdp-2, -0x1.966410c3db1b4p-3, 0x1.75748c4a7c10dp-6,
    -0x1.7b8c894d951f2p-2, 0x1.f2228b2649a69p-4, 0x1.4d52c5734f8e2p-11,
    -0x1.1349f57f2501ep-5, 0x1.5fee586c26b16p-3, -0x1.7c2232986844fp-3,
    0x1.5a07b5040efb5p-3, -0x0p+0, -0x1.31020dad420d2p-8, 0x1.72c26734115fbp-1,
    -0x0p+0, -0x1.6904b4a0c53ap-2, 0x1.1743e4bb1c13cp-3, -0x1.7e07b2e662097p-5,
    -0x1.2276f25981124p-2, 0x1.46d57040271ffp-2, -0x1.da1f68e47f9ffp-4,
    -0x1.75748c4a7c10dp-4, 0x1.3c4a726b5199dp-1, -0x1.4c17076edbc31p-6,
    -0x1.b87684eaa9207p-9, 0x1.6f0d47543157dp-4, -0x1.429a7bb878cd3p-3};

// make_energy_cost (2 scan points, 2 + 2 + 2 bumps): residuals, then the nine
// Jacobian blocks
const std::vector<double> energy_cost = {
    0x1.4f721d33c2adcp-3, 0x1.0e64793a843f2p+2, 0x1.33733226c3b93p+0,
    0x1.6573d188f42ffp+0, 0x1.f543369b9d7fdp+1, 0x1.4d341d084e833p+0,
    0x1.53ae7d566cf42p+1, 0x1.ece3de6149c72p-1, 0x1.098075f6fd222p+2,
    0x1.ceea1c68ec52dp+0, -0x1.1ecc8de2ac322p+1, 0x1.79455219a847ep-1,
    -0x1.4453c60029f19p+1, 0x1.5f468ec52a413p+0, 0x1.df68d7908190bp+0,
    0x1.b0652cf7db45ep+0, 0x1.3804a6ae767f4p+3, 0x1.b346ed17786fdp+1,
    0x1.ff21530ccd57cp-1, 0x1.627bc1c8ff118p-2, 0x1.a56ac8bd94de2p+0,
    0x1.5388ab96534b4p+0, -0x1.b3ea547d6c3a4p-1, 0x1.5124439c9482ap-2,
    -0x1.71585249d4d34p+0, 0x1.7538495d9d025p+0, 0x1.e926ce96aa0cp-1,
    0x1.c60f7eec2821ep-3, 0x1.37c2b959c1daap+0, 0x1.22738a6c3d014p+0,
    -0x1.b033b034c9844p-2, -0x1.1b9044d80cabfp-3, -0x1.c9d60c561d3e4p+0,
    -0x1.99791e6eaa893p-2, 0x1.8200ad57aa596p-2, -0x1.cc745025c1f95p-1,
    0x1.5a99550ac20b7p+2, -0x1.a3070686471eap+0};

// refine (CERES_LM): curve_rmse, c (4), mu (4), eta (4)
const std::vector<double> refine = {
    0x1.ca1fa6a790f9fp-14, 0x1.031f17b60f007p+1, -0x1.4ff3da5ce7722p+0,
    0x1.68a663bdf840ep-1, -0x1.9ebbac23e02fp-2, 0x1.801dc6343ffd6p+0,
    0x1.3241f6fc26c46p+1, 0x1.cce6b83c4482ap+1, 0x1.32fbe3e66c0d7p+2,
    0x1.32303ad668aap+0, 0x1.ca3967989b278p-1, 0x1.966cc6f07f1b6p-1,
    0x1.196c1d4607f18p+0};

// joint_refine (CERES_LM, 500 it): max weighted |E residual|, then (c, mu,
// eta) of phi, rho, F
const std::vector<double> joint = {
    0x0p+0, 0x1.ffffffffffffap+0, -0x1.4cccccccccce7p+0, 0x1.666666666667ep-1,
    -0x1.99999999999c7p-2, 0x1.8p+0, 0x1.3333333333338p+1, 0x1.cccccccccccc4p+1,
    0x1.3333333333332p+2, 0x1.3333333333332p+0, 0x1.cccccccccccd1p-1,
    0x1.999999999996bp-1, 0x1.1999999999999p+0, 0x1.333333333334bp-1,
    -0x1.99999999999c1p-3, 0x1.3333333333324p-3, 0x1.0000000000005p+1,
    0x1.999999999997fp+1, 0x1.199999999999dp+2, 0x1.cccccccccccc8p-1,
    0x1.9999999999975p-1, 0x1.cccccccccccd8p-1, -0x1p+1, -0x1.999999999999bp-1,
    0x1.0000000000001p-1, 0x1p+0, 0x1.0000000000001p+2, 0x1.bffffffffffffp+2,
    0x1.ccccccccccccdp-2, 0x1.9999999999999p-2, 0x1.cccccccccccccp-2};

// fit() ladder (n_max 5, no jitters): n, curve_rmse, c, mu, eta
const std::vector<double> ladder = {
    0x1.4p+2, 0x1.cd235748f11d5p-16, 0x1.578fb199ac63cp+1,
    -0x1.1b3b69cf792efp-2, 0x1.199fa886a4d2fp+1, -0x1.0e55e3e1f1f8p-3,
    0x1.f76aff767b9f9p-8, -0x1.cp+1, 0x1.be8fa12fd6789p+2,
    -0x1.45bed42aa8218p+1, 0x1.17cdc50ec517p-1, 0x1.9144e21ce6f16p+1,
    0x1.7ebe1b679f275p-2, 0x1.9f2ae33efc44bp-4, 0x1.2f9274eabc15ap-2,
    0x1.1713122b8fc57p-1, 0x1.d826164c7cab6p-1};

}  // namespace compact_signed_golden

#endif  // TADAH_HPO_TESTS_COMPACT_FITTER_SIGNED_GOLDENS_H
