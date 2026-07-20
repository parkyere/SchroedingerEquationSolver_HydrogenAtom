// RED: the exact prolate-spheroidal H2+ eigensolver (ses.spheroidal).
//
// H2+ separates EXACTLY in prolate spheroidal coordinates into two coupled
// 1D ODEs (angular in eta, radial in xi), solved numerically -- the same
// reduce-to-1D-and-synthesize methodology as the hydrogen radial atlas.
// Oracles are the published electronic energies E_elec = -2 p^2 / R^2 (no
// 1/R): 1sigma_g/2p sigma_u*/2sigma_g/1pi_u at R=2 (Scott et al.
// arXiv:physics/0607081; Turbiner arXiv:1401.8009), and the R=1/2/4 bond
// curve of E_total = E_elec + 1/R. Synthesis: <H> of the sampled orbital
// in the two-center Coulomb landscape reproduces E_elec.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>

import ses.spheroidal;
import ses.field;
import ses.grid;
import ses.observables;
import ses.potential;
import ses.vec;

namespace {

using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;
using ses::Vec3d;

// Electronic energies (Ha) at R = 2.0, tolerance loose enough for a coarse
// finite-difference solve but tight enough to pin the physics.
TEST(Spheroidal, KnownOrbitalEnergiesAtEquilibrium) {
    const double R = 2.0;
    const ses::H2plusOrbital sg = ses::h2plus_orbital(R, 0, 0, 0);
    EXPECT_NEAR(sg.energy, -1.1026342, 0.01) << "1sigma_g electronic";
    EXPECT_EQ(sg.parity, +1) << "1sigma_g is gerade";

    const ses::H2plusOrbital su = ses::h2plus_orbital(R, 0, 1, 0);
    EXPECT_NEAR(su.energy, -0.6675344, 0.01) << "2p sigma_u* electronic";
    EXPECT_EQ(su.parity, -1) << "2p sigma_u* is ungerade";

    const ses::H2plusOrbital sg2 = ses::h2plus_orbital(R, 0, 0, 1);
    EXPECT_NEAR(sg2.energy, -0.3608649, 0.01) << "2sigma_g electronic";
    EXPECT_EQ(sg2.parity, +1);

    const ses::H2plusOrbital pu = ses::h2plus_orbital(R, 1, 0, 0);
    EXPECT_NEAR(pu.energy, -0.4287723, 0.01) << "1pi_u electronic";
    EXPECT_EQ(pu.parity, -1) << "1pi_u is ungerade";

    // Energy ordering of the four lowest MOs.
    EXPECT_LT(sg.energy, su.energy);
    EXPECT_LT(su.energy, pu.energy);
    EXPECT_LT(pu.energy, sg2.energy);
}

TEST(Spheroidal, GroundEnergyVsInternuclearDistance) {
    // E_elec of 1sigma_g at three separations (Turbiner Table I / Scott).
    EXPECT_NEAR(ses::h2plus_orbital(1.0, 0, 0, 0).energy, -1.4517863, 0.02);
    EXPECT_NEAR(ses::h2plus_orbital(2.0, 0, 0, 0).energy, -1.1026342, 0.01);
    EXPECT_NEAR(ses::h2plus_orbital(4.0, 0, 0, 0).energy, -0.7960849, 0.01);

    // The chemical bond: E_total = E_elec + 1/R dips near R = 2.
    const double et1 = ses::h2plus_orbital(1.0, 0, 0, 0).energy + 1.0 / 1.0;
    const double et2 = ses::h2plus_orbital(2.0, 0, 0, 0).energy + 1.0 / 2.0;
    const double et4 = ses::h2plus_orbital(4.0, 0, 0, 0).energy + 1.0 / 4.0;
    EXPECT_LT(et2, et1) << "the bond binds vs compressed";
    EXPECT_LT(et2, et4) << "the bond binds vs stretched";
}

// Synthesizing the orbital onto a Cartesian grid and taking <H> in the
// two-center regularized Coulomb potential reproduces E_elec: the solve and
// the synthesis are consistent end to end.
TEST(Spheroidal, SynthesizedOrbitalHasTheSolvedEnergy) {
    const double R = 2.0;
    const Grid1D ax{-16.0, 16.0, 128};
    const Grid3D g{ax, ax, ax};
    const ses::H2plusOrbital sg = ses::h2plus_orbital(R, 0, 0, 0);
    const Field3D psi = ses::synthesize_h2plus(g, sg, 0);

    // gerade: even under inversion through the bond midpoint (origin).
    const int c = 64;  // coord 0 (128 points over +-16, index 64 -> ~0)
    EXPECT_GT(std::norm(psi(c, c, c)), 0.0);

    const std::vector<double> v = ses::regularized_coulomb_potential(
        g, 1.0, {{-R / 2, 0.0, 0.0}, {R / 2, 0.0, 0.0}});
    const double e = ses::mean_energy(psi, v);
    // Coarse grid + regularized cusp: a few percent of E_elec.
    EXPECT_NEAR(e, sg.energy, 0.05) << "synthesized <H> matches the solve";
}

TEST(Spheroidal, PiUOrbitalHasAnAxisNode) {
    const double R = 2.0;
    const Grid1D ax{-16.0, 16.0, 128};
    const Grid3D g{ax, ax, ax};
    const ses::H2plusOrbital pu = ses::h2plus_orbital(R, 1, 0, 0);
    // partner 0 = cos(phi) = y/rho -> nodal plane y = 0.
    const Field3D psi = ses::synthesize_h2plus(g, pu, 0);
    double node = 0.0;
    double bulk = 0.0;
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double w = std::norm(psi(i, j, k));
                if (j == 64) {
                    node += w;
                } else {
                    bulk += w;
                }
            }
        }
    }
    EXPECT_LT(node, 0.02 * bulk) << "1pi_u (cos phi) has the y = 0 nodal plane";
}

}  // namespace
