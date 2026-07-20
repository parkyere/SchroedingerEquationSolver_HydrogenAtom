// RED: the 2D circular HO ladder. a_R-dag on the isotropic ground adds
// EXACTLY one omega quantum (and +1 angular momentum: the state grows a
// node-free e^{i phi} vortex); a_R on the ground annihilates (norm ~ 0);
// the up-then-down round trip lands back on the ground.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

import ses.lattice2d;
import ses.field;
import ses.grid;
import ses.observables;

namespace {

ses::Field3D ho_ground(const ses::Grid3D& g, double omega) {
    ses::Field3D psi{g};
    for (int j = 0; j < g.y.n; ++j) {
        const double y = g.y.coord(j);
        for (int i = 0; i < g.x.n; ++i) {
            const double x = g.x.coord(i);
            psi(i, j, 0) = std::exp(-0.5 * omega * (x * x + y * y));
        }
    }
    ses::normalize(psi);
    return psi;
}

double overlap2(const ses::Field3D& a, const ses::Field3D& b) {
    std::complex<double> ov{};
    for (std::size_t c = 0; c < a.data().size(); ++c) {
        ov += std::conj(a.data()[c]) * b.data()[c];
    }
    ov *= a.grid().x.spacing() * a.grid().y.spacing() *
          a.grid().z.spacing();
    return std::norm(ov);
}

TEST(Ho2dLadder, RaisesExactlyOneOmegaAndRoundTrips) {
    const double omega = 0.5;
    const ses::Grid3D g{ses::Grid1D{-14.0, 14.0, 128},
                        ses::Grid1D{-14.0, 14.0, 128},
                        ses::Grid1D{0.0, 2.0, 1}};
    std::vector<double> v(static_cast<std::size_t>(g.size()));
    for (int j = 0; j < g.y.n; ++j) {
        const double y = g.y.coord(j);
        for (int i = 0; i < g.x.n; ++i) {
            const double x = g.x.coord(i);
            v[static_cast<std::size_t>(g.flat(i, j, 0))] =
                0.5 * omega * omega * (x * x + y * y);
        }
    }
    const ses::Field3D ground = ho_ground(g, omega);
    const double e0 = ses::mean_energy(ground, v);
    EXPECT_NEAR(e0, omega, 0.01 * omega);  // E_00 = omega (2D zero point)

    ses::Field3D up = ses::ho2d_ladder(ground, omega, true);
    const double n_up = ses::norm_sq(up);
    ASSERT_GT(n_up, 0.5);  // a-dag on the ground has unit weight
    ses::normalize(up);
    const double e1 = ses::mean_energy(up, v);
    EXPECT_NEAR(e1 - e0, omega, 0.02 * omega);  // exactly one quantum

    ses::Field3D down = ses::ho2d_ladder(up, omega, false);
    ses::normalize(down);
    EXPECT_GT(overlap2(down, ground), 0.999);  // round trip

    ses::Field3D dead = ses::ho2d_ladder(ground, omega, false);
    EXPECT_LT(ses::norm_sq(dead), 1e-3);  // a|0> = 0: refuse signal
}

}  // namespace
