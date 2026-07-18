// RED: the 1D periodic lattice (solid-state) core.
//
// Potential: V(x) = V0 sin^2(kL x) -- SMOOTH, so the FFT split-operator
// keeps spectral accuracy (the user's own call: Kronig-Penney's kinks
// would Gibbs-ring in the plane-wave basis). This is exactly the optical-
// lattice / Mathieu problem, lattice constant a = pi / kL, reciprocal
// vector G = 2 kL.
//
// Band structure: V0 sin^2 = V0/2 - (V0/4)(e^{i2kLx} + e^{-i2kLx}) has a
// SINGLE harmonic, so the central equation in the plane-wave basis
// {q + m G} is symmetric TRIDIAGONAL: diagonal (q + m G)^2/2 + V0/2,
// off-diagonal -V0/4. lattice_bands diagonalizes it exactly.
//   - V0 = 0: the folded free parabola.
//   - small V0: first gap at the zone edge = V0/2 (first-order PT: twice
//     the |G| Fourier amplitude V0/4).
//
// Bloch oscillations: a uniform force F breaks the periodic box as a
// potential -F x, but enters EXACTLY as the comoving gauge A(t) = -F t:
// the kinetic phase e^{-i (k - A)^2 dt / 2} is rebuilt each step with the
// MIDPOINT A (midpoint is exact for linear A). Contracts:
//   - V = 0: <x>(t) = x0 + k0 t + F t^2/2 EXACTLY (Ehrenfest for a linear
//     potential; validates the gauge to round-off).
//   - V = lattice: the packet does NOT run away -- quasimomentum sweeps
//     the zone, q(t) = q0 + F t, and <x> oscillates with the Bloch period
//     T_B = G / F, returning near its start while a free particle would
//     have fallen F T_B^2 / 2 away.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

import ses.bloch;
import ses.field;
import ses.grid;
import ses.wavepacket;

namespace {

TEST(LatticeBands, FreeLimitFoldsTheParabola) {
    const double kl = 1.0;
    const double g2 = 2.0 * kl;  // reciprocal vector
    for (const double q : {0.0, 0.3, 0.9}) {
        const std::vector<double> e = ses::lattice_bands(0.0, kl, q, 4);
        ASSERT_EQ(e.size(), 4u);
        // Expected: (q + m G)^2 / 2 over m = ..., sorted ascending.
        std::vector<double> want;
        for (int m = -3; m <= 3; ++m) {
            const double k = q + m * g2;
            want.push_back(0.5 * k * k);
        }
        std::sort(want.begin(), want.end());
        for (int n = 0; n < 4; ++n) {
            EXPECT_NEAR(e[static_cast<std::size_t>(n)],
                        want[static_cast<std::size_t>(n)], 1e-9)
                << "band " << n << " at q = " << q;
        }
    }
}

TEST(LatticeBands, FirstGapAtTheZoneEdgeIsHalfV0) {
    const double kl = 1.0;
    const double v0 = 0.1;  // weak lattice: first-order PT regime
    const std::vector<double> e = ses::lattice_bands(v0, kl, kl, 2);
    EXPECT_NEAR(e[1] - e[0], 0.5 * v0, 0.1 * 0.5 * v0);
    // And the bands are monotone across the half zone (no crossing).
    const std::vector<double> mid = ses::lattice_bands(v0, kl, 0.5 * kl, 2);
    EXPECT_LT(mid[0], e[0]);
    EXPECT_GT(mid[1], e[1]);
}

TEST(TiltedSplitOperator1D, FreeTiltIsExactlyUniformAcceleration) {
    const ses::Grid1D g{-40.0, 40.0, 2048};
    const std::vector<double> zero(static_cast<std::size_t>(g.n), 0.0);
    const double f = 0.3;
    const double dt = 0.005;
    ses::TiltedSplitOperator1D prop{g, zero, dt, f};
    const double k0 = 1.0;
    const double x0 = -20.0;
    ses::Field1D psi = ses::gaussian_wavepacket(g, x0, 3.0, k0);
    const int steps = 1600;  // T = 8
    prop.step(psi, steps);
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < g.n; ++i) {
        const double w = std::norm(psi[i]);
        num += g.coord(i) * w;
        den += w;
    }
    const double t = steps * dt;
    EXPECT_NEAR(num / den, x0 + k0 * t + 0.5 * f * t * t, 1e-6);
    EXPECT_NEAR(prop.drift(), f * t, 1e-12);
}

TEST(TiltedSplitOperator1D, BlochOscillationReturnsInsteadOfRunningAway) {
    // 26 lattice periods (a = pi) in the periodic box; s = V0/E_R = 3
    // (open gaps, negligible Zener at F a << gap^2 / bandwidth).
    const double kl = 1.0;
    const double a = std::numbers::pi / kl;
    const ses::Grid1D g{-13.0 * a, 13.0 * a, 2048};
    const double v0 = 1.5;
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        const double s = std::sin(kl * g.coord(i));
        v[static_cast<std::size_t>(i)] = v0 * s * s;
    }
    const double f = 0.05;
    const double t_b = 2.0 * kl / f;  // Bloch period G / F = 40
    const double dt = 0.01;
    ses::TiltedSplitOperator1D prop{g, v, dt, f};
    // Broad packet at rest on a well minimum: ground band, q ~ 0.
    ses::Field1D psi = ses::gaussian_wavepacket(g, 0.0, 6.0, 0.0);
    auto mean_x = [&] {
        double num = 0.0;
        double den = 0.0;
        for (int i = 0; i < g.n; ++i) {
            const double w = std::norm(psi[i]);
            num += g.coord(i) * w;
            den += w;
        }
        return num / den;
    };
    const double x0 = mean_x();
    const int steps = static_cast<int>(t_b / dt + 0.5);
    double excursion = 0.0;
    for (int s = 0; s < steps; s += 50) {
        prop.step(psi, std::min(50, steps - s));
        excursion = std::max(excursion, std::abs(mean_x() - x0));
    }
    const double free_fall = 0.5 * f * t_b * t_b;  // = 40
    EXPECT_LT(excursion, 0.15 * free_fall);  // never runs away
    EXPECT_LT(std::abs(mean_x() - x0), 1.5);  // and RETURNS at T_B
}

}  // namespace
