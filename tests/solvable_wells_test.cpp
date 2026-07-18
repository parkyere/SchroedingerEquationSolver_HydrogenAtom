// RED: the license physics of the three new 1D "solvable well" scenes,
// plus the shared oracle they all lean on:
//
//  - bound_states_1d(grid, V, count): the lowest eigenpairs of ANY 1D
//    potential via the radial engine's Dirichlet FD tridiagonal solver
//    (the 1D box [xmin, xmax] IS a radial problem at l = 0). Verified
//    against the HO ladder and the closed-form Morse spectrum.
//  - Double well: the ground doublet's splitting dE drives a full
//    left <-> right tunneling oscillation with period 2 pi / dE (the
//    ammonia-inversion demo): psi_L = (psi_0 + psi_1)/sqrt(2) transfers
//    to the right well at t = pi / dE.
//  - Poschl-Teller: V = -l(l+1)/(2 a^2) sech^2(x/a) at integer l is
//    REFLECTIONLESS for every incident energy; an equal-depth, equal-area
//    square well reflects visibly at the same energy.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>

import ses.field;
import ses.grid;
import ses.ladder;
import ses.observables;
import ses.potential;
import ses.propagator;
import ses.spectrum1d;
import ses.wavepacket;

namespace {

double overlap_sq(const ses::Field1D& a, const ses::Field1D& b) {
    std::complex<double> acc{};
    for (int i = 0; i < a.size(); ++i) {
        acc += std::conj(a[i]) * b[i];
    }
    acc *= a.grid().spacing();
    return std::norm(acc);
}

int node_count(const ses::Field1D& f) {
    int nodes = 0;
    double prev = 0.0;
    for (int i = 0; i < f.size(); ++i) {
        const double re = f[i].real();
        if (std::abs(re) < 1e-6) {
            continue;
        }
        if (prev != 0.0 && (re > 0.0) != (prev > 0.0)) {
            ++nodes;
        }
        prev = re;
    }
    return nodes;
}

TEST(BoundStates1D, ReproducesTheHarmonicLadderInABox) {
    const ses::Grid1D g{-20.0, 20.0, 512};
    const double w = 0.5;
    const std::vector<double> v = ses::harmonic_potential(g, w);
    const std::vector<ses::Bound1D> s = ses::bound_states_1d(g, v, 6);
    ASSERT_EQ(s.size(), 6u);
    for (int k = 0; k < 6; ++k) {
        EXPECT_NEAR(s[static_cast<std::size_t>(k)].energy, (k + 0.5) * w, 1e-4)
            << "E_" << k;
        EXPECT_NEAR(ses::norm_sq(s[static_cast<std::size_t>(k)].psi), 1.0, 1e-9)
            << "norm_" << k;
        EXPECT_EQ(node_count(s[static_cast<std::size_t>(k)].psi), k)
            << "nodes_" << k;
    }
    // Cross-validate the FD eigenvector against the Hermite oracle.
    for (int k = 0; k < 4; ++k) {
        EXPECT_NEAR(overlap_sq(s[static_cast<std::size_t>(k)].psi,
                               ses::ho_eigenstate(g, w, k)),
                    1.0, 1e-6)
            << "overlap_" << k;
    }
    // Orthogonality across the set.
    EXPECT_NEAR(overlap_sq(s[0].psi, s[1].psi), 0.0, 1e-9);
    EXPECT_NEAR(overlap_sq(s[2].psi, s[5].psi), 0.0, 1e-9);
}

TEST(BoundStates1D, MorseSpectrumMatchesTheClosedForm) {
    // E_n = w0 (n + 1/2) - (alpha^2/2)(n + 1/2)^2, w0 = alpha sqrt(2 d):
    // the anharmonic ladder whose spacing SHRINKS toward dissociation.
    const ses::Grid1D g{-80.0, 80.0, 2048};
    const double d = 0.3;
    const double alpha = 0.12;
    const double x0 = -30.0;
    const std::vector<double> v = ses::morse_potential(g, d, alpha, x0);
    const std::vector<ses::Bound1D> s = ses::bound_states_1d(g, v, 8);
    const double w0 = alpha * std::sqrt(2.0 * d);
    int bound = 0;
    for (int n = 0; n < 8; ++n) {
        const double e = s[static_cast<std::size_t>(n)].energy;
        if (e < d - 2e-3) {
            const double exact =
                w0 * (n + 0.5) - 0.5 * alpha * alpha * (n + 0.5) * (n + 0.5);
            EXPECT_NEAR(e, exact, 5e-4) << "E_" << n;
            ++bound;
        }
    }
    EXPECT_EQ(bound, 6);  // nu = sqrt(2d)/alpha ~ 6.45 -> exactly 6 states
    // The anharmonic signature: the top gap is well under the bottom gap.
    const double gap0 = s[1].energy - s[0].energy;
    const double gap4 = s[5].energy - s[4].energy;
    EXPECT_LT(gap4, 0.6 * gap0);
}

TEST(DoubleWell1D, SplittingDrivesTheFullTunnelingOscillation) {
    // psi_L = (psi_0 + psi_1)/sqrt(2) starts left (P_L > 0.95) and lands
    // right (P_R > 0.85) after t = pi / dE -- the two-level physics of the
    // ammonia inversion, verified through the REAL split-step propagator.
    const ses::Grid1D g{-16.0, 16.0, 1024};
    const double vb = 0.12;
    const double a = 6.0;
    const std::vector<double> v = ses::double_well_potential(g, vb, a);
    const std::vector<ses::Bound1D> s = ses::bound_states_1d(g, v, 2);
    const double de = s[1].energy - s[0].energy;
    EXPECT_GT(de, 3e-3);  // params keep the demo in a sane band
    EXPECT_LT(de, 5e-2);
    EXPECT_LT(s[1].energy, vb);  // a genuine below-barrier doublet

    ses::Field1D psi{g};
    for (int i = 0; i < g.n; ++i) {
        psi[i] = (s[0].psi[i] + s[1].psi[i]);
    }
    ses::normalize(psi);
    EXPECT_GT(ses::probability_in_range(psi, g.xmin, 0.0), 0.95);

    const double dt = 0.04;
    const int steps = static_cast<int>(std::lround(3.14159265358979 / de / dt));
    ses::SplitOperator1D prop{g, v, dt};
    prop.step(psi, steps);
    EXPECT_GT(ses::probability_in_range(psi, 0.0, g.xmax), 0.85)
        << "after " << steps << " steps (dE = " << de << ")";
}

TEST(PoschlTeller1D, IsReflectionlessWhereTheSquareWellReflects) {
    // Same packet (E = 0.125), two attractive wells of EQUAL depth and
    // integrated area: the lambda = 2 sech^2 well transmits everything;
    // the sharp-edged square well reflects a visible fraction.
    const ses::Grid1D g{-60.0, 60.0, 2048};
    const double v0 = 0.375;  // lambda (lambda+1) / (2 a^2), lambda=2, a=2
    const double dt = 0.04;
    const int steps = 2250;  // t = 90 au: transit complete, nothing at walls
    auto reflected = [&](const std::vector<double>& v) {
        ses::Field1D psi = ses::gaussian_wavepacket(g, -25.0, 4.0, 0.5);
        ses::SplitOperator1D prop{g, v, dt};
        prop.step(psi, steps);
        return ses::probability_in_range(psi, g.xmin, -10.0);
    };
    const double r_pt =
        reflected(ses::poschl_teller_potential(g, v0, 2.0));
    const double r_sq =
        reflected(ses::barrier_potential(g, -v0, -2.0, 2.0));
    EXPECT_LT(r_pt, 5e-3) << "sech^2 must not reflect";
    EXPECT_GT(r_sq, 3e-2) << "the square well must reflect visibly";
    EXPECT_GT(r_sq, 10.0 * (r_pt + 1e-6));
}

}  // namespace
