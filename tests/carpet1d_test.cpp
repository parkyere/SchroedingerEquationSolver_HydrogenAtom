// RED: the quantum-carpet ring contracts. T_rev = L^2 / pi; the free
// ring packet revives EXACTLY there (spectral dispersion is exact, no
// walls to smear it), is nowhere near revived at an incommensurate time,
// and at T/2 reappears as the clone displaced by half the ring.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

import ses.scenario.carpet1d_director;
import ses.field;
import ses.grid;
import ses.propagator;
import ses.wavepacket;

namespace {

double overlap2(const ses::Field1D& a, const ses::Field1D& b) {
    std::complex<double> ov{};
    for (int i = 0; i < a.size(); ++i) {
        ov += std::conj(a[i]) * b[i];
    }
    ov *= a.grid().spacing();
    return std::norm(ov);
}

TEST(Carpet1d, RingRevivesExactlyAtLSquaredOverPi) {
    const double half = 20.0;
    const ses::Grid1D g{-half, half, 1024};
    const double t_rev = ses_shell::carpet_revival_time(2.0 * half);
    EXPECT_NEAR(t_rev, 1600.0 / 3.14159265358979323846, 1e-9);
    const std::vector<double> zero(static_cast<std::size_t>(g.n), 0.0);
    const double dt = 0.05;
    const ses::SplitOperator1D prop{g, zero, dt};
    const ses::Field1D psi0 = ses::gaussian_wavepacket(g, -5.0, 2.0, 1.0);
    ses::Field1D psi = psi0;
    const int n_mid = static_cast<int>(0.37 * t_rev / dt + 0.5);
    prop.step(psi, n_mid);
    EXPECT_LT(overlap2(psi, psi0), 0.5);  // scrambled mid-carpet...
    const int n_rest =
        static_cast<int>(t_rev / dt + 0.5) - n_mid;
    prop.step(psi, n_rest);
    EXPECT_GT(overlap2(psi, psi0), 0.99);  // ...full revival at T_rev
}

TEST(Carpet1d, HalfRevivalIsTheHalfRingClone) {
    const double half = 20.0;
    const ses::Grid1D g{-half, half, 1024};
    const double t_rev = ses_shell::carpet_revival_time(2.0 * half);
    const std::vector<double> zero(static_cast<std::size_t>(g.n), 0.0);
    const double dt = 0.05;
    const ses::SplitOperator1D prop{g, zero, dt};
    const ses::Field1D psi0 = ses::gaussian_wavepacket(g, -5.0, 2.0, 1.0);
    // The expected half-Talbot image: psi0 displaced by L/2 around the
    // ring (launch at -5 -> clone at +15).
    ses::Field1D clone{g};
    for (int i = 0; i < g.n; ++i) {
        const int j = (i + g.n / 2) % g.n;
        clone[i] = psi0[j];
    }
    ses::Field1D psi = psi0;
    prop.step(psi, static_cast<int>(0.5 * t_rev / dt + 0.5));
    EXPECT_GT(overlap2(psi, clone), 0.99);
    EXPECT_LT(overlap2(psi, psi0), 0.05);  // NOT at the launch site
}

}  // namespace
