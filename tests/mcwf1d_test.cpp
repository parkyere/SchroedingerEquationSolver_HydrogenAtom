// RED: the 1D photon-loss MCWF step. A forced jump on the even cat
// |a> + |-a> lands on the ODD cat (parity flip -- the fringe inversion
// per lost photon); the no-jump conditional damping bleeds <n> at
// exactly kappa (coherent states stay coherent, alpha -> alpha
// e^{-kappa t/2}).

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

import ses.mcwf1d;
import ses.observables;
import ses.propagator;
import ses.wavepacket;

namespace {

ses::Field1D cat(const ses::Grid1D& g, double omega, double x0, int sign) {
    const double sig = 1.0 / std::sqrt(2.0 * omega);
    ses::Field1D a = ses::gaussian_wavepacket(g, x0, sig, 0.0);
    ses::Field1D b = ses::gaussian_wavepacket(g, -x0, sig, 0.0);
    for (int i = 0; i < g.n; ++i) {
        a[i] += static_cast<double>(sign) * b[i];
    }
    ses::normalize(a);
    return a;
}

double overlap2(const ses::Field1D& a, const ses::Field1D& b) {
    std::complex<double> ov{};
    for (int i = 0; i < a.size(); ++i) {
        ov += std::conj(a[i]) * b[i];
    }
    ov *= a.grid().spacing();
    return std::norm(ov);
}

TEST(Mcwf1d, JumpFlipsTheCatParity) {
    const ses::Grid1D g{-12.0, 12.0, 512};
    const double omega = 1.0;
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        const double x = g.coord(i);
        v[static_cast<std::size_t>(i)] = 0.5 * omega * omega * x * x;
    }
    const ses::ImaginaryTimePropagator1D damp{g, v, 1e-4};
    ses::Field1D psi = cat(g, omega, 3.0, +1);
    // u = 0 forces the jump (<n> ~ alpha^2 = 4.5 > 0).
    const bool jumped =
        ses::photon_loss_step(psi, omega, v, 0.05, 0.01, 0.0, damp);
    EXPECT_TRUE(jumped);
    EXPECT_GT(overlap2(psi, cat(g, omega, 3.0, -1)), 0.9);   // odd cat
    EXPECT_LT(overlap2(psi, cat(g, omega, 3.0, +1)), 0.05);  // even gone
}

TEST(Mcwf1d, NoJumpDampingBleedsNAtKappa) {
    const ses::Grid1D g{-12.0, 12.0, 512};
    const double omega = 1.0;
    const double kappa = 0.5;
    const double dt = 0.01;
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        const double x = g.coord(i);
        v[static_cast<std::size_t>(i)] = 0.5 * omega * omega * x * x;
    }
    const ses::SplitOperator1D prop{g, v, dt};
    const ses::ImaginaryTimePropagator1D damp{g, v,
                                              kappa * dt / (2.0 * omega)};
    const double sig = 1.0 / std::sqrt(2.0 * omega);
    ses::Field1D psi = ses::gaussian_wavepacket(g, 3.0, sig, 0.0);
    auto n_of = [&](const ses::Field1D& f) {
        return ses::mean_energy(f, v) / omega - 0.5;
    };
    const double n0 = n_of(psi);
    ASSERT_GT(n0, 4.0);  // alpha^2 = x0^2 omega / 2 = 4.5
    for (int s = 0; s < 400; ++s) {  // T = 4, u = 1: never jump
        prop.step(psi, 1);
        ses::photon_loss_step(psi, omega, v, kappa, dt, 1.0, damp);
    }
    const double ratio = n_of(psi) / n0;
    const double pred = std::exp(-kappa * 4.0);  // e^-2 = 0.135
    EXPECT_NEAR(ratio, pred, 0.15 * pred + 0.02);
}

}  // namespace
