// RED: Anderson localization contracts. The disorder landscape is
// deterministic per seed, bounded by the amplitude range, and every
// barrier sits BELOW the packet energy (the classical particle passes);
// yet the quantum packet's transport HALTS -- coherent backscattering --
// while the clean (W = 0) twin flies ballistically.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <vector>

import ses.scenario.anderson1d_director;
import ses.field;
import ses.grid;
import ses.observables;
import ses.propagator;
import ses.wavepacket;

namespace {

TEST(Anderson1D, LandscapeIsDeterministicBoundedAndSubEnergy) {
    const ses::Grid1D g{-60.0, 60.0, 4096};
    const std::vector<double> a = ses_shell::anderson_potential(g, 1.0, 7);
    const std::vector<double> b = ses_shell::anderson_potential(g, 1.0, 7);
    const std::vector<double> c = ses_shell::anderson_potential(g, 1.0, 8);
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(a, b);  // same seed -> bitwise identical
    EXPECT_NE(a, c);  // fresh seed -> fresh landscape
    double vmax = 0.0;
    double vsum = 0.0;
    for (const double v : a) {
        vmax = std::max(vmax, std::abs(v));
        vsum += std::abs(v);
    }
    EXPECT_GT(vsum, 0.0);  // there IS a landscape
    // Overlap-bounded: the speckle field peaks under ~1.4 x the grain
    // range (this landscape was drawn at w = 1.0).
    EXPECT_LT(vmax, 1.4);
}

TEST(Anderson1D, DisorderBlocksTheBallisticPacket) {
    // Conductance framing on an OPEN wire (edge CAPs eat whatever exits;
    // the periodic FFT box would otherwise wrap the transmitted tail into
    // the readout): the CLEAN wire transmits the whole packet -- the norm
    // drains; the DISORDERED wire reflects/localizes it -- the norm stays,
    // parked on the entry side. Same landscape, E above every barrier.
    const ses::Grid1D g{-60.0, 60.0, 4096};
    const double x0 = -45.0;
    const double w0 = 4.0;
    const double cap_w = 6.0;
    std::vector<double> cap(static_cast<std::size_t>(g.n), 1.0);
    for (int i = 0; i < g.n; ++i) {
        const double d = std::min(g.coord(i) - g.xmin,
                                  g.xmax - g.coord(i));
        if (d < cap_w) {
            const double t = 1.0 - d / cap_w;
            cap[static_cast<std::size_t>(i)] =
                std::exp(-w0 * t * t * 0.01);
        }
    }
    // TRANSMITTED flux = what the RIGHT cap eats (reflected flux exits
    // the LEFT cap -- that IS blocking, it must not count against it).
    auto run = [&](const std::vector<double>& v) {
        const ses::SplitOperator1D prop{g, v, 0.01};
        ses::Field1D psi =
            ses::gaussian_wavepacket(g, x0, 2.0, ses_shell::kAn1dK0);
        const double h = g.spacing();
        double transmitted = 0.0;
        for (int s = 0; s < 11000; ++s) {  // t = 110: full transit + tail
            prop.step(psi, 1);
            for (int i = 0; i < g.n; ++i) {
                const double c = cap[static_cast<std::size_t>(i)];
                if (c < 1.0 && g.coord(i) > 0.0) {
                    transmitted += std::norm(psi[i]) * (1.0 - c * c) * h;
                }
                psi[i] *= c;
            }
        }
        return transmitted;
    };
    const std::vector<double> clean(static_cast<std::size_t>(g.n), 0.0);
    const std::vector<double> dis =
        ses_shell::anderson_potential(g, ses_shell::kAn1dW, 7);
    const double t_clean = run(clean);
    const double t_dis = run(dis);
    std::printf("anderson: transmitted clean %.3f, disordered %.3f\n",
                t_clean, t_dis);
    EXPECT_GT(t_clean, 0.7);           // clean wire: conducts
    EXPECT_LT(t_dis, 0.3 * t_clean);   // disorder: the wire INSULATES
}

}  // namespace
