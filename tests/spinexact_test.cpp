// RED: the exact 4x4 Heisenberg contracts. A product boot round-trips
// the per-site Bloch vectors; the all-up ferromagnet is an eigenstate
// (arrows frozen at +z); a single flipped spin's magnon HOPS while total
// sigma_z is conserved; the Neel product ENTANGLES (arrows shrink --
// the thing the mean-field ansatz cannot do); energy is conserved and
// the norm stays round-off.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

import ses.spinexact;

namespace {

ses::SpinLattice product_updown(int flip_site) {
    ses::SpinLattice l;
    l.nx = ses::kExactSide;
    l.ny = ses::kExactSide;
    l.s.assign(ses::kExactSites, ses::spinor_from_bloch(0.0, 0.0, 1.0));
    if (flip_site >= 0) {
        l.s[static_cast<std::size_t>(flip_site)] =
            ses::spinor_from_bloch(0.0, 0.0, -1.0);
    }
    return l;
}

double norm2(const ses::SpinState16& s) {
    double n = 0.0;
    for (const auto& z : s.c) {
        n += std::norm(z);
    }
    return n;
}

double total_sz(const ses::SpinState16& s) {
    double t = 0.0;
    for (int i = 0; i < ses::kExactSites; ++i) {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        ses::exact_site_bloch(s, i, &x, &y, &z);
        t += z;
    }
    return t;
}

TEST(SpinExact, ProductBootRoundTripsTheArrows) {
    ses::SpinLattice l;
    l.nx = ses::kExactSide;
    l.ny = ses::kExactSide;
    l.s.resize(ses::kExactSites);
    for (int i = 0; i < ses::kExactSites; ++i) {
        const double th = 0.3 + 0.11 * i;
        const double ph = 0.7 * i;
        l.s[static_cast<std::size_t>(i)] = ses::spinor_from_bloch(
            std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph),
            std::cos(th));
    }
    const ses::SpinState16 s = ses::exact_from_product(l);
    ASSERT_EQ(s.c.size(), ses::kExactDim);
    EXPECT_NEAR(norm2(s), 1.0, 1e-12);
    for (int i = 0; i < ses::kExactSites; ++i) {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        ses::exact_site_bloch(s, i, &x, &y, &z);
        double lx = 0.0;
        double ly = 0.0;
        double lz = 0.0;
        ses::bloch_vector(l.s[static_cast<std::size_t>(i)], &lx, &ly,
                          &lz);
        EXPECT_NEAR(x, lx, 1e-12);
        EXPECT_NEAR(y, ly, 1e-12);
        EXPECT_NEAR(z, lz, 1e-12);
    }
}

TEST(SpinExact, FerroIsStationaryAndMagnonHopsConservingSz) {
    ses::SpinState16 up = ses::exact_from_product(product_updown(-1));
    for (int k = 0; k < 200; ++k) {
        ses::exact_step(up, 0.0, 0.0, 0.4, 0.5, 0.01);
    }
    for (int i = 0; i < ses::kExactSites; ++i) {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        ses::exact_site_bloch(up, i, &x, &y, &z);
        EXPECT_NEAR(z, 1.0, 1e-9);  // all-up: an exact eigenstate
    }

    ses::SpinState16 mg = ses::exact_from_product(product_updown(5));
    const double sz0 = total_sz(mg);
    double z5_min = 1.0;
    double z_other_min = 1.0;
    for (int k = 0; k < 400; ++k) {
        ses::exact_step(mg, 0.0, 0.0, 0.0, 0.5, 0.01);
    }
    EXPECT_NEAR(total_sz(mg), sz0, 1e-9);  // sum sigma_z conserved
    double x = 0.0;
    double y = 0.0;
    double z5 = 0.0;
    ses::exact_site_bloch(mg, 5, &x, &y, &z5);
    EXPECT_GT(z5, -0.9);  // the flipped spin delocalized...
    double zmin = 1.0;
    for (int i = 0; i < ses::kExactSites; ++i) {
        if (i == 5) {
            continue;
        }
        double zz = 0.0;
        ses::exact_site_bloch(mg, i, &x, &y, &zz);
        zmin = std::min(zmin, zz);
    }
    EXPECT_LT(zmin, 0.99);  // ...into the neighbors (the magnon hopped)
    (void)z5_min;
    (void)z_other_min;
}

TEST(SpinExact, NeelEntanglesArrowsShrinkEnergyConserved) {
    ses::SpinLattice l;
    l.nx = ses::kExactSide;
    l.ny = ses::kExactSide;
    l.s.resize(ses::kExactSites);
    for (int yy = 0; yy < ses::kExactSide; ++yy) {
        for (int xx = 0; xx < ses::kExactSide; ++xx) {
            const double sgn = ((xx + yy) & 1) != 0 ? -1.0 : 1.0;
            l.s[static_cast<std::size_t>(yy * ses::kExactSide + xx)] =
                ses::spinor_from_bloch(0.0, 0.0, sgn);
        }
    }
    ses::SpinState16 s = ses::exact_from_product(l);
    const double e0 = ses::exact_energy(s, 0.0, 0.0, 0.1, 0.5);
    ASSERT_NE(e0, 0.0);
    for (int k = 0; k < 500; ++k) {
        ses::exact_step(s, 0.0, 0.0, 0.1, 0.5, 0.01);
    }
    EXPECT_NEAR(norm2(s), 1.0, 1e-10);
    EXPECT_NEAR(ses::exact_energy(s, 0.0, 0.0, 0.1, 0.5), e0,
                1e-3 * std::abs(e0) + 1e-3);
    double mean_len = 0.0;
    for (int i = 0; i < ses::kExactSites; ++i) {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        ses::exact_site_bloch(s, i, &x, &y, &z);
        mean_len += std::sqrt(x * x + y * y + z * z);
    }
    mean_len /= ses::kExactSites;
    EXPECT_LT(mean_len, 0.8);  // ENTANGLED: the product ansatz cannot
}

TEST(SpinExact, MeasurementCollapsesASite) {
    ses::SpinLattice l = product_updown(-1);
    l.s[3] = ses::spinor_from_bloch(1.0, 0.0, 0.0);  // site 3 on +x
    ses::SpinState16 s = ses::exact_from_product(l);
    const int out = ses::exact_measure_z(s, 3, 0.4);  // p_up = 1/2
    EXPECT_NE(out, 0);
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    ses::exact_site_bloch(s, 3, &x, &y, &z);
    EXPECT_NEAR(z, static_cast<double>(out), 1e-12);  // eigenstate now
    EXPECT_NEAR(norm2(s), 1.0, 1e-12);
    // The basis-rotation helper: rotate site 3 by pi about y maps
    // +-z -> -+z (up to phase).
    ses::exact_site_rotate(s, 3, 0.0, 1.0, 0.0,
                           3.14159265358979323846);
    ses::exact_site_bloch(s, 3, &x, &y, &z);
    EXPECT_NEAR(z, -static_cast<double>(out), 1e-12);
}

}  // namespace
