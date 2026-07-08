// RED: orbital-free angular-decomposition projection.
//
// The tracked eigenstates factorize as |n> = (u_nl(r)/r) Y_lm(Omega) (soft-
// Coulomb is central). The direct grid inner product <n|psi> can therefore be
// REORGANIZED: deposit each cell's Y_lm(cell)*psi(cell)*dV into radial bins
// with a linear (cloud-in-cell) weight that folds in the 1/r Jacobian and the
// radial interpolation -> g_lm[c][j]; then <n|psi> = sum_j u_nl[j] g_lm[c][j],
// a cheap 1-D radial dot. The grid pass is done ONCE for all (l,m) up to l_max
// and is INDEPENDENT of the number of states -- so it needs no resident 3-D
// atlas and scales to a larger basis / 512^3.
//
// THE gate (this file): the reorganized amplitude must EQUAL the direct grid
// inner product <(u/r)Y_lm | psi> to ~1e-12 -- it is the same sum, merely
// reassociated. The reference orbital is built by fill_orbital (the un-
// normalized (u_interp/r)Y_lm the projection algebraically equals).

#include <core/projection.hpp>

#include <core/complex.hpp>
#include <core/field.hpp>
#include <core/grid.hpp>
#include <core/harmonics.hpp>
#include <core/radial.hpp>
#include <core/wavepacket.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

using ses::Grid1D;
using ses::Grid3D;
using ses::Vec3d;

// An arbitrary smooth radial u(r) with u(0)=0. The reorganization identity is
// algebraic -- it holds for ANY u (normalized or not, eigenstate or not),
// because amp*sqrt(norm2) reconstructs the raw dot regardless.
std::vector<double> make_u(const ses::RadialGrid& rg, int kind) {
    std::vector<double> u(static_cast<std::size_t>(rg.n));
    for (int i = 0; i < rg.n; ++i) {
        const double r = rg.r(i);
        u[static_cast<std::size_t>(i)] =
            (kind == 0) ? r * std::exp(-r) : r * r * std::exp(-0.5 * r);
    }
    return u;
}

TEST(RadialAngularProjection, ExactReorgIdentity) {
    const Grid1D ax{-10.0, 10.0, 48};
    const Grid3D g{ax, ax, ax};
    const ses::RadialGrid rg{10.0, 159};
    const std::vector<std::vector<double>> u_by_level = {make_u(rg, 0), make_u(rg, 1)};

    // A genuinely COMPLEX test state (nonzero k0 gives psi a phase).
    const ses::Field3D psi = ses::gaussian_wavepacket(
        g, Vec3d{1.2, 0.6, -0.4}, Vec3d{1.7, 1.7, 1.7}, Vec3d{0.3, -0.2, 0.1});

    // Cover l = 0 (worst near-origin), l = 1, and l = 5 (highest harmonic).
    const std::vector<ses::ProjectorState> states = {
        {0, 0, 0}, {1, 1, 0}, {1, 1, 1}, {1, 1, -1}, {0, 5, 3}, {1, 5, 0}};

    const ses::RadialAngularProjection proj =
        ses::project_radial_angular(psi, rg, u_by_level, states, 5);

    ASSERT_EQ(proj.amp.size(), states.size());
    for (std::size_t s = 0; s < states.size(); ++s) {
        const ses::ProjectorState& st = states[s];
        ses::Field3D orbital{g};
        ses::fill_orbital(orbital, g, rg,
                          u_by_level[static_cast<std::size_t>(st.level)], st.l, st.m);
        const ses::Complex<double> oracle = ses::inner_product(orbital, psi);
        // raw (un-normalized) amplitude = amp * sqrt(norm2) = sum_j u[j] g_lm[j].
        const ses::Complex<double> raw =
            proj.amp[s] * std::sqrt(proj.norm2[static_cast<std::size_t>(s)]);
        const double tol = 1e-11 * (1.0 + std::abs(oracle.real()) + std::abs(oracle.imag()));
        EXPECT_NEAR(raw.real(), oracle.real(), tol) << "state " << s;
        EXPECT_NEAR(raw.imag(), oracle.imag(), tol) << "state " << s;
    }
}

// The two boundary branches that most easily drift from fill_orbital: a cell at
// the exact origin (r < h) deposits to bin 0 only and only the l=0 harmonic
// (Y_{l>0}(0)=0); a cell beyond rmax is excluded entirely (the box-corner
// continuum, correctly part of the deficit).
TEST(RadialAngularProjection, OriginAndRmaxShape) {
    const Grid1D ax{-10.0, 10.0, 48};
    const Grid3D g{ax, ax, ax};
    const ses::RadialGrid rg{10.0, 159};
    const std::vector<std::vector<double>> u_by_level = {make_u(rg, 0)};
    const std::vector<ses::ProjectorState> states = {{0, 0, 0}, {0, 1, 0}, {0, 2, 0}};

    // (a) support only at the exact origin cell (coord(24) = 0, r = 0 < h).
    {
        ses::Field3D psi{g};
        psi(24, 24, 24) = ses::Complex<double>{1.0, 0.0};
        const ses::RadialAngularProjection proj =
            ses::project_radial_angular(psi, rg, u_by_level, states, 5);
        // l=0: only bin 0 receives (weight 1/h); every other bin is exactly 0.
        EXPECT_GT(std::abs(proj.g_lm[static_cast<std::size_t>(ses::lm_index(0, 0))][0]), 0.0);
        for (int jr = 1; jr < rg.n; ++jr) {
            EXPECT_EQ(std::abs(proj.g_lm[static_cast<std::size_t>(ses::lm_index(0, 0))]
                                       [static_cast<std::size_t>(jr)]),
                      0.0);
        }
        // all l>0 harmonics vanish at the origin -> those components are 0.
        for (int l = 1; l <= 5; ++l) {
            for (int m = -l; m <= l; ++m) {
                for (int jr = 0; jr < rg.n; ++jr) {
                    EXPECT_EQ(std::abs(proj.g_lm[static_cast<std::size_t>(ses::lm_index(l, m))]
                                               [static_cast<std::size_t>(jr)]),
                              0.0);
                }
            }
        }
    }
    // (b) support only beyond rmax (corner cell, r ~ 16.6 > 10): all amps zero.
    {
        ses::Field3D psi{g};
        psi(47, 47, 47) = ses::Complex<double>{1.0, 0.0};
        const ses::RadialAngularProjection proj =
            ses::project_radial_angular(psi, rg, u_by_level, states, 5);
        for (const ses::Complex<double>& a : proj.amp) {
            EXPECT_EQ(std::abs(a), 0.0);
        }
    }
}

// The angular part is exact analytically: projecting a pure Y_{1,0} orbital
// recovers the (1,0) amplitude and gives ~0 for the orthogonal (1,+-1) (exact
// by x/y symmetry) and a small grid-limited residual for a different l.
TEST(RadialAngularProjection, AngularOrthogonality) {
    const Grid1D ax{-10.0, 10.0, 48};
    const Grid3D g{ax, ax, ax};
    const ses::RadialGrid rg{10.0, 159};
    const std::vector<std::vector<double>> u_by_level = {make_u(rg, 0)};
    ses::Field3D psi{g};
    ses::fill_orbital(psi, g, rg, u_by_level[0], 1, 0);  // a real 2p_z profile
    const std::vector<ses::ProjectorState> states = {
        {0, 1, 0}, {0, 1, 1}, {0, 1, -1}, {0, 3, 0}};
    const ses::RadialAngularProjection proj =
        ses::project_radial_angular(psi, rg, u_by_level, states, 5);
    const double a10 = std::abs(proj.amp[0]);
    EXPECT_GT(a10, 1e-3);
    // Same-l cross-m orthogonality is near-exact (Y_1,+-1 _|_ Y_10 by x/y parity).
    EXPECT_LT(std::abs(proj.amp[1]), 1e-8 * a10);
    EXPECT_LT(std::abs(proj.amp[2]), 1e-8 * a10);
    // Different-l (Y_30 vs Y_10) is orthogonal only in the continuum; on a
    // coarse Cartesian grid the near-origin shells are angularly undersampled,
    // leaving a residual (~5% at 48^3). Crucially that residual is the GRID's,
    // NOT the projector's -- the projector reproduces the DIRECT grid inner
    // product exactly, so it is no worse than the resident-atlas path it
    // replaces (which would inner-product the same two grid orbitals).
    ses::Field3D o30{g};
    ses::fill_orbital(o30, g, rg, u_by_level[0], 3, 0);
    const ses::Complex<double> direct = ses::inner_product(o30, psi);
    const ses::Complex<double> raw3 = proj.amp[3] * std::sqrt(proj.norm2[3]);
    const double tol = 1e-11 * (1.0 + std::abs(direct.real()) + std::abs(direct.imag()));
    EXPECT_NEAR(raw3.real(), direct.real(), tol);
    EXPECT_NEAR(raw3.imag(), direct.imag(), tol);
}

// The core scaling claim: the grid pass (g_lm) is done ONCE and is INDEPENDENT
// of the number of states -- only the cheap 1-D radial dots scale with #states.
TEST(RadialAngularProjection, StateCountIndependence) {
    const Grid1D ax{-10.0, 10.0, 40};
    const Grid3D g{ax, ax, ax};
    const ses::RadialGrid rg{10.0, 159};
    const std::vector<std::vector<double>> u_by_level = {make_u(rg, 0), make_u(rg, 1)};
    const ses::Field3D psi = ses::gaussian_wavepacket(
        g, Vec3d{0.8, -0.5, 0.3}, Vec3d{1.6, 1.6, 1.6}, Vec3d{0.2, 0.1, -0.15});

    const std::vector<ses::ProjectorState> few = {{0, 1, 0}, {1, 2, 1}};
    std::vector<ses::ProjectorState> many = few;
    for (int l = 0; l <= 5; ++l) {
        for (int m = -l; m <= l; ++m) {
            many.push_back({0, l, m});
        }
    }
    const ses::RadialAngularProjection pf =
        ses::project_radial_angular(psi, rg, u_by_level, few, 5);
    const ses::RadialAngularProjection pm =
        ses::project_radial_angular(psi, rg, u_by_level, many, 5);
    // Identical g_lm (same one pass) regardless of the state list.
    for (int c = 0; c < ses::lm_count(5); ++c) {
        for (int jr = 0; jr < rg.n; ++jr) {
            EXPECT_EQ(pf.g_lm[static_cast<std::size_t>(c)][static_cast<std::size_t>(jr)],
                      pm.g_lm[static_cast<std::size_t>(c)][static_cast<std::size_t>(jr)]);
        }
    }
    // Overlapping amplitudes (few's states are many's first two) identical.
    for (std::size_t s = 0; s < few.size(); ++s) {
        EXPECT_EQ(pf.amp[s], pm.amp[s]);
    }
}

// Deterministic: fixed-order accumulation, no atomics/RNG -> two runs are
// bit-for-bit identical (the property the GPU kernel must preserve so gpucheck
// stays a clean bitwise-close comparison against this oracle).
TEST(RadialAngularProjection, DeterministicRepro) {
    const Grid1D ax{-9.0, 9.0, 40};
    const Grid3D g{ax, ax, ax};
    const ses::RadialGrid rg{9.0, 143};
    const std::vector<std::vector<double>> u_by_level = {make_u(rg, 0)};
    const ses::Field3D psi = ses::gaussian_wavepacket(
        g, Vec3d{0.5, 0.5, -0.5}, Vec3d{1.5, 1.5, 1.5}, Vec3d{0.1, -0.1, 0.2});
    const std::vector<ses::ProjectorState> states = {{0, 0, 0}, {0, 2, -1}, {0, 4, 2}};
    const ses::RadialAngularProjection a =
        ses::project_radial_angular(psi, rg, u_by_level, states, 5);
    const ses::RadialAngularProjection b =
        ses::project_radial_angular(psi, rg, u_by_level, states, 5);
    for (std::size_t s = 0; s < states.size(); ++s) {
        EXPECT_EQ(a.amp[s], b.amp[s]);
    }
    for (int c = 0; c < ses::lm_count(5); ++c) {
        for (int jr = 0; jr < rg.n; ++jr) {
            EXPECT_EQ(a.g_lm[static_cast<std::size_t>(c)][static_cast<std::size_t>(jr)],
                      b.g_lm[static_cast<std::size_t>(c)][static_cast<std::size_t>(jr)]);
        }
    }
}

}  // namespace
