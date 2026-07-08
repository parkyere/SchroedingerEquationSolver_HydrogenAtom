#pragma once

// Orbital-free angular-decomposition projection of a wavefunction onto the
// tracked eigenstate manifold -- computes every amplitude <n|psi> WITHOUT
// holding a resident 3-D orbital atlas.
//
// Because the soft-Coulomb potential is central, each eigenstate factorizes
// exactly as |n> = (u_nl(r)/r) Y_lm(Omega). The direct grid inner product
//     <n|psi> = sum_cells (u_interp(r)/r) Y_lm(cell) psi(cell) dV
// is therefore just a reorganization: deposit the CELL-side factors
// Y_lm(cell) psi(cell) dV into radial bins with a weight that folds in the 1/r
// Jacobian and the SAME linear (cloud-in-cell) radial interpolation that
// synthesize_orbital/fill_orbital use, giving
//     g_lm[c][j] = sum_cells W_j(r_cell) Y_lm(cell) psi(cell) dV,
// after which each amplitude is a cheap 1-D radial dot
//     <n|psi> = sum_j u_nl[j] g_lm[lm(l,m)][j].
// The grid pass is done ONCE for all (l,m) up to l_max and is INDEPENDENT of
// the number of states -- so it needs no 3-D atlas and scales to a larger
// basis / a 512^3 grid (where a resident atlas is physically impossible).
//
// This is the pure CPU-double reference (the oracle the GPU deposit kernel is
// verified against). The deposit shape W_j(r) MUST mirror fill_orbital
// (core/harmonics.hpp) exactly, so u_over_r(r) = sum_j u[j] W_j(r) and the
// reorganized sum equals the direct inner product to rounding.

#include <core/complex.hpp>
#include <core/field.hpp>
#include <core/grid.hpp>
#include <core/harmonics.hpp>  // real_spherical_harmonic
#include <core/radial.hpp>     // RadialGrid

#include <cmath>
#include <cstddef>
#include <vector>

namespace ses {

// A tracked state's generative seed: which radial level (index into the
// caller's u tables) and its angular quantum numbers. NO 3-D orbital.
struct ProjectorState {
    int level;
    int l;
    int m;
};

// Flat index of a real harmonic (l, m): l in [0, l_max], m in [-l, l].
inline int lm_index(int l, int m) { return l * l + (l + m); }
inline int lm_count(int l_max) { return (l_max + 1) * (l_max + 1); }

struct RadialAngularProjection {
    std::vector<Complex<double>> amp;               // <n|psi>, unit-normalized
    std::vector<double> norm2;                      // N_n = sum_j u[j]^2 h (=1 for eigen-u)
    std::vector<std::vector<Complex<double>>> g_lm;  // [lm_count][n_radial], built once
    // amp[n] = (sum_j u[j] g_lm[lm(n)][j]) / sqrt(N_n);
    // the RAW amplitude (== the direct grid inner product) = amp[n] * sqrt(N_n).
};

// One grid pass over psi -> g_lm; then a 1-D radial dot per state -> amplitudes.
inline RadialAngularProjection project_radial_angular(
    const Field3D& psi, const RadialGrid& rgrid,
    const std::vector<std::vector<double>>& u_by_level,
    const std::vector<ProjectorState>& states, int l_max = 5) {
    const Grid3D& g = psi.grid();
    const int nx = g.x.n;
    const int ny = g.y.n;
    const int nz = g.z.n;
    const double h = rgrid.h();
    const double rmax = rgrid.rmax;
    const int nr = rgrid.n;
    const double dV = g.cell_volume();
    const int ncomp = lm_count(l_max);

    RadialAngularProjection out;
    out.g_lm.assign(static_cast<std::size_t>(ncomp),
                    std::vector<Complex<double>>(static_cast<std::size_t>(nr),
                                                 Complex<double>{}));

    // --- the deposit: Y_lm(cell) psi(cell) dV into radial bins, 1/r Jacobian
    //     folded into the linear cloud-in-cell weight (mirrors fill_orbital) ---
    for (int k = 0; k < nz; ++k) {
        const double z = g.z.coord(k);
        for (int j = 0; j < ny; ++j) {
            const double y = g.y.coord(j);
            for (int i = 0; i < nx; ++i) {
                const double x = g.x.coord(i);
                const double r = std::sqrt(x * x + y * y + z * z);
                if (r >= rmax) {
                    continue;  // outside the radial box -> no deposit (deficit)
                }
                // Deposit bins/weights, exactly the fill_orbital branches:
                //   r < h        : bin 0, weight 1/h (the u[0]/h origin limit);
                //   h <= r < rmax: linear tent /r into i0 and (if present) i0+1;
                int b0 = 0;
                int b1 = -1;
                double w0 = 0.0;
                double w1 = 0.0;
                if (r < h) {
                    b0 = 0;
                    w0 = 1.0 / h;
                } else {
                    const double t = r / h - 1.0;  // r_i = (i+1) h
                    const int i0 = static_cast<int>(t);
                    const double frac = t - static_cast<double>(i0);
                    b0 = i0;
                    w0 = (1.0 - frac) / r;
                    if (i0 + 1 < nr) {
                        b1 = i0 + 1;
                        w1 = frac / r;  // outermost node = pinned u(rmax)=0: dropped
                    }
                }
                const Complex<double> pdV = psi(i, j, k) * dV;
                for (int l = 0; l <= l_max; ++l) {
                    for (int m = -l; m <= l; ++m) {
                        const double Y = real_spherical_harmonic(l, m, x, y, z);
                        const Complex<double> contrib = pdV * Y;
                        const std::size_t c = static_cast<std::size_t>(lm_index(l, m));
                        out.g_lm[c][static_cast<std::size_t>(b0)] += contrib * w0;
                        if (b1 >= 0) {
                            out.g_lm[c][static_cast<std::size_t>(b1)] += contrib * w1;
                        }
                    }
                }
            }
        }
    }

    // --- per-state amplitudes: raw = sum_j u[j] g_lm[lm][j]; norm = sum u^2 h ---
    out.amp.assign(states.size(), Complex<double>{});
    out.norm2.assign(states.size(), 0.0);
    for (std::size_t s = 0; s < states.size(); ++s) {
        const ProjectorState& st = states[s];
        const std::vector<double>& u =
            u_by_level[static_cast<std::size_t>(st.level)];
        const std::vector<Complex<double>>& gc =
            out.g_lm[static_cast<std::size_t>(lm_index(st.l, st.m))];
        Complex<double> raw{};
        double n2 = 0.0;
        for (int jr = 0; jr < nr; ++jr) {
            raw += u[static_cast<std::size_t>(jr)] * gc[static_cast<std::size_t>(jr)];
            n2 += u[static_cast<std::size_t>(jr)] * u[static_cast<std::size_t>(jr)] * h;
        }
        out.norm2[s] = n2;
        out.amp[s] = (n2 > 0.0) ? raw * (1.0 / std::sqrt(n2)) : Complex<double>{};
    }
    return out;
}

}  // namespace ses
