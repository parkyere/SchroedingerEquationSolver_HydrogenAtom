#pragma once

// Soft position measurement (Gaussian POVM): the electron is "observed" not
// into a Dirac delta but into a sharp-with-Gaussian-spread packet that the
// TDSE then re-evolves.
//
// Randomness stays OUT of core: callers supply a uniform draw u in [0,1) and
// sample_collapse_index inverts the discrete CDF of |psi|^2 in flat-index
// order (the cell-volume factor is uniform and cancels).

#include <core/complex.hpp>
#include <core/field.hpp>
#include <core/grid.hpp>
#include <core/vec.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace ses {

// Projective energy measurement over eigenstate populations P_n =
// |<phi_n|psi>|^2. Because the tracked bound manifold is incomplete (no
// continuum states), the P_n sum to <= 1; the deficit 1 - sum(P) is the
// probability of an "outside the tracked manifold" outcome. Given a uniform
// draw u in [0,1), returns the collapsed eigenstate index n (psi should then
// be projected onto phi_n), or -1 for the outside outcome (leave psi as is).
// CDF inversion in index order; randomness stays with the caller.
inline int sample_energy_eigenstate(const std::vector<double>& populations, double u) {
    double cum = 0.0;
    for (std::size_t n = 0; n < populations.size(); ++n) {
        cum += populations[n];
        if (u < cum) {
            return static_cast<int>(n);
        }
    }
    return -1;  // fell into the 1 - sum(P) deficit: continuum / untracked
}

// First flat index whose cumulative probability exceeds u * total.
inline int sample_collapse_index(const Field3D& psi, double u) {
    double total = 0.0;
    for (const Complex<double>& z : psi.data()) {
        total += norm_sq(z);
    }
    const double target = u * total;
    double cum = 0.0;
    const std::size_t n = psi.data().size();
    for (std::size_t i = 0; i < n; ++i) {
        cum += norm_sq(psi.data()[i]);
        if (cum > target) {
            return static_cast<int>(i);
        }
    }
    return static_cast<int>(n - 1);  // u ~ 1 with rounding: last occupied cell
}

// psi <- psi * exp(-|r - center|^2 / (4 sigma_m^2)), renormalized. Same
// amplitude convention as gaussian_wavepacket, so Gaussian x Gaussian
// posteriors are analytic.
inline void collapse_wavepacket(Field3D& psi, Vec3d center, double sigma_m) {
    const Grid3D& g = psi.grid();
    const double inv4s2 = 1.0 / (4.0 * sigma_m * sigma_m);
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double dx = g.x.coord(i) - center.x;
                const double dy = g.y.coord(j) - center.y;
                const double dz = g.z.coord(k) - center.z;
                const double mask = std::exp(-(dx * dx + dy * dy + dz * dz) * inv4s2);
                psi(i, j, k) = mask * psi(i, j, k);
            }
        }
    }
    normalize(psi);
}

}  // namespace ses
