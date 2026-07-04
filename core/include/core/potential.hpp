#pragma once

// Potential builders: real-valued V sampled on the grid, fed to the
// split-operator propagator.
//
// The soft-Coulomb regularization -Z/sqrt(dx^2 + a^2) is the standard 1D
// stand-in for the singular -Z/|x|: finite everywhere (deepest value -Z/a at
// the nucleus), so a grid point may sit exactly on the nucleus.

#include <core/grid.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace ses {

// V(x) = 1/2 omega^2 (x - x0)^2
inline std::vector<double> harmonic_potential(const Grid1D& g, double omega, double x0 = 0.0) {
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        const double dx = g.coord(i) - x0;
        v[static_cast<std::size_t>(i)] = 0.5 * omega * omega * dx * dx;
    }
    return v;
}

// V(x) = -Z / sqrt((x - x0)^2 + a^2)
inline std::vector<double> soft_coulomb_potential(const Grid1D& g, double Z, double a,
                                                  double x0 = 0.0) {
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        const double dx = g.coord(i) - x0;
        v[static_cast<std::size_t>(i)] = -Z / std::sqrt(dx * dx + a * a);
    }
    return v;
}

}  // namespace ses
