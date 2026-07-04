#pragma once

// Gaussian wavepacket factory (atomic units):
//     psi(x) = (2 pi s^2)^(-1/4) exp(-(x-x0)^2 / (4 s^2)) exp(i k0 x)
// |psi|^2 is Gaussian(mean x0, std dev s); the exp(i k0 x) phase carries mean
// momentum k0. The result is normalized on the grid (the continuum amplitude
// is already unit-norm; a final discrete normalize absorbs sampling error).

#include <core/complex.hpp>
#include <core/field.hpp>
#include <core/grid.hpp>

#include <cmath>
#include <numbers>

namespace ses {

inline Field1D gaussian_wavepacket(const Grid1D& g, double x0, double sigma, double k0) {
    Field1D psi{g};
    const double amp = std::pow(2.0 * std::numbers::pi * sigma * sigma, -0.25);
    for (int i = 0; i < psi.size(); ++i) {
        const double x = g.coord(i);
        const double envelope = amp * std::exp(-(x - x0) * (x - x0) / (4.0 * sigma * sigma));
        psi[i] = Complex<double>{envelope * std::cos(k0 * x), envelope * std::sin(k0 * x)};
    }
    normalize(psi);
    return psi;
}

}  // namespace ses
