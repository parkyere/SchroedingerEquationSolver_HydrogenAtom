#pragma once

// Expectation values over a Field1D. All observables are scale-invariant
// (they divide by the discrete norm), so they are valid on unnormalized
// fields as well.

#include <core/complex.hpp>
#include <core/fft.hpp>
#include <core/field.hpp>
#include <core/spectral.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace ses {

// <x> = sum x_i |psi_i|^2 / sum |psi_i|^2   (grid weight h cancels)
inline double mean_position(const Field1D& f) {
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < f.size(); ++i) {
        const double w = norm_sq(f[i]);
        num += f.grid().coord(i) * w;
        den += w;
    }
    return num / den;
}

// sigma_x = sqrt(<x^2> - <x>^2)
inline double sigma_x(const Field1D& f) {
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < f.size(); ++i) {
        const double x = f.grid().coord(i);
        const double w = norm_sq(f[i]);
        num += x * x * w;
        den += w;
    }
    const double mean = mean_position(f);
    return std::sqrt(num / den - mean * mean);
}

// <p> = sum k_j |phi_j|^2 / sum |phi_j|^2 with phi = fft(psi)  (weights cancel)
inline double mean_momentum(const Field1D& f) {
    std::vector<Complex<double>> phi(static_cast<std::size_t>(f.size()));
    for (int i = 0; i < f.size(); ++i) {
        phi[static_cast<std::size_t>(i)] = f[i];
    }
    fft(phi);
    const std::vector<double> k = wavenumbers(f.grid());
    double num = 0.0;
    double den = 0.0;
    for (std::size_t j = 0; j < phi.size(); ++j) {
        const double w = norm_sq(phi[j]);
        num += k[j] * w;
        den += w;
    }
    return num / den;
}

}  // namespace ses
