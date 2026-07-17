module;
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>
export module ses.ladder;
export import ses.field;
export import ses.grid;
import ses.fft;
import ses.spectral;


// 1D harmonic-oscillator ladder operators (atomic units, m = hbar = 1):
//
//     a    = sqrt(omega/2) x + 1/sqrt(2 omega) d/dx
//     adag = sqrt(omega/2) x - 1/sqrt(2 omega) d/dx
//
// acting on eigenstates: adag|n> = sqrt(n+1)|n+1>, a|n> = sqrt(n)|n-1>,
// a|0> = 0. The derivative is spectral (FFT ik multiply), consistent with
// the split-operator periodic grid, so repeated applications climb/descend
// the Fock chain cleanly to near machine epsilon.
//
// Both entry points return ||O psi||^2 BEFORE the internal renormalization
// (the sqrt(n+1) / sqrt(n) counting weight squared) and leave psi normalized.
// ladder_lower is guarded: ||a psi||^2 = <N> can vanish (a|0> = 0), in which
// case psi is left UNCHANGED and the tiny return value is the caller's
// forbidden-transition signal. ladder_raise needs no guard:
// ||adag psi||^2 = <N> + 1 >= 1 on any normalized state.


export namespace ses {

namespace ladder_detail {

// (sqrt(omega/2) x + deriv_sign / sqrt(2 omega) d/dx) psi, spectral d/dx.
inline std::vector<std::complex<double>> apply(const Field1D& psi, double omega,
                                               double deriv_sign) {
    const Grid1D& g = psi.grid();
    std::vector<std::complex<double>> dpsi = psi.data();
    fft(dpsi);
    const std::vector<double> k = wavenumbers(g);
    for (std::size_t j = 0; j < dpsi.size(); ++j) {
        dpsi[j] *= std::complex<double>{0.0, k[j]};
    }
    ifft(dpsi);

    const double cx = std::sqrt(0.5 * omega);
    const double cd = deriv_sign / std::sqrt(2.0 * omega);
    std::vector<std::complex<double>> out(dpsi.size());
    for (int i = 0; i < psi.size(); ++i) {
        const std::size_t s = static_cast<std::size_t>(i);
        out[s] = cx * g.coord(i) * psi[i] + cd * dpsi[s];
    }
    return out;
}

// ||v||^2 with the grid weight h (the Field1D norm convention).
inline double norm_sq_h(const std::vector<std::complex<double>>& v, double h) noexcept {
    double acc = 0.0;
    for (const std::complex<double>& z : v) {
        acc += std::norm(z);
    }
    return acc * h;
}

// Normalize `v` into psi's storage.
inline void store_normalized(Field1D& psi, const std::vector<std::complex<double>>& v,
                             double norm2) noexcept {
    const double inv = 1.0 / std::sqrt(norm2);
    for (int i = 0; i < psi.size(); ++i) {
        psi[i] = inv * v[static_cast<std::size_t>(i)];
    }
}

}  // namespace ladder_detail

// adag: psi <- adag psi / ||adag psi||; returns ||adag psi||^2 (n+1 on |n>).
inline double ladder_raise(Field1D& psi, double omega) {
    const std::vector<std::complex<double>> out =
        ladder_detail::apply(psi, omega, -1.0);
    const double norm2 = ladder_detail::norm_sq_h(out, psi.grid().spacing());
    ladder_detail::store_normalized(psi, out, norm2);
    return norm2;
}

// a: psi <- a psi / ||a psi|| unless annihilated (||a psi||^2 < vanish_eps,
// e.g. the ground state), in which case psi is untouched; returns ||a psi||^2
// (n on |n>, <N> in general) either way.
inline double ladder_lower(Field1D& psi, double omega, double vanish_eps = 1e-6) {
    const std::vector<std::complex<double>> out =
        ladder_detail::apply(psi, omega, +1.0);
    const double norm2 = ladder_detail::norm_sq_h(out, psi.grid().spacing());
    if (norm2 < vanish_eps) {
        return norm2;
    }
    ladder_detail::store_normalized(psi, out, norm2);
    return norm2;
}

}  // namespace ses
