module;
#include <cmath>
#include <complex>
#include <cstddef>
#include <utility>
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

// The exact HO eigenstate |n> built DIRECTLY from the normalized
// Hermite-Gauss recurrence in x-space (atomic units, m = hbar = 1):
//     psi_0 = (omega/pi)^{1/4} exp(-omega x^2 / 2)
//     psi_1 = sqrt(2 omega) x psi_0
//     psi_{k} = sqrt(2 omega / k) x psi_{k-1} - sqrt((k-1)/k) psi_{k-2}
// The NORMALIZED recurrence keeps every intermediate O(1) (no 2^n n!
// overflow) and -- crucially -- uses NO derivative, so unlike the ladder
// chain it suffers no spectral round-off amplification: it is exact to
// round-off for every level the grid can represent. A final discrete
// normalize absorbs the sampling error. This is the ground-truth oracle
// the ladder is measured against, and lets the scene jump to any level.
inline Field1D ho_eigenstate(const Grid1D& g, double omega, int n) {
    const double pi = 3.14159265358979323846;
    Field1D prev{g};   // psi_{k-2}
    Field1D cur{g};    // psi_{k-1}
    const double a0 = std::pow(omega / pi, 0.25);
    for (int i = 0; i < g.n; ++i) {
        const double x = g.coord(i);
        cur[i] = a0 * std::exp(-0.5 * omega * x * x);  // psi_0
    }
    for (int k = 1; k <= n; ++k) {
        const double c1 = std::sqrt(2.0 * omega / k);
        const double c2 = std::sqrt(static_cast<double>(k - 1) / k);
        Field1D next{g};
        for (int i = 0; i < g.n; ++i) {
            next[i] = c1 * g.coord(i) * cur[i] - c2 * prev[i];
        }
        prev = std::move(cur);
        cur = std::move(next);
    }
    normalize(cur);
    return cur;
}

// The largest Fock level the FFT ladder reaches cleanly on a given grid.
// Each raise amplifies the k_max round-off floor by g/sqrt(n+1) with
// g = k_max/sqrt(2 omega) (the derivative term's top-of-band gain), so the
// accumulated noise amplitude after N raises is ~ eps0 g^N / sqrt(N!).
// The cap is the last N keeping that below a display-invisible bound
// (norm^2 error <~ 1e-8). eps0 ~ 3e-16 is the measured effective FFT
// round-off floor per apply (see tests/ladder_test.cpp for the calibration
// history: 1024 points, g ~ 114, lost the chain at n = 8 exactly as this
// model predicts).
inline int ladder_cap(double omega, double k_max) noexcept {
    const double gain = k_max / std::sqrt(2.0 * omega);
    const double eps0 = 3e-16;   // effective per-apply round-off floor
    const double bound = 1e-4;   // noise amplitude ceiling (norm^2 ~ 1e-8)
    double amp = eps0;
    int n = 0;
    while (n < 64) {
        amp *= gain / std::sqrt(n + 1.0);
        if (amp > bound) {
            break;
        }
        ++n;
    }
    return n;
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
