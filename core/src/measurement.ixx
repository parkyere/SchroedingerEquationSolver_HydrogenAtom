module;
#include <complex>
#include <cmath>
#include <cstddef>
#include <vector>
#include <cstdint>
export module ses.measurement;
export import ses.grid;
export import ses.vec;
export import ses.field;
import ses.parallel;


// Soft position measurement (Gaussian POVM). Randomness stays OUT of core:
// callers supply u in [0,1); samplers invert a discrete CDF in flat-index
// order (cell-volume cancels). POVM consistency: Kraus mask e^{-(r-c)^2/(4
// s^2)} gives E_c = M_c^dag M_c, so the outcome density is |psi|^2 blurred by
// std sigma_m, not raw |psi|^2 -- see sample_povm_index.


export namespace ses {

// Incomplete manifold (sum(P) <= 1): returns eigenstate index, or -1 for the
// 1 - sum(P) untracked/continuum deficit (caller then projects the manifold
// OUT, psi <- (1 - P)|psi>).
inline int sample_energy_eigenstate(const std::vector<double>& populations, double u) noexcept {
    double cum = 0.0;
    for (std::size_t n = 0; n < populations.size(); ++n) {
        cum += populations[n];
        if (u < cum) {
            return static_cast<int>(n);
        }
    }
    return -1;
}

inline int sample_collapse_index(const Field3D& psi, double u) noexcept {
    double total = 0.0;
    for (const std::complex<double>& z : psi.data()) {
        total += std::norm(z);
    }
    const double target = u * total;
    double cum = 0.0;
    const std::size_t n = psi.data().size();
    for (std::size_t i = 0; i < n; ++i) {
        cum += std::norm(psi.data()[i]);
        if (cum > target) {
            return static_cast<int>(i);
        }
    }
    return static_cast<int>(n - 1);  // u ~ 1 rounding fallback
}

// Per-axis Gaussian (std sigma_m) convolution of |psi|^2; periodic wrap
// matches the grid FFT topology.
inline std::vector<double> povm_outcome_density(const Field3D& psi,
                                                double sigma_m) {
    const Grid3D& g = psi.grid();
    const int nx = g.x.n;
    const int ny = g.y.n;
    const int nz = g.z.n;
    std::vector<double> d(psi.data().size());
    for (std::size_t i = 0; i < d.size(); ++i) {
        d[i] = std::norm(psi.data()[i]);
    }
    std::vector<double> tmp(d.size());
    const Grid1D* axes[3] = {&g.x, &g.y, &g.z};
    const int strides[3] = {1, nx, nx * ny};
    for (int a = 0; a < 3; ++a) {
        const int n = axes[a]->n;
        const double h = axes[a]->spacing();
        const int radius = static_cast<int>(std::ceil(4.0 * sigma_m / h));
        std::vector<double> w(static_cast<std::size_t>(2 * radius + 1));
        double sum = 0.0;
        for (int t = -radius; t <= radius; ++t) {
            const double x = t * h / sigma_m;
            sum += w[static_cast<std::size_t>(t + radius)] = std::exp(-0.5 * x * x);
        }
        for (double& v : w) {
            v /= sum;
        }
        const int stride = strides[a];
        const int lines = nx * ny * nz / n;
        parallel_for(lines, [&](int line) {
            // reinsert axis-a into the flattened remaining-dims counter
            const int base = line % stride + (line / stride) * stride * n;
            for (int p = 0; p < n; ++p) {
                double acc = 0.0;
                for (int t = -radius; t <= radius; ++t) {
                    const int q = (p + t % n + n) % n;
                    acc += w[static_cast<std::size_t>(t + radius)] *
                           d[static_cast<std::size_t>(base + q * stride)];
                }
                tmp[static_cast<std::size_t>(base + p * stride)] = acc;
            }
        });
        d.swap(tmp);
    }
    return d;
}

// Signed-m amplitudes from the real cos/sin pair. Convention (ses.harmonics,
// test-pinned): Y_{l,+|m|} ~ cos(m phi), Y_{l,-|m|} ~ sin(m phi).
struct SignedM {
    std::complex<double> plus;
    std::complex<double> minus;
};
inline SignedM signed_m_amplitudes(std::complex<double> c_cos,
                                   std::complex<double> c_sin) noexcept {
    const double inv = 1.0 / std::sqrt(2.0);
    const std::complex<double> i{0.0, 1.0};
    return SignedM{inv * (c_cos - i * c_sin), inv * (c_cos + i * c_sin)};
}

// Inverse of signed_m_amplitudes: real cos/sin pair from one kept signed-m
// component (both outcomes sum back to the original pair).
struct RealPair {
    std::complex<double> c_cos;
    std::complex<double> c_sin;
};
inline RealPair pair_from_signed_m(std::complex<double> a, int sign) noexcept {
    const double inv = 1.0 / std::sqrt(2.0);
    const std::complex<double> i{0.0, 1.0};
    return RealPair{inv * a, static_cast<double>(sign) * inv * (i * a)};
}

// Sample from the POVM density, not raw |psi|^2: outcomes can land on a node
// of |psi|^2 that a sigma_m detector cannot resolve.
inline int sample_povm_index(const Field3D& psi, double sigma_m, double u) {
    const std::vector<double> d = povm_outcome_density(psi, sigma_m);
    double total = 0.0;
    for (double p : d) {
        total += p;
    }
    const double target = u * total;
    double cum = 0.0;
    for (std::size_t i = 0; i < d.size(); ++i) {
        cum += d[i];
        if (cum > target) {
            return static_cast<int>(i);
        }
    }
    return static_cast<int>(d.size() - 1);  // u ~ 1 with rounding
}

// Same amplitude convention as gaussian_wavepacket (4 sigma^2, not 2), so
// Gaussian x Gaussian posteriors stay analytic.
inline void collapse_wavepacket(Field3D& psi, Vec3d center, double sigma_m) noexcept {
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
