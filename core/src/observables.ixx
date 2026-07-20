module;
#include <complex>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
#include <cstdint>
export module ses.observables;
export import ses.spectral;
export import ses.vec;
export import ses.fft;
export import ses.field;


// Scale-invariant (norm cancels) -> valid on unnormalized fields.


export namespace ses {

// Guards: den==0 -> 0 (avoid 0/0 NaN in readouts); clamp negative variance
// (FP cancellation) before sqrt.
inline double obs_ratio(double num, double den) noexcept {
    return den > 0.0 ? num / den : 0.0;
}
inline double obs_sigma(double second_moment, double mean) noexcept {
    return std::sqrt(std::max(0.0, second_moment - mean * mean));
}

inline double mean_position(const Field1D& f) noexcept {
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < f.size(); ++i) {
        const double w = std::norm(f[i]);
        num += f.grid().coord(i) * w;
        den += w;
    }
    return obs_ratio(num, den);
}

inline double sigma_x(const Field1D& f) noexcept {
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < f.size(); ++i) {
        const double x = f.grid().coord(i);
        const double w = std::norm(f[i]);
        num += x * x * w;
        den += w;
    }
    return obs_sigma(obs_ratio(num, den), mean_position(f));
}

inline double mean_momentum(const Field1D& f) {
    std::vector<std::complex<double>> phi = f.data();
    fft(phi);
    const std::vector<double> k = wavenumbers(f.grid());
    double num = 0.0;
    double den = 0.0;
    for (std::size_t j = 0; j < phi.size(); ++j) {
        const double w = std::norm(phi[j]);
        num += k[j] * w;
        den += w;
    }
    return obs_ratio(num, den);
}

// Each term normalized separately -> Parseval factor cancels within the term.
inline double mean_energy(const Field1D& f, const std::vector<double>& potential) {
    double num_v = 0.0;
    double den_x = 0.0;
    for (int i = 0; i < f.size(); ++i) {
        const double w = std::norm(f[i]);
        num_v += potential[static_cast<std::size_t>(i)] * w;
        den_x += w;
    }

    std::vector<std::complex<double>> phi = f.data();
    fft(phi);
    const std::vector<double> k = wavenumbers(f.grid());
    double num_t = 0.0;
    double den_k = 0.0;
    for (std::size_t j = 0; j < phi.size(); ++j) {
        const double w = std::norm(phi[j]);
        num_t += 0.5 * k[j] * k[j] * w;
        den_k += w;
    }

    return obs_ratio(num_v, den_x) + obs_ratio(num_t, den_k);
}

// Var(H) = ||(H-<H>)psi||^2/||psi||^2 residual form
// (LOCK: observables_test HighEnergyEigenstate).
inline double energy_variance(const Field1D& f, const std::vector<double>& potential) {
    std::vector<std::complex<double>> hpsi = f.data();
    fft(hpsi);
    const std::vector<double> k = wavenumbers(f.grid());
    for (std::size_t j = 0; j < hpsi.size(); ++j) {
        hpsi[j] *= 0.5 * k[j] * k[j];
    }
    ifft(hpsi);
    double num_h = 0.0;
    double den = 0.0;
    for (int i = 0; i < f.size(); ++i) {
        const std::size_t s = static_cast<std::size_t>(i);
        hpsi[s] += potential[s] * f[i];
        den += std::norm(f[i]);
        num_h += (std::conj(f[i]) * hpsi[s]).real();
    }
    const double mean = obs_ratio(num_h, den);
    // Residual, not naive <H^2>-<H>^2: the latter's E^2-scaled cancellation
    // floor defeats the absolute 1e-8 eigenstate gate at high rungs.
    double num_var = 0.0;
    for (int i = 0; i < f.size(); ++i) {
        const std::size_t s = static_cast<std::size_t>(i);
        num_var += std::norm(hpsi[s] - mean * f[i]);
    }
    return obs_ratio(num_var, den);
}

// Deliberately NOT scale-invariant (unlike the others): tunneling readout is
// measured against the initial unit norm, so absorbed flux must reduce it.
inline double probability_in_range(const Field1D& f, double a, double b) noexcept {
    double acc = 0.0;
    for (int i = 0; i < f.size(); ++i) {
        const double x = f.grid().coord(i);
        if (x >= a && x < b) {
            acc += std::norm(f[i]);
        }
    }
    return acc * f.grid().spacing();
}

// ---- 3D observables ----

inline Vec3d mean_position(const Field3D& f) noexcept {
    const Grid3D& g = f.grid();
    Vec3d num{};
    double den = 0.0;
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double w = std::norm(f(i, j, k));
                num.x += g.x.coord(i) * w;
                num.y += g.y.coord(j) * w;
                num.z += g.z.coord(k) * w;
                den += w;
            }
        }
    }
    return Vec3d{obs_ratio(num.x, den), obs_ratio(num.y, den),
                 obs_ratio(num.z, den)};
}

inline Vec3d sigma_position(const Field3D& f) noexcept {
    const Grid3D& g = f.grid();
    Vec3d num{};
    double den = 0.0;
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double w = std::norm(f(i, j, k));
                const double x = g.x.coord(i);
                const double y = g.y.coord(j);
                const double z = g.z.coord(k);
                num.x += x * x * w;
                num.y += y * y * w;
                num.z += z * z * w;
                den += w;
            }
        }
    }
    const Vec3d m = mean_position(f);
    return Vec3d{obs_sigma(obs_ratio(num.x, den), m.x),
                 obs_sigma(obs_ratio(num.y, den), m.y),
                 obs_sigma(obs_ratio(num.z, den), m.z)};
}

inline Vec3d mean_momentum(const Field3D& f) {
    Field3D phi = f;
    fft(phi);
    const Grid3D& g = f.grid();
    const std::vector<double> kx = wavenumbers(g.x);
    const std::vector<double> ky = wavenumbers(g.y);
    const std::vector<double> kz = wavenumbers(g.z);
    Vec3d num{};
    double den = 0.0;
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double w = std::norm(phi(i, j, k));
                num.x += kx[static_cast<std::size_t>(i)] * w;
                num.y += ky[static_cast<std::size_t>(j)] * w;
                num.z += kz[static_cast<std::size_t>(k)] * w;
                den += w;
            }
        }
    }
    return Vec3d{obs_ratio(num.x, den), obs_ratio(num.y, den),
                 obs_ratio(num.z, den)};
}

// mass default 1.0: exact division by 1.0 is a no-op -> oracles untouched.
inline double mean_energy(const Field3D& f, const std::vector<double>& potential,
                          double mass = 1.0) {
    double num_v = 0.0;
    double den_x = 0.0;
    for (std::size_t i = 0; i < f.data().size(); ++i) {
        const double w = std::norm(f.data()[i]);
        num_v += potential[i] * w;
        den_x += w;
    }

    Field3D phi = f;
    fft(phi);
    const Grid3D& g = f.grid();
    const std::vector<double> kx = wavenumbers(g.x);
    const std::vector<double> ky = wavenumbers(g.y);
    const std::vector<double> kz = wavenumbers(g.z);
    double num_t = 0.0;
    double den_k = 0.0;
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double w = std::norm(phi(i, j, k));
                const double kxx = kx[static_cast<std::size_t>(i)];
                const double kyy = ky[static_cast<std::size_t>(j)];
                const double kzz = kz[static_cast<std::size_t>(k)];
                num_t += 0.5 * (kxx * kxx + kyy * kyy + kzz * kzz) / mass * w;
                den_k += w;
            }
        }
    }

    return obs_ratio(num_v, den_x) + obs_ratio(num_t, den_k);
}

}  // namespace ses
