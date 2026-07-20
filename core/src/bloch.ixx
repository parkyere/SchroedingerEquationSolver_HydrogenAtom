module;
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>
export module ses.bloch;
export import ses.field;
export import ses.grid;
import ses.fft;
import ses.spectral;


// 1D periodic lattice V(x) = V0 sin^2(kL x). sin^2 (not Kronig-Penney kinks) keeps
// the FFT plane-wave basis spectral (optical-lattice / Mathieu). Central equation:
// symmetric tridiagonal over {q + m G} (G = 2 kL): diag (q+mG)^2/2 + V0/2, off-diag
// -V0/4. Comoving gauge A(t) = -F t; -F x would break the periodic box.


export namespace ses {

namespace bloch_detail {

// Count eigenvalues below lambda. RATIO-form Sturm (not product): the product
// form sticks at 0 when a dyadic midpoint hits a leading-minor eigenvalue
// exactly; the ratio nudges past it (LAPACK dlaebz style).
inline int sturm_count(const std::vector<double>& d,
                       const std::vector<double>& o, double lambda) {
    int count = 0;
    double q = 1.0;
    for (std::size_t i = 0; i < d.size(); ++i) {
        const double off2 = i > 0 ? o[i - 1] * o[i - 1] : 0.0;
        if (q == 0.0) {
            q = -1e-300;
        }
        q = (d[i] - lambda) - off2 / q;
        if (q < 0.0) {
            ++count;
        }
    }
    return count;
}

}  // namespace bloch_detail

inline std::vector<double> lattice_bands(double v0, double kl, double q,
                                         int n_bands) {
    const double g2 = 2.0 * kl;
    const int m_max = std::max(8, n_bands + 6);
    const int n = 2 * m_max + 1;
    std::vector<double> d(static_cast<std::size_t>(n));
    std::vector<double> o(static_cast<std::size_t>(n - 1), -0.25 * v0);
    for (int m = -m_max; m <= m_max; ++m) {
        const double k = q + m * g2;
        d[static_cast<std::size_t>(m + m_max)] = 0.5 * k * k + 0.5 * v0;
    }
    // Sturm bisection per band (Gershgorin bracket).
    double lo = d[0];
    double hi = d[0];
    for (std::size_t i = 0; i < d.size(); ++i) {
        const double r = (i > 0 ? std::abs(o[i - 1]) : 0.0) +
                         (i + 1 < d.size() ? std::abs(o[i]) : 0.0);
        lo = std::min(lo, d[i] - r);
        hi = std::max(hi, d[i] + r);
    }
    std::vector<double> bands;
    bands.reserve(static_cast<std::size_t>(n_bands));
    for (int band = 0; band < n_bands; ++band) {
        double a = lo;
        double b = hi;
        for (int it = 0; it < 200; ++it) {
            const double mid = 0.5 * (a + b);
            if (bloch_detail::sturm_count(d, o, mid) <= band) {
                a = mid;
            } else {
                b = mid;
            }
        }
        bands.push_back(0.5 * (a + b));
    }
    return bands;
}

// Comoving-gauge tilt A(t) = -F t: kinetic phase rebuilt each step at midpoint A
// (exact for linear A), unlike the fixed table in SplitOperator1D.
class TiltedSplitOperator1D {
public:
    TiltedSplitOperator1D(const Grid1D& g, const std::vector<double>& potential,
                          double dt, double force)
        : dt_(dt), force_(force), k_(wavenumbers(g)) {
        assert(static_cast<int>(potential.size()) == g.n);
        half_v_.resize(potential.size());
        for (std::size_t i = 0; i < potential.size(); ++i) {
            const double th = -0.5 * potential[i] * dt;
            half_v_[i] = std::complex<double>{std::cos(th), std::sin(th)};
        }
        kinetic_.resize(k_.size());
    }

    double dt() const noexcept { return dt_; }
    // quasimomentum drift F*t = -A(t)
    double drift() const noexcept { return force_ * t_; }
    void reset_time() noexcept { t_ = 0.0; }

    void step(Field1D& psi, int nsteps = 1) {
        assert(psi.data().size() == half_v_.size());
        for (int s = 0; s < nsteps; ++s) {
            const double a_mid = -force_ * (t_ + 0.5 * dt_);
            for (std::size_t j = 0; j < k_.size(); ++j) {
                const double km = k_[j] - a_mid;
                const double th = -0.5 * km * km * dt_;
                kinetic_[j] =
                    std::complex<double>{std::cos(th), std::sin(th)};
            }
            apply_phase(half_v_, psi.data());
            fft(psi.data());
            apply_phase(kinetic_, psi.data());
            ifft(psi.data());
            apply_phase(half_v_, psi.data());
            t_ += dt_;
        }
    }

private:
    static void apply_phase(const std::vector<std::complex<double>>& phase,
                            std::vector<std::complex<double>>& a) noexcept {
        for (std::size_t i = 0; i < a.size(); ++i) {
            a[i] = a[i] * phase[i];
        }
    }

    double dt_;
    double force_;
    double t_ = 0.0;
    std::vector<double> k_;
    std::vector<std::complex<double>> half_v_;
    std::vector<std::complex<double>> kinetic_;
};

}  // namespace ses
