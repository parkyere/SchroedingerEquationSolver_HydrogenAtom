module;
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <utility>
#include <vector>
export module ses.ladder;
export import ses.field;
export import ses.grid;
import ses.fft;
import ses.parallel;
import ses.spectral;


// 1D HO ladder operators, atomic units (m = hbar = 1).


export namespace ses {

namespace ladder_detail {

// Shared a/adag kernel; d/dx spectral (FFT ik).
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

inline double norm_sq_h(const std::vector<std::complex<double>>& v, double h) noexcept {
    double acc = 0.0;
    for (const std::complex<double>& z : v) {
        acc += std::norm(z);
    }
    return acc * h;
}

inline void store_normalized(Field1D& psi, const std::vector<std::complex<double>>& v,
                             double norm2) noexcept {
    const double inv = 1.0 / std::sqrt(norm2);
    for (int i = 0; i < psi.size(); ++i) {
        psi[i] = inv * v[static_cast<std::size_t>(i)];
    }
}

// ---- scaled Hermite chain ------------------------------------------------
// Plain seed exp(-omega x^2/2) underflows to EXACT zero (arg > ~745), zeroing
// the outer lobes of deep states forever. Carried as mantissa * 2^exponent;
// power-of-two rescales are exact, so in-range values stay bitwise identical.

constexpr double kChainHi = 0x1p+500;
constexpr double kChainLo = 0x1p-500;
constexpr double kChainDn = 0x1p-512;
constexpr double kChainUp = 0x1p+512;
constexpr int kChainExp = 512;

// Seed psi_0 as (mantissa, exponent): exact where plain exp is normal,
// log2-domain split otherwise.
inline void chain_seed(double a0, double omega, double x, double& m, int& e) {
    const double arg = 0.5 * omega * x * x;
    if (arg < 690.0) {
        m = a0 * std::exp(-arg);
        e = 0;
        return;
    }
    const double bits = -arg * 1.4426950408889634;  // log2(psi_0 / a0)
    const int shift = static_cast<int>(std::floor(bits));
    m = a0 * std::exp2(bits - shift);
    e = shift;
}

// One recurrence rung on a scaled (prev, cur) pair sharing one exponent.
// Same evaluation order as the plain chain: bitwise-identical until a rescale
// fires (only outside (2^-500, 2^500), where plain was denormal anyway).
inline void chain_advance(double c1x, double c2, double& p, double& c,
                          int& e) {
    const double next = c1x * c - c2 * p;
    p = c;
    c = next;
    const double ac = std::abs(c);
    if (ac > kChainHi || std::abs(p) > kChainHi) {
        p *= kChainDn;
        c *= kChainDn;
        e += kChainExp;
    } else if (ac < kChainLo && std::abs(p) < kChainLo && e != 0) {
        p *= kChainUp;
        c *= kChainUp;
        e -= kChainExp;
    }
}

// Whole-grid scaled chain: SoA rows (p, c, e) advanced level by level.
// advance_to is transposed (each worker walks its own points through all
// pending levels), per-point work independent -> deterministic. Copying the
// object is the snapshot ho_level_cap's bisection rewinds to.
class ScaledChain {
  public:
    ScaledChain(const Grid1D& g, double omega)
        : g_(&g),
          omega_(omega),
          p_(static_cast<std::size_t>(g.n), 0.0),
          c_(static_cast<std::size_t>(g.n)),
          e_(static_cast<std::size_t>(g.n), 0) {
        const double pi = 3.14159265358979323846;
        const double a0 = std::pow(omega / pi, 0.25);
        parallel_for(g.n, [&](int i) {
            const std::size_t s = static_cast<std::size_t>(i);
            chain_seed(a0, omega_, g_->coord(i), c_[s], e_[s]);
        });
    }

    int level() const noexcept { return level_; }

    void advance_to(int target) {
        if (target <= level_) {
            return;
        }
        ensure_coeffs(target);
        const int from = level_;
        parallel_ranges(g_->n, [&](int, int begin, int end) {
            for (int i = begin; i < end; ++i) {
                const std::size_t s = static_cast<std::size_t>(i);
                const double x = g_->coord(i);
                double pp = p_[s];
                double cc = c_[s];
                int ee = e_[s];
                for (int k = from + 1; k <= target; ++k) {
                    chain_advance(r1_[static_cast<std::size_t>(k)] * x,
                                  r2_[static_cast<std::size_t>(k)], pp, cc,
                                  ee);
                }
                p_[s] = pp;
                c_[s] = cc;
                e_[s] = ee;
            }
        });
        level_ = target;
    }

    double value(int i) const noexcept {
        const std::size_t s = static_cast<std::size_t>(i);
        return std::ldexp(c_[s], e_[s]);
    }

  private:
    void ensure_coeffs(int target) {
        const std::size_t need = static_cast<std::size_t>(target) + 1;
        while (r1_.size() < need) {
            const double k = static_cast<double>(r1_.size());
            r1_.push_back(std::sqrt(2.0 * omega_ / k));
            r2_.push_back(std::sqrt((k - 1.0) / k));
        }
    }

    const Grid1D* g_;
    double omega_;
    int level_ = 0;
    std::vector<double> r1_{0.0};  // sqrt(2 omega / k), index k >= 1
    std::vector<double> r2_{0.0};  // sqrt((k-1)/k)
    std::vector<double> p_;        // mantissa psi_{level-1}
    std::vector<double> c_;        // mantissa psi_{level}
    std::vector<int> e_;           // shared per-point exponent
};

}  // namespace ladder_detail

// adag: psi <- adag psi / ||adag psi||; returns ||adag psi||^2 (n+1 on |n>).
inline double ladder_raise(Field1D& psi, double omega) {
    const std::vector<std::complex<double>> out =
        ladder_detail::apply(psi, omega, -1.0);
    const double norm2 = ladder_detail::norm_sq_h(out, psi.grid().spacing());
    ladder_detail::store_normalized(psi, out, norm2);
    return norm2;
}

// Exact HO eigenstate |n> from the normalized Hermite-Gauss recurrence in
// x-space -- no derivative, so no spectral round-off (exact to round-off for
// every representable level). The ground-truth oracle the ladder is measured
// against.
inline Field1D ho_eigenstate(const Grid1D& g, double omega, int n) {
    ladder_detail::ScaledChain chain{g, omega};
    chain.advance_to(n);
    Field1D cur{g};
    parallel_for(g.n, [&](int i) { cur[i] = chain.value(i); });
    normalize(cur);
    return cur;
}

// Eigenbasis decomposition of psi up to e_max: (E_n = (n+1/2) omega,
// |<n|psi>|^2) pairs -- the superposition spectrum, not emitted photons.
// CONTRACT: tests/ho_spectrum_test.cpp.
inline std::vector<std::pair<double, double>> ho1d_spectrum(
    const Field1D& psi, double omega, double e_max) {
    std::vector<std::pair<double, double>> lines;
    const double h = psi.grid().spacing();
    for (int n = 0;; ++n) {
        const double e = (n + 0.5) * omega;
        if (e > e_max) {
            break;
        }
        const Field1D basis = ho_eigenstate(psi.grid(), omega, n);
        std::complex<double> ov{};
        for (int i = 0; i < psi.size(); ++i) {
            ov += std::conj(basis[i]) * psi[i];
        }
        ov *= h;
        lines.emplace_back(e, std::norm(ov));
    }
    return lines;
}

// Representability ceiling: largest level whose Hermite oracle stays faithful
// on the grid (energy within 0.1% of (n+1/2)w). Box-limited for a soft well,
// Nyquist-limited for a stiff one; caps the STABLE rungs (>> ladder_cap).
inline int ho_level_cap(const Grid1D& g, double omega) {
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        const double x = g.coord(i);
        v[static_cast<std::size_t>(i)] = 0.5 * omega * omega * x * x;
    }
    const std::vector<double> k = wavenumbers(g);
    // Faithful = (a) contained AND (b) energy-exact.
    // (a) catches box truncation that (b) CANNOT: a clipped Hermite slice still
    //     satisfies k(x)^2/2 + V = E at every sample, so its grid energy stays
    //     within ~1e-5 of (n+1/2)w with turning points outside the box. Test:
    //     boundary density below 1e-6 of the bulk.
    // (b) catches the Nyquist side (aliased <T>); scale-invariant.
    std::vector<std::complex<double>> phi(static_cast<std::size_t>(g.n));
    auto faithful = [&](const ladder_detail::ScaledChain& chain, int n) {
        parallel_for(g.n, [&](int i) {
            phi[static_cast<std::size_t>(i)] = chain.value(i);
        });
        double num_v = 0.0;
        double den_x = 0.0;
        double bulk = 0.0;
        for (std::size_t j = 0; j < phi.size(); ++j) {
            const double w = std::norm(phi[j]);
            num_v += v[j] * w;
            den_x += w;
            bulk = std::max(bulk, w);
        }
        const double edge = std::max(std::norm(phi.front()),
                                     std::norm(phi.back()));
        if (edge > 1e-6 * bulk) {
            return false;  // leaks out of the box: not this grid's state
        }
        fft(phi);
        double num_t = 0.0;
        double den_k = 0.0;
        for (std::size_t j = 0; j < phi.size(); ++j) {
            const double w = std::norm(phi[j]);
            num_t += 0.5 * k[j] * k[j] * w;
            den_k += w;
        }
        const double e = num_t / den_k + num_v / den_x;
        const double e_exact = (n + 0.5) * omega;
        return std::abs(e - e_exact) <= 1e-3 * e_exact;
    };
    // Probe bound: a safety net above the physics ceilings, not the cap. The
    // measured boundary can overhang n_box (faithfulness fades over an Airy
    // transition width), so 1.5x + 64 clears it; the scan stops at the first
    // failure, so a generous bound only guards the loop. Grid-derived so fine
    // grids (ceiling in the thousands) are not clipped.
    const double x_edge = std::min(std::abs(g.xmin), std::abs(g.xmax));
    const double k_max = 3.14159265358979323846 / g.spacing();
    const double n_box = 0.5 * omega * x_edge * x_edge;
    const double n_nyq = 0.5 * k_max * k_max / omega;
    const int bound = std::max(
        1,
        static_cast<int>(std::min(1.5 * std::min(n_box, n_nyq), 1048576.0)) +
            64);
    // Representability is MONOTONE in n (once lost, never regained), so:
    // stride-scan, snapshot at the last good check, then bisect the last
    // window from the snapshot. Runs on every omega-slider apply, up to 64k
    // points per level.
    const int stride = std::clamp(bound / 64, 16, 512);
    ladder_detail::ScaledChain chain{g, omega};
    ladder_detail::ScaledChain snap = chain;
    int last_good = 0;
    int first_bad = -1;
    while (chain.level() < bound) {
        chain.advance_to(std::min(chain.level() + stride, bound));
        if (faithful(chain, chain.level())) {
            last_good = chain.level();
            snap = chain;
        } else {
            first_bad = chain.level();
            break;
        }
    }
    if (first_bad < 0) {
        return last_good;  // clean through the physics bound
    }
    while (first_bad - last_good > 1) {
        const int mid = last_good + (first_bad - last_good) / 2;
        chain = snap;
        chain.advance_to(mid);
        if (faithful(chain, mid)) {
            last_good = mid;
            snap = std::move(chain);
        } else {
            first_bad = mid;
        }
    }
    return last_good;
}

// Ladder step in the truncated Fock basis |0..n_top> -- superposition
// counterpart of ladder_rung_stable. Project c_n = <n|psi>, act on the
// coefficients, resynthesize from noise-free oracles: no spectral derivative,
// so it works at ANY grid k_max. *out_residual gets the outside-band weight;
// psi is written only when the band holds the state (residual <= 1/2) AND the
// result does not vanish. Returns counting norm^2 inside the band.
inline double ladder_fock(Field1D& psi, double omega, bool up, int n_top,
                          double* out_residual = nullptr,
                          double vanish_eps = 1e-6) {
    const Grid1D& g = psi.grid();
    const double h = g.spacing();
    const double psi_n2 = norm_sq(psi);
    // Pass 1: project c_n = <n|psi> off ONE scaled chain -- no basis storage
    // (O(band*grid) memory would forbid deep bands). Early stop once the band
    // holds all but 1e-10 of the state: every unscanned |c_n|^2 is then bounded
    // by that leftover, so the residual stays honest. Sums use the chunk-ordered
    // parallel reduction (bitwise-deterministic).
    struct DotAcc {
        std::complex<double> dot{};
        double n2 = 0.0;
        DotAcc& operator+=(const DotAcc& o) {
            dot += o.dot;
            n2 += o.n2;
            return *this;
        }
    };
    ladder_detail::ScaledChain chain{g, omega};
    std::vector<std::complex<double>> c;
    std::vector<double> inv_norm;  // 1 / ||chain level||_h, cached for pass 2
    double inside = 0.0;
    for (int n = 0; n <= n_top; ++n) {
        chain.advance_to(n);
        const DotAcc acc =
            parallel_sum(g.n, DotAcc{}, [&](int i) {
                const double val = chain.value(i);
                return DotAcc{val * psi[i], val * val};
            });
        const double inv = 1.0 / std::sqrt(acc.n2 * h);
        inv_norm.push_back(inv);
        c.push_back(acc.dot * h * inv);
        inside += std::norm(c.back());
        if (inside >= (1.0 - 1e-10) * psi_n2) {
            break;  // the band holds the whole state; higher c_n ~ 0
        }
    }
    const int m = static_cast<int>(c.size()) - 1;  // last projected level
    const double residual = std::max(0.0, 1.0 - inside / psi_n2);
    if (out_residual != nullptr) {
        *out_residual = residual;
    }
    // Exact coefficient action on the scanned band.
    const int nd = m + (up ? 2 : 1);
    std::vector<std::complex<double>> d(static_cast<std::size_t>(nd));
    double norm2 = 0.0;
    if (up) {
        for (int n = 0; n <= m; ++n) {
            d[static_cast<std::size_t>(n + 1)] =
                std::sqrt(n + 1.0) * c[static_cast<std::size_t>(n)];
            norm2 += (n + 1.0) * std::norm(c[static_cast<std::size_t>(n)]);
        }
    } else {
        for (int n = 0; n + 1 <= m; ++n) {
            d[static_cast<std::size_t>(n)] =
                std::sqrt(n + 1.0) * c[static_cast<std::size_t>(n + 1)];
            norm2 += (n + 1.0) * std::norm(c[static_cast<std::size_t>(n + 1)]);
        }
    }
    if (residual > 0.5 || norm2 < vanish_eps) {
        return norm2;  // outside the band, or annihilated: psi untouched
    }
    // Pass 2: resynthesize psi = sum d_n |n> off a fresh chain, reusing the
    // cached level norms (the up-shift top level n = m+1 was not normed in
    // pass 1; measured here).
    std::vector<std::complex<double>> out(static_cast<std::size_t>(g.n));
    ladder_detail::ScaledChain synth{g, omega};
    for (int n = 0; n < nd; ++n) {
        synth.advance_to(n);
        if (d[static_cast<std::size_t>(n)] == std::complex<double>{}) {
            continue;
        }
        double inv = 0.0;
        if (n < static_cast<int>(inv_norm.size())) {
            inv = inv_norm[static_cast<std::size_t>(n)];
        } else {
            const double lvl_n2 = parallel_sum(g.n, 0.0, [&](int i) {
                const double val = synth.value(i);
                return val * val;
            });
            inv = 1.0 / std::sqrt(lvl_n2 * h);
        }
        const std::complex<double> w = d[static_cast<std::size_t>(n)] * inv;
        parallel_for(g.n, [&](int i) {
            out[static_cast<std::size_t>(i)] += w * synth.value(i);
        });
    }
    const double inv_total = 1.0 / std::sqrt(norm2);
    parallel_for(g.n, [&](int i) {
        psi[i] = inv_total * out[static_cast<std::size_t>(i)];
    });
    return norm2;
}

// Largest Fock level the FFT ladder reaches cleanly -- MEASURED, not modeled:
// adag's round-off gains (derivative k_max/sqrt(2w), x-term x_max*sqrt(w/2))
// make the cap non-monotone with no closed form, so probe directly. Returns
// the last level matching the Hermite oracle within defect_tol (1e-6 ~ 0.1%
// amplitude). ~cap FFTs, only on omega change.
inline int ladder_cap(const Grid1D& g, double omega, double defect_tol = 1e-6) {
    Field1D psi = ho_eigenstate(g, omega, 0);
    int cap = 0;
    for (int n = 1; n <= 64; ++n) {
        ladder_raise(psi, omega);
        const Field1D oracle = ho_eigenstate(g, omega, n);
        std::complex<double> ov{};
        for (int i = 0; i < g.n; ++i) {
            ov += std::conj(psi[i]) * oracle[i];
        }
        ov *= g.spacing();  // grid-weighted overlap (both normalized)
        if (1.0 - std::norm(ov) > defect_tol) {
            break;  // ladder diverged from the clean eigenstate here
        }
        cap = n;
    }
    return cap;
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

// Noise-free ladder rung for a state KNOWN to be eigenstate |n_from> (caller's
// Var(H) classifier is the gate). The raw spectral operator still ACTS -- it
// supplies the counting norm^2 and the global phase -- but the state body is
// rebuilt from the direct Hermite oracle carrying that phase, so the round-off
// floor RESETS each rung instead of compounding (cap becomes ho_level_cap, not
// the raw-chain noise cap). At the ground the operator refuses (returns ~0,
// psi untouched). If psi is NOT the claimed eigenstate (oracle overlap < 1/2),
// the raw result is kept: caller misclassified, honest output stands.
inline double ladder_rung_stable(Field1D& psi, double omega, int n_from,
                                 bool up, double vanish_eps = 1e-6) {
    const Grid1D& g = psi.grid();
    Field1D raw = psi;
    double norm2 = 0.0;
    if (up) {
        norm2 = ladder_raise(raw, omega);
    } else {
        norm2 = ladder_lower(raw, omega, vanish_eps);
        if (norm2 < vanish_eps) {
            return norm2;  // annihilation: psi untouched, caller's signal
        }
    }
    const int target = up ? n_from + 1 : n_from - 1;
    const Field1D oracle = ho_eigenstate(g, omega, target);
    std::complex<double> ov{};
    for (int i = 0; i < g.n; ++i) {
        ov += std::conj(oracle[i]) * raw[i];
    }
    ov *= g.spacing();
    if (std::norm(ov) < 0.5) {
        psi = std::move(raw);  // misclassified input: keep the raw output
        return norm2;
    }
    const std::complex<double> phase = ov / std::abs(ov);
    for (int i = 0; i < g.n; ++i) {
        psi[i] = phase * oracle[i];
    }
    return norm2;
}

}  // namespace ses
