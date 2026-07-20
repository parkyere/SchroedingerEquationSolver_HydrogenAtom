module;
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
export module ses.scenario.kepler_seed;
export import ses.scenario.manifold_spec;
import ses.measurement;


// Rydberg (Kepler) wave-packet seed onto the real-Y_lm manifold; circular
// states |n, l=m=n-1> beat at the Kepler rate E_{n+1}-E_n ~ 1/n^3.
// CONTRACT: tests/kepler_test.cpp (pair table, purity, orbit rate).


export namespace ses_shell {

// Defaults: n_bar mid-manifold (r ~ n^2 ~ 20 a0 fits the +-80 box).
inline constexpr double kKeplerNBar = 4.5;
inline constexpr double kKeplerSigmaN = 1.0;

struct KeplerPair {
    int n;  // circular: l = m = n - 1
    int idx_cos;
    int idx_sin;
};

// Circular pairs scanned off kStateSpec (no hardcoded indices).
inline std::array<KeplerPair, 5> kepler_pairs() {
    std::array<KeplerPair, 5> pairs{};
    std::size_t out = 0;
    for (int i = 0; i < kNumStates && out < pairs.size(); ++i) {
        const StateSpec& sc = kStateSpec[i];
        if (sc.l < 1 || sc.m != sc.l || state_n(i) != sc.l + 1) {
            continue;
        }
        int sin_idx = -1;
        for (int j = 0; j < kNumStates; ++j) {
            if (kStateSpec[j].level == sc.level && kStateSpec[j].m == -sc.l) {
                sin_idx = j;
                break;
            }
        }
        pairs[out++] = KeplerPair{sc.l + 1, i, sin_idx};
    }
    return pairs;
}

inline std::array<std::complex<double>, kNumStates> kepler_coefficients(
    double n_bar, double sigma) {
    std::array<std::complex<double>, kNumStates> c{};
    double sum = 0.0;
    for (const KeplerPair& p : kepler_pairs()) {
        const double dn = p.n - n_bar;
        const double w = std::exp(-dn * dn / (4.0 * sigma * sigma));
        const ses::RealPair rp = ses::pair_from_signed_m(w, +1);
        c[static_cast<std::size_t>(p.idx_cos)] = rp.c_cos;
        c[static_cast<std::size_t>(p.idx_sin)] = rp.c_sin;
        sum += w * w;
    }
    if (sum > 0.0) {
        const double inv = 1.0 / std::sqrt(sum);
        for (auto& z : c) {
            z *= inv;
        }
    }
    return c;
}

}  // namespace ses_shell
