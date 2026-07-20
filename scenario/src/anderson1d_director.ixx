module;
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>
export module ses.scenario.anderson1d_director;
export import ses.scenario.line1d_director;
import ses.wavepacket;
import ses.observables;


// Anderson localization in 1D: a packet with energy ABOVE every barrier
// (a classical particle sails through) is STOPPED by coherent multiple
// scattering off a random landscape -- in 1D every eigenstate is
// exponentially localized for any disorder. Sharp scatterers on purpose
// (sigma < lambda): smooth bumps are transparent at k (the corral fence
// lesson), sharp ones backscatter. Clean (W = 0) contrast: ballistic
// flight. CONTRACT: tests/anderson1d_test.cpp.


export namespace ses_shell {

// SPECKLE disorder (the cold-atom realization, Billy et al. 2008): dense
// overlapping bumps -> a smooth random field with correlation length ~
// sigma, every correlation cell a scatterer. Strength W ~ E (the standard
// Anderson regime): sub-E-everywhere fields proved far too transparent
// (measured <= 36% blocking over 100 Bohr at every sub-E tuning) -- the
// localization length simply exceeds any reasonable stage.
constexpr double kAn1dSpacing = 0.6;    // speckle grain spacing (Bohr)
constexpr double kAn1dBumpSigma = 0.3;  // correlation length
constexpr double kAn1dK0 = 1.2;         // E = 0.72 Ha
constexpr double kAn1dW = 1.2;          // grain amplitude range [-W, W]

// Random landscape: Gaussian bumps at every lattice site, amplitudes
// uniform in [-w, w] from the SEEDED mt19937 (deterministic per seed).
inline std::vector<double> anderson_potential(const ses::Grid1D& g, double w,
                                              unsigned seed) {
    std::vector<double> v(static_cast<std::size_t>(g.n), 0.0);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> amp(-w, w);
    const double margin = 2.0;  // no scatterer inside the box lip
    const int s_lo = static_cast<int>(
        std::ceil((g.xmin + margin) / kAn1dSpacing));
    const int s_hi = static_cast<int>(
        std::floor((g.xmax - margin) / kAn1dSpacing));
    for (int s = s_lo; s <= s_hi; ++s) {
        const double xs = s * kAn1dSpacing;
        const double a = amp(rng);
        for (int i = 0; i < g.n; ++i) {
            const double dx = g.coord(i) - xs;
            if (std::abs(dx) > 5.0 * kAn1dBumpSigma) {
                continue;
            }
            v[static_cast<std::size_t>(i)] +=
                a * std::exp(-dx * dx /
                             (2.0 * kAn1dBumpSigma * kAn1dBumpSigma));
        }
    }
    return v;
}

}  // namespace ses_shell
