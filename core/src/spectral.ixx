module;
#include <cstddef>
#include <numbers>
#include <vector>
export module ses.spectral;
import ses.grid;


// FFT-bin -> physical wavenumber, fftfreq layout.


export namespace ses {

inline std::vector<double> wavenumbers(const Grid1D& g) {
    const double dk = 2.0 * std::numbers::pi / (g.xmax - g.xmin);
    std::vector<double> k(static_cast<std::size_t>(g.n));
    for (int j = 0; j < g.n; ++j) {
        // 2j<n not j<n/2: keeps DC=0 at n=1, matches fftfreq split for all n.
        const int shifted = (2 * j < g.n) ? j : j - g.n;
        k[static_cast<std::size_t>(j)] = dk * shifted;
    }
    return k;
}

}  // namespace ses
