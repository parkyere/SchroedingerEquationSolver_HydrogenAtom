module;
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>
export module ses.interference;
export import ses.field;
export import ses.grid;
import ses.fft;
import ses.spectral;


// Double-slit support in the TRANSVERSE frame: the 1D axis is the
// coordinate parallel to the slit plane, propagation toward the screen is
// time (t = z/v -- the exact paraxial/Fresnel reduction of the real 2D
// experiment, not an approximation). The infinite wall with two slits acts
// at one instant as an aperture: the wavefront is windowed to the two
// openings, the wall absorbs the rest, and the surviving state is
// renormalized (post-selection on transmission). From that instant the
// momentum distribution |psi~(k)|^2 IS the accumulated screen pattern:
// free flight preserves it exactly, and in the far field the arrival
// density at angle k is cos^2((k d + phi)/2) fringes under the
// single-slit sinc^2(k w / 2) envelope. phi is the Aharonov-Bohm phase of
// a solenoid tucked behind the wall between the slits (its exact reduced
// effect: slit 2 is multiplied by e^{i phi}) -- fringes shift by phi/d,
// the envelope does not move (Chambers' experiment).


export namespace ses {

// Aperture of an infinite wall pierced by two slits of width w centered at
// x = -d/2 and x = +d/2; the +x slit carries the solenoid phase e^{i phi}.
inline std::vector<std::complex<double>> double_slit_aperture(
    const Grid1D& g, double d, double w, double phi) {
    std::vector<std::complex<double>> ap(static_cast<std::size_t>(g.n));
    const std::complex<double> ab{std::cos(phi), std::sin(phi)};
    for (int i = 0; i < g.n; ++i) {
        const double x = g.coord(i);
        if (std::abs(x + 0.5 * d) <= 0.5 * w) {
            ap[static_cast<std::size_t>(i)] = 1.0;
        } else if (std::abs(x - 0.5 * d) <= 0.5 * w) {
            ap[static_cast<std::size_t>(i)] = ab;
        }
    }
    return ap;
}

// psi -> aperture * psi, renormalized (post-selection on the transmitted
// electron); returns the transmitted fraction the wall did not absorb.
// A vanishing overlap (both slits outside the beam) leaves psi untouched.
inline double apply_aperture(Field1D& psi,
                             const std::vector<std::complex<double>>& ap) {
    double frac = 0.0;
    for (int i = 0; i < psi.size(); ++i) {
        frac += std::norm(ap[static_cast<std::size_t>(i)] * psi[i]);
    }
    frac *= psi.grid().spacing();
    if (frac <= 0.0) {
        return 0.0;
    }
    const double inv = 1.0 / std::sqrt(frac);
    for (int i = 0; i < psi.size(); ++i) {
        psi[i] = inv * ap[static_cast<std::size_t>(i)] * psi[i];
    }
    return frac;
}

// The FFT-bin wavenumbers rearranged ascending -- the screen's angle axis.
inline std::vector<double> momentum_axis(const Grid1D& g) {
    const std::vector<double> k = wavenumbers(g);
    // wavenumbers() is in FFT bin order (0..k_max, then negative); the
    // ascending order is the negative half followed by the positive half.
    std::vector<double> axis;
    axis.reserve(k.size());
    const std::size_t half = (k.size() + 1) / 2;  // bins 0..half-1 are >= 0
    for (std::size_t j = half; j < k.size(); ++j) {
        axis.push_back(k[j]);
    }
    for (std::size_t j = 0; j < half; ++j) {
        axis.push_back(k[j]);
    }
    return axis;
}

// |psi~(k)|^2 aligned to momentum_axis, normalized so that
// sum * (2 pi / L) = 1 -- the far-field screen (arrival density per unit
// angle-wavenumber), CONSTANT under free flight.
inline std::vector<double> momentum_spectrum(const Field1D& psi) {
    const Grid1D& g = psi.grid();
    std::vector<std::complex<double>> ft = psi.data();
    fft(ft);
    // Parseval on the grid: sum |ft|^2 = n * sum |psi|^2. Normalize |psi~|^2
    // to unit integral over k (bin width 2 pi / L).
    double total = 0.0;
    for (const std::complex<double>& z : ft) {
        total += std::norm(z);
    }
    const double l = g.xmax - g.xmin;
    const double bin = 2.0 * 3.14159265358979323846 / l;
    const double scale = total > 0.0 ? 1.0 / (total * bin) : 0.0;
    std::vector<double> spec;
    spec.reserve(ft.size());
    const std::size_t half = (ft.size() + 1) / 2;
    for (std::size_t j = half; j < ft.size(); ++j) {
        spec.push_back(std::norm(ft[j]) * scale);
    }
    for (std::size_t j = 0; j < half; ++j) {
        spec.push_back(std::norm(ft[j]) * scale);
    }
    return spec;
}

}  // namespace ses
