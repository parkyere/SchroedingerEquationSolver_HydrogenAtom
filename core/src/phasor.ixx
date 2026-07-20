module;
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>
export module ses.phasor;
export import ses.colormap;
export import ses.field;
export import ses.grid;
import ses.parallel;


export namespace ses {

// r = r_scale |psi|^2 radius, arg(psi) twist about x; via |psi|*(Re,Im), no transcendentals.
inline std::vector<float> phasor_curve(const Field1D& psi, double r_scale) {
    const Grid1D& g = psi.grid();
    std::vector<float> v(static_cast<std::size_t>(3 * g.n));
    parallel_for(g.n, [&](int i) {
        const double a = r_scale * std::abs(psi[i]);
        const std::size_t o = static_cast<std::size_t>(3 * i);
        v[o + 0] = static_cast<float>(g.coord(i));
        v[o + 1] = static_cast<float>(a * psi[i].real());
        v[o + 2] = static_cast<float>(a * psi[i].imag());
    });
    return v;
}

// z=0 shadow band, y in [-r,+r], r = r_scale |psi|^2; TRIANGLE_STRIP two verts/point (lower,upper winding).
inline std::vector<float> density_band(const Field1D& psi, double r_scale) {
    const Grid1D& g = psi.grid();
    std::vector<float> v(static_cast<std::size_t>(6 * g.n));
    parallel_for(g.n, [&](int i) {
        const float x = static_cast<float>(g.coord(i));
        const float r = static_cast<float>(r_scale * std::norm(psi[i]));
        const std::size_t o = static_cast<std::size_t>(6 * i);
        v[o + 0] = x;
        v[o + 1] = -r;
        v[o + 2] = 0.0f;
        v[o + 3] = x;
        v[o + 4] = r;
        v[o + 5] = 0.0f;
    });
    return v;
}

// rgba for density_band; same phase_color as volume view (cross-scene hue), premultiplied for overlay blend.
inline std::vector<float> phase_band_colors(const Field1D& psi, float alpha) {
    const Grid1D& g = psi.grid();
    std::vector<float> c(static_cast<std::size_t>(8 * g.n));
    // per-point atan2: pool-chunked, disjoint writes -> deterministic.
    parallel_for(g.n, [&](int i) {
        const Rgb col = phase_color(std::arg(psi[i]));
        const std::size_t o = static_cast<std::size_t>(8 * i);
        const float r = static_cast<float>(col.r) * alpha;
        const float gg = static_cast<float>(col.g) * alpha;
        const float b = static_cast<float>(col.b) * alpha;
        c[o + 0] = r;
        c[o + 1] = gg;
        c[o + 2] = b;
        c[o + 3] = alpha;
        c[o + 4] = r;
        c[o + 5] = gg;
        c[o + 6] = b;
        c[o + 7] = alpha;
    });
    return c;
}

// z=0 potential polyline; y clamped to y_clamp so steep walls stay in frame.
inline std::vector<float> potential_curve(const Grid1D& g, const std::vector<double>& pot,
                                          double e_scale, double y_clamp) {
    std::vector<float> v(static_cast<std::size_t>(3 * g.n));
    for (int i = 0; i < g.n; ++i) {
        const std::size_t o = static_cast<std::size_t>(3 * i);
        const double y = std::min(pot[static_cast<std::size_t>(i)] * e_scale, y_clamp);
        v[o + 0] = static_cast<float>(g.coord(i));
        v[o + 1] = static_cast<float>(y);
        v[o + 2] = 0.0f;
    }
    return v;
}

}  // namespace ses
