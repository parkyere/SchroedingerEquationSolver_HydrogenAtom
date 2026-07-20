module;
#include <complex>
#include <algorithm>
#include <cmath>
export module ses.cross_section;

import ses.colormap;
import ses.vec;
import ses.grid;


// SINGLE SOURCE OF TRUTH mirrored by shaders: volume.frag (clip), slice.vert
// (quad), slice.frag (sample + colour).


export namespace ses {

// Clamp ray [tn, t_stop] to half-space sign*(p[axis]-offset) <= 0, p = eye+t*dir.
struct ClipInterval {
    double tn;
    double t_stop;
    bool visible;
};
inline ClipInterval clip_ray_interval(double tn, double t_stop, double eye_c,
                                      double dir_c, double sign,
                                      double offset) noexcept {
    const double a = sign * dir_c;
    const double b = sign * (offset - eye_c);
    if (std::abs(a) < 1e-8) {
        return ClipInterval{tn, t_stop, b >= 0.0};  // parallel: one side whole
    }
    const double tp = b / a;
    if (a > 0.0) {
        t_stop = std::min(t_stop, tp);
    } else {
        tn = std::max(tn, tp);
    }
    return ClipInterval{tn, t_stop, t_stop > tn};
}

// World pos of quad corner k=0..5 (two CCW tris) on offset-plane along axis.
inline Vec3d slice_quad_corner(int axis, double offset, const Grid3D& g,
                               int k) noexcept {
    static const double kST[6][2] = {{0, 0}, {1, 0}, {1, 1},
                                     {0, 0}, {1, 1}, {0, 1}};
    const int u = (axis + 1) % 3;
    const int w = (axis + 2) % 3;
    const Grid1D* ax[3] = {&g.x, &g.y, &g.z};
    double c[3];
    c[axis] = offset;
    c[u] = ax[u]->xmin + kST[k][0] * (ax[u]->xmax - ax[u]->xmin);
    c[w] = ax[w]->xmin + kST[k][1] * (ax[w]->xmax - ax[w]->xmin);
    return Vec3d{c[0], c[1], c[2]};
}

// map: 0 density, 1 Re(psi), 2 phase; mirrors slice.frag col/alpha.
struct SliceShade {
    Rgb col;
    double alpha;
};
inline SliceShade slice_shade(int map, std::complex<double> psi,
                              double inv_peak) noexcept {
    const double dens = std::norm(psi) * inv_peak;
    const double amp = std::sqrt(std::max(inv_peak, 0.0));
    Rgb col{};
    double bright;
    if (map == 1) {
        const double r = std::clamp(psi.real() * amp, -1.0, 1.0);
        const double m = std::abs(r);
        const Rgb tgt = (r >= 0.0) ? Rgb{1.0, 0.55, 0.15} : Rgb{0.15, 0.45, 1.0};
        col = Rgb{0.03 + (tgt.r - 0.03) * m, 0.03 + (tgt.g - 0.03) * m,
                  0.03 + (tgt.b - 0.03) * m};
        bright = m;
    } else if (map == 2) {
        col = phase_color(std::atan2(psi.imag(), psi.real()));
        bright = std::sqrt(std::clamp(dens, 0.0, 1.0));
        const double tint = 0.25 + 0.75 * bright;
        col = Rgb{col.r * tint, col.g * tint, col.b * tint};
    } else {
        const double d = std::clamp(dens, 0.0, 1.0);
        col = magnitude_color(d);
        bright = d;
    }
    return SliceShade{col, std::clamp(0.45 + 0.5 * bright, 0.0, 0.95)};
}

}  // namespace ses
