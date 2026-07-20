module;
#include <utility>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <vector>
export module ses.volume;

import ses.colormap;
import ses.vec;


// Shader ray marcher mirrors these formulas; correctness pinned here
// (tests/volume_test.cpp) since shaders can't be unit-tested.


export namespace ses {

struct RayHit {
    bool hit{};
    double t_near{};
    double t_far{};
};

// Slab method. Raw interval: t_near < 0 means origin inside; caller clamps max(t_near, 0).
inline constexpr RayHit ray_box(Vec3d origin, Vec3d dir, Vec3d box_min, Vec3d box_max) noexcept {
    double t_near = -std::numeric_limits<double>::infinity();
    double t_far = std::numeric_limits<double>::infinity();

    const double o[3] = {origin.x, origin.y, origin.z};
    const double d[3] = {dir.x, dir.y, dir.z};
    const double lo[3] = {box_min.x, box_min.y, box_min.z};
    const double hi[3] = {box_max.x, box_max.y, box_max.z};

    for (int axis = 0; axis < 3; ++axis) {
        if (d[axis] == 0.0) {
            if (o[axis] < lo[axis] || o[axis] > hi[axis]) {
                return RayHit{false, 0.0, 0.0};
            }
            continue;
        }
        double t1 = (lo[axis] - o[axis]) / d[axis];
        double t2 = (hi[axis] - o[axis]) / d[axis];
        if (t1 > t2) {
            std::swap(t1, t2);
        }
        t_near = std::max(t_near, t1);
        t_far = std::min(t_far, t2);
        if (t_near > t_far) {
            return RayHit{false, 0.0, 0.0};
        }
    }
    return RayHit{true, t_near, t_far};
}

// dir must be unit length. Raw interval: t_near < 0 when origin inside; tangent -> t_near == t_far.
inline RayHit ray_sphere(Vec3d origin, Vec3d dir, Vec3d center, double radius) noexcept {
    const Vec3d oc = origin - center;
    const double b = dot(oc, dir);
    const double c = dot(oc, oc) - radius * radius;
    const double disc = b * b - c;
    if (disc < 0.0) {
        return RayHit{false, 0.0, 0.0};
    }
    const double s = std::sqrt(disc);
    return RayHit{true, -b - s, -b + s};
}

// Beer-Lambert opacity of one step.
inline double sample_alpha(double density01, double absorbance, double step) noexcept {
    return 1.0 - std::exp(-absorbance * density01 * step);
}

struct Rgba {
    double r{};
    double g{};
    double b{};
    double a{};
};

struct VolumeSample {
    Rgb color;
    double alpha{};
};

// Premultiplied front-to-back compositing; the saturation early-out is a pure optimization.
inline Rgba composite_front_to_back(const std::vector<VolumeSample>& samples) noexcept {
    Rgba out{};
    for (const VolumeSample& s : samples) {
        const double w = (1.0 - out.a) * s.alpha;
        out.r += w * s.color.r;
        out.g += w * s.color.g;
        out.b += w * s.color.b;
        out.a += w;
        if (out.a >= 0.999) {
            break;
        }
    }
    return out;
}

// phase_color baked to a LUT at bin centers so the GPU samples verified colors, not a re-derived formula.
inline std::vector<Rgb> phase_lut(int n) {
    std::vector<Rgb> lut(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double theta =
            -std::numbers::pi + 2.0 * std::numbers::pi * (i + 0.5) / n;
        lut[static_cast<std::size_t>(i)] = phase_color(theta);
    }
    return lut;
}

}  // namespace ses
