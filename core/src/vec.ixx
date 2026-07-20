module;
#include <cmath>
export module ses.vec;


export namespace ses {

struct Vec3d {
    double x{};
    double y{};
    double z{};
};

constexpr Vec3d operator+(Vec3d a, Vec3d b) noexcept { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
constexpr Vec3d operator-(Vec3d a, Vec3d b) noexcept { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
constexpr Vec3d operator*(double s, Vec3d v) noexcept { return {s * v.x, s * v.y, s * v.z}; }

constexpr double dot(Vec3d a, Vec3d b) noexcept { return a.x * b.x + a.y * b.y + a.z * b.z; }

// Right-handed convention: cross(x_hat, y_hat) = +z_hat.
constexpr Vec3d cross(Vec3d a, Vec3d b) noexcept {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline double length(Vec3d v) noexcept { return std::sqrt(dot(v, v)); }

inline Vec3d normalized(Vec3d v) noexcept {
    const double inv = 1.0 / length(v);
    return {inv * v.x, inv * v.y, inv * v.z};
}

}  // namespace ses
