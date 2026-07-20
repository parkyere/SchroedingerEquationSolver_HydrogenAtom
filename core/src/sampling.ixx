module;
#include <complex>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>
export module ses.sampling;
export import ses.grid;
export import ses.vec;
export import ses.marching_cubes;
export import ses.field;
export import ses.colormap;


// atan2 the interpolated COMPLEX value, not the phase directly: constant-phase
// regions stay constant, amplitude cancels in the ratio.


export namespace ses {

namespace sampling_detail {

// clamp i to n-2 so i+1 stays in range; t=1 covers the last cell.
inline std::pair<int, double> cell_and_t(double u, const Grid1D& axis) noexcept {
    const double s = (u - axis.xmin) / axis.spacing();
    int i = static_cast<int>(std::floor(s));
    i = std::clamp(i, 0, axis.n - 2);
    return {i, s - i};
}

}  // namespace sampling_detail

inline std::complex<double> sample_trilinear(const Field3D& f, Vec3d p) noexcept {
    const Grid3D& g = f.grid();
    const auto [i, tx] = sampling_detail::cell_and_t(p.x, g.x);
    const auto [j, ty] = sampling_detail::cell_and_t(p.y, g.y);
    const auto [k, tz] = sampling_detail::cell_and_t(p.z, g.z);

    auto lerp = [](std::complex<double> a, std::complex<double> b, double t) {
        return a + t * (b - a);
    };

    const std::complex<double> c00 = lerp(f(i, j, k), f(i + 1, j, k), tx);
    const std::complex<double> c10 = lerp(f(i, j + 1, k), f(i + 1, j + 1, k), tx);
    const std::complex<double> c01 = lerp(f(i, j, k + 1), f(i + 1, j, k + 1), tx);
    const std::complex<double> c11 = lerp(f(i, j + 1, k + 1), f(i + 1, j + 1, k + 1), tx);
    return lerp(lerp(c00, c10, ty), lerp(c01, c11, ty), tz);
}

inline std::vector<Rgb> phase_colors(const Mesh& mesh, const Field3D& psi) {
    std::vector<Rgb> colors;
    colors.reserve(mesh.vertices.size());
    for (const Vec3d& v : mesh.vertices) {
        const std::complex<double> s = sample_trilinear(psi, v);
        colors.push_back(phase_color(std::atan2(s.imag(), s.real())));
    }
    return colors;
}

}  // namespace ses
