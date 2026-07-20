module;
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>
export module ses.emission;
export import ses.decay;
export import ses.field;
import ses.vec;
import ses.grid;


// Semiclassical (Larmor) dipole emission, atomic units. Coherent-superposition
// only: P == 0 for a pure eigenstate (its decay = Einstein-A jumps, ses.decay).
// P = (2/3) alpha^3 |d_ddot|^2, d_ddot = <grad V> (Ehrenfest).


export namespace ses {

// PRECONDITION: psi normalized; result is the raw integral (NOT norm-invariant),
// so unnormalized psi scales it by the norm. GPU mean_force oracle shares this.
inline Vec3d mean_potential_gradient(const Field3D& psi, const std::vector<double>& v,
                                     const Grid3D& g) noexcept {
    const int nx = g.x.n;
    const int ny = g.y.n;
    const int nz = g.z.n;
    const double inv2hx = 1.0 / (2.0 * g.x.spacing());
    const double inv2hy = 1.0 / (2.0 * g.y.spacing());
    const double inv2hz = 1.0 / (2.0 * g.z.spacing());
    Vec3d acc{};
    for (int k = 0; k < nz; ++k) {
        const int kp = (k + 1) % nz;
        const int km = (k - 1 + nz) % nz;
        for (int j = 0; j < ny; ++j) {
            const int jp = (j + 1) % ny;
            const int jm = (j - 1 + ny) % ny;
            for (int i = 0; i < nx; ++i) {
                const int ip = (i + 1) % nx;
                const int im = (i - 1 + nx) % nx;
                const double rho = std::norm(psi(i, j, k));
                const double gx =
                    (v[static_cast<std::size_t>(g.flat(ip, j, k))] -
                     v[static_cast<std::size_t>(g.flat(im, j, k))]) * inv2hx;
                const double gy =
                    (v[static_cast<std::size_t>(g.flat(i, jp, k))] -
                     v[static_cast<std::size_t>(g.flat(i, jm, k))]) * inv2hy;
                const double gz =
                    (v[static_cast<std::size_t>(g.flat(i, j, kp))] -
                     v[static_cast<std::size_t>(g.flat(i, j, km))]) * inv2hz;
                acc.x += rho * gx;
                acc.y += rho * gy;
                acc.z += rho * gz;
            }
        }
    }
    const double dv = g.cell_volume();
    return Vec3d{acc.x * dv, acc.y * dv, acc.z * dv};
}

inline constexpr double larmor_power(const Vec3d& dipole_accel) noexcept {
    const double a3 = kFineStructureConstant * kFineStructureConstant *
                      kFineStructureConstant;
    return (2.0 / 3.0) * a3 * dot(dipole_accel, dipole_accel);
}

}  // namespace ses
