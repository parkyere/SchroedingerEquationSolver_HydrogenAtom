module;
#include <complex>
#include <algorithm>
export module ses.flow;
import ses.vec;


// Atomic units (hbar = m_e = 1). std::complex rides in the GMF (not exported):
// consumers naming it must #include <complex> themselves.


export namespace ses {

// Probability current j = Im(conj(psi) grad psi) = rho v (Madelung).
inline Vec3d probability_current(std::complex<double> psi, std::complex<double> dpsi_dx,
                                 std::complex<double> dpsi_dy,
                                 std::complex<double> dpsi_dz) noexcept {
    const auto jc = [&](std::complex<double> d) {
        return psi.real() * d.imag() - psi.imag() * d.real();
    };
    return Vec3d{jc(dpsi_dx), jc(dpsi_dy), jc(dpsi_dz)};
}

// Bohmian velocity v = j / rho, rho = |psi|^2; guarded where rho->0 (nodes).
inline Vec3d bohmian_velocity(std::complex<double> psi, std::complex<double> dpsi_dx,
                              std::complex<double> dpsi_dy,
                              std::complex<double> dpsi_dz) noexcept {
    const double rho = psi.real() * psi.real() + psi.imag() * psi.imag();
    const Vec3d j = probability_current(psi, dpsi_dx, dpsi_dy, dpsi_dz);
    const double inv = rho > 1e-12 ? 1.0 / rho : 0.0;
    return Vec3d{j.x * inv, j.y * inv, j.z * inv};
}

// Trail fade: oldest tail transparent -> newest head opaque.
inline double trail_fade(int v, int trail_len) noexcept {
    if (trail_len <= 1) {
        return 1.0;
    }
    const double t =
        static_cast<double>(v) / static_cast<double>(trail_len - 1);
    return std::clamp(t, 0.0, 1.0);
}

}  // namespace ses
