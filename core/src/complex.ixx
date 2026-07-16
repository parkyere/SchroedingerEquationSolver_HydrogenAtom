module;
#include <complex>
export module ses.complex;

export namespace ses {
template <typename T>
using Complex = std::complex<T>;

template <typename T>
constexpr T norm_sq(const Complex<T>& z) noexcept {
    return std::norm(z);
}
}  // namespace ses
