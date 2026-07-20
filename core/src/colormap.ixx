module;
#include <algorithm>
#include <cmath>
#include <numbers>
export module ses.colormap;

export namespace ses {

struct Rgb {
    double r{};
    double g{};
    double b{};
};

// HSV hue wheel, S=V=1; theta in [-pi, pi].
inline Rgb phase_color(double theta) noexcept {
    double h = (theta + std::numbers::pi) / (2.0 * std::numbers::pi) * 6.0;
    h = h - 6.0 * std::floor(h / 6.0);
    const double x = 1.0 - std::abs(std::fmod(h, 2.0) - 1.0);
    switch (static_cast<int>(h)) {
        case 0: return {1.0, x, 0.0};
        case 1: return {x, 1.0, 0.0};
        case 2: return {0.0, 1.0, x};
        case 3: return {0.0, x, 1.0};
        case 4: return {x, 0.0, 1.0};
        default: return {1.0, 0.0, x};
    }
}

inline Rgb magnitude_color(double t) noexcept {
    t = std::clamp(t, 0.0, 1.0);
    return {std::pow(t, 1.6), std::pow(t, 0.9), 0.25 + 0.75 * t};
}

}  // namespace ses
