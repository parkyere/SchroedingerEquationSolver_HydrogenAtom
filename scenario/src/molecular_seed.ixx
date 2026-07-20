module;
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <complex>
#include <numbers>
#include <random>
#include <vector>
export module ses.scenario.molecular_seed;
export import ses.field;
export import ses.grid;


// H2+ one-electron MO seeds (bond axis = x). Deflated ITP seeded in an irrep
// sector lands that irrep's lowest orbital; SigmaG2 (2sigma_g) needs
// deflation vs 1sigma_g.


export namespace ses_shell {

enum class MolOrbital { SigmaG, SigmaU, PiUy, PiUz, SigmaG2 };

// Only the irrep matters; ITP polishes the radial profile.
ses::Field3D molecular_orbital_seed(const ses::Grid3D& g, MolOrbital sym) {
    constexpr double kSeedSigma = 2.0;  // Bohr
    const double s2 = 4.0 * kSeedSigma * kSeedSigma;
    ses::Field3D f{g};
    for (int k = 0; k < g.z.n; ++k) {
        const double z = g.z.coord(k);
        for (int j = 0; j < g.y.n; ++j) {
            const double y = g.y.coord(j);
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i);
                const double env = std::exp(-(x * x + y * y + z * z) / s2);
                double factor = 1.0;
                switch (sym) {
                    case MolOrbital::SigmaG:
                        factor = 1.0;
                        break;
                    case MolOrbital::SigmaU:
                        factor = x;
                        break;
                    case MolOrbital::PiUy:
                        factor = y;
                        break;
                    case MolOrbital::PiUz:
                        factor = z;
                        break;
                    case MolOrbital::SigmaG2:
                        factor = 1.0 - (x * x + y * y + z * z) / s2;
                        break;
                }
                f(i, j, k) = std::complex<double>{factor * env, 0.0};
            }
        }
    }
    ses::normalize(f);
    return f;
}

// Symmetry-broken seed: random-phase Gaussian blobs, deterministic in `seed`.
ses::Field3D random_molecular_seed(const ses::Grid3D& g, std::uint64_t seed) {
    constexpr int kBlobs = 5;
    constexpr double kReach = 0.30;
    constexpr double kBlobSigma = 2.5;
    const double s2 = 4.0 * kBlobSigma * kBlobSigma;
    const double bx = std::max(std::abs(g.x.xmin), std::abs(g.x.xmax));
    const double by = std::max(std::abs(g.y.xmin), std::abs(g.y.xmax));
    const double bz = std::max(std::abs(g.z.xmin), std::abs(g.z.xmax));

    std::mt19937_64 rng{seed};
    std::uniform_real_distribution<double> pos(-1.0, 1.0);
    std::uniform_real_distribution<double> phase(0.0,
                                                 2.0 * std::numbers::pi);
    std::uniform_real_distribution<double> amp(0.5, 1.0);

    struct Blob {
        double cx, cy, cz;
        std::complex<double> w;
    };
    std::vector<Blob> blobs(kBlobs);
    for (Blob& b : blobs) {
        b.cx = kReach * bx * pos(rng);
        b.cy = kReach * by * pos(rng);
        b.cz = kReach * bz * pos(rng);
        const double ph = phase(rng);
        b.w = amp(rng) * std::complex<double>{std::cos(ph), std::sin(ph)};
    }

    ses::Field3D f{g};
    for (int k = 0; k < g.z.n; ++k) {
        const double z = g.z.coord(k);
        for (int j = 0; j < g.y.n; ++j) {
            const double y = g.y.coord(j);
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i);
                std::complex<double> acc{};
                for (const Blob& b : blobs) {
                    const double dx = x - b.cx;
                    const double dy = y - b.cy;
                    const double dz = z - b.cz;
                    acc += b.w * std::exp(-(dx * dx + dy * dy + dz * dz) / s2);
                }
                f(i, j, k) = acc;
            }
        }
    }
    ses::normalize(f);
    return f;
}

}  // namespace ses_shell
