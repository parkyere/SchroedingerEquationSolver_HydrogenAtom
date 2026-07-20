module;
#include <complex>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>
export module ses.fft;
export import ses.field;
import ses.parallel;


// Hand-written radix-2 Cooley-Tukey FFT (purist reinvention boundary: no FFTW).
//
// Convention (matches FFTW/NumPy, pinned by tests/fft_test.cpp):
//     forward:  X_k = sum_n x_n e^{-2 pi i k n / N}   (unnormalized)
//     inverse:  x_n = (1/N) sum_k X_k e^{+2 pi i k n / N}


export namespace ses {

// Twiddle table w[j] = e^{-2 pi i j / n}, computed directly not by the w *= wlen
// recurrence (drifts off the unit circle, damps norm; caught by
// SplitOperator.ConservesNorm). Stateless (no thread_local): 3D passes share one
// read-only table per axis, cheaper than a per-thread cache.
inline std::vector<std::complex<double>> fft_twiddles(std::size_t n) {
    std::vector<std::complex<double>> w(n / 2);
    const double ang = -2.0 * std::numbers::pi / static_cast<double>(n);
    for (std::size_t j = 0; j < n / 2; ++j) {
        const double th = ang * static_cast<double>(j);
        w[j] = std::complex<double>{std::cos(th), std::sin(th)};
    }
    return w;
}

inline void fft(std::complex<double>* a, std::size_t n, const std::complex<double>* w) {
    // Runtime guard not assert: under NDEBUG a mis-sized axis spins/garbles
    // (bit-reversal assumes a power of two).
    if ((n & (n - 1)) != 0) {
        throw std::invalid_argument("ses::fft: size must be a power of two");
    }
    if (n < 2) {
        return;
    }

    // Bit-reversal permutation.
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }

    // Butterfly passes. Stage len uses w_len^j = w[j * n/len].
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const std::size_t stride = n / len;
        for (std::size_t i = 0; i < n; i += len) {
            for (std::size_t j = 0; j < len / 2; ++j) {
                const std::complex<double> u = a[i + j];
                const std::complex<double> v = a[i + j + len / 2] * w[j * stride];
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
            }
        }
    }
}

inline void fft(std::complex<double>* a, std::size_t n) {
    if ((n & (n - 1)) != 0) {
        throw std::invalid_argument("ses::fft: size must be a power of two");
    }
    if (n < 2) {
        return;
    }
    const std::vector<std::complex<double>> w = fft_twiddles(n);
    fft(a, n, w.data());
}

inline void fft(std::vector<std::complex<double>>& a) { fft(a.data(), a.size()); }

// Conjugation identity: ifft(X) = conj(fft(conj(X))) / N
inline void ifft(std::vector<std::complex<double>>& a) {
    for (std::complex<double>& z : a) {
        z = conj(z);
    }
    fft(a);
    const double inv = 1.0 / static_cast<double>(a.size());
    for (std::complex<double>& z : a) {
        z = inv * conj(z);
    }
}

// 3D forward: 1D FFT per axis. x-lines contiguous; y/z gathered into per-worker
// scratch. Each line owned by one worker -> bitwise identical to serial.
inline void fft(Field3D& f) {
    std::vector<std::complex<double>>& a = f.data();
    const Grid3D& g = f.grid();
    const int nx = g.x.n;
    const int ny = g.y.n;
    const int nz = g.z.n;

    // Validate up front: throw on the caller's thread, never inside the pool.
    if (((nx & (nx - 1)) | (ny & (ny - 1)) | (nz & (nz - 1))) != 0) {
        throw std::invalid_argument("ses::fft: size must be a power of two");
    }

    {
        const std::vector<std::complex<double>> w =
            fft_twiddles(static_cast<std::size_t>(nx));
        parallel_for(nz * ny, [&](int m) {
            const int k = m / ny;
            const int j = m % ny;
            fft(a.data() + g.flat(0, j, k), static_cast<std::size_t>(nx), w.data());
        });
    }

    // Scratch line per worker, reused across the y and z passes.
    std::vector<std::vector<std::complex<double>>> scratch(
        static_cast<std::size_t>(parallel_workers()));

    {
        const std::vector<std::complex<double>> w =
            fft_twiddles(static_cast<std::size_t>(ny));
        parallel_ranges(nz * nx, [&](int worker, int begin, int end) {
            std::vector<std::complex<double>>& line =
                scratch[static_cast<std::size_t>(worker)];
            line.resize(static_cast<std::size_t>(ny));
            for (int m = begin; m < end; ++m) {
                const int k = m / nx;
                const int i = m % nx;
                for (int j = 0; j < ny; ++j) {
                    line[static_cast<std::size_t>(j)] =
                        a[static_cast<std::size_t>(g.flat(i, j, k))];
                }
                fft(line.data(), line.size(), w.data());
                for (int j = 0; j < ny; ++j) {
                    a[static_cast<std::size_t>(g.flat(i, j, k))] =
                        line[static_cast<std::size_t>(j)];
                }
            }
        });
    }

    {
        const std::vector<std::complex<double>> w =
            fft_twiddles(static_cast<std::size_t>(nz));
        parallel_ranges(ny * nx, [&](int worker, int begin, int end) {
            std::vector<std::complex<double>>& line =
                scratch[static_cast<std::size_t>(worker)];
            line.resize(static_cast<std::size_t>(nz));
            for (int m = begin; m < end; ++m) {
                const int j = m / nx;
                const int i = m % nx;
                for (int k = 0; k < nz; ++k) {
                    line[static_cast<std::size_t>(k)] =
                        a[static_cast<std::size_t>(g.flat(i, j, k))];
                }
                fft(line.data(), line.size(), w.data());
                for (int k = 0; k < nz; ++k) {
                    a[static_cast<std::size_t>(g.flat(i, j, k))] =
                        line[static_cast<std::size_t>(k)];
                }
            }
        });
    }
}

// 3D inverse: conjugation identity. Elementwise loops threaded
// (disjoint -> bitwise identical to serial).
inline void ifft(Field3D& f) {
    std::vector<std::complex<double>>& a = f.data();
    const int n = static_cast<int>(a.size());
    parallel_for(n, [&](int i) {
        a[static_cast<std::size_t>(i)] = conj(a[static_cast<std::size_t>(i)]);
    });
    fft(f);
    const double inv = 1.0 / static_cast<double>(f.size());
    parallel_for(n, [&](int i) {
        a[static_cast<std::size_t>(i)] = inv * conj(a[static_cast<std::size_t>(i)]);
    });
}

}  // namespace ses
