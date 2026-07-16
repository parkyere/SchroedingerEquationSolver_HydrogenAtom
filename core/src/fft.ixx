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
// Sizes must be powers of two.


export namespace ses {

// Twiddle table w[j] = e^{-2 pi i j / n}, j = 0 .. n/2-1, computed directly
// so every factor sits within ~1 ulp of the unit circle. (The recurrence
// w *= wlen drifts off the circle and damps the norm -- caught by
// SplitOperator.ConservesNorm.) Stateless by design (no thread_local: MSVC
// modules miscompile it with omp, and a table per 3D axis pass is cheaper
// than a per-thread cache anyway); the 3D passes share one read-only table.
inline std::vector<std::complex<double>> fft_twiddles(std::size_t n) {
    std::vector<std::complex<double>> w(n / 2);
    const double ang = -2.0 * std::numbers::pi / static_cast<double>(n);
    for (std::size_t j = 0; j < n / 2; ++j) {
        const double th = ang * static_cast<double>(j);
        w[j] = std::complex<double>{std::cos(th), std::sin(th)};
    }
    return w;
}

// In-place forward transform of a contiguous line of length n against a
// prebuilt fft_twiddles(n) table. Iterative: bit-reversal permutation, then
// butterfly passes of doubling length.
inline void fft(std::complex<double>* a, std::size_t n, const std::complex<double>* w) {
    // Runtime guard, not just assert: under NDEBUG a mis-sized axis would
    // silently produce garbage or spin (the bit-reversal loop assumes a
    // power of two).
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

    // Butterfly passes: len = 2, 4, ..., n. Stage len uses w_len^j = w[j * n/len].
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

// In-place inverse transform via the conjugation identity:
//     ifft(X) = conj(fft(conj(X))) / N
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

// 3D forward transform: 1D FFT per axis (x-lines contiguous in the x-fastest
// layout; y/z lines gathered into per-worker scratch). Each line is owned by
// exactly one worker, so the threaded result is BITWISE IDENTICAL to serial.
inline void fft(Field3D& f) {
    std::vector<std::complex<double>>& a = f.data();
    const Grid3D& g = f.grid();
    const int nx = g.x.n;
    const int ny = g.y.n;
    const int nz = g.z.n;

    // Validate every axis up front: a mis-sized grid must throw on the
    // caller's thread, never inside the worker pool.
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

    // One gather/scatter line per worker, reused across the y and z passes
    // (the thread_local scratch of the OpenMP version, made explicit).
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

// 3D inverse: conjugation identity with N = nx*ny*nz. The elementwise loops
// are threaded (disjoint elements: bitwise identical to serial).
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
