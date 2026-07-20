export module ses.grid;


// Periodic grid (required by the split-operator Fourier propagator): xmax aliases
// to xmin -- not a grid point -- so spacing divides by n, not n-1.


export namespace ses {

struct Grid1D {
    double xmin{};
    double xmax{};
    int n{};

    constexpr int size() const noexcept { return n; }
    constexpr double spacing() const noexcept { return (xmax - xmin) / n; }
    constexpr double coord(int i) const noexcept { return xmin + i * spacing(); }
};

// Flat layout x-fastest: x-lines contiguous for the 3D FFT, and matches the
// row-major order GPU 3D-texture uploads expect.
struct Grid3D {
    Grid1D x{};
    Grid1D y{};
    Grid1D z{};

    constexpr int size() const noexcept { return x.n * y.n * z.n; }
    constexpr int flat(int i, int j, int k) const noexcept { return i + x.n * (j + y.n * k); }
    constexpr double cell_volume() const noexcept { return x.spacing() * y.spacing() * z.spacing(); }
};

}  // namespace ses
