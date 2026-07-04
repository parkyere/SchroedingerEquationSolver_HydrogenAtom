// Contract specification for ses::Complex<T>.
//
// The reinvention boundary excludes the C++ STANDARD LIBRARY (only third-
// party libraries are reinvented), so ses::Complex is an alias of
// std::complex plus the norm_sq convenience. These tests pin the arithmetic
// contract the whole core relies on -- construction, +,-,*,/ (built with
// -fcx-limited-range: the naive formulas, no Annex G NaN fixups), scalar
// multiply, conj (via ADL into std), abs, and norm_sq == |z|^2.
//
// Oracle: exact arithmetic identities (i*i = -1, |3-4i| = 5, ...). All simple
// operations use values exactly representable in binary floating point, so
// EXPECT_EQ is legitimate there; sqrt/division results use EXPECT_DOUBLE_EQ.

#include <core/complex.hpp>

#include <gtest/gtest.h>

namespace {

using Cd = ses::Complex<double>;

TEST(Complex, DefaultConstructsToZero) {
    constexpr Cd z{};
    EXPECT_EQ(z.real(), 0.0);
    EXPECT_EQ(z.imag(), 0.0);
}

TEST(Complex, AggregateConstruction) {
    constexpr Cd z{3.0, -4.0};
    EXPECT_EQ(z.real(), 3.0);
    EXPECT_EQ(z.imag(), -4.0);
}

TEST(Complex, Addition) {
    constexpr Cd s = Cd{1.0, 2.0} + Cd{3.0, -5.0};
    EXPECT_EQ(s.real(), 4.0);
    EXPECT_EQ(s.imag(), -3.0);
}

TEST(Complex, Subtraction) {
    constexpr Cd d = Cd{1.0, 2.0} - Cd{3.0, -5.0};
    EXPECT_EQ(d.real(), -2.0);
    EXPECT_EQ(d.imag(), 7.0);
}

TEST(Complex, MultiplicationSatisfiesISquaredEqualsMinusOne) {
    constexpr Cd i{0.0, 1.0};
    constexpr Cd ii = i * i;
    EXPECT_EQ(ii.real(), -1.0);
    EXPECT_EQ(ii.imag(), 0.0);
}

TEST(Complex, MultiplicationGeneralCase) {
    // (1+2i)(3+4i) = 3 + 4i + 6i + 8i^2 = -5 + 10i
    constexpr Cd p = Cd{1.0, 2.0} * Cd{3.0, 4.0};
    EXPECT_EQ(p.real(), -5.0);
    EXPECT_EQ(p.imag(), 10.0);
}

TEST(Complex, ScalarMultiplicationFromBothSides) {
    constexpr Cd l = 2.0 * Cd{1.0, -3.0};
    constexpr Cd r = Cd{1.0, -3.0} * 2.0;
    EXPECT_EQ(l.real(), 2.0);
    EXPECT_EQ(l.imag(), -6.0);
    EXPECT_EQ(r.real(), 2.0);
    EXPECT_EQ(r.imag(), -6.0);
}

TEST(Complex, Conjugate) {
    constexpr Cd c = conj(Cd{3.0, -4.0});
    EXPECT_EQ(c.real(), 3.0);
    EXPECT_EQ(c.imag(), 4.0);
}

TEST(Complex, NormSquaredIsSquaredMagnitude) {
    // |3-4i|^2 = 9 + 16 = 25 (this is the probability-density operation).
    // Qualified call: ADL associates std::complex with std, not ses.
    constexpr double n = ses::norm_sq(Cd{3.0, -4.0});
    EXPECT_EQ(n, 25.0);
}

TEST(Complex, AbsIsMagnitude) {
    EXPECT_DOUBLE_EQ(abs(Cd{3.0, -4.0}), 5.0);
}

TEST(Complex, DivisionByComplex) {
    // (-5+10i)/(3+4i) = (1+2i)  [inverse of the multiplication case above]
    const Cd q = Cd{-5.0, 10.0} / Cd{3.0, 4.0};
    EXPECT_DOUBLE_EQ(q.real(), 1.0);
    EXPECT_DOUBLE_EQ(q.imag(), 2.0);
}

TEST(Complex, MultiplicationConjugateGivesRealNormSq) {
    // z * conj(z) must be purely real and equal |z|^2
    constexpr Cd z{3.0, -4.0};
    constexpr Cd zz = z * conj(z);
    EXPECT_EQ(zz.real(), 25.0);
    EXPECT_EQ(zz.imag(), 0.0);
}

}  // namespace
