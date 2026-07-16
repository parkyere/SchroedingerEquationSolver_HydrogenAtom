// RED: specification for ses.parallel -- the project's own worker-pool
// parallelism, replacing OpenMP (whose pragmas MSVC miscompiles or crashes
// inside C++20 module interfaces; see core/src/parallel.ixx).
//
// Contract:
//  - parallel_for(n, body):    body(i) once per i in [0, n); disjoint writes.
//  - parallel_sum(n, init, f): sum of f(i); chunk boundaries depend only on n
//    and partials combine in chunk order, so the result is BITWISE identical
//    run-to-run and for any worker count (OpenMP reduction never promised
//    this; the CPU is the oracle, so determinism outranks speed).
//  - parallel_ranges(n, body): body(worker, begin, end) over disjoint chunks
//    covering [0, n); 0 <= worker < parallel_workers() indexes per-worker
//    scratch (the thread_local replacement).
//  - Nested calls (a body that itself calls parallel_*) must not deadlock.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

import ses.parallel;

namespace {

TEST(ParallelWorkers, AtLeastOne) {
    EXPECT_GE(ses::parallel_workers(), 1);
}

TEST(ParallelFor, CoversEveryIndexExactlyOnce) {
    for (const int n : {0, 1, 3, 64, 10007}) {
        std::vector<int> hits(static_cast<std::size_t>(n), 0);
        ses::parallel_for(n, [&](int i) { ++hits[static_cast<std::size_t>(i)]; });
        for (int i = 0; i < n; ++i) {
            ASSERT_EQ(hits[static_cast<std::size_t>(i)], 1) << "n=" << n << " i=" << i;
        }
    }
}

TEST(ParallelFor, NestedCallDoesNotDeadlockAndStaysCorrect) {
    const int outer = 4;
    const int inner = 1000;
    std::vector<std::int64_t> sums(outer, 0);
    ses::parallel_for(outer, [&](int o) {
        // Serial accumulation into this iteration's own slot; the inner
        // parallel_for exercises re-entrancy, not shared writes.
        std::int64_t s = 0;
        std::vector<int> hits(inner, 0);
        ses::parallel_for(inner, [&](int i) { ++hits[static_cast<std::size_t>(i)]; });
        for (int i = 0; i < inner; ++i) {
            s += hits[static_cast<std::size_t>(i)] * (i + 1);
        }
        sums[static_cast<std::size_t>(o)] = s;
    });
    const std::int64_t expect = static_cast<std::int64_t>(inner) * (inner + 1) / 2;
    for (int o = 0; o < outer; ++o) {
        EXPECT_EQ(sums[static_cast<std::size_t>(o)], expect);
    }
}

// Adversarial magnitudes (e^-30 .. e^+30, alternating signs): any change in
// summation order shows up in the low bits, so bitwise-equal repeats prove
// the combine order is fixed.
TEST(ParallelSum, BitwiseDeterministicAcrossRuns) {
    const int n = 4001;
    auto term = [](int i) {
        const double m = std::exp(30.0 * std::sin(0.7 * i));
        return (i % 2 == 0) ? m : -m;
    };
    const double first = ses::parallel_sum(n, 0.0, term);
    for (int rep = 0; rep < 50; ++rep) {
        const double again = ses::parallel_sum(n, 0.0, term);
        ASSERT_EQ(first, again) << "rep=" << rep;  // bitwise, not approx
    }
    double serial = 0.0;
    for (int i = 0; i < n; ++i) {
        serial += term(i);
    }
    EXPECT_NEAR(first, serial, 1e-9 * std::abs(serial));
}

TEST(ParallelSum, ComplexAccumulatorAndEmptyRange) {
    const int n = 513;
    auto term = [](int i) {
        return std::complex<double>{std::cos(0.1 * i), std::sin(0.1 * i)};
    };
    const std::complex<double> par = ses::parallel_sum(n, std::complex<double>{}, term);
    std::complex<double> serial{};
    for (int i = 0; i < n; ++i) {
        serial += term(i);
    }
    EXPECT_NEAR(par.real(), serial.real(), 1e-12);
    EXPECT_NEAR(par.imag(), serial.imag(), 1e-12);
    EXPECT_EQ(ses::parallel_sum(0, 42.0, [](int) { return 1.0; }), 42.0);
}

TEST(ParallelRanges, DisjointCoverageAndWorkerIndexBounds) {
    const int n = 12345;
    const int workers = ses::parallel_workers();
    std::vector<int> hits(static_cast<std::size_t>(n), 0);
    std::vector<int> used(static_cast<std::size_t>(workers), 0);
    ses::parallel_ranges(n, [&](int worker, int begin, int end) {
        ASSERT_GE(worker, 0);
        ASSERT_LT(worker, workers);
        ASSERT_LT(begin, end);
        used[static_cast<std::size_t>(worker)] = 1;  // own slot: no race
        for (int i = begin; i < end; ++i) {
            ++hits[static_cast<std::size_t>(i)];
        }
    });
    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(hits[static_cast<std::size_t>(i)], 1) << "i=" << i;
    }
    ses::parallel_ranges(0, [&](int, int, int) { FAIL() << "n=0 must not call body"; });
}

}  // namespace
