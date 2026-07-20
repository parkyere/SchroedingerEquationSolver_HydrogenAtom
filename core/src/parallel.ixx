module;
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
export module ses.parallel;


// Worker pool, not OpenMP: MSVC miscompiles `#pragma omp` in an exported module
// interface. Chunk boundaries depend only on n; partials combine in chunk order
// -> bitwise-identical across worker count and machine (determinism contract).


namespace ses::par_detail {

// One region at a time (region_mutex_); caller is worker 0, pool threads 1..W-1.
// Region ends only after every participant leaves the Job -> no dangling stack reads.
class Pool {
public:
    Pool() {
        const unsigned hc = std::thread::hardware_concurrency();
        workers_ = static_cast<int>(hc == 0 ? 1 : hc);
        if (const char* env = std::getenv("SES_PARALLEL_WORKERS")) {
            const int v = std::atoi(env);
            if (v >= 1) {
                workers_ = v;
            }
        }
        ids_.resize(static_cast<std::size_t>(workers_));
        ids_[0] = std::thread::id{};  // slot 0 = coordinator
        for (int w = 1; w < workers_; ++w) {
            threads_.emplace_back([this, w](std::stop_token st) { worker_loop(st, w); });
            ids_[static_cast<std::size_t>(w)] = threads_.back().get_id();
        }
    }

    int workers() const noexcept { return workers_; }

    // Calling thread's pool index (coordinator -> 0), or -1 if outside any region.
    int current_worker() const noexcept {
        const std::thread::id me = std::this_thread::get_id();
        if (me == coordinator_.load(std::memory_order_acquire)) {
            return 0;
        }
        for (int w = 1; w < workers_; ++w) {
            if (ids_[static_cast<std::size_t>(w)] == me) {
                return w;
            }
        }
        return -1;
    }

    // fn(chunk, worker). Nested call (inside a running region) runs serially on
    // the caller's slot -- no deadlock, scratch stays exclusive.
    void run(int chunks, const std::function<void(int, int)>& fn) {
        if (chunks <= 0) {
            return;
        }
        const int nested_worker = current_worker();
        if (nested_worker >= 0) {
            for (int c = 0; c < chunks; ++c) {
                fn(c, nested_worker);
            }
            return;
        }
        std::lock_guard<std::mutex> region(region_mutex_);
        Job job{&fn, chunks};
        job_.store(&job, std::memory_order_release);
        coordinator_.store(std::this_thread::get_id(), std::memory_order_release);
        gen_.fetch_add(1, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(m_);  // pair with cv_ sleepers
        }
        cv_.notify_all();
        work(job, 0);
        // Spin first: back-to-back micro-regions finish in microseconds; sleeping
        // here would pay a needless wakeup.
        for (int s = 0; s < kSpin; ++s) {
            if (job.exited.load(std::memory_order_acquire) == workers_) {
                break;
            }
        }
        if (job.exited.load(std::memory_order_acquire) != workers_) {
            std::unique_lock<std::mutex> lk(m_);
            cv_done_.wait(lk, [&] {
                return job.exited.load(std::memory_order_acquire) == workers_;
            });
        }
        job_.store(nullptr, std::memory_order_release);
        coordinator_.store(std::thread::id{}, std::memory_order_release);
    }

private:
    struct Job {
        const std::function<void(int, int)>* fn;
        int chunks;
        std::atomic<int> next{0};
        std::atomic<int> exited{0};
    };

    // Spin budget before kernel sleep: waking a deep-C-state core costs tens of us,
    // dominating back-to-back micro-regions. Bounded, then cv-sleep so idle pools park.
    static constexpr int kSpin = 1 << 15;

    void work(Job& job, int worker) {
        for (;;) {
            const int c = job.next.fetch_add(1, std::memory_order_relaxed);
            if (c >= job.chunks) {
                break;
            }
            (*job.fn)(c, worker);
        }
        if (job.exited.fetch_add(1) + 1 == workers_) {
            { std::lock_guard<std::mutex> lk(m_); }  // pair with cv_done_ waiter
            cv_done_.notify_all();
        }
    }

    void worker_loop(std::stop_token st, int worker) {
        std::uint64_t seen = 0;
        for (;;) {
            // Fast path: spin gen_, no lock, no thundering herd.
            bool woke = false;
            for (int s = 0; s < kSpin; ++s) {
                if (gen_.load(std::memory_order_acquire) != seen) {
                    woke = true;
                    break;
                }
                if ((s & 1023) == 1023 && st.stop_requested()) {
                    return;
                }
            }
            if (!woke) {
                std::unique_lock<std::mutex> lk(m_);
                if (!cv_.wait(lk, st, [&] {
                        return gen_.load(std::memory_order_acquire) != seen;
                    })) {
                    return;  // stop requested
                }
            }
            seen = gen_.load(std::memory_order_acquire);
            Job* job = job_.load(std::memory_order_acquire);
            if (job != nullptr) {
                work(*job, worker);
            }
        }
    }

    int workers_{1};
    std::vector<std::thread::id> ids_;
    std::mutex region_mutex_;
    std::mutex m_;
    std::condition_variable_any cv_;
    std::condition_variable cv_done_;
    std::atomic<std::uint64_t> gen_{0};
    std::atomic<Job*> job_{nullptr};
    std::atomic<std::thread::id> coordinator_{};
    // LAST member on purpose: reverse destruction order joins the jthreads (their
    // stop_callback notifies cv_) while the mutexes/cvs above are still alive.
    // Declared earlier, ~cv_ would run with workers parked -> UB that HANGS at exit.
    std::vector<std::jthread> threads_;
};

// NON-inline on purpose: an `inline` function in a module interface is instantiated
// per importing TU, and MSVC can DUPLICATE its function-local static -> two Pools,
// one unconstructed, worker derefs a null mutex. Module-linkage = compiled once = one pool.
Pool& pool() {
    static Pool p;
    return p;
}

// <= 64 chunks; boundaries depend only on n, never on worker count (determinism).
// int64 intermediate: (n + 63) must not wrap for n near INT_MAX.
inline int chunk_size(int n) noexcept {
    return static_cast<int>((static_cast<std::int64_t>(n) + 63) / 64);
}
inline int chunk_count(int n) noexcept {
    const int cs = chunk_size(n);
    return cs == 0 ? 0 : (n + cs - 1) / cs;
}

}  // namespace ses::par_detail

export namespace ses {

// Pool width (>= 1); parallel_ranges' worker index stays below this.
inline int parallel_workers() { return par_detail::pool().workers(); }

// body(i) once per i in [0, n); iterations must write disjoint memory.
template <class Body>
void parallel_for(int n, Body&& body) {
    const int cs = par_detail::chunk_size(n);
    par_detail::pool().run(par_detail::chunk_count(n), [&](int c, int) {
        const int end = (c + 1) * cs < n ? (c + 1) * cs : n;
        for (int i = c * cs; i < end; ++i) {
            body(i);
        }
    });
}

// init + sum of body(i) over [0, n); chunk partials combined in chunk order
// -> bitwise-deterministic reduction.
template <class T, class Body>
T parallel_sum(int n, T init, Body&& body) {
    const int cs = par_detail::chunk_size(n);
    const int chunks = par_detail::chunk_count(n);
    std::vector<T> partials(static_cast<std::size_t>(chunks), T{});
    par_detail::pool().run(chunks, [&](int c, int) {
        const int end = (c + 1) * cs < n ? (c + 1) * cs : n;
        T acc{};
        for (int i = c * cs; i < end; ++i) {
            acc += body(i);
        }
        partials[static_cast<std::size_t>(c)] = acc;
    });
    T result = init;
    for (int c = 0; c < chunks; ++c) {
        result += partials[static_cast<std::size_t>(c)];
    }
    return result;
}

// body(worker, begin, end) over disjoint chunks covering [0, n). worker keys
// per-worker scratch (thread_local made explicit); outputs depend on [begin, end)
// only, never on worker.
template <class Body>
void parallel_ranges(int n, Body&& body) {
    const int cs = par_detail::chunk_size(n);
    par_detail::pool().run(par_detail::chunk_count(n), [&](int c, int worker) {
        const int end = (c + 1) * cs < n ? (c + 1) * cs : n;
        body(worker, c * cs, end);
    });
}

}  // namespace ses
