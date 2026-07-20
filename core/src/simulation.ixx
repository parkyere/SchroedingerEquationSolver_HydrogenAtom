module;
#include <optional>
#include <utility>
#include <vector>
#include <cstdint>
#include <cassert>
export module ses.simulation;
export import ses.grid;
export import ses.vec;
export import ses.imaginary_time;
export import ses.propagator;
export import ses.measurement;
export import ses.field;
export import ses.wavepacket;


// Contract (test-pinned): only gaussian_wavepacket + SplitOperator3D, no hidden physics.


export namespace ses {

class WavepacketSimulation {
public:
    struct Config {
        Grid3D grid;
        std::vector<double> potential;
        Vec3d r0;
        Vec3d sigma;
        Vec3d k0;
        double dt;
    };

    explicit WavepacketSimulation(Config cfg)
        : grid_(cfg.grid),
          dt_(cfg.dt),
          potential_(std::move(cfg.potential)),
          propagator_(cfg.grid, potential_, cfg.dt),
          psi_(gaussian_wavepacket(cfg.grid, cfg.r0, cfg.sigma, cfg.k0)) {}

    void advance(int nsteps) {
        propagator_.step(psi_, nsteps);
        steps_ += nsteps;
    }

    // Relaxer keyed on dtau (decay tables); tau != t, no real-time advance.
    void relax(int nsteps, double dtau) {
        if (!relaxer_ || relaxer_dtau_ != dtau) {
            relaxer_.emplace(grid_, potential_, dtau);
            relaxer_dtau_ = dtau;
        }
        relaxer_->relax(psi_, nsteps);
    }

    // u = injected uniform[0,1) draw selecting the sigma_m-blurred POVM outcome cell.
    Vec3d measure(double u, double sigma_m) {
        const int idx = sample_povm_index(psi_, sigma_m, u);
        const int i = idx % grid_.x.n;
        const int j = (idx / grid_.x.n) % grid_.y.n;
        const int k = idx / (grid_.x.n * grid_.y.n);
        const Vec3d center{grid_.x.coord(i), grid_.y.coord(j), grid_.z.coord(k)};
        collapse_wavepacket(psi_, center, sigma_m);
        return center;
    }

    constexpr double time() const noexcept { return steps_ * dt_; }
    constexpr double dt() const noexcept { return dt_; }
    const Grid3D& grid() const noexcept { return grid_; }
    const Field3D& psi() const noexcept { return psi_; }
    const std::vector<double>& potential() const noexcept { return potential_; }
    const SplitOperator3D& propagator() const noexcept { return propagator_; }

    // Inject a GPU-evolved field back into this CPU session.
    void set_psi(const Field3D& psi) {
        assert(psi.data().size() == psi_.data().size());
        psi_ = psi;
    }
    std::vector<double> density() const { return probability_density(psi_); }

private:
    Grid3D grid_;
    double dt_;
    std::vector<double> potential_;
    SplitOperator3D propagator_;
    Field3D psi_;
    long long steps_ = 0;
    std::optional<ImaginaryTimePropagator3D> relaxer_;
    double relaxer_dtau_ = 0.0;
};

}  // namespace ses
