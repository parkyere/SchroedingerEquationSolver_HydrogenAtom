module;
#include <algorithm>
#include <cstddef>
#include <string>
export module ses.scenario.tunneling_director;
export import ses.scenario.base_director;

export namespace ses_shell {

constexpr double kTunnelBox = 80.0;      // Bohr; h = 0.625 (256^3)
constexpr double kTunnelV0 = 0.25;       // Ha
// kappa = sqrt(2(V0-E)) = 0.5; analytic T = [1 + sinh^2(kappa w)]^-1 ~ 2.7%
constexpr double kTunnelXLo = 0.0;
constexpr double kTunnelXHi = 5.0;
constexpr double kTunnelK0 = 0.5;        // mean E = k^2/2 = 0.125 < V0
constexpr double kTunnelLaunchX = -30.0;
constexpr double kTunnelSigma = 5.0;

class TunnelingDirector final : public BaseDirector, public TunnelApi {
public:
    TunnelingDirector() : BaseDirector(make()) {}

    TunnelApi* tunnel() override { return this; }

    // Selftest hook.
    double transmitted_max() const override { return t_max_; }

protected:
    ses::WavepacketSimulation remake_simulation() const override { return make(); }
    const char* scene_name() const override { return "Quantum tunneling"; }
    double absorber_width() const override { return 10.0; }
    bool relax_allowed() const override { return false; }  // no bound target

    // No marker: an origin ball would misread as a nucleus; renderer draws the slab.
    int marker_count() const override { return 0; }
    bool barrier_slab(double& lo, double& hi) const override {
        lo = kTunnelXLo;
        hi = kTunnelXHi;
        return true;
    }
    // Slight angle keeps the 3D depth cue on the wall.
    double default_camera_azimuth() const override { return 0.18; }
    double default_camera_elevation() const override { return 0.22; }

    std::string title_suffix() override {
        return strf("  V0 = %.2f Ha, E = %.3f Ha (forbidden)  P(x<%.0f) %.3f | "
                    "P(x>%.0f) %.3f (max T %.3f)",
                    kTunnelV0, 0.5 * kTunnelK0 * kTunnelK0, kTunnelXLo, p_left_,
                    kTunnelXHi, p_right_, t_max_);
    }

    // Full-field readback ~10 ms: probe every 3rd title tick (~0.5 s).
    void after_step_batch() override {
        if (!gpu_title_due_ || ++probe_phase_ % 3 != 0) {
            return;
        }
        // Host-wait: readback consumes POST-step psi (same-queue order carries no memory dependency).
        engine_.wait_async();
        if (!engine_.readback(readback_buf_)) {
            return;  // GPU readback failed: skip this probe (no stale/OOB read)
        }
        const ses::Grid3D& g = sim_.grid();
        const double dv = g.cell_volume();
        const int nx = g.x.n;
        double left = 0.0;
        double right = 0.0;
        const std::size_t cells = readback_buf_.size() / 2;
        for (std::size_t idx = 0; idx < cells; ++idx) {
            const double re = readback_buf_[2 * idx];
            const double im = readback_buf_[2 * idx + 1];
            const double d = (re * re + im * im) * dv;
            // x-fastest layout: the x index is idx % nx.
            const double x = g.x.coord(static_cast<int>(idx % nx));
            if (x < kTunnelXLo) {
                left += d;
            } else if (x >= kTunnelXHi) {
                right += d;
            }
        }
        p_left_ = left;
        p_right_ = right;
        t_max_ = std::max(t_max_, p_right_);
        title_dirty_ = true;
    }

    void after_reset() override {
        p_left_ = 0.0;
        p_right_ = 0.0;
        t_max_ = 0.0;
        probe_phase_ = 0;
    }

private:
    static ses::WavepacketSimulation make() {
        const ses::Grid1D axis{-kTunnelBox, kTunnelBox, 256};
        const ses::Grid3D grid{axis, axis, axis};
        return ses::WavepacketSimulation{ses::WavepacketSimulation::Config{
            grid,
            ses::barrier_potential(grid, kTunnelV0, kTunnelXLo, kTunnelXHi),
            ses::Vec3d{kTunnelLaunchX, 0.0, 0.0},
            ses::Vec3d{kTunnelSigma, kTunnelSigma, kTunnelSigma},
            ses::Vec3d{kTunnelK0, 0.0, 0.0},
            0.04,  // dt
        }};
    }

    double p_left_ = 0.0;
    double p_right_ = 0.0;
    double t_max_ = 0.0;
    int probe_phase_ = 0;
};

}  // namespace ses_shell
