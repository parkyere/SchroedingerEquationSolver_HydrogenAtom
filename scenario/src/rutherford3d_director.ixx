module;
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
export module ses.scenario.rutherford3d_director;
export import ses.scenario.base_director;
import ses.potential;


// Rutherford scattering: Gaussian packet fired head-on at a repulsive Coulomb
// center.
//
// Unit-mass analog: a real alpha (m~7300, ~5 MeV) is classical and unresolvable
// on a grid. Scattering depends on eta = Z_eff/v (Z_eff = 2Z), not on m and E
// separately, so keep Z_eff physical and scale ENERGY to the alpha velocity:
// E=25 Ha at m=1 -> v=7.07 a.u. -> eta=22.3, matching the gold-alpha experiment.
// Scaling Z instead breaks eta. Packet turns at r_min >> a cell, so it never
// samples the regularized core spike.


export namespace ses_shell {

constexpr double kRu3dBox = 40.0;
constexpr int kRu3dPoints = 256;
constexpr double kRu3dZDefault = 79.0;  // gold
constexpr double kRu3dZProj = 2.0;      // alpha (helium)
constexpr double kRu3dEDefault = 25.0;  // incident KE (Ha)
constexpr double kRu3dEMin = 5.0;
constexpr double kRu3dEMax = 80.0;
constexpr double kRu3dZMin = 5.0;
constexpr double kRu3dZMax = 100.0;
constexpr double kRu3dLaunchX = -30.0;
constexpr double kRu3dSigma = 4.0;
constexpr double kRu3dDt = 0.01;        // small: V*dt/2 moderate at the turn

class Rutherford3DDirector final : public BaseDirector, public RutherfordApi {
public:
    Rutherford3DDirector() : BaseDirector(make(kRu3dEDefault, kRu3dZDefault)) {}

    RutherfordApi* rutherford() override { return this; }

    // ---- RutherfordApi ----
    void set_energy(double e) override {
        const double v = std::clamp(e, kRu3dEMin, kRu3dEMax);
        if (v != energy_) {
            energy_ = v;
            reset_simulation();
        }
    }
    double energy() const override { return energy_; }
    void set_z(double z) override {
        const double v = std::clamp(z, kRu3dZMin, kRu3dZMax);
        if (v != z_) {
            z_ = v;
            reset_simulation();
        }
    }
    double z() const override { return z_; }
    void refire() override { reset_simulation(); }
    double turning_point() const override {
        return kRu3dZProj * z_ / energy_;
    }
    double closest_approach() const override { return r_min_seen_; }
    double backscattered_fraction() const override { return back_; }

    // Base reset re-uploads psi only; also push the new potential to the engine.
    void reset_simulation() override {
        BaseDirector::reset_simulation();
        if (gpu_ok_) {
            engine_.wait_async();
            engine_.set_potential(sim_.potential());
            engine_.set_potential_gradient(sim_.potential());
        }
    }

protected:
    ses::WavepacketSimulation remake_simulation() const override {
        return make(energy_, z_);
    }
    const char* scene_name() const override {
        return "Rutherford scattering (repulsive Coulomb)";
    }
    double absorber_width() const override { return 10.0; }
    bool relax_allowed() const override { return false; }  // no bound target

    // gold nucleus (CPK color)
    int marker_count() const override { return 1; }
    SceneMarker marker(int /*i*/) const override {
        return {0.0f, 0.0f, 0.0f, 0.6f, 0.95f, 0.78f, 0.20f};
    }

    double default_camera_azimuth() const override { return 0.22; }
    double default_camera_elevation() const override { return 0.22; }
    double default_camera_distance() const override { return 110.0; }

    std::string title_suffix() override {
        const double v = std::sqrt(2.0 * energy_);
        const double eta = kRu3dZProj * z_ / v;
        return strf("  Au(Z=%.0f) <- He++ (alpha; v=%.2f a.u., eta=%.1f ~ real "
                    "5 MeV)  E = %.1f Ha  r_min = %.1f bohr  closest <r> = %.1f "
                    " backscatter %.2f",
                    z_, v, eta, energy_, turning_point(),
                    r_min_seen_ < 1e8 ? r_min_seen_ : 0.0, back_);
    }

    // Readback ~10 ms; probe every 3rd title tick.
    void after_step_batch() override {
        if (!gpu_title_due_ || ++probe_phase_ % 3 != 0) {
            return;
        }
        engine_.wait_async();
        if (!engine_.readback(readback_buf_)) {
            return;
        }
        const ses::Grid3D& g = sim_.grid();
        const int nx = g.x.n;
        const int ny = g.y.n;
        const std::size_t cells = readback_buf_.size() / 2;
        double mr = 0.0;
        double den = 0.0;
        double back = 0.0;
        for (std::size_t idx = 0; idx < cells; ++idx) {
            const double re = readback_buf_[2 * idx];
            const double im = readback_buf_[2 * idx + 1];
            const double d = re * re + im * im;
            if (d <= 0.0) {
                continue;
            }
            const int i = static_cast<int>(idx % nx);
            const int j = static_cast<int>((idx / nx) % ny);
            const int k = static_cast<int>(idx / (nx * ny));
            const double x = g.x.coord(i);
            const double y = g.y.coord(j);
            const double z = g.z.coord(k);
            mr += std::sqrt(x * x + y * y + z * z) * d;
            den += d;
            if (x < kRu3dLaunchX) {
                back += d;
            }
        }
        if (den > 0.0) {
            const double mean_r = mr / den;
            r_min_seen_ = std::min(r_min_seen_, mean_r);
            back_ = std::max(back_, back / den);
            title_dirty_ = true;
        }
    }

    void after_reset() override {
        r_min_seen_ = 1e9;
        back_ = 0.0;
        probe_phase_ = 0;
    }

private:
    static ses::WavepacketSimulation make(double e, double z) {
        const ses::Grid1D axis{-kRu3dBox, kRu3dBox, kRu3dPoints};
        const ses::Grid3D grid{axis, axis, axis};
        // Repulsive: attractive generator + negative charge -2Z -> V = +2Z/r.
        const double p0 = std::sqrt(2.0 * e);
        return ses::WavepacketSimulation{ses::WavepacketSimulation::Config{
            grid,
            ses::regularized_coulomb_potential(grid, -kRu3dZProj * z,
                                               ses::Vec3d{}),
            ses::Vec3d{kRu3dLaunchX, 0.0, 0.0},
            ses::Vec3d{kRu3dSigma, kRu3dSigma, kRu3dSigma},
            ses::Vec3d{p0, 0.0, 0.0},
            kRu3dDt,
        }};
    }

    double energy_ = kRu3dEDefault;
    double z_ = kRu3dZDefault;
    double r_min_seen_ = 1e9;
    double back_ = 0.0;
    int probe_phase_ = 0;
};

}  // namespace ses_shell
