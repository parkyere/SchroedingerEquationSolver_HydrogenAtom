module;
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <vector>
export module ses.scenario.molecule_director;
export import ses.scenario.base_director;
import ses.wavepacket;


// One-electron molecules with FIXED nuclei (Born-Oppenheimer): the engine
// propagates the single electron in a multi-center Coulomb landscape; the
// nuclei are geometry, not dynamics. Two scenes share the machinery:
//
//  - H2+ (two bare protons, regularized cells): sigma_g piles bond charge
//    between the nuclei, the deflated sigma_u carries the nodal plane; the
//    R knob scans E_total(R) = E_elec + 1/R -- the chemical bond is its
//    minimum. (H2+ separates in prolate spheroidal coordinates: it is the
//    two-center member of the "solvable" family.)
//  - Benzene toy (six equal SOFT cores on a ring): the uniform hexagon's
//    ground state delocalizes evenly and its first excited pair is
//    degenerate; the Kekule 1-2-1-2 alternation keeps the pair degenerate
//    (D3h still has the 2-dim irrep!) but piles the bond charge onto the
//    short bonds. One-electron toy: symmetry and delocalization, NOT the
//    full six-pi-electron aromatic stabilization.
//
// State preparation is a CHAIN: ITP ground, then deflated excited states
// against the captured lower ones (fp32 state buffers, engine-resident),
// mirroring the hydrogen director's deflation flow. prepare(k) is async;
// completion is the ITP energy plateau. License physics (32^3 CPU):
// tests/molecule_test.cpp.


export namespace ses_shell {

class MoleculeDirectorBase : public BaseDirector, public MoleculeApi {
public:
    explicit MoleculeDirectorBase(ses::WavepacketSimulation sim)
        : BaseDirector(std::move(sim)) {}

    MoleculeApi* molecule() override { return this; }

    // ---- MoleculeApi ----
    bool prepared(int k) const override {
        return k >= 0 && k < kStates && prepared_[k];
    }
    double energy(int k) const override {
        return (k >= 0 && k < kStates) ? e_[k] : 0.0;
    }
    void prepare(int k) override {
        if (!gpu_ok_ || k < 0 || k >= kStates) {
            return;  // the deflation chain runs on the GPU path only
        }
        want_ = k;
        advance_chain();
    }
    double nuclear_repulsion() const override {
        const std::vector<ses::Vec3d> c = centers();
        double acc = 0.0;
        for (std::size_t i = 0; i < c.size(); ++i) {
            for (std::size_t j = i + 1; j < c.size(); ++j) {
                const double dx = c[i].x - c[j].x;
                const double dy = c[i].y - c[j].y;
                const double dz = c[i].z - c[j].z;
                acc += 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz);
            }
        }
        return acc;
    }
    int geometry() const override { return geom_; }
    double parameter() const override { return param_; }
    void set_geometry(int variant) override {
        geom_ = variant;
        param_ = geometry_parameter(variant);
        apply_geometry();
    }
    void set_parameter(double p) override {
        geom_ = -1;  // custom knob value
        param_ = clamp_parameter(p);
        apply_geometry();
    }

    bool handle_key(char key) override {
        if (key == '2') {
            prepare(0);
            return true;
        }
        if (key == '3') {
            prepare(1);
            return true;
        }
        if (key == '4' && kStates > 2) {
            prepare(2);
            return true;
        }
        return false;
    }

    bool center_marker() const override { return false; }

protected:
    static constexpr int kStates = 3;

    // ---- scene hooks ----
    virtual std::vector<ses::Vec3d> centers() const = 0;
    virtual ses::Field3D excited_seed(int k) const = 0;
    virtual double geometry_parameter(int variant) const = 0;
    virtual double clamp_parameter(double p) const = 0;
    virtual void geometry_changed() {}  // rebuild markers etc.

    // Swap the nuclear geometry under a fresh electron: new potential on
    // both the CPU truth and the engine, stale state caches dropped, and
    // the ground relax restarted automatically (an E(R)-scan step).
    void apply_geometry() {
        if (gpu_ok_) {
            engine_.wait_async();
        }
        sim_ = remake_simulation();
        free_state_buffers();
        cpu_is_truth_ = true;
        gpu_time_ = 0.0;
        pending_gpu_steps_ = 0;
        stepping_ = BaseStepping::RealTime;
        if (gpu_ok_) {
            engine_.release_relax_tables();  // tables bake the OLD potential
            engine_.set_potential(sim_.potential());
            engine_.set_potential_gradient(sim_.potential());
        }
        geometry_changed();
        title_dirty_ = true;
        if (gpu_ok_) {
            prepare(0);
        } else {
            stage_active_view();
        }
    }

    void advance_chain() {
        int next = -1;
        for (int k = 0; k <= want_; ++k) {
            if (!prepared_[k]) {
                next = k;
                break;
            }
        }
        if (next < 0) {
            return;  // everything up to want_ is already cached
        }
        start_relax_for(next);
    }

    void start_relax_for(int k) {
        if (!ensure_relax_tables()) {
            return;
        }
        if (mode_ != BaseViewMode::Cloud) {
            mode_ = BaseViewMode::Cloud;
        }
        deflate_.clear();
        for (int i = 0; i < k; ++i) {
            deflate_.push_back(buf_[i]);
        }
        sim_.set_psi(k == 0 ? ground_seed() : excited_seed(k));
        cpu_is_truth_ = true;  // run_frame uploads the seed to the engine
        target_ = k;
        relax_plateau_ = 0;
        relax_prev_energy_ = 0.0;
        stepping_ = BaseStepping::Relaxing;
    }

    // Deflated ITP batch + plateau auto-complete + state capture.
    void run_relax_batch() override {
        const ses_vk::Engine::RelaxStats stats =
            deflate_.empty()
                ? engine_.relax_step(pending_gpu_steps_)
                : engine_.relax_deflated_step(deflate_, pending_gpu_steps_);
        relax_energy_display_ = stats.energy;
        if (stats.peak > 0.0) {
            peak_ = stats.peak;
        }
        norm_display_ = 1.0;
        if (gpu_title_due_) {
            if (std::abs(stats.energy - relax_prev_energy_) < 5e-5) {
                ++relax_plateau_;
            } else {
                relax_plateau_ = 0;
            }
            relax_prev_energy_ = stats.energy;
            if (relax_plateau_ >= 12) {
                relax_plateau_ = 0;
                complete_relax();
            }
        }
    }

    void complete_relax() {
        stepping_ = BaseStepping::RealTime;
        // Capture the converged state as an engine-resident deflation
        // buffer + record its energy for the HUD/arcs.
        if (engine_.readback(readback_buf_)) {
            ses::Field3D f{sim_.grid()};
            for (std::size_t i = 0; i < f.data().size(); ++i) {
                f.data()[i] = std::complex<double>{readback_buf_[2 * i],
                                                   readback_buf_[2 * i + 1]};
            }
            ses::normalize(f);
            if (buf_[target_] >= 0) {
                engine_.release_state(buf_[target_]);
            }
            buf_[target_] = engine_.create_state_buffer(f.data());
            if (buf_[target_] >= 0) {
                e_[target_] = relax_energy_display_;
                prepared_[target_] = true;
            }
        }
        title_dirty_ = true;
        if (target_ < want_ && prepared_[target_]) {
            advance_chain();  // keep climbing the requested chain
        } else {
            engine_.release_relax_tables();
        }
    }

    void free_state_buffers() {
        for (int k = 0; k < kStates; ++k) {
            if (buf_[k] >= 0 && gpu_ok_) {
                engine_.release_state(buf_[k]);
            }
            buf_[k] = -1;
            prepared_[k] = false;
            e_[k] = 0.0;
        }
        deflate_.clear();
        want_ = 0;
        target_ = 0;
    }

    void after_reset() override { free_state_buffers(); }

    ses::Field3D ground_seed() const {
        return ses::gaussian_wavepacket(sim_.grid(), ses::Vec3d{},
                                        ses::Vec3d{1.8, 1.8, 1.8},
                                        ses::Vec3d{});
    }

    int geom_ = 0;
    double param_ = 0.0;
    int buf_[kStates] = {-1, -1, -1};
    double e_[kStates] = {0.0, 0.0, 0.0};
    bool prepared_[kStates] = {false, false, false};
    int want_ = 0;
    int target_ = 0;
    std::vector<int> deflate_;
};

// ---- H2+ ------------------------------------------------------------------

constexpr double kH2pBox = 20.0;   // Bohr half-extent, 256^3 (h ~ 0.156)
constexpr int kH2pPoints = 256;
constexpr double kH2pDt = 0.04;
constexpr double kH2pRDefault = 2.0;  // near the true equilibrium R ~ 2.0
constexpr double kH2pRMin = 1.0;
constexpr double kH2pRMax = 8.0;

class H2PlusDirector final : public MoleculeDirectorBase {
public:
    H2PlusDirector() : MoleculeDirectorBase(make(kH2pRDefault)) {
        param_ = snap_r(kH2pRDefault);
        rebuild_markers();
    }

    double default_camera_azimuth() const override { return 0.35; }
    double default_camera_elevation() const override { return 0.28; }
    double default_camera_distance() const override { return 55.0; }

    int overlay_curve_count() const override { return 2; }
    OverlayCurve overlay_curve(int i) const override {
        // Two proton diamonds (warm orange, like the atomic marker).
        return {marker_[i == 0 ? 0 : 1].data(), 5, 1.0f, 0.45f, 0.20f, 1.0f};
    }

protected:
    const char* scene_name() const override { return "H2+ molecular ion"; }

    ses::WavepacketSimulation remake_simulation() const override {
        return make(param_ > 0.0 ? param_ : kH2pRDefault);
    }

    std::vector<ses::Vec3d> centers() const override {
        const double d = 0.5 * snap_r(param_);
        return {{-d, 0.0, 0.0}, {d, 0.0, 0.0}};
    }
    // sigma_u: the x-odd seed keeps the flow in the odd sector; deflation
    // against sigma_g is the belt to that suspender.
    ses::Field3D excited_seed(int /*k*/) const override {
        const ses::Grid3D& g = sim_.grid();
        ses::Field3D seed{g};
        for (int k = 0; k < g.z.n; ++k) {
            for (int j = 0; j < g.y.n; ++j) {
                for (int i = 0; i < g.x.n; ++i) {
                    const double x = g.x.coord(i);
                    const double y = g.y.coord(j);
                    const double z = g.z.coord(k);
                    const double env =
                        std::exp(-(x * x + y * y + z * z) / (4.0 * 4.0));
                    seed(i, j, k) = std::complex<double>{x * env, 0.0};
                }
            }
        }
        ses::normalize(seed);
        return seed;
    }
    double geometry_parameter(int variant) const override {
        return variant == 1 ? 6.0 : kH2pRDefault;  // 0 = equilibrium, 1 = stretched
    }
    double clamp_parameter(double p) const override {
        return snap_r(std::clamp(p, kH2pRMin, kH2pRMax));
    }
    void geometry_changed() override { rebuild_markers(); }

    std::string title_suffix() override {
        const double rep = nuclear_repulsion();
        std::string s = strf("  R = %.3f Bohr (fixed nuclei)", snap_r(param_));
        if (prepared_[0]) {
            s += strf("  E(sigma_g) = %.4f, E_total = %.4f Ha", e_[0],
                      e_[0] + rep);
        }
        if (prepared_[1]) {
            s += strf("  E(sigma_u) = %.4f (antibonding)", e_[1]);
        }
        s += "  keys: 2 sigma_g / 3 sigma_u, R slider scans the bond";
        return s;
    }

private:
    static ses::WavepacketSimulation make(double r) {
        const ses::Grid1D axis{-kH2pBox, kH2pBox, kH2pPoints};
        const ses::Grid3D grid{axis, axis, axis};
        const double d = 0.5 * snap_r(r);
        return ses::WavepacketSimulation{ses::WavepacketSimulation::Config{
            grid,
            ses::regularized_coulomb_potential(
                grid, 1.0, {{-d, 0.0, 0.0}, {d, 0.0, 0.0}}),
            ses::Vec3d{},
            ses::Vec3d{1.8, 1.8, 1.8},
            ses::Vec3d{},
            kH2pDt,
        }};
    }

    // Snap the HALF-distance to a grid point so both regularized nucleus
    // cells stay honest (R in multiples of 2h).
    static double snap_r(double r) {
        const double h = 2.0 * kH2pBox / kH2pPoints;
        const double m =
            std::max(1.0, std::round(std::clamp(r, kH2pRMin, kH2pRMax) /
                                     (2.0 * h)));
        return 2.0 * h * m;
    }

    void rebuild_markers() {
        const std::vector<ses::Vec3d> c = centers();
        for (int n = 0; n < 2; ++n) {
            const float x = static_cast<float>(c[static_cast<std::size_t>(n)].x);
            const float s = 0.5f;
            marker_[n] = {x - s, 0.0f, 0.0f, x, s, 0.0f, x + s, 0.0f, 0.0f,
                          x, -s, 0.0f, x - s, 0.0f, 0.0f};
        }
    }

    std::vector<float> marker_[2];
};

// ---- Benzene one-electron toy --------------------------------------------

constexpr double kBzBox = 12.0;   // Bohr half-extent, 256^3 (h ~ 0.094)
constexpr int kBzPoints = 256;
constexpr double kBzDt = 0.04;
constexpr double kBzRingR = 2.63;   // C-C 1.39 A in bohr
constexpr double kBzSoftA = 0.8;    // soft-core pseudopotential width
constexpr double kBzKekDeg = 5.0;   // Kekule angle alternation (deg)
constexpr double kBzKekMax = 10.0;

class BenzeneDirector final : public MoleculeDirectorBase {
public:
    BenzeneDirector() : MoleculeDirectorBase(make(0.0)) {
        param_ = 0.0;  // uniform ring at boot
        rebuild_markers();
    }

    double default_camera_azimuth() const override { return 0.3; }
    double default_camera_elevation() const override { return 0.7; }
    double default_camera_distance() const override { return 40.0; }

    int overlay_curve_count() const override { return 1; }
    OverlayCurve overlay_curve(int /*i*/) const override {
        // The closed ring skeleton: the bond alternation IS visible in it.
        return {ring_marker_.data(), 7, 1.0f, 0.45f, 0.20f, 1.0f};
    }

protected:
    const char* scene_name() const override {
        return "Benzene ring (one-electron toy)";
    }

    ses::WavepacketSimulation remake_simulation() const override {
        return make(param_);
    }

    std::vector<ses::Vec3d> centers() const override {
        return ring(param_);
    }
    // In-plane displaced seeds pick up the ring-momentum pair.
    ses::Field3D excited_seed(int k) const override {
        const ses::Vec3d at = k == 1
                                  ? ses::Vec3d{kBzRingR, 0.4, 0.0}
                                  : ses::Vec3d{-0.5, kBzRingR, 0.0};
        return ses::gaussian_wavepacket(sim_.grid(), at,
                                        ses::Vec3d{1.5, 1.5, 1.2},
                                        ses::Vec3d{});
    }
    double geometry_parameter(int variant) const override {
        return variant == 1 ? kBzKekDeg : 0.0;  // 0 = uniform, 1 = Kekule
    }
    double clamp_parameter(double p) const override {
        return std::clamp(p, 0.0, kBzKekMax);
    }
    void geometry_changed() override { rebuild_markers(); }

    std::string title_suffix() override {
        std::string s =
            param_ == 0.0
                ? std::string("  UNIFORM ring (X-ray's verdict)")
                : strf("  KEKULE 1-2-1-2 (delta = %.1f deg)", param_);
        if (prepared_[0]) {
            s += strf("  E0 = %.4f", e_[0]);
        }
        if (prepared_[1]) {
            s += strf("  E1 = %.4f", e_[1]);
        }
        if (prepared_[2]) {
            s += strf("  E2 = %.4f (pair split %.1e: D3h keeps the "
                      "degeneracy)",
                      e_[2], std::abs(e_[2] - e_[1]));
        }
        s += "  keys: 2/3/4 states; one-electron toy, not 6-pi aromaticity";
        return s;
    }

private:
    static std::vector<ses::Vec3d> ring(double delta_deg) {
        const double kPi = 3.14159265358979323846;
        std::vector<ses::Vec3d> c;
        for (int i = 0; i < 6; ++i) {
            const double th =
                kPi / 3.0 * i +
                (i % 2 == 0 ? 1.0 : -1.0) * delta_deg * kPi / 180.0;
            c.push_back({kBzRingR * std::cos(th), kBzRingR * std::sin(th), 0.0});
        }
        return c;
    }

    static ses::WavepacketSimulation make(double delta_deg) {
        const ses::Grid1D axis{-kBzBox, kBzBox, kBzPoints};
        const ses::Grid3D grid{axis, axis, axis};
        return ses::WavepacketSimulation{ses::WavepacketSimulation::Config{
            grid,
            ses::soft_coulomb_potential(grid, 1.0, kBzSoftA, ring(delta_deg)),
            ses::Vec3d{},
            ses::Vec3d{1.8, 1.8, 1.2},
            ses::Vec3d{},
            kBzDt,
        }};
    }

    void rebuild_markers() {
        const std::vector<ses::Vec3d> c = centers();
        ring_marker_.clear();
        for (int i = 0; i <= 6; ++i) {
            const ses::Vec3d& p = c[static_cast<std::size_t>(i % 6)];
            ring_marker_.push_back(static_cast<float>(p.x));
            ring_marker_.push_back(static_cast<float>(p.y));
            ring_marker_.push_back(static_cast<float>(p.z));
        }
    }

    std::vector<float> ring_marker_;
};

}  // namespace ses_shell
