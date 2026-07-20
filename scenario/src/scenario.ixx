module;
#include <volk.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
export module ses.scenario;
export import ses.vk.device;
export import ses.grid;
export import ses.marching_cubes;
export import ses.colormap;


// One framework-neutral interface per demo; the shell owns one ScenarioDirector
// (--scene=) and drives it through this contract.
// volk.h: VK_* macros (macros never cross module boundaries) + inoculates the TU
// against ses.vk.device's GMF/textual std redefinitions.


export namespace ses_shell {

// Capability seams: the shell never down-casts; a scene returning non-null from
// the matching ScenarioDirector accessor exposes that control surface.
struct HydrogenApi {
    virtual ~HydrogenApi() = default;
    virtual void set_relaxing() = 0;
    virtual void relax_to_excited() = 0;
    virtual void relax_to_2s() = 0;
    virtual void excite_n3() = 0;
    virtual void toggle_decay() = 0;
    virtual void toggle_laser() = 0;
    virtual void measure_energy_now() = 0;
    virtual void measure_n_shell_now() = 0;
    virtual void measure_l_now() = 0;
    virtual void measure_m_now() = 0;
    virtual int last_partial_outcome() const = 0;
    virtual void set_efield_e0(double e0) = 0;
    virtual void set_bfield_b(double b) = 0;
    virtual double bfield_b() const = 0;  // director truth (panel syncs back)
    virtual void toggle_bfield_axis() = 0;
    virtual int bfield_axis() const = 0;
    virtual double ionized_fraction() const = 0;
    virtual void set_mcwf_damping(bool on) = 0;
    virtual bool mcwf_damping() const = 0;
    virtual double channel_a(int from, int to) const = 0;
    virtual double state_energy(int idx) const = 0;
    virtual long long photon_count() const = 0;
    // Jump-photon energies (eV) since reset, emission order. spectro_max_ev =
    // strip full scale (ionization limit); 0 hides the strip.
    virtual int spectro_count() const { return 0; }
    virtual double spectro_ev(int /*i*/) const { return 0.0; }
    virtual double spectro_max_ev() const { return 0.0; }
    virtual int last_measured_index() const = 0;
    virtual void seed_kepler() = 0;  // circular-state Rydberg packet (K)
    virtual double mean_z() = 0;
    virtual double mean_x() = 0;
    virtual double mean_y() = 0;
    virtual double peak_excited_population() const = 0;
    virtual void debug_prepare_state(int idx) = 0;
    virtual double probe_population(int idx) = 0;
    virtual void debug_prepare_superposition(int a, int b) = 0;
};

struct TunnelApi {
    virtual ~TunnelApi() = default;
    virtual double transmitted_max() const = 0;
};

// 1D harmonic-oscillator ladder (the 1D trap scene).
struct Ladder1dApi {
    virtual ~Ladder1dApi() = default;
    // Fock level n; -1 = superposition (classified by Var(H)).
    virtual int level() const = 0;
    virtual double level_energy() const = 0;  // live <H> (Ha)
    // a-dag (up) / a (down); false = refused (a|0> = 0, or spectral-band cap).
    virtual bool ladder(bool up) = 0;
    // Well stiffness omega (width ~ 1/sqrt(w)); setting it is a sudden QUENCH
    // (psi kept, breathes in the new well; reset target becomes the new ground).
    virtual void set_omega(double w) = 0;
    virtual double omega() const = 0;
    // Largest level reachable from the CURRENT state: eigenstate via the
    // oracle-rebuilt path (ses.ladder ho_level_cap); superposition in the
    // truncated Fock basis (ses.ladder ladder_fock band).
    virtual int max_level() const = 0;
    // Random coherent superposition over low Fock levels (PURE; a wavefunction
    // solver cannot hold a density-matrix mixture).
    virtual void random_superposition() = 0;
    // Schrodinger-cat lens: two-lobe cat |a> + |-a>, cavity photon-loss MCWF
    // (each lost photon flips the parity).
    virtual void cat() = 0;
    virtual void toggle_loss() = 0;
    virtual bool loss_on() const = 0;
    virtual long long jump_count() const = 0;
    // Linear-combination spectrum strip (0..100 eV). Weights change only on
    // mutations (unitary evolution preserves |c_n|^2), so recompute is lazy.
    virtual int spectrum_count() = 0;
    virtual double spectrum_ev(int i) = 0;
    virtual double spectrum_weight(int i) = 0;
};

struct DoubleWellApi {
    virtual ~DoubleWellApi() = default;
    virtual double splitting() const = 0;  // dE = E1 - E0 (Ha)
    virtual double p_left() const = 0;
    virtual double p_right() const = 0;
    // Re-prepare psi_L in a well with the new barrier.
    virtual void set_barrier(double vb) = 0;
    virtual double barrier() const = 0;
};

// Reflectionless Poschl-Teller scattering scene.
struct ReflectApi {
    virtual ~ReflectApi() = default;
    // Largest negative-momentum fraction while most norm still in box (honest
    // R; absorbed flux must not inflate it).
    virtual double reflected_max() const = 0;
    virtual bool square_well() const = 0;
    virtual void toggle_well() = 0;  // sech^2 <-> equal square, relaunch
};

// One-electron fixed-nuclei molecules (Born-Oppenheimer): ITP ground then the
// deflated excited chain. prepare(k) is ASYNC (relax over frames); poll prepared(k).
struct MoleculeApi {
    virtual ~MoleculeApi() = default;
    virtual bool prepared(int k) const = 0;  // state k solved and cached
    virtual double energy(int k) const = 0;  // captured E_k (Ha); 0 = none
    virtual void prepare(int k) = 0;         // relax the chain up to k
    virtual double nuclear_repulsion() const = 0;  // sum_{i<j} Z^2 / r_ij
    virtual void set_geometry(int variant) = 0;    // scene-defined presets
    virtual int geometry() const = 0;
    virtual void set_parameter(double p) = 0;  // scene knob (R / delta)
    virtual double parameter() const = 0;
    // P(|r| < radius): containment probe (arc contract -- a bound state must
    // STAY on the nuclei).
    virtual double containment(double radius) = 0;
    // Exposed orbitals: prepare(k) valid for k in [0, state_count); label is
    // the term symbol.
    virtual int state_count() const = 0;
    virtual const char* orbital_label(int k) const = 0;
    // Drop a random normalized wavefunction (a superposition, not an eigenstate).
    virtual void seed_random() = 0;
};

// Morse anharmonic ladder (shrinking gaps).
struct MorseApi {
    virtual ~MorseApi() = default;
    virtual int level() const = 0;           // -1 = pair superposition
    virtual double level_energy() const = 0;
    virtual bool jump(bool up) = 0;
    virtual int bound_count() const = 0;
};

// 2D double-slit + Aharonov-Bohm: sliders re-fire a fresh electron; flux is the
// solenoid buried mid-wall (exact Peierls link phases, B = 0 on every path).
struct SlitApi {
    virtual ~SlitApi() = default;
    virtual void set_separation(double d) = 0;
    virtual double separation() const = 0;
    virtual void set_width(double w) = 0;
    virtual double width() const = 0;
    virtual void set_flux(double phi) = 0;
    virtual double flux() const = 0;
    virtual void refire() = 0;
    virtual double transmitted_fraction() const = 0;
    // Accumulated screen density at the row nearest y (arcs probe fringes).
    virtual double screen_at(double y) const = 0;
};

// Landau / cyclotron scene: uniform B along z on the 2D lattice.
struct LandauApi {
    virtual ~LandauApi() = default;
    virtual void set_field(double b) = 0;
    virtual double field() const = 0;
    virtual void set_k0(double k0) = 0;
    virtual double k0() const = 0;
    virtual void refire() = 0;
    virtual double omega_c() const = 0;       // = B
    virtual double radius_pred() const = 0;   // k0 / B
    virtual double orbit_x() const = 0;       // live <x>
    virtual double orbit_y() const = 0;       // live <y>
    virtual double mean_n() const = 0;        // <E>/B - 1/2 (Landau index)
    // Recorded AT the crossings (arcs poll too coarsely): distance from the T/2
    // antipode and from the T = 2 pi / B start. -1 until reached.
    virtual double antipode_dist() const = 0;
    virtual double closure_dist() const = 0;
    // Landau-level ladder (ses.lattice2d landau_ladder): a-dag / a jump one
    // cyclotron quantum; false = refused (lowest level).
    virtual bool ladder(bool up) = 0;
};

// 1D periodic-lattice (Bloch) scene: V0 sin^2(kL x) + tilt force F.
struct BlochApi {
    virtual ~BlochApi() = default;
    virtual void set_depth(double v0) = 0;
    virtual double depth() const = 0;
    virtual void set_force(double f) = 0;
    virtual double force() const = 0;
    virtual void refire() = 0;
    virtual double bloch_period() const = 0;  // G/F (0 = no tilt)
    virtual double quasimomentum() const = 0; // q(t) folded to the BZ
    virtual double mean_x() const = 0;        // live <x>
    virtual double excursion() const = 0;     // max |<x> - x0| since fire
};

// 1993 IBM quantum corral: adatoms on a ring, standing waves inside the fence.
// States capture ASYNC over frames (poll relaxing()); energies in Ha.
struct CorralApi {
    virtual ~CorralApi() = default;
    virtual void set_radius(double r) = 0;
    virtual double radius() const = 0;
    virtual double mass() const = 0;  // Cu(111) surface-state m* (a.u.)
    virtual void relax_next() = 0;  // capture the next (deflated) state
    virtual int captured() const = 0;
    virtual double energy(int k) const = 0;
    virtual double confinement() const = 0;  // probability inside the ring
    virtual bool relaxing() const = 0;
    virtual void fire_packet() = 0;  // scatter a packet off the fence
    // Standing wave AT the Fermi energy (k_F R ~ j0_10, ~10 nodes), not the ground.
    virtual void fermi_wave() = 0;
};

// 2D quantum dot: parabolic confinement + optional uniform B -- the
// Fock-Darwin problem. Ground relax is ASYNC (poll relaxing()).
struct QdotApi {
    virtual ~QdotApi() = default;
    virtual void set_omega0(double w0) = 0;
    virtual double omega0() const = 0;
    virtual void set_field(double b) = 0;
    virtual double field() const = 0;
    virtual void relax_ground() = 0;
    virtual bool relaxing() const = 0;
    virtual double energy_meas() const = 0;
    virtual double energy_pred() const = 0;  // Omega = sqrt(w0^2 + B^2/4)
    virtual void fire_displaced() = 0;  // coherent orbit / rosette
    // 2D-HO extensions: circular ladder (B = 0 only -- gauge), a random coherent
    // packet, and the pick-and-gather grab (time freezes while held).
    virtual bool ho_ladder(bool up) = 0;
    virtual void random_packet() = 0;
    virtual void begin_grab(double x, double y) = 0;
    virtual void update_grab(double strength) = 0;
    virtual void end_grab() = 0;
    virtual bool grabbing() const = 0;
    // Fock-Darwin linear-combination spectrum (0..100 eV), lazy.
    virtual int spectrum_count() = 0;
    virtual double spectrum_ev(int i) = 0;
    virtual double spectrum_weight(int i) = 0;
};

// Pinned electron spin on the Bloch sphere (Pauli two-level stage).
struct SpinApi {
    virtual ~SpinApi() = default;
    virtual void set_b(int axis, double v) = 0;  // B free to point anywhere
    virtual double b(int axis) const = 0;
    virtual void set_e(int axis, double v) = 0;  // pinned spin: flux only
    virtual double e(int axis) const = 0;
    virtual void toggle_rf() = 0;  // co-rotating Rabi drive at omega_L
    virtual bool rf_on() const = 0;
    virtual void pulse(bool half) = 0;  // pi/2 (true) / pi about x
    virtual void spin_echo() = 0;       // detuned-ensemble sequence
    virtual double echo_peak() const = 0;
    virtual double bloch_x() = 0;
    virtual double bloch_y() = 0;
    virtual double bloch_z() = 0;
    virtual int last_outcome() const = 0;
};

// Interacting spins: mean-field Heisenberg lattice of Bloch spheres.
struct SpinsApi {
    virtual ~SpinsApi() = default;
    virtual void set_j(double j) = 0;  // exchange: >0 ferro, <0 Neel
    virtual double j() const = 0;
    virtual void set_alpha(double a) = 0;  // Gilbert damping
    virtual double alpha() const = 0;
    virtual void set_b(int axis, double v) = 0;
    virtual double b(int axis) const = 0;
    virtual void seed_random() = 0;
    virtual void seed_ferro() = 0;
    virtual void seed_neel() = 0;
    virtual double magnetization() = 0;
    virtual double staggered() = 0;
    // Exact Heisenberg <-> mean-field product switch. In exact mode entanglement
    // is real, so per-site arrows SHRINK below 1; arrow_mean() = mean |<sigma_i>|.
    virtual void set_exact(bool on) = 0;
    virtual bool exact_mode() const = 0;
    virtual double arrow_mean() = 0;
};

// Quantum bouncer: gravity + mirror, the Airy ladder.
struct BouncerApi {
    virtual ~BouncerApi() = default;
    virtual void relax_ground() = 0;  // instant ITP anneal to Airy 1
    virtual void drop() = 0;          // packet from height: bounce/revive
    virtual double energy() const = 0;
    virtual double airy_e1() const = 0;  // ideal hard-floor E1
};

// Quantum point contact: conductance staircase in the gap width.
struct QpcApi {
    virtual ~QpcApi() = default;
    virtual void set_gap(double w) = 0;  // constriction width (refires)
    virtual double gap() const = 0;
    virtual void fire() = 0;
    virtual double transmitted() const = 0;  // right-cap flux tally
    virtual int open_channels() const = 0;   // floor(w k0 / pi)
};

// Quantum carpet: free ring, temporal Talbot weave.
struct CarpetApi {
    virtual ~CarpetApi() = default;
    virtual void refire() = 0;
    virtual double revival_time() const = 0;     // T_rev = L^2 / pi
    virtual double revival_overlap() const = 0;  // |<psi0|psi>|^2 live
    // Row-cadence maxima (frame polling would miss the ~2 au peak):
    virtual double mid_scramble_max() const = 0;  // max in (0.15, 0.6) T
    virtual double best_revival() const = 0;      // max past 0.6 T
};

// Anderson localization: 1D speckle wire, conductance framing.
struct AndersonApi {
    virtual ~AndersonApi() = default;
    virtual void set_disorder(double w) = 0;  // speckle strength (refires)
    virtual double disorder() const = 0;
    virtual void reroll() = 0;  // fresh landscape (new seed, refires)
    virtual void refire() = 0;
    virtual double transmitted() const = 0;  // right-cap flux tally
    virtual double survived() const = 0;     // on-stage norm
};

// Quantum billiard: circle (integrable) vs Bunimovich stadium (chaotic).
struct BilliardApi {
    virtual ~BilliardApi() = default;
    virtual void fire() = 0;          // relaunch the tangential packet
    virtual void toggle_shape() = 0;  // circle <-> stadium (refires)
    virtual bool stadium() const = 0;
    virtual void toggle_avg_view() = 0;  // live |psi|^2 <-> time average
    virtual bool avg_view() const = 0;
    virtual double avg_center_fraction() const = 0;  // caustic metric
};

// Rutherford scattering: Gaussian packet fired head-on at a REPULSIVE Coulomb
// center (3D Coulomb generator, flipped sign). Sliders set incident KE E and
// target charge Z (defaults: gold Z=79 vs alpha charge 2 -> V = +2Z/r);
// classical closest approach r_min = 2Z/E.
struct RutherfordApi {
    virtual ~RutherfordApi() = default;
    virtual void set_energy(double e) = 0;  // incident KE (Ha); refires
    virtual double energy() const = 0;
    virtual void set_z(double z) = 0;  // target nuclear charge; refires
    virtual double z() const = 0;
    virtual void refire() = 0;
    virtual double turning_point() const = 0;      // classical r_min = 2Z/E
    virtual double closest_approach() const = 0;    // min <r> seen (arc)
    virtual double backscattered_fraction() const = 0;  // returned upstream
};

// A 1D-scene overlay primitive: packed (x, y, z) world-space triples; LINE_STRIP,
// or with `fill` a TRIANGLE_STRIP sheet. `rgba` (premultiplied) REPLACES the
// constant color. Both pointers stay valid until the next run_frame().
struct OverlayCurve {
    const float* xyz = nullptr;
    int count = 0;
    float r = 1.0f;
    float g = 1.0f;
    float b = 1.0f;
    float a = 1.0f;
    bool fill = false;
    const float* rgba = nullptr;  // per-vertex premultiplied color
};

// A nucleus marker BALL (world space, symbolic radius): proton / CPK atoms.
// Shaded as a sphere both views (mesh in Surface, raymarch in Cloud).
struct SceneMarker {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float radius = 0.35f;
    float r = 1.0f;
    float g = 1.0f;
    float b = 1.0f;
};

class ScenarioDirector {
public:
    virtual ~ScenarioDirector() = default;

    // Capability queries: non-null only for the scene that implements them.
    virtual HydrogenApi* hydrogen() { return nullptr; }
    virtual TunnelApi* tunnel() { return nullptr; }
    virtual Ladder1dApi* ladder1d() { return nullptr; }
    virtual DoubleWellApi* doublewell() { return nullptr; }
    virtual ReflectApi* reflect() { return nullptr; }
    virtual MorseApi* morse() { return nullptr; }
    virtual MoleculeApi* molecule() { return nullptr; }
    virtual SlitApi* slit() { return nullptr; }
    virtual LandauApi* landau() { return nullptr; }
    virtual BlochApi* bloch() { return nullptr; }
    virtual CorralApi* corral() { return nullptr; }
    virtual QdotApi* qdot() { return nullptr; }
    virtual BilliardApi* billiard() { return nullptr; }
    virtual AndersonApi* anderson() { return nullptr; }
    virtual CarpetApi* carpet() { return nullptr; }
    virtual QpcApi* qpc() { return nullptr; }
    virtual BouncerApi* bouncer() { return nullptr; }
    virtual SpinApi* spin() { return nullptr; }
    virtual SpinsApi* spins() { return nullptr; }
    virtual RutherfordApi* rutherford() { return nullptr; }

    // 1D-scene overlay polylines (phasor curve + potential profile); 3D scenes
    // return 0.
    virtual int overlay_curve_count() const { return 0; }
    virtual OverlayCurve overlay_curve(int /*i*/) const { return {}; }

    // Photons from quantum jumps (0 if none); generic so arcs can probe every
    // jump-capable scene.
    virtual long long photon_count() const { return 0; }

    // Scene-prop hints, display only (physics never reads them): nucleus marker
    // BALLS (default single origin sphere; molecules supply a CPK list; barrier
    // scenes return 0) and a potential slab [lo, hi) on x.
    virtual int marker_count() const { return 1; }
    virtual SceneMarker marker(int /*i*/) const {
        return {0.0f, 0.0f, 0.0f, 0.35f, 1.0f, 0.45f, 0.20f};
    }
    virtual bool barrier_slab(double& /*lo*/, double& /*hi*/) const {
        return false;
    }

    // Scene-chosen boot camera (shell owns it afterwards); generic 3/4 view,
    // tunneling wants the slab edge-on.
    virtual double default_camera_azimuth() const { return 0.6; }
    virtual double default_camera_elevation() const { return 0.4; }

    // ---- lifecycle ----
    virtual const ses::Grid3D& grid() const = 0;
    virtual void init_compute(ses_vk::DeviceContext& ctx, bool device_ok,
                              std::int64_t free_vram_bytes) = 0;
    virtual void release_gpu() = 0;
    virtual bool compute_attempted() const = 0;
    virtual bool gpu_ok() const = 0;

    // ---- per-frame / per-tick ----
    virtual void run_frame() = 0;
    virtual void tick() = 0;

    // ---- controls every scenario supports ----
    // Real time = x1: routing back must also clear the time scale, or a raised
    // slider survives as a sticky speedup. NVI: scenes override do_set_real_time().
    void set_real_time() {
        do_set_real_time();
        set_time_scale(1);
    }
    virtual void do_set_real_time() = 0;
    virtual void reset_simulation() = 0;
    virtual void measure_now() = 0;
    virtual void toggle_view_mode() = 0;
    // Scenario-specific keys (upper-case letters / digits); true = handled.
    virtual bool handle_key(char key) = 0;

    // ---- state the shell gates on ----
    virtual bool solving() const = 0;      // startup solve owns the GPU state
    virtual bool scene_ready() const = 0;  // demo fully armed (selftest gate)

    // Camera start distance framing this scene's box (Bohr).
    virtual double default_camera_distance() const { return 150.0; }

    // Visualized time scale: multiply the steps SUPPLIED per tick, not dt --
    // more steps per frame, never larger ones, so accuracy is preserved (GPU
    // saturation just lowers fps).
    virtual void set_time_scale(int scale) { (void)scale; }
    virtual int time_scale() const { return 1; }

    // Total simulated time and integrator step (au); the perf readout derives
    // au/s from these.
    virtual double sim_time() const { return 0.0; }
    virtual double sim_dt() const { return 0.0; }
    // Integrator steps a x1 tick SUPPLIES; the honest baseline is ticks/s x dt x THIS.
    virtual int steps_per_tick_x1() const { return 1; }

    // ---- display accessors (FrameInput assembly + title) ----
    virtual bool cloud() const = 0;
    virtual double peak() const = 0;
    virtual VkImageView psi_volume_view() = 0;
    // Low-res fp32 Bohmian-velocity volume for streaklines (null -> renderer
    // skips flow; only GPU cloud scenes provide it).
    virtual VkImageView flow_velocity_view() { return VK_NULL_HANDLE; }
    // GPU surface mesh (non-null when extracted on-GPU); renderer draws these
    // and ignores mesh()/colors().
    virtual VkBuffer surface_vbuf() const { return VK_NULL_HANDLE; }
    virtual VkBuffer surface_indirect() const { return VK_NULL_HANDLE; }
    // GPU-surface vertex cap; overflow clamps to a clean prefix (engine warns once).
    static constexpr int kMcMaxTris = 1000000;
    virtual float next_flash_intensity() = 0;
    virtual bool take_volume_written() = 0;
    virtual bool take_volume_dirty() = 0;
    virtual bool take_mesh_dirty() = 0;
    virtual void mark_display_dirty() = 0;
    virtual bool take_title_dirty() = 0;
    virtual const std::vector<float>& psi_staging() const = 0;
    virtual const ses::Mesh& mesh() const = 0;
    virtual const std::vector<ses::Rgb>& colors() const = 0;
    virtual std::string title_text() = 0;
};

}  // namespace ses_shell
