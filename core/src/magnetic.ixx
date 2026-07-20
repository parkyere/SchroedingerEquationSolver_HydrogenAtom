module;
#include <cmath>
#include <cstddef>
#include <vector>
export module ses.magnetic;
export import ses.grid;
export import ses.propagator;
export import ses.rotation;
export import ses.field;


// Magnetic split-operator: uniform B along axis, symmetric gauge, atomic units.
//   H = p^2/2 + V + (B/2) L_axis + (B^2/8) rho_perp^2.
// Paramagnetic (B/2)L_axis = exact three-shear rotate_axis; diamagnetic -> core V.
// Strang: R(a).[halfV.kin.halfV].R(a), a = (B/2)(dt/2). Unitary.


export namespace ses {

class MagneticPropagator3D {
public:
    // axis: 0=x, 1=y, 2=z
    MagneticPropagator3D(const Grid3D& g, const std::vector<double>& v, double dt,
                         double bfield, int axis = 2)
        : dt_(dt), half_angle_(0.5 * bfield * (0.5 * dt)), axis_(axis),
          veff_(build_veff(g, v, bfield, axis)), core_(g, veff_, dt) {}

    constexpr double dt() const noexcept { return dt_; }

    const std::vector<double>& effective_potential() const noexcept { return veff_; }

    // Adjacent half-rotations across a step boundary merge to the full Larmor angle (B/2) dt.
    void step(Field3D& psi, int nsteps = 1) const {
        for (int s = 0; s < nsteps; ++s) {
            if (half_angle_ != 0.0) {
                rotate_axis(psi, axis_, half_angle_);
            }
            core_.step(psi, 1);
            if (half_angle_ != 0.0) {
                rotate_axis(psi, axis_, half_angle_);
            }
        }
    }

private:
    static std::vector<double> build_veff(const Grid3D& g, const std::vector<double>& v,
                                          double bfield, int axis) {
        std::vector<double> veff = v;
        const double c = bfield * bfield / 8.0;
        if (c != 0.0) {
            for (int k = 0; k < g.z.n; ++k) {
                for (int j = 0; j < g.y.n; ++j) {
                    for (int i = 0; i < g.x.n; ++i) {
                        const double coord[3] = {g.x.coord(i), g.y.coord(j),
                                                 g.z.coord(k)};
                        double perp2 = 0.0;
                        for (int a = 0; a < 3; ++a) {
                            if (a != axis) {
                                perp2 += coord[a] * coord[a];
                            }
                        }
                        veff[static_cast<std::size_t>(g.flat(i, j, k))] += c * perp2;
                    }
                }
            }
        }
        return veff;
    }

    double dt_;
    double half_angle_;
    int axis_;
    std::vector<double> veff_;
    SplitOperator3D core_;
};

}  // namespace ses
