module;
#include <cmath>
export module ses.camera;
import ses.vec;


// Column-major Mat4: element(row r, col c) = m[c*4 + r]. Right-handed view,
// camera looks down -Z. NDC depth [-1,+1] (OpenGL); the Vulkan renderer
// applies its own y-flip/depth-remap. Contract: tests/camera_test.cpp.


export namespace ses {

struct Mat4 {
    double m[16]{};

    static constexpr Mat4 identity() noexcept {
        Mat4 r;
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0;
        return r;
    }

    static constexpr Mat4 translation(Vec3d t) noexcept {
        Mat4 r = identity();
        r.m[12] = t.x;
        r.m[13] = t.y;
        r.m[14] = t.z;
        return r;
    }

    static constexpr Mat4 scale(double s) noexcept {
        Mat4 r = identity();
        r.m[0] = r.m[5] = r.m[10] = s;
        return r;
    }
};

inline constexpr Mat4 operator*(const Mat4& a, const Mat4& b) noexcept {
    Mat4 r;
    for (int c = 0; c < 4; ++c) {
        for (int row = 0; row < 4; ++row) {
            double acc = 0.0;
            for (int k = 0; k < 4; ++k) {
                acc += a.m[k * 4 + row] * b.m[c * 4 + k];
            }
            r.m[c * 4 + row] = acc;
        }
    }
    return r;
}

inline constexpr Vec3d transform_point(const Mat4& a, Vec3d p) noexcept {
    const double x = a.m[0] * p.x + a.m[4] * p.y + a.m[8] * p.z + a.m[12];
    const double y = a.m[1] * p.x + a.m[5] * p.y + a.m[9] * p.z + a.m[13];
    const double z = a.m[2] * p.x + a.m[6] * p.y + a.m[10] * p.z + a.m[14];
    const double w = a.m[3] * p.x + a.m[7] * p.y + a.m[11] * p.z + a.m[15];
    return {x / w, y / w, z / w};
}

// gluLookAt.
inline Mat4 look_at(Vec3d eye, Vec3d center, Vec3d up) noexcept {
    const Vec3d f = normalized(center - eye);
    const Vec3d s = normalized(cross(f, up));
    const Vec3d u = cross(s, f);

    Mat4 r = Mat4::identity();
    r.m[0] = s.x;
    r.m[4] = s.y;
    r.m[8] = s.z;
    r.m[1] = u.x;
    r.m[5] = u.y;
    r.m[9] = u.z;
    r.m[2] = -f.x;
    r.m[6] = -f.y;
    r.m[10] = -f.z;
    r.m[12] = -dot(s, eye);
    r.m[13] = -dot(u, eye);
    r.m[14] = dot(f, eye);
    return r;
}

// gluPerspective; fovy radians.
inline Mat4 perspective(double fovy, double aspect, double znear, double zfar) noexcept {
    const double f = 1.0 / std::tan(fovy / 2.0);
    Mat4 r;
    r.m[0] = f / aspect;
    r.m[5] = f;
    r.m[10] = (zfar + znear) / (znear - zfar);
    r.m[11] = -1.0;
    r.m[14] = 2.0 * zfar * znear / (znear - zfar);
    return r;
}

// azimuth=elevation=0 -> +Z axis; +azimuth -> +X, +elevation -> +Y.
inline Vec3d orbit_eye(double azimuth, double elevation, double distance, Vec3d target) noexcept {
    const double ce = std::cos(elevation);
    return target + distance * Vec3d{ce * std::sin(azimuth), std::sin(elevation),
                                     ce * std::cos(azimuth)};
}

// Unproject an NDC point onto the z=0 stage plane. Contract:
// tests/pick_test.cpp round-trips perspective * look_at.
inline bool unproject_to_z0(double azimuth, double elevation,
                            double distance, double fovy, double aspect,
                            double ndc_x, double ndc_y, double* out_x,
                            double* out_y) noexcept {
    const Vec3d eye = orbit_eye(azimuth, elevation, distance, Vec3d{});
    const Vec3d f = normalized(Vec3d{} - eye);
    const Vec3d s = normalized(cross(f, Vec3d{0.0, 1.0, 0.0}));
    const Vec3d u = cross(s, f);
    const double tf = std::tan(0.5 * fovy);
    const double cx = ndc_x * tf * aspect;
    const double cy = ndc_y * tf;
    const Vec3d dir{f.x + s.x * cx + u.x * cy, f.y + s.y * cx + u.y * cy,
                    f.z + s.z * cx + u.z * cy};
    if (std::abs(dir.z) < 1e-12) {
        return false;  // ray parallel to plane
    }
    const double t = -eye.z / dir.z;
    if (t <= 0.0) {
        return false;  // behind the eye
    }
    *out_x = eye.x + t * dir.x;
    *out_y = eye.y + t * dir.y;
    return true;
}

}  // namespace ses
