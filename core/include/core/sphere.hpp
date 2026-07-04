#pragma once

// UV-sphere triangle soup for the nucleus marker. Lattice directions are
// computed once and shared by adjacent triangles (longitude wraps by index),
// so seams and poles weld bitwise -- watertight by construction. Normals are
// the exact unit radials.

#include <core/marching_cubes.hpp>
#include <core/vec.hpp>

#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

namespace ses {

inline Mesh sphere_mesh(Vec3d center, double radius, int rings, int segments) {
    // Unit directions on a (rings+1) x segments lattice; theta = 0 is the
    // +z pole, theta = pi the -z pole.
    std::vector<Vec3d> dir(static_cast<std::size_t>((rings + 1) * segments));
    for (int i = 0; i <= rings; ++i) {
        const double theta = std::numbers::pi * i / rings;
        const double st = std::sin(theta);
        const double ct = std::cos(theta);
        for (int j = 0; j < segments; ++j) {
            const double phi = 2.0 * std::numbers::pi * j / segments;
            dir[static_cast<std::size_t>(i * segments + j)] =
                Vec3d{st * std::cos(phi), st * std::sin(phi), ct};
        }
    }
    auto at = [&](int i, int j) {
        return dir[static_cast<std::size_t>(i * segments + (j % segments))];
    };

    Mesh m;
    auto emit = [&](Vec3d u) {
        m.vertices.push_back(center + radius * u);
        m.normals.push_back(u);
    };

    for (int j = 0; j < segments; ++j) {
        // top cap: pole row 0 collapses to a single point
        emit(at(0, j));
        emit(at(1, j));
        emit(at(1, j + 1));
        // middle bands
        for (int i = 1; i + 1 < rings; ++i) {
            const Vec3d a = at(i, j);
            const Vec3d b = at(i + 1, j);
            const Vec3d c = at(i + 1, j + 1);
            const Vec3d d = at(i, j + 1);
            emit(a);
            emit(b);
            emit(c);
            emit(a);
            emit(c);
            emit(d);
        }
        // bottom cap: pole row `rings` collapses to a single point
        emit(at(rings - 1, j));
        emit(at(rings, j));
        emit(at(rings - 1, j + 1));
    }
    return m;
}

}  // namespace ses
