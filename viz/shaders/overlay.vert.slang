// 1D-scene overlay polyline vertex: LINE_STRIP / TRIANGLE_STRIP over one
// packed float SSBO laid out [positions 3f...][colors 4f...], all curves
// back to back. gl_VertexIndex includes the draw's firstVertex, so one
// buffer serves several curves -- each draw passes its base as firstVertex
// and its color via push constants: either the constant premultiplied
// color (the white phasor curve keeps phase as GEOMETRY, the twist about
// the x axis), or, with params.x set, per-vertex premultiplied rgba pulled
// from the color region at float-index 4 * gl_VertexIndex + params.y (the
// phase-hued density band on the z = 0 plane; the offset folded out the
// position base host-side and may be negative, hence the int math).
import glsl;

[[vk::binding(0, 0)]] StructuredBuffer<float, Std430DataLayout> pts;

// Layout-compatible prefix of the shared MeshUbo (mvp + eye); only the mvp
// is read here.
struct Ubo {
    float4x4 mvp;
    float4 eye;
};
[[vk::binding(1, 0)]] ConstantBuffer<Ubo, Std140DataLayout> ubo;

struct Push {
    float4 color;   // premultiplied (rgb scaled by a), a = coverage
    float4 params;  // x: 1 = per-vertex rgba, y: color float-index offset
};
[[vk::push_constant]] ConstantBuffer<Push> push;

struct VSOut {
    float4 pos : SV_Position;
    [[vk::location(0)]] float4 v_color : TEXCOORD0;
};

[shader("vertex")]
VSOut main() {
    VSOut o;
    uint vid = uint(gl_VertexIndex);
    float3 p = float3(pts[3 * vid + 0], pts[3 * vid + 1], pts[3 * vid + 2]);
    if (push.params.x > 0.5) {
        int ca = 4 * int(vid) + int(push.params.y);
        o.v_color = float4(pts[ca + 0], pts[ca + 1], pts[ca + 2],
                           pts[ca + 3]);
    } else {
        o.v_color = push.color;
    }
    o.pos = mul(ubo.mvp, float4(p, 1.0));
    return o;
}
