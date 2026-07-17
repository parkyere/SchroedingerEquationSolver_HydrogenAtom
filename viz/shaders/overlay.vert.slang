// 1D-scene overlay polyline vertex: LINE_STRIP over a packed (x, y, z)
// float SSBO holding all curves back to back. gl_VertexIndex includes the
// draw's firstVertex, so one buffer serves several curves -- each draw
// passes its base as firstVertex and its constant color as a push constant.
// No per-vertex color on purpose: the 1D scenes encode phase as geometry
// (the phasor twist about the x axis), not as hue.
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
    float4 color;  // premultiplied (rgb scaled by a), a = coverage
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
    o.v_color = push.color;
    o.pos = mul(ubo.mvp, float4(p, 1.0));
    return o;
}
