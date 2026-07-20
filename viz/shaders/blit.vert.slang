// Fullscreen-triangle.
// gl_VertexIndex (glsl module) avoids SV_VertexID's shaderDrawParameters capability.
import glsl;

struct VSOut {
    float4 pos : SV_Position;
    [[vk::location(0)]] float2 v_uv : TEXCOORD0;
};

[shader("vertex")]
VSOut main() {
    VSOut o;
    uint vid = uint(gl_VertexIndex);
    float2 pos = float2(float((vid << 1) & 2), float(vid & 2));
    o.v_uv = pos;
    o.pos = float4(pos * 2.0 - 1.0, 0.0, 1.0);
    return o;
}
