// fp16 pack halves the resident-state atlas footprint.
// packHalf2x16 (glsl module -> GLSL.std.450 op) needs no shaderFloat16 capability; f32tof16 would emit one.
import glsl;

[[vk::binding(0, 0)]] StructuredBuffer<float2, Std430DataLayout> src;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint, Std430DataLayout> dst;

struct Params {
    uint n;
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i < params.n) {
        dst[i] = packHalf2x16(src[i]);
    }
}
