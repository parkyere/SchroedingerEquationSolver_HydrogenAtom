// Unpack fp16-packed uint (src) -> complex fp32 field float2 (dst); inverse of
// pack_half. unpackHalf2x16 needs no shaderFloat16 capability.
import glsl;

[[vk::binding(6, 0)]] StructuredBuffer<uint, Std430DataLayout> src;
[[vk::binding(0, 0)]] RWStructuredBuffer<float2, Std430DataLayout> dst;

struct Params {
    uint n;
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i < params.n) {
        dst[i] = unpackHalf2x16(src[i]);
    }
}
