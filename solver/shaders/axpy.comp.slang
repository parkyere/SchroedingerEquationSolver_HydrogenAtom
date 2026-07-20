// c = <phi|psi>: deflation projection (relax_deflated step).
// std140: vec2 c at offset 8 (uint n + 4B pad).
import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> psi;
[[vk::binding(3, 0)]] StructuredBuffer<Complex, Std430DataLayout> phi;

struct Params {
    uint n;
    float2 c;
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= params.n) {
        return;
    }
    psi[i] = psi[i] - Complex(params.c) * phi[i];
}
