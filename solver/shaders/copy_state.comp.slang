// psi <- src, quantum-jump collapse. Plain assignment is bitwise-exact (no arithmetic touches the values).
import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> psi;
[[vk::binding(3, 0)]] StructuredBuffer<Complex, Std430DataLayout> src;

struct Params {
    uint n;
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i < params.n) {
        psi[i] = src[i];
    }
}
