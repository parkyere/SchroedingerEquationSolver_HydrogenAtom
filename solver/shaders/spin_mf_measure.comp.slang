// Mean-field Born measurement: each site independently measured along axis n by
// its Bloch projection, collapsed to the |+-n> eigenstate. Per-site uniform
// draws u[i] come from the host RNG (a scalar, not a CPU engine). fp32 mirror
// of ses::spin_measure. CONTRACT: vkcheck check_spin_mf_measure.
import complex;
import spin;

[[vk::binding(0, 0)]] RWStructuredBuffer<float4, Std430DataLayout> spinors;
[[vk::binding(1, 0)]] RWStructuredBuffer<float, Std430DataLayout> bloch;

struct Axis {
    float nx, ny, nz, pad;
};
[[vk::binding(2, 0)]] ConstantBuffer<Axis, Std140DataLayout> axis;
[[vk::binding(3, 0)]] RWStructuredBuffer<float, Std430DataLayout> uu;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= 16u) {
        return;
    }
    float3 n = float3(axis.nx, axis.ny, axis.nz);
    Spinor s = load_spinor(spinors[i]);
    float3 b = bloch_of(s);
    float p_plus = 0.5 * (1.0 + dot(b, n));
    float sgn = (uu[i] < p_plus) ? 1.0 : -1.0;
    s = spinor_from_bloch(sgn * n);  // collapse to the outcome eigenstate
    spinors[i] = store_spinor(s);
    float3 pb = bloch_of(s);
    bloch[3u * i + 0u] = pb.x;
    bloch[3u * i + 1u] = pb.y;
    bloch[3u * i + 2u] = pb.z;
}
