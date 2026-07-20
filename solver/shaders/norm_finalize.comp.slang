// Single-workgroup fixed-order tree-sum (no atomics) => bitwise-deterministic.
// Folds only partials[].x (|psi|^2); the fp64 energy/peak diagnostic is finished
// on the host from these same partials, so this kernel just writes inv for the
// fused rescale in the same submit (no host round-trip).
[[vk::binding(0, 0)]] StructuredBuffer<float2, Std430DataLayout> partials;
[[vk::binding(2, 0)]] RWStructuredBuffer<float, Std430DataLayout> renorm;  // [0] = inv = 1/sqrt(norm_sq)

struct Params {
    uint ngroups;       // valid partials (== kGroups)
    float cell_volume;  // dV, so sum*dV = <psi|psi>
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

groupshared float s_sum[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint tid : SV_GroupIndex) {
    s_sum[tid] = (tid < params.ngroups) ? partials[tid].x : 0.0;
    GroupMemoryBarrierWithGroupSync();
    for (uint half_len = 128u; half_len > 0u; half_len >>= 1u) {
        if (tid < half_len) {
            s_sum[tid] += s_sum[tid + half_len];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) {
        float norm_sq = s_sum[0] * params.cell_volume;
        renorm[0] = (norm_sq > 0.0) ? rsqrt(norm_sq) : 0.0;
    }
}
