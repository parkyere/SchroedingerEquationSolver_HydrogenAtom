// STUB (RED): zeros out the Bloch buffer so check_spin_bloch fails; the real
// per-site partial-trace reduction lands in the green commit.
import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> state;
[[vk::binding(1, 0)]] RWStructuredBuffer<float, Std430DataLayout> outv;

struct Params {
    uint half_n;
    uint pad0;
    uint pad1;
    uint pad2;
};
[[vk::binding(2, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    if (gtid.x == 0u) {
        outv[3u * gid.x + 0u] = 0.0;
        outv[3u * gid.x + 1u] = 0.0;
        outv[3u * gid.x + 2u] = 0.0;
    }
}
