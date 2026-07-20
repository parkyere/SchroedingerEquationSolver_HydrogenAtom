// Kinetic phase; coef = -dt/2, k tables in FFT freq order.
import complex;
import grid;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> psi;
[[vk::binding(1, 0)]] StructuredBuffer<float, Std430DataLayout> kx2;
[[vk::binding(2, 0)]] StructuredBuffer<float, Std430DataLayout> ky2;
[[vk::binding(3, 0)]] StructuredBuffer<float, Std430DataLayout> kz2;

struct Params {
    uint n;
    int nx;
    int ny;
    float coef;
};
[[vk::binding(4, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= params.n) {
        return;
    }
    int3 k = unflatten_pow2(i, params.nx, params.ny);
    float a = params.coef * (kx2[k.x] + ky2[k.y] + kz2[k.z]);
    psi[i] = psi[i] * cis(a);
}
