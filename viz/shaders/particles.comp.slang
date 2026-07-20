// Probability-flow streaklines: head advects a Bohmian step v = j/rho,
// j = Im(conj(psi) grad psi) (au, ses.flow::probability_current). Respawn =
// rejection-sample |psi|^2 (equivariant: streaks stay |psi|^2-distributed).
[[vk::binding(0, 0)]] RWStructuredBuffer<float4, Std430DataLayout> trail;

struct Params {
    float4 box_min;
    float4 box_max;
    float dt;        // au
    float inv_peak;  // 1/peak |psi|^2
    float lifetime;  // frames
    float frame;     // RNG decorrelation
    int n_streaks;
    int trail_len;
    int animate;
    int reset;
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[[vk::binding(2, 0)]] Sampler3D<float4> psi_tex;  // fp16 display volume
[[vk::binding(3, 0)]] Sampler3D<float4> vel_tex;  // fp32 Bohmian velocity

float2 psi_at(float3 p) {
    // half-texel: grid point i at center (i+0.5)/n
    uint w, h, d;
    psi_tex.GetDimensions(w, h, d);
    float3 uvw = (p - params.box_min.xyz) / (params.box_max.xyz - params.box_min.xyz) +
                 0.5 / float3(w, h, d);
    return psi_tex.SampleLevel(uvw, 0.0).xy;
}

// fp32 (not fp16 display): real eigenstate reads v ~ 0, no spurious pulsation.
float3 vel_at(float3 p) {
    uint w, h, d;
    vel_tex.GetDimensions(w, h, d);
    float3 uvw = (p - params.box_min.xyz) / (params.box_max.xyz - params.box_min.xyz) +
                 0.5 / float3(w, h, d);
    return vel_tex.SampleLevel(uvw, 0.0).xyz;
}

// PCG-ish hash -> [0,1).
float rnd(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint w = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    w = (w >> 22u) ^ w;
    return float(w) / 4294967296.0;
}

// Rejection-sample respawn vs |psi|^2; on fail keep last draw (re-gated next frame).
float3 respawn_seed(inout uint rng) {
    float3 p = params.box_min.xyz;
    for (int t = 0; t < 4; ++t) {
        float3 cand = params.box_min.xyz +
                      float3(rnd(rng), rnd(rng), rnd(rng)) *
                          (params.box_max.xyz - params.box_min.xyz);
        p = cand;
        float2 cs = psi_at(cand);
        if (rnd(rng) < dot(cs, cs) * params.inv_peak) {
            break;
        }
    }
    return p;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    if (i >= uint(params.n_streaks)) {
        return;
    }
    uint L = uint(params.trail_len);
    uint base = i * L;
    float4 head = trail[base + L - 1u];
    float3 p = head.xyz;
    float age = head.w;

    uint rng = i * 9781u + uint(params.frame) * 6271u + 12345u;

    float2 s = psi_at(p);
    float rho = dot(s, s);
    bool dead = age > params.lifetime || rho * params.inv_peak < 1e-4 ||
                any(p < params.box_min.xyz) ||
                any(p > params.box_max.xyz);

    if (params.reset != 0 || dead) {
        float3 seed = respawn_seed(rng);
        float na = params.lifetime * rnd(rng) * 0.25;  // stagger lifetimes
        for (uint k = 0u; k < L; ++k) {
            trail[base + k] = float4(seed, na);
        }
        return;
    }

    if (params.animate != 0) {
        float3 v = vel_at(p);  // already node-guarded/clamped
        p += v * params.dt;
        age += 1.0;

        // One thread owns its L slots; ascending copy reads k+1 before overwrite -> no aliasing.
        for (uint k = 0u; k + 1u < L; ++k) {
            trail[base + k] = trail[base + k + 1u];
        }
        trail[base + L - 1u] = float4(p, age);
    }
}
