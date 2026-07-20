// psi_tex = RG16F (re, im). Transcribes tested ses.volume formulas. UBO (binding 0) shared with the vertex stage.
import vec;

struct FSIn {
    float4 fragCoord : SV_Position;
    [[vk::location(0)]] float3 v_world : TEXCOORD0;
};

struct Ubo {
    float4x4 mvp;
    float4 eye;
    float4 box_min;
    float4 box_max;
    float4 proton_center;   // reserved (layout)
    float4 proton_color;    // reserved (layout)
    float inv_peak;
    float absorbance;
    float proton_radius;    // reserved (layout)
    float jitter_frame;
    float4 clip;   // .x enable, .y axis(0/1/2), .z sign(+-1), .w offset
    float4 slice;  // slice pass; shared UBO layout
    float4 barrier;  // .x enable, .y x_lo, .z x_hi, .w fog density
    // marker_cr = xyz center + w radius; marker_col = rgb; count = marker_meta.x
    float4 marker_cr[64];
    float4 marker_col[64];
    float4 marker_meta;
};
[[vk::binding(0, 0)]] ConstantBuffer<Ubo, Std140DataLayout> ubo;

[[vk::binding(1, 0)]] Sampler3D<float4> psi_tex;
[[vk::binding(2, 0)]] Sampler1D<float4> phase_tex;
[[vk::binding(3, 0)]] Sampler3D<float4> occ_tex;     // dilated block max (NEAREST)
[[vk::binding(4, 0)]] Sampler3D<float4> shadow_tex;  // key-light transmittance

float2 ray_box(float3 o, float3 d) {
    float3 inv = 1.0 / d;
    float3 t1 = (ubo.box_min.xyz - o) * inv;
    float3 t2 = (ubo.box_max.xyz - o) * inv;
    float3 tmin = min(t1, t2);
    float3 tmax = max(t1, t2);
    return float2(max(max(tmin.x, tmin.y), tmin.z),
                  min(min(tmax.x, tmax.y), tmax.z));
}

// Mirrors tested ses::ray_sphere; assumes unit direction.
float2 ray_sphere(float3 o, float3 d, float3 center, float radius) {
    float3 oc = o - center;
    float b = dot(oc, d);
    float c = dot(oc, oc) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0) {
        return float2(1.0, -1.0);  // empty interval = miss
    }
    float s = sqrt(disc);
    return float2(-b - s, -b + s);
}

[shader("fragment")]
float4 main(FSIn input) : SV_Target {
    float3 v_world = input.v_world;
    float3 dir = normalize(v_world - ubo.eye.xyz);
    float2 t = ray_box(ubo.eye.xyz, dir);
    float tn = max(t.x, 0.0);
    if (t.y <= tn) {
        discard;
    }
    float t_stop = t.y;

    // Mirrors ses::clip_ray_interval / ses.cross_section: keep half-space clip.z*(p[axis]-offset) <= 0.
    if (ubo.clip.x > 0.5) {
        int ax = int(ubo.clip.y + 0.5);
        float a = ubo.clip.z * comp(dir, ax);
        float b = ubo.clip.z * (ubo.clip.w - comp(ubo.eye.xyz, ax));
        if (abs(a) < 1e-8) {
            if (b < 0.0) discard;  // whole ray on the cut side
        } else {
            float tp = b / a;
            if (a > 0.0) t_stop = min(t_stop, tp);
            else tn = max(tn, tp);
        }
        if (t_stop <= tn) discard;
    }

    // Stop march at nearest marker: front cloud fogs it, back is occluded.
    int hit_marker = -1;
    float hit_t = 0.0;
    int n_markers = int(ubo.marker_meta.x + 0.5);
    for (int m = 0; m < n_markers; ++m) {
        float2 sp = ray_sphere(ubo.eye.xyz, dir, ubo.marker_cr[m].xyz,
                               ubo.marker_cr[m].w);
        if (sp.x <= sp.y && sp.x > tn && sp.x < t_stop) {
            hit_marker = m;
            hit_t = sp.x;
            t_stop = sp.x;
        }
    }

    const int kSteps = 384;  // ~0.42 Bohr/sample across the box
    float step_len = (t_stop - tn) / float(kSteps);
    // Interleaved gradient noise + per-frame golden-ratio rotate: kills banding, decorrelates accumulation.
    float2 jp = input.fragCoord.xy + 5.588238 * ubo.jitter_frame;
    float jitter = frac(52.9829189 * frac(0.06711056 * jp.x + 0.00583715 * jp.y));

    // Half-texel align: texel i center = (i+0.5)/n vs grid point i, else cloud is half a cell off.
    uint tw, th, tdp;
    psi_tex.GetDimensions(tw, th, tdp);
    float3 half_texel = 0.5 / float3(tw, th, tdp);
    float3 ext = ubo.box_max.xyz - ubo.box_min.xyz;
    // Empty-space skip: jump a block where occupancy is empty; fine-march at original step elsewhere.
    float skip_len = min(min(ext.x, ext.y), ext.z) / 32.0;

    float3 C = float3(0.0, 0.0, 0.0);
    float A = 0.0;
    float A_b = 0.0;  // barrier slab opacity budget
    float t_cur = tn + jitter * step_len;
    int fine_budget = kSteps;
    while (t_cur < t_stop && fine_budget > 0) {
        float3 pc = ubo.eye.xyz + t_cur * dir;
        float3 uvw_c = (pc - ubo.box_min.xyz) / ext;
        // Barrier slab has no occupancy: don't block-skip near it or the wall vanishes.
        bool near_slab = ubo.barrier.x > 0.5 &&
                         pc.x >= ubo.barrier.y - skip_len &&
                         pc.x < ubo.barrier.z + skip_len;
        if (!near_slab && occ_tex.Sample(uvw_c).r < 1e-5) {
            t_cur += skip_len;
            continue;
        }
        for (int k = 0; k < 8 && t_cur < t_stop && fine_budget > 0 &&
                        A < 0.999; ++k) {
            float3 p = ubo.eye.xyz + t_cur * dir;
            float3 uvw = (p - ubo.box_min.xyz) / ext + half_texel;
            float2 s = psi_tex.Sample(uvw).rg;
            float dens = dot(s, s) * ubo.inv_peak;
            float alpha = 1.0 - exp(-ubo.absorbance * dens * step_len);
            float phase01 =
                (atan2(s.g, s.r) + 3.14159265358979) / 6.28318530717959;
            float3 col = phase_tex.Sample(phase01).rgb;
            // 0.35 ambient floor keeps nodes readable.
            float T = shadow_tex.Sample(uvw_c).r;
            col *= 0.35 + 0.65 * T;
            float w = (1.0 - A) * alpha;
            C += w * col;
            A += w;
            // Barrier fog in the same march; opacity capped (A_b) so grazing rays don't white out.
            if (ubo.barrier.x > 0.5 && p.x >= ubo.barrier.y &&
                p.x < ubo.barrier.z && A_b < 0.35) {
                float ba = 1.0 - exp(-ubo.barrier.w * step_len);
                float bw = min((1.0 - A) * ba, 0.35 - A_b);
                C += bw * float3(0.50, 0.56, 0.68);
                A += bw;
                A_b += bw;
            }
            t_cur += step_len;
            --fine_budget;
        }
        if (A >= 0.999) {
            break;
        }
    }

    if (hit_marker >= 0) {
        float3 p = ubo.eye.xyz + hit_t * dir;
        float3 n = (p - ubo.marker_cr[hit_marker].xyz) /
                   ubo.marker_cr[hit_marker].w;
        float diffuse = max(dot(n, -dir), 0.0);
        float spec = pow(diffuse, 32.0);
        float3 shaded =
            ubo.marker_col[hit_marker].xyz * (0.25 + 0.75 * diffuse) +
            float3(0.2, 0.2, 0.2) * spec;
        C += (1.0 - A) * shaded;
        A = 1.0;
    }

    return float4(C, A);  // premultiplied alpha
}
