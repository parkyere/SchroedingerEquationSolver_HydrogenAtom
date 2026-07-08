#pragma once

// Orbital synthesis + fp16 (half) codec on the GPU (Stage 3). Builds an
// eigenstate psi = (u/r) Y_lm straight into a new normalized GPU buffer (no CPU
// field), plus the small-VRAM fp16 store: pack/unpack a state to/from packed
// half floats so the whole n<=6 manifold fits an 8 GB card. The tested fp32
// consumers never change -- an fp16 state is decoded to a scratch fp32 buffer
// on demand. Extracted from GpuEngine, verified by sesolver_gpucheck's
// synthesis (T10) and fp16 roundtrip checks.
//
// NOT a standalone header: gpu_engine.hpp #includes this INSIDE namespace
// ses_gpu, after its includes and shared utilities (bind::, build_program,
// make_buffer, run_norm_peak, run_scale, NormPeak, GpuState, and kSynthSrc /
// kPackHalfSrc / kUnpackHalfSrc from gpu_shaders.hpp) are in scope.

struct OrbitalSynth {
    GLuint synth_prog_ = 0;
    GLuint pack_prog_ = 0;
    GLuint unpack_prog_ = 0;
    GLuint radial_u_buf_ = 0;          // radial u_nl(r) table for synthesis
    std::size_t radial_u_count_ = 0;   // current radial table length

    bool build(Gl& gl) {
        synth_prog_ = build_program(gl, kSynthSrc, "orbital synthesis");
        pack_prog_ = build_program(gl, kPackHalfSrc, "pack half");
        unpack_prog_ = build_program(gl, kUnpackHalfSrc, "unpack half");
        return synth_prog_ != 0 && pack_prog_ != 0 && unpack_prog_ != 0;
    }

    // ---- fp16 (half) atlas storage -------------------------------------
    // A small-VRAM fallback: store an eigenstate as packed fp16 (uint per
    // complex cell, HALF the fp32 footprint) so the whole n<=6 manifold fits an
    // 8 GB card. The tested fp32 consumers (inner product / dipole) never
    // change -- an fp16 state is unpacked to a scratch fp32 buffer on demand
    // (decode-on-use). Accuracy: ~1e-3 relative, safe for populations/dipoles.

    // Allocate a write-once fp16 state buffer (cells uints = half the fp32 size).
    GLuint make_half_state_buffer(Gl& gl, std::size_t cells) {
        GLuint buf = 0;
        gl.glGenBuffers(1, &buf);
        gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
        gl.glBufferData(GL_SHADER_STORAGE_BUFFER,
                        static_cast<GLsizeiptr>(cells * sizeof(GLuint)), nullptr,
                        GL_STATIC_COPY);
        return buf;
    }

    // dst_half <- packHalf2x16(src_fp32).
    void pack_to_half(Gl& gl, GLuint src_fp32, GLuint dst_half, std::size_t cells) {
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPsi, src_fp32);
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kAux, dst_half);
        gl.glUseProgram(pack_prog_);
        gl.glUniform1ui(gl.glGetUniformLocation(pack_prog_, "n"),
                        static_cast<GLuint>(cells));
        gl.glDispatchCompute(static_cast<GLuint>((cells + 255) / 256), 1, 1);
        gl.glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    // dst_fp32 <- unpackHalf2x16(src_half).
    void unpack_from_half(Gl& gl, GLuint src_half, GLuint dst_fp32, std::size_t cells) {
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kAux, src_half);
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPsi, dst_fp32);
        gl.glUseProgram(unpack_prog_);
        gl.glUniform1ui(gl.glGetUniformLocation(unpack_prog_, "n"),
                        static_cast<GLuint>(cells));
        gl.glDispatchCompute(static_cast<GLuint>((cells + 255) / 256), 1, 1);
        gl.glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    // Upload the radial u_nl(r) table (fp32) that kSynthSrc interpolates.
    void upload_radial(Gl& gl, const std::vector<double>& u) {
        std::vector<float> f(u.size());
        for (std::size_t i = 0; i < u.size(); ++i) {
            f[i] = static_cast<float>(u[i]);
        }
        if (radial_u_buf_ == 0 || radial_u_count_ != u.size()) {
            if (radial_u_buf_ != 0) {
                gl.glDeleteBuffers(1, &radial_u_buf_);
            }
            radial_u_buf_ = make_buffer(gl, f);
            radial_u_count_ = u.size();
        } else {
            gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, radial_u_buf_);
            gl.glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                               static_cast<GLsizeiptr>(f.size() * sizeof(float)), f.data());
        }
    }

    // Synthesize psi = (u/r) Y_lm INTO A NEW normalized state buffer on the GPU
    // -- the atlas builds each eigenstate straight on the GPU, no CPU field.
    // Never touches st.psi_buf_, so it is safe to call mid-simulation. h_radial =
    // rmax/(n_radial+1). *out_peak (if given) receives the normalized peak
    // |psi|^2. The buffer is GL_STATIC_COPY (a write-once eigenstate).
    GLuint synthesize_state(Gl& gl, const GpuState& st, const std::vector<double>& u,
                            int l, int m, double h_radial, double rmax, int n_radial,
                            double* out_peak = nullptr, double* out_norm2 = nullptr) {
        upload_radial(gl, u);
        GLuint buf = 0;
        gl.glGenBuffers(1, &buf);
        gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
        gl.glBufferData(GL_SHADER_STORAGE_BUFFER,
                        static_cast<GLsizeiptr>(2 * st.cells_ * sizeof(float)), nullptr,
                        GL_STATIC_COPY);
        // Synthesize into the new buffer (binding 0), reading the radial table.
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kPsi, buf);
        gl.glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind::kRadial, radial_u_buf_);
        gl.glUseProgram(synth_prog_);
        gl.glUniform1ui(gl.glGetUniformLocation(synth_prog_, "n"),
                        static_cast<GLuint>(st.cells_));
        gl.glUniform1i(gl.glGetUniformLocation(synth_prog_, "nx"), st.grid_.x.n);
        gl.glUniform1i(gl.glGetUniformLocation(synth_prog_, "ny"), st.grid_.y.n);
        gl.glUniform3f(gl.glGetUniformLocation(synth_prog_, "box_min"),
                       static_cast<float>(st.grid_.x.xmin), static_cast<float>(st.grid_.y.xmin),
                       static_cast<float>(st.grid_.z.xmin));
        gl.glUniform3f(gl.glGetUniformLocation(synth_prog_, "cell_h"),
                       static_cast<float>(st.grid_.x.spacing()),
                       static_cast<float>(st.grid_.y.spacing()),
                       static_cast<float>(st.grid_.z.spacing()));
        gl.glUniform1i(gl.glGetUniformLocation(synth_prog_, "l"), l);
        gl.glUniform1i(gl.glGetUniformLocation(synth_prog_, "m"), m);
        gl.glUniform1f(gl.glGetUniformLocation(synth_prog_, "h_radial"),
                       static_cast<float>(h_radial));
        gl.glUniform1f(gl.glGetUniformLocation(synth_prog_, "rmax"),
                       static_cast<float>(rmax));
        gl.glUniform1i(gl.glGetUniformLocation(synth_prog_, "n_radial"), n_radial);
        gl.glDispatchCompute(static_cast<GLuint>((st.cells_ + 255) / 256), 1, 1);
        gl.glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        // Normalize in place: the norm reduction and scale act on binding 0
        // (still `buf`), exactly the core's normalize().
        const NormPeak np = run_norm_peak(gl, st.norm_prog_, st.partials_buf_, st.cells_);
        const double norm_sq = np.sum * st.grid_.cell_volume();  // ||raw (u/r)Ylm||^2_grid
        const double inv = (norm_sq > 0.0) ? 1.0 / std::sqrt(norm_sq) : 0.0;
        run_scale(gl, st.scale_prog_, st.cells_, static_cast<float>(inv));
        if (out_peak != nullptr) {
            *out_peak = (norm_sq > 0.0) ? np.peak / norm_sq : 0.0;
        }
        // The pre-normalization grid norm: the projection normalizes populations
        // by this (|<n|psi>|^2 = |raw dot|^2 / norm2_grid) to stay value-identical
        // to the retired inner_with_psi(grid-normalized orbital) path.
        if (out_norm2 != nullptr) {
            *out_norm2 = norm_sq;
        }
        return buf;
    }

    // Synthesize psi = (u/r) Y_lm straight into a resident fp16 buffer: build +
    // normalize in fp32 (the tested path), then pack to half and free the fp32
    // temp. HALF the resident footprint. *out_peak (if given) is the fp32
    // normalized peak (pre-pack), matching synthesize_state.
    GLuint synthesize_state_half(Gl& gl, const GpuState& st, const std::vector<double>& u,
                                 int l, int m, double h_radial, double rmax, int n_radial,
                                 double* out_peak = nullptr, double* out_norm2 = nullptr) {
        const GLuint fp32 =
            synthesize_state(gl, st, u, l, m, h_radial, rmax, n_radial, out_peak, out_norm2);
        const GLuint half = make_half_state_buffer(gl, st.cells_);
        pack_to_half(gl, fp32, half, st.cells_);
        gl.glDeleteBuffers(1, &fp32);
        return half;
    }
};
