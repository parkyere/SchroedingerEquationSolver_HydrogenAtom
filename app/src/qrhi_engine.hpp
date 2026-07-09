#pragma once

// QRhi (Vulkan-backend) analog of ses_gpu::GpuEngine (gpu_engine.hpp): the
// split-operator Strang step orchestrated on QRhi compute. The kernels are the
// same decorated Vulkan-GLSL that sesolver_qrhicheck bakes; the engine loads the
// baked .qsb from the Qt resource system, so the caller must bake them into its
// target (phase_multiply / conj_scale / fft_line<N>).
//
// Scope of this first cut: a CUBIC grid (nx == ny == nz == N) with a baked
// fft_line<N> shader -- enough to verify the step against SplitOperator3D. The
// psi buffer is a single Static VkBuffer (GPU-written in place, read back), which
// is also what a later VkFFT drop-in needs (one durable handle).

#include <core/complex.hpp>
#include <core/grid.hpp>

#include <rhi/qrhi.h>

#include <QFile>
#include <QString>

#include <cstdio>
#include <vector>

namespace ses_qrhi {

inline std::vector<float> to_rg32f(const std::vector<ses::Complex<double>>& src) {
    std::vector<float> out(2 * src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        out[2 * i] = static_cast<float>(src[i].real());
        out[2 * i + 1] = static_cast<float>(src[i].imag());
    }
    return out;
}

inline QShader load_qsb(const QString& path) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) {
        std::fprintf(stderr, "QrhiEngine: cannot open %s\n", qPrintable(path));
        return {};
    }
    return QShader::fromSerialized(f.readAll());
}

class QrhiEngine {
public:
    // Returns false if a shader is missing or a resource fails to build. half_v
    // / kinetic are SplitOperator3D's phase tables; psi0 the initial field.
    bool initialize(QRhi* rhi, const ses::Grid3D& grid,
                    const std::vector<ses::Complex<double>>& half_v,
                    const std::vector<ses::Complex<double>>& kinetic,
                    const std::vector<ses::Complex<double>>& psi0) {
        rhi_ = rhi;
        n_ = grid.x.n;
        cells_ = static_cast<std::size_t>(grid.size());
        if (grid.y.n != n_ || grid.z.n != n_) {
            std::fprintf(stderr, "QrhiEngine: only cubic grids supported\n");
            return false;
        }

        const QShader mulcs = load_qsb(QStringLiteral(":/shaders/phase_multiply.comp.qsb"));
        const QShader conjcs = load_qsb(QStringLiteral(":/shaders/conj_scale.comp.qsb"));
        const QShader fftcs =
            load_qsb(QStringLiteral(":/shaders/fft_line%1.comp.qsb").arg(n_));
        if (!mulcs.isValid() || !conjcs.isValid() || !fftcs.isValid()) { return false; }

        const std::vector<float> psi_f = to_rg32f(psi0);
        const std::vector<float> half_f = to_rg32f(half_v);
        const std::vector<float> kin_f = to_rg32f(kinetic);
        const quint32 bytes = static_cast<quint32>(psi_f.size() * sizeof(float));

        psi_.reset(rhi_->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, bytes));
        half_.reset(rhi_->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, bytes));
        kin_.reset(rhi_->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, bytes));
        if (!psi_->create() || !half_->create() || !kin_->create()) { return false; }

        // UBO param blocks (std140). Values are constant; re-uploaded each frame
        // (Dynamic uniform buffers are per-frame-slot in QRhi).
        muln_ = MulParams{ static_cast<quint32>(cells_), 0, 0, 0 };
        conj1_ = ConjParams{ static_cast<quint32>(cells_), 1.0f, 0.0f, 0.0f };
        conjN_ = ConjParams{ static_cast<quint32>(cells_),
                             1.0f / static_cast<float>(cells_), 0.0f, 0.0f };
        const int nn = n_ * n_;
        fftp_[0] = FftParams{ nn, n_, 0, 1, nn, 0, 0, 0 };        // x-lines
        fftp_[1] = FftParams{ n_, 1, nn, n_, nn, 0, 0, 0 };       // y-lines
        fftp_[2] = FftParams{ nn, 1, 0, nn, nn, 0, 0, 0 };        // z-lines

        muln_ubo_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer,
                                        sizeof(MulParams)));
        conj1_ubo_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer,
                                         sizeof(ConjParams)));
        conjN_ubo_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer,
                                         sizeof(ConjParams)));
        if (!muln_ubo_->create() || !conj1_ubo_->create() || !conjN_ubo_->create()) {
            return false;
        }
        for (int a = 0; a < 3; ++a) {
            fft_ubo_[a].reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer,
                                              sizeof(FftParams)));
            if (!fft_ubo_[a]->create()) { return false; }
        }

        // SRBs. phase_multiply: psi(0) rw, phase(1) ro, n(2) ubo. conj_scale:
        // psi(0) rw, params(1) ubo. fft_line: psi(0) rw, axis(1) ubo.
        using B = QRhiShaderResourceBinding;
        const auto cs = B::ComputeStage;
        mul_half_srb_.reset(rhi_->newShaderResourceBindings());
        mul_half_srb_->setBindings({ B::bufferLoadStore(0, cs, psi_.data()),
                                     B::bufferLoad(1, cs, half_.data()),
                                     B::uniformBuffer(2, cs, muln_ubo_.data()) });
        mul_kin_srb_.reset(rhi_->newShaderResourceBindings());
        mul_kin_srb_->setBindings({ B::bufferLoadStore(0, cs, psi_.data()),
                                    B::bufferLoad(1, cs, kin_.data()),
                                    B::uniformBuffer(2, cs, muln_ubo_.data()) });
        conj1_srb_.reset(rhi_->newShaderResourceBindings());
        conj1_srb_->setBindings({ B::bufferLoadStore(0, cs, psi_.data()),
                                  B::uniformBuffer(1, cs, conj1_ubo_.data()) });
        conjN_srb_.reset(rhi_->newShaderResourceBindings());
        conjN_srb_->setBindings({ B::bufferLoadStore(0, cs, psi_.data()),
                                  B::uniformBuffer(1, cs, conjN_ubo_.data()) });
        for (int a = 0; a < 3; ++a) {
            fft_srb_[a].reset(rhi_->newShaderResourceBindings());
            fft_srb_[a]->setBindings({ B::bufferLoadStore(0, cs, psi_.data()),
                                       B::uniformBuffer(1, cs, fft_ubo_[a].data()) });
        }
        if (!mul_half_srb_->create() || !mul_kin_srb_->create() || !conj1_srb_->create() ||
            !conjN_srb_->create() || !fft_srb_[0]->create() || !fft_srb_[1]->create() ||
            !fft_srb_[2]->create()) {
            return false;
        }

        mul_pipe_.reset(rhi_->newComputePipeline());
        mul_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, mulcs));
        mul_pipe_->setShaderResourceBindings(mul_half_srb_.data());
        conj_pipe_.reset(rhi_->newComputePipeline());
        conj_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, conjcs));
        conj_pipe_->setShaderResourceBindings(conj1_srb_.data());
        fft_pipe_.reset(rhi_->newComputePipeline());
        fft_pipe_->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, fftcs));
        fft_pipe_->setShaderResourceBindings(fft_srb_[0].data());
        if (!mul_pipe_->create() || !conj_pipe_->create() || !fft_pipe_->create()) {
            return false;
        }

        // Upload the static buffers (psi/half/kin) once.
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return false; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->uploadStaticBuffer(psi_.data(), psi_f.data());
        u->uploadStaticBuffer(half_.data(), half_f.data());
        u->uploadStaticBuffer(kin_.data(), kin_f.data());
        cb->resourceUpdate(u);
        rhi_->endOffscreenFrame();
        return true;
    }

    // psi <- (halfV . IFFT . kin . FFT . halfV)^nsteps psi, the split-operator
    // Strang step -- the same dispatch chain as ses_gpu::GpuEngine::step. All
    // dispatches share the psi buffer, so QRhi inserts the barriers between them.
    void step(int nsteps) {
        const int mul_groups = static_cast<int>((cells_ + 255) / 256);
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return; }
        QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
        u->updateDynamicBuffer(muln_ubo_.data(), 0, sizeof(MulParams), &muln_);
        u->updateDynamicBuffer(conj1_ubo_.data(), 0, sizeof(ConjParams), &conj1_);
        u->updateDynamicBuffer(conjN_ubo_.data(), 0, sizeof(ConjParams), &conjN_);
        for (int a = 0; a < 3; ++a) {
            u->updateDynamicBuffer(fft_ubo_[a].data(), 0, sizeof(FftParams), &fftp_[a]);
        }
        cb->beginComputePass(u);
        for (int s = 0; s < nsteps; ++s) {
            multiply(cb, mul_half_srb_.data(), mul_groups);
            fft3(cb);
            multiply(cb, mul_kin_srb_.data(), mul_groups);
            conjscale(cb, conj1_srb_.data(), mul_groups);
            fft3(cb);
            conjscale(cb, conjN_srb_.data(), mul_groups);
            multiply(cb, mul_half_srb_.data(), mul_groups);
        }
        cb->endComputePass();
        rhi_->endOffscreenFrame();
    }

    // Interleaved RG floats, 2 per cell.
    void readback(std::vector<float>& out) {
        const quint32 bytes = static_cast<quint32>(2 * cells_ * sizeof(float));
        QRhiCommandBuffer* cb = nullptr;
        if (rhi_->beginOffscreenFrame(&cb) != QRhi::FrameOpSuccess) { return; }
        QRhiReadbackResult rb;
        QRhiResourceUpdateBatch* d = rhi_->nextResourceUpdateBatch();
        d->readBackBuffer(psi_.data(), 0, bytes, &rb);
        cb->resourceUpdate(d);
        rhi_->endOffscreenFrame();
        out.assign(reinterpret_cast<const float*>(rb.data.constData()),
                   reinterpret_cast<const float*>(rb.data.constData()) + 2 * cells_);
    }

private:
    struct alignas(16) MulParams { quint32 n, p0, p1, p2; };
    struct alignas(16) ConjParams { quint32 n; float scale; float p0, p1; };
    struct alignas(16) FftParams { qint32 mod_a, mul_b, mul_c, stride, n_lines, p0, p1, p2; };

    void multiply(QRhiCommandBuffer* cb, QRhiShaderResourceBindings* srb, int groups) {
        cb->setComputePipeline(mul_pipe_.data());
        cb->setShaderResources(srb);
        cb->dispatch(groups, 1, 1);
    }
    void conjscale(QRhiCommandBuffer* cb, QRhiShaderResourceBindings* srb, int groups) {
        cb->setComputePipeline(conj_pipe_.data());
        cb->setShaderResources(srb);
        cb->dispatch(groups, 1, 1);
    }
    void fft3(QRhiCommandBuffer* cb) {
        cb->setComputePipeline(fft_pipe_.data());
        for (int a = 0; a < 3; ++a) {
            cb->setShaderResources(fft_srb_[a].data());
            cb->dispatch(n_ * n_, 1, 1);
        }
    }

    QRhi* rhi_ = nullptr;
    int n_ = 0;
    std::size_t cells_ = 0;
    MulParams muln_{};
    ConjParams conj1_{};
    ConjParams conjN_{};
    FftParams fftp_[3]{};

    QScopedPointer<QRhiBuffer> psi_, half_, kin_;
    QScopedPointer<QRhiBuffer> muln_ubo_, conj1_ubo_, conjN_ubo_, fft_ubo_[3];
    QScopedPointer<QRhiShaderResourceBindings> mul_half_srb_, mul_kin_srb_, conj1_srb_,
        conjN_srb_, fft_srb_[3];
    QScopedPointer<QRhiComputePipeline> mul_pipe_, conj_pipe_, fft_pipe_;
};

}  // namespace ses_qrhi
