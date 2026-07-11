#version 450

// Sample the ses_vk scene image (imported via QRhiTexture::createFrom).
layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;
layout(binding = 0) uniform sampler2D scene;

void main() {
    frag_color = texture(scene, v_uv);
}
