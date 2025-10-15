#version 460
// Simple fullscreen-ish colored triangle (vertex-only data via gl_VertexIndex)
layout(location = 0) out vec3 vColor;

const vec2 kPositions[3] = vec2[](
vec2(0.0, -0.6),
vec2(0.6, 0.6),
vec2(-0.6, 0.6)
);
const vec3 kColors[3] = vec3[](
vec3(1.0, 0.1, 0.1),
vec3(0.1, 1.0, 0.1),
vec3(0.1, 0.1, 1.0)
);

void main() {
    gl_Position = vec4(kPositions[gl_VertexIndex], 0.0, 1.0);
    vColor = kColors[gl_VertexIndex];
}

