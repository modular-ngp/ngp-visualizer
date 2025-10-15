#version 460 core
#extension GL_GOOGLE_include_directive : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in float inDensity;
layout(location = 2) in vec3 inColor;
layout(location = 3) in float inFeature;
layout(location = 4) in float inRadius;
layout(location = 5) in float inEmissive;

layout(location = 0) out vec3 vColor;
layout(location = 1) out float vDensity;
layout(location = 2) out float vFeature;
layout(location = 3) out float vRadius;
layout(location = 4) out float vEmissive;
layout(location = 5) out vec3 vWorldPos;
layout(location = 6) out float vViewDepth;

layout(set = 0, binding = 0) uniform CameraUniform {
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    vec4 cameraPos;
    vec4 viewport;   // width, height, near, far
    vec4 boundsMin;
    vec4 boundsMax;
    vec4 extras;     // time, dt, pointCount, padding
} uCamera;

layout(push_constant) uniform PushConstants {
    float pointScale;
    float intensityScale;
    float opacityScale;
    float densityMin;
    float densityMax;
    float gamma;
    float splatSharpness;
    float depthFade;
    uint  colorMode;
    uint  animate;
    float hueShift;
    float pad0;
} pc;

void main() {
    vec4 world = vec4(inPosition, 1.0);
    vec4 viewPos = uCamera.view * world;
    gl_Position = uCamera.proj * viewPos;

    float viewportHeight = max(uCamera.viewport.y, 1.0);
    float distance = max(1e-4, -viewPos.z);
    float worldRadius = max(1e-4, inRadius * pc.pointScale);
    float sizePixels = worldRadius * viewportHeight / distance;
    gl_PointSize = clamp(sizePixels, 1.5, 220.0);

    vColor = inColor;
    vDensity = inDensity;
    vFeature = inFeature;
    vRadius = worldRadius;
    vEmissive = inEmissive;
    vWorldPos = inPosition;
    vViewDepth = distance;
}
