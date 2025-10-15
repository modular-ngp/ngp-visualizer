#include "vk_engine.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <imgui.h>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.h>
#include <vv_camera.h>
#include <shmx_client.h>

#ifndef VK_CHECK
#define VK_CHECK(expr)                                                                                                             \
    do {                                                                                                                           \
        VkResult _vk_result_ = (expr);                                                                                             \
        if (_vk_result_ != VK_SUCCESS) throw std::runtime_error("Vulkan error: " + std::to_string(static_cast<int>(_vk_result_))); \
    } while (false)
#endif

namespace {

    struct GpuBuffer {
        VkBuffer buffer{VK_NULL_HANDLE};
        VmaAllocation allocation{VK_NULL_HANDLE};
        VmaAllocationInfo info{};
        VkDeviceSize size{0};
    };

    struct alignas(16) CameraUniforms {
        std::array<float, 16> view{};
        std::array<float, 16> proj{};
        std::array<float, 16> viewProj{};
        std::array<float, 4> cameraPos{};
        std::array<float, 4> viewport{};
        std::array<float, 4> boundsMin{};
        std::array<float, 4> boundsMax{};
        std::array<float, 4> extras{};
    };

    struct alignas(16) PushConstants {
        float pointScale{1.0f};
        float intensityScale{1.0f};
        float opacityScale{1.0f};
        float densityMin{0.0f};
        float densityMax{1.0f};
        float gamma{1.0f};
        float splatSharpness{2.5f};
        float depthFade{0.18f};
        uint32_t colorMode{0};
        uint32_t animate{1};
        float hueShift{0.0f};
        float pad0{0.0f};
    };

    struct PointVertex {
        float px;
        float py;
        float pz;
        float density;
        float r;
        float g;
        float b;
        float feature;
        float radius;
        float emissive;
    };

    struct UiState {
        float pointScale{1.2f};
        float intensityScale{1.15f};
        float opacityScale{1.0f};
        float gamma{1.0f};
        float splatSharpness{2.6f};
        float depthFade{0.22f};
        float hueShift{0.0f};
        int colorMode{0};
        bool animate{true};
        std::array<float, 2> densityRange{0.05f, 1.0f};
        std::array<float, 3> background{0.04f, 0.045f, 0.07f};
        bool showHistogram{true};
        std::array<char, 64> streamName{};
        bool autoReconnect{true};
        bool autoFrameOnFirstRemote{true};
        bool autoResetDensityOnFirstRemote{true};
        bool requestDensityReset{false};
        int maxPoints{200000};
        bool showTrainingMetrics{true};

        UiState() {
            streamName.fill(0);
            constexpr const char* def = "ngp_debug";
            std::strncpy(streamName.data(), def, streamName.size() - 1);
        }
    };

    struct PointFieldStats {
        float densityMin{std::numeric_limits<float>::max()};
        float densityMax{std::numeric_limits<float>::lowest()};
        std::array<float, 64> histogram{};
    };

    struct StreamingMetrics {
        uint64_t frameSeq{0};
        uint32_t iteration{0};
        double timestamp{0.0};
        float loss{std::numeric_limits<float>::quiet_NaN()};
        float psnr{std::numeric_limits<float>::quiet_NaN()};
        float learningRate{std::numeric_limits<float>::quiet_NaN()};
        vv::float3 cameraPos{};
        vv::float3 cameraTarget{};
        bool hasCameraPos{false};
        bool hasCameraTarget{false};
    };

    struct StreamingState {
        bool enabled{true};
        bool connected{false};
        std::string activeStream{"ngp_debug"};
        std::string lastError{};
        uint32_t staticGen{0};
        uint64_t lastFrameId{std::numeric_limits<uint64_t>::max()};
        std::chrono::steady_clock::time_point lastFrameTime{};
        size_t rawPointCount{0};
        size_t usedPointCount{0};
        bool firstFrameReceived{false};
        StreamingMetrics metrics{};
    };

    namespace ngp_streams {
        inline constexpr uint32_t FrameSeq     = 1;
        inline constexpr uint32_t Timestamp    = 2;
        inline constexpr uint32_t Iteration    = 3;
        inline constexpr uint32_t Positions    = 100;
        inline constexpr uint32_t Colors       = 101;
        inline constexpr uint32_t Normals      = 102;
        inline constexpr uint32_t Density      = 200;
        inline constexpr uint32_t Opacity      = 201;
        inline constexpr uint32_t Features     = 202;
        inline constexpr uint32_t Loss         = 300;
        inline constexpr uint32_t Psnr         = 301;
        inline constexpr uint32_t LearningRate = 302;
        inline constexpr uint32_t CameraPos    = 400;
        inline constexpr uint32_t CameraTarget = 401;
        inline constexpr uint32_t CameraMatrix = 402;
    }

    float saturate(float v) {
        return std::clamp(v, 0.0f, 1.0f);
    }

    std::array<float, 3> lerp_color(const std::array<float, 3>& a, const std::array<float, 3>& b, float t) {
        return {a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t};
    }

    std::array<float, 3> mul_color(const std::array<float, 3>& c, float s) {
        return {c[0] * s, c[1] * s, c[2] * s};
    }

    std::array<float, 3> add_color(const std::array<float, 3>& a, const std::array<float, 3>& b) {
        return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
    }

    std::array<float, 3> clamp_color(const std::array<float, 3>& c) {
        return {saturate(c[0]), saturate(c[1]), saturate(c[2])};
    }

    std::vector<char> load_spirv(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("Failed to open shader: " + path);
        const std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(static_cast<size_t>(size));
        file.read(buffer.data(), size);
        return buffer;
    }

    std::array<float, 3> palette_viridis(float t) {
        static constexpr std::array<std::array<float, 3>, 8> stops{{
            {0.267004f, 0.004874f, 0.329415f},
            {0.282327f, 0.094955f, 0.417331f},
            {0.253935f, 0.265254f, 0.529983f},
            {0.206756f, 0.371758f, 0.553117f},
            {0.163625f, 0.471133f, 0.558148f},
            {0.134692f, 0.658636f, 0.517649f},
            {0.477504f, 0.821444f, 0.318195f},
            {0.993248f, 0.906157f, 0.143936f},
        }};
        t                  = saturate(t);
        const float scaled = t * static_cast<float>(stops.size() - 1);
        const size_t idx   = static_cast<size_t>(std::floor(scaled));
        const size_t next  = std::min(idx + 1, stops.size() - 1);
        const float f      = scaled - static_cast<float>(idx);
        return lerp_color(stops[idx], stops[next], f);
    }

    class PointCloudRenderer final : public IRenderer {
    public:
        PointCloudRenderer()           = default;
        ~PointCloudRenderer() override = default;

        void query_required_device_caps(RendererCaps& caps) override {
            caps.api_version         = VK_API_VERSION_1_3;
            caps.enable_imgui        = true;
            caps.allow_async_compute = false;
        }

        void get_capabilities(const EngineContext&, RendererCaps& caps) override {
            caps.presentation_mode = PresentationMode::EngineBlit;
            caps.color_attachments = {AttachmentRequest{.name = "dense_field_color", .format = VK_FORMAT_R16G16B16A16_SFLOAT, .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, .samples = VK_SAMPLE_COUNT_1_BIT, .aspect = VK_IMAGE_ASPECT_COLOR_BIT, .initial_layout = VK_IMAGE_LAYOUT_GENERAL}};
            caps.presentation_attachment = "dense_field_color";
            caps.depth_attachment        = AttachmentRequest{.name = "dense_field_depth", .format = VK_FORMAT_D32_SFLOAT, .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, .samples = VK_SAMPLE_COUNT_1_BIT, .aspect = VK_IMAGE_ASPECT_DEPTH_BIT, .initial_layout = VK_IMAGE_LAYOUT_UNDEFINED};
            caps.uses_depth              = VK_TRUE;
            caps.frames_in_flight        = 2;
        }

        void initialize(const EngineContext& eng, const RendererCaps& caps, const FrameContext&) override {
            dev_                 = eng.device;
            allocator_           = eng.allocator;
            descriptorAllocator_ = eng.descriptorAllocator;
            services_            = eng.services;
            colorFormat_         = caps.color_attachments.front().format;
            depthFormat_         = caps.depth_attachment ? caps.depth_attachment->format : VK_FORMAT_UNDEFINED;

            build_point_field();
            create_uniform_buffer();
            create_point_buffer();
            create_descriptor_layout();
            allocate_descriptor_set();
            create_pipeline_layout();
            create_pipeline();

            streaming_             = StreamingState{};
            streaming_.activeStream = sanitize_stream_name(ui_.streamName);
            streaming_.enabled      = true;
            streaming_.lastFrameId  = std::numeric_limits<uint64_t>::max();
            lastConnectAttempt_      = std::chrono::steady_clock::time_point{};
            if (ui_.autoReconnect) {
                try_connect(true);
            }

            camera_.set_scene_bounds(sceneBounds_);
            camera_.frame_scene(1.08f);
        }

        void destroy(const EngineContext& eng, const RendererCaps&) override {
            if (eng.device != VK_NULL_HANDLE) {
                vkDeviceWaitIdle(eng.device);
            }
            disconnect();
            destroy_pipeline();
            destroy_descriptor_layout();
            destroy_buffer(pointBuffer_);
            destroy_buffer(uniformBuffer_);
            descriptorAllocator_ = nullptr;
            allocator_           = nullptr;
            services_            = nullptr;
            dev_                 = VK_NULL_HANDLE;
        }

        void on_event(const SDL_Event& e, const EngineContext& eng, const FrameContext* frm) override {
            camera_.handle_event(e, &eng, frm);
        }

        void simulate(const EngineContext&, const FrameContext& frm) override {
            camera_.update(frm.dt_sec, static_cast<int>(frm.extent.width), static_cast<int>(frm.extent.height));
        }

        void update(const EngineContext& eng, const FrameContext&) override {
            poll_stream_data(eng);
        }

        void record_graphics(VkCommandBuffer cmd, const EngineContext& eng, const FrameContext& frm) override {
            if (pointBuffer_.buffer == VK_NULL_HANDLE || pipeline_ == VK_NULL_HANDLE || frm.color_attachments.empty()) return;

            const auto frameStart = std::chrono::high_resolution_clock::now();
            update_uniforms(frm);
            if (services_ == nullptr && eng.services != nullptr) {
                services_ = eng.services;
            }

            const auto& colorTarget = frm.color_attachments.front();
            transition_image(cmd, colorTarget, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

            const AttachmentView* depthTarget = frm.depth_attachment;
            if (depthTarget) {
                transition_image(cmd, *depthTarget, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
            }

            VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            colorAttachment.imageView   = colorTarget.view;
            colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            colorAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachment.clearValue  = {.color = {{ui_.background[0], ui_.background[1], ui_.background[2], 1.0f}}};

            VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            if (depthTarget) {
                depthAttachment.imageView   = depthTarget->view;
                depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
                depthAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
                depthAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
                depthAttachment.clearValue  = {.depthStencil = {1.0f, 0}};
            }

            VkRenderingInfo renderingInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
            renderingInfo.renderArea           = {{0, 0}, frm.extent};
            renderingInfo.layerCount           = 1;
            renderingInfo.colorAttachmentCount = 1;
            renderingInfo.pColorAttachments    = &colorAttachment;
            if (depthTarget) renderingInfo.pDepthAttachment = &depthAttachment;

            vkCmdBeginRendering(cmd, &renderingInfo);
            record_point_draw(cmd, frm);
            vkCmdEndRendering(cmd);

            transition_image(cmd, colorTarget, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
            if (depthTarget) {
                transition_image(cmd, *depthTarget, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
            }

            const auto frameEnd = std::chrono::high_resolution_clock::now();
            stats_.cpu_ms       = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();
            stats_.draw_calls   = pointCount_ ? 1 : 0;
            stats_.triangles    = 0;
        }

        void on_imgui(const EngineContext& eng, const FrameContext& frm) override {
            if (services_ == nullptr && eng.services != nullptr) {
                services_ = eng.services;
            }
            if (auto* tabs = static_cast<vv_ui::TabsHost*>(services_)) {
                tabs->set_main_window_title("Instant-NGP Dense Field Debugger");
                FrameContext frame_copy = frm;
                tabs->add_tab("Field Controls", [this, frame_copy] { draw_field_controls_panel(frame_copy); });
                tabs->add_tab("Camera", [this] { camera_.imgui_panel_contents(); });
                tabs->add_overlay([this] { camera_.imgui_draw_mini_axis_gizmo(); });
                tabs->add_overlay([this] { camera_.imgui_draw_nav_overlay_space_tint(0x20141432); });
            } else {
                if (ImGui::Begin("Field Controls")) {
                    draw_field_controls_panel(frm);
                }
                ImGui::End();
                if (ImGui::Begin("Camera")) {
                    camera_.imgui_panel_contents();
                }
                ImGui::End();
            }
        }

        [[nodiscard]] RendererStats get_stats() const override {
            return stats_;
        }

    private:
        void draw_field_controls_panel(const FrameContext& frm) {
            ImGui::SeparatorText("Streaming");
            bool edited = ImGui::InputText("Stream", ui_.streamName.data(), ui_.streamName.size());
            if (edited && !ui_.autoReconnect) {
                streaming_.activeStream = sanitize_stream_name(ui_.streamName);
            }
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                std::string desired = sanitize_stream_name(ui_.streamName);
                if (streaming_.connected && desired != streaming_.activeStream) {
                    disconnect();
                }
                if (ui_.autoReconnect) {
                    streaming_.activeStream = desired;
                    try_connect(true);
                }
            }
            ImGui::SameLine();
            if (ImGui::Button(streaming_.connected ? "Disconnect" : "Connect")) {
                if (streaming_.connected) {
                    disconnect();
                } else {
                    try_connect(true);
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Apply##stream")) {
                disconnect();
                try_connect(true);
            }

            ImGui::Checkbox("Auto Reconnect", &ui_.autoReconnect);
            ImGui::SameLine();
            ImGui::Checkbox("Auto Frame First Remote", &ui_.autoFrameOnFirstRemote);
            ImGui::SameLine();
            ImGui::Checkbox("Auto Reset Density", &ui_.autoResetDensityOnFirstRemote);

            if (ImGui::Button("Reset Density Window")) {
                ui_.requestDensityReset = true;
                reset_density_window();
            }
            ImGui::SameLine();
            if (ImGui::Button("Frame Scene##manual")) {
                camera_.frame_scene(1.05f);
            }

            const auto now          = std::chrono::steady_clock::now();
            const bool hasFrameTime = streaming_.lastFrameTime != std::chrono::steady_clock::time_point{};
            const double ageSeconds = streaming_.connected && hasFrameTime ? std::chrono::duration<double>(now - streaming_.lastFrameTime).count() : std::numeric_limits<double>::infinity();
            const bool stale        = streaming_.connected && ageSeconds > 1.0;
            ImVec4 statusColor      = streaming_.connected ? (stale ? ImVec4(0.95f, 0.75f, 0.2f, 1.0f) : ImVec4(0.45f, 0.85f, 0.45f, 1.0f)) : ImVec4(0.9f, 0.45f, 0.45f, 1.0f);
            const char* statusText  = streaming_.connected ? (stale ? "Connected (stale)" : "Connected") : "Disconnected";
            ImGui::TextColored(statusColor, "%s", statusText);
            if (streaming_.connected) {
                ImGui::SameLine();
                ImGui::TextDisabled("stream: %s", streaming_.activeStream.c_str());
            }
            if (!streaming_.lastError.empty()) {
                ImGui::TextColored(ImVec4(0.95f, 0.55f, 0.35f, 1.0f), "Last error: %s", streaming_.lastError.c_str());
            }

            if (streaming_.connected) {
                ImGui::Text("Frame seq: %llu | Frame ID: %llu | Iteration: %u",
                            static_cast<unsigned long long>(streaming_.metrics.frameSeq),
                            static_cast<unsigned long long>(streaming_.lastFrameId == std::numeric_limits<uint64_t>::max() ? 0ull : streaming_.lastFrameId),
                            streaming_.metrics.iteration);
                if (std::isfinite(ageSeconds)) {
                    ImGui::Text("Last update: %.2f s ago", ageSeconds);
                }
            ImGui::Text("Points: %zu raw -> %zu displayed (limit %d)",
                            streaming_.rawPointCount,
                            streaming_.usedPointCount,
                            ui_.maxPoints);
                ImGui::Text("Timestamp: %.3f", streaming_.metrics.timestamp);
            }

            ImGui::Checkbox("Show Training Metrics", &ui_.showTrainingMetrics);
            if (ui_.showTrainingMetrics && streaming_.connected) {
                ImGui::SeparatorText("Training Metrics");
                if (!std::isnan(streaming_.metrics.loss)) {
                    ImGui::Text("Loss: %.5f", streaming_.metrics.loss);
                }
                if (!std::isnan(streaming_.metrics.psnr)) {
                    ImGui::Text("PSNR: %.2f dB", streaming_.metrics.psnr);
                }
                if (!std::isnan(streaming_.metrics.learningRate)) {
                    ImGui::Text("Learning rate: %.6f", streaming_.metrics.learningRate);
                }
                if (streaming_.metrics.hasCameraPos) {
                    ImGui::Text("Camera pos: (%.2f, %.2f, %.2f)",
                                streaming_.metrics.cameraPos.x,
                                streaming_.metrics.cameraPos.y,
                                streaming_.metrics.cameraPos.z);
                }
                if (streaming_.metrics.hasCameraTarget) {
                    ImGui::Text("Camera target: (%.2f, %.2f, %.2f)",
                                streaming_.metrics.cameraTarget.x,
                                streaming_.metrics.cameraTarget.y,
                                streaming_.metrics.cameraTarget.z);
                }
            }

            ImGui::SeparatorText("Sampling & Appearance");
            if (ui_.maxPoints < 1000) ui_.maxPoints = 1000;
            ImGui::SliderInt("Max Points", &ui_.maxPoints, 1000, 1000000, "%d", ImGuiSliderFlags_Logarithmic);

            const float totalPoints = static_cast<float>(pointCount_);
            ImGui::Text("Samples: %.0f", totalPoints);
            ImGui::SameLine();
            ImGui::TextDisabled("CPU %.2f ms", stats_.cpu_ms);

            ImGui::SliderFloat("Point Scale", &ui_.pointScale, 0.2f, 6.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("Intensity", &ui_.intensityScale, 0.2f, 3.5f, "%.2f");
            ImGui::SliderFloat("Opacity", &ui_.opacityScale, 0.05f, 3.0f, "%.2f");
            ImGui::SliderFloat("Splat Sharpness", &ui_.splatSharpness, 1.0f, 6.0f, "%.2f");
            ImGui::SliderFloat("Depth Fade", &ui_.depthFade, 0.0f, 1.2f, "%.2f");
            ImGui::SliderFloat("Gamma", &ui_.gamma, 0.6f, 2.2f, "%.2f");
            ImGui::SliderFloat("Hue Shift", &ui_.hueShift, -0.5f, 0.5f, "%.2f");

            ImGui::SeparatorText("Density Window");
            ImGui::DragFloatRange2("##density_range", ui_.densityRange.data(), ui_.densityRange.data() + 1, 0.005f, statsInfo_.densityMin, statsInfo_.densityMax, "Min %.3f", "Max %.3f");
            if (ui_.densityRange[0] > ui_.densityRange[1]) {
                std::swap(ui_.densityRange[0], ui_.densityRange[1]);
            }

            static const char* kColorModes[] = {"Neural RGB", "Density Viridis", "Height Bands", "Feature Divergence"};
            ImGui::Combo("Color Mode", &ui_.colorMode, kColorModes, IM_ARRAYSIZE(kColorModes));
            ImGui::Checkbox("Animate Tint", &ui_.animate);
            ImGui::ColorEdit3("Background", ui_.background.data(), ImGuiColorEditFlags_NoInputs);

            ImGui::Spacing();
            ImGui::Checkbox("Show Histogram", &ui_.showHistogram);
            if (ui_.showHistogram && pointCount_ > 0) {
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.45f, 0.75f, 1.0f, 1.0f));
                ImGui::PlotHistogram("##density_histogram", statsInfo_.histogram.data(), static_cast<int>(statsInfo_.histogram.size()), 0, nullptr, 0.0f, 1.0f, ImVec2(0, 80));
                ImGui::PopStyleColor();
            }

            ImGui::SeparatorText("Frame Stats");
            const auto& camState = camera_.state();
            ImGui::Text("Camera yaw/pitch: %.1f deg / %.1f deg", camState.yaw_deg, camState.pitch_deg);
            ImGui::Text("Distance: %.2f", camState.distance);
            ImGui::Text("Frame dt: %.3f s (%.2f ms)", frm.dt_sec, frm.dt_sec * 1000.0f);
        }
        void build_point_field() {
            pointsCpu_.clear();
            statsInfo_   = PointFieldStats{};
            sceneBounds_ = vv::BoundingBox{};

            constexpr int gridRes    = 80;
            constexpr float halfSize = 1.0f;
            const float step         = (halfSize * 2.0f) / static_cast<float>(gridRes - 1);
            std::mt19937 rng(42);
            std::uniform_real_distribution<float> jitter(-0.5f, 0.5f);

            auto gaussian = [](const vv::float3& p, const vv::float3& c, float r, float gain) {
                const vv::float3 d = p - c;
                const float dist2  = vv::dot(d, d);
                return gain * std::exp(-dist2 / (r * r));
            };

            auto radial_swirl = [](const vv::float3& p) {
                const float radial = std::sqrt(p.x * p.x + p.z * p.z);
                const float swirl  = std::sin(8.0f * p.y + 5.0f * radial);
                return std::exp(-radial * 2.6f) * (0.5f + 0.5f * swirl);
            };

            auto filament = [](const vv::float3& p, const vv::float3& axis, float falloff) {
                vv::float3 diff = p - axis;
                diff.y *= 0.6f;
                const float dist = vv::length(diff);
                const float osc  = std::sin(12.0f * p.x + 9.0f * p.z);
                return std::exp(-dist * falloff) * (0.6f + 0.4f * osc);
            };

            auto update_bounds = [&](const vv::float3& p) {
                if (!sceneBounds_.valid) {
                    sceneBounds_.min = sceneBounds_.max = p;
                    sceneBounds_.valid                  = true;
                } else {
                    sceneBounds_.min.x = std::min(sceneBounds_.min.x, p.x);
                    sceneBounds_.min.y = std::min(sceneBounds_.min.y, p.y);
                    sceneBounds_.min.z = std::min(sceneBounds_.min.z, p.z);
                    sceneBounds_.max.x = std::max(sceneBounds_.max.x, p.x);
                    sceneBounds_.max.y = std::max(sceneBounds_.max.y, p.y);
                    sceneBounds_.max.z = std::max(sceneBounds_.max.z, p.z);
                }
            };

            float densityMin = std::numeric_limits<float>::max();
            float densityMax = std::numeric_limits<float>::lowest();

            pointsCpu_.reserve(static_cast<size_t>(gridRes) * gridRes * gridRes / 2);

            for (int iz = 0; iz < gridRes; ++iz) {
                for (int iy = 0; iy < gridRes; ++iy) {
                    for (int ix = 0; ix < gridRes; ++ix) {
                        vv::float3 p{-halfSize + ix * step, -halfSize + iy * step, -halfSize + iz * step};

                        float density = 0.0f;
                        density += gaussian(p, vv::make_float3(0.05f, 0.0f, 0.0f), 0.68f, 1.0f);
                        density += gaussian(p, vv::make_float3(0.35f, 0.18f, -0.22f), 0.45f, 0.82f);
                        density += gaussian(p, vv::make_float3(-0.3f, -0.24f, 0.35f), 0.35f, 0.74f);
                        const float swirl     = radial_swirl(p);
                        const float filamentA = filament(p, vv::make_float3(-0.45f, -0.2f, 0.0f), 4.5f);
                        const float filamentB = filament(p, vv::make_float3(0.42f, 0.3f, -0.25f), 5.8f);
                        density += 0.46f * swirl + 0.33f * filamentA + 0.28f * filamentB;
                        density = std::max(0.0f, density - 0.04f);

                        if (density < 0.05f) continue;

                        const float jitterScale = step * 0.35f;
                        p.x += jitter(rng) * jitterScale;
                        p.y += jitter(rng) * jitterScale;
                        p.z += jitter(rng) * jitterScale;

                        const float densityNorm = saturate((density - 0.05f) / 0.95f);
                        const float height      = saturate((p.y + halfSize) / (halfSize * 2.0f));
                        const float radial      = std::sqrt(p.x * p.x + p.z * p.z);
                        const float feature     = saturate(swirl * 0.9f + filamentA * 0.6f + filamentB * 0.45f);
                        const float emissive    = saturate(0.35f * swirl + 0.55f * filamentA + 0.25f * filamentB);

                        auto base   = palette_viridis(densityNorm);
                        auto warm   = std::array<float, 3>{0.95f, 0.58f, 0.25f};
                        auto cool   = std::array<float, 3>{0.18f, 0.36f, 0.72f};
                        auto accent = std::array<float, 3>{0.9f, 0.9f, 1.2f};

                        auto color = add_color(mul_color(base, 0.55f + 0.45f * height), mul_color(warm, 0.25f * densityNorm + 0.5f * emissive));
                        color      = add_color(color, mul_color(cool, 0.18f * (1.0f - height) * (0.5f + 0.5f * feature)));
                        color      = add_color(color, mul_color(accent, 0.12f * emissive));
                        color      = clamp_color(color);

                        PointVertex v{};
                        v.px       = p.x;
                        v.py       = p.y;
                        v.pz       = p.z;
                        v.density  = density;
                        v.r        = color[0];
                        v.g        = color[1];
                        v.b        = color[2];
                        v.feature  = feature;
                        v.radius   = 0.0055f + 0.022f * densityNorm * (0.45f + 0.55f * (1.0f - saturate(radial)));
                        v.emissive = emissive;

                        pointsCpu_.push_back(v);
                        update_bounds(p);

                        densityMin = std::min(densityMin, density);
                        densityMax = std::max(densityMax, density);
                    }
                }
            }

            const std::array<vv::float3, 4> guideLights = {vv::make_float3(-0.55f, 0.62f, 0.25f), vv::make_float3(0.48f, -0.52f, -0.4f), vv::make_float3(0.12f, 0.75f, -0.6f), vv::make_float3(-0.2f, -0.7f, 0.15f)};
            for (const auto& pos : guideLights) {
                PointVertex v{};
                v.px       = pos.x;
                v.py       = pos.y;
                v.pz       = pos.z;
                v.density  = densityMax * 0.9f;
                v.r        = 1.6f;
                v.g        = 1.2f;
                v.b        = 0.6f;
                v.feature  = 1.0f;
                v.radius   = 0.03f;
                v.emissive = 1.0f;
                pointsCpu_.push_back(v);
                update_bounds(pos);
            }

            pointCount_                 = static_cast<uint32_t>(pointsCpu_.size());
            streaming_.rawPointCount  = pointsCpu_.size();
            streaming_.usedPointCount = pointsCpu_.size();

            rebuild_stats_from_points(densityMin, densityMax, true);
            if (ui_.autoResetDensityOnFirstRemote || ui_.requestDensityReset) {
                reset_density_window();
                ui_.requestDensityReset = false;
            }
        }

        void create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaAllocationCreateFlags flags, GpuBuffer& out) {
            VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            info.size        = size;
            info.usage       = usage;
            info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo allocInfo{};
            allocInfo.flags = flags;
            allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

            out.size = size;
            VK_CHECK(vmaCreateBuffer(allocator_, &info, &allocInfo, &out.buffer, &out.allocation, &out.info));
        }

        void create_point_buffer() {
            if (pointsCpu_.empty()) return;
            const VkDeviceSize size = sizeof(PointVertex) * pointsCpu_.size();
            create_buffer(size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, pointBuffer_);
            std::memcpy(pointBuffer_.info.pMappedData, pointsCpu_.data(), static_cast<size_t>(size));
            VK_CHECK(vmaFlushAllocation(allocator_, pointBuffer_.allocation, 0, size));
        }

        void create_uniform_buffer() {
            create_buffer(sizeof(CameraUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, uniformBuffer_);
            std::memset(uniformBuffer_.info.pMappedData, 0, sizeof(CameraUniforms));
        }

        void destroy_buffer(GpuBuffer& buffer) {
            if (buffer.buffer != VK_NULL_HANDLE) {
                vmaUnmapMemory(allocator_, buffer.allocation);
                vmaDestroyBuffer(allocator_, buffer.buffer, buffer.allocation);
            }
            buffer = {};
        }

        VkShaderModule load_shader_module(const std::string& path) {
            const auto spirv = load_spirv(path);
            VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            info.codeSize = spirv.size();
            info.pCode    = reinterpret_cast<const uint32_t*>(spirv.data());
            VkShaderModule module{};
            VK_CHECK(vkCreateShaderModule(dev_, &info, nullptr, &module));
            return module;
        }

        void create_descriptor_layout() {
            VkDescriptorSetLayoutBinding binding{};
            binding.binding         = 0;
            binding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            binding.descriptorCount = 1;
            binding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

            VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            info.bindingCount = 1;
            info.pBindings    = &binding;
            VK_CHECK(vkCreateDescriptorSetLayout(dev_, &info, nullptr, &descriptorLayout_));
        }

        void allocate_descriptor_set() {
            descriptorSet_ = descriptorAllocator_->allocate(dev_, descriptorLayout_);
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffer_.buffer;
            bufferInfo.offset = 0;
            bufferInfo.range  = sizeof(CameraUniforms);

            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet          = descriptorSet_;
            write.dstBinding      = 0;
            write.descriptorCount = 1;
            write.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            write.pBufferInfo     = &bufferInfo;
            vkUpdateDescriptorSets(dev_, 1, &write, 0, nullptr);
        }

        void destroy_descriptor_layout() {
            if (descriptorLayout_ != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(dev_, descriptorLayout_, nullptr);
                descriptorLayout_ = VK_NULL_HANDLE;
            }
        }

        void create_pipeline_layout() {
            VkPushConstantRange range{};
            range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            range.offset     = 0;
            range.size       = sizeof(PushConstants);

            VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            info.setLayoutCount         = 1;
            info.pSetLayouts            = &descriptorLayout_;
            info.pushConstantRangeCount = 1;
            info.pPushConstantRanges    = &range;
            VK_CHECK(vkCreatePipelineLayout(dev_, &info, nullptr, &pipelineLayout_));
        }

        void create_pipeline() {
            VkShaderModule vs = load_shader_module("shaders/point_cloud.vert.spv");
            VkShaderModule fs = load_shader_module("shaders/point_cloud.frag.spv");

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vs;
            stages[0].pName  = "main";
            stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = fs;
            stages[1].pName  = "main";

            VkVertexInputBindingDescription binding{};
            binding.binding   = 0;
            binding.stride    = sizeof(PointVertex);
            binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

            std::array<VkVertexInputAttributeDescription, 6> attrs{};
            attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(PointVertex, px)};
            attrs[1] = {1, 0, VK_FORMAT_R32_SFLOAT, offsetof(PointVertex, density)};
            attrs[2] = {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(PointVertex, r)};
            attrs[3] = {3, 0, VK_FORMAT_R32_SFLOAT, offsetof(PointVertex, feature)};
            attrs[4] = {4, 0, VK_FORMAT_R32_SFLOAT, offsetof(PointVertex, radius)};
            attrs[5] = {5, 0, VK_FORMAT_R32_SFLOAT, offsetof(PointVertex, emissive)};

            VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            vertexInput.vertexBindingDescriptionCount   = 1;
            vertexInput.pVertexBindingDescriptions      = &binding;
            vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
            vertexInput.pVertexAttributeDescriptions    = attrs.data();

            VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

            VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            viewport.viewportCount = 1;
            viewport.scissorCount  = 1;

            VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            raster.cullMode    = VK_CULL_MODE_NONE;
            raster.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            raster.lineWidth   = 1.0f;
            raster.polygonMode = VK_POLYGON_MODE_FILL;

            VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            depth.depthTestEnable   = VK_TRUE;
            depth.depthWriteEnable  = VK_FALSE;
            depth.depthCompareOp    = VK_COMPARE_OP_LESS_OR_EQUAL;
            depth.stencilTestEnable = VK_FALSE;

            VkPipelineColorBlendAttachmentState blendAttachment{};
            blendAttachment.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            blendAttachment.blendEnable         = VK_TRUE;
            blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blendAttachment.colorBlendOp        = VK_BLEND_OP_ADD;
            blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            blendAttachment.alphaBlendOp        = VK_BLEND_OP_ADD;

            VkPipelineColorBlendStateCreateInfo colorBlend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            colorBlend.attachmentCount = 1;
            colorBlend.pAttachments    = &blendAttachment;

            std::array<VkDynamicState, 2> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dynamic.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
            dynamic.pDynamicStates    = dynamicStates.data();

            VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
            rendering.colorAttachmentCount    = 1;
            rendering.pColorAttachmentFormats = &colorFormat_;
            if (depthFormat_ != VK_FORMAT_UNDEFINED) {
                rendering.depthAttachmentFormat = depthFormat_;
            }

            VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            pipelineInfo.stageCount          = 2;
            pipelineInfo.pStages             = stages;
            pipelineInfo.pVertexInputState   = &vertexInput;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pViewportState      = &viewport;
            pipelineInfo.pRasterizationState = &raster;
            pipelineInfo.pMultisampleState   = &multisample;
            pipelineInfo.pDepthStencilState  = &depth;
            pipelineInfo.pColorBlendState    = &colorBlend;
            pipelineInfo.pDynamicState       = &dynamic;
            pipelineInfo.layout              = pipelineLayout_;
            pipelineInfo.pNext               = &rendering;

            const VkResult res = vkCreateGraphicsPipelines(dev_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_);
            vkDestroyShaderModule(dev_, vs, nullptr);
            vkDestroyShaderModule(dev_, fs, nullptr);
            VK_CHECK(res);
        }

        void destroy_pipeline() {
            if (pipeline_ != VK_NULL_HANDLE) {
                vkDestroyPipeline(dev_, pipeline_, nullptr);
                pipeline_ = VK_NULL_HANDLE;
            }
            if (pipelineLayout_ != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(dev_, pipelineLayout_, nullptr);
                pipelineLayout_ = VK_NULL_HANDLE;
            }
        }

        void update_uniforms(const FrameContext& frm) {
            CameraUniforms uniforms{};
            const auto view     = camera_.view_matrix();
            const auto proj     = camera_.proj_matrix();
            const auto viewProj = vv::mul(proj, view);
            uniforms.view       = view.m;
            uniforms.proj       = proj.m;
            uniforms.viewProj   = viewProj.m;

            const auto eye     = camera_.eye_position();
            const auto& state  = camera_.state();
            uniforms.cameraPos = {eye.x, eye.y, eye.z, 1.0f};
            uniforms.viewport  = {static_cast<float>(frm.extent.width), static_cast<float>(frm.extent.height), state.znear, state.zfar};
            uniforms.boundsMin = {sceneBounds_.min.x, sceneBounds_.min.y, sceneBounds_.min.z, 0.0f};
            uniforms.boundsMax = {sceneBounds_.max.x, sceneBounds_.max.y, sceneBounds_.max.z, 0.0f};
            uniforms.extras    = {static_cast<float>(frm.time_sec), static_cast<float>(frm.dt_sec), static_cast<float>(pointCount_), 0.0f};

            std::memcpy(uniformBuffer_.info.pMappedData, &uniforms, sizeof(CameraUniforms));
            VK_CHECK(vmaFlushAllocation(allocator_, uniformBuffer_.allocation, 0, sizeof(CameraUniforms)));
        }

        void record_point_draw(VkCommandBuffer cmd, const FrameContext& frm) {
            VkViewport viewport{};
            viewport.width    = static_cast<float>(frm.extent.width);
            viewport.height   = static_cast<float>(frm.extent.height);
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(cmd, 0, 1, &viewport);

            VkRect2D scissor{{0, 0}, frm.extent};
            vkCmdSetScissor(cmd, 0, 1, &scissor);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &pointBuffer_.buffer, &offset);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_, 0, 1, &descriptorSet_, 0, nullptr);

            PushConstants push{};
            push.pointScale     = ui_.pointScale;
            push.intensityScale = ui_.intensityScale;
            push.opacityScale   = ui_.opacityScale;
            push.densityMin     = ui_.densityRange[0];
            push.densityMax     = ui_.densityRange[1];
            push.gamma          = ui_.gamma;
            push.splatSharpness = ui_.splatSharpness;
            push.depthFade      = ui_.depthFade;
            push.colorMode      = static_cast<uint32_t>(ui_.colorMode);
            push.animate        = ui_.animate ? 1U : 0U;
            push.hueShift       = ui_.hueShift;

            vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &push);
            vkCmdDraw(cmd, pointCount_, 1, 0, 0);

            stats_.draw_calls = pointCount_ ? 1 : 0;
        }

        static std::string sanitize_stream_name(const std::array<char, 64>& buffer) {
            std::string s(buffer.data());
            const auto first = s.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) return {};
            const auto last = s.find_last_not_of(" \t\r\n");
            return s.substr(first, last - first + 1);
        }

        void reset_density_window() {
            float min = statsInfo_.densityMin;
            float max = statsInfo_.densityMax;
            if (!(min <= max)) {
                min = 0.0f;
                max = 1.0f;
            }
            ui_.densityRange[0] = min;
            ui_.densityRange[1] = max;
        }

        void poll_stream_data(const EngineContext& eng) {
            if (!streaming_.enabled) return;

            if (services_ == nullptr && eng.services != nullptr) {
                services_ = eng.services;
            }

            if (!streaming_.connected) {
                if (ui_.autoReconnect) {
                    try_connect(false);
                }
                return;
            }

            const auto* header = shmxClient_.header();
            if (!is_header_valid(header)) {
                streaming_.lastError = "Shared memory header invalid";
                disconnect();
                return;
            }

            const uint32_t staticGen = header->static_gen.load(std::memory_order_acquire);
            if (staticGen != streaming_.staticGen) {
                shmx::StaticState state;
                if (shmxClient_.refresh_static(state)) {
                    shmxStatic_          = std::move(state);
                    streaming_.staticGen = shmxStatic_.static_gen;
                }
            }

            shmx::FrameView fv;
            if (!shmxClient_.latest(fv) || fv.fh == nullptr || fv.checksum_mismatch != 0u) {
                return;
            }

            const uint64_t frameId = fv.fh->frame_id.load(std::memory_order_acquire);
            if (frameId == streaming_.lastFrameId) {
                return;
            }

            shmx::DecodedFrame decoded;
            if (!shmx::Client::decode(fv, decoded)) {
                streaming_.lastError = "Failed to decode frame";
                return;
            }

            if (update_point_cloud_from_frame(fv, decoded)) {
                streaming_.lastFrameId   = frameId;
                streaming_.lastFrameTime = std::chrono::steady_clock::now();
                streaming_.lastError.clear();
            }
        }

        void try_connect(bool force) {
            const auto now = std::chrono::steady_clock::now();
            if (!force && (now - lastConnectAttempt_) < std::chrono::milliseconds(500)) {
                return;
            }
            lastConnectAttempt_ = now;

            std::string desired = sanitize_stream_name(ui_.streamName);
            if (desired.empty()) {
                streaming_.lastError = "Stream name is empty";
                return;
            }

            if (streaming_.connected && desired == streaming_.activeStream) {
                return;
            }

            shmxClient_.close();
            streaming_             = StreamingState{};
            streaming_.activeStream = desired;
            streaming_.enabled      = true;

            if (!shmxClient_.open(desired)) {
                streaming_.lastError = "Failed to open stream '" + desired + "'";
                streaming_.connected = false;
                return;
            }

            streaming_.connected    = true;
            streaming_.lastFrameId  = std::numeric_limits<uint64_t>::max();
            streaming_.lastError.clear();
            ui_.requestDensityReset = ui_.autoResetDensityOnFirstRemote;

            shmx::StaticState state;
            if (shmxClient_.refresh_static(state)) {
                shmxStatic_          = std::move(state);
                streaming_.staticGen = shmxStatic_.static_gen;
            }
        }

        void disconnect() {
            shmxClient_.close();
            streaming_             = StreamingState{};
            streaming_.activeStream = sanitize_stream_name(ui_.streamName);
            streaming_.enabled      = true;
            streaming_.lastError.clear();
            streaming_.lastFrameId  = std::numeric_limits<uint64_t>::max();
            streaming_.rawPointCount = 0;
            streaming_.usedPointCount = 0;
        }

        static bool is_header_valid(const shmx::GlobalHeader* GH) {
            if (!GH) return false;
            if (GH->magic != shmx::MAGIC || GH->ver_major != shmx::VER_MAJOR || GH->ver_minor != shmx::VER_MINOR || GH->endianness != shmx::ENDIAN_TAG) return false;
            if (GH->slot_stride == 0u || GH->slots_offset == 0u || GH->frame_bytes_cap == 0u) return false;
            if (GH->reader_slot_stride == 0u || GH->readers_offset == 0u) return false;
            if (GH->control_stride != 0u && (GH->control_offset == 0u || GH->control_per_reader == 0u)) return false;
            return true;
        }

        const shmx::DecodedItem* find_stream(const shmx::DecodedFrame& frame, uint32_t id) const {
            for (const auto& entry : frame.streams) {
                if (entry.first == id) return &entry.second;
            }
            return nullptr;
        }

        template <typename T>
        static std::optional<T> read_scalar(const shmx::DecodedItem* item) {
            if (!item || item->bytes < sizeof(T) || item->elem_count == 0) return std::nullopt;
            T value{};
            std::memcpy(&value, item->ptr, sizeof(T));
            return value;
        }

        bool update_point_cloud_from_frame(const shmx::FrameView& fv, const shmx::DecodedFrame& df) {
            const auto* positionsItem = find_stream(df, ngp_streams::Positions);
            if (!positionsItem || positionsItem->elem_count == 0) {
                streaming_.lastError = "Frame missing positions";
                return false;
            }

            const size_t rawCount = positionsItem->elem_count;
            streaming_.rawPointCount = rawCount;

            streaming_.metrics.loss         = std::numeric_limits<float>::quiet_NaN();
            streaming_.metrics.psnr         = std::numeric_limits<float>::quiet_NaN();
            streaming_.metrics.learningRate = std::numeric_limits<float>::quiet_NaN();
            streaming_.metrics.hasCameraPos    = false;
            streaming_.metrics.hasCameraTarget = false;

            const size_t posComponents = std::max<std::size_t>(1, positionsItem->bytes / (positionsItem->elem_count * sizeof(float)));
            if (posComponents < 3) {
                streaming_.lastError = "Positions stream expected 3 components";
                return false;
            }

            const float* posPtr = reinterpret_cast<const float*>(positionsItem->ptr);
            const auto* colorItem = find_stream(df, ngp_streams::Colors);
            const float* colorPtr = colorItem ? reinterpret_cast<const float*>(colorItem->ptr) : nullptr;
            const size_t colorComponents = colorItem ? std::max<std::size_t>(1, colorItem->bytes / (colorItem->elem_count * sizeof(float))) : 0;

            const auto* densityItem = find_stream(df, ngp_streams::Density);
            const float* densityPtr = densityItem ? reinterpret_cast<const float*>(densityItem->ptr) : nullptr;
            const size_t densityComponents = densityItem ? std::max<std::size_t>(1, densityItem->bytes / (densityItem->elem_count * sizeof(float))) : 0;

            if (rawCount == 0) {
                streaming_.usedPointCount = 0;
                return false;
            }

            size_t targetCount = rawCount;
            const size_t maxPts = static_cast<size_t>(std::max(ui_.maxPoints, 1));
            if (rawCount > maxPts) targetCount = maxPts;
            const size_t stride = std::max<std::size_t>(1, rawCount / targetCount);

            pointsCpu_.resize(targetCount);
            sceneBounds_ = {};
            float densityMin = std::numeric_limits<float>::max();
            float densityMax = std::numeric_limits<float>::lowest();

            for (size_t dst = 0; dst < targetCount; ++dst) {
                const size_t src = std::min(dst * stride, rawCount - 1);
                auto& v          = pointsCpu_[dst];
                const float* p   = posPtr + src * posComponents;
                v.px             = p[0];
                v.py             = p[1];
                v.pz             = p[2];

                if (!sceneBounds_.valid) {
                    sceneBounds_.min = sceneBounds_.max = vv::make_float3(v.px, v.py, v.pz);
                    sceneBounds_.valid                  = true;
                } else {
                    sceneBounds_.min.x = std::min(sceneBounds_.min.x, v.px);
                    sceneBounds_.min.y = std::min(sceneBounds_.min.y, v.py);
                    sceneBounds_.min.z = std::min(sceneBounds_.min.z, v.pz);
                    sceneBounds_.max.x = std::max(sceneBounds_.max.x, v.px);
                    sceneBounds_.max.y = std::max(sceneBounds_.max.y, v.py);
                    sceneBounds_.max.z = std::max(sceneBounds_.max.z, v.pz);
                }

                float density = 0.5f;
                if (densityPtr && densityItem->elem_count > src) {
                    density = densityPtr[src * densityComponents];
                }
                v.density = density;
                densityMin = std::min(densityMin, density);
                densityMax = std::max(densityMax, density);

                if (colorPtr && colorItem->elem_count > src) {
                    const float* c = colorPtr + src * colorComponents;
                    v.r            = c[0];
                    v.g            = colorComponents > 1 ? c[1] : c[0];
                    v.b            = colorComponents > 2 ? c[2] : c[0];
                } else {
                    v.r = v.g = v.b = 0.0f;
                }

                v.feature  = 0.0f;
                v.radius   = 0.0f;
                v.emissive = 0.0f;
            }

            if (!(densityMax > densityMin)) {
                densityMin = 0.0f;
                densityMax = 1.0f;
            }

            streaming_.usedPointCount = targetCount;

            if (auto frameSeq = read_scalar<uint64_t>(find_stream(df, ngp_streams::FrameSeq))) {
                streaming_.metrics.frameSeq = *frameSeq;
            } else {
                streaming_.metrics.frameSeq = fv.fh->frame_id.load(std::memory_order_relaxed);
            }
            if (auto iter = read_scalar<uint32_t>(find_stream(df, ngp_streams::Iteration))) {
                streaming_.metrics.iteration = *iter;
            }
            if (auto timestamp = read_scalar<double>(find_stream(df, ngp_streams::Timestamp))) {
                streaming_.metrics.timestamp = *timestamp;
            } else {
                streaming_.metrics.timestamp = fv.fh->sim_time;
            }
            if (auto loss = read_scalar<float>(find_stream(df, ngp_streams::Loss))) {
                streaming_.metrics.loss = *loss;
            }
            if (auto psnr = read_scalar<float>(find_stream(df, ngp_streams::Psnr))) {
                streaming_.metrics.psnr = *psnr;
            }
            if (auto lr = read_scalar<float>(find_stream(df, ngp_streams::LearningRate))) {
                streaming_.metrics.learningRate = *lr;
            }
            if (const auto* camPosItem = find_stream(df, ngp_streams::CameraPos); camPosItem && camPosItem->bytes >= sizeof(float) * 3) {
                std::memcpy(&streaming_.metrics.cameraPos, camPosItem->ptr, sizeof(float) * 3);
                streaming_.metrics.hasCameraPos = true;
            }
            if (const auto* camTargetItem = find_stream(df, ngp_streams::CameraTarget); camTargetItem && camTargetItem->bytes >= sizeof(float) * 3) {
                std::memcpy(&streaming_.metrics.cameraTarget, camTargetItem->ptr, sizeof(float) * 3);
                streaming_.metrics.hasCameraTarget = true;
            }

            rebuild_stats_from_points(densityMin, densityMax, colorPtr != nullptr);
            upload_point_data();
            camera_.set_scene_bounds(sceneBounds_);

            if (!streaming_.firstFrameReceived) {
                if (ui_.autoFrameOnFirstRemote && sceneBounds_.valid) {
                    camera_.frame_scene(1.08f);
                }
                if (ui_.autoResetDensityOnFirstRemote) {
                    reset_density_window();
                }
            }
            if (ui_.requestDensityReset) {
                reset_density_window();
                ui_.requestDensityReset = false;
            }
            streaming_.firstFrameReceived = true;
            return true;
        }

        void rebuild_stats_from_points(float densityMin, float densityMax, bool hasExplicitColors) {
            statsInfo_ = PointFieldStats{};
            if (pointsCpu_.empty()) {
                statsInfo_.densityMin = 0.0f;
                statsInfo_.densityMax = 1.0f;
                return;
            }
            if (!(densityMax > densityMin)) {
                densityMin = 0.0f;
                densityMax = 1.0f;
            }
            statsInfo_.densityMin = densityMin;
            statsInfo_.densityMax = densityMax;
            statsInfo_.histogram.fill(0.0f);
            const float range = std::max(1e-6f, densityMax - densityMin);
            float maxBin = 0.0f;
            for (auto& v : pointsCpu_) {
                float norm = std::clamp((v.density - densityMin) / range, 0.0f, 1.0f);
                v.feature  = norm;
                v.radius   = 0.0025f + 0.02f * norm;
                v.emissive = 0.06f * norm;
                if (!hasExplicitColors) {
                    auto col = palette_viridis(norm);
                    v.r      = col[0];
                    v.g      = col[1];
                    v.b      = col[2];
                } else {
                    v.r = saturate(v.r);
                    v.g = saturate(v.g);
                    v.b = saturate(v.b);
                }
                const size_t bin = std::min<std::size_t>(statsInfo_.histogram.size() - 1, static_cast<size_t>(norm * (statsInfo_.histogram.size() - 1)));
                statsInfo_.histogram[bin] += 1.0f;
                maxBin = std::max(maxBin, statsInfo_.histogram[bin]);
            }
            if (maxBin > 1e-6f) {
                for (float& bin : statsInfo_.histogram) {
                    bin /= maxBin;
                }
            }
        }

        void ensure_point_buffer_capacity(std::size_t count) {
            if (count == 0) return;
            const VkDeviceSize needed = sizeof(PointVertex) * count;
            if (pointBuffer_.buffer != VK_NULL_HANDLE && needed <= pointBuffer_.size) return;
            const VkDeviceSize previous = pointBuffer_.size;
            destroy_buffer(pointBuffer_);
            VkDeviceSize allocSize = std::max<VkDeviceSize>(needed, previous > 0 ? previous * 2 : needed);
            allocSize              = std::max<VkDeviceSize>(allocSize, static_cast<VkDeviceSize>(256ull * 1024ull));
            create_buffer(allocSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, pointBuffer_);
        }

        void upload_point_data() {
            if (pointsCpu_.empty()) {
                pointCount_ = 0;
                return;
            }
            ensure_point_buffer_capacity(pointsCpu_.size());
            const VkDeviceSize size = sizeof(PointVertex) * pointsCpu_.size();
            std::memcpy(pointBuffer_.info.pMappedData, pointsCpu_.data(), static_cast<size_t>(size));
            VK_CHECK(vmaFlushAllocation(allocator_, pointBuffer_.allocation, 0, size));
            pointCount_ = static_cast<uint32_t>(pointsCpu_.size());
        }

        void transition_image(VkCommandBuffer cmd, const AttachmentView& view, VkImageLayout oldLayout, VkImageLayout newLayout, VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
            VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            barrier.srcStageMask     = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            barrier.srcAccessMask    = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
            barrier.dstStageMask     = dstStage;
            barrier.dstAccessMask    = dstAccess;
            barrier.oldLayout        = oldLayout;
            barrier.newLayout        = newLayout;
            barrier.image            = view.image;
            barrier.subresourceRange = {view.aspect, 0, 1, 0, 1};

            VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers    = &barrier;
            vkCmdPipelineBarrier2(cmd, &dep);
        }

    private:
        VkDevice dev_{VK_NULL_HANDLE};
        VmaAllocator allocator_{nullptr};
        DescriptorAllocator* descriptorAllocator_{nullptr};
        void* services_{nullptr};

        vv::CameraService camera_;
        vv::BoundingBox sceneBounds_{};

        std::vector<PointVertex> pointsCpu_;
        PointFieldStats statsInfo_{};
        UiState ui_{};
        uint32_t pointCount_{0};

        shmx::Client shmxClient_{};
        shmx::StaticState shmxStatic_{};
        StreamingState streaming_{};
        std::chrono::steady_clock::time_point lastConnectAttempt_{};

        GpuBuffer uniformBuffer_{};
        GpuBuffer pointBuffer_{};

        VkDescriptorSetLayout descriptorLayout_{VK_NULL_HANDLE};
        VkDescriptorSet descriptorSet_{VK_NULL_HANDLE};
        VkPipelineLayout pipelineLayout_{VK_NULL_HANDLE};
        VkPipeline pipeline_{VK_NULL_HANDLE};
        VkFormat colorFormat_{VK_FORMAT_B8G8R8A8_UNORM};
        VkFormat depthFormat_{VK_FORMAT_D32_SFLOAT};

        RendererStats stats_{};
    };

} // namespace

int main() {
    try {
        VulkanEngine engine;
        engine.configure_window(1600, 900, "Instant-NGP Dense Field Debugger");
        engine.set_renderer(std::make_unique<PointCloudRenderer>());
        engine.init();
        engine.run();
        engine.cleanup();
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "Fatal: %s\n", ex.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
