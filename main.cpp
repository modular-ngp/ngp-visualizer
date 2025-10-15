#include "vk_engine.h"
#include <fstream>
#include <imgui.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

#ifndef VK_CHECK
#define VK_CHECK(x)                                                                          \
    do {                                                                                     \
        VkResult r = (x);                                                                    \
        if (r != VK_SUCCESS) throw std::runtime_error("Vulkan error: " + std::to_string(r)); \
    } while (false)
#endif

static std::vector<char> load_spv(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("open " + p);
    size_t s = (size_t) f.tellg();
    f.seekg(0);
    std::vector<char> d(s);
    f.read(d.data(), s);
    return d;
}

static VkShaderModule make_shader(VkDevice d, const std::vector<char>& b) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = b.size();
    ci.pCode    = reinterpret_cast<const uint32_t*>(b.data());
    VkShaderModule m{};
    VK_CHECK(vkCreateShaderModule(d, &ci, nullptr, &m));
    return m;
}

class TriangleRenderer : public IRenderer {
public:
    // ============================================================
    // IRenderer Interface Implementation
    // ============================================================

    void query_required_device_caps(RendererCaps& c) override {
        c.allow_async_compute = false;
    }

    void get_capabilities(const EngineContext&, RendererCaps& c) override {
        c                            = RendererCaps{};
        c.presentation_mode          = PresentationMode::EngineBlit;
        c.preferred_swapchain_format = VK_FORMAT_B8G8R8A8_UNORM;
        c.color_attachments          = {AttachmentRequest{.name = "color", .format = VK_FORMAT_B8G8R8A8_UNORM, .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, .samples = VK_SAMPLE_COUNT_1_BIT, .aspect = VK_IMAGE_ASPECT_COLOR_BIT, .initial_layout = VK_IMAGE_LAYOUT_GENERAL}};
        c.presentation_attachment    = "color";
    }

    void initialize(const EngineContext& e, const RendererCaps& c, const FrameContext&) override {
        initialize_device_state(e, c);
        initialize_pipeline_resources();
    }

    void destroy(const EngineContext& e, const RendererCaps&) override {
        destroy_pipeline_resources();
        dev = VK_NULL_HANDLE;
    }

    void record_graphics(VkCommandBuffer cmd, const EngineContext&, const FrameContext& f) override {
        if (!is_ready_to_render(f)) return;

        const auto& target = f.color_attachments.front();

        prepare_for_rendering(cmd, target);
        execute_rendering(cmd, target, f.extent);
        finalize_rendering(cmd, target);
    }

private:
    // ============================================================
    // Initialization & Cleanup
    // ============================================================

    void initialize_device_state(const EngineContext& e, const RendererCaps& c) {
        dev = e.device;
        fmt = c.color_attachments.empty() ? VK_FORMAT_B8G8R8A8_UNORM : c.color_attachments.front().format;
    }

    void initialize_pipeline_resources() {
        create_pipeline_layout();
        create_graphics_pipeline();
    }

    void destroy_pipeline_resources() {
        cleanup_pipeline();
        cleanup_pipeline_layout();
    }

    // ============================================================
    // Pipeline Creation
    // ============================================================

    void create_pipeline_layout() {
        VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        VK_CHECK(vkCreatePipelineLayout(dev, &lci, nullptr, &layout));
    }

    void create_graphics_pipeline() {
        auto [vs, fs]                             = load_shaders();
        VkPipelineShaderStageCreateInfo stages[2] = {create_shader_stage_info(VK_SHADER_STAGE_VERTEX_BIT, vs), create_shader_stage_info(VK_SHADER_STAGE_FRAGMENT_BIT, fs)};

        VkGraphicsPipelineCreateInfo pci = build_pipeline_create_info(stages);
        VK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pci, nullptr, &pipe));

        destroy_shaders(vs, fs);
    }

    std::pair<VkShaderModule, VkShaderModule> load_shaders() {
        VkShaderModule vs = load_shader_module("shaders/triangle.vert.spv");
        VkShaderModule fs = load_shader_module("shaders/triangle.frag.spv");
        return {vs, fs};
    }

    VkShaderModule load_shader_module(const std::string& path) {
        return make_shader(dev, load_spv(path));
    }

    void destroy_shaders(VkShaderModule vs, VkShaderModule fs) {
        vkDestroyShaderModule(dev, vs, nullptr);
        vkDestroyShaderModule(dev, fs, nullptr);
    }

    VkGraphicsPipelineCreateInfo build_pipeline_create_info(VkPipelineShaderStageCreateInfo* stages) {
        // Store state objects as members to avoid dangling pointers
        m_renderingInfo      = create_rendering_info();
        m_vertexInputState   = create_vertex_input_state();
        m_inputAssemblyState = create_input_assembly_state();
        m_viewportState      = create_viewport_state();
        m_rasterizationState = create_rasterization_state();
        m_multisampleState   = create_multisample_state();
        m_colorBlendState    = create_color_blend_state();
        m_dynamicState       = create_dynamic_state();

        VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pci.pNext               = &m_renderingInfo;
        pci.stageCount          = 2;
        pci.pStages             = stages;
        pci.pVertexInputState   = &m_vertexInputState;
        pci.pInputAssemblyState = &m_inputAssemblyState;
        pci.pViewportState      = &m_viewportState;
        pci.pRasterizationState = &m_rasterizationState;
        pci.pMultisampleState   = &m_multisampleState;
        pci.pColorBlendState    = &m_colorBlendState;
        pci.pDynamicState       = &m_dynamicState;
        pci.layout              = layout;
        return pci;
    }

    // ============================================================
    // Pipeline State Creation Helpers
    // ============================================================

    VkPipelineShaderStageCreateInfo create_shader_stage_info(VkShaderStageFlagBits stage, VkShaderModule module) {
        VkPipelineShaderStageCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        info.stage  = stage;
        info.module = module;
        info.pName  = "main";
        return info;
    }

    VkPipelineVertexInputStateCreateInfo create_vertex_input_state() {
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        return vi;
    }

    VkPipelineInputAssemblyStateCreateInfo create_input_assembly_state() {
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        return ia;
    }

    VkPipelineViewportStateCreateInfo create_viewport_state() {
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1;
        vp.scissorCount  = 1;
        return vp;
    }

    VkPipelineRasterizationStateCreateInfo create_rasterization_state() {
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode    = VK_CULL_MODE_NONE;
        rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth   = 1.0f;
        return rs;
    }

    VkPipelineMultisampleStateCreateInfo create_multisample_state() {
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        return ms;
    }

    VkPipelineColorBlendStateCreateInfo create_color_blend_state() {
        m_colorBlendAttachment                = {};
        m_colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1;
        cb.pAttachments    = &m_colorBlendAttachment;
        return cb;
    }

    VkPipelineDynamicStateCreateInfo create_dynamic_state() {
        m_dynamicStates[0] = VK_DYNAMIC_STATE_VIEWPORT;
        m_dynamicStates[1] = VK_DYNAMIC_STATE_SCISSOR;

        VkPipelineDynamicStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        ds.dynamicStateCount = 2;
        ds.pDynamicStates    = m_dynamicStates;
        return ds;
    }

    VkPipelineRenderingCreateInfo create_rendering_info() {
        VkPipelineRenderingCreateInfo r{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        r.colorAttachmentCount    = 1;
        r.pColorAttachmentFormats = &fmt;
        return r;
    }

    // ============================================================
    // Rendering Operations
    // ============================================================

    bool is_ready_to_render(const FrameContext& f) const {
        return pipe && !f.color_attachments.empty();
    }

    void prepare_for_rendering(VkCommandBuffer cmd, const AttachmentView& target) {
        transition_image_layout(cmd, target, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }

    void execute_rendering(VkCommandBuffer cmd, const AttachmentView& target, VkExtent2D extent) {
        begin_rendering(cmd, target, extent);
        draw_triangle(cmd, extent);
        end_rendering(cmd);
    }

    void finalize_rendering(VkCommandBuffer cmd, const AttachmentView& target) {
        transition_image_layout(cmd, target, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    }

    // ============================================================
    // Image Layout Transitions
    // ============================================================

    void transition_image_layout(VkCommandBuffer cmd, const AttachmentView& target, VkImageLayout oldLayout, VkImageLayout newLayout) {
        auto [srcStage, dstStage, srcAccess, dstAccess] = get_barrier_params(oldLayout, newLayout);

        VkImageMemoryBarrier2 barrier = create_image_barrier(target, oldLayout, newLayout, srcStage, dstStage, srcAccess, dstAccess);
        execute_pipeline_barrier(cmd, barrier);
    }

    std::tuple<VkPipelineStageFlags2, VkPipelineStageFlags2, VkAccessFlags2, VkAccessFlags2> get_barrier_params(VkImageLayout oldLayout, VkImageLayout newLayout) {
        if (oldLayout == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
            return {VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT};
        } else // COLOR_ATTACHMENT_OPTIMAL to GENERAL
        {
            return {VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
        }
    }

    VkImageMemoryBarrier2 create_image_barrier(const AttachmentView& target, VkImageLayout oldLayout, VkImageLayout newLayout, VkPipelineStageFlags2 srcStage, VkPipelineStageFlags2 dstStage, VkAccessFlags2 srcAccess, VkAccessFlags2 dstAccess) {
        VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.srcStageMask     = srcStage;
        barrier.dstStageMask     = dstStage;
        barrier.srcAccessMask    = srcAccess;
        barrier.dstAccessMask    = dstAccess;
        barrier.oldLayout        = oldLayout;
        barrier.newLayout        = newLayout;
        barrier.image            = target.image;
        barrier.subresourceRange = {target.aspect, 0, 1, 0, 1};
        return barrier;
    }

    void execute_pipeline_barrier(VkCommandBuffer cmd, const VkImageMemoryBarrier2& barrier) {
        VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        depInfo.imageMemoryBarrierCount = 1;
        depInfo.pImageMemoryBarriers    = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    // ============================================================
    // Render Pass Operations
    // ============================================================

    void begin_rendering(VkCommandBuffer cmd, const AttachmentView& target, VkExtent2D extent) {
        VkRenderingAttachmentInfo colorAttachment = create_color_attachment(target);
        VkRenderingInfo renderInfo                = create_rendering_info_for_pass(extent, colorAttachment);
        vkCmdBeginRendering(cmd, &renderInfo);
    }

    VkRenderingAttachmentInfo create_color_attachment(const AttachmentView& target) {
        VkClearValue clearValue{.color = {{0.05f, 0.07f, 0.12f, 1.0f}}};

        VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        colorAttachment.imageView   = target.view;
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue  = clearValue;
        return colorAttachment;
    }

    VkRenderingInfo create_rendering_info_for_pass(VkExtent2D extent, const VkRenderingAttachmentInfo& colorAttachment) {
        VkRenderingInfo renderInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
        renderInfo.renderArea           = {{0, 0}, extent};
        renderInfo.layerCount           = 1;
        renderInfo.colorAttachmentCount = 1;
        renderInfo.pColorAttachments    = &colorAttachment;
        return renderInfo;
    }

    void end_rendering(VkCommandBuffer cmd) {
        vkCmdEndRendering(cmd);
    }

    // ============================================================
    // Drawing Operations
    // ============================================================

    void draw_triangle(VkCommandBuffer cmd, VkExtent2D extent) {
        bind_pipeline(cmd);
        set_viewport_and_scissor(cmd, extent);
        issue_draw_call(cmd);
    }

    void bind_pipeline(VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    }

    void set_viewport_and_scissor(VkCommandBuffer cmd, VkExtent2D extent) {
        VkViewport viewport = create_viewport(extent);
        VkRect2D scissor    = create_scissor(extent);

        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
    }

    VkViewport create_viewport(VkExtent2D extent) {
        VkViewport viewport{};
        viewport.width    = static_cast<float>(extent.width);
        viewport.height   = static_cast<float>(extent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        return viewport;
    }

    VkRect2D create_scissor(VkExtent2D extent) {
        return VkRect2D{{0, 0}, extent};
    }

    void issue_draw_call(VkCommandBuffer cmd) {
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }

    // ============================================================
    // Cleanup Operations
    // ============================================================

    void cleanup_pipeline() {
        if (pipe) {
            vkDestroyPipeline(dev, pipe, nullptr);
            pipe = VK_NULL_HANDLE;
        }
    }

    void cleanup_pipeline_layout() {
        if (layout) {
            vkDestroyPipelineLayout(dev, layout, nullptr);
            layout = VK_NULL_HANDLE;
        }
    }

    // ============================================================
    // Member Variables
    // ============================================================

    VkDevice dev{VK_NULL_HANDLE};
    VkPipelineLayout layout{VK_NULL_HANDLE};
    VkPipeline pipe{VK_NULL_HANDLE};
    VkFormat fmt{VK_FORMAT_B8G8R8A8_UNORM};

    // Pipeline state objects
    VkPipelineRenderingCreateInfo m_renderingInfo{};
    VkPipelineVertexInputStateCreateInfo m_vertexInputState{};
    VkPipelineInputAssemblyStateCreateInfo m_inputAssemblyState{};
    VkPipelineViewportStateCreateInfo m_viewportState{};
    VkPipelineRasterizationStateCreateInfo m_rasterizationState{};
    VkPipelineMultisampleStateCreateInfo m_multisampleState{};
    VkPipelineColorBlendStateCreateInfo m_colorBlendState{};
    VkPipelineDynamicStateCreateInfo m_dynamicState{};

    // Dynamic state array
    VkDynamicState m_dynamicStates[2]{};
    VkPipelineColorBlendAttachmentState m_colorBlendAttachment{};
};

int main() {
    try {
        VulkanEngine e;
        e.configure_window(1280, 720, "ex00_basic_window");
        e.set_renderer(std::make_unique<TriangleRenderer>());
        e.init();
        e.run();
        e.cleanup();
    } catch (const std::exception& ex) {
        fprintf(stderr, "Fatal: %s\n", ex.what());
        return 1;
    }
    return 0;
}
