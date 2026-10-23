#include "application.hpp"

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

namespace application {

namespace {
constexpr float pi = 3.14159265358979323846f;
constexpr uint32_t object_count = 3;
struct Vec3 { float x, y, z; };
struct Mat4 { float data[16]{}; };

Mat4 identity() {
	Mat4 m{};
	for (int i = 0; i < 4; ++i) m.data[i * 5] = 1;
	return m;
}

// Column-major matrices match GLSL; transforms act on column vectors.
Mat4 operator*(const Mat4& a, const Mat4& b) {
	Mat4 m{};
	for (int c = 0; c < 4; ++c)
		for (int r = 0; r < 4; ++r)
			for (int k = 0; k < 4; ++k)
				m.data[c*4+r] += a.data[k*4+r] * b.data[c*4+k];
	return m;
}

Mat4 translation(Vec3 p) {
	auto m = identity();
	m.data[12] = p.x; m.data[13] = p.y; m.data[14] = p.z;
	return m;
}
Mat4 scaling(Vec3 s) {
	auto m = identity();
	m.data[0] = s.x; m.data[5] = s.y; m.data[10] = s.z;
	return m;
}
Mat4 rotation(int axis, float angle) {
	auto m = identity();
	const int a = (axis+1)%3, b = (axis+2)%3;
	m.data[a*4+a] = m.data[b*4+b] = std::cos(angle);
	m.data[a*4+b] = std::sin(angle);
	m.data[b*4+a] = -std::sin(angle);
	return m;
}
Mat4 projection(bool ortho, float aspect, float fov, float size) {
	constexpr float near_plane = 0.1f, far_plane = 100;
	Mat4 m{};
	if (ortho) {
		m.data[0] = 1/(size*aspect); m.data[5] = -1/size;
		m.data[10] = 1/(near_plane-far_plane);
		m.data[14] = near_plane/(near_plane-far_plane); m.data[15] = 1;
	} else {
		const float f = 1/std::tan(fov*pi/360);
		m.data[0] = f/aspect; m.data[5] = -f;
		m.data[10] = far_plane/(near_plane-far_plane); m.data[11] = -1;
		m.data[14] = near_plane*far_plane/(near_plane-far_plane);
	}
	return m;
}

constexpr std::array<Vec3, 6> positions = {{{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}}};
constexpr std::array<uint16_t, 24> indices = {2,4,0, 2,1,4, 2,5,1, 2,0,5, 3,0,4, 3,4,1, 3,1,5, 3,5,0};
struct Vertex { Vec3 position; Vec3 color; };
struct alignas(16) Uniform { Mat4 mvp; Mat4 model; float tint[4]; float options[4]; };
static_assert(sizeof(Uniform) == 160 && offsetof(Uniform, tint) == 128);
struct Buffer {
	VkBuffer handle = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	void* mapped = nullptr;
};
struct Object {
	Vec3 position{}, angles{22,32,0}, scale{1,1,1};
	float tint[4]{1,1,1,1};
	bool visible = true;
	Buffer uniform;
	VkDescriptorSet descriptor = VK_NULL_HANDLE;
};
Buffer vertex_buffer, index_buffer;
std::array<Object, object_count> objects;
VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;
int selected_object = 0;
bool multiple_objects = false, orthographic = false, procedural_colors = true;
bool playing = false, show_demo = false;
float animation_speed = 1, orbit_radius = 1.1f, orbit_height = 0.45f;
float fov = 50, ortho_size = 3.6f, camera_distance = 9;
double animation_time = 0, previous_time = -1;

void advance_animation(double time) {
	const double dt = previous_time < 0 ? 0 : std::max(0.0,time-previous_time);
	previous_time = time;
	if (playing) animation_time += dt*animation_speed;
}

void reset_settings() {
	selected_object = 0;
	multiple_objects = orthographic = playing = show_demo = false;
	procedural_colors = true;
	animation_speed = 1; orbit_radius = 1.1f; orbit_height = 0.45f;
	fov = 50; ortho_size = 3.6f; camera_distance = 9;
	animation_time = 0; previous_time = -1;
	for (auto& o : objects) {
		o.position = {}; o.angles = {22,32,0}; o.scale = {1,1,1}; o.visible = true;
		std::fill(std::begin(o.tint), std::end(o.tint), 1.0f);
	}
	objects[1].position = {-2.2f,0,-1}; objects[2].position = {2.2f,0,-1};
	objects[1].scale = objects[2].scale = {0.7f,0.7f,0.7f};
	objects[1].tint[1] = 0.55f; objects[2].tint[0] = 0.5f;
}
void check(VkResult result, const char* name) {
	if (result != VK_SUCCESS) throw std::runtime_error(std::string(name)+" failed: "+std::to_string(result));
}
void create_buffer(Buffer& b, VkDeviceSize size, VkBufferUsageFlags usage, const void* data = nullptr) {
	const VkBufferCreateInfo info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
		.usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
	const VmaAllocationCreateInfo allocation{
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO};
	VmaAllocationInfo details{};
	check(vmaCreateBuffer(graphics::internal::context.allocator, &info, &allocation,
		&b.handle, &b.allocation, &details), "vmaCreateBuffer");
	b.mapped = details.pMappedData;
	if (data) {
		std::memcpy(b.mapped, data, static_cast<size_t>(size));
		check(vmaFlushAllocation(graphics::internal::context.allocator, b.allocation, 0, size), "vmaFlushAllocation");
	}
}
void destroy_buffer(Buffer& b) {
	if (b.handle) vmaDestroyBuffer(graphics::internal::context.allocator, b.handle, b.allocation);
	b = {};
}
VkShaderModule load_shader(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) throw std::runtime_error(std::string("Cannot open ")+path+"; run from project root");
	const auto size = file.tellg();
	if (size <= 0 || static_cast<size_t>(size)%4) throw std::runtime_error("Invalid SPIR-V size");
	std::vector<uint32_t> code(static_cast<size_t>(size)/4);
	file.seekg(0);
	if (!file.read(reinterpret_cast<char*>(code.data()), size)) throw std::runtime_error("Cannot read shader");
	const VkShaderModuleCreateInfo info{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = code.size()*sizeof(uint32_t), .pCode = code.data()};
	VkShaderModule module = VK_NULL_HANDLE;
	check(vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module), "vkCreateShaderModule");
	return module;
}
void create_pipeline() {
	auto& context = graphics::internal::context;
	VkShaderModule vert = VK_NULL_HANDLE, frag = VK_NULL_HANDLE;
	try {
		vert = load_shader("shaders/octahedron.vert.spv");
		frag = load_shader("shaders/octahedron.frag.spv");
		const VkPipelineShaderStageCreateInfo stages[] = {
			{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vert, .pName = "main"},
			{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = frag, .pName = "main"}};
		const VkVertexInputBindingDescription binding{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
		const VkVertexInputAttributeDescription attributes[] = {
			{0,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,position)},
			{1,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,color)}};
		const VkPipelineVertexInputStateCreateInfo input{.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &binding,
			.vertexAttributeDescriptionCount = 2, .pVertexAttributeDescriptions = attributes};
		const VkPipelineInputAssemblyStateCreateInfo assembly{.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
		const VkPipelineViewportStateCreateInfo viewport{.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1, .scissorCount = 1};
		const VkPipelineRasterizationStateCreateInfo raster{.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1};
		const VkPipelineMultisampleStateCreateInfo samples{.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
		const VkPipelineDepthStencilStateCreateInfo depth{.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE, .depthCompareOp = VK_COMPARE_OP_LESS};
		const VkPipelineColorBlendAttachmentState attachment{.colorWriteMask =
			VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT};
		const VkPipelineColorBlendStateCreateInfo blend{.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			.attachmentCount = 1, .pAttachments = &attachment};
		const VkDynamicState states[] = {VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
		const VkPipelineDynamicStateCreateInfo dynamic{.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
			.dynamicStateCount = 2, .pDynamicStates = states};
		const VkPipelineLayoutCreateInfo layout{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1, .pSetLayouts = &descriptor_layout};
		check(vkCreatePipelineLayout(context.device, &layout, nullptr, &pipeline_layout), "vkCreatePipelineLayout");
		const VkGraphicsPipelineCreateInfo info{.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2, .pStages = stages, .pVertexInputState = &input, .pInputAssemblyState = &assembly,
			.pViewportState = &viewport, .pRasterizationState = &raster, .pMultisampleState = &samples,
			.pDepthStencilState = &depth, .pColorBlendState = &blend, .pDynamicState = &dynamic,
			.layout = pipeline_layout, .renderPass = context.render_pass};
		check(vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline), "vkCreateGraphicsPipelines");
	} catch (...) {
		if (frag) vkDestroyShaderModule(context.device, frag, nullptr);
		if (vert) vkDestroyShaderModule(context.device, vert, nullptr);
		throw;
	}
	vkDestroyShaderModule(context.device, frag, nullptr);
	vkDestroyShaderModule(context.device, vert, nullptr);
}
Mat4 model_matrix(const Object& o, uint32_t index) {
	Vec3 p = o.position, a = o.angles;
	if (index == 0) {
		const float t = static_cast<float>(animation_time);
		p.x += orbit_radius*std::sin(t); p.y += orbit_height*std::sin(2*t); p.z += orbit_radius*0.5f*std::sin(3*t);
		a.x += t*25; a.y += t*50; a.z += t*15;
	}
	return translation(p)*rotation(2,a.z*pi/180)*rotation(1,a.y*pi/180)*rotation(0,a.x*pi/180)*scaling(o.scale);
}
} // namespace

bool initialize() {
	reset_settings();
	try {
		std::array<Vertex, 6> vertices{};
		for (size_t i = 0; i < vertices.size(); ++i) {
			const auto p = positions[i];
			vertices[i] = {p, {(p.x+1)*0.5f,(p.y+1)*0.5f,(p.z+1)*0.5f}};
		}
		create_buffer(vertex_buffer, sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertices.data());
		create_buffer(index_buffer, sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indices.data());
		const VkDescriptorSetLayoutBinding binding{.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT};
		const VkDescriptorSetLayoutCreateInfo layout{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = 1, .pBindings = &binding};
		auto device = graphics::internal::context.device;
		check(vkCreateDescriptorSetLayout(device, &layout, nullptr, &descriptor_layout), "vkCreateDescriptorSetLayout");
		const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,object_count};
		const VkDescriptorPoolCreateInfo pool{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.maxSets = object_count, .poolSizeCount = 1, .pPoolSizes = &pool_size};
		check(vkCreateDescriptorPool(device, &pool, nullptr, &descriptor_pool), "vkCreateDescriptorPool");
		for (auto& o : objects) {
			create_buffer(o.uniform, sizeof(Uniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
			const VkDescriptorSetAllocateInfo allocation{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool, .descriptorSetCount = 1, .pSetLayouts = &descriptor_layout};
			check(vkAllocateDescriptorSets(device, &allocation, &o.descriptor), "vkAllocateDescriptorSets");
			const VkDescriptorBufferInfo buffer{o.uniform.handle,0,sizeof(Uniform)};
			const VkWriteDescriptorSet write{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = o.descriptor, .dstBinding = 0, .descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &buffer};
			vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
		}
		create_pipeline();
		return true;
	} catch (const std::exception& error) {
		std::cerr << error.what() << '\n';
		shutdown();
		return false;
	}
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);
	if (pipeline) vkDestroyPipeline(context.device, pipeline, nullptr);
	if (pipeline_layout) vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
	if (descriptor_pool) vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	if (descriptor_layout) vkDestroyDescriptorSetLayout(context.device, descriptor_layout, nullptr);
	pipeline = VK_NULL_HANDLE; pipeline_layout = VK_NULL_HANDLE;
	descriptor_pool = VK_NULL_HANDLE; descriptor_layout = VK_NULL_HANDLE;
	for (auto& o : objects) { destroy_buffer(o.uniform); o.descriptor = VK_NULL_HANDLE; }
	destroy_buffer(index_buffer); destroy_buffer(vertex_buffer);
}

void update(double time) {
	advance_animation(time);
	ImGui::SetNextWindowPos({16,16}, ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize({340,660}, ImGuiCond_FirstUseEver);
	if (ImGui::Begin("Lab 1 - Regular octahedron")) {
		ImGui::TextUnformatted("Variant 2");
		ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
		if (ImGui::CollapsingHeader("Projection", ImGuiTreeNodeFlags_DefaultOpen)) {
			if (ImGui::RadioButton("Perspective", !orthographic)) orthographic = false;
			ImGui::SameLine();
			if (ImGui::RadioButton("Orthographic", orthographic)) orthographic = true;
			if (orthographic) ImGui::SliderFloat("Half height", &ortho_size, 1, 8);
			else ImGui::SliderFloat("Field of view", &fov, 25, 90, "%.0f deg");
			ImGui::SliderFloat("Camera distance", &camera_distance, 4, 16);
		}
		if (ImGui::CollapsingHeader("Objects", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Checkbox("Multiple objects", &multiple_objects);
			if (!multiple_objects) selected_object = 0;
			if (multiple_objects) ImGui::Combo("Selected", &selected_object, "Octahedron 1\0Octahedron 2\0Octahedron 3\0");
			auto& o = objects[selected_object];
			ImGui::Checkbox("Visible", &o.visible);
			ImGui::SliderFloat3("Position", &o.position.x, -4, 4);
			ImGui::SliderFloat3("Rotation", &o.angles.x, -180, 180, "%.0f deg");
			ImGui::SliderFloat3("Scale", &o.scale.x, 0.1f, 3);
			ImGui::ColorEdit3("Color", o.tint);
			ImGui::Checkbox("Vertex colors", &procedural_colors);
		}
		if (ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Checkbox("Play", &playing);
			ImGui::SliderFloat("Speed", &animation_speed, 0.1f, 3);
			ImGui::SliderFloat("Radius", &orbit_radius, 0, 2);
			ImGui::SliderFloat("Height", &orbit_height, 0, 1.5f);
			if (ImGui::Button("Restart")) animation_time = 0;
		}
		if (ImGui::Button("Reset scene")) reset_settings();
		ImGui::Checkbox("ImGui demo", &show_demo);
	}
	ImGui::End();
	if (show_demo) ImGui::ShowDemoWindow(&show_demo);
}

void render(const graphics::internal::FrameData& fd) {
	if (!fd.command_buffer || !fd.framebuffer) return;
	auto& context = graphics::internal::context;
	const auto extent = context.swapchain_extent;
	if (!extent.width || !extent.height) return;
	const float sidebar = std::min(368.0f, static_cast<float>(extent.width) * 0.45f);
	const float scene_width = static_cast<float>(extent.width) - sidebar;
	const Mat4 pv = projection(orthographic,scene_width/extent.height,fov,ortho_size)
		*translation({0,0,-camera_distance});
	// prepare() waited for the previous frame's fence, so mapped UBOs are safe to write.
	for (uint32_t i = 0; i < object_count; ++i) {
		Uniform uniform{};
		uniform.model = model_matrix(objects[i],i);
		uniform.mvp = pv*uniform.model;
		std::copy(std::begin(objects[i].tint),std::end(objects[i].tint),uniform.tint);
		uniform.options[0] = procedural_colors ? 1.0f : 0.0f;
		std::memcpy(objects[i].uniform.mapped,&uniform,sizeof(uniform));
		check(vmaFlushAllocation(context.allocator,objects[i].uniform.allocation,0,sizeof(uniform)),"vmaFlushAllocation");
	}
	check(vkResetCommandBuffer(fd.command_buffer,0),"vkResetCommandBuffer");
	const VkCommandBufferBeginInfo begin{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
	check(vkBeginCommandBuffer(fd.command_buffer,&begin),"vkBeginCommandBuffer");
	const VkClearValue clear[] = {{.color = {{0.075f,0.09f,0.115f,1}}},{.depthStencil = {1,0}}};
	const VkRenderPassBeginInfo pass{.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass, .framebuffer = fd.framebuffer,
		.renderArea = {.extent = extent}, .clearValueCount = 2, .pClearValues = clear};
	vkCmdBeginRenderPass(fd.command_buffer,&pass,VK_SUBPASS_CONTENTS_INLINE);
	const VkViewport viewport{sidebar,0,scene_width,static_cast<float>(extent.height),0,1};
	const VkRect2D scissor{{0,0},extent};
	vkCmdSetViewport(fd.command_buffer,0,1,&viewport);
	vkCmdSetScissor(fd.command_buffer,0,1,&scissor);
	vkCmdBindPipeline(fd.command_buffer,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline);
	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer,0,1,&vertex_buffer.handle,&offset);
	vkCmdBindIndexBuffer(fd.command_buffer,index_buffer.handle,0,VK_INDEX_TYPE_UINT16);
	for (uint32_t i = 0; i < (multiple_objects ? object_count : 1); ++i) {
		if (!objects[i].visible) continue;
		vkCmdBindDescriptorSets(fd.command_buffer,VK_PIPELINE_BIND_POINT_GRAPHICS,
			pipeline_layout,0,1,&objects[i].descriptor,0,nullptr);
		vkCmdDrawIndexed(fd.command_buffer,static_cast<uint32_t>(indices.size()),1,0,0,0);
	}
	vkCmdEndRenderPass(fd.command_buffer);
	check(vkEndCommandBuffer(fd.command_buffer),"vkEndCommandBuffer");
}

bool self_test() {
	bool passed = true;
	auto expect = [&](bool condition, const char* name) {
		std::cout << (condition ? "PASS " : "FAIL ") << name << '\n'; passed &= condition;
	};
	auto close = [](float a, float b) { return std::abs(a-b) < 0.0001f; };
	std::map<std::pair<uint16_t,uint16_t>,int> edges;
	bool regular = true, outward = true;
	for (size_t i = 0; i < indices.size(); i += 3) {
		const auto a = positions[indices[i]], b = positions[indices[i+1]], c = positions[indices[i+2]];
		const Vec3 u{b.x-a.x,b.y-a.y,b.z-a.z}, v{c.x-a.x,c.y-a.y,c.z-a.z};
		const Vec3 n{u.y*v.z-u.z*v.y,u.z*v.x-u.x*v.z,u.x*v.y-u.y*v.x};
		outward &= n.x*a.x+n.y*a.y+n.z*a.z > 0;
		for (size_t j = 0; j < 3; ++j) {
			const auto x = indices[i+j], y = indices[i+(j+1)%3];
			++edges[std::minmax(x,y)];
			const auto p = positions[x], q = positions[y];
			regular &= close((p.x-q.x)*(p.x-q.x)+(p.y-q.y)*(p.y-q.y)+(p.z-q.z)*(p.z-q.z),2);
		}
	}
	expect(regular && edges.size() == 12,"12 equal edges of length sqrt(2)");
	expect(std::all_of(edges.begin(),edges.end(),[](const auto& e) { return e.second == 2; }),"closed mesh: two faces per edge");
	expect(outward,"eight outward-facing triangles");
	for (bool ortho : {false,true}) {
		const auto p = projection(ortho,16.0f/9,50,3);
		auto depth = [&](float z) { return (p.data[10]*z+p.data[14])/(p.data[11]*z+p.data[15]); };
		expect(close(depth(-0.1f),0) && close(depth(-100),1),ortho ? "orthographic depth [0,1]" : "perspective depth [0,1]");
		expect(p.data[5] < 0,"Vulkan Y correction");
		expect(close(p.data[0]*(16.0f/9),-p.data[5]),"aspect ratio preserves shape proportions");
	}
	const auto m = translation({2,3,4})*rotation(2,pi/2)*scaling({2,3,4});
	expect(close(m.data[0]+m.data[12],2) && close(m.data[1]+m.data[13],5) && close(m.data[2]+m.data[14],4),"scale -> rotate -> translate known point");
	reset_settings();
	advance_animation(0);
	playing = true; advance_animation(1);
	playing = false; advance_animation(10);
	expect(std::abs(animation_time-1) < 0.000001,"pause preserves animation phase");
	playing = true; animation_speed = 2; advance_animation(11);
	expect(std::abs(animation_time-3) < 0.000001,"resume and speed change do not accumulate paused time");
	reset_settings();
	return passed;
}

void verification_step(uint32_t frame) {
	if (frame == 0) { multiple_objects = true; playing = true; }
	if (frame == 60) orthographic = true;
	if (frame == 120) { playing = false; objects[1].scale = {1.1f,0.4f,0.8f}; objects[2].visible = false; }
	if (frame == 180) { orthographic = false; procedural_colors = false; objects[2].visible = true; objects[0].tint[1] = 0.4f; }
}

} // namespace application
