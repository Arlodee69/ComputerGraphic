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
Vec3 operator+(Vec3 a, Vec3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
Vec3 operator*(Vec3 a, float s) { return {a.x*s,a.y*s,a.z*s}; }
float dot(Vec3 a, Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
Vec3 cross(Vec3 a, Vec3 b) { return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
Vec3 normalized(Vec3 v) {
	const float length = std::sqrt(dot(v,v));
	return length > 0.00001f ? v*(1/length) : Vec3{0,-1,0};
}
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
struct Vertex { Vec3 position; Vec3 color; Vec3 normal; };
struct alignas(16) Uniform { Mat4 model; float albedo[4]; float specular[4]; float parameters[4]; };
struct alignas(16) SceneUniform {
	Mat4 view_projection;
	float camera[4], ambient[4], direction[4], directional_color[4];
	uint32_t counts[4];
};
struct alignas(16) PointGPU { float position_intensity[4], color[4]; };
struct alignas(16) SpotGPU { float position_intensity[4], direction_inner[4], color_outer[4]; };
struct Push { Mat4 model; float color[4]; };
static_assert(sizeof(Uniform) == 112 && offsetof(Uniform, albedo) == 64);
static_assert(sizeof(SceneUniform) == 144 && offsetof(SceneUniform, counts) == 128);
static_assert(sizeof(PointGPU) == 32 && sizeof(SpotGPU) == 48 && sizeof(Push) == 80);
struct Buffer {
	VkBuffer handle = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	void* mapped = nullptr;
	VkDeviceSize size = 0;
};
struct Object {
	Vec3 position{}, angles{22,32,0}, scale{1,1,1};
	float tint[4]{1,1,1,1};
	float specular[4]{1,1,1,1};
	float shininess = 48;
	bool visible = true;
	Buffer uniform;
	VkDescriptorSet descriptor = VK_NULL_HANDLE;
};
Buffer vertex_buffer;
std::array<Object, object_count+1> objects;
Buffer scene_buffer, point_buffer, spot_buffer, line_buffer;
VkDescriptorSetLayout scene_layout = VK_NULL_HANDLE;
VkDescriptorSet scene_descriptor = VK_NULL_HANDLE;
VkPipeline line_pipeline = VK_NULL_HANDLE;
uint32_t line_vertex_count = 0;
struct Light {
	Vec3 position{2,2,2}, direction{-0.5f,-1,-0.5f};
	float color[3]{1,0.65f,0.25f};
	float intensity = 18, inner_angle = 15, outer_angle = 27;
	bool enabled = true;
};
std::vector<Light> point_lights, spot_lights;
Vec3 camera_position{0,2.2f,9}, camera_angles{-12,0,0};
Vec3 directional_direction{-0.4f,-1,-0.6f};
float directional_color[3]{1,0.95f,0.85f}, directional_intensity = 1.2f;
float ambient_color[3]{0.65f,0.75f,1}, ambient_intensity = 0.12f;
bool directional_enabled = true, show_markers = true, show_floor = true;
int selected_point = 0, selected_spot = 0;
VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;
int selected_object = 0;
bool multiple_objects = false, orthographic = false, procedural_colors = true;
bool playing = false, show_demo = false;
float animation_speed = 1, orbit_radius = 1.1f, orbit_height = 0.45f;
float fov = 50, ortho_size = 3.6f;
double animation_time = 0, previous_time = -1;
float sidebar_width = 390;
float cone_factor(float cosine, float inner, float outer) {
	if (inner-outer < 0.00001f) return cosine >= inner ? 1.0f : 0.0f;
	const float t = std::clamp((cosine-outer)/(inner-outer),0.0f,1.0f);
	return t*t*(3-2*t);
}
Mat4 view_matrix() {
	return rotation(2,-camera_angles.z*pi/180)*rotation(0,-camera_angles.x*pi/180)
		*rotation(1,-camera_angles.y*pi/180)*translation(camera_position*-1);
}

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
	fov = 50; ortho_size = 3.6f;
	animation_time = 0; previous_time = -1;
	for (auto& o : objects) {
		o.position = {}; o.angles = {22,32,0}; o.scale = {1,1,1}; o.visible = true;
		std::fill(std::begin(o.tint), std::end(o.tint), 1.0f);
	}
	objects[1].position = {-2.2f,0,-1}; objects[2].position = {2.2f,0,-1};
	objects[1].scale = objects[2].scale = {0.7f,0.7f,0.7f};
	objects[1].tint[1] = 0.55f; objects[2].tint[0] = 0.5f;
	objects[3].position = {0,-1.5f,0}; objects[3].scale = {6,0.08f,6};
	objects[3].angles = {}; objects[3].tint[0] = 0.3f; objects[3].tint[1] = 0.34f; objects[3].tint[2] = 0.38f;
	for (auto& o : objects) { std::fill(std::begin(o.specular),std::end(o.specular),1.0f); o.shininess = 48; }
	camera_position = {0,2.2f,9}; camera_angles = {-12,0,0};
	multiple_objects = true; procedural_colors = false;
	directional_enabled = show_markers = show_floor = true;
	directional_direction = {-0.4f,-1,-0.6f}; directional_intensity = 1.2f;
	directional_color[0] = 1; directional_color[1] = 0.95f; directional_color[2] = 0.85f;
	ambient_color[0] = 0.65f; ambient_color[1] = 0.75f; ambient_color[2] = 1; ambient_intensity = 0.12f;
	point_lights = {Light{}};
	Light spot; spot.position = {-2,3,2}; spot.direction = {0.3f,-0.8f,-0.6f};
	spot.color[0] = 0.25f; spot.color[1] = 0.6f; spot.color[2] = 1; spot.intensity = 35;
	spot_lights = {spot}; selected_point = selected_spot = 0;
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
	b.size = size;
	if (data) {
		std::memcpy(b.mapped, data, static_cast<size_t>(size));
		check(vmaFlushAllocation(graphics::internal::context.allocator, b.allocation, 0, size), "vmaFlushAllocation");
	}
}
void destroy_buffer(Buffer& b) {
	if (b.handle) vmaDestroyBuffer(graphics::internal::context.allocator, b.handle, b.allocation);
	b = {};
}
void write_buffer(Buffer& b, const void* data, VkDeviceSize size) {
	if (!size) return;
	std::memcpy(b.mapped,data,static_cast<size_t>(size));
	check(vmaFlushAllocation(graphics::internal::context.allocator,b.allocation,0,size),"vmaFlushAllocation");
}
void bind_buffer(VkDescriptorSet set, uint32_t binding, VkDescriptorType type, const Buffer& b) {
	const VkDescriptorBufferInfo info{b.handle,0,b.size};
	const VkWriteDescriptorSet write{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet = set,
		.dstBinding = binding,.descriptorCount = 1,.descriptorType = type,.pBufferInfo = &info};
	vkUpdateDescriptorSets(graphics::internal::context.device,1,&write,0,nullptr);
}
void ensure_storage(Buffer& b, VkDeviceSize bytes, uint32_t binding) {
	if (b.size >= bytes) return;
	VkPhysicalDeviceProperties properties{};
	vkGetPhysicalDeviceProperties(graphics::internal::context.physical_device,&properties);
	if (bytes > properties.limits.maxStorageBufferRange) throw std::runtime_error("Too many lights for this GPU");
	Buffer replacement;
	create_buffer(replacement,bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	destroy_buffer(b); b = replacement;
	bind_buffer(scene_descriptor,binding,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,b);
}
std::vector<Vertex> make_mesh() {
	std::vector<Vertex> mesh;
	auto triangle = [&](Vec3 a, Vec3 b, Vec3 c) {
		const auto n = normalized(cross(b-a,c-a));
		for (auto p : {a,b,c}) mesh.push_back({p,{(p.x+1)*0.5f,(p.y+1)*0.5f,(p.z+1)*0.5f},n});
	};
	for (size_t i = 0; i < indices.size(); i += 3) triangle(positions[indices[i]],positions[indices[i+1]],positions[indices[i+2]]);
	// Separate vertices per face preserve the octahedron's flat normals.
	for (int axis = 0; axis < 3; ++axis) for (float sign : {-1.0f,1.0f}) {
		Vec3 n{}, u{}, v{};
		if (axis == 0) { n = {sign,0,0}; u = {0,1,0}; }
		if (axis == 1) { n = {0,sign,0}; u = {0,0,1}; }
		if (axis == 2) { n = {0,0,sign}; u = {1,0,0}; }
		v = cross(n,u);
		triangle(n-u-v,n+u-v,n+u+v); triangle(n-u-v,n+u+v,n-u+v);
	}
	return mesh;
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
			{1,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,color)},
			{2,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,normal)}};
		const VkPipelineVertexInputStateCreateInfo input{.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &binding,
			.vertexAttributeDescriptionCount = 3, .pVertexAttributeDescriptions = attributes};
		VkPipelineInputAssemblyStateCreateInfo assembly{.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
		const VkPipelineViewportStateCreateInfo viewport{.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
			.viewportCount = 1, .scissorCount = 1};
		VkPipelineRasterizationStateCreateInfo raster{.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
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
		const VkDescriptorSetLayout layouts[] = {scene_layout,descriptor_layout};
		const VkPushConstantRange push{VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(Push)};
		const VkPipelineLayoutCreateInfo layout{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 2, .pSetLayouts = layouts,.pushConstantRangeCount = 1,.pPushConstantRanges = &push};
		check(vkCreatePipelineLayout(context.device, &layout, nullptr, &pipeline_layout), "vkCreatePipelineLayout");
		const VkGraphicsPipelineCreateInfo info{.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2, .pStages = stages, .pVertexInputState = &input, .pInputAssemblyState = &assembly,
			.pViewportState = &viewport, .pRasterizationState = &raster, .pMultisampleState = &samples,
			.pDepthStencilState = &depth, .pColorBlendState = &blend, .pDynamicState = &dynamic,
			.layout = pipeline_layout, .renderPass = context.render_pass};
		check(vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline), "vkCreateGraphicsPipelines");
		assembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
		raster.cullMode = VK_CULL_MODE_NONE;
		check(vkCreateGraphicsPipelines(context.device,VK_NULL_HANDLE,1,&info,nullptr,&line_pipeline),"create line pipeline");
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
		const auto mesh = make_mesh();
		create_buffer(vertex_buffer,mesh.size()*sizeof(Vertex),VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,mesh.data());
		const VkDescriptorSetLayoutBinding binding{.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT};
		const VkDescriptorSetLayoutCreateInfo layout{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = 1, .pBindings = &binding};
		auto device = graphics::internal::context.device;
		check(vkCreateDescriptorSetLayout(device, &layout, nullptr, &descriptor_layout), "vkCreateDescriptorSetLayout");
		const VkDescriptorSetLayoutBinding scene_bindings[] = {
			{0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
			{1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
			{2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr}};
		const VkDescriptorSetLayoutCreateInfo scene_info{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = 3,.pBindings = scene_bindings};
		check(vkCreateDescriptorSetLayout(device,&scene_info,nullptr,&scene_layout),"create scene layout");
		const VkDescriptorPoolSize pool_sizes[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,object_count+2},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2}};
		const VkDescriptorPoolCreateInfo pool{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.maxSets = object_count+2, .poolSizeCount = 2, .pPoolSizes = pool_sizes};
		check(vkCreateDescriptorPool(device, &pool, nullptr, &descriptor_pool), "vkCreateDescriptorPool");
		const VkDescriptorSetAllocateInfo scene_allocation{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = descriptor_pool,.descriptorSetCount = 1,.pSetLayouts = &scene_layout};
		check(vkAllocateDescriptorSets(device,&scene_allocation,&scene_descriptor),"allocate scene descriptor");
		create_buffer(scene_buffer,sizeof(SceneUniform),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
		bind_buffer(scene_descriptor,0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,scene_buffer);
		ensure_storage(point_buffer,sizeof(PointGPU),1); ensure_storage(spot_buffer,sizeof(SpotGPU),2);
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
	if (line_pipeline) vkDestroyPipeline(context.device,line_pipeline,nullptr);
	if (pipeline) vkDestroyPipeline(context.device, pipeline, nullptr);
	if (pipeline_layout) vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
	if (descriptor_pool) vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	if (descriptor_layout) vkDestroyDescriptorSetLayout(context.device, descriptor_layout, nullptr);
	if (scene_layout) vkDestroyDescriptorSetLayout(context.device,scene_layout,nullptr);
	scene_layout = VK_NULL_HANDLE; scene_descriptor = VK_NULL_HANDLE; line_pipeline = VK_NULL_HANDLE;
	pipeline = VK_NULL_HANDLE; pipeline_layout = VK_NULL_HANDLE;
	descriptor_pool = VK_NULL_HANDLE; descriptor_layout = VK_NULL_HANDLE;
	for (auto& o : objects) { destroy_buffer(o.uniform); o.descriptor = VK_NULL_HANDLE; }
	destroy_buffer(vertex_buffer);
	destroy_buffer(scene_buffer); destroy_buffer(point_buffer); destroy_buffer(spot_buffer); destroy_buffer(line_buffer);
}

void edit_lights(const char* label, std::vector<Light>& lights, int& selected, bool spot) {
	if (!ImGui::CollapsingHeader(label)) return;
	ImGui::PushID(label);
	if (ImGui::Button("Add")) { lights.emplace_back(); selected = static_cast<int>(lights.size())-1; }
	ImGui::SameLine();
	ImGui::BeginDisabled(lights.empty());
	if (ImGui::Button("Delete")) { lights.erase(lights.begin()+selected); selected = 0; }
	ImGui::EndDisabled();
	ImGui::Text("Count: %d",static_cast<int>(lights.size()));
	if (!lights.empty()) {
		selected = std::clamp(selected,0,static_cast<int>(lights.size())-1);
		ImGui::SliderInt("Selected",&selected,0,static_cast<int>(lights.size())-1);
		auto& light = lights[selected];
		ImGui::Checkbox("Enabled",&light.enabled);
		ImGui::DragFloat3("Position",&light.position.x,0.05f,-20,20);
		ImGui::ColorEdit3("Color",light.color);
		ImGui::SliderFloat("Intensity",&light.intensity,0,100);
		if (spot) {
			ImGui::SliderFloat3("Direction",&light.direction.x,-1,1);
			ImGui::SliderFloat("Inner angle",&light.inner_angle,0,80,"%.1f deg");
			ImGui::SliderFloat("Outer angle",&light.outer_angle,light.inner_angle,85,"%.1f deg");
			light.outer_angle = std::max(light.outer_angle,light.inner_angle);
		}
	}
	ImGui::PopID();
}

void update_camera(float dt) {
	auto& io = ImGui::GetIO();
	if (ImGui::IsMouseDown(ImGuiMouseButton_Right) && io.MousePos.x > sidebar_width && !io.WantCaptureMouse) {
		camera_angles.y -= io.MouseDelta.x*0.15f;
		camera_angles.x = std::clamp(camera_angles.x-io.MouseDelta.y*0.15f,-89.0f,89.0f);
	}
	if (io.WantCaptureKeyboard) return;
	const float yaw = camera_angles.y*pi/180, pitch = camera_angles.x*pi/180;
	const Vec3 forward{-std::sin(yaw)*std::cos(pitch),std::sin(pitch),-std::cos(yaw)*std::cos(pitch)};
	const auto right = normalized(cross(forward,{0,1,0}));
	Vec3 movement{};
	if (ImGui::IsKeyDown(ImGuiKey_W)) movement = movement+forward;
	if (ImGui::IsKeyDown(ImGuiKey_S)) movement = movement-forward;
	if (ImGui::IsKeyDown(ImGuiKey_D)) movement = movement+right;
	if (ImGui::IsKeyDown(ImGuiKey_A)) movement = movement-right;
	if (ImGui::IsKeyDown(ImGuiKey_E)) movement.y += 1;
	if (ImGui::IsKeyDown(ImGuiKey_Q)) movement.y -= 1;
	if (dot(movement,movement) > 0) camera_position = camera_position+normalized(movement)*(dt*4);
}

void update(double time) {
	const float dt = previous_time < 0 ? 0 : static_cast<float>(std::clamp(time-previous_time,0.0,0.1));
	update_camera(dt);
	advance_animation(time);
	sidebar_width = std::min(390.0f,ImGui::GetIO().DisplaySize.x*0.45f);
	ImGui::SetNextWindowPos({12,12}, ImGuiCond_Always);
	ImGui::SetNextWindowSize({sidebar_width-24,std::max(180.0f,ImGui::GetIO().DisplaySize.y-24)}, ImGuiCond_Always);
	if (ImGui::Begin("Lab 2 - Camera and lighting",nullptr,ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoResize)) {
		ImGui::PushItemWidth(std::max(100.0f,sidebar_width-180));
		ImGui::TextUnformatted("Variant 2");
		ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
		if (ImGui::CollapsingHeader("Projection", ImGuiTreeNodeFlags_DefaultOpen)) {
			if (ImGui::RadioButton("Perspective", !orthographic)) orthographic = false;
			ImGui::SameLine();
			if (ImGui::RadioButton("Orthographic", orthographic)) orthographic = true;
			if (orthographic) ImGui::SliderFloat("Half height", &ortho_size, 1, 8);
			else ImGui::SliderFloat("Field of view", &fov, 25, 90, "%.0f deg");
			ImGui::DragFloat3("Camera position",&camera_position.x,0.05f,-30,30);
			ImGui::SliderFloat3("Camera rotation",&camera_angles.x,-180,180,"%.0f deg");
			camera_angles.x = std::clamp(camera_angles.x,-89.0f,89.0f);
		}
		if (ImGui::CollapsingHeader("Environment",ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::ColorEdit3("Ambient color",ambient_color);
			ImGui::SliderFloat("Ambient intensity",&ambient_intensity,0,1);
			ImGui::Checkbox("Light markers",&show_markers);
			ImGui::SameLine(); ImGui::Checkbox("Floor",&show_floor);
		}
		if (ImGui::CollapsingHeader("Directional light",ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Checkbox("Directional enabled",&directional_enabled);
			ImGui::SliderFloat3("Light direction",&directional_direction.x,-1,1);
			ImGui::ColorEdit3("Light color",directional_color);
			ImGui::SliderFloat("Light intensity",&directional_intensity,0,5);
		}
		edit_lights("Point lights",point_lights,selected_point,false);
		edit_lights("Spot lights",spot_lights,selected_spot,true);
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
			ImGui::ColorEdit3("Specular color",o.specular);
			ImGui::SliderFloat("Shininess",&o.shininess,1,256);
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
		ImGui::PopItemWidth();
	}
	ImGui::End();
	if (show_demo) ImGui::ShowDemoWindow(&show_demo);
}

void render(const graphics::internal::FrameData& fd) {
	if (!fd.command_buffer || !fd.framebuffer) return;
	auto& context = graphics::internal::context;
	const auto extent = context.swapchain_extent;
	if (!extent.width || !extent.height) return;
	const float sidebar = std::min(sidebar_width, static_cast<float>(extent.width) * 0.45f);
	const float scene_width = static_cast<float>(extent.width) - sidebar;
	const Mat4 pv = projection(orthographic,scene_width/extent.height,fov,ortho_size)
		*view_matrix();
	SceneUniform scene{};
	scene.view_projection = pv;
	scene.camera[0] = camera_position.x; scene.camera[1] = camera_position.y; scene.camera[2] = camera_position.z;
	std::copy(ambient_color,ambient_color+3,scene.ambient); scene.ambient[3] = ambient_intensity;
	const auto direction = normalized(directional_direction);
	scene.direction[0] = direction.x; scene.direction[1] = direction.y; scene.direction[2] = direction.z;
	scene.direction[3] = directional_enabled ? directional_intensity : 0;
	std::copy(directional_color,directional_color+3,scene.directional_color);
	scene.counts[0] = static_cast<uint32_t>(point_lights.size()); scene.counts[1] = static_cast<uint32_t>(spot_lights.size());
	write_buffer(scene_buffer,&scene,sizeof(scene));
	std::vector<PointGPU> points;
	for (const auto& light : point_lights) {
		PointGPU point{{light.position.x,light.position.y,light.position.z,light.enabled ? light.intensity : 0},{}};
		std::copy(light.color,light.color+3,point.color); points.push_back(point);
	}
	std::vector<SpotGPU> spots;
	for (const auto& light : spot_lights) {
		const auto d = normalized(light.direction);
		SpotGPU spot{{light.position.x,light.position.y,light.position.z,light.enabled ? light.intensity : 0},
			{d.x,d.y,d.z,std::cos(light.inner_angle*pi/180)},
			{light.color[0],light.color[1],light.color[2],std::cos(light.outer_angle*pi/180)}};
		spots.push_back(spot);
	}
	ensure_storage(point_buffer,std::max<size_t>(1,points.size())*sizeof(PointGPU),1);
	ensure_storage(spot_buffer,std::max<size_t>(1,spots.size())*sizeof(SpotGPU),2);
	write_buffer(point_buffer,points.data(),points.size()*sizeof(PointGPU));
	write_buffer(spot_buffer,spots.data(),spots.size()*sizeof(SpotGPU));
	std::vector<Vertex> lines;
	if (show_markers) for (const auto& light : spot_lights) {
		const auto d = normalized(light.direction);
		const auto u = normalized(cross(d,std::abs(d.y) < 0.9f ? Vec3{0,1,0} : Vec3{1,0,0}));
		const auto v = cross(d,u);
		auto line = [&](Vec3 a, Vec3 b) { lines.push_back({a,{1,1,1},{0,1,0}}); lines.push_back({b,{1,1,1},{0,1,0}}); };
		constexpr float length = 2.5f;
		for (float angle : {light.inner_angle,light.outer_angle}) {
			const float radius = length*std::tan(angle*pi/180);
			auto ring = [&](float phase) { return light.position+d*length+(u*std::cos(phase)+v*std::sin(phase))*radius; };
			for (int j = 0; j < 48; ++j) line(ring(2*pi*j/48),ring(2*pi*(j+1)/48));
			for (int j = 0; j < 4; ++j) line(light.position,ring(pi*j/2));
		}
		line(light.position,light.position+d*length);
	}
	line_vertex_count = static_cast<uint32_t>(lines.size());
	if (!lines.empty()) {
		const auto bytes = lines.size()*sizeof(Vertex);
		if (line_buffer.size < bytes) { Buffer replacement; create_buffer(replacement,bytes,VK_BUFFER_USAGE_VERTEX_BUFFER_BIT); destroy_buffer(line_buffer); line_buffer = replacement; }
		write_buffer(line_buffer,lines.data(),bytes);
	}
	// prepare() waited for the previous frame's fence, so mapped UBOs are safe to write.
	for (uint32_t i = 0; i < objects.size(); ++i) {
		Uniform uniform{};
		uniform.model = model_matrix(objects[i],i);
		std::copy(std::begin(objects[i].tint),std::end(objects[i].tint),uniform.albedo);
		std::copy(std::begin(objects[i].specular),std::end(objects[i].specular),uniform.specular);
		uniform.parameters[0] = objects[i].shininess; uniform.parameters[1] = procedural_colors && i != 3 ? 1.0f : 0.0f;
		write_buffer(objects[i].uniform,&uniform,sizeof(uniform));
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
	vkCmdBindDescriptorSets(fd.command_buffer,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline_layout,0,1,&scene_descriptor,0,nullptr);
	const Push normal_draw{identity(),{0,0,0,0}};
	vkCmdPushConstants(fd.command_buffer,pipeline_layout,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(Push),&normal_draw);
	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer,0,1,&vertex_buffer.handle,&offset);
	for (uint32_t i = 0; i < objects.size(); ++i) {
		if (i > 0 && i < object_count && !multiple_objects) continue;
		if (i == 3 && !show_floor) continue;
		if (!objects[i].visible) continue;
		vkCmdBindDescriptorSets(fd.command_buffer,VK_PIPELINE_BIND_POINT_GRAPHICS,
			pipeline_layout,1,1,&objects[i].descriptor,0,nullptr);
		vkCmdDraw(fd.command_buffer,i == 3 ? 36 : 24,1,i == 3 ? 24 : 0,0);
	}
	if (show_markers) {
		vkCmdBindDescriptorSets(fd.command_buffer,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline_layout,1,1,&objects[0].descriptor,0,nullptr);
		auto marker = [&](const Light& light) {
			const Push push{translation(light.position)*scaling({0.08f,0.08f,0.08f}),{light.color[0],light.color[1],light.color[2],1}};
			vkCmdPushConstants(fd.command_buffer,pipeline_layout,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(Push),&push);
			vkCmdDraw(fd.command_buffer,36,1,24,0);
		};
		for (const auto& light : point_lights) marker(light);
		for (const auto& light : spot_lights) marker(light);
		if (line_vertex_count) {
			vkCmdBindPipeline(fd.command_buffer,VK_PIPELINE_BIND_POINT_GRAPHICS,line_pipeline);
			vkCmdBindVertexBuffers(fd.command_buffer,0,1,&line_buffer.handle,&offset);
			const Push push{identity(),{1,1,1,1}};
			vkCmdPushConstants(fd.command_buffer,pipeline_layout,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(Push),&push);
			vkCmdDraw(fd.command_buffer,line_vertex_count,1,0,0);
		}
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
	const auto mesh = make_mesh();
	bool normals_valid = mesh.size() == 60;
	for (size_t i = 0; i < mesh.size(); i += 3) {
		const auto tangent_a = mesh[i+1].position-mesh[i].position;
		const auto tangent_b = mesh[i+2].position-mesh[i].position;
		for (size_t j = 0; j < 3; ++j) {
			const auto n = mesh[i+j].normal;
			normals_valid &= close(dot(n,n),1) && close(dot(n,tangent_a),0) && close(dot(n,tangent_b),0);
		}
	}
	expect(normals_valid,"procedural face normals: unit length and perpendicular to triangles");
	auto transform = [](const Mat4& matrix, Vec3 p, float w) {
		return Vec3{matrix.data[0]*p.x+matrix.data[4]*p.y+matrix.data[8]*p.z+matrix.data[12]*w,
			matrix.data[1]*p.x+matrix.data[5]*p.y+matrix.data[9]*p.z+matrix.data[13]*w,
			matrix.data[2]*p.x+matrix.data[6]*p.y+matrix.data[10]*p.z+matrix.data[14]*w};
	};
	const auto camera_origin = transform(view_matrix(),camera_position,1);
	expect(close(dot(camera_origin,camera_origin),0),"view matrix maps camera position to origin");
	camera_angles = {0,90,0};
	const auto ahead = transform(view_matrix(),camera_position+Vec3{-1,0,0},1);
	expect(close(ahead.x,0) && close(ahead.y,0) && close(ahead.z,-1),"yaw rotates camera viewing direction");
	const auto r = rotation(1,0.7f)*rotation(0,0.3f);
	const auto tangent = transform(r*scaling({2,0.4f,3}),{1,-1,0},0);
	const auto normal = transform(r*scaling({0.5f,2.5f,1.0f/3}),normalized({1,1,1}),0);
	expect(close(dot(normal,tangent),0),"inverse-transpose normals remain perpendicular after nonuniform scale");
	const float inner = std::cos(15*pi/180), outer = std::cos(30*pi/180);
	expect(close(cone_factor(inner,inner,outer),1) && close(cone_factor(outer,inner,outer),0)
		&& close(cone_factor((inner+outer)/2,inner,outer),0.5f),"spot cone: full center, smooth middle, zero outside");
	expect(close(cone_factor(inner,inner,inner),1) && close(cone_factor(outer,inner,inner),0),"equal spot angles avoid division by zero");
	expect(close((1.0f/(4*4))/(1.0f/(2*2)),0.25f),"inverse-square attenuation quarters radiance at twice the distance");
	reset_settings();
	return passed;
}

void verification_step(uint32_t frame) {
	if (frame == 0) { reset_settings(); point_lights.clear(); spot_lights.clear(); show_markers = false; std::cout << "VERIFY directional\n" << std::flush; }
	if (frame == 45) { directional_enabled = false; point_lights.emplace_back(); std::cout << "VERIFY point\n" << std::flush; }
	if (frame == 90) { point_lights.clear(); Light spot; spot.position = {0,3,3}; spot.direction = {0,-1,-1}; spot.intensity = 60; spot_lights = {spot}; std::cout << "VERIFY spot\n" << std::flush; }
	if (frame == 135) {
		reset_settings(); point_lights.resize(4); spot_lights.resize(3); show_markers = false;
		point_lights[1].position = {-3,1,2}; point_lights[2].position = {0,3,-2}; point_lights[3].position = {3,1,-2};
		spot_lights[1].inner_angle = spot_lights[1].outer_angle = 20;
		playing = true; std::cout << "VERIFY multiple\n" << std::flush;
	}
	if (frame == 180) { directional_enabled = false; point_lights.clear(); spot_lights.clear(); playing = false; ambient_intensity = 0.3f; std::cout << "VERIFY ambient\n" << std::flush; }
	if (frame == 225) { directional_enabled = true; objects[0].scale = {1.4f,0.5f,0.7f}; objects[0].shininess = 128; objects[1].tint[0] = 0.2f; std::cout << "VERIFY material\n" << std::flush; }
	if (frame == 270) { orthographic = true; camera_position = {3,3,9}; camera_angles = {-16,18,4}; std::cout << "VERIFY camera\n" << std::flush; }
	if (frame == 315) { reset_settings(); std::cout << "VERIFY markers\n" << std::flush; }
}

} // namespace application
