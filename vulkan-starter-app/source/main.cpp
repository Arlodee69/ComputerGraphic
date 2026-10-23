#include <cstdint>
#include <climits>

#include <iostream>
#include <cstring>
#include <chrono>
#include <thread>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>

#include "graphics_internal.hpp"
#include "application.hpp"

namespace {

constexpr int32_t default_window_width = 1280;
constexpr int32_t default_window_height = 720;

constexpr char default_window_title[] = "Vulkan Starter App";

GLFWwindow* glfw_window;

} // namespace

int main(int argc, char** argv) {
	if (argc > 1 && std::strcmp(argv[1], "--self-test") == 0) return application::self_test() ? EXIT_SUCCESS : EXIT_FAILURE;
	const bool verify = argc > 1 && std::strcmp(argv[1], "--verify") == 0;
	uint32_t frame = 0;
	int status = EXIT_SUCCESS;

	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW\n";
		return EXIT_FAILURE;
	}

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	glfw_window = glfwCreateWindow(default_window_width, default_window_height,
	                               default_window_title, nullptr, nullptr);
	if (glfw_window == nullptr) {
		status = EXIT_FAILURE;
		goto err_null_window;
	}

	glfwSetFramebufferSizeCallback(glfw_window, [](GLFWwindow*, int width, int height){
		if (width == 0 || height == 0) {
			return;
		}

		graphics::internal::resize(width, height);
	});

	if (ImGui::CreateContext() == nullptr) {
		std::cerr << "Failed to create ImGUI context\n";
		status = EXIT_FAILURE;
		goto err_imgui_init;
	}

	if (!ImGui_ImplGlfw_InitForVulkan(glfw_window, true)) {
		std::cerr << "Failed to initialize ImGUI GLFW backend for Vulkan renderer\n";
		status = EXIT_FAILURE;
		goto err_imgui_glfw_init;
	}

	if (!graphics::internal::initialize(glfw_window)) {
		std::cerr << "Failed to initialize graphics\n";
		status = EXIT_FAILURE;
		goto err_graphics_init;
	}

	if (!application::initialize()) {
		std::cerr << "Failed to initialize application\n";
		status = EXIT_FAILURE;
		goto err_application_init;
	}

	while (!glfwWindowShouldClose(glfw_window)) {
		const double time = glfwGetTime();

		glfwPollEvents();
		int width, height;
		glfwGetFramebufferSize(glfw_window, &width, &height);
		if (width == 0 || height == 0) { glfwWaitEvents(); continue; }
		ImGui_ImplGlfw_NewFrame();

		ImGui::NewFrame();
		if (verify) application::verification_step(frame);
		application::update(time);
		ImGui::Render();

		graphics::internal::FrameData fd = graphics::internal::prepare();
		if (!fd.command_buffer) { status = EXIT_FAILURE; break; }
		try {
			application::render(fd);
		} catch (const std::exception& error) {
			std::cerr << error.what() << '\n';
			status = EXIT_FAILURE;
			break;
		}
		graphics::internal::submitAndPresent();
		if (verify && frame == 90) glfwSetWindowSize(glfw_window, 1024, 768);
		if (verify && frame == 150) glfwSetWindowSize(glfw_window, 1280, 720);
		if (verify && ++frame == 240) glfwSetWindowShouldClose(glfw_window, GLFW_TRUE);
		if (verify) std::this_thread::sleep_for(std::chrono::milliseconds(30));
	}

	application::shutdown();
err_application_init:
	graphics::internal::shutdown();
err_graphics_init:
	ImGui_ImplGlfw_Shutdown();
err_imgui_glfw_init:
	ImGui::DestroyContext();
err_imgui_init:
	glfwDestroyWindow(glfw_window);
err_null_window:
	glfwTerminate();

	return status;
}
