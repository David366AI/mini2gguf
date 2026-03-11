cmake_minimum_required(VERSION 3.20)

project(mini2gguf LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MINI2GGUF_USE_VULKAN "Enable Vulkan backend via ggml" OFF)

set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if (MINI2GGUF_USE_VULKAN)
	set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(ggml)
add_subdirectory(runtime)
add_subdirectory(examples)
