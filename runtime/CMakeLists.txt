add_library(mini2gguf_runtime
    src/model_runtime.cpp
    src/yolo_utils.cpp
)

target_include_directories(mini2gguf_runtime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mini2gguf_runtime
    PUBLIC
        ggml::ggml
)
