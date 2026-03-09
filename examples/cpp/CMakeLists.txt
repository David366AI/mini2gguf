add_executable(mini2gguf_infer_demo
    infer_demo.cpp
)

target_link_libraries(mini2gguf_infer_demo
    PRIVATE
        mini2gguf_runtime
)

add_executable(mini2gguf_yolo_dynamic_demo
    yolo_dynamic_demo.cpp
    ../../ggml/examples/yolo/yolo-image.cpp
)

target_include_directories(mini2gguf_yolo_dynamic_demo
    PRIVATE
    ../../ggml/examples
        ../../ggml/examples/yolo
)

target_link_libraries(mini2gguf_yolo_dynamic_demo
    PRIVATE
        mini2gguf_runtime
)
