add_executable(infer_performance_test
    infer_performance_test.cpp
)

target_link_libraries(infer_performance_test
    PRIVATE
        mini2gguf_runtime
)

add_executable(yolo_demo
    yolo_demo.cpp
    ../ggml/examples/yolo/yolo-image.cpp
)

target_include_directories(yolo_demo
    PRIVATE
        ../ggml/examples
        ../ggml/examples/yolo
)

target_link_libraries(yolo_demo
    PRIVATE
        mini2gguf_runtime
)
