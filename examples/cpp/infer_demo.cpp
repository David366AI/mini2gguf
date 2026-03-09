#include "mini2gguf/model_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

static double bytes_to_mb(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

static std::string dims_to_string(const std::vector<int64_t> & dims) {
    std::string out = "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        out += std::to_string(dims[i]);
        if (i + 1 < dims.size()) {
            out += " x ";
        }
    }
    out += "]";
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> <model_name> [fill_value] [bench_iters] [backend]" << std::endl;
        std::cerr << "Example: " << argv[0] << " assets/models/yolo/converted blank_16" << std::endl;
        return 1;
    }

    const std::string model_dir = argv[1];
    const std::string model_name = argv[2];
    const float fill_value = argc >= 4 ? std::strtof(argv[3], nullptr) : 0.0f;
    const int bench_iters = argc >= 5 ? std::max(1, std::atoi(argv[4])) : 1;
    const std::string backend = argc >= 6 ? argv[5] : "auto";

    setenv("MINI2GGUF_BACKEND", backend.c_str(), 1);

    mini2gguf::DynamicModel model;
    const auto load_start = std::chrono::steady_clock::now();
    if (!model.load(model_dir, model_name)) {
        std::cerr << "load failed: " << model.last_error() << std::endl;
        return 2;
    }
    const auto load_end = std::chrono::steady_clock::now();
    const double load_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();
    std::cout << "load time: " << load_ms << " ms" << std::endl;
    std::cout << "backend: " << model.backend_name() << std::endl;
    std::cout << "device: " << model.backend_device_name() << std::endl;

    const auto & inputs = model.input_tensors();
    const auto & outputs_info = model.output_tensors();
    std::cout << "inputs: " << inputs.size() << ", outputs: " << outputs_info.size() << std::endl;
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::cout << "input[" << i << "] " << inputs[i].name
                  << " type=" << inputs[i].data_type
                  << " shape=" << dims_to_string(inputs[i].dims) << std::endl;
    }
    for (size_t i = 0; i < outputs_info.size(); ++i) {
        std::cout << "output[" << i << "] " << outputs_info[i].name
                  << " type=" << outputs_info[i].data_type
                  << " shape=" << dims_to_string(outputs_info[i].dims) << std::endl;
    }

    std::cout << "weight buffer: " << model.weight_buffer_bytes() << " bytes ("
              << bytes_to_mb(model.weight_buffer_bytes()) << " MB)" << std::endl;

    const int64_t input_elements = model.input_elements();
    if (input_elements <= 0) {
        std::cerr << "invalid model input elements: " << input_elements << std::endl;
        return 3;
    }

    std::vector<float> input(static_cast<size_t>(input_elements), fill_value);
    std::vector<std::vector<float>> outputs;

    double avg_compute_ms = 0.0;
    if (!model.benchmark_compute(input, bench_iters, avg_compute_ms)) {
        std::cerr << "benchmark failed: " << model.last_error() << std::endl;
        return 4;
    }
    std::cout << "benchmark set+compute avg over " << bench_iters << " runs: " << avg_compute_ms << " ms" << std::endl;

    const auto infer_start = std::chrono::steady_clock::now();
    if (!model.infer_all(input, outputs)) {
        std::cerr << "infer failed: " << model.last_error() << std::endl;
        return 5;
    }
    const auto infer_end = std::chrono::steady_clock::now();
    const double infer_ms = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
    std::cout << "infer time: " << infer_ms << " ms" << std::endl;

    std::cout << "compute buffer(static ctx alloc): " << model.last_compute_buffer_bytes() << " bytes ("
              << bytes_to_mb(model.last_compute_buffer_bytes()) << " MB)" << std::endl;
    std::cout << "compute buffer(peak estimate): " << model.last_compute_peak_bytes() << " bytes ("
              << bytes_to_mb(model.last_compute_peak_bytes()) << " MB)" << std::endl;

    size_t total_elements = 0;
    for (const auto & output : outputs) {
        total_elements += output.size();
    }
    std::cout << "infer ok, outputs: " << outputs.size() << ", total output elements: " << total_elements << std::endl;
    for (size_t output_index = 0; output_index < outputs.size(); ++output_index) {
        const std::vector<float> & output = outputs[output_index];
        std::cout << "output[" << output_index << "] elements: " << output.size() << std::endl;
        // const size_t preview = std::min<size_t>(10, output.size());
        // for (size_t i = 0; i < preview; ++i) {
        //     std::cout << "out[" << output_index << "][" << i << "] = " << output[i] << std::endl;
        // }
    }

    model.unload();
    return 0;
}
