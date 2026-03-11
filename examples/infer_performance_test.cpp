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

struct cli_options {
    std::string model_path;
    float fill_value = 0.0f;
    int bench_iters = 1;
    std::string backend = "auto";
};

static void print_usage(const char * prog) {
    std::cerr << "Usage: " << prog << " -m <model.gguf> [-f fill_value] [-b betch_iters] [-d backend]" << std::endl;
    std::cerr << "  -m, --model         gguf model path + filename (required)" << std::endl;
    std::cerr << "  -f, --fill_value    input fill value (default: 0.0)" << std::endl;
    std::cerr << "  -b, --betch_iters   benchmark iterations (default: 1)" << std::endl;
    std::cerr << "      --bench_iters   alias of --betch_iters" << std::endl;
    std::cerr << "  -d, --backend       backend name (default: auto)" << std::endl;
}

static bool parse_cli_options(int argc, char ** argv, cli_options & opts, std::string & error, bool & show_help) {
    error.clear();
    show_help = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const std::string & opt, std::string & out) -> bool {
            if (i + 1 >= argc) {
                error = "missing value for option " + opt;
                return false;
            }
            out = argv[++i];
            return true;
        };

        if (arg == "-h" || arg == "--help") {
            show_help = true;
            return true;
        }
        if (arg == "-m" || arg == "--model") {
            if (!need_value(arg, opts.model_path)) {
                return false;
            }
            continue;
        }
        if (arg == "-f" || arg == "--fill_value") {
            std::string value;
            if (!need_value(arg, value)) {
                return false;
            }
            char * end = nullptr;
            const float parsed = std::strtof(value.c_str(), &end);
            if (end == value.c_str() || (end != nullptr && *end != '\0')) {
                error = "invalid float for " + arg + ": " + value;
                return false;
            }
            opts.fill_value = parsed;
            continue;
        }
        if (arg == "-b" || arg == "--betch_iters" || arg == "--bench_iters") {
            std::string value;
            if (!need_value(arg, value)) {
                return false;
            }
            char * end = nullptr;
            const long parsed = std::strtol(value.c_str(), &end, 10);
            if (end == value.c_str() || (end != nullptr && *end != '\0') || parsed <= 0) {
                error = "invalid positive integer for " + arg + ": " + value;
                return false;
            }
            opts.bench_iters = static_cast<int>(parsed);
            continue;
        }
        if (arg == "-d" || arg == "--backend") {
            if (!need_value(arg, opts.backend)) {
                return false;
            }
            continue;
        }

        error = "unknown option: " + arg;
        return false;
    }

    if (opts.model_path.empty()) {
        error = "missing required option --model/-m";
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    cli_options opts;
    std::string cli_error;
    bool show_help = false;
    if (!parse_cli_options(argc, argv, opts, cli_error, show_help)) {
        print_usage(argv[0]);
        std::cerr << "error: " << cli_error << std::endl;
        return 1;
    }
    if (show_help) {
        print_usage(argv[0]);
        return 0;
    }

    const size_t sep = opts.model_path.find_last_of("/\\");
    const std::string model_dir = sep == std::string::npos ? "." : opts.model_path.substr(0, sep);
    const std::string model_name = sep == std::string::npos ? opts.model_path : opts.model_path.substr(sep + 1);
    if (model_name.empty()) {
        std::cerr << "error: invalid --model path: " << opts.model_path << std::endl;
        return 1;
    }

    setenv("MINI2GGUF_BACKEND", opts.backend.c_str(), 1);

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

    std::vector<float> input(static_cast<size_t>(input_elements), opts.fill_value);
    std::vector<std::vector<float>> outputs;

    double avg_compute_ms = 0.0;
    if (!model.benchmark_compute(input, opts.bench_iters, avg_compute_ms)) {
        std::cerr << "benchmark failed: " << model.last_error() << std::endl;
        return 4;
    }
    std::cout << "benchmark set+compute avg over " << opts.bench_iters << " runs: " << avg_compute_ms << " ms" << std::endl;

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
