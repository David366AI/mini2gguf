#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;
struct ggml_backend;
struct ggml_backend_buffer;

namespace mini2gguf {

class DynamicModel {
public:
    struct TensorInfo {
        std::string name;
        std::string data_type;
        std::vector<int64_t> dims;
    };

    DynamicModel();
    ~DynamicModel();

    DynamicModel(const DynamicModel &) = delete;
    DynamicModel & operator=(const DynamicModel &) = delete;

    bool load(const std::string & model_dir, const std::string & model_name);
    bool infer(const std::vector<float> & input, std::vector<float> & output);
    bool infer_all(const std::vector<float> & input, std::vector<std::vector<float>> & outputs);
    bool benchmark_compute(const std::vector<float> & input, int repeats, double & avg_ms);
    void unload();
    int64_t input_elements() const;
    size_t weight_buffer_bytes() const;
    size_t last_compute_buffer_bytes() const;
    size_t last_compute_peak_bytes() const;
    const std::vector<TensorInfo> & input_tensors() const;
    const std::vector<TensorInfo> & output_tensors() const;
    const std::unordered_map<std::string, std::string> & model_metadata() const;
    std::string model_metadata_value(const std::string & key, const std::string & fallback = "") const;
    const std::string & backend_name() const;
    const std::string & backend_device_name() const;

    bool is_loaded() const;
    const std::string & last_error() const;

    struct NodeDef {
        std::string name;
        std::string op_type;
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;

        int axis = 0;
        std::vector<int64_t> axes;
        int keepdims = 1;
        std::vector<int64_t> split;
        std::vector<int> perm;
        std::vector<int> strides;
        std::vector<int> dilations;
        std::vector<int> pads;
        std::vector<int> kernel_shape;
        std::vector<int> output_padding;
        std::vector<int64_t> output_shape;
        std::string auto_pad;
        int group = 1;

        float alpha = 0.01f;
        float epsilon = 1e-5f;
        float momentum = 0.9f;
        int to = 0;
        int largest = 1;
        int sorted = 1;
        int fmod = 0;
        int ceil_mode = 0;
        int count_include_pad = 0;

        std::string const_value_name;
    };

    struct HostTensor {
        int type = 0;
        std::vector<int64_t> ne;
        std::vector<uint8_t> bytes;
    };

private:

    bool load_graph_json(const std::string & graph_path);
    bool load_graph_json_text(const std::string & graph_json, const std::string & source_name);
    bool load_weights_gguf(const std::string & weights_path);
    bool build_compute_graph();

    bool build_and_run_graph(const std::vector<float> & input, std::vector<std::vector<float>> & outputs);

    bool set_error(const std::string & message);

private:
    std::string model_dir_;
    std::string model_name_;
    std::string last_error_;
    std::string backend_name_;
    std::string backend_device_name_;

    bool loaded_ = false;

    ggml_backend * backend_ = nullptr;
    ggml_context * weight_ctx_ = nullptr;
    ggml_backend_buffer * weight_buffer_ = nullptr;

    std::vector<TensorInfo> graph_inputs_;
    std::vector<TensorInfo> graph_outputs_;
    std::vector<TensorInfo> graph_initializers_;
    std::vector<NodeDef> nodes_;
    std::unordered_map<std::string, std::string> model_metadata_;

    std::unordered_map<std::string, ggml_tensor *> weight_tensors_;
    std::unordered_map<std::string, int> tensor_rank_by_name_;

    ggml_context * compute_ctx_ = nullptr;
    ggml_cgraph * compute_graph_ = nullptr;
    void * compute_allocr_ = nullptr;
    ggml_tensor * input_tensor_ = nullptr;
    std::vector<ggml_tensor *> output_tensors_;
    std::vector<std::string> node_output_names_;
    std::vector<ggml_tensor *> node_output_tensors_;
    std::vector<std::string> debug_dump_names_;
    std::vector<ggml_tensor *> debug_dump_tensors_;

    size_t weight_buffer_bytes_ = 0;
    size_t last_compute_buffer_bytes_ = 0;
    size_t last_compute_peak_bytes_ = 0;
};

} // namespace mini2gguf
