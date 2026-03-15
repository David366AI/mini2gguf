#pragma once

#include "mini2gguf/model_runtime.hpp"

#include <string>
#include <vector>

namespace mini2gguf {

struct CrnnPreprocessOptions {
    float pixel_mean = 0.9252072f;
    float pixel_std = 0.20963529f;
    float pad_value = 1.0f;
};

struct CrnnPostprocessOptions {
    int blank_index = 0;
};

bool parse_crnn_dict_metadata(
    const std::string & dict_metadata,
    std::vector<std::string> & dict);

bool preprocess_crnn_input(
    const float * image_chw,
    int image_w,
    int image_h,
    int image_c,
    const DynamicModel::TensorInfo & input_info,
    std::vector<float> & input_tensor,
    std::string & error,
    const CrnnPreprocessOptions & options = {});

bool postprocess_crnn_outputs(
    const std::vector<std::vector<float>> & outputs,
    const std::vector<DynamicModel::TensorInfo> & output_infos,
    const std::vector<std::string> & dict,
    std::string & text,
    std::string & error,
    const CrnnPostprocessOptions & options = {});

bool postprocess_crnn_outputs(
    const std::vector<std::vector<float>> & outputs,
    const std::vector<DynamicModel::TensorInfo> & output_infos,
    const std::vector<std::string> & dict,
    std::vector<std::string> & texts,
    std::string & error,
    const CrnnPostprocessOptions & options = {});

} // namespace mini2gguf
