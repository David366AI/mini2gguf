#pragma once

#include "mini2gguf/model_runtime.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mini2gguf {

struct YoloBox {
    float x = 0.0f;
    float y = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
};

struct YoloDetection {
    YoloBox bbox;
    std::vector<float> prob;
    float objectness = 0.0f;
    std::vector<uint8_t> mask;
    int mask_w = 0;
    int mask_h = 0;
};

struct YoloPostprocessOptions {
    int image_w = 0;
    int image_h = 0;
    int net_w = 0;
    int net_h = 0;
    int model_version = -1;
    float conf_thres = 0.25f;
    float iou_thres = 0.45f;
    bool agnostic_nms = false;
};

bool postprocess_yolo_outputs(
    const std::vector<std::vector<float>> & outputs,
    const std::vector<DynamicModel::TensorInfo> & output_infos,
    const YoloPostprocessOptions & options,
    std::vector<YoloDetection> & detections,
    std::string & error);

} // namespace mini2gguf
