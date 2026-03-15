#include "mini2gguf/crnn_utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace mini2gguf {
namespace {

struct CrnnOutputLayout {
    int64_t batch = 0;
    int64_t timesteps = 0;
    int64_t classes = 0;
    bool ntc = true;
};

static inline size_t chw_index(int c, int y, int x, int h, int w) {
    return static_cast<size_t>(c) * static_cast<size_t>(h) * static_cast<size_t>(w) +
           static_cast<size_t>(y) * static_cast<size_t>(w) +
           static_cast<size_t>(x);
}

static float clamp01(float v) {
    return std::min(1.0f, std::max(0.0f, v));
}

static int64_t score_layout_candidate(const CrnnOutputLayout & cand, size_t total_size, size_t dict_size) {
    if (cand.batch <= 0 || cand.timesteps <= 0 || cand.classes <= 1) {
        return -1;
    }
    if (static_cast<size_t>(cand.batch) * static_cast<size_t>(cand.timesteps) * static_cast<size_t>(cand.classes) != total_size) {
        return -1;
    }

    int64_t score = 0;
    if (dict_size + 1 == static_cast<size_t>(cand.classes)) {
        score += 6;
    } else if (dict_size == static_cast<size_t>(cand.classes)) {
        score += 5;
    }
    if (cand.timesteps >= cand.classes) {
        score += 2;
    }
    if (cand.classes < 512) {
        score += 1;
    }
    return score;
}

static bool infer_crnn_output_layout(
    const std::vector<float> & output,
    const DynamicModel::TensorInfo & output_info,
    size_t dict_size,
    CrnnOutputLayout & layout) {
    const std::vector<int64_t> & dims = output_info.dims;
    std::vector<CrnnOutputLayout> candidates;

    if (dims.size() == 3) {
        const int64_t b = dims[0];
        candidates.push_back({b, dims[1], dims[2], true});
        candidates.push_back({b, dims[2], dims[1], false});
    } else if (dims.size() == 2) {
        candidates.push_back({1, dims[0], dims[1], true});
        candidates.push_back({1, dims[1], dims[0], false});
    }

    int64_t best_score = -1;
    for (const CrnnOutputLayout & cand : candidates) {
        const int64_t score = score_layout_candidate(cand, output.size(), dict_size);
        if (score > best_score) {
            best_score = score;
            layout = cand;
        }
    }
    if (best_score >= 0) {
        return true;
    }

    if (dict_size == 0 || output.empty()) {
        return false;
    }
    const int64_t classes = static_cast<int64_t>(dict_size + 1);
    if (classes <= 1 || (output.size() % static_cast<size_t>(classes)) != 0) {
        return false;
    }
    layout.batch = 1;
    layout.classes = classes;
    layout.timesteps = static_cast<int64_t>(output.size() / static_cast<size_t>(classes));
    layout.ntc = true;
    return layout.timesteps > 0;
}

static int argmax_class(
    const std::vector<float> & output,
    const CrnnOutputLayout & layout,
    int64_t batch_index,
    int64_t timestep_index) {
    int best_class = 0;
    float best_value = -std::numeric_limits<float>::infinity();
    for (int64_t c = 0; c < layout.classes; ++c) {
        size_t idx = 0;
        if (layout.ntc) {
            idx = static_cast<size_t>(batch_index * layout.timesteps * layout.classes +
                                      timestep_index * layout.classes + c);
        } else {
            idx = static_cast<size_t>(batch_index * layout.classes * layout.timesteps +
                                      c * layout.timesteps + timestep_index);
        }
        const float v = output[idx];
        if (c == 0 || v > best_value) {
            best_value = v;
            best_class = static_cast<int>(c);
        }
    }
    return best_class;
}

static std::string decode_word_python_compatible(
    const std::vector<int> & word,
    const std::vector<std::string> & dict,
    int blank_index) {
    std::string out;
    if (word.empty() || dict.empty()) {
        return out;
    }

    const int last = word.back();
    for (size_t i = 0; i < word.size(); ++i) {
        const int index = word[i];

        if (word.size() >= 3 && i == 0 && word[1] == 0 && word[0] == word[2] && last != -1) {
            continue;
        }
        if (i + 1 < word.size() && word[i] == word[i + 1] && last != -1) {
            continue;
        }
        if (index == blank_index) {
            continue;
        }

        const int mapped = index - 1;
        if (mapped < 0 || mapped >= static_cast<int>(dict.size())) {
            continue;
        }
        out += dict[static_cast<size_t>(mapped)];
    }
    return out;
}

} // namespace

bool parse_crnn_dict_metadata(
    const std::string & dict_metadata,
    std::vector<std::string> & dict) {
    dict.clear();

    if (dict_metadata.empty()) {
        return false;
    }

    // Backward compatibility: older models stored dict as newline-separated lines.
    if (dict_metadata.find('\n') != std::string::npos || dict_metadata.find('\r') != std::string::npos) {
        std::istringstream iss(dict_metadata);
        std::string line;
        while (std::getline(iss, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty()) {
                continue;
            }
            dict.push_back(line);
        }
        return !dict.empty();
    }

    // Current format: model.dict is a plain concatenated UTF-8 character string.
    for (size_t i = 0; i < dict_metadata.size();) {
        const unsigned char c = static_cast<unsigned char>(dict_metadata[i]);
        size_t cp_len = 1;
        if ((c & 0x80u) == 0x00u) {
            cp_len = 1;
        } else if ((c & 0xE0u) == 0xC0u) {
            cp_len = 2;
        } else if ((c & 0xF0u) == 0xE0u) {
            cp_len = 3;
        } else if ((c & 0xF8u) == 0xF0u) {
            cp_len = 4;
        }

        if (i + cp_len > dict_metadata.size()) {
            cp_len = 1;
        }
        dict.push_back(dict_metadata.substr(i, cp_len));
        i += cp_len;
    }
    return !dict.empty();
}

bool preprocess_crnn_input(
    const float * image_chw,
    int image_w,
    int image_h,
    int image_c,
    const DynamicModel::TensorInfo & input_info,
    std::vector<float> & input_tensor,
    std::string & error,
    const CrnnPreprocessOptions & options) {
    input_tensor.clear();
    error.clear();

    if (image_chw == nullptr) {
        error = "CRNN preprocess image buffer is null";
        return false;
    }
    if (image_w <= 0 || image_h <= 0) {
        error = "CRNN preprocess image shape is invalid";
        return false;
    }
    if (image_c != 1 && image_c != 3) {
        error = "CRNN preprocess only supports input image channels 1 or 3";
        return false;
    }
    if (std::abs(options.pixel_std) < 1e-12f) {
        error = "CRNN preprocess pixel_std must be non-zero";
        return false;
    }
    if (input_info.dims.size() < 4) {
        error = "CRNN preprocess requires model input rank >= 4";
        return false;
    }

    const int64_t batch = input_info.dims[0];
    const int64_t target_c64 = input_info.dims[1];
    const int64_t target_h64 = input_info.dims[2];
    const int64_t target_w64 = input_info.dims[3];
    if (batch != 1 || target_c64 <= 0 || target_h64 <= 0 || target_w64 <= 0) {
        error = "CRNN preprocess requires fixed input shape [1,C,H,W]";
        return false;
    }

    const int target_c = static_cast<int>(target_c64);
    const int target_h = static_cast<int>(target_h64);
    const int target_w = static_cast<int>(target_w64);
    if (target_c != 1 && target_c != 3) {
        error = "CRNN preprocess supports model input channels C=1 or C=3";
        return false;
    }

    const int resized_w = std::max(1, static_cast<int>((static_cast<double>(image_w) / static_cast<double>(image_h)) * target_h));
    const int copy_w = std::min(resized_w, target_w);

    const float pad_pixel = clamp01(options.pad_value);
    const float pad_norm = (pad_pixel - options.pixel_mean) / options.pixel_std;
    input_tensor.assign(static_cast<size_t>(target_c) * static_cast<size_t>(target_h) * static_cast<size_t>(target_w), pad_norm);

    auto read_src = [&](int c, int y, int x) -> float {
        return image_chw[chw_index(c, y, x, image_h, image_w)];
    };

    auto fetch_target_pixel = [&](int target_channel, int y, int x) -> float {
        if (target_c == 1) {
            if (image_c == 1) {
                return read_src(0, y, x);
            }
            const float r = read_src(0, y, x);
            const float g = read_src(1, y, x);
            const float b = read_src(2, y, x);
            return 0.299f * r + 0.587f * g + 0.114f * b;
        }

        if (image_c == 1) {
            return read_src(0, y, x);
        }
        return read_src(target_channel, y, x);
    };

    for (int c = 0; c < target_c; ++c) {
        for (int y = 0; y < target_h; ++y) {
            const float src_y = ((static_cast<float>(y) + 0.5f) * static_cast<float>(image_h) / static_cast<float>(target_h)) - 0.5f;
            const int y0 = std::max(0, std::min(image_h - 1, static_cast<int>(std::floor(src_y))));
            const int y1 = std::max(0, std::min(image_h - 1, y0 + 1));
            const float dy = std::max(0.0f, std::min(1.0f, src_y - static_cast<float>(y0)));

            for (int x = 0; x < copy_w; ++x) {
                const float src_x = ((static_cast<float>(x) + 0.5f) * static_cast<float>(image_w) / static_cast<float>(resized_w)) - 0.5f;
                const int x0 = std::max(0, std::min(image_w - 1, static_cast<int>(std::floor(src_x))));
                const int x1 = std::max(0, std::min(image_w - 1, x0 + 1));
                const float dx = std::max(0.0f, std::min(1.0f, src_x - static_cast<float>(x0)));

                const float v00 = fetch_target_pixel(c, y0, x0);
                const float v01 = fetch_target_pixel(c, y0, x1);
                const float v10 = fetch_target_pixel(c, y1, x0);
                const float v11 = fetch_target_pixel(c, y1, x1);
                const float top = v00 + (v01 - v00) * dx;
                const float bot = v10 + (v11 - v10) * dx;
                const float v = top + (bot - top) * dy;

                input_tensor[chw_index(c, y, x, target_h, target_w)] = (v - options.pixel_mean) / options.pixel_std;
            }
        }
    }

    return true;
}

bool postprocess_crnn_outputs(
    const std::vector<std::vector<float>> & outputs,
    const std::vector<DynamicModel::TensorInfo> & output_infos,
    const std::vector<std::string> & dict,
    std::vector<std::string> & texts,
    std::string & error,
    const CrnnPostprocessOptions & options) {
    texts.clear();
    error.clear();

    if (outputs.empty() || output_infos.empty()) {
        error = "CRNN postprocess expects at least 1 output tensor";
        return false;
    }
    if (dict.empty()) {
        error = "CRNN dict is empty";
        return false;
    }

    const std::vector<float> & output = outputs[0];
    if (output.empty()) {
        error = "CRNN output[0] is empty";
        return false;
    }

    CrnnOutputLayout layout;
    if (!infer_crnn_output_layout(output, output_infos[0], dict.size(), layout)) {
        error = "unsupported CRNN output layout for output[0]";
        return false;
    }

    texts.resize(static_cast<size_t>(layout.batch));
    for (int64_t b = 0; b < layout.batch; ++b) {
        std::vector<int> word;
        word.reserve(static_cast<size_t>(layout.timesteps));
        for (int64_t t = 0; t < layout.timesteps; ++t) {
            word.push_back(argmax_class(output, layout, b, t));
        }
        texts[static_cast<size_t>(b)] = decode_word_python_compatible(word, dict, options.blank_index);
    }

    return true;
}

bool postprocess_crnn_outputs(
    const std::vector<std::vector<float>> & outputs,
    const std::vector<DynamicModel::TensorInfo> & output_infos,
    const std::vector<std::string> & dict,
    std::string & text,
    std::string & error,
    const CrnnPostprocessOptions & options) {
    text.clear();
    std::vector<std::string> texts;
    if (!postprocess_crnn_outputs(outputs, output_infos, dict, texts, error, options)) {
        return false;
    }
    if (!texts.empty()) {
        text = texts[0];
    }
    return true;
}

} // namespace mini2gguf
