#include "mini2gguf/model_runtime.hpp"

#include "yolo-image.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

struct box {
    float x, y, w, h;
};

struct detection {
    box bbox;
    std::vector<float> prob;
    float objectness;
};

struct yolo_layer {
    int classes = 80;
    std::vector<int> mask;
    std::vector<float> anchors;
    std::vector<float> predictions;
    int w;
    int h;

    int entry_index(int location, int entry) const {
        int n = location / (w * h);
        int loc = location % (w * h);
        return n * w * h * (4 + classes + 1) + entry * w * h + loc;
    }
};

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

static void print_output_stats(const std::string & name, const std::vector<float> & v) {
    if (v.empty()) {
        std::cout << "output-stats " << name << ": empty" << std::endl;
        return;
    }

    float vmin = std::numeric_limits<float>::infinity();
    float vmax = -std::numeric_limits<float>::infinity();
    double sum = 0.0;
    for (float x : v) {
        vmin = std::min(vmin, x);
        vmax = std::max(vmax, x);
        sum += x;
    }
    std::cout << "output-stats " << name
              << ": n=" << v.size()
              << " min=" << vmin
              << " max=" << vmax
              << " mean=" << (sum / static_cast<double>(v.size()))
              << std::endl;
}

static void print_output_values(const std::string & name, const std::vector<float> & v, size_t limit) {
    std::cout << "output-values " << name << ": [";
    const size_t n = std::min(limit, v.size());
    for (size_t i = 0; i < n; ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << v[i];
    }
    if (v.size() > n) {
        std::cout << ", ...";
    }
    std::cout << "]" << std::endl;
}

static std::string to_lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

static int parse_leading_int(const std::string & text, int fallback) {
    if (text.empty()) {
        return fallback;
    }
    size_t i = 0;
    bool negative = false;
    if (text[i] == '+' || text[i] == '-') {
        negative = (text[i] == '-');
        ++i;
    }
    size_t begin = i;
    while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
        ++i;
    }
    if (i == begin) {
        return fallback;
    }
    int value = std::atoi(text.substr(begin, i - begin).c_str());
    return negative ? -value : value;
}

static int infer_yolo_version_from_name(const std::string & model_name) {
    const std::string lower = to_lower_copy(model_name);
    const size_t pos = lower.find("yolo");
    if (pos == std::string::npos) {
        return -1;
    }
    const size_t begin = pos + 4;
    size_t end = begin;
    while (end < lower.size() && std::isdigit(static_cast<unsigned char>(lower[end]))) {
        ++end;
    }
    if (end == begin) {
        return -1;
    }
    return std::atoi(lower.substr(begin, end - begin).c_str());
}

static std::string detect_model_family(const mini2gguf::DynamicModel & model, const std::string & model_name) {
    std::string family = model.model_metadata_value("model.family");
    if (family.empty()) {
        family = model.model_metadata_value("family");
    }
    if (family.empty() && to_lower_copy(model_name).find("yolo") != std::string::npos) {
        family = "yolo";
    }
    return to_lower_copy(family);
}

static int detect_model_version(const mini2gguf::DynamicModel & model, const std::string & model_name) {
    std::string version = model.model_metadata_value("model.version");
    int parsed = parse_leading_int(version, -1);
    if (parsed >= 0) {
        return parsed;
    }

    return infer_yolo_version_from_name(model_name);
}


static bool load_labels(const std::string & filename, std::vector<std::string> & labels) {
    std::ifstream file_in(filename);
    if (!file_in) {
        return false;
    }
    std::string line;
    while (std::getline(file_in, line)) {
        labels.push_back(line);
    }
    return !labels.empty();
}

static bool load_alphabet(const std::string & data_dir, std::vector<yolo_image> & alphabet) {
    alphabet.resize(8 * 128);
    for (int j = 0; j < 8; j++) {
        for (int i = 32; i < 127; i++) {
            char fname[512];
            std::snprintf(fname, sizeof(fname), "%s/labels/%d_%d.png", data_dir.c_str(), i, j);
            if (!load_image(fname, alphabet[j * 128 + i])) {
                std::cerr << "Cannot load '" << fname << "'" << std::endl;
                return false;
            }
        }
    }
    return true;
}

static void activate_array(float * x, const int n) {
    for (int i = 0; i < n; i++) {
        x[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
}

static void apply_yolo(yolo_layer & layer) {
    int w = layer.w;
    int h = layer.h;
    int n_masks = static_cast<int>(layer.mask.size());
    float * data = layer.predictions.data();
    for (int n = 0; n < n_masks; n++) {
        int index = layer.entry_index(n * w * h, 0);
        activate_array(data + index, 2 * w * h);
        index = layer.entry_index(n * w * h, 4);
        activate_array(data + index, (1 + layer.classes) * w * h);
    }
}

static box get_yolo_box(const yolo_layer & layer, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride) {
    const float * predictions = layer.predictions.data();
    box b;
    b.x = (i + predictions[index + 0 * stride]) / lw;
    b.y = (j + predictions[index + 1 * stride]) / lh;
    b.w = std::exp(predictions[index + 2 * stride]) * layer.anchors[2 * n] / w;
    b.h = std::exp(predictions[index + 3 * stride]) * layer.anchors[2 * n + 1] / h;
    return b;
}

static void correct_yolo_box(box & b, int im_w, int im_h, int net_w, int net_h) {
    int new_w = 0;
    int new_h = 0;
    if (((float) net_w / im_w) < ((float) net_h / im_h)) {
        new_w = net_w;
        new_h = (im_h * net_w) / im_w;
    } else {
        new_h = net_h;
        new_w = (im_w * net_h) / im_h;
    }
    b.x = (b.x - (net_w - new_w) / 2.0f / net_w) / ((float) new_w / net_w);
    b.y = (b.y - (net_h - new_h) / 2.0f / net_h) / ((float) new_h / net_h);
    b.w *= (float) net_w / new_w;
    b.h *= (float) net_h / new_h;
}

static void get_yolo_detections(const yolo_layer & layer, std::vector<detection> & detections, int im_w, int im_h, int netw, int neth, float thresh) {
    int w = layer.w;
    int h = layer.h;
    int n_masks = static_cast<int>(layer.mask.size());
    const float * predictions = layer.predictions.data();
    for (int i = 0; i < w * h; i++) {
        for (int n = 0; n < n_masks; n++) {
            int obj_index = layer.entry_index(n * w * h + i, 4);
            float objectness = predictions[obj_index];
            if (objectness <= thresh) {
                continue;
            }
            detection det;
            int box_index = layer.entry_index(n * w * h + i, 0);
            int row = i / w;
            int col = i % w;
            det.bbox = get_yolo_box(layer, layer.mask[n], box_index, col, row, w, h, netw, neth, w * h);
            correct_yolo_box(det.bbox, im_w, im_h, netw, neth);
            det.objectness = objectness;
            det.prob.resize(layer.classes);
            for (int j = 0; j < layer.classes; j++) {
                int class_index = layer.entry_index(n * w * h + i, 4 + 1 + j);
                float prob = objectness * predictions[class_index];
                det.prob[j] = (prob > thresh) ? prob : 0;
            }
            detections.push_back(det);
        }
    }
}

static float overlap(float x1, float w1, float x2, float w2) {
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

static float box_intersection(const box & a, const box & b) {
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0) {
        return 0;
    }
    return w * h;
}

static float box_union(const box & a, const box & b) {
    float i = box_intersection(a, b);
    return a.w * a.h + b.w * b.h - i;
}

static float box_iou(const box & a, const box & b) {
    return box_intersection(a, b) / box_union(a, b);
}

static float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static void scale_box_xyxy_from_net_to_img(
    float & x1,
    float & y1,
    float & x2,
    float & y2,
    int net_w,
    int net_h,
    int img_w,
    int img_h) {
    const float gain = std::min(
        static_cast<float>(net_h) / static_cast<float>(img_h),
        static_cast<float>(net_w) / static_cast<float>(img_w));
    const float pad_x = (static_cast<float>(net_w) - static_cast<float>(img_w) * gain) * 0.5f;
    const float pad_y = (static_cast<float>(net_h) - static_cast<float>(img_h) * gain) * 0.5f;

    x1 = (x1 - pad_x) / gain;
    y1 = (y1 - pad_y) / gain;
    x2 = (x2 - pad_x) / gain;
    y2 = (y2 - pad_y) / gain;

    x1 = clampf(x1, 0.0f, static_cast<float>(img_w - 1));
    y1 = clampf(y1, 0.0f, static_cast<float>(img_h - 1));
    x2 = clampf(x2, 0.0f, static_cast<float>(img_w - 1));
    y2 = clampf(y2, 0.0f, static_cast<float>(img_h - 1));
}

static box xyxy_to_cxcywh(float x1, float y1, float x2, float y2, int img_w, int img_h) {
    box b;
    const float cx = 0.5f * (x1 + x2);
    const float cy = 0.5f * (y1 + y2);
    const float w = std::max(0.0f, x2 - x1);
    const float h = std::max(0.0f, y2 - y1);

    b.x = cx / static_cast<float>(img_w);
    b.y = cy / static_cast<float>(img_h);
    b.w = w / static_cast<float>(img_w);
    b.h = h / static_cast<float>(img_h);
    return b;
}

static void do_nms_sort(std::vector<detection> & dets, int classes, float thresh) {
    int k = static_cast<int>(dets.size()) - 1;
    for (int i = 0; i <= k; ++i) {
        if (dets[i].objectness == 0) {
            std::swap(dets[i], dets[k]);
            --k;
            --i;
        }
    }
    int total = k + 1;
    for (int c = 0; c < classes; ++c) {
        std::sort(dets.begin(), dets.begin() + total, [=](const detection & a, const detection & b) {
            return a.prob[c] > b.prob[c];
        });
        for (int i = 0; i < total; ++i) {
            if (dets[i].prob[c] == 0) {
                continue;
            }
            box a = dets[i].bbox;
            for (int j = i + 1; j < total; ++j) {
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh) {
                    dets[j].prob[c] = 0;
                }
            }
        }
    }
}

static float get_color(int c, int x, int max) {
    float colors[6][3] = {{1, 0, 1}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}};
    float ratio = ((float) x / max) * 5;
    int i = static_cast<int>(std::floor(ratio));
    int j = static_cast<int>(std::ceil(ratio));
    ratio -= i;
    return (1 - ratio) * colors[i][c] + ratio * colors[j][c];
}

static void draw_detections(yolo_image & im, const std::vector<detection> & dets, float thresh, const std::vector<std::string> & labels, const std::vector<yolo_image> & alphabet) {
    int classes = static_cast<int>(labels.size());
    for (int i = 0; i < static_cast<int>(dets.size()); i++) {
        std::string labelstr;
        int cl = -1;
        for (int j = 0; j < static_cast<int>(dets[i].prob.size()); j++) {
            if (dets[i].prob[j] > thresh) {
                if (cl < 0) {
                    labelstr = labels[j];
                    cl = j;
                } else {
                    labelstr += ", ";
                    labelstr += labels[j];
                }
                std::printf("%s: %.0f%%\n", labels[j].c_str(), dets[i].prob[j] * 100);
            }
        }
        if (cl >= 0) {
            int width = im.h * .006;
            int offset = cl * 123457 % classes;
            float red = get_color(2, offset, classes);
            float green = get_color(1, offset, classes);
            float blue = get_color(0, offset, classes);
            float rgb[3];

            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
            box b = dets[i].bbox;

            int left = (b.x - b.w / 2.) * im.w;
            int right = (b.x + b.w / 2.) * im.w;
            int top = (b.y - b.h / 2.) * im.h;
            int bot = (b.y + b.h / 2.) * im.h;

            if (left < 0) left = 0;
            if (right > im.w - 1) right = im.w - 1;
            if (top < 0) top = 0;
            if (bot > im.h - 1) bot = im.h - 1;

            draw_box_width(im, left, top, right, bot, width, red, green, blue);
            yolo_image label = get_label(alphabet, labelstr, (im.h * .03));
            draw_label(im, top + width, left, label, rgb);
        }
    }
}

static bool postprocess_yolo_v4(
    const std::vector<std::vector<float>> & outputs,
    int img_w,
    int img_h,
    int net_w,
    int net_h,
    float thresh,
    std::vector<detection> & detections,
    std::string & error) {
    if (outputs.size() < 2) {
        error = "YOLO <= v4 postprocess expects at least 2 outputs";
        return false;
    }

    yolo_layer yolo16{80, {3, 4, 5}, {10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319}, outputs[0], 13, 13};
    yolo_layer yolo23{80, {0, 1, 2}, {10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319}, outputs[1], 26, 26};

    apply_yolo(yolo16);
    apply_yolo(yolo23);

    detections.clear();
    get_yolo_detections(yolo16, detections, img_w, img_h, net_w, net_h, thresh);
    get_yolo_detections(yolo23, detections, img_w, img_h, net_w, net_h, thresh);
    do_nms_sort(detections, 80, 0.45f);
    return true;
}

static bool postprocess_yolo_v26(
    const std::vector<std::vector<float>> & outputs,
    int img_w,
    int img_h,
    int net_w,
    int net_h,
    float conf_thresh,
    std::vector<detection> & detections,
    std::string & error) {
    if (outputs.empty()) {
        error = "YOLO v26 postprocess expects at least 1 output";
        return false;
    }

    const std::vector<float> & pred = outputs[0];
    if (pred.empty() || (pred.size() % 6) != 0) {
        error = "YOLO v26 output must be Kx6";
        return false;
    }

    constexpr int classes = 80;
    constexpr float nms_iou = 0.25f;
    const size_t k = pred.size() / 6;

    detections.clear();
    detections.reserve(k);
    size_t kept_conf = 0;
    float max_score = -1.0f;
    float min_score = 1e9f;

    for (size_t i = 0; i < k; ++i) {
        const float x1_raw = pred[i * 6 + 0];
        const float y1_raw = pred[i * 6 + 1];
        const float x2_raw = pred[i * 6 + 2];
        const float y2_raw = pred[i * 6 + 3];
        const float score = pred[i * 6 + 4];
        const int cls = static_cast<int>(std::lround(pred[i * 6 + 5]));
        max_score = std::max(max_score, score);
        min_score = std::min(min_score, score);

        if (score < conf_thresh) {
            continue;
        }
        ++kept_conf;
        if (cls < 0 || cls >= classes) {
            continue;
        }

        float x1 = x1_raw;
        float y1 = y1_raw;
        float x2 = x2_raw;
        float y2 = y2_raw;

        // Some exports may output cx,cy,w,h. Auto-convert only when obvious.
        if (x2 < x1 || y2 < y1) {
            const float cx = x1_raw;
            const float cy = y1_raw;
            const float w = std::max(0.0f, x2_raw);
            const float h = std::max(0.0f, y2_raw);
            x1 = cx - 0.5f * w;
            y1 = cy - 0.5f * h;
            x2 = cx + 0.5f * w;
            y2 = cy + 0.5f * h;
        }

        scale_box_xyxy_from_net_to_img(x1, y1, x2, y2, net_w, net_h, img_w, img_h);
        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        detection det;
        det.bbox = xyxy_to_cxcywh(x1, y1, x2, y2, img_w, img_h);
        det.objectness = score;
        det.prob.assign(classes, 0.0f);
        det.prob[cls] = score;
        detections.push_back(std::move(det));
    }

    std::cout << "yolo26 postprocess: layout=[x1,y1,x2,y2,score,cls], conf_keep=" << kept_conf
              << ", pre_nms=" << detections.size()
              << ", score_range=[" << min_score << "," << max_score << "]"
              << std::endl;

    do_nms_sort(detections, classes, nms_iou);
    std::cout << "yolo26 postprocess: post_nms=" << detections.size() << std::endl;
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> <model_name> <image_path> [output_path] [thresh] [backend]" << std::endl;
        return 1;
    }

    const std::string model_dir = argv[1];
    const std::string model_name = argv[2];
    const std::string image_path = argv[3];
    const std::string output_path = argc >= 5 ? argv[4] : "predictions_dynamic.jpg";
    const float thresh = argc >= 6 ? std::strtof(argv[5], nullptr) : 0.5f;
    const std::string backend = argc >= 7 ? argv[6] : "auto";

    setenv("MINI2GGUF_BACKEND", backend.c_str(), 1);

    mini2gguf::DynamicModel model;
    if (!model.load(model_dir, model_name)) {
        std::cerr << "load failed: " << model.last_error() << std::endl;
        return 2;
    }

    std::cout << "backend: " << model.backend_name() << std::endl;
    std::cout << "device: " << model.backend_device_name() << std::endl;

    const auto & input_infos = model.input_tensors();
    const auto & output_infos = model.output_tensors();

    // print input and output tensor info
    
    std::cout << "inputs: " << input_infos.size() << ", outputs: " << output_infos.size() << std::endl;
    for (size_t i = 0; i < input_infos.size(); ++i) {
        std::cout << "input[" << i << "] " << input_infos[i].name
                  << " type=" << input_infos[i].data_type
                  << " shape=" << dims_to_string(input_infos[i].dims) << std::endl;
    }
    for (size_t i = 0; i < output_infos.size(); ++i) {
        std::cout << "output[" << i << "] " << output_infos[i].name
                  << " type=" << output_infos[i].data_type
                  << " shape=" << dims_to_string(output_infos[i].dims) << std::endl;
    }
    
    if (input_infos.empty() || output_infos.size() < 1) {
        std::cerr << "unexpected model IO layout, require 1 input and at least 1 output" << std::endl;
        return 3;
    }


    int net_w = 416;
    int net_h = 416;
    if (input_infos[0].dims.size() >= 4) {
        net_h = static_cast<int>(input_infos[0].dims[2]);
        net_w = static_cast<int>(input_infos[0].dims[3]);
    }

    yolo_image img;
    if (!load_image(image_path.c_str(), img)) {
        std::cerr << "failed to load image: " << image_path << std::endl;
        return 4;
    }
    yolo_image sized = letterbox_image(img, net_w, net_h);

    std::vector<float> input = sized.data;
    std::vector<std::vector<float>> outputs;
    if (!model.infer_all(input, outputs)) {
        std::cerr << "infer failed: " << model.last_error() << std::endl;
        return 5;
    }
    const bool debug_output_stats = std::getenv("MINI2GGUF_DEBUG_OUTPUT_STATS") != nullptr;
    const bool debug_output_values = std::getenv("MINI2GGUF_DEBUG_OUTPUT_VALUES") != nullptr;
    size_t debug_output_values_n = 10;
    if (const char *v = std::getenv("MINI2GGUF_DEBUG_OUTPUT_VALUES_N")) {
        const int parsed = std::atoi(v);
        if (parsed > 0) {
            debug_output_values_n = static_cast<size_t>(parsed);
        }
    }
    if (debug_output_stats) {
        for (size_t i = 0; i < outputs.size(); ++i) {
            const std::string & out_name = i < output_infos.size() ? output_infos[i].name : std::string("output_") + std::to_string(i);
            print_output_stats(out_name, outputs[i]);
        }
    }
    if (debug_output_values) {
        for (size_t i = 0; i < outputs.size(); ++i) {
            const std::string & out_name = i < output_infos.size() ? output_infos[i].name : std::string("output_") + std::to_string(i);
            print_output_values(out_name, outputs[i], debug_output_values_n);
        }
    }

    std::vector<detection> detections;
    std::string postprocess_error;
    const std::string family = detect_model_family(model, model_name);
    const int version = detect_model_version(model, model_name);

    std::cout << "model.family=" << (family.empty() ? "unknown" : family)
              << " model.version=" << (version >= 0 ? std::to_string(version) : "unknown") << std::endl;

    if (family == "yolo") {
        if (version > 0 && version <= 4) {
            if (!postprocess_yolo_v4(outputs, img.w, img.h, net_w, net_h, thresh, detections, postprocess_error)) {
                std::cerr << "postprocess failed: " << postprocess_error << std::endl;
                return 6;
            }
        } else if (version == 26) {
            if (!postprocess_yolo_v26(outputs, img.w, img.h, net_w, net_h, thresh, detections, postprocess_error)) {
                std::cerr << "postprocess failed: " << postprocess_error << std::endl;
                return 6;
            }
        } else {
            std::cerr << "postprocess failed: unsupported yolo version: "
                      << (version >= 0 ? std::to_string(version) : "unknown") << std::endl;
            return 6;
        }
    } else {
        std::cerr << "postprocess failed: unsupported model family: "
                  << (family.empty() ? "unknown" : family) << std::endl;
        return 6;
    }

    const std::string data_dir = "ggml/examples/yolo/data";
    std::vector<std::string> labels;
    if (!load_labels(data_dir + "/coco.names", labels)) {
        std::cerr << "failed to load labels from " << data_dir << "/coco.names" << std::endl;
        return 7;
    }
    std::vector<yolo_image> alphabet;
    if (!load_alphabet(data_dir, alphabet)) {
        std::cerr << "failed to load alphabet from " << data_dir << "/labels" << std::endl;
        return 8;
    }

    draw_detections(img, detections, thresh, labels, alphabet);
    if (!save_image(img, output_path.c_str(), 80)) {
        std::cerr << "failed to save output image: " << output_path << std::endl;
        return 9;
    }

    std::cout << "saved detection image to " << output_path << std::endl;
    return 0;
}
