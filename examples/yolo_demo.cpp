#include "mini2gguf/crnn_utils.hpp"
#include "mini2gguf/model_runtime.hpp"
#include "mini2gguf/yolo_utils.hpp"

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

using box = mini2gguf::YoloBox;
using detection = mini2gguf::YoloDetection;

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
    const size_t begin = i;
    while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
        ++i;
    }
    if (i == begin) {
        return fallback;
    }
    const int value = std::atoi(text.substr(begin, i - begin).c_str());
    return negative ? -value : value;
}

static int infer_yolo_version_from_name(const std::string & model_name) {
    const std::string lower = to_lower_copy(model_name);
    const size_t pos = lower.find("yolo");
    if (pos == std::string::npos) {
        return -1;
    }
    size_t begin = pos + 4;
    while (begin < lower.size() &&
           (lower[begin] == 'v' || lower[begin] == '_' || lower[begin] == '-' || lower[begin] == ' ')) {
        ++begin;
    }
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
    const std::string version = model.model_metadata_value("model.version");
    const int parsed = parse_leading_int(version, -1);
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
            if (!load_image(fname, alphabet[static_cast<size_t>(j) * 128 + static_cast<size_t>(i)])) {
                std::cerr << "Cannot load '" << fname << "'" << std::endl;
                return false;
            }
        }
    }
    return true;
}

static float get_color(int c, int x, int max) {
    float colors[6][3] = {{1, 0, 1}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}};
    float ratio = (static_cast<float>(x) / max) * 5;
    int i = static_cast<int>(std::floor(ratio));
    int j = static_cast<int>(std::ceil(ratio));
    ratio -= i;
    return (1 - ratio) * colors[i][c] + ratio * colors[j][c];
}

static void draw_detections(
    yolo_image & im,
    const std::vector<detection> & dets,
    float thresh,
    const std::vector<std::string> & labels,
    const std::vector<yolo_image> & alphabet) {
    const int classes = static_cast<int>(labels.size());
    for (int i = 0; i < static_cast<int>(dets.size()); i++) {
        std::string labelstr;
        int cl = -1;
        const int prob_count = static_cast<int>(dets[static_cast<size_t>(i)].prob.size());
        for (int j = 0; j < prob_count && j < classes; j++) {
            if (dets[static_cast<size_t>(i)].prob[static_cast<size_t>(j)] > thresh) {
                if (cl < 0) {
                    labelstr = labels[static_cast<size_t>(j)];
                    cl = j;
                } else {
                    labelstr += ", ";
                    labelstr += labels[static_cast<size_t>(j)];
                }
                std::printf("%s: %.0f%%\n", labels[static_cast<size_t>(j)].c_str(),
                            dets[static_cast<size_t>(i)].prob[static_cast<size_t>(j)] * 100);
            }
        }
        if (cl >= 0) {
            const int width = im.h * .006;
            const int offset = cl * 123457 % classes;
            const float red = get_color(2, offset, classes);
            const float green = get_color(1, offset, classes);
            const float blue = get_color(0, offset, classes);
            float rgb[3];

            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
            const box b = dets[static_cast<size_t>(i)].bbox;

            int left = static_cast<int>((b.x - b.w / 2.0f) * im.w);
            int right = static_cast<int>((b.x + b.w / 2.0f) * im.w);
            int top = static_cast<int>((b.y - b.h / 2.0f) * im.h);
            int bot = static_cast<int>((b.y + b.h / 2.0f) * im.h);

            if (left < 0) {
                left = 0;
            }
            if (right > im.w - 1) {
                right = im.w - 1;
            }
            if (top < 0) {
                top = 0;
            }
            if (bot > im.h - 1) {
                bot = im.h - 1;
            }

            if (!dets[static_cast<size_t>(i)].mask.empty() &&
                dets[static_cast<size_t>(i)].mask_w == im.w &&
                dets[static_cast<size_t>(i)].mask_h == im.h) {
                const float alpha = 0.35f;
                for (int y = top; y <= bot; ++y) {
                    for (int x = left; x <= right; ++x) {
                        const size_t midx = static_cast<size_t>(y) * static_cast<size_t>(im.w) + static_cast<size_t>(x);
                        if (dets[static_cast<size_t>(i)].mask[midx] == 0) {
                            continue;
                        }
                        const float r0 = im.get_pixel(x, y, 0);
                        const float g0 = im.get_pixel(x, y, 1);
                        const float b0 = im.get_pixel(x, y, 2);
                        im.set_pixel(x, y, 0, r0 * (1.0f - alpha) + red * alpha);
                        im.set_pixel(x, y, 1, g0 * (1.0f - alpha) + green * alpha);
                        im.set_pixel(x, y, 2, b0 * (1.0f - alpha) + blue * alpha);
                    }
                }
            }

            draw_box_width(im, left, top, right, bot, width, red, green, blue);
            yolo_image label = get_label(alphabet, labelstr, (im.h * .03f));
            draw_label(im, top + width, left, label, rgb);
        }
    }
}

struct cli_options {
    std::string model_spec;
    std::string model_dir;
    std::string model_name;
    std::string image_path;
    std::string output_path = "predictions_dynamic.jpg";
    std::string backend = "auto";
    float conf_thres = 0.5f;
    float iou_thres = 0.45f;
    bool agnostic_nms = false;
};

static bool ends_with(const std::string & value, const std::string & suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool parse_float_01(const std::string & value, const char * opt_name, float & out, std::string & error) {
    char * end = nullptr;
    const float parsed = std::strtof(value.c_str(), &end);
    if (end == value.c_str() || (end != nullptr && *end != '\0') || !std::isfinite(parsed)) {
        error = std::string("invalid value for ") + opt_name + ": " + value;
        return false;
    }
    if (parsed < 0.0f || parsed > 1.0f) {
        error = std::string("value out of range [0,1] for ") + opt_name + ": " + value;
        return false;
    }
    out = parsed;
    return true;
}

static bool resolve_model_spec(const std::string & model_spec, std::string & model_dir, std::string & model_name, std::string & error) {
    if (model_spec.empty()) {
        error = "model path/name is empty";
        return false;
    }

    const size_t sep = model_spec.find_last_of("/\\");
    const std::string base = (sep == std::string::npos) ? model_spec : model_spec.substr(sep + 1);
    model_dir = (sep == std::string::npos) ? "assets/models/yolo" : model_spec.substr(0, sep);
    if (model_dir.empty()) {
        model_dir = ".";
    }

    if (ends_with(base, "_weights.gguf")) {
        model_name = base.substr(0, base.size() - std::string("_weights.gguf").size());
    } else if (ends_with(base, "_graph.json")) {
        model_name = base.substr(0, base.size() - std::string("_graph.json").size());
    } else if (ends_with(base, ".gguf")) {
        model_name = base.substr(0, base.size() - std::string(".gguf").size());
    } else if (ends_with(base, ".json")) {
        model_name = base.substr(0, base.size() - std::string(".json").size());
    } else {
        model_name = base;
    }

    if (ends_with(model_name, "_weights")) {
        model_name = model_name.substr(0, model_name.size() - std::string("_weights").size());
    }
    if (ends_with(model_name, "_graph")) {
        model_name = model_name.substr(0, model_name.size() - std::string("_graph").size());
    }

    if (model_name.empty()) {
        error = "failed to resolve model name from --model/-m";
        return false;
    }
    return true;
}

static void print_usage(const char * prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << " -m <model_name|*.gguf> -i <image_path> [options]\n\n"
              << "Required:\n"
              << "  -m, --model       model name or path to .gguf\n"
              << "  -i, --input       input image path\n\n"
              << "Options:\n"
              << "  -o, --output      output image path (default: predictions_dynamic.jpg)\n"
              << "      --conf, --conf_thres  confidence threshold in [0,1] (default: 0.5)\n"
              << "      --iou,  --iou_thres   IoU threshold in [0,1] (default: 0.45)\n"
              << "  -a, --agnostic    class-agnostic NMS\n"
              << "  -b, --backend     backend name (default: auto)\n"
              << "  -h, --help        show this message\n\n"
              << "Legacy format (still supported):\n"
              << "  " << prog << " <model_dir> <model_name> <image_path> [output_path] [conf_thres] [backend]"
              << std::endl;
}

static bool parse_cli_options(int argc, char ** argv, cli_options & opt, bool & show_help, std::string & error) {
    show_help = false;
    error.clear();

    if (argc >= 4 && argv[1][0] != '-' && argv[2][0] != '-' && argv[3][0] != '-') {
        opt.model_dir = argv[1];
        opt.model_name = argv[2];
        opt.image_path = argv[3];
        if (argc >= 5) {
            opt.output_path = argv[4];
        }
        if (argc >= 6) {
            if (!parse_float_01(argv[5], "conf_thres", opt.conf_thres, error)) {
                return false;
            }
        }
        if (argc >= 7) {
            opt.backend = argv[6];
        }
        if (argc > 7) {
            error = "too many positional arguments";
            return false;
        }
        return true;
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const std::string & option_name, std::string & value_out) -> bool {
            if (i + 1 >= argc) {
                error = "missing value for option " + option_name;
                return false;
            }
            value_out = argv[++i];
            return true;
        };

        if (arg == "-h" || arg == "--help") {
            show_help = true;
            return true;
        }
        if (arg == "-m" || arg == "--model") {
            if (!need_value(arg, opt.model_spec)) {
                return false;
            }
            continue;
        }
        if (arg == "-i" || arg == "--input") {
            if (!need_value(arg, opt.image_path)) {
                return false;
            }
            continue;
        }
        if (arg == "-o" || arg == "--output") {
            if (!need_value(arg, opt.output_path)) {
                return false;
            }
            continue;
        }
        if (arg == "-b" || arg == "--backend") {
            if (!need_value(arg, opt.backend)) {
                return false;
            }
            continue;
        }
        if (arg == "--conf" || arg == "--conf_thres") {
            std::string value;
            if (!need_value(arg, value)) {
                return false;
            }
            if (!parse_float_01(value, arg.c_str(), opt.conf_thres, error)) {
                return false;
            }
            continue;
        }
        if (arg == "--iou" || arg == "--iou_thres") {
            std::string value;
            if (!need_value(arg, value)) {
                return false;
            }
            if (!parse_float_01(value, arg.c_str(), opt.iou_thres, error)) {
                return false;
            }
            continue;
        }
        if (arg == "-a" || arg == "--agnostic") {
            opt.agnostic_nms = true;
            continue;
        }

        error = "unknown option: " + arg;
        return false;
    }

    if (opt.model_spec.empty()) {
        error = "missing required option --model/-m";
        return false;
    }
    if (opt.image_path.empty()) {
        error = "missing required option --input/-i";
        return false;
    }
    if (!resolve_model_spec(opt.model_spec, opt.model_dir, opt.model_name, error)) {
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    cli_options opts;
    bool show_help = false;
    std::string cli_error;
    if (!parse_cli_options(argc, argv, opts, show_help, cli_error)) {
        print_usage(argv[0]);
        if (!cli_error.empty()) {
            std::cerr << "error: " << cli_error << std::endl;
            return 1;
        }
        return 0;
    }
    if (show_help) {
        print_usage(argv[0]);
        return 0;
    }

    setenv("MINI2GGUF_BACKEND", opts.backend.c_str(), 1);

    mini2gguf::DynamicModel model;
    if (!model.load(opts.model_dir, opts.model_name)) {
        std::cerr << "load failed: " << model.last_error() << std::endl;
        return 2;
    }

    std::cout << "backend: " << model.backend_name() << std::endl;
    std::cout << "device: " << model.backend_device_name() << std::endl;

    const auto & input_infos = model.input_tensors();
    const auto & output_infos = model.output_tensors();

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

    if (input_infos.empty() || output_infos.empty()) {
        std::cerr << "unexpected model IO layout, require 1 input and at least 1 output" << std::endl;
        return 3;
    }

    const std::string family = detect_model_family(model, opts.model_name);
    const int version = detect_model_version(model, opts.model_name);

    std::cout << "model.family=" << (family.empty() ? "unknown" : family)
              << " model.version=" << (version >= 0 ? std::to_string(version) : "unknown") << std::endl;

    int net_w = 416;
    int net_h = 416;
    if (input_infos[0].dims.size() >= 4) {
        net_h = static_cast<int>(input_infos[0].dims[2]);
        net_w = static_cast<int>(input_infos[0].dims[3]);
    }
    if (net_h <= 0 || net_w <= 0) {
        std::cerr << "invalid input shape for model input[0]: " << dims_to_string(input_infos[0].dims) << std::endl;
        return 3;
    }

    yolo_image img;
    if (!load_image(opts.image_path.c_str(), img)) {
        std::cerr << "failed to load image: " << opts.image_path << std::endl;
        return 4;
    }

    std::vector<float> input;
    if (family == "crnn") {
        std::string preprocess_error;
        mini2gguf::CrnnPreprocessOptions pp;
        if (!mini2gguf::preprocess_crnn_input(
                img.data.data(),
                img.w,
                img.h,
                img.c,
                input_infos[0],
                input,
                preprocess_error,
                pp)) {
            std::cerr << "CRNN preprocess failed: " << preprocess_error << std::endl;
            return 4;
        }
    } else {
        yolo_image sized = letterbox_image(img, net_w, net_h);
        input = sized.data;
    }

    std::vector<std::vector<float>> outputs;
    if (!model.infer_all(input, outputs)) {
        std::cerr << "infer failed: " << model.last_error() << std::endl;
        return 5;
    }

    const bool debug_output_stats = std::getenv("MINI2GGUF_DEBUG_OUTPUT_STATS") != nullptr;
    const bool debug_output_values = std::getenv("MINI2GGUF_DEBUG_OUTPUT_VALUES") != nullptr;
    size_t debug_output_values_n = 10;
    if (const char * v = std::getenv("MINI2GGUF_DEBUG_OUTPUT_VALUES_N")) {
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

    if (family == "yolo") {
        std::cout << "postprocess: conf_thres=" << opts.conf_thres
                  << " iou_thres=" << opts.iou_thres
                  << " agnostic_nms=" << (opts.agnostic_nms ? "true" : "false")
                  << std::endl;

        mini2gguf::YoloPostprocessOptions pp;
        pp.image_w = img.w;
        pp.image_h = img.h;
        pp.net_w = net_w;
        pp.net_h = net_h;
        pp.model_version = version;
        pp.conf_thres = opts.conf_thres;
        pp.iou_thres = opts.iou_thres;
        pp.agnostic_nms = opts.agnostic_nms;

        if (!mini2gguf::postprocess_yolo_outputs(outputs, output_infos, pp, detections, postprocess_error)) {
            std::cerr << "postprocess failed: " << postprocess_error << std::endl;
            return 6;
        }
    } else if (family == "crnn") {
        std::vector<std::string> dict;
        const std::string dict_metadata = model.model_metadata_value("model.dict");
        if (!mini2gguf::parse_crnn_dict_metadata(dict_metadata, dict)) {
            std::cerr << "postprocess failed: missing or empty model.dict metadata for CRNN model" << std::endl;
            return 6;
        }

        std::string crnn_text;
        mini2gguf::CrnnPostprocessOptions pp;
        if (!mini2gguf::postprocess_crnn_outputs(outputs, output_infos, dict, crnn_text, postprocess_error, pp)) {
            std::cerr << "postprocess failed: " << postprocess_error << std::endl;
            return 6;
        }

        std::cout << "crnn.text=" << crnn_text << std::endl;
        return 0;
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

    draw_detections(img, detections, opts.conf_thres, labels, alphabet);
    if (!save_image(img, opts.output_path.c_str(), 80)) {
        std::cerr << "failed to save output image: " << opts.output_path << std::endl;
        return 9;
    }

    std::cout << "saved detection image to " << opts.output_path << std::endl;
    return 0;
}
