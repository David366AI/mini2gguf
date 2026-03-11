#include "mini2gguf/yolo_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace mini2gguf {
namespace {

struct yolo_layer {
    int classes = 80;
    std::vector<int> mask;
    std::vector<float> anchors;
    std::vector<float> predictions;
    int w = 0;
    int h = 0;

    int entry_index(int location, int entry) const {
        const int n = location / (w * h);
        const int loc = location % (w * h);
        return n * w * h * (4 + classes + 1) + entry * w * h + loc;
    }
};

struct yolo_xyxy_candidate {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float score = 0.0f;
    int cls = -1;
};

static void activate_array(float * x, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
}

static void apply_yolo(yolo_layer & layer) {
    const int w = layer.w;
    const int h = layer.h;
    const int n_masks = static_cast<int>(layer.mask.size());
    float * data = layer.predictions.data();
    for (int n = 0; n < n_masks; n++) {
        int index = layer.entry_index(n * w * h, 0);
        activate_array(data + index, 2 * w * h);
        index = layer.entry_index(n * w * h, 4);
        activate_array(data + index, (1 + layer.classes) * w * h);
    }
}

static YoloBox get_yolo_box(
    const yolo_layer & layer,
    int n,
    int index,
    int i,
    int j,
    int lw,
    int lh,
    int w,
    int h,
    int stride) {
    const float * predictions = layer.predictions.data();
    YoloBox b;
    b.x = (i + predictions[index + 0 * stride]) / lw;
    b.y = (j + predictions[index + 1 * stride]) / lh;
    b.w = std::exp(predictions[index + 2 * stride]) * layer.anchors[2 * n] / w;
    b.h = std::exp(predictions[index + 3 * stride]) * layer.anchors[2 * n + 1] / h;
    return b;
}

static void correct_yolo_box(YoloBox & b, int im_w, int im_h, int net_w, int net_h) {
    int new_w = 0;
    int new_h = 0;
    if ((static_cast<float>(net_w) / im_w) < (static_cast<float>(net_h) / im_h)) {
        new_w = net_w;
        new_h = (im_h * net_w) / im_w;
    } else {
        new_h = net_h;
        new_w = (im_w * net_h) / im_h;
    }
    b.x = (b.x - (net_w - new_w) / 2.0f / net_w) / (static_cast<float>(new_w) / net_w);
    b.y = (b.y - (net_h - new_h) / 2.0f / net_h) / (static_cast<float>(new_h) / net_h);
    b.w *= static_cast<float>(net_w) / new_w;
    b.h *= static_cast<float>(net_h) / new_h;
}

static void get_yolo_detections(
    const yolo_layer & layer,
    std::vector<YoloDetection> & detections,
    int im_w,
    int im_h,
    int netw,
    int neth,
    float thresh) {
    const int w = layer.w;
    const int h = layer.h;
    const int n_masks = static_cast<int>(layer.mask.size());
    const float * predictions = layer.predictions.data();
    for (int i = 0; i < w * h; i++) {
        for (int n = 0; n < n_masks; n++) {
            const int obj_index = layer.entry_index(n * w * h + i, 4);
            const float objectness = predictions[obj_index];
            if (objectness <= thresh) {
                continue;
            }

            YoloDetection det;
            const int box_index = layer.entry_index(n * w * h + i, 0);
            const int row = i / w;
            const int col = i % w;
            det.bbox = get_yolo_box(layer, layer.mask[n], box_index, col, row, w, h, netw, neth, w * h);
            correct_yolo_box(det.bbox, im_w, im_h, netw, neth);
            det.objectness = objectness;
            det.prob.resize(static_cast<size_t>(layer.classes));
            for (int j = 0; j < layer.classes; j++) {
                const int class_index = layer.entry_index(n * w * h + i, 4 + 1 + j);
                const float prob = objectness * predictions[class_index];
                det.prob[static_cast<size_t>(j)] = (prob > thresh) ? prob : 0.0f;
            }
            detections.push_back(std::move(det));
        }
    }
}

static float overlap(float x1, float w1, float x2, float w2) {
    const float l1 = x1 - w1 / 2;
    const float l2 = x2 - w2 / 2;
    const float left = l1 > l2 ? l1 : l2;
    const float r1 = x1 + w1 / 2;
    const float r2 = x2 + w2 / 2;
    const float right = r1 < r2 ? r1 : r2;
    return right - left;
}

static float box_intersection(const YoloBox & a, const YoloBox & b) {
    const float w = overlap(a.x, a.w, b.x, b.w);
    const float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0) {
        return 0;
    }
    return w * h;
}

static float box_union(const YoloBox & a, const YoloBox & b) {
    const float i = box_intersection(a, b);
    return a.w * a.h + b.w * b.h - i;
}

static float box_iou(const YoloBox & a, const YoloBox & b) {
    return box_intersection(a, b) / box_union(a, b);
}

static float clampf(float v, float lo, float hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
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

static YoloBox xyxy_to_cxcywh(float x1, float y1, float x2, float y2, int img_w, int img_h) {
    YoloBox b;
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

static void do_nms_sort(std::vector<YoloDetection> & dets, int classes, float thresh, bool agnostic_nms) {
    int k = static_cast<int>(dets.size()) - 1;
    for (int i = 0; i <= k; ++i) {
        if (dets[static_cast<size_t>(i)].objectness == 0.0f) {
            std::swap(dets[static_cast<size_t>(i)], dets[static_cast<size_t>(k)]);
            --k;
            --i;
        }
    }
    const int total = k + 1;
    if (total <= 0) {
        return;
    }

    auto det_best_prob = [](const YoloDetection & det) -> float {
        float best = 0.0f;
        for (float p : det.prob) {
            best = std::max(best, p);
        }
        return best;
    };

    if (agnostic_nms) {
        std::sort(dets.begin(), dets.begin() + total, [&](const YoloDetection & a, const YoloDetection & b) {
            return det_best_prob(a) > det_best_prob(b);
        });
        for (int i = 0; i < total; ++i) {
            if (det_best_prob(dets[static_cast<size_t>(i)]) == 0.0f) {
                continue;
            }
            const YoloBox a = dets[static_cast<size_t>(i)].bbox;
            for (int j = i + 1; j < total; ++j) {
                if (det_best_prob(dets[static_cast<size_t>(j)]) == 0.0f) {
                    continue;
                }
                const YoloBox b = dets[static_cast<size_t>(j)].bbox;
                if (box_iou(a, b) > thresh) {
                    std::fill(dets[static_cast<size_t>(j)].prob.begin(), dets[static_cast<size_t>(j)].prob.end(), 0.0f);
                    dets[static_cast<size_t>(j)].objectness = 0.0f;
                }
            }
        }
        return;
    }

    for (int c = 0; c < classes; ++c) {
        std::sort(dets.begin(), dets.begin() + total, [=](const YoloDetection & a, const YoloDetection & b) {
            return a.prob[static_cast<size_t>(c)] > b.prob[static_cast<size_t>(c)];
        });
        for (int i = 0; i < total; ++i) {
            if (dets[static_cast<size_t>(i)].prob[static_cast<size_t>(c)] == 0.0f) {
                continue;
            }
            const YoloBox a = dets[static_cast<size_t>(i)].bbox;
            for (int j = i + 1; j < total; ++j) {
                const YoloBox b = dets[static_cast<size_t>(j)].bbox;
                if (box_iou(a, b) > thresh) {
                    dets[static_cast<size_t>(j)].prob[static_cast<size_t>(c)] = 0.0f;
                }
            }
        }
    }
}

static float box_iou_xyxy(const yolo_xyxy_candidate & a, const yolo_xyxy_candidate & b) {
    const float inter_x1 = std::max(a.x1, b.x1);
    const float inter_y1 = std::max(a.y1, b.y1);
    const float inter_x2 = std::min(a.x2, b.x2);
    const float inter_y2 = std::min(a.y2, b.y2);
    const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    const float inter = inter_w * inter_h;
    const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    const float uni = area_a + area_b - inter;
    if (uni <= 0.0f) {
        return 0.0f;
    }
    return inter / uni;
}

static bool infer_yolo_layout(
    const std::vector<float> & pred,
    const std::vector<int64_t> & dims,
    int min_attrs,
    int fallback_attrs,
    bool fallback_attrs_first,
    int & attrs,
    int & boxes,
    bool & attrs_first) {
    attrs = -1;
    boxes = -1;
    attrs_first = true;
    if (pred.empty()) {
        return false;
    }

    auto set_layout = [&](int64_t maybe_attrs, int64_t maybe_boxes, bool maybe_attrs_first) -> bool {
        if (maybe_attrs < min_attrs || maybe_attrs > 1024 || maybe_boxes <= 0) {
            return false;
        }
        const size_t expected = static_cast<size_t>(maybe_attrs) * static_cast<size_t>(maybe_boxes);
        if (expected != pred.size()) {
            return false;
        }
        attrs = static_cast<int>(maybe_attrs);
        boxes = static_cast<int>(maybe_boxes);
        attrs_first = maybe_attrs_first;
        return true;
    };

    if (dims.size() >= 3 && dims[0] == 1 && dims[1] > 0 && dims[2] > 0) {
        const size_t expected = static_cast<size_t>(dims[0]) * static_cast<size_t>(dims[1]) * static_cast<size_t>(dims[2]);
        if (expected == pred.size()) {
            const bool d1_ok = dims[1] >= min_attrs && dims[1] <= 1024;
            const bool d2_ok = dims[2] >= min_attrs && dims[2] <= 1024;
            if (d1_ok && !d2_ok) {
                return set_layout(dims[1], dims[2], true);
            }
            if (!d1_ok && d2_ok) {
                return set_layout(dims[2], dims[1], false);
            }
            if (d1_ok && d2_ok) {
                if (dims[1] <= dims[2]) {
                    return set_layout(dims[1], dims[2], true);
                }
                return set_layout(dims[2], dims[1], false);
            }
        }
    }

    if (dims.size() >= 2 && dims[dims.size() - 2] > 0 && dims[dims.size() - 1] > 0) {
        const int64_t a = dims[dims.size() - 2];
        const int64_t b = dims[dims.size() - 1];
        const size_t expected = static_cast<size_t>(a) * static_cast<size_t>(b);
        if (expected == pred.size()) {
            const bool a_ok = a >= min_attrs && a <= 1024;
            const bool b_ok = b >= min_attrs && b <= 1024;
            if (a_ok && !b_ok) {
                return set_layout(a, b, true);
            }
            if (!a_ok && b_ok) {
                return set_layout(b, a, false);
            }
            if (a_ok && b_ok) {
                if (a <= b) {
                    return set_layout(a, b, true);
                }
                return set_layout(b, a, false);
            }
        }
    }

    if (fallback_attrs >= min_attrs && (pred.size() % static_cast<size_t>(fallback_attrs)) == 0) {
        attrs = fallback_attrs;
        boxes = static_cast<int>(pred.size() / static_cast<size_t>(fallback_attrs));
        attrs_first = fallback_attrs_first;
        return true;
    }

    if ((pred.size() % 8400) == 0) {
        const int maybe_attrs = static_cast<int>(pred.size() / 8400);
        if (maybe_attrs >= min_attrs && maybe_attrs <= 1024) {
            attrs = maybe_attrs;
            boxes = 8400;
            attrs_first = true;
            return true;
        }
    }

    return false;
}

static bool infer_yolo_v8_layout(
    const std::vector<float> & pred,
    const std::vector<int64_t> & dims,
    int & attrs,
    int & boxes,
    bool & attrs_first) {
    return infer_yolo_layout(pred, dims, 5, 84, true, attrs, boxes, attrs_first);
}

static bool infer_yolo_v5_layout(
    const std::vector<float> & pred,
    const std::vector<int64_t> & dims,
    int & attrs,
    int & boxes,
    bool & attrs_first) {
    return infer_yolo_layout(pred, dims, 6, 85, false, attrs, boxes, attrs_first);
}

static bool postprocess_yolo_v4(
    const std::vector<std::vector<float>> & outputs,
    const YoloPostprocessOptions & options,
    std::vector<YoloDetection> & detections,
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
    get_yolo_detections(
        yolo16,
        detections,
        options.image_w,
        options.image_h,
        options.net_w,
        options.net_h,
        options.conf_thres);
    get_yolo_detections(
        yolo23,
        detections,
        options.image_w,
        options.image_h,
        options.net_w,
        options.net_h,
        options.conf_thres);
    do_nms_sort(detections, 80, options.iou_thres, options.agnostic_nms);
    return true;
}

static bool postprocess_yolo_v26(
    const std::vector<std::vector<float>> & outputs,
    const YoloPostprocessOptions & options,
    std::vector<YoloDetection> & detections,
    std::string & error) {
    if (outputs.empty()) {
        error = "YOLO postprocess expects at least 1 output";
        return false;
    }

    const std::vector<float> & pred = outputs[0];
    if (pred.empty() || (pred.size() % 6) != 0) {
        error = "YOLO output must be Kx6";
        return false;
    }

    constexpr int classes = 80;
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

        if (score < options.conf_thres) {
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

        scale_box_xyxy_from_net_to_img(
            x1, y1, x2, y2,
            options.net_w, options.net_h,
            options.image_w, options.image_h);
        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        YoloDetection det;
        det.bbox = xyxy_to_cxcywh(x1, y1, x2, y2, options.image_w, options.image_h);
        det.objectness = score;
        det.prob.assign(classes, 0.0f);
        det.prob[static_cast<size_t>(cls)] = score;
        detections.push_back(std::move(det));
    }

    std::cout << "yolo postprocess: layout=[x1,y1,x2,y2,score,cls], conf_keep=" << kept_conf
              << ", pre_nms=" << detections.size()
              << ", score_range=[" << min_score << "," << max_score << "]"
              << std::endl;

    do_nms_sort(detections, classes, options.iou_thres, options.agnostic_nms);
    std::cout << "yolo postprocess: post_nms=" << detections.size() << std::endl;
    return true;
}

static bool postprocess_yolo_v8(
    const std::vector<std::vector<float>> & outputs,
    const std::vector<DynamicModel::TensorInfo> & output_infos,
    const YoloPostprocessOptions & options,
    std::vector<YoloDetection> & detections,
    std::string & error) {
    if (outputs.empty()) {
        error = "YOLO postprocess expects at least 1 output";
        return false;
    }

    const std::vector<float> & pred = outputs[0];
    if (pred.empty()) {
        error = "YOLO output[0] is empty";
        return false;
    }

    std::vector<int64_t> dims;
    if (!output_infos.empty()) {
        dims = output_infos[0].dims;
    }

    int attrs = -1;
    int num_boxes = -1;
    bool attrs_first = true;
    if (!infer_yolo_v8_layout(pred, dims, attrs, num_boxes, attrs_first)) {
        error = "YOLO output shape is not supported, expect [1,(nc+4),N] or [1,N,(nc+4)]";
        return false;
    }

    const int classes = attrs - 4;
    if (classes <= 0) {
        error = "YOLO output has invalid class count";
        return false;
    }
    if (static_cast<size_t>(attrs) * static_cast<size_t>(num_boxes) != pred.size()) {
        error = "YOLO output size does not match parsed shape";
        return false;
    }

    constexpr int max_det = 300;
    constexpr int max_nms = 30000;
    const bool debug_boxes = std::getenv("MINI2GGUF_DEBUG_YOLO8_BOXES") != nullptr;

    auto at = [&](int box_idx, int attr_idx) -> float {
        if (attrs_first) {
            return pred[static_cast<size_t>(attr_idx) * static_cast<size_t>(num_boxes) + static_cast<size_t>(box_idx)];
        }
        return pred[static_cast<size_t>(box_idx) * static_cast<size_t>(attrs) + static_cast<size_t>(attr_idx)];
    };

    std::vector<yolo_xyxy_candidate> candidates;
    candidates.reserve(static_cast<size_t>(num_boxes));
    for (int i = 0; i < num_boxes; ++i) {
        float best_conf = -std::numeric_limits<float>::infinity();
        int best_cls = -1;
        for (int c = 0; c < classes; ++c) {
            const float score = at(i, 4 + c);
            if (score > best_conf) {
                best_conf = score;
                best_cls = c;
            }
        }
        if (best_conf <= options.conf_thres || best_cls < 0) {
            continue;
        }

        const float cx = at(i, 0);
        const float cy = at(i, 1);
        const float w = at(i, 2);
        const float h = at(i, 3);
        if (w <= 0.0f || h <= 0.0f) {
            continue;
        }

        yolo_xyxy_candidate cand;
        cand.x1 = cx - 0.5f * w;
        cand.y1 = cy - 0.5f * h;
        cand.x2 = cx + 0.5f * w;
        cand.y2 = cy + 0.5f * h;
        cand.score = best_conf;
        cand.cls = best_cls;
        candidates.push_back(cand);
    }

    if (candidates.empty()) {
        detections.clear();
        std::cout << "yolo postprocess: layout=[1x" << attrs << "x" << num_boxes
                  << "], conf_keep=0, post_nms=0" << std::endl;
        return true;
    }

    std::sort(candidates.begin(), candidates.end(), [](const yolo_xyxy_candidate & a, const yolo_xyxy_candidate & b) {
        return a.score > b.score;
    });
    if (debug_boxes) {
        const size_t topk = std::min<size_t>(10, candidates.size());
        std::cout << "yolo debug: top raw candidates" << std::endl;
        for (size_t i = 0; i < topk; ++i) {
            const auto & c = candidates[i];
            std::cout << "  [" << i << "] cls=" << c.cls
                      << " score=" << c.score
                      << " xyxy=(" << c.x1 << "," << c.y1 << "," << c.x2 << "," << c.y2 << ")"
                      << " wh=(" << (c.x2 - c.x1) << "," << (c.y2 - c.y1) << ")"
                      << std::endl;
        }
    }
    if (static_cast<int>(candidates.size()) > max_nms) {
        candidates.resize(static_cast<size_t>(max_nms));
    }

    std::vector<char> suppressed(candidates.size(), 0);
    std::vector<size_t> keep;
    keep.reserve(std::min(static_cast<size_t>(max_det), candidates.size()));
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        keep.push_back(i);
        if (static_cast<int>(keep.size()) >= max_det) {
            break;
        }
        const yolo_xyxy_candidate & a = candidates[i];
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }
            const yolo_xyxy_candidate & b = candidates[j];
            if (!options.agnostic_nms && a.cls != b.cls) {
                continue;
            }
            if (box_iou_xyxy(a, b) > options.iou_thres) {
                suppressed[j] = 1;
            }
        }
    }

    detections.clear();
    detections.reserve(keep.size());
    for (size_t idx : keep) {
        yolo_xyxy_candidate cand = candidates[idx];
        scale_box_xyxy_from_net_to_img(
            cand.x1, cand.y1, cand.x2, cand.y2,
            options.net_w, options.net_h,
            options.image_w, options.image_h);
        if (cand.x2 <= cand.x1 || cand.y2 <= cand.y1) {
            continue;
        }
        if (debug_boxes) {
            std::cout << "yolo debug: kept cls=" << cand.cls
                      << " score=" << cand.score
                      << " scaled_xyxy=(" << cand.x1 << "," << cand.y1 << "," << cand.x2 << "," << cand.y2 << ")"
                      << std::endl;
        }

        YoloDetection det;
        det.bbox = xyxy_to_cxcywh(cand.x1, cand.y1, cand.x2, cand.y2, options.image_w, options.image_h);
        det.objectness = cand.score;
        det.prob.assign(static_cast<size_t>(classes), 0.0f);
        det.prob[static_cast<size_t>(cand.cls)] = cand.score;
        detections.push_back(std::move(det));
    }

    std::cout << "yolo postprocess: layout=[1x" << attrs << "x" << num_boxes
              << "], conf_keep=" << candidates.size()
              << ", post_nms=" << detections.size()
              << std::endl;
    return true;
}

static bool postprocess_yolo_v5(
    const std::vector<std::vector<float>> & outputs,
    const std::vector<DynamicModel::TensorInfo> & output_infos,
    const YoloPostprocessOptions & options,
    std::vector<YoloDetection> & detections,
    std::string & error) {
    if (outputs.empty()) {
        error = "YOLO postprocess expects at least 1 output";
        return false;
    }

    const std::vector<float> & pred = outputs[0];
    if (pred.empty()) {
        error = "YOLO output[0] is empty";
        return false;
    }

    std::vector<int64_t> dims;
    if (!output_infos.empty()) {
        dims = output_infos[0].dims;
    }

    int attrs = -1;
    int num_boxes = -1;
    bool attrs_first = true;
    if (!infer_yolo_v5_layout(pred, dims, attrs, num_boxes, attrs_first)) {
        error = "YOLOv5 output shape is not supported, expect [1,(nc+5),N] or [1,N,(nc+5)]";
        return false;
    }

    const int classes = attrs - 5;
    if (classes <= 0) {
        error = "YOLOv5 output has invalid class count";
        return false;
    }
    if (static_cast<size_t>(attrs) * static_cast<size_t>(num_boxes) != pred.size()) {
        error = "YOLOv5 output size does not match parsed shape";
        return false;
    }

    constexpr int max_det = 300;
    constexpr int max_nms = 30000;
    const bool debug_boxes = std::getenv("MINI2GGUF_DEBUG_YOLO5_BOXES") != nullptr;

    auto at = [&](int box_idx, int attr_idx) -> float {
        if (attrs_first) {
            return pred[static_cast<size_t>(attr_idx) * static_cast<size_t>(num_boxes) + static_cast<size_t>(box_idx)];
        }
        return pred[static_cast<size_t>(box_idx) * static_cast<size_t>(attrs) + static_cast<size_t>(attr_idx)];
    };

    std::vector<yolo_xyxy_candidate> candidates;
    candidates.reserve(static_cast<size_t>(num_boxes));
    for (int i = 0; i < num_boxes; ++i) {
        const float obj_conf = at(i, 4);
        if (obj_conf <= options.conf_thres) {
            continue;
        }

        float best_conf = -std::numeric_limits<float>::infinity();
        int best_cls = -1;
        for (int c = 0; c < classes; ++c) {
            const float score = obj_conf * at(i, 5 + c);
            if (score > best_conf) {
                best_conf = score;
                best_cls = c;
            }
        }
        if (best_conf <= options.conf_thres || best_cls < 0) {
            continue;
        }

        const float cx = at(i, 0);
        const float cy = at(i, 1);
        const float w = at(i, 2);
        const float h = at(i, 3);
        if (w <= 0.0f || h <= 0.0f) {
            continue;
        }

        yolo_xyxy_candidate cand;
        cand.x1 = cx - 0.5f * w;
        cand.y1 = cy - 0.5f * h;
        cand.x2 = cx + 0.5f * w;
        cand.y2 = cy + 0.5f * h;
        cand.score = best_conf;
        cand.cls = best_cls;
        candidates.push_back(cand);
    }

    if (candidates.empty()) {
        detections.clear();
        std::cout << "yolo postprocess(v5): layout=[1x" << attrs << "x" << num_boxes
                  << "], conf_keep=0, post_nms=0" << std::endl;
        return true;
    }

    std::sort(candidates.begin(), candidates.end(), [](const yolo_xyxy_candidate & a, const yolo_xyxy_candidate & b) {
        return a.score > b.score;
    });
    if (debug_boxes) {
        const size_t topk = std::min<size_t>(10, candidates.size());
        std::cout << "yolo(v5) debug: top raw candidates" << std::endl;
        for (size_t i = 0; i < topk; ++i) {
            const auto & c = candidates[i];
            std::cout << "  [" << i << "] cls=" << c.cls
                      << " score=" << c.score
                      << " xyxy=(" << c.x1 << "," << c.y1 << "," << c.x2 << "," << c.y2 << ")"
                      << " wh=(" << (c.x2 - c.x1) << "," << (c.y2 - c.y1) << ")"
                      << std::endl;
        }
    }
    if (static_cast<int>(candidates.size()) > max_nms) {
        candidates.resize(static_cast<size_t>(max_nms));
    }

    std::vector<char> suppressed(candidates.size(), 0);
    std::vector<size_t> keep;
    keep.reserve(std::min(static_cast<size_t>(max_det), candidates.size()));
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        keep.push_back(i);
        if (static_cast<int>(keep.size()) >= max_det) {
            break;
        }
        const yolo_xyxy_candidate & a = candidates[i];
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }
            const yolo_xyxy_candidate & b = candidates[j];
            if (!options.agnostic_nms && a.cls != b.cls) {
                continue;
            }
            if (box_iou_xyxy(a, b) > options.iou_thres) {
                suppressed[j] = 1;
            }
        }
    }

    detections.clear();
    detections.reserve(keep.size());
    for (size_t idx : keep) {
        yolo_xyxy_candidate cand = candidates[idx];
        scale_box_xyxy_from_net_to_img(
            cand.x1, cand.y1, cand.x2, cand.y2,
            options.net_w, options.net_h,
            options.image_w, options.image_h);
        if (cand.x2 <= cand.x1 || cand.y2 <= cand.y1) {
            continue;
        }
        if (debug_boxes) {
            std::cout << "yolo(v5) debug: kept cls=" << cand.cls
                      << " score=" << cand.score
                      << " scaled_xyxy=(" << cand.x1 << "," << cand.y1 << "," << cand.x2 << "," << cand.y2 << ")"
                      << std::endl;
        }

        YoloDetection det;
        det.bbox = xyxy_to_cxcywh(cand.x1, cand.y1, cand.x2, cand.y2, options.image_w, options.image_h);
        det.objectness = cand.score;
        det.prob.assign(static_cast<size_t>(classes), 0.0f);
        det.prob[static_cast<size_t>(cand.cls)] = cand.score;
        detections.push_back(std::move(det));
    }

    std::cout << "yolo postprocess(v5): layout=[1x" << attrs << "x" << num_boxes
              << "], conf_keep=" << candidates.size()
              << ", post_nms=" << detections.size()
              << std::endl;
    return true;
}

} // namespace

bool postprocess_yolo_outputs(
    const std::vector<std::vector<float>> & outputs,
    const std::vector<DynamicModel::TensorInfo> & output_infos,
    const YoloPostprocessOptions & options,
    std::vector<YoloDetection> & detections,
    std::string & error) {
    if (options.model_version > 0 && options.model_version <= 4) {
        return postprocess_yolo_v4(outputs, options, detections, error);
    }
    if (options.model_version == 26 || options.model_version == 10) {
        return postprocess_yolo_v26(outputs, options, detections, error);
    }
    if (options.model_version == 5) {
        return postprocess_yolo_v5(outputs, output_infos, options, detections, error);
    }
    if (options.model_version == 8 || options.model_version == 9 || options.model_version == 11) {
        return postprocess_yolo_v8(outputs, output_infos, options, detections, error);
    }

    error = "unsupported yolo version: " +
            (options.model_version >= 0 ? std::to_string(options.model_version) : std::string("unknown"));
    return false;
}

} // namespace mini2gguf

