#include "mini2gguf/model_runtime.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace mini2gguf
{

    namespace
    {

        static std::string to_lower_copy(std::string value)
        {
            std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c)
                           { return static_cast<char>(std::tolower(c)); });
            return value;
        }

        static bool contains_case_insensitive(const std::string &text, const std::string &pattern)
        {
            const std::string lower_text = to_lower_copy(text);
            const std::string lower_pattern = to_lower_copy(pattern);
            return lower_text.find(lower_pattern) != std::string::npos;
        }

        static int parse_trailing_index(const std::string &value, const std::string &prefix)
        {
            if (value.size() <= prefix.size())
            {
                return 0;
            }
            const std::string suffix = value.substr(prefix.size());
            for (char c : suffix)
            {
                if (!std::isdigit(static_cast<unsigned char>(c)))
                {
                    return 0;
                }
            }
            return std::atoi(suffix.c_str());
        }

        static ggml_backend_t init_backend_by_name_and_index(const std::string &name_filter, int selected_index)
        {
            int match_index = 0;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
            {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (dev == nullptr)
                {
                    continue;
                }
                const char *dev_name_c = ggml_backend_dev_name(dev);
                const char *dev_desc_c = ggml_backend_dev_description(dev);
                const std::string dev_name = dev_name_c ? dev_name_c : "";
                const std::string dev_desc = dev_desc_c ? dev_desc_c : "";

                if (!contains_case_insensitive(dev_name, name_filter) &&
                    !contains_case_insensitive(dev_desc, name_filter))
                {
                    continue;
                }

                if (match_index == selected_index)
                {
                    return ggml_backend_dev_init(dev, nullptr);
                }
                ++match_index;
            }
            return nullptr;
        }

        static bool env_flag_enabled(const char *name)
        {
            const char *value = std::getenv(name);
            if (value == nullptr)
            {
                return false;
            }
            const std::string lower = to_lower_copy(value);
            if (lower.empty() || lower == "0" || lower == "false" || lower == "off" || lower == "no")
            {
                return false;
            }
            return true;
        }

        static int env_int_or_default(const char *name, int fallback, int min_value = 1)
        {
            const char *value = std::getenv(name);
            if (value == nullptr)
            {
                return fallback;
            }
            const int parsed = std::atoi(value);
            if (parsed < min_value)
            {
                return fallback;
            }
            return parsed;
        }

        struct NodeProfileTiming
        {
            int node_index = -1;
            std::string tensor_name;
            std::string ggml_op;
            double elapsed_ms = 0.0;
        };

        struct SchedulerNodeProfileState
        {
            std::unordered_map<const ggml_tensor *, int> node_index_by_ptr;
            std::vector<NodeProfileTiming> timings;
            std::chrono::steady_clock::time_point last_start = std::chrono::steady_clock::now();
            bool has_start = false;
        };

        static bool scheduler_profile_callback(struct ggml_tensor *t, bool ask, void *user_data)
        {
            auto *state = reinterpret_cast<SchedulerNodeProfileState *>(user_data);
            if (state == nullptr || t == nullptr)
            {
                return false;
            }

            if (ask)
            {
                state->last_start = std::chrono::steady_clock::now();
                state->has_start = true;
                // Return true so scheduler computes this node separately.
                return true;
            }

            const auto end = std::chrono::steady_clock::now();
            const double elapsed_ms = state->has_start
                                          ? std::chrono::duration<double, std::milli>(end - state->last_start).count()
                                          : 0.0;
            state->has_start = false;

            NodeProfileTiming timing;
            auto it = state->node_index_by_ptr.find(t);
            if (it != state->node_index_by_ptr.end())
            {
                timing.node_index = it->second;
            }
            const char *name_c = ggml_get_name(t);
            timing.tensor_name = name_c != nullptr ? name_c : "";
            const char *op_desc_c = ggml_op_desc(t);
            if (op_desc_c != nullptr && op_desc_c[0] != '\0')
            {
                timing.ggml_op = op_desc_c;
            }
            else
            {
                const char *op_name_c = ggml_op_name(t->op);
                timing.ggml_op = op_name_c != nullptr ? op_name_c : "unknown";
            }
            timing.elapsed_ms = elapsed_ms;
            state->timings.push_back(std::move(timing));
            return true;
        }

        struct JsonValue
        {
            enum class Type
            {
                Null,
                Bool,
                Number,
                String,
                Array,
                Object
            };

            Type type = Type::Null;
            bool boolean = false;
            double number = 0.0;
            std::string string;
            std::vector<JsonValue> array;
            std::unordered_map<std::string, JsonValue> object;
        };

        class JsonParser
        {
        public:
            explicit JsonParser(const std::string &text) : text_(text) {}

            JsonValue parse()
            {
                skip_ws();
                JsonValue value = parse_value();
                skip_ws();
                if (pos_ != text_.size())
                {
                    throw std::runtime_error("json trailing characters");
                }
                return value;
            }

        private:
            void skip_ws()
            {
                while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_])))
                {
                    ++pos_;
                }
            }

            char peek() const
            {
                if (pos_ >= text_.size())
                {
                    throw std::runtime_error("json unexpected end");
                }
                return text_[pos_];
            }

            char take()
            {
                char c = peek();
                ++pos_;
                return c;
            }

            void expect(char expected)
            {
                const char got = take();
                if (got != expected)
                {
                    throw std::runtime_error("json expected different character");
                }
            }

            bool consume_literal(const char *lit)
            {
                size_t i = 0;
                while (lit[i] != '\0')
                {
                    if (pos_ + i >= text_.size() || text_[pos_ + i] != lit[i])
                    {
                        return false;
                    }
                    ++i;
                }
                pos_ += i;
                return true;
            }

            JsonValue parse_value()
            {
                skip_ws();
                const char c = peek();
                if (c == '{')
                {
                    return parse_object();
                }
                if (c == '[')
                {
                    return parse_array();
                }
                if (c == '"')
                {
                    JsonValue v;
                    v.type = JsonValue::Type::String;
                    v.string = parse_string();
                    return v;
                }
                if (c == 't')
                {
                    if (!consume_literal("true"))
                    {
                        throw std::runtime_error("json invalid true literal");
                    }
                    JsonValue v;
                    v.type = JsonValue::Type::Bool;
                    v.boolean = true;
                    return v;
                }
                if (c == 'f')
                {
                    if (!consume_literal("false"))
                    {
                        throw std::runtime_error("json invalid false literal");
                    }
                    JsonValue v;
                    v.type = JsonValue::Type::Bool;
                    v.boolean = false;
                    return v;
                }
                if (c == 'n')
                {
                    if (!consume_literal("null"))
                    {
                        throw std::runtime_error("json invalid null literal");
                    }
                    JsonValue v;
                    v.type = JsonValue::Type::Null;
                    return v;
                }
                return parse_number();
            }

            std::string parse_string()
            {
                expect('"');
                std::string out;
                while (true)
                {
                    if (pos_ >= text_.size())
                    {
                        throw std::runtime_error("json unterminated string");
                    }
                    char c = take();
                    if (c == '"')
                    {
                        break;
                    }
                    if (c == '\\')
                    {
                        if (pos_ >= text_.size())
                        {
                            throw std::runtime_error("json bad escape");
                        }
                        char e = take();
                        switch (e)
                        {
                        case '"':
                            out.push_back('"');
                            break;
                        case '\\':
                            out.push_back('\\');
                            break;
                        case '/':
                            out.push_back('/');
                            break;
                        case 'b':
                            out.push_back('\b');
                            break;
                        case 'f':
                            out.push_back('\f');
                            break;
                        case 'n':
                            out.push_back('\n');
                            break;
                        case 'r':
                            out.push_back('\r');
                            break;
                        case 't':
                            out.push_back('\t');
                            break;
                        case 'u':
                            if (pos_ + 4 > text_.size())
                            {
                                throw std::runtime_error("json bad unicode escape");
                            }
                            pos_ += 4;
                            out.push_back('?');
                            break;
                        default:
                            throw std::runtime_error("json unknown escape");
                        }
                    }
                    else
                    {
                        out.push_back(c);
                    }
                }
                return out;
            }

            JsonValue parse_number()
            {
                const size_t start = pos_;
                if (text_[pos_] == '-')
                {
                    ++pos_;
                }
                while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_])))
                {
                    ++pos_;
                }
                if (pos_ < text_.size() && text_[pos_] == '.')
                {
                    ++pos_;
                    while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_])))
                    {
                        ++pos_;
                    }
                }
                if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E'))
                {
                    ++pos_;
                    if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-'))
                    {
                        ++pos_;
                    }
                    while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_])))
                    {
                        ++pos_;
                    }
                }

                const std::string token = text_.substr(start, pos_ - start);
                char *end_ptr = nullptr;
                const double value = std::strtod(token.c_str(), &end_ptr);
                if (end_ptr == token.c_str())
                {
                    throw std::runtime_error("json invalid number");
                }

                JsonValue v;
                v.type = JsonValue::Type::Number;
                v.number = value;
                return v;
            }

            JsonValue parse_array()
            {
                expect('[');
                JsonValue v;
                v.type = JsonValue::Type::Array;

                skip_ws();
                if (peek() == ']')
                {
                    take();
                    return v;
                }

                while (true)
                {
                    v.array.push_back(parse_value());
                    skip_ws();
                    const char c = take();
                    if (c == ']')
                    {
                        break;
                    }
                    if (c != ',')
                    {
                        throw std::runtime_error("json expected comma in array");
                    }
                }
                return v;
            }

            JsonValue parse_object()
            {
                expect('{');
                JsonValue v;
                v.type = JsonValue::Type::Object;

                skip_ws();
                if (peek() == '}')
                {
                    take();
                    return v;
                }

                while (true)
                {
                    skip_ws();
                    const std::string key = parse_string();
                    skip_ws();
                    expect(':');
                    JsonValue value = parse_value();
                    v.object.emplace(key, std::move(value));
                    skip_ws();
                    const char c = take();
                    if (c == '}')
                    {
                        break;
                    }
                    if (c != ',')
                    {
                        throw std::runtime_error("json expected comma in object");
                    }
                }
                return v;
            }

        private:
            const std::string &text_;
            size_t pos_ = 0;
        };

        static const JsonValue *find_obj_key(const JsonValue &obj, const std::string &key)
        {
            if (obj.type != JsonValue::Type::Object)
            {
                return nullptr;
            }
            auto it = obj.object.find(key);
            if (it == obj.object.end())
            {
                return nullptr;
            }
            return &it->second;
        }

        static std::string get_string(const JsonValue &obj, const std::string &key, const std::string &fallback = "")
        {
            const JsonValue *value = find_obj_key(obj, key);
            if (value == nullptr || value->type != JsonValue::Type::String)
            {
                return fallback;
            }
            return value->string;
        }

        static int64_t get_i64(const JsonValue &obj, const std::string &key, int64_t fallback = 0)
        {
            const JsonValue *value = find_obj_key(obj, key);
            if (value == nullptr || value->type != JsonValue::Type::Number)
            {
                return fallback;
            }
            return static_cast<int64_t>(value->number);
        }

        static double get_f64(const JsonValue &obj, const std::string &key, double fallback = 0.0)
        {
            const JsonValue *value = find_obj_key(obj, key);
            if (value == nullptr || value->type != JsonValue::Type::Number)
            {
                return fallback;
            }
            return value->number;
        }

        static std::vector<std::string> get_string_array(const JsonValue &obj, const std::string &key)
        {
            std::vector<std::string> out;
            const JsonValue *value = find_obj_key(obj, key);
            if (value == nullptr || value->type != JsonValue::Type::Array)
            {
                return out;
            }
            out.reserve(value->array.size());
            for (const JsonValue &item : value->array)
            {
                if (item.type == JsonValue::Type::String)
                {
                    out.push_back(item.string);
                }
            }
            return out;
        }

        static std::vector<int64_t> get_i64_array(const JsonValue &obj, const std::string &key)
        {
            std::vector<int64_t> out;
            const JsonValue *value = find_obj_key(obj, key);
            if (value == nullptr || value->type != JsonValue::Type::Array)
            {
                return out;
            }
            out.reserve(value->array.size());
            for (const JsonValue &item : value->array)
            {
                if (item.type == JsonValue::Type::Number)
                {
                    out.push_back(static_cast<int64_t>(item.number));
                }
            }
            return out;
        }

        static std::vector<int> to_i32_vec(const std::vector<int64_t> &input)
        {
            std::vector<int> out;
            out.reserve(input.size());
            for (int64_t value : input)
            {
                out.push_back(static_cast<int>(value));
            }
            return out;
        }

        static bool json_scalar_to_string(const JsonValue &value, std::string &out)
        {
            switch (value.type)
            {
            case JsonValue::Type::String:
                out = value.string;
                return true;
            case JsonValue::Type::Number:
            {
                std::ostringstream oss;
                oss << value.number;
                out = oss.str();
                return true;
            }
            case JsonValue::Type::Bool:
                out = value.boolean ? "true" : "false";
                return true;
            default:
                return false;
            }
        }

        static bool gguf_kv_value_to_string(const gguf_context *ctx, int64_t key_id, std::string &out)
        {
            switch (gguf_get_kv_type(ctx, key_id))
            {
            case GGUF_TYPE_STRING:
            {
                const char *value = gguf_get_val_str(ctx, key_id);
                out = value != nullptr ? value : "";
                return true;
            }
            case GGUF_TYPE_BOOL:
                out = gguf_get_val_bool(ctx, key_id) ? "true" : "false";
                return true;
            case GGUF_TYPE_UINT8:
                out = std::to_string(static_cast<unsigned int>(gguf_get_val_u8(ctx, key_id)));
                return true;
            case GGUF_TYPE_INT8:
                out = std::to_string(static_cast<int>(gguf_get_val_i8(ctx, key_id)));
                return true;
            case GGUF_TYPE_UINT16:
                out = std::to_string(static_cast<unsigned int>(gguf_get_val_u16(ctx, key_id)));
                return true;
            case GGUF_TYPE_INT16:
                out = std::to_string(static_cast<int>(gguf_get_val_i16(ctx, key_id)));
                return true;
            case GGUF_TYPE_UINT32:
                out = std::to_string(gguf_get_val_u32(ctx, key_id));
                return true;
            case GGUF_TYPE_INT32:
                out = std::to_string(gguf_get_val_i32(ctx, key_id));
                return true;
            case GGUF_TYPE_UINT64:
                out = std::to_string(gguf_get_val_u64(ctx, key_id));
                return true;
            case GGUF_TYPE_INT64:
                out = std::to_string(gguf_get_val_i64(ctx, key_id));
                return true;
            case GGUF_TYPE_FLOAT32:
            {
                std::ostringstream oss;
                oss << gguf_get_val_f32(ctx, key_id);
                out = oss.str();
                return true;
            }
            case GGUF_TYPE_FLOAT64:
            {
                std::ostringstream oss;
                oss << gguf_get_val_f64(ctx, key_id);
                out = oss.str();
                return true;
            }
            default:
                return false;
            }
        }

        static int64_t numel_from_dims(const std::vector<int64_t> &dims)
        {
            int64_t n = 1;
            for (int64_t d : dims)
            {
                n *= d;
            }
            return n;
        }

        static std::array<int64_t, 4> onnx_dims_to_ggml_ne(const std::vector<int64_t> &dims)
        {
            std::array<int64_t, 4> ne = {1, 1, 1, 1};
            const int rank = static_cast<int>(dims.size());
            for (int i = 0; i < rank && i < 4; ++i)
            {
                // Keep rank-local reverse mapping:
                // ONNX dims [d0..d{r-1}] -> ggml ne[r-1-i] = d_i.
                // For rank<4 this packs into low ggml dims and matches GGUF tensor shapes.
                ne[rank - 1 - i] = dims[i] <= 0 ? 1 : dims[i];
            }
            return ne;
        }

        static int map_onnx_axis_to_ggml(int axis, int rank)
        {
            int a = axis;
            if (a < 0)
            {
                a += rank;
            }
            if (a < 0 || a >= rank)
            {
                throw std::runtime_error("axis out of range");
            }
            return rank - 1 - a;
        }

        static std::vector<int> map_onnx_perm_to_ggml(const std::vector<int> &perm)
        {
            const int rank = static_cast<int>(perm.size());
            std::vector<int> mapped(rank, 0);
            for (int j = 0; j < rank; ++j)
            {
                const int onnx_out_axis = rank - 1 - j;
                const int onnx_src_axis = perm[onnx_out_axis];
                mapped[j] = rank - 1 - onnx_src_axis;
            }
            return mapped;
        }

        static bool same_shape(const ggml_tensor *a, const ggml_tensor *b)
        {
            for (int i = 0; i < 4; ++i)
            {
                if (a->ne[i] != b->ne[i])
                {
                    return false;
                }
            }
            return true;
        }

        static std::vector<int64_t> to_i64_vector(const DynamicModel::HostTensor &host)
        {
            std::vector<int64_t> values;
            if (host.bytes.empty())
            {
                return values;
            }
            const size_t element_count = host.bytes.size() / ggml_type_size(static_cast<ggml_type>(host.type));
            values.resize(element_count);

            if (host.type == GGML_TYPE_I64)
            {
                std::memcpy(values.data(), host.bytes.data(), element_count * sizeof(int64_t));
                return values;
            }
            if (host.type == GGML_TYPE_I32)
            {
                const auto *src = reinterpret_cast<const int32_t *>(host.bytes.data());
                for (size_t i = 0; i < element_count; ++i)
                {
                    values[i] = src[i];
                }
                return values;
            }
            if (host.type == GGML_TYPE_F32)
            {
                const auto *src = reinterpret_cast<const float *>(host.bytes.data());
                for (size_t i = 0; i < element_count; ++i)
                {
                    values[i] = static_cast<int64_t>(src[i]);
                }
                return values;
            }
            if (host.type == GGML_TYPE_F16)
            {
                const auto *src = reinterpret_cast<const ggml_fp16_t *>(host.bytes.data());
                for (size_t i = 0; i < element_count; ++i)
                {
                    values[i] = static_cast<int64_t>(ggml_fp16_to_fp32(src[i]));
                }
                return values;
            }

            throw std::runtime_error("unsupported shape tensor type");
        }

        static std::vector<float> to_f32_vector(const DynamicModel::HostTensor &host)
        {
            std::vector<float> values;
            if (host.bytes.empty())
            {
                return values;
            }
            const size_t element_count = host.bytes.size() / ggml_type_size(static_cast<ggml_type>(host.type));
            values.resize(element_count);

            if (host.type == GGML_TYPE_F32)
            {
                std::memcpy(values.data(), host.bytes.data(), element_count * sizeof(float));
                return values;
            }
            if (host.type == GGML_TYPE_F16)
            {
                const auto *src = reinterpret_cast<const ggml_fp16_t *>(host.bytes.data());
                for (size_t i = 0; i < element_count; ++i)
                {
                    values[i] = ggml_fp16_to_fp32(src[i]);
                }
                return values;
            }
            if (host.type == GGML_TYPE_I64)
            {
                const auto *src = reinterpret_cast<const int64_t *>(host.bytes.data());
                for (size_t i = 0; i < element_count; ++i)
                {
                    values[i] = static_cast<float>(src[i]);
                }
                return values;
            }

            throw std::runtime_error("unsupported scale tensor type");
        }

        static std::vector<int64_t> tensor_to_onnx_dims(const ggml_tensor *t, int rank)
        {
            std::vector<int64_t> dims;
            dims.reserve(rank);
            for (int i = 0; i < rank && i < 4; ++i)
            {
                dims.push_back(t->ne[rank - 1 - i]);
            }
            return dims;
        }

        static ggml_type onnx_tensor_type_to_ggml(int onnx_to)
        {
            // ONNX TensorProto DataType values used in this project:
            // 1 = FLOAT, 6 = INT32, 7 = INT64, 10 = FLOAT16
            switch (onnx_to)
            {
            case 1:
                return GGML_TYPE_F32;
            case 6:
                return GGML_TYPE_I32;
            case 7:
                return GGML_TYPE_I64;
            case 10:
                return GGML_TYPE_F16;
            default:
                throw std::runtime_error("unsupported ONNX cast target type");
            }
        }

        static ggml_tensor *maybe_broadcast_to(ggml_context *ctx, ggml_tensor *candidate, ggml_tensor *target)
        {
            if (same_shape(candidate, target))
            {
                return candidate;
            }
            if (ggml_can_repeat(candidate, target))
            {
                return ggml_repeat(ctx, candidate, target);
            }
            return candidate;
        }

    } // namespace

    DynamicModel::DynamicModel() = default;

    DynamicModel::~DynamicModel()
    {
        unload();
    }

    bool DynamicModel::set_error(const std::string &message)
    {
        last_error_ = message;
        return false;
    }

    bool DynamicModel::is_loaded() const
    {
        return loaded_;
    }

    const std::string &DynamicModel::last_error() const
    {
        return last_error_;
    }

    const std::string &DynamicModel::backend_name() const
    {
        return backend_name_;
    }

    const std::string &DynamicModel::backend_device_name() const
    {
        return backend_device_name_;
    }

    const std::vector<DynamicModel::TensorInfo> &DynamicModel::input_tensors() const
    {
        return graph_inputs_;
    }

    const std::vector<DynamicModel::TensorInfo> &DynamicModel::output_tensors() const
    {
        return graph_outputs_;
    }

    const std::unordered_map<std::string, std::string> &DynamicModel::model_metadata() const
    {
        return model_metadata_;
    }

    std::string DynamicModel::model_metadata_value(const std::string &key, const std::string &fallback) const
    {
        const auto it = model_metadata_.find(key);
        return it != model_metadata_.end() ? it->second : fallback;
    }

    size_t DynamicModel::weight_buffer_bytes() const
    {
        return weight_buffer_bytes_;
    }

    size_t DynamicModel::last_compute_buffer_bytes() const
    {
        return last_compute_buffer_bytes_;
    }

    size_t DynamicModel::last_compute_peak_bytes() const
    {
        return last_compute_peak_bytes_;
    }

    int64_t DynamicModel::input_elements() const
    {
        if (graph_inputs_.empty())
        {
            return 0;
        }
        return numel_from_dims(graph_inputs_.front().dims);
    }

    bool DynamicModel::load(const std::string &model_dir, const std::string &model_name)
    {
        unload();

        model_dir_ = model_dir;
        model_name_ = model_name;
        backend_name_.clear();
        backend_device_name_.clear();

        const std::string graph_path = model_dir + "/" + model_name + "_graph.json";
        const std::string weights_path = model_dir + "/" + model_name + "_weights.gguf";

        if (!load_graph_json(graph_path))
        {
            return false;
        }
        if (!load_weights_gguf(weights_path))
        {
            const std::string error = last_error_;
            unload();
            return set_error(error);
        }
        if (!build_compute_graph())
        {
            const std::string error = last_error_;
            unload();
            return set_error(error);
        }

        loaded_ = true;
        return true;
    }

    void DynamicModel::unload()
    {
        loaded_ = false;
        last_error_.clear();

        nodes_.clear();
        graph_inputs_.clear();
        graph_outputs_.clear();
        graph_initializers_.clear();
        weight_tensors_.clear();
        tensor_rank_by_name_.clear();
        model_metadata_.clear();
        node_output_names_.clear();
        node_output_tensors_.clear();
        weight_buffer_bytes_ = 0;
        last_compute_buffer_bytes_ = 0;
        last_compute_peak_bytes_ = 0;

        input_tensor_ = nullptr;
        output_tensors_.clear();
        compute_graph_ = nullptr;

        if (compute_allocr_ != nullptr)
        {
            ggml_gallocr_free(reinterpret_cast<ggml_gallocr_t>(compute_allocr_));
            compute_allocr_ = nullptr;
        }
        if (compute_ctx_ != nullptr)
        {
            ggml_free(compute_ctx_);
            compute_ctx_ = nullptr;
        }

        if (weight_buffer_ != nullptr)
        {
            ggml_backend_buffer_free(reinterpret_cast<ggml_backend_buffer_t>(weight_buffer_));
            weight_buffer_ = nullptr;
        }
        if (weight_ctx_ != nullptr)
        {
            ggml_free(weight_ctx_);
            weight_ctx_ = nullptr;
        }
        if (backend_ != nullptr)
        {
            ggml_backend_free(reinterpret_cast<ggml_backend_t>(backend_));
            backend_ = nullptr;
        }
    }

    bool DynamicModel::load_graph_json(const std::string &graph_path)
    {
        std::ifstream file(graph_path);
        if (!file)
        {
            return set_error("failed to open graph json: " + graph_path);
        }

        std::stringstream buffer;
        buffer << file.rdbuf();

        {
            JsonValue root;
            try
            {
                root = JsonParser(buffer.str()).parse();
            }
            catch (const std::exception &e)
            {
                return set_error(std::string("failed to parse graph json: ") + e.what());
            }

            model_metadata_.clear();
            const JsonValue *model_metadata = find_obj_key(root, "model_metadata");
            if (model_metadata != nullptr && model_metadata->type == JsonValue::Type::Object)
            {
                for (const auto &entry : model_metadata->object)
                {
                    std::string value;
                    if (!json_scalar_to_string(entry.second, value))
                    {
                        continue;
                    }
                    model_metadata_[entry.first] = value;
                }
            }

            const JsonValue *graph = find_obj_key(root, "graph");
            if (graph == nullptr || graph->type != JsonValue::Type::Object)
            {
                return set_error("graph json missing graph object");
            }

            auto parse_tensor_infos = [](const JsonValue *array_ptr)
            {
                std::vector<TensorInfo> result;
                if (array_ptr == nullptr || array_ptr->type != JsonValue::Type::Array)
                {
                    return result;
                }
                for (const JsonValue &item : array_ptr->array)
                {
                    if (item.type != JsonValue::Type::Object)
                    {
                        continue;
                    }
                    TensorInfo info;
                    info.name = get_string(item, "name", "");
                    info.data_type = get_string(item, "data_type", "");
                    info.dims = get_i64_array(item, "dims");
                    result.push_back(std::move(info));
                }
                return result;
            };

            graph_inputs_ = parse_tensor_infos(find_obj_key(*graph, "inputs"));
            graph_outputs_ = parse_tensor_infos(find_obj_key(*graph, "outputs"));
            graph_initializers_ = parse_tensor_infos(find_obj_key(*graph, "initializers"));

            tensor_rank_by_name_.clear();
            auto register_tensor_rank = [&](const std::vector<TensorInfo> &infos)
            {
                for (const TensorInfo &info : infos)
                {
                    if (info.name.empty())
                    {
                        continue;
                    }
                    tensor_rank_by_name_[info.name] = static_cast<int>(info.dims.size());
                }
            };
            register_tensor_rank(graph_inputs_);
            register_tensor_rank(graph_outputs_);
            register_tensor_rank(graph_initializers_);
            register_tensor_rank(parse_tensor_infos(find_obj_key(*graph, "value_info")));

            nodes_.clear();
            const JsonValue *nodes_value = find_obj_key(*graph, "nodes");
            if (nodes_value == nullptr || nodes_value->type != JsonValue::Type::Array)
            {
                return set_error("graph json missing nodes array");
            }

            nodes_.reserve(nodes_value->array.size());
            for (const JsonValue &node_json : nodes_value->array)
            {
                if (node_json.type != JsonValue::Type::Object)
                {
                    continue;
                }

                NodeDef node;
                node.name = get_string(node_json, "name", "");
                node.op_type = get_string(node_json, "op_type", "");
                node.inputs = get_string_array(node_json, "inputs");
                node.outputs = get_string_array(node_json, "outputs");

                const JsonValue *attrs = find_obj_key(node_json, "attributes");
                if (attrs != nullptr && attrs->type == JsonValue::Type::Object)
                {
                    node.axis = static_cast<int>(get_i64(*attrs, "axis", 0));
                    node.axes = get_i64_array(*attrs, "axes");
                    node.keepdims = static_cast<int>(get_i64(*attrs, "keepdims", 1));
                    node.split = get_i64_array(*attrs, "split");
                    node.perm = to_i32_vec(get_i64_array(*attrs, "perm"));
                    node.strides = to_i32_vec(get_i64_array(*attrs, "strides"));
                    node.dilations = to_i32_vec(get_i64_array(*attrs, "dilations"));
                    node.pads = to_i32_vec(get_i64_array(*attrs, "pads"));
                    node.kernel_shape = to_i32_vec(get_i64_array(*attrs, "kernel_shape"));
                    node.auto_pad = get_string(*attrs, "auto_pad", "");
                    node.group = static_cast<int>(get_i64(*attrs, "group", 1));
                    node.alpha = static_cast<float>(get_f64(*attrs, "alpha", 0.01));
                    node.epsilon = static_cast<float>(get_f64(*attrs, "epsilon", 1e-5));
                    node.momentum = static_cast<float>(get_f64(*attrs, "momentum", 0.9));
                    node.to = static_cast<int>(get_i64(*attrs, "to", 0));
                    node.largest = static_cast<int>(get_i64(*attrs, "largest", 1));
                    node.sorted = static_cast<int>(get_i64(*attrs, "sorted", 1));
                    node.fmod = static_cast<int>(get_i64(*attrs, "fmod", 0));

                    const JsonValue *value = find_obj_key(*attrs, "value");
                    if (value != nullptr && value->type == JsonValue::Type::Object)
                    {
                        node.const_value_name = get_string(*value, "name", "");
                    }
                }

                nodes_.push_back(std::move(node));
            }
        }

        std::stringstream().swap(buffer);

        return true;
    }

    bool DynamicModel::load_weights_gguf(const std::string &weights_path)
    {
        ggml_backend_load_all();

        const char *backend_env = std::getenv("MINI2GGUF_BACKEND");
        std::string backend_request = backend_env != nullptr ? to_lower_copy(backend_env) : "auto";
        if (backend_request.empty())
        {
            backend_request = "auto";
        }

        ggml_backend_t backend = nullptr;
        if (backend_request == "auto")
        {
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
            if (backend == nullptr)
            {
                backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
            }
        }
        else if (backend_request == "cpu")
        {
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        }
        else if (backend_request == "gpu")
        {
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        }
        else if (backend_request.rfind("vulkan", 0) == 0)
        {
            const int index = parse_trailing_index(backend_request, "vulkan");
            backend = init_backend_by_name_and_index("vulkan", index);
        }
        else if (backend_request.rfind("cuda", 0) == 0)
        {
            const int index = parse_trailing_index(backend_request, "cuda");
            backend = init_backend_by_name_and_index("cuda", index);
        }
        else
        {
            return set_error("unsupported backend request: " + backend_request + " (use auto|cpu|gpu|vulkanN|cudaN)");
        }

        if (backend == nullptr)
        {
            return set_error("failed to init ggml backend for request: " + backend_request);
        }
        backend_ = reinterpret_cast<ggml_backend *>(backend);
        int n_threads = std::max(1U, std::thread::hardware_concurrency() / 2);
        if (const char *env_threads = std::getenv("MINI2GGUF_NUM_THREADS"))
        {
            const int parsed = std::atoi(env_threads);
            if (parsed > 0)
            {
                n_threads = parsed;
            }
        }
        fprintf(stderr, "mini2gguf cpu threads=%d\n", n_threads);

        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        backend_name_ = ggml_backend_name(backend);
        if (dev != nullptr)
        {
            const char *dev_name = ggml_backend_dev_name(dev);
            const char *dev_desc = ggml_backend_dev_description(dev);
            backend_device_name_ = (dev_name ? dev_name : "unknown");
            if (dev_desc != nullptr && std::strlen(dev_desc) > 0)
            {
                backend_device_name_ += " (";
                backend_device_name_ += dev_desc;
                backend_device_name_ += ")";
            }
        }
        else
        {
            backend_device_name_ = "unknown";
        }

        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg)
        {
            auto ggml_backend_set_n_threads_fn =
                (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (ggml_backend_set_n_threads_fn)
            {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
        }

        ggml_context *tmp_ctx = nullptr;
        gguf_init_params params = {
            false,
            &tmp_ctx,
        };

        gguf_context *gguf_ctx = gguf_init_from_file(weights_path.c_str(), params);
        if (gguf_ctx == nullptr)
        {
            return set_error("failed to load gguf: " + weights_path);
        }

        const int64_t n_kv = gguf_get_n_kv(gguf_ctx);
        for (int64_t i = 0; i < n_kv; ++i)
        {
            const char *key = gguf_get_key(gguf_ctx, i);
            if (key == nullptr)
            {
                continue;
            }
            std::string value;
            if (!gguf_kv_value_to_string(gguf_ctx, i, value))
            {
                continue;
            }
            model_metadata_[key] = std::move(value);
        }

        auto cleanup_gguf = [&]()
        {
            if (gguf_ctx != nullptr)
            {
                gguf_free(gguf_ctx);
                gguf_ctx = nullptr;
            }
            if (tmp_ctx != nullptr)
            {
                ggml_free(tmp_ctx);
                tmp_ctx = nullptr;
            }
        };

        const int n_tensors = gguf_get_n_tensors(gguf_ctx);
        ggml_init_params init_params = {
            static_cast<size_t>(ggml_tensor_overhead() * (n_tensors + 8)),
            nullptr,
            true,
        };

        weight_ctx_ = ggml_init(init_params);
        if (weight_ctx_ == nullptr)
        {
            cleanup_gguf();
            return set_error("failed to init weight context");
        }

        for (int i = 0; i < n_tensors; ++i)
        {
            const char *name_c = gguf_get_tensor_name(gguf_ctx, i);
            if (name_c == nullptr)
            {
                continue;
            }
            const std::string name(name_c);

            ggml_tensor *src = ggml_get_tensor(tmp_ctx, name.c_str());
            if (src == nullptr)
            {
                continue;
            }

            ggml_tensor *dst = ggml_dup_tensor(weight_ctx_, src);
            ggml_set_name(dst, name.c_str());
            weight_tensors_[name] = dst;
        }

        weight_buffer_ = reinterpret_cast<ggml_backend_buffer *>(ggml_backend_alloc_ctx_tensors(weight_ctx_, backend));
        if (weight_buffer_ == nullptr)
        {
            cleanup_gguf();
            return set_error("failed to alloc backend buffer");
        }
        weight_buffer_bytes_ = ggml_backend_buffer_get_size(reinterpret_cast<ggml_backend_buffer_t>(weight_buffer_));

        for (auto &item : weight_tensors_)
        {
            const std::string &name = item.first;
            ggml_tensor *dst = item.second;
            ggml_tensor *src = ggml_get_tensor(tmp_ctx, name.c_str());
            if (src == nullptr)
            {
                continue;
            }
            ggml_backend_tensor_set(dst, ggml_get_data(src), 0, ggml_nbytes(src));
        }

        cleanup_gguf();
        return true;
    }

    bool DynamicModel::build_compute_graph()
    {
        if (graph_inputs_.empty() || graph_outputs_.empty())
        {
            return set_error("graph has no inputs or outputs");
        }
        const bool debug_node_shape = std::getenv("MINI2GGUF_DEBUG_NODE_SHAPE") != nullptr;
        if (debug_node_shape)
        {
            fprintf(stderr, "[node-shape] build_compute_graph nodes=%zu\n", nodes_.size());
        }

        const TensorInfo &input_info = graph_inputs_.front();
        const int graph_size_hint = std::max<int>(GGML_DEFAULT_GRAPH_SIZE, static_cast<int>(nodes_.size() * 8 + 256));
        ggml_init_params compute_params = {
            static_cast<size_t>(ggml_tensor_overhead() * graph_size_hint + ggml_graph_overhead()),
            nullptr,
            true,
        };
        ggml_context *ctx = ggml_init(compute_params);
        if (ctx == nullptr)
        {
            return set_error("failed to init compute context");
        }

        auto fail_with_cleanup = [&](const std::string &msg)
        {
            ggml_free(ctx);
            return set_error(msg);
        };

        std::unordered_map<std::string, ggml_tensor *> values;
        values.reserve(nodes_.size() * 2 + weight_tensors_.size());

        std::unordered_map<std::string, const NodeDef *> producer_by_output;
        producer_by_output.reserve(nodes_.size() * 2);
        for (const auto &n : nodes_)
        {
            for (const auto &out_name : n.outputs)
            {
                producer_by_output[out_name] = &n;
            }
        }

        auto resolve_host_tensor = [&](const std::string &name, HostTensor &out) -> bool
        {
            auto vit = values.find(name);
            if (vit == values.end())
            {
                return false;
            }
            ggml_tensor *t = vit->second;
            out.type = static_cast<int>(t->type);
            out.ne = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
            const size_t bytes = ggml_nbytes(t);
            out.bytes.resize(bytes);
            if (t->buffer != nullptr)
            {
                ggml_backend_tensor_get(t, out.bytes.data(), 0, bytes);
            }
            else
            {
                std::memcpy(out.bytes.data(), t->data, bytes);
            }
            return true;
        };

        auto try_fuse_silu = [&](const std::string &sigmoid_name, const std::string &x_name) -> ggml_tensor *
        {
            auto pit = producer_by_output.find(sigmoid_name);
            if (pit == producer_by_output.end())
            {
                return nullptr;
            }
            const NodeDef *p = pit->second;
            if (p == nullptr || p->op_type != "Sigmoid" || p->inputs.size() != 1 || p->inputs[0] != x_name)
            {
                return nullptr;
            }
            auto xit = values.find(x_name);
            if (xit == values.end())
            {
                return nullptr;
            }
            ggml_tensor *x = xit->second;
            if (!ggml_is_contiguous(x))
            {
                x = ggml_cont(ctx, x);
            }
            return ggml_silu(ctx, x);
        };

        for (const auto &initializer : graph_initializers_)
        {
            auto it = weight_tensors_.find(initializer.name);
            if (it == weight_tensors_.end())
            {
                return fail_with_cleanup("initializer tensor missing in gguf: " + initializer.name);
            }
            values[initializer.name] = it->second;
        }

        const auto input_ne = onnx_dims_to_ggml_ne(input_info.dims);
        ggml_tensor *input_tensor = ggml_new_tensor_4d(
            ctx,
            GGML_TYPE_F32,
            input_ne[0],
            input_ne[1],
            input_ne[2],
            input_ne[3]);
        ggml_set_name(input_tensor, input_info.name.c_str());
        values[input_info.name] = input_tensor;

        // Some exported graphs are not strictly topologically ordered.
        // Build a dependency-respecting execution order from available tensor names.
        std::vector<const NodeDef *> execution_nodes;
        execution_nodes.reserve(nodes_.size());
        std::vector<bool> scheduled(nodes_.size(), false);
        std::unordered_set<std::string> available;
        available.reserve(graph_initializers_.size() + nodes_.size() * 2 + 8);
        available.insert(input_info.name);
        for (const auto &initializer : graph_initializers_)
        {
            available.insert(initializer.name);
        }

        size_t scheduled_count = 0;
        while (scheduled_count < nodes_.size())
        {
            bool progress = false;
            for (size_t i = 0; i < nodes_.size(); ++i)
            {
                if (scheduled[i])
                {
                    continue;
                }

                const NodeDef &node = nodes_[i];
                if (node.op_type == "Constant")
                {
                    scheduled[i] = true;
                    ++scheduled_count;
                    execution_nodes.push_back(&node);
                    for (const auto &out : node.outputs)
                    {
                        available.insert(out);
                    }
                    progress = true;
                    continue;
                }

                bool ready = true;
                for (const std::string &input_name : node.inputs)
                {
                    if (input_name.empty())
                    {
                        continue;
                    }
                    if (available.find(input_name) == available.end())
                    {
                        ready = false;
                        break;
                    }
                }
                if (!ready)
                {
                    continue;
                }

                scheduled[i] = true;
                ++scheduled_count;
                execution_nodes.push_back(&node);
                for (const auto &out : node.outputs)
                {
                    available.insert(out);
                }
                progress = true;
            }

            if (!progress)
            {
                // Fallback to original order for unresolved remainder;
                // later node execution may still provide explicit error details.
                for (size_t i = 0; i < nodes_.size(); ++i)
                {
                    if (!scheduled[i])
                    {
                        execution_nodes.push_back(&nodes_[i]);
                        scheduled[i] = true;
                        ++scheduled_count;
                    }
                }
            }
        }

        for (const NodeDef *node_ptr : execution_nodes)
        {
            const NodeDef &node = *node_ptr;
            const bool print_node_dims = debug_node_shape;
            auto print_tensor_ne = [&](const char *io_tag, int io_index, const std::string &tensor_name)
            {
                auto it = values.find(tensor_name);
                if (it == values.end() || it->second == nullptr)
                {
                    fprintf(stderr, "[node-shape] %s[%d] %s: <missing>\n", io_tag, io_index, tensor_name.c_str());
                    return;
                }
                const ggml_tensor *t = it->second;
                const char *type_name_c = ggml_type_name(t->type);
                const std::string type_name = to_lower_copy(type_name_c ? std::string(type_name_c) : std::string("unknown"));
                fprintf(stderr,
                        "[node-shape] %s[%d] %s: dim=[%lld,%lld,%lld,%lld] type=%s\n",
                        io_tag,
                        io_index,
                        tensor_name.c_str(),
                        (long long)t->ne[0],
                        (long long)t->ne[1],
                        (long long)t->ne[2],
                        (long long)t->ne[3],
                        type_name.c_str());
            };
            auto print_node_inputs = [&]()
            {
                if (!print_node_dims)
                {
                    return;
                }
                fprintf(stderr, "[node-shape] node=%s op=%s\n", node.name.c_str(), node.op_type.c_str());
                for (size_t i = 0; i < node.inputs.size(); ++i)
                {
                    if (node.inputs[i].empty())
                    {
                        continue;
                    }
                    print_tensor_ne("in", static_cast<int>(i), node.inputs[i]);
                }
            };
            auto print_node_outputs = [&]()
            {
                if (!print_node_dims)
                {
                    return;
                }
                for (size_t i = 0; i < node.outputs.size(); ++i)
                {
                    if (node.outputs[i].empty())
                    {
                        continue;
                    }
                    print_tensor_ne("out", static_cast<int>(i), node.outputs[i]);
                }
            };
            struct ScopeExit
            {
                std::function<void()> fn;
                ~ScopeExit()
                {
                    if (fn)
                    {
                        fn();
                    }
                }
            };
            ScopeExit shape_log_exit{print_node_outputs};

            print_node_inputs();
            if (node.op_type != "Constant")
            {
                for (const std::string &input_name : node.inputs)
                {
                    if (input_name.empty())
                    {
                        continue;
                    }
                    if (values.find(input_name) == values.end())
                    {
                        return fail_with_cleanup("input tensor not ready for node " + node.name + ": " + input_name);
                    }
                }
            }

            if (node.op_type == "Constant")
            {
                if (node.outputs.empty())
                {
                    return fail_with_cleanup("Constant without output: " + node.name);
                }
                auto wit = weight_tensors_.find(node.const_value_name);
                if (wit == weight_tensors_.end())
                {
                    return fail_with_cleanup("constant tensor missing in gguf: " + node.const_value_name);
                }
                values[node.outputs[0]] = wit->second;
                continue;
            }

            if (node.op_type == "Conv")
            {
                ggml_tensor *x = values.at(node.inputs[0]);
                ggml_tensor *w = values.at(node.inputs[1]);
                const bool force_conv_f32 = std::getenv("MINI2GGUF_FORCE_CONV_F32") != nullptr;

                if (force_conv_f32 && w->type == GGML_TYPE_F16)
                {
                    w = ggml_cast(ctx, w, GGML_TYPE_F32);
                    if (x->type == GGML_TYPE_F16)
                    {
                        x = ggml_cast(ctx, x, GGML_TYPE_F32);
                    }
                }

                const std::vector<int> strides = node.strides.empty() ? std::vector<int>{1, 1} : node.strides;
                const std::vector<int> dilations = node.dilations.empty() ? std::vector<int>{1, 1} : node.dilations;
                const std::vector<int> pads = node.pads.empty() ? std::vector<int>{0, 0, 0, 0} : node.pads;
                const std::vector<int> kernel = node.kernel_shape.empty() ? std::vector<int>{static_cast<int>(w->ne[1]), static_cast<int>(w->ne[0])} : node.kernel_shape;

                const int s0 = strides.size() > 1 ? strides[1] : strides[0];
                const int s1 = strides[0];
                const int d0 = dilations.size() > 1 ? dilations[1] : dilations[0];
                const int d1 = dilations[0];

                int p0 = pads.size() > 1 ? pads[1] : pads[0];
                int p1 = pads[0];
                if (node.pads.empty() && !node.auto_pad.empty() && node.auto_pad != "NOTSET")
                {
                    const int k0 = kernel.size() > 1 ? kernel[1] : kernel[0];
                    const int k1 = kernel[0];
                    const int eff_k0 = (k0 - 1) * d0 + 1;
                    const int eff_k1 = (k1 - 1) * d1 + 1;
                    const int64_t out_w = (x->ne[0] + s0 - 1) / s0;
                    const int64_t out_h = (x->ne[1] + s1 - 1) / s1;
                    const int64_t total_pad_w = std::max<int64_t>(0, (out_w - 1) * s0 + eff_k0 - x->ne[0]);
                    const int64_t total_pad_h = std::max<int64_t>(0, (out_h - 1) * s1 + eff_k1 - x->ne[1]);

                    const bool same_lower = node.auto_pad == "SAME_LOWER";
                    p0 = static_cast<int>(same_lower ? (total_pad_w + 1) / 2 : total_pad_w / 2);
                    p1 = static_cast<int>(same_lower ? (total_pad_h + 1) / 2 : total_pad_h / 2);
                }

                ggml_tensor *y = nullptr;
                ggml_tensor *conv_bias = node.inputs.size() >= 3 ? values.at(node.inputs[2]) : nullptr;
                bool fused_conv_bias = false;
                if (node.group == 1)
                {
                    if (w->ne[2] != x->ne[2])
                    {
                        std::ostringstream oss;
                        oss << "Conv channel mismatch in " << node.name
                            << " weight.ne2=" << w->ne[2]
                            << " input.ne2=" << x->ne[2]
                            << " (input=" << node.inputs[0] << ", weight=" << node.inputs[1] << ")";
                        return fail_with_cleanup(oss.str());
                    }
                    const bool disable_cpu_conv_direct = env_flag_enabled("MINI2GGUF_DISABLE_CPU_CONV_DIRECT");
                    const ggml_backend_t backend = reinterpret_cast<ggml_backend_t>(backend_);
                    const ggml_backend_dev_t dev = backend != nullptr ? ggml_backend_get_device(backend) : nullptr;
                    const bool is_cpu_backend = dev != nullptr && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
                    const bool weight_type_ok = w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_F32;
                    const bool input_type_ok = x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_F32;
                    const bool can_use_direct = !disable_cpu_conv_direct && is_cpu_backend && weight_type_ok && input_type_ok;

                    if (can_use_direct)
                    {
                        ggml_tensor *w_direct = w;
                        if (!ggml_is_contiguous(w_direct))
                        {
                            w_direct = ggml_cont(ctx, w_direct);
                        }

                        ggml_tensor *x_direct = x;
                        if (x_direct->type != GGML_TYPE_F32)
                        {
                            x_direct = ggml_cast(ctx, x_direct, GGML_TYPE_F32);
                        }

                        ggml_tensor *bias_direct = nullptr;
                        if (conv_bias != nullptr)
                        {
                            const bool bias_1d = ggml_n_dims(conv_bias) == 1 && conv_bias->ne[0] == w_direct->ne[3];
                            const bool bias_4d = ggml_n_dims(conv_bias) == 4 &&
                                conv_bias->ne[0] == 1 &&
                                conv_bias->ne[1] == 1 &&
                                conv_bias->ne[2] == w_direct->ne[3] &&
                                (conv_bias->ne[3] == 1 || conv_bias->ne[3] == x_direct->ne[3]);
                            if (bias_1d || bias_4d)
                            {
                                bias_direct = conv_bias;
                                if (bias_direct->type != x_direct->type)
                                {
                                    bias_direct = ggml_cast(ctx, bias_direct, x_direct->type);
                                }
                            }
                        }

                        if (bias_direct != nullptr)
                        {
                            y = ggml_conv_2d_direct_bias(ctx, w_direct, x_direct, bias_direct, s0, s1, p0, p1, d0, d1);
                            fused_conv_bias = true;
                        }
                        else
                        {
                            y = ggml_conv_2d_direct(ctx, w_direct, x_direct, s0, s1, p0, p1, d0, d1);
                        }
                    }
                    else
                    {
                        y = ggml_conv_2d(ctx, w, x, s0, s1, p0, p1, d0, d1);
                    }
                }
                else
                {
                    const bool is_depthwise =
                        (node.group == x->ne[2]) &&
                        (w->ne[2] == 1) &&
                        (w->ne[3] == x->ne[2]);
                    if (!is_depthwise)
                    {
                        return fail_with_cleanup("Conv grouped mode unsupported (non-depthwise): " + node.name);
                    }
                    const bool disable_cpu_conv_direct = env_flag_enabled("MINI2GGUF_DISABLE_CPU_CONV_DIRECT");
                    const ggml_backend_t backend = reinterpret_cast<ggml_backend_t>(backend_);
                    const ggml_backend_dev_t dev = backend != nullptr ? ggml_backend_get_device(backend) : nullptr;
                    const bool is_cpu_backend = dev != nullptr && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
                    const bool can_use_dw_direct = !disable_cpu_conv_direct && is_cpu_backend;

                    if (can_use_dw_direct)
                    {
                        ggml_tensor *w_direct = w;
                        ggml_tensor *x_direct = x;

                        if (w_direct->type != GGML_TYPE_F32)
                        {
                            w_direct = ggml_cast(ctx, w_direct, GGML_TYPE_F32);
                        }
                        if (x_direct->type != GGML_TYPE_F32)
                        {
                            x_direct = ggml_cast(ctx, x_direct, GGML_TYPE_F32);
                        }

                        if (!ggml_is_contiguous(w_direct))
                        {
                            w_direct = ggml_cont(ctx, w_direct);
                        }
                        if (!ggml_is_contiguous(x_direct) && !ggml_is_contiguous_channels(x_direct))
                        {
                            x_direct = ggml_cont(ctx, x_direct);
                        }

                        ggml_tensor *bias_direct = nullptr;
                        if (conv_bias != nullptr)
                        {
                            const bool bias_1d = ggml_n_dims(conv_bias) == 1 && conv_bias->ne[0] == x_direct->ne[2];
                            const bool bias_4d = ggml_n_dims(conv_bias) == 4 &&
                                conv_bias->ne[0] == 1 &&
                                conv_bias->ne[1] == 1 &&
                                conv_bias->ne[2] == x_direct->ne[2] &&
                                (conv_bias->ne[3] == 1 || conv_bias->ne[3] == x_direct->ne[3]);
                            if (bias_1d || bias_4d)
                            {
                                bias_direct = conv_bias;
                                if (bias_direct->type != x_direct->type)
                                {
                                    bias_direct = ggml_cast(ctx, bias_direct, x_direct->type);
                                }
                            }
                        }

                        if (bias_direct != nullptr)
                        {
                            y = ggml_conv_2d_dw_direct_bias(ctx, w_direct, x_direct, bias_direct, s0, s1, p0, p1, d0, d1);
                            fused_conv_bias = true;
                        }
                        else
                        {
                            y = ggml_conv_2d_dw_direct(ctx, w_direct, x_direct, s0, s1, p0, p1, d0, d1);
                        }
                    }
                    else
                    {
                        y = ggml_conv_2d_dw(ctx, w, x, s0, s1, p0, p1, d0, d1);
                    }
                }
                if (node.inputs.size() >= 3 && !fused_conv_bias)
                {
                    ggml_tensor *b = values.at(node.inputs[2]);
                    ggml_tensor *b_for_add = b;
                    if (ggml_n_dims(b_for_add) == 1)
                    {
                        b_for_add = ggml_reshape_4d(ctx, b_for_add, 1, 1, b_for_add->ne[0], 1);
                    }
                    if (!ggml_can_repeat(b_for_add, y))
                    {
                        return fail_with_cleanup("Conv bias cannot be broadcast to output tensor: " + node.name);
                    }
                    if (b_for_add->type != y->type)
                    {
                        b_for_add = ggml_cast(ctx, b_for_add, y->type);
                    }
                    // ggml_add supports broadcast (ggml_can_repeat on src1), so avoid
                    // materializing an explicit REPEAT tensor for Conv bias.
                    y = ggml_add(ctx, y, b_for_add);
                }
                values[node.outputs[0]] = y;
                continue;
            }

            if (node.op_type == "Sigmoid")
            {
                ggml_tensor *x = values.at(node.inputs[0]);
                if (!ggml_is_contiguous(x))
                {
                    x = ggml_cont(ctx, x);
                }
                values[node.outputs[0]] = ggml_sigmoid(ctx, x);
                continue;
            }

            if (node.op_type == "LeakyRelu")
            {
                ggml_tensor *x = values.at(node.inputs[0]);
                values[node.outputs[0]] = ggml_leaky_relu(ctx, x, node.alpha, false);
                continue;
            }

            if (node.op_type == "BatchNormalization")
            {
                if (node.inputs.size() < 5)
                {
                    return fail_with_cleanup("BatchNormalization requires 5 inputs: " + node.name);
                }

                ggml_tensor *x = values.at(node.inputs[0]);
                ggml_tensor *scale = values.at(node.inputs[1]);
                ggml_tensor *bias = values.at(node.inputs[2]);
                ggml_tensor *mean = values.at(node.inputs[3]);
                ggml_tensor *var = values.at(node.inputs[4]);

                auto repeat_to_x = [&](ggml_tensor *t, const char *label) -> ggml_tensor *
                {
                    ggml_tensor *out = t;
                    if (out->type != x->type)
                    {
                        out = ggml_cast(ctx, out, x->type);
                    }
                    if (ggml_n_dims(out) == 1)
                    {
                        out = ggml_reshape_4d(ctx, out, 1, 1, out->ne[0], 1);
                    }
                    if (!ggml_can_repeat(out, x))
                    {
                        std::ostringstream oss;
                        oss << "BatchNormalization " << label << " cannot be broadcast: " << node.name;
                        throw std::runtime_error(oss.str());
                    }
                    return ggml_repeat(ctx, out, x);
                };

                ggml_tensor *scale_r = repeat_to_x(scale, "scale");
                ggml_tensor *bias_r = repeat_to_x(bias, "bias");
                ggml_tensor *mean_r = repeat_to_x(mean, "mean");
                ggml_tensor *var_r = repeat_to_x(var, "var");

                ggml_tensor *centered = ggml_sub(ctx, x, mean_r);
                ggml_tensor *stddev = ggml_sqrt(ctx, var_r);
                ggml_tensor *normalized = ggml_div(ctx, centered, stddev);
                ggml_tensor *scaled = ggml_mul(ctx, normalized, scale_r);
                values[node.outputs[0]] = ggml_add(ctx, scaled, bias_r);
                continue;
            }

            if (node.op_type == "Cast")
            {
                ggml_tensor *x = values.at(node.inputs[0]);
                ggml_type to_type = GGML_TYPE_COUNT;
                const bool skip_input_f16_cast = std::getenv("MINI2GGUF_SKIP_INPUT_F16_CAST") != nullptr;
                try
                {
                    to_type = onnx_tensor_type_to_ggml(node.to);
                }
                catch (const std::exception &)
                {
                    return fail_with_cleanup("unsupported Cast target type: " + std::to_string(node.to) + " in " + node.name);
                }

                if (x->type == to_type)
                {
                    values[node.outputs[0]] = x;
                    continue;
                }

                if (skip_input_f16_cast &&
                    node.name == "graph_input_cast0" &&
                    (to_type == GGML_TYPE_F16 || to_type == GGML_TYPE_BF16))
                {
                    values[node.outputs[0]] = x;
                    continue;
                }

                if ((x->type == GGML_TYPE_I32 || x->type == GGML_TYPE_I16 || x->type == GGML_TYPE_I8) && to_type == GGML_TYPE_F16)
                {
                    ggml_tensor *tmp = ggml_cast(ctx, x, GGML_TYPE_F32);
                    values[node.outputs[0]] = ggml_cast(ctx, tmp, GGML_TYPE_F16);
                    continue;
                }

                values[node.outputs[0]] = ggml_cast(ctx, x, to_type);
                continue;
            }

            if (node.op_type == "Sub")
            {
                ggml_tensor *a = values.at(node.inputs[0]);
                ggml_tensor *b = values.at(node.inputs[1]);
                int rank_a = 4;
                int rank_b = 4;
                if (auto it = tensor_rank_by_name_.find(node.inputs[0]); it != tensor_rank_by_name_.end() && it->second >= 1 && it->second <= 4)
                {
                    rank_a = it->second;
                }
                if (auto it = tensor_rank_by_name_.find(node.inputs[1]); it != tensor_rank_by_name_.end() && it->second >= 1 && it->second <= 4)
                {
                    rank_b = it->second;
                }
                if (rank_a != rank_b)
                {
                    auto align_rank = [&](ggml_tensor *t, int from_rank, int to_rank) -> ggml_tensor *
                    {
                        if (from_rank >= to_rank)
                        {
                            return t;
                        }
                        std::vector<int64_t> dims = tensor_to_onnx_dims(t, from_rank);
                        while ((int)dims.size() < to_rank)
                        {
                            dims.insert(dims.begin(), 1);
                        }
                        if (!ggml_is_contiguous(t))
                        {
                            t = ggml_cont(ctx, t);
                        }
                        const auto ne = onnx_dims_to_ggml_ne(dims);
                        if (to_rank == 2)
                        {
                            return ggml_reshape_2d(ctx, t, ne[0], ne[1]);
                        }
                        if (to_rank == 3)
                        {
                            return ggml_reshape_3d(ctx, t, ne[0], ne[1], ne[2]);
                        }
                        return ggml_reshape_4d(ctx, t, ne[0], ne[1], ne[2], ne[3]);
                    };
                    if (rank_a < rank_b)
                    {
                        a = align_rank(a, rank_a, rank_b);
                    }
                    else
                    {
                        b = align_rank(b, rank_b, rank_a);
                    }
                }
                if (a->type != b->type)
                {
                    a = ggml_cast(ctx, a, GGML_TYPE_F32);
                    b = ggml_cast(ctx, b, GGML_TYPE_F32);
                }
                a = maybe_broadcast_to(ctx, a, b);
                b = maybe_broadcast_to(ctx, b, a);
                if (!ggml_can_repeat(b, a))
                {
                    if (ggml_nelements(a) == ggml_nelements(b))
                    {
                        if (!ggml_is_contiguous(a))
                        {
                            a = ggml_cont(ctx, a);
                        }
                        if (!ggml_is_contiguous(b))
                        {
                            b = ggml_cont(ctx, b);
                        }
                        b = ggml_reshape_4d(ctx, b, a->ne[0], a->ne[1], a->ne[2], a->ne[3]);
                    }
                    else
                    {
                        std::ostringstream oss;
                        oss << "Sub broadcast mismatch in " << node.name
                            << " a.ne=" << a->ne[0] << "," << a->ne[1] << "," << a->ne[2] << "," << a->ne[3]
                            << " b.ne=" << b->ne[0] << "," << b->ne[1] << "," << b->ne[2] << "," << b->ne[3];
                        return fail_with_cleanup(oss.str());
                    }
                }
                values[node.outputs[0]] = ggml_sub(ctx, a, b);
                continue;
            }

            if (node.op_type == "Div")
            {
                ggml_tensor *a = values.at(node.inputs[0]);
                ggml_tensor *b = values.at(node.inputs[1]);

                if (a->type == GGML_TYPE_I32)
                {
                    HostTensor b_host;
                    if (!resolve_host_tensor(node.inputs[1], b_host))
                    {
                        return fail_with_cleanup("Div integer path expects scalar divisor: " + node.name);
                    }
                    const std::vector<int64_t> b_vals = to_i64_vector(b_host);
                    if (b_vals.empty())
                    {
                        return fail_with_cleanup("Div divisor tensor is empty: " + node.name);
                    }
                    const float divisor = static_cast<float>(b_vals[0]);
                    if (std::abs(divisor) < 1e-12f)
                    {
                        return fail_with_cleanup("Div divisor is zero: " + node.name);
                    }

                    ggml_tensor *a_f = ggml_cast(ctx, a, GGML_TYPE_F32);
                    ggml_tensor *q_f = ggml_floor(ctx, ggml_scale(ctx, a_f, 1.0f / divisor));
                    values[node.outputs[0]] = ggml_cast(ctx, q_f, GGML_TYPE_I32);
                    continue;
                }

                if (a->type != b->type)
                {
                    a = ggml_cast(ctx, a, GGML_TYPE_F32);
                    b = ggml_cast(ctx, b, GGML_TYPE_F32);
                }
                a = maybe_broadcast_to(ctx, a, b);
                b = maybe_broadcast_to(ctx, b, a);
                values[node.outputs[0]] = ggml_div(ctx, a, b);
                continue;
            }

            if (node.op_type == "Mod")
            {
                ggml_tensor *a = values.at(node.inputs[0]);
                ggml_tensor *b = values.at(node.inputs[1]);
                if (node.fmod != 0)
                {
                    return fail_with_cleanup("Mod fmod=1 is not supported: " + node.name);
                }

                const auto is_integral = [](ggml_type t)
                {
                    return t == GGML_TYPE_I8 || t == GGML_TYPE_I16 || t == GGML_TYPE_I32 || t == GGML_TYPE_I64;
                };

                if (!is_integral(a->type) || !is_integral(b->type))
                {
                    return fail_with_cleanup("Mod only supports integral tensors (I8/I16/I32/I64): " + node.name);
                }

                if (a->type != b->type)
                {
                    // ggml_mod requires identical input types, while ggml_cast does not
                    // currently support all integer-to-integer casts (e.g. I64->I32).
                    if (a->type == GGML_TYPE_I32 && b->type == GGML_TYPE_I64)
                    {
                        HostTensor b_host;
                        if (!resolve_host_tensor(node.inputs[1], b_host))
                        {
                            return fail_with_cleanup("Mod type mismatch requires const divisor for I64->I32: " + node.name);
                        }
                        const std::vector<int64_t> b_vals = to_i64_vector(b_host);
                        if (b_vals.size() != 1)
                        {
                            return fail_with_cleanup("Mod I64->I32 divisor must be scalar: " + node.name);
                        }

                        const float divisor = static_cast<float>(b_vals[0]);
                        if (std::abs(divisor) < 1e-12f)
                        {
                            return fail_with_cleanup("Mod divisor is zero: " + node.name);
                        }

                        ggml_tensor *a_f = ggml_cast(ctx, a, GGML_TYPE_F32);
                        ggml_tensor *q_f = ggml_floor(ctx, ggml_scale(ctx, a_f, 1.0f / divisor));
                        ggml_tensor *r_f = ggml_sub(ctx, a_f, ggml_scale(ctx, q_f, divisor));
                        values[node.outputs[0]] = ggml_cast(ctx, r_f, GGML_TYPE_I32);
                        continue;
                    }
                    else
                    {
                        return fail_with_cleanup("Mod mixed integer types unsupported without safe cast: " + node.name);
                    }
                }

                a = maybe_broadcast_to(ctx, a, b);
                b = maybe_broadcast_to(ctx, b, a);
                if (!ggml_can_repeat(b, a))
                {
                    return fail_with_cleanup("Mod broadcast mismatch: " + node.name);
                }
                values[node.outputs[0]] = ggml_mod(ctx, a, b);
                continue;
            }

            if (node.op_type == "MatMul")
            {
                if (node.inputs.size() < 2 || node.outputs.empty())
                {
                    return fail_with_cleanup("MatMul expects 2 inputs and 1 output: " + node.name);
                }
                ggml_tensor *a = values.at(node.inputs[0]);
                ggml_tensor *b = values.at(node.inputs[1]);
                auto can_mul_mat = [](const ggml_tensor *t0, const ggml_tensor *t1)
                {
                    return (t0->ne[0] == t1->ne[0]) &&
                           (t1->ne[2] % t0->ne[2] == 0) &&
                           (t1->ne[3] % t0->ne[3] == 0);
                };

                // ONNX MatMul: (..., M, K) x (..., K, N).
                // With ONNX->ggml dim mapping, input B must be transposed before ggml_mul_mat.
                // Otherwise when K==N (e.g. 400x400) shape checks still pass but math becomes A @ B^T.
                ggml_tensor *b_t = ggml_transpose(ctx, b);
                ggml_tensor *lhs = b_t;
                ggml_tensor *rhs = a;
                if (!can_mul_mat(lhs, rhs))
                {
                    // Conservative fallback for non-standard layouts.
                    lhs = b;
                    rhs = a;
                    if (!can_mul_mat(lhs, rhs))
                    {
                        ggml_tensor *a_t = ggml_transpose(ctx, a);
                        if (can_mul_mat(b_t, a_t))
                        {
                            lhs = b_t;
                            rhs = a_t;
                        }
                        else if (can_mul_mat(b, a_t))
                        {
                            lhs = b;
                            rhs = a_t;
                        }
                        else
                        {
                            return fail_with_cleanup("MatMul shape mismatch: " + node.name);
                        }
                    }
                }

                // ggml_mul_mat forbids transposed tensor as src0 ("a").
                if (ggml_is_transposed(lhs))
                {
                    lhs = ggml_cont(ctx, lhs);
                }
                if (ggml_is_transposed(rhs))
                {
                    rhs = ggml_cont(ctx, rhs);
                }

                ggml_tensor *y = ggml_mul_mat(ctx, lhs, rhs);
                values[node.outputs[0]] = y;
                continue;
            }

            if (node.op_type == "Softmax")
            {
                ggml_tensor *x = values.at(node.inputs[0]);
                int input_rank = 4;
                auto rit = tensor_rank_by_name_.find(node.inputs[0]);
                if (rit != tensor_rank_by_name_.end() && rit->second >= 1 && rit->second <= 4)
                {
                    input_rank = rit->second;
                }
                const int axis_ggml = map_onnx_axis_to_ggml(node.axis, input_rank);

                ggml_tensor *x_soft = x;
                int perm[4] = {0, 1, 2, 3};
                int inv_perm[4] = {0, 1, 2, 3};
                if (axis_ggml != 0)
                {
                    int next = 1;
                    perm[0] = axis_ggml;
                    for (int i = 0; i < 4; ++i)
                    {
                        if (i != axis_ggml)
                        {
                            perm[next++] = i;
                        }
                    }
                    for (int i = 0; i < 4; ++i)
                    {
                        inv_perm[perm[i]] = i;
                    }
                    x_soft = ggml_permute(ctx, x, perm[0], perm[1], perm[2], perm[3]);
                }

                ggml_tensor *y = ggml_soft_max(ctx, x_soft);
                if (axis_ggml != 0)
                {
                    y = ggml_permute(ctx, y, inv_perm[0], inv_perm[1], inv_perm[2], inv_perm[3]);
                }
                values[node.outputs[0]] = y;
                continue;
            }

            if (node.op_type == "Slice")
            {
                if (node.inputs.size() < 4)
                {
                    return fail_with_cleanup("Slice expects at least 4 inputs: " + node.name);
                }
                ggml_tensor *x = values.at(node.inputs[0]);
                if (!ggml_is_contiguous(x))
                {
                    // Slice below is built with ggml_view_4d, which cannot represent arbitrary nb[0].
                    // Materialize permuted/view tensors first to keep offsets/strides correct.
                    x = ggml_cont(ctx, x);
                }
                HostTensor starts_t, ends_t, axes_t;
                if (!resolve_host_tensor(node.inputs[1], starts_t) ||
                    !resolve_host_tensor(node.inputs[2], ends_t) ||
                    !resolve_host_tensor(node.inputs[3], axes_t))
                {
                    return fail_with_cleanup("Slice const inputs missing: " + node.name);
                }
                const std::vector<int64_t> starts = to_i64_vector(starts_t);
                const std::vector<int64_t> ends = to_i64_vector(ends_t);
                const std::vector<int64_t> axes = to_i64_vector(axes_t);
                if (starts.size() != 1 || ends.size() != 1 || axes.size() != 1)
                {
                    return fail_with_cleanup("Slice only supports single axis currently: " + node.name);
                }

                int input_rank = 4;
                auto rit = tensor_rank_by_name_.find(node.inputs[0]);
                if (rit != tensor_rank_by_name_.end() && rit->second >= 1 && rit->second <= 4)
                {
                    input_rank = rit->second;
                }
                const int axis_ggml = map_onnx_axis_to_ggml(static_cast<int>(axes[0]), input_rank);
                const int64_t dim = x->ne[axis_ggml];

                int64_t start = starts[0] < 0 ? starts[0] + dim : starts[0];
                int64_t end = ends[0] < 0 ? ends[0] + dim : ends[0];
                start = std::max<int64_t>(0, std::min<int64_t>(start, dim));
                end = std::max<int64_t>(start, std::min<int64_t>(end, dim));

                std::array<int64_t, 4> ne = {x->ne[0], x->ne[1], x->ne[2], x->ne[3]};
                ne[axis_ggml] = end - start;
                const size_t byte_offset = static_cast<size_t>(start) * x->nb[axis_ggml];
                values[node.outputs[0]] = ggml_view_4d(ctx, x, ne[0], ne[1], ne[2], ne[3], x->nb[1], x->nb[2], x->nb[3], byte_offset);
                continue;
            }

            if (node.op_type == "Unsqueeze")
            {
                if (node.inputs.size() < 2)
                {
                    return fail_with_cleanup("Unsqueeze expects data + axes: " + node.name);
                }
                ggml_tensor *x = values.at(node.inputs[0]);
                HostTensor axes_t;
                if (!resolve_host_tensor(node.inputs[1], axes_t))
                {
                    return fail_with_cleanup("Unsqueeze axes tensor missing: " + node.name);
                }
                std::vector<int64_t> axes = to_i64_vector(axes_t);
                if (axes.size() != 1)
                {
                    return fail_with_cleanup("Unsqueeze only supports single axis currently: " + node.name);
                }

                int input_rank = 4;
                auto rit = tensor_rank_by_name_.find(node.inputs[0]);
                if (rit != tensor_rank_by_name_.end() && rit->second >= 1 && rit->second <= 4)
                {
                    input_rank = rit->second;
                }
                std::vector<int64_t> dims = tensor_to_onnx_dims(x, input_rank);
                int axis = static_cast<int>(axes[0]);
                if (axis < 0)
                {
                    axis += input_rank + 1;
                }
                if (axis < 0 || axis > input_rank)
                {
                    return fail_with_cleanup("Unsqueeze axis out of range: " + node.name);
                }
                dims.insert(dims.begin() + axis, 1);

                if (!ggml_is_contiguous(x))
                {
                    x = ggml_cont(ctx, x);
                }
                const auto ne = onnx_dims_to_ggml_ne(dims);
                if (dims.size() == 2)
                {
                    values[node.outputs[0]] = ggml_reshape_2d(ctx, x, ne[0], ne[1]);
                }
                else if (dims.size() == 3)
                {
                    values[node.outputs[0]] = ggml_reshape_3d(ctx, x, ne[0], ne[1], ne[2]);
                }
                else
                {
                    values[node.outputs[0]] = ggml_reshape_4d(ctx, x, ne[0], ne[1], ne[2], ne[3]);
                }
                continue;
            }

            if (node.op_type == "Flatten")
            {
                ggml_tensor *x = values.at(node.inputs[0]);
                int input_rank = 4;
                auto rit = tensor_rank_by_name_.find(node.inputs[0]);
                if (rit != tensor_rank_by_name_.end() && rit->second >= 1 && rit->second <= 4)
                {
                    input_rank = rit->second;
                }
                std::vector<int64_t> dims = tensor_to_onnx_dims(x, input_rank);
                int axis = node.axis;
                if (axis < 0)
                {
                    axis += input_rank;
                }
                if (axis < 0 || axis > input_rank)
                {
                    return fail_with_cleanup("Flatten axis out of range: " + node.name);
                }
                int64_t d0 = 1;
                int64_t d1 = 1;
                for (int i = 0; i < axis; ++i)
                {
                    d0 *= dims[i];
                }
                for (int i = axis; i < input_rank; ++i)
                {
                    d1 *= dims[i];
                }
                if (!ggml_is_contiguous(x))
                {
                    x = ggml_cont(ctx, x);
                }
                const auto ne = onnx_dims_to_ggml_ne({d0, d1});
                values[node.outputs[0]] = ggml_reshape_2d(ctx, x, ne[0], ne[1]);
                continue;
            }

            if (node.op_type == "Tile")
            {
                if (node.inputs.size() < 2)
                {
                    return fail_with_cleanup("Tile expects 2 inputs: " + node.name);
                }
                ggml_tensor *x = values.at(node.inputs[0]);
                HostTensor rep_t;
                if (!resolve_host_tensor(node.inputs[1], rep_t))
                {
                    return fail_with_cleanup("Tile repeats tensor missing: " + node.name);
                }
                const std::vector<int64_t> reps = to_i64_vector(rep_t);

                int input_rank = 4;
                auto rit = tensor_rank_by_name_.find(node.inputs[0]);
                if (rit != tensor_rank_by_name_.end() && rit->second >= 1 && rit->second <= 4)
                {
                    input_rank = rit->second;
                }
                std::vector<int64_t> dims = tensor_to_onnx_dims(x, input_rank);
                if (static_cast<int>(reps.size()) != input_rank)
                {
                    return fail_with_cleanup("Tile repeats rank mismatch: " + node.name);
                }
                for (int i = 0; i < input_rank; ++i)
                {
                    dims[i] *= reps[i];
                }
                const auto ne = onnx_dims_to_ggml_ne(dims);
                values[node.outputs[0]] = ggml_repeat_4d(ctx, x, ne[0], ne[1], ne[2], ne[3]);
                continue;
            }

            if (node.op_type == "TopK")
            {
                if (node.inputs.size() < 2 || node.outputs.size() < 2)
                {
                    return fail_with_cleanup("TopK expects 2 inputs and 2 outputs: " + node.name);
                }
                if (node.largest != 1 || node.sorted != 1)
                {
                    return fail_with_cleanup("TopK only supports largest=1 sorted=1 currently: " + node.name);
                }

                ggml_tensor *x = values.at(node.inputs[0]);
                if (x->type != GGML_TYPE_F32)
                {
                    return fail_with_cleanup("TopK currently requires F32 input: " + node.name);
                }

                HostTensor k_t;
                if (!resolve_host_tensor(node.inputs[1], k_t))
                {
                    return fail_with_cleanup("TopK K tensor missing: " + node.name);
                }
                const std::vector<int64_t> k_vals = to_i64_vector(k_t);
                const int k = k_vals.empty() ? 1 : static_cast<int>(k_vals[0]);
                if (k <= 0)
                {
                    return fail_with_cleanup("TopK invalid k: " + node.name);
                }

                int input_rank = 4;
                auto rit = tensor_rank_by_name_.find(node.inputs[0]);
                if (rit != tensor_rank_by_name_.end() && rit->second >= 1 && rit->second <= 4)
                {
                    input_rank = rit->second;
                }
                const int axis_ggml = map_onnx_axis_to_ggml(node.axis, input_rank);
                if (axis_ggml != 0)
                {
                    return fail_with_cleanup("TopK currently only supports axis mapping to ggml dim0: " + node.name);
                }

                ggml_tensor *indices = ggml_argsort_top_k(ctx, x, k); // I32
                ggml_tensor *values_topk = ggml_gather_elements(ctx, x, indices, 0);
                values[node.outputs[0]] = values_topk;
                values[node.outputs[1]] = indices;
                continue;
            }

            if (node.op_type == "Add" || node.op_type == "Mul")
            {
                if (node.op_type == "Mul" && node.inputs.size() >= 2)
                {
                    if (ggml_tensor *fused = try_fuse_silu(node.inputs[0], node.inputs[1]))
                    {
                        values[node.outputs[0]] = fused;
                        continue;
                    }
                    if (ggml_tensor *fused = try_fuse_silu(node.inputs[1], node.inputs[0]))
                    {
                        values[node.outputs[0]] = fused;
                        continue;
                    }
                }

                ggml_tensor *a = values.at(node.inputs[0]);
                ggml_tensor *b = values.at(node.inputs[1]);
                int rank_a = 4;
                int rank_b = 4;
                if (auto it = tensor_rank_by_name_.find(node.inputs[0]); it != tensor_rank_by_name_.end() && it->second >= 1 && it->second <= 4)
                {
                    rank_a = it->second;
                }
                if (auto it = tensor_rank_by_name_.find(node.inputs[1]); it != tensor_rank_by_name_.end() && it->second >= 1 && it->second <= 4)
                {
                    rank_b = it->second;
                }
                if (rank_a != rank_b)
                {
                    auto align_rank = [&](ggml_tensor *t, int from_rank, int to_rank) -> ggml_tensor *
                    {
                        if (from_rank >= to_rank)
                        {
                            return t;
                        }
                        std::vector<int64_t> dims = tensor_to_onnx_dims(t, from_rank);
                        while ((int)dims.size() < to_rank)
                        {
                            dims.insert(dims.begin(), 1);
                        }
                        if (!ggml_is_contiguous(t))
                        {
                            t = ggml_cont(ctx, t);
                        }
                        const auto ne = onnx_dims_to_ggml_ne(dims);
                        if (to_rank == 2)
                        {
                            return ggml_reshape_2d(ctx, t, ne[0], ne[1]);
                        }
                        if (to_rank == 3)
                        {
                            return ggml_reshape_3d(ctx, t, ne[0], ne[1], ne[2]);
                        }
                        return ggml_reshape_4d(ctx, t, ne[0], ne[1], ne[2], ne[3]);
                    };
                    if (rank_a < rank_b)
                    {
                        a = align_rank(a, rank_a, rank_b);
                    }
                    else
                    {
                        b = align_rank(b, rank_b, rank_a);
                    }
                }

                if (a->type != b->type)
                {
                    a = ggml_cast(ctx, a, GGML_TYPE_F32);
                    b = ggml_cast(ctx, b, GGML_TYPE_F32);
                }
                a = maybe_broadcast_to(ctx, a, b);
                b = maybe_broadcast_to(ctx, b, a);
                if (!ggml_can_repeat(b, a))
                {
                    if (ggml_nelements(a) == ggml_nelements(b))
                    {
                        if (!ggml_is_contiguous(a))
                        {
                            a = ggml_cont(ctx, a);
                        }
                        if (!ggml_is_contiguous(b))
                        {
                            b = ggml_cont(ctx, b);
                        }
                        b = ggml_reshape_4d(ctx, b, a->ne[0], a->ne[1], a->ne[2], a->ne[3]);
                    }
                    else
                    {
                        return fail_with_cleanup("Add/Mul broadcast mismatch: " + node.name);
                    }
                }
                values[node.outputs[0]] = node.op_type == "Add" ? ggml_add(ctx, a, b) : ggml_mul(ctx, a, b);
                continue;
            }

            if (node.op_type == "Pow")
            {
                ggml_tensor *x = values.at(node.inputs[0]);
                HostTensor exp_tensor;
                if (!resolve_host_tensor(node.inputs[1], exp_tensor))
                {
                    return fail_with_cleanup("Pow exponent tensor missing: " + node.inputs[1]);
                }
                const std::vector<float> exp_values = to_f32_vector(exp_tensor);
                const float exponent = exp_values.empty() ? 1.0f : exp_values[0];
                if (std::abs(exponent - 2.0f) < 1e-6f)
                {
                    if (!ggml_is_contiguous(x))
                    {
                        x = ggml_cont(ctx, x);
                    }
                    values[node.outputs[0]] = ggml_sqr(ctx, x);
                }
                else if (std::abs(exponent - 1.0f) < 1e-6f)
                {
                    values[node.outputs[0]] = x;
                }
                else
                {
                    return fail_with_cleanup("unsupported Pow exponent");
                }
                continue;
            }

            if (node.op_type == "MaxPool")
            {
                ggml_tensor *x = values.at(node.inputs[0]);
                const std::vector<int> kernel = node.kernel_shape.empty() ? std::vector<int>{2, 2} : node.kernel_shape;
                const std::vector<int> strides = node.strides.empty() ? std::vector<int>{2, 2} : node.strides;
                const std::vector<int> pads = node.pads.empty() ? std::vector<int>{0, 0, 0, 0} : node.pads;

                const int k0 = kernel.size() > 1 ? kernel[1] : kernel[0];
                const int k1 = kernel[0];
                const int s0 = strides.size() > 1 ? strides[1] : strides[0];
                const int s1 = strides[0];
                float p0 = static_cast<float>(pads.size() > 1 ? pads[1] : pads[0]);
                float p1 = static_cast<float>(pads[0]);
                if (node.pads.empty() && !node.auto_pad.empty() && node.auto_pad != "NOTSET")
                {
                    const int64_t out_w = (x->ne[0] + s0 - 1) / s0;
                    const int64_t out_h = (x->ne[1] + s1 - 1) / s1;
                    const int64_t total_pad_w = std::max<int64_t>(0, (out_w - 1) * s0 + k0 - x->ne[0]);
                    const int64_t total_pad_h = std::max<int64_t>(0, (out_h - 1) * s1 + k1 - x->ne[1]);
                    p0 = static_cast<float>(total_pad_w) * 0.5f;
                    p1 = static_cast<float>(total_pad_h) * 0.5f;
                }
                values[node.outputs[0]] = ggml_pool_2d(ctx, x, GGML_OP_POOL_MAX, k0, k1, s0, s1, p0, p1);
                continue;
            }

            if (node.op_type == "ReduceMax")
            {
                ggml_tensor *x = values.at(node.inputs[0]);
                if (x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_F16)
                {
                    return fail_with_cleanup("ReduceMax only supports F16/F32: " + node.name);
                }

                std::vector<int64_t> axes = node.axes;
                if (axes.empty() && node.inputs.size() >= 2)
                {
                    HostTensor axes_tensor;
                    if (resolve_host_tensor(node.inputs[1], axes_tensor))
                    {
                        axes = to_i64_vector(axes_tensor);
                    }
                }
                if (axes.empty())
                {
                    axes.push_back(0);
                }
                if (axes.size() != 1)
                {
                    return fail_with_cleanup("ReduceMax only supports single axis currently: " + node.name);
                }

                int input_rank = 4;
                auto rit = tensor_rank_by_name_.find(node.inputs[0]);
                if (rit != tensor_rank_by_name_.end() && rit->second >= 1 && rit->second <= 4)
                {
                    input_rank = rit->second;
                }

                const int axis_ggml = map_onnx_axis_to_ggml(static_cast<int>(axes[0]), input_rank);
                if (axis_ggml < 0 || axis_ggml > 3)
                {
                    return fail_with_cleanup("ReduceMax axis out of range: " + node.name);
                }

                ggml_tensor *y = ggml_reduce_max(ctx, x, axis_ggml);

                if (node.keepdims == 0)
                {
                    std::array<int64_t, 4> ne = {y->ne[0], y->ne[1], y->ne[2], y->ne[3]};
                    std::vector<int64_t> squeezed;
                    squeezed.reserve(3);
                    for (int i = 0; i < 4; ++i)
                    {
                        if (i != axis_ggml)
                        {
                            squeezed.push_back(ne[i]);
                        }
                    }

                    if (squeezed.empty())
                    {
                        y = ggml_reshape_1d(ctx, y, 1);
                    }
                    else if (squeezed.size() == 1)
                    {
                        y = ggml_reshape_1d(ctx, y, squeezed[0]);
                    }
                    else if (squeezed.size() == 2)
                    {
                        y = ggml_reshape_2d(ctx, y, squeezed[0], squeezed[1]);
                    }
                    else
                    {
                        y = ggml_reshape_3d(ctx, y, squeezed[0], squeezed[1], squeezed[2]);
                    }
                }
                values[node.outputs[0]] = y;
                continue;
            }

            if (node.op_type == "Transpose")
            {
                ggml_tensor *x = values.at(node.inputs[0]);
                int input_rank = 4;
                if (auto it = tensor_rank_by_name_.find(node.inputs[0]); it != tensor_rank_by_name_.end() && it->second >= 1 && it->second <= 4)
                {
                    input_rank = it->second;
                }

                std::vector<int> perm;
                if (node.perm.empty())
                {
                    perm.resize(input_rank);
                    for (int i = 0; i < input_rank; ++i)
                    {
                        perm[i] = i;
                    }
                }
                else
                {
                    perm = node.perm;
                }

                if (static_cast<int>(perm.size()) != input_rank)
                {
                    return fail_with_cleanup("Transpose perm rank mismatch: " + node.name);
                }

                std::vector<int> mapped_rank = map_onnx_perm_to_ggml(perm);
                std::array<int, 4> mapped = {0, 1, 2, 3};
                for (int i = 0; i < input_rank; ++i)
                {
                    mapped[i] = mapped_rank[i];
                }
                values[node.outputs[0]] = ggml_permute(ctx, x, mapped[0], mapped[1], mapped[2], mapped[3]);
                continue;
            }

            if (node.op_type == "Reshape")
            {
                ggml_tensor *x = values.at(node.inputs[0]);
                HostTensor shape_tensor;
                if (!resolve_host_tensor(node.inputs[1], shape_tensor))
                {
                    return fail_with_cleanup("Reshape shape tensor missing: " + node.inputs[1]);
                }

                std::vector<int64_t> shape = to_i64_vector(shape_tensor);
                if (shape.empty() || shape.size() > 4)
                {
                    return fail_with_cleanup("unsupported reshape rank");
                }

                int input_rank = 4;
                if (auto it = tensor_rank_by_name_.find(node.inputs[0]); it != tensor_rank_by_name_.end() && it->second >= 1 && it->second <= 4)
                {
                    input_rank = it->second;
                }
                std::vector<int64_t> input_dims_onnx = tensor_to_onnx_dims(x, input_rank);
                int64_t known_product = 1;
                int infer_idx = -1;
                for (size_t i = 0; i < shape.size(); ++i)
                {
                    if (shape[i] == 0)
                    {
                        if (i >= input_dims_onnx.size())
                        {
                            return fail_with_cleanup("Reshape uses 0 for out-of-range input axis: " + node.name);
                        }
                        shape[i] = input_dims_onnx[i];
                    }
                    else if (shape[i] == -1)
                    {
                        infer_idx = static_cast<int>(i);
                    }
                    else
                    {
                        known_product *= shape[i];
                    }
                }
                const int64_t total = ggml_nelements(x);
                if (infer_idx >= 0)
                {
                    shape[infer_idx] = total / known_product;
                }

                const auto ne = onnx_dims_to_ggml_ne(shape);
                ggml_tensor *y = nullptr;
                if (!ggml_is_contiguous(x))
                {
                    x = ggml_cont(ctx, x);
                }
                if (shape.size() == 1)
                {
                    y = ggml_reshape_1d(ctx, x, ne[0]);
                }
                else if (shape.size() == 2)
                {
                    y = ggml_reshape_2d(ctx, x, ne[0], ne[1]);
                }
                else if (shape.size() == 3)
                {
                    y = ggml_reshape_3d(ctx, x, ne[0], ne[1], ne[2]);
                }
                else
                {
                    y = ggml_reshape_4d(ctx, x, ne[0], ne[1], ne[2], ne[3]);
                }
                values[node.outputs[0]] = y;
                continue;
            }

            if (node.op_type == "Split")
            {
                ggml_tensor *x = values.at(node.inputs[0]);
                if (!ggml_is_contiguous(x))
                {
                    // Split uses ggml_view_4d; force contiguous source when input comes
                    // from transpose/permute views to avoid stride corruption.
                    x = ggml_cont(ctx, x);
                }
                int input_rank = 4;
                if (auto it = tensor_rank_by_name_.find(node.inputs[0]); it != tensor_rank_by_name_.end() && it->second >= 1 && it->second <= 4)
                {
                    input_rank = it->second;
                }
                const int axis_ggml = map_onnx_axis_to_ggml(node.axis, input_rank);
                std::vector<int64_t> splits = node.split;

                if (splits.empty() && node.inputs.size() >= 2)
                {
                    HostTensor split_tensor;
                    if (resolve_host_tensor(node.inputs[1], split_tensor))
                    {
                        splits = to_i64_vector(split_tensor);
                    }
                }
                if (splits.empty())
                {
                    const int n = static_cast<int>(node.outputs.size());
                    const int64_t each = x->ne[axis_ggml] / n;
                    splits.assign(n, each);
                }

                int64_t offset = 0;
                for (size_t i = 0; i < node.outputs.size(); ++i)
                {
                    std::array<int64_t, 4> ne = {x->ne[0], x->ne[1], x->ne[2], x->ne[3]};
                    ne[axis_ggml] = splits[i];
                    const size_t byte_offset = static_cast<size_t>(offset) * x->nb[axis_ggml];
                    ggml_tensor *view = ggml_view_4d(ctx, x, ne[0], ne[1], ne[2], ne[3], x->nb[1], x->nb[2], x->nb[3], byte_offset);
                    values[node.outputs[i]] = view;
                    offset += splits[i];
                }
                continue;
            }

            if (node.op_type == "Concat")
            {
                int input_rank = 4;
                if (auto it = tensor_rank_by_name_.find(node.inputs[0]); it != tensor_rank_by_name_.end() && it->second >= 1 && it->second <= 4)
                {
                    input_rank = it->second;
                }
                const int axis_ggml = map_onnx_axis_to_ggml(node.axis, input_rank);
                ggml_tensor *y = values.at(node.inputs[0]);
                for (size_t i = 1; i < node.inputs.size(); ++i)
                {
                    ggml_tensor *rhs = values.at(node.inputs[i]);
                    if (rhs->type != y->type)
                    {
                        rhs = ggml_cast(ctx, rhs, y->type);
                    }
                    for (int d = 0; d < 4; ++d)
                    {
                        if (d == axis_ggml)
                        {
                            continue;
                        }
                        if (y->ne[d] != rhs->ne[d])
                        {
                            std::ostringstream oss;
                            oss << "Concat shape mismatch in " << node.name
                                << " at dim " << d
                                << " lhs.ne=" << y->ne[0] << "," << y->ne[1] << "," << y->ne[2] << "," << y->ne[3]
                                << " rhs.ne=" << rhs->ne[0] << "," << rhs->ne[1] << "," << rhs->ne[2] << "," << rhs->ne[3]
                                << " axis_ggml=" << axis_ggml;
                            return fail_with_cleanup(oss.str());
                        }
                    }
                    y = ggml_concat(ctx, y, rhs, axis_ggml);
                }
                values[node.outputs[0]] = y;
                continue;
            }

            if (node.op_type == "Gather")
            {
                // Gather fallback for yolo26n pattern:
                // data [N, C], indices [B, K], axis=0 -> output [B, K, C]
                if (node.inputs.size() < 2 || node.outputs.empty())
                {
                    return fail_with_cleanup("Gather expects 2 inputs and 1 output: " + node.name);
                }
                if (node.axis != 0)
                {
                    return fail_with_cleanup("Gather currently only supports axis=0: " + node.name);
                }

                ggml_tensor *data = values.at(node.inputs[0]);
                ggml_tensor *indices = values.at(node.inputs[1]);
                if (indices->type != GGML_TYPE_I32 && indices->type != GGML_TYPE_I64)
                {
                    return fail_with_cleanup("Gather indices type must be I32/I64: " + node.name);
                }

                int data_rank = 4;
                int idx_rank = 4;
                if (auto it = tensor_rank_by_name_.find(node.inputs[0]); it != tensor_rank_by_name_.end() && it->second >= 1 && it->second <= 4)
                {
                    data_rank = it->second;
                }
                if (auto it = tensor_rank_by_name_.find(node.inputs[1]); it != tensor_rank_by_name_.end() && it->second >= 1 && it->second <= 4)
                {
                    idx_rank = it->second;
                }
                if (data_rank != 2 || idx_rank != 2)
                {
                    return fail_with_cleanup("Gather currently expects rank-2 data and rank-2 indices: " + node.name);
                }

                std::vector<int64_t> data_dims = tensor_to_onnx_dims(data, data_rank); // [N, C]
                std::vector<int64_t> idx_dims = tensor_to_onnx_dims(indices, idx_rank); // [B, K]

                ggml_tensor *data_for_gather = data;
                const ggml_type data_orig_type = data_for_gather->type;
                if (data_for_gather->type != GGML_TYPE_F32 &&
                    data_for_gather->type != GGML_TYPE_F16 &&
                    data_for_gather->type != GGML_TYPE_I32)
                {
                    return fail_with_cleanup("Gather data currently supports F16/F32/I32 only: " + node.name);
                }

                if (!ggml_is_contiguous(data_for_gather))
                {
                    data_for_gather = ggml_cont(ctx, data_for_gather);
                }
                if (data_for_gather->type == GGML_TYPE_I32)
                {
                    // ggml_gather_elements currently supports F16/F32 data only.
                    data_for_gather = ggml_cast(ctx, data_for_gather, GGML_TYPE_F32);
                }

                // Normalize indices to I32 for downstream gather path.
                ggml_tensor *indices_i32 = indices;
                if (indices_i32->type == GGML_TYPE_I64)
                {
                    indices_i32 = ggml_cast(ctx, indices_i32, GGML_TYPE_I32);
                }
                else if (indices_i32->type != GGML_TYPE_I32)
                {
                    return fail_with_cleanup("Gather indices type must be I32/I64: " + node.name);
                }

                if (!ggml_is_contiguous(indices_i32))
                {
                    indices_i32 = ggml_cont(ctx, indices_i32);
                }

                // Use GatherElements on reshaped tensors:
                // data3 [1, N, C], idx3 [B, K, 1], gather axis=1.
                const auto data3_ne = onnx_dims_to_ggml_ne({1, data_dims[0], data_dims[1]});
                ggml_tensor *data3 = ggml_reshape_3d(ctx, data_for_gather, data3_ne[0], data3_ne[1], data3_ne[2]);
                const auto idx3_ne = onnx_dims_to_ggml_ne({idx_dims[0], idx_dims[1], 1});
                ggml_tensor *idx3 = ggml_reshape_3d(ctx, indices_i32, idx3_ne[0], idx3_ne[1], idx3_ne[2]);

                ggml_tensor *y = ggml_gather_elements(ctx, data3, idx3, map_onnx_axis_to_ggml(1, 3));

                // Keep output dtype consistent with original data tensor.
                if (y->type != data_orig_type)
                {
                    y = ggml_cast(ctx, y, data_orig_type);
                }
                values[node.outputs[0]] = y;
                continue;
            }

            if (node.op_type == "GatherElements")
            {
                if (node.inputs.size() < 2 || node.outputs.empty())
                {
                    return fail_with_cleanup("GatherElements expects 2 inputs and 1 output: " + node.name);
                }

                ggml_tensor *data = values.at(node.inputs[0]);
                ggml_tensor *indices = values.at(node.inputs[1]);

                if (data->type != GGML_TYPE_F32 && data->type != GGML_TYPE_F16)
                {
                    return fail_with_cleanup("GatherElements data type must be F16/F32: " + node.name);
                }
                if (indices->type != GGML_TYPE_I32 && indices->type != GGML_TYPE_I64)
                {
                    return fail_with_cleanup("GatherElements indices type must be I32/I64: " + node.name);
                }

                int data_rank = 4;
                int idx_rank = 4;
                if (auto it = tensor_rank_by_name_.find(node.inputs[0]); it != tensor_rank_by_name_.end() && it->second >= 1 && it->second <= 4)
                {
                    data_rank = it->second;
                }
                if (auto it = tensor_rank_by_name_.find(node.inputs[1]); it != tensor_rank_by_name_.end() && it->second >= 1 && it->second <= 4)
                {
                    idx_rank = it->second;
                }
                if (data_rank != idx_rank)
                {
                    return fail_with_cleanup("GatherElements rank mismatch: " + node.name);
                }

                const int axis_ggml = map_onnx_axis_to_ggml(node.axis, data_rank);
                if (axis_ggml < 0 || axis_ggml > 3)
                {
                    return fail_with_cleanup("GatherElements axis out of range: " + node.name);
                }

                for (int d = 0; d < 4; ++d)
                {
                    if (d == axis_ggml)
                    {
                        continue;
                    }
                    if (data->ne[d] != indices->ne[d])
                    {
                        std::ostringstream oss;
                        oss << "GatherElements shape mismatch in " << node.name
                            << " at dim " << d
                            << " data.ne=" << data->ne[0] << "," << data->ne[1] << "," << data->ne[2] << "," << data->ne[3]
                            << " indices.ne=" << indices->ne[0] << "," << indices->ne[1] << "," << indices->ne[2] << "," << indices->ne[3]
                            << " axis_ggml=" << axis_ggml;
                        return fail_with_cleanup(oss.str());
                    }
                }

                ggml_tensor *y = ggml_gather_elements(ctx, data, indices, axis_ggml);
                values[node.outputs[0]] = y;
                continue;
            }

            if (node.op_type == "Resize")
            {
                ggml_tensor *x = values.at(node.inputs[0]);
                std::vector<int64_t> target_sizes;
                std::vector<float> scales;

                if (node.inputs.size() >= 4)
                {
                    HostTensor sizes_tensor;
                    if (resolve_host_tensor(node.inputs[3], sizes_tensor))
                    {
                        target_sizes = to_i64_vector(sizes_tensor);
                    }
                }
                if (target_sizes.empty() && node.inputs.size() >= 3)
                {
                    HostTensor scales_tensor;
                    if (resolve_host_tensor(node.inputs[2], scales_tensor))
                    {
                        scales = to_f32_vector(scales_tensor);
                    }
                }

                if (!target_sizes.empty())
                {
                    const auto ne = onnx_dims_to_ggml_ne(target_sizes);
                    values[node.outputs[0]] = ggml_interpolate(ctx, x, ne[0], ne[1], ne[2], ne[3], GGML_SCALE_MODE_NEAREST);
                }
                else if (!scales.empty() && scales.size() >= 4)
                {
                    const int factor_h = static_cast<int>(std::round(scales[2]));
                    const int factor_w = static_cast<int>(std::round(scales[3]));
                    if (factor_h == factor_w && factor_h > 0)
                    {
                        values[node.outputs[0]] = ggml_upscale(ctx, x, factor_h, GGML_SCALE_MODE_NEAREST);
                    }
                    else
                    {
                        std::vector<int64_t> current_onnx = {x->ne[3], x->ne[2], x->ne[1], x->ne[0]};
                        std::vector<int64_t> resized_onnx = {
                            static_cast<int64_t>(std::llround(current_onnx[0] * scales[0])),
                            static_cast<int64_t>(std::llround(current_onnx[1] * scales[1])),
                            static_cast<int64_t>(std::llround(current_onnx[2] * scales[2])),
                            static_cast<int64_t>(std::llround(current_onnx[3] * scales[3]))};
                        const auto ne = onnx_dims_to_ggml_ne(resized_onnx);
                        values[node.outputs[0]] = ggml_interpolate(ctx, x, ne[0], ne[1], ne[2], ne[3], GGML_SCALE_MODE_NEAREST);
                    }
                }
                else
                {
                    return fail_with_cleanup("unsupported Resize node missing scales/sizes: " + node.name);
                }
                continue;
            }

            return fail_with_cleanup("unsupported op type: " + node.op_type);
        }

        for (const auto &node : nodes_)
        {
            for (const std::string &output_name : node.outputs)
            {
                auto vit = values.find(output_name);
                if (vit == values.end() || vit->second == nullptr)
                {
                    continue;
                }
                ggml_set_name(vit->second, output_name.c_str());
            }
        }

        std::vector<ggml_tensor *> outputs;
        outputs.reserve(graph_outputs_.size());
        for (const auto &output_info : graph_outputs_)
        {
            auto oit = values.find(output_info.name);
            if (oit == values.end())
            {
                return fail_with_cleanup("output tensor not produced: " + output_info.name);
            }
            ggml_tensor *out = oit->second;
            // Some graph outputs are views (non-contiguous). Materialize here so
            // host-side output reads always observe logical tensor order.
            if (!ggml_is_contiguous(out))
            {
                out = ggml_cont(ctx, out);
            }
            outputs.push_back(out);
        }

        ggml_cgraph *gf = ggml_new_graph(ctx);
        for (ggml_tensor *out : outputs)
        {
            ggml_build_forward_expand(gf, out);
        }

        ggml_gallocr_t gallocr = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(reinterpret_cast<ggml_backend_t>(backend_)));
        if (gallocr == nullptr)
        {
            return fail_with_cleanup("failed to create gallocr");
        }

        if (!ggml_gallocr_alloc_graph(gallocr, gf))
        {
            ggml_gallocr_free(gallocr);
            return fail_with_cleanup("failed to allocate graph with gallocr");
        }

        last_compute_peak_bytes_ = ggml_gallocr_get_buffer_size(gallocr, 0);
        last_compute_buffer_bytes_ = last_compute_peak_bytes_;

        compute_ctx_ = ctx;
        compute_graph_ = gf;
        compute_allocr_ = reinterpret_cast<void *>(gallocr);
        input_tensor_ = input_tensor;
        output_tensors_ = std::move(outputs);

        return true;
    }

    bool DynamicModel::infer(const std::vector<float> &input, std::vector<float> &output)
    {
        std::vector<std::vector<float>> outputs;
        if (!infer_all(input, outputs))
        {
            return false;
        }
        if (outputs.empty())
        {
            return set_error("no output tensors available");
        }
        output = std::move(outputs.front());
        return true;
    }

    bool DynamicModel::infer_all(const std::vector<float> &input, std::vector<std::vector<float>> &outputs)
    {
        if (!loaded_)
        {
            return set_error("model not loaded");
        }
        try
        {
            return build_and_run_graph(input, outputs);
        }
        catch (const std::exception &e)
        {
            return set_error(std::string("inference failed: ") + e.what());
        }
    }

    bool DynamicModel::benchmark_compute(const std::vector<float> &input, int repeats, double &avg_ms)
    {
        if (!loaded_)
        {
            return set_error("model not loaded");
        }
        if (repeats <= 0)
        {
            return set_error("benchmark repeats must be > 0");
        }
        if (graph_inputs_.empty() || compute_graph_ == nullptr || input_tensor_ == nullptr)
        {
            return set_error("compute graph is not initialized");
        }

        const TensorInfo &input_info = graph_inputs_.front();
        const int64_t expected_input_elements = numel_from_dims(input_info.dims);
        if (expected_input_elements != static_cast<int64_t>(input.size()))
        {
            std::ostringstream oss;
            oss << "input element count mismatch, expect " << expected_input_elements << " got " << input.size();
            return set_error(oss.str());
        }

        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i)
        {
            ggml_backend_tensor_set(input_tensor_, input.data(), 0, input.size() * sizeof(float));

            const ggml_status status = ggml_backend_graph_compute(
                reinterpret_cast<ggml_backend_t>(backend_),
                compute_graph_);
            if (status != GGML_STATUS_SUCCESS)
            {
                return set_error("ggml backend graph compute failed");
            }
        }
        const auto end = std::chrono::steady_clock::now();
        const double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        avg_ms = total_ms / static_cast<double>(repeats);

        const bool enable_op_profile = !env_flag_enabled("MINI2GGUF_DISABLE_PROFILE_OP");
        if (enable_op_profile)
        {
            ggml_backend_t backend = reinterpret_cast<ggml_backend_t>(backend_);
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            if (dev != nullptr && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU)
            {
                SchedulerNodeProfileState profile_state;
                const int profile_n_nodes = ggml_graph_n_nodes(compute_graph_);
                profile_state.timings.reserve(static_cast<size_t>(profile_n_nodes));
                for (int i = 0; i < profile_n_nodes; ++i)
                {
                    profile_state.node_index_by_ptr[ggml_graph_node(compute_graph_, i)] = i;
                }

                ggml_backend_t backends[] = {backend};
                const size_t graph_size = std::max<size_t>(
                    GGML_DEFAULT_GRAPH_SIZE,
                    static_cast<size_t>(ggml_graph_size(compute_graph_)));
                ggml_backend_sched_t sched = ggml_backend_sched_new(
                    backends,
                    nullptr,
                    1,
                    graph_size,
                    false,
                    true);

                if (sched != nullptr)
                {
                    ggml_backend_sched_set_eval_callback(
                        sched,
                        scheduler_profile_callback,
                        &profile_state);

                    ggml_backend_tensor_set(input_tensor_, input.data(), 0, input.size() * sizeof(float));

                    const ggml_status profile_status = ggml_backend_sched_graph_compute(
                        sched,
                        compute_graph_);
                    ggml_backend_sched_free(sched);

                    if (profile_status == GGML_STATUS_SUCCESS && !profile_state.timings.empty())
                    {
                        struct OpBucket
                        {
                            double total_ms = 0.0;
                            int count = 0;
                        };

                        std::unordered_map<std::string, OpBucket> by_op;
                        by_op.reserve(profile_state.timings.size());
                        double total_profile_ms = 0.0;
                        for (const NodeProfileTiming &timing : profile_state.timings)
                        {
                            OpBucket &bucket = by_op[timing.ggml_op];
                            bucket.total_ms += timing.elapsed_ms;
                            bucket.count += 1;
                            total_profile_ms += timing.elapsed_ms;
                        }

                        std::vector<std::pair<std::string, OpBucket>> rows;
                        rows.reserve(by_op.size());
                        for (const auto &kv : by_op)
                        {
                            rows.push_back(kv);
                        }
                        std::sort(
                            rows.begin(),
                            rows.end(),
                            [](const std::pair<std::string, OpBucket> &a, const std::pair<std::string, OpBucket> &b)
                            {
                                return a.second.total_ms > b.second.total_ms;
                            });

                        const int top_k = env_int_or_default(
                            "MINI2GGUF_PROFILE_OP_TOPK",
                            static_cast<int>(rows.size()),
                            0);
                        std::printf("mini2gguf profile op-category breakdown (single-run):\n");
                        int printed = 0;
                        for (const auto &row : rows)
                        {
                            if (top_k > 0 && printed >= top_k)
                            {
                                break;
                            }
                            const double pct = total_profile_ms > 0.0
                                                   ? (row.second.total_ms * 100.0 / total_profile_ms)
                                                   : 0.0;
                            std::printf(
                                "  op=%s total=%.6f ms count=%d (%.2f%%)\n",
                                row.first.c_str(),
                                row.second.total_ms,
                                row.second.count,
                                pct);
                            ++printed;
                        }
                    }
                }
            }
        }

        return true;
    }

    bool DynamicModel::build_and_run_graph(const std::vector<float> &input, std::vector<std::vector<float>> &outputs)
    {
        if (graph_inputs_.empty() || graph_outputs_.empty() || compute_graph_ == nullptr || input_tensor_ == nullptr || output_tensors_.empty())
        {
            return set_error("compute graph is not initialized");
        }

        const TensorInfo &input_info = graph_inputs_.front();
        const int64_t expected_input_elements = numel_from_dims(input_info.dims);
        if (expected_input_elements != static_cast<int64_t>(input.size()))
        {
            std::ostringstream oss;
            oss << "input element count mismatch, expect " << expected_input_elements << " got " << input.size();
            return set_error(oss.str());
        }

        ggml_backend_tensor_set(input_tensor_, input.data(), 0, input.size() * sizeof(float));

        const ggml_status status = ggml_backend_graph_compute(
            reinterpret_cast<ggml_backend_t>(backend_),
            compute_graph_);
        if (status != GGML_STATUS_SUCCESS)
        {
            return set_error("ggml backend graph compute failed");
        }

        outputs.clear();
        outputs.resize(output_tensors_.size());

        for (size_t output_index = 0; output_index < output_tensors_.size(); ++output_index)
        {
            ggml_tensor *output_tensor = output_tensors_[output_index];
            const size_t out_elements = static_cast<size_t>(ggml_nelements(output_tensor));
            std::vector<float> &output = outputs[output_index];
            output.resize(out_elements);

            if (output_tensor->type == GGML_TYPE_F32)
            {
                if (output_tensor->buffer != nullptr)
                {
                    ggml_backend_tensor_get(output_tensor, output.data(), 0, out_elements * sizeof(float));
                }
                else
                {
                    std::memcpy(output.data(), output_tensor->data, out_elements * sizeof(float));
                }
            }
            else if (output_tensor->type == GGML_TYPE_F16)
            {
                std::vector<ggml_fp16_t> temp(out_elements);
                if (output_tensor->buffer != nullptr)
                {
                    ggml_backend_tensor_get(output_tensor, temp.data(), 0, temp.size() * sizeof(ggml_fp16_t));
                }
                else
                {
                    std::memcpy(temp.data(), output_tensor->data, temp.size() * sizeof(ggml_fp16_t));
                }
                for (size_t i = 0; i < out_elements; ++i)
                {
                    output[i] = ggml_fp16_to_fp32(temp[i]);
                }
            }
            else
            {
                std::ostringstream oss;
                oss << "unsupported output tensor type at index " << output_index;
                return set_error(oss.str());
            }
        }

        return true;
    }

} // namespace mini2gguf
