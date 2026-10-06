#include "preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace preprocess {

    // Minimal single-input/single-output ONNX Runtime wrapper (CUDA provider optional).
    struct OnnxEngine {
        Ort::Env env;
        Ort::SessionOptions so;
        std::unique_ptr<Ort::Session> session;
        Ort::MemoryInfo mi;
        std::string input_name;
        std::vector<std::string> output_names;

        OnnxEngine(const std::string& path, bool use_cuda)
            : env(ORT_LOGGING_LEVEL_ERROR, "preprocess"),
              mi(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
            if (use_cuda) {
                OrtCUDAProviderOptions cuda{};
                cuda.device_id = 0;
                so.AppendExecutionProvider_CUDA(cuda);
            }
            so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            session = std::make_unique<Ort::Session>(env, path.c_str(), so);
            Ort::AllocatorWithDefaultOptions alloc;
            input_name = session->GetInputNameAllocated(0, alloc).get();
            for (size_t i = 0; i < session->GetOutputCount(); ++i) {
                output_names.push_back(session->GetOutputNameAllocated(i, alloc).get());
            }
        }

        std::vector<float> Run(const std::vector<float>& input,
                               const std::vector<int64_t>& shape,
                               std::vector<int64_t>* out_shape,
                               size_t out_index = 0) {
            Ort::Value t = Ort::Value::CreateTensor<float>(mi, const_cast<float*>(input.data()),
                                                           input.size(), shape.data(), shape.size());
            const char* in_names[]  = {input_name.c_str()};
            const char* out_names[] = {output_names[out_index].c_str()};
            auto outputs            = session->Run(Ort::RunOptions{nullptr}, in_names, &t, 1, out_names, 1);
            auto info               = outputs[0].GetTensorTypeAndShapeInfo();
            if (out_shape != nullptr) {
                *out_shape = info.GetShape();
            }
            const float* data = outputs[0].GetTensorData<float>();
            return std::vector<float>(data, data + info.GetElementCount());
        }
    };

    static cv::Mat ToGray3(const cv::Mat& bgr) {
        cv::Mat gray, gray3;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(gray, gray3, cv::COLOR_GRAY2BGR);
        return gray3;
    }

    static cv::Mat ResizeLongest(const cv::Mat& img, int target) {
        if (target <= 0) {
            return img;
        }
        int w = img.cols, h = img.rows;
        int longest = std::max(w, h);
        if (longest == target) {
            return img;
        }
        double scale = static_cast<double>(target) / longest;
        cv::Mat out;
        cv::resize(img, out, cv::Size(std::max(1, (int)std::lround(w * scale)),
                                      std::max(1, (int)std::lround(h * scale))),
                   0, 0, cv::INTER_AREA);
        return out;
    }

    static cv::Mat CannyMap(const cv::Mat& bgr, const Options& opt) {
        cv::Mat gray, blur, edges, out;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);
        cv::Canny(blur, edges, opt.canny_low, opt.canny_high);
        cv::cvtColor(edges, out, cv::COLOR_GRAY2BGR);
        return out;
    }

    // Approximate the ControlNet "lineart" annotator: thin dark lines on white.
    static cv::Mat LineartMap(const cv::Mat& bgr, const Options& opt) {
        cv::Mat gray, blur, edge, out;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blur, cv::Size(3, 3), 0);
        cv::adaptiveThreshold(255 - blur, edge, 255,
                              cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY,
                              9, -opt.lineart_thresh);
        cv::bitwise_not(edge, out);  // white lines on black (matches canny/hed style)
        cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);
        return out;
    }

    // Approximate the ControlNet "HED" soft-edge annotator (no HED model required).
    static cv::Mat HedApproxMap(const cv::Mat& bgr, const Options& opt) {
        cv::Mat gray, blur, gx, gy, mag, norm32, out;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blur, cv::Size(3, 3), 0);
        cv::Sobel(blur, gx, CV_32F, 1, 0, 3);
        cv::Sobel(blur, gy, CV_32F, 0, 1, 3);
        cv::magnitude(gx, gy, mag);
        cv::GaussianBlur(mag, mag, cv::Size(9, 9), 0);
        cv::normalize(mag, norm32, 0, 255, cv::NORM_MINMAX);
        norm32.convertTo(out, CV_8U);
        cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);
        return out;
    }

    static std::string SaveMap(const cv::Mat& bgr, const std::string& output_path, const Options& opt) {
        cv::Mat out = bgr;
        if (opt.invert) {
            cv::bitwise_not(out, out);
        }
        if (!cv::imwrite(output_path, out)) {
            return "failed to write output: " + output_path;
        }
        return "";
    }

    // Approximate the ControlNet "MLSD" line-segment annotator with OpenCV LSD (no model).
    static cv::Mat MlsdMap(const cv::Mat& bgr, const Options& opt) {
        (void)opt;
        cv::Mat gray, out = cv::Mat::zeros(bgr.size(), CV_8UC3);
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
        std::vector<cv::Vec4f> lines;
        lsd->detect(gray, lines);
        double min_len = 0.02 * std::max(bgr.cols, bgr.rows);
        for (const auto& l : lines) {
            cv::Point p1((int)l[0], (int)l[1]), p2((int)l[2], (int)l[3]);
            if (cv::norm(p2 - p1) < min_len) continue;
            cv::line(out, p1, p2, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        }
        return out;
    }

    // Depth-Anything-V2-Small: ImageNet-normalized NCHW RGB -> min/max-normalized grayscale
    // (brighter = closer), which is the ControlNet depth convention.
    static std::string DepthMap(const cv::Mat& bgr, const Options& opt, cv::Mat& out_map) {
        OnnxEngine engine(opt.depth_model, opt.use_cuda);
        int H = opt.depth_size, W = opt.depth_size;
        if (H % 14 != 0 || W % 14 != 0) {
            return "depth-size must be a multiple of 14";
        }
        cv::Mat rgb, resized;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        cv::resize(rgb, resized, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
        resized.convertTo(resized, CV_32F, 1.0 / 255.0);
        const float mean[3] = {0.485f, 0.456f, 0.406f};
        const float stdv[3] = {0.229f, 0.224f, 0.225f};
        std::vector<float> input(3 * H * W);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const cv::Vec3f p = resized.at<cv::Vec3f>(y, x);
                for (int c = 0; c < 3; ++c) {
                    input[c * H * W + y * W + x] = (p[c] - mean[c]) / stdv[c];
                }
            }
        }
        std::vector<int64_t> oshape;
        std::vector<float> odata = engine.Run(input, {1, 3, H, W}, &oshape);
        if (oshape.size() < 2) {
            return "unexpected depth output rank";
        }
        int64_t oh = oshape[oshape.size() - 2];
        int64_t ow = oshape[oshape.size() - 1];
        cv::Mat depth(oh, ow, CV_32F, odata.data());
        cv::Mat d8, dres;
        cv::normalize(depth, d8, 0, 255, cv::NORM_MINMAX);
        d8.convertTo(d8, CV_8U);
        cv::resize(d8, dres, bgr.size(), 0, 0, cv::INTER_CUBIC);
        cv::cvtColor(dres, out_map, cv::COLOR_GRAY2BGR);
        return "";
    }

    // DWPose (YOLOX person detector + wholebody pose estimator) -> OpenPose 18-keypoint
    // skeleton image, the ControlNet pose convention.
    static std::string PoseMap(const cv::Mat& bgr, const Options& opt, cv::Mat& out_map) {
        OnnxEngine det(opt.pose_det_model, opt.use_cuda);
        OnnxEngine pose(opt.pose_est_model, opt.use_cuda);
        const int W = bgr.cols, H = bgr.rows;

        // --- person detection (YOLOX, 640 letterbox, 0-255 RGB) ---
        float r       = std::min(640.f / W, 640.f / H);
        int nw        = std::max(1, (int)std::lround(W * r));
        int nh        = std::max(1, (int)std::lround(H * r));
        cv::Mat lbox(640, 640, CV_8UC3, cv::Scalar(114, 114, 114));
        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
        resized.copyTo(lbox(cv::Rect(0, 0, nw, nh)));

        std::vector<float> din(3 * 640 * 640);
        for (int y = 0; y < 640; ++y) {
            for (int x = 0; x < 640; ++x) {
                const cv::Vec3b p = lbox.at<cv::Vec3b>(y, x);
                din[0 * 640 * 640 + y * 640 + x] = p[2];  // R
                din[1 * 640 * 640 + y * 640 + x] = p[1];  // G
                din[2 * 640 * 640 + y * 640 + x] = p[0];  // B
            }
        }
        std::vector<int64_t> dshape;
        std::vector<float> raw = det.Run(din, {1, 3, 640, 640}, &dshape);

        struct Box { float x1, y1, x2, y2, score; };
        std::vector<Box> boxes;
        const int strides[3] = {8, 16, 32};
        int idx              = 0;
        for (int s = 0; s < 3; ++s) {
            const int st = strides[s];
            const int g  = 640 / st;
            for (int gy = 0; gy < g; ++gy) {
                for (int gx = 0; gx < g; ++gx) {
                    const float* p = &raw[(size_t)idx * 85];
                    ++idx;
                    float score = p[4] * p[5];  // obj * person class
                    if (score < opt.pose_det_thresh) continue;
                    float cx = (p[0] + gx) * st;
                    float cy = (p[1] + gy) * st;
                    float w  = std::exp(p[2]) * st;
                    float h  = std::exp(p[3]) * st;
                    boxes.push_back({cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score});
                }
            }
        }
        std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) { return a.score > b.score; });
        std::vector<Box> keep;
        for (const auto& b : boxes) {
            bool ok = true;
            for (const auto& k : keep) {
                float xx1 = std::max(b.x1, k.x1), yy1 = std::max(b.y1, k.y1);
                float xx2 = std::min(b.x2, k.x2), yy2 = std::min(b.y2, k.y2);
                float iw = std::max(0.f, xx2 - xx1), ih = std::max(0.f, yy2 - yy1);
                float inter = iw * ih;
                float uni   = (b.x2 - b.x1) * (b.y2 - b.y1) + (k.x2 - k.x1) * (k.y2 - k.y1) - inter;
                if (inter / (uni + 1e-6f) > 0.45f) {
                    ok = false;
                    break;
                }
            }
            if (ok) keep.push_back(b);
        }

        // --- pose per person -> OpenPose 18 skeleton ---
        // OpenPose limb order (0-indexed) and colors are the standard ControlNet annotator.
        static const int limb[17][2] = {{1, 2}, {1, 5}, {2, 3}, {3, 4}, {5, 6}, {6, 7}, {1, 8}, {8, 9}, {9, 10}, {1, 11}, {11, 12}, {12, 13}, {1, 0}, {0, 14}, {14, 16}, {0, 15}, {15, 17}};
        static const int col[18][3]  = {{255, 0, 0}, {255, 85, 0}, {255, 170, 0}, {255, 255, 0}, {170, 255, 0}, {85, 255, 0}, {0, 255, 0}, {0, 255, 85}, {0, 255, 170}, {0, 255, 255}, {0, 170, 255}, {0, 85, 255}, {0, 0, 255}, {85, 0, 255}, {170, 0, 255}, {255, 0, 255}, {255, 0, 170}, {255, 0, 85}};
        // COCO-17 (DWPose body) -> OpenPose-18 index mapping (OP1 neck is interpolated).
        static const int coco_to_op[17] = {0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10};

        cv::Mat out = cv::Mat::zeros(H, W, CV_8UC3);
        for (const auto& b : keep) {
            float x1 = b.x1 / r, y1 = b.y1 / r, x2 = b.x2 / r, y2 = b.y2 / r;
            float cx = (x1 + x2) / 2, cy = (y1 + y2) / 2, bw = x2 - x1, bh = y2 - y1;
            const float aspect = 288.f / 384.f;
            if (bw > bh * aspect) bh = bw / aspect;
            else bw = bh * aspect;
            float ox = cx - bw / 2, oy = cy - bh / 2;
            float sx = 288.f / bw, sy = 384.f / bh;
            cv::Mat M = (cv::Mat_<float>(2, 3) << sx, 0, -ox * sx, 0, sy, -oy * sy);
            cv::Mat warped;
            cv::warpAffine(bgr, warped, M, cv::Size(288, 384), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
            warped.convertTo(warped, CV_32F);
            const float mean[3] = {123.675f, 116.28f, 103.53f};
            const float stdv[3] = {58.395f, 57.12f, 57.375f};
            std::vector<float> pin(3 * 384 * 288);
            for (int y = 0; y < 384; ++y) {
                for (int x = 0; x < 288; ++x) {
                    const cv::Vec3f p = warped.at<cv::Vec3f>(y, x);
                    pin[0 * 384 * 288 + y * 288 + x] = (p[2] - mean[0]) / stdv[0];
                    pin[1 * 384 * 288 + y * 288 + x] = (p[1] - mean[1]) / stdv[1];
                    pin[2 * 384 * 288 + y * 288 + x] = (p[0] - mean[2]) / stdv[2];
                }
            }
            std::vector<int64_t> xshape, yshape;
            std::vector<float> simcc_x = pose.Run(pin, {1, 3, 384, 288}, &xshape, 0);  // [133, 576]
            std::vector<float> simcc_y = pose.Run(pin, {1, 3, 384, 288}, &yshape, 1);  // [133, 768]
            if (xshape.size() < 2 || yshape.size() < 2) {
                return "unexpected pose output rank";
            }
            const int K  = (int)xshape[xshape.size() - 2];
            const int BX = (int)xshape[xshape.size() - 1];
            const int BY = (int)yshape[yshape.size() - 1];

            std::vector<cv::Point> op(18);
            for (int k = 0; k < 17 && k < K; ++k) {
                int bx = 0, by = 0;
                for (int i = 1; i < BX; ++i)
                    if (simcc_x[k * BX + i] > simcc_x[k * BX + bx]) bx = i;
                for (int i = 1; i < BY; ++i)
                    if (simcc_y[k * BY + i] > simcc_y[k * BY + by]) by = i;
                float kx = bx / 2.0f / sx + ox;
                float ky = by / 2.0f / sy + oy;
                op[coco_to_op[k]] = cv::Point((int)kx, (int)ky);
            }
            // neck = midpoint of shoulders
            op[1] = cv::Point((op[5].x + op[2].x) / 2, (op[5].y + op[2].y) / 2);
            for (int i = 0; i < 17; ++i) {
                cv::line(out, op[limb[i][0]], op[limb[i][1]], cv::Scalar(col[i][0], col[i][1], col[i][2]), 4, cv::LINE_AA);
            }
            for (int i = 0; i < 18; ++i) {
                cv::circle(out, op[i], 4, cv::Scalar(col[i % 18][0], col[i % 18][1], col[i % 18][2]), -1, cv::LINE_AA);
            }
        }
        out_map = out;
        return "";
    }

    std::string Process(const std::string& mode,
                        const std::string& input_path,
                        const std::string& output_path,
                        const Options& options) {
        cv::Mat bgr = cv::imread(input_path, cv::IMREAD_COLOR);
        if (bgr.empty()) {
            return "failed to read input image: " + input_path;
        }
        if (options.target_size > 0) {
            bgr = ResizeLongest(bgr, options.target_size);
        }

        if (mode == "canny") {
            return SaveMap(CannyMap(bgr, options), output_path, options);
        }
        if (mode == "lineart") {
            return SaveMap(LineartMap(bgr, options), output_path, options);
        }
        if (mode == "gray") {
            return SaveMap(ToGray3(bgr), output_path, options);
        }
        if (mode == "hed_approx" || mode == "hed") {
            return SaveMap(HedApproxMap(bgr, options), output_path, options);
        }
        if (mode == "mlsd") {
            return SaveMap(MlsdMap(bgr, options), output_path, options);
        }
        if (mode == "depth") {
            cv::Mat map;
            std::string err = DepthMap(bgr, options, map);
            if (!err.empty()) {
                return err;
            }
            return SaveMap(map, output_path, options);
        }
        if (mode == "pose") {
            cv::Mat map;
            std::string err = PoseMap(bgr, options, map);
            if (!err.empty()) {
                return err;
            }
            return SaveMap(map, output_path, options);
        }

        return "unknown or unbuilt mode: " + mode;
    }

}  // namespace preprocess
