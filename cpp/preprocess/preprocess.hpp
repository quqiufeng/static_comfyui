#ifndef __COMFY_PREPROCESS_HPP__
#define __COMFY_PREPROCESS_HPP__

#include <string>

// Condition-map preprocessors for Z-Image Fun-ControlNet (and other ControlNet backends).
//
// Zero-dependency modes (OpenCV only): canny, lineart, gray, hed_approx
// Model-based modes (ONNX Runtime): depth, pose, mlsd
namespace preprocess {

    struct Options {
        // Shared
        int target_size = 0;  // 0 = keep original size; else longest side resized to this
        bool invert = false;  // invert the output (white/black swap)

        // Canny / lineart
        double canny_low  = 100.0;
        double canny_high = 200.0;
        int lineart_thresh = 8;  // adaptive threshold block-size bias (smaller = more lines)

        // Depth (Depth-Anything-V2-Small)
        std::string depth_model = "/data/models/image/depth_anything_v2_small/model.onnx";
        int depth_size          = 518;  // must be a multiple of 14

        // Pose (DWPose: yolox detector + pose estimator)
        std::string pose_det_model = "/data/models/image/dwpose/yolox_l.onnx";
        std::string pose_est_model = "/data/models/image/dwpose/dw-ll_ucoco_384.onnx";
        float pose_det_thresh      = 0.3f;

        // MLSD
        std::string mlsd_model = "/data/models/image/mlsd_large_512_fp32.onnx";
        float mlsd_score_thresh = 0.1f;
        float mlsd_dist_thresh  = 0.1f;

        bool use_cuda = true;
    };

    // mode: canny | lineart | gray | hed_approx | depth | pose | mlsd
    // Returns empty string on success, otherwise an error message.
    std::string Process(const std::string& mode,
                        const std::string& input_path,
                        const std::string& output_path,
                        const Options& options = {});

}  // namespace preprocess

#endif  // __COMFY_PREPROCESS_HPP__
