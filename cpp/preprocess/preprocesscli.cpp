// preprocesscli — condition-map preprocessors for Z-Image Fun-ControlNet.
//
//   preprocess <mode> <input> <output> [options]
//   mode: canny | lineart | gray | hed | depth | pose | mlsd
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "preprocess.hpp"

static void print_usage(const char* argv0) {
    std::fprintf(stderr,
                 "Usage: %s <mode> <input> <output> [options]\n"
                 "  modes: canny | lineart | gray | hed | depth | pose | mlsd\n"
                 "  --target <int>         resize longest side (0 = keep, default 0)\n"
                 "  --invert               invert output\n"
                 "  --canny-low <f>        canny low threshold (default 100)\n"
                 "  --canny-high <f>       canny high threshold (default 200)\n"
                 "  --lineart-thresh <int> lineart threshold bias (default 8)\n"
                 "  --depth-model <path>   Depth-Anything-V2 ONNX\n"
                 "  --depth-size <int>     depth input size, multiple of 14 (default 518)\n"
                 "  --pose-det <path>      DWPose person detector ONNX\n"
                 "  --pose-est <path>      DWPose pose estimator ONNX\n"
                 "  --mlsd-model <path>    M-LSD ONNX\n"
                 "  --cpu                  disable CUDA provider\n",
                 argv0);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }
    std::string mode = argv[1];
    std::string input = argv[2];
    std::string output = argv[3];

    preprocess::Options opt;
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : nullptr; };
        if (a == "--target") {
            if (auto v = next()) opt.target_size = std::atoi(v);
        } else if (a == "--invert") {
            opt.invert = true;
        } else if (a == "--canny-low") {
            if (auto v = next()) opt.canny_low = std::atof(v);
        } else if (a == "--canny-high") {
            if (auto v = next()) opt.canny_high = std::atof(v);
        } else if (a == "--lineart-thresh") {
            if (auto v = next()) opt.lineart_thresh = std::atoi(v);
        } else if (a == "--depth-model") {
            if (auto v = next()) opt.depth_model = v;
        } else if (a == "--depth-size") {
            if (auto v = next()) opt.depth_size = std::atoi(v);
        } else if (a == "--pose-det") {
            if (auto v = next()) opt.pose_det_model = v;
        } else if (a == "--pose-est") {
            if (auto v = next()) opt.pose_est_model = v;
        } else if (a == "--mlsd-model") {
            if (auto v = next()) opt.mlsd_model = v;
        } else if (a == "--cpu") {
            opt.use_cuda = false;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    std::string err = preprocess::Process(mode, input, output, opt);
    if (!err.empty()) {
        std::fprintf(stderr, "preprocess failed: %s\n", err.c_str());
        return 1;
    }
    std::printf("saved %s\n", output.c_str());
    return 0;
}
