// img2prompt.cpp — CLI: Florence-2 image -> text prompt (pure C++ / ONNX Runtime).
#include "florence2.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    flo::Options opt;
    if (const char* d = std::getenv("FLORENCE2_MODEL_DIR")) opt.model_dir = d;
    std::string image;
    bool cpu = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model-dir" && i + 1 < argc) {
            opt.model_dir = argv[++i];
        } else if (a == "--task" && i + 1 < argc) {
            opt.task = argv[++i];
        } else if (a == "--beams" && i + 1 < argc) {
            opt.num_beams = std::stoi(argv[++i]);
        } else if (a == "--max-new-tokens" && i + 1 < argc) {
            opt.max_new_tokens = std::stoi(argv[++i]);
        } else if (a == "--cpu") {
            cpu = true;
        } else if (a == "-h" || a == "--help") {
            std::cerr << "usage: " << argv[0]
                      << " [--model-dir DIR] [--task TASK] [--beams N] [--max-new-tokens N] [--cpu] IMAGE\n"
                         "tasks: <CAPTION> <DETAILED_CAPTION> <MORE_DETAILED_CAPTION> <OCR>\n";
            return 0;
        } else {
            image = a;
        }
    }
    if (image.empty()) {
        std::cerr << "error: missing IMAGE argument\n";
        return 2;
    }
    opt.use_cuda = !cpu;

    try {
        flo::Florence2 model(opt.model_dir, opt.use_cuda);
        std::cerr << "[img2prompt] execution provider: " << model.provider() << "\n";
        int w = 0, h = 0;
        std::string text = model.Caption(image, opt, &w, &h);
        std::cerr << "[img2prompt] task=" << opt.task << " beams=" << opt.num_beams << " (" << w << "x" << h
                  << ")\n";
        std::cout << text << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[img2prompt] error: " << e.what() << "\n";
        return 1;
    }
}
