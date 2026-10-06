// facecli.cpp — CLI: 人脸检测 / 对齐裁剪 / 人脸解析（face-parsing）
#include "face.hpp"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage:\n"
                     "  facecli detect <img>\n"
                     "  facecli crop   <img> <out.png> [size=512]\n"
                     "  facecli parse  <img> <out.png> [color|face|skin|hair]\n";
        return 2;
    }
    std::string cmd = argv[1];
    try {
        face::FaceEngine eng;
        if (cmd == "detect") {
            if (argc < 3) return 2;
            auto fs = eng.Detect(argv[2]);
            std::ostringstream os;
            os << "[";
            for (size_t i = 0; i < fs.size(); ++i) {
                if (i) os << ",";
                os << "{\"score\":" << fs[i].score << ",\"box\":[" << fs[i].box[0] << "," << fs[i].box[1] << ","
                   << fs[i].box[2] << "," << fs[i].box[3] << "]}";
            }
            os << "]";
            std::cout << os.str() << std::endl;
            return 0;
        } else if (cmd == "crop") {
            if (argc < 4) return 2;
            int size = argc > 4 ? std::atoi(argv[4]) : 512;
            bool ok = eng.CropLargest(argv[2], argv[3], size);
            if (!ok) { std::cerr << "no face detected\n"; return 1; }
            std::cout << argv[3] << std::endl;
            return 0;
        } else if (cmd == "parse") {
            if (argc < 4) return 2;
            std::string mode = argc > 4 ? argv[4] : "face";
            eng.Parse(argv[2], argv[3], mode);
            std::cout << argv[3] << std::endl;
            return 0;
        }
        std::cerr << "unknown command: " << cmd << "\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "[facecli] error: " << e.what() << "\n";
        return 1;
    }
}
