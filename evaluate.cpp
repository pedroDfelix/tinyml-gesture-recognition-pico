// Runs the C++ model on the recorded CSV data (windows of 20 readings, step 5),
// prints a confusion matrix and writes every prediction to cpp_predictions.csv,
// so it can be compared with the Python model (compare_with_python.py).
//
// Usage: ./evaluate <folder with dados_*.csv> [output.csv]
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "gesture_model.h"

using gesture::Sample;

static constexpr int EVAL_STEP = 5;

static std::vector<Sample> read_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "cannot open " << path << "\n";
        std::exit(1);
    }
    std::vector<Sample> data;
    std::string line;
    std::getline(f, line);  // header: x,y,z,label
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::string x, y, z;
        if (!std::getline(ss, x, ',') || !std::getline(ss, y, ',') || !std::getline(ss, z, ','))
            continue;
        data.push_back({static_cast<int16_t>(std::stoi(x)), static_cast<int16_t>(std::stoi(y)),
                        static_cast<int16_t>(std::stoi(z))});
    }
    return data;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <data folder> [output.csv]\n";
        return 1;
    }
    const std::string dir = argv[1];
    const std::string out_path = argc > 2 ? argv[2] : "cpp_predictions.csv";

    // Same class order as LABELS in main.py.
    const char* files[4] = {"dados_circulo.csv", "dados_quadrado.csv", "dados_repouso.csv",
                            "dados_triangulo.csv"};

    std::ofstream out(out_path);
    out << "class,start,pred_cpp,p0,p1,p2,p3\n";

    int confusion[4][4] = {};
    long total = 0, correct = 0;
    double seconds = 0.0;

    for (int c = 0; c < 4; ++c) {
        std::vector<Sample> data = read_csv(dir + "/" + files[c]);
        for (size_t s = 0; s + gesture::WINDOW <= data.size(); s += EVAL_STEP) {
            float input[gesture::WINDOW * gesture::AXES];
            float probs[4];
            auto t0 = std::chrono::steady_clock::now();
            gesture::preprocess(&data[s], input);
            int p = gesture::predict(input, probs);
            seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

            confusion[c][p]++;
            total++;
            correct += (p == c);
            char row[160];
            std::snprintf(row, sizeof row, "%d,%zu,%d,%.6f,%.6f,%.6f,%.6f\n", c, s, p, probs[0],
                          probs[1], probs[2], probs[3]);
            out << row;
        }
    }

    std::printf("Windows: %ld, accuracy on all recorded windows: %.2f%%\n", total,
                100.0 * correct / total);
    std::printf("Rows = real, columns = predicted (Circle, Square, Rest, Triangle)\n");
    for (int r = 0; r < 4; ++r) {
        std::printf("%-9s", gesture::class_name(r));
        for (int c = 0; c < 4; ++c) std::printf("%6d", confusion[r][c]);
        std::printf("\n");
    }
    std::printf("Average inference time on this PC: %.2f us per window\n",
                1e6 * seconds / total);
    std::printf("Predictions written to %s\n", out_path.c_str());

    // Streaming test: feed each recording reading by reading through the same
    // logic as main.py (rest filter, step of 10, 75% confidence) and count what
    // the Pico would have shown on the LCD.
    std::printf("\nStreaming, as on the Pico (gestures reported per recording):\n");
    for (int c = 0; c < 4; ++c) {
        std::vector<Sample> data = read_csv(dir + "/" + files[c]);
        gesture::Classifier clf;
        int reported[4] = {};
        for (const Sample& s : data) {
            int g = clf.push(s);
            if (g != gesture::NONE) reported[g]++;
        }
        std::printf("%-20s -> Circle %4d  Square %4d  Triangle %4d\n", files[c], reported[0],
                    reported[1], reported[3]);
    }
    return 0;
}
