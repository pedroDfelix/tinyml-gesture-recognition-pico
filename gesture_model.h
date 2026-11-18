// C++ port of the gesture classifier that runs on the Pico in main.py / ai_puro.py.
// No dynamic memory and no dependencies beyond <cmath>, so the same code can be
// built with the Pico SDK.
#pragma once

#include <cstdint>

namespace gesture {

constexpr int WINDOW = 20;        // readings per window (PONTOS_POR_JANELA)
constexpr int AXES = 3;           // x, y, z
constexpr int STEP = 10;          // main.py keeps the last 10 readings after each window
constexpr int REST_RANGE = 40;    // rest filter: x and y range below this -> skip
constexpr float MIN_CONFIDENCE = 0.75f;

enum Class : int { CIRCLE = 0, SQUARE = 1, REST = 2, TRIANGLE = 3, NONE = -1 };
const char* class_name(int c);

struct Sample { int16_t x, y, z; };

// Same preprocessing as main.py: each reading relative to the first one, divided by 500.
void preprocess(const Sample window[WINDOW], float out[WINDOW * AXES]);

// Forward pass: dense(24, ReLU) -> dense(12, ReLU) -> dense(4, softmax).
// Returns the predicted class; probabilities are written to probs.
int predict(const float input[WINDOW * AXES], float probs[4]);

// Streaming classifier that reproduces the loop in main.py: buffer readings,
// skip windows where the sensor is not moving, report a gesture only above 75%.
class Classifier {
public:
    // Feed one accelerometer reading. Returns a class when a gesture is
    // recognised, NONE otherwise. confidence is set when a class is returned.
    int push(Sample s, float* confidence = nullptr);
    void reset() { count_ = 0; }

private:
    void drop_oldest(int n);
    Sample buffer_[WINDOW];
    int count_ = 0;
};

}  // namespace gesture
