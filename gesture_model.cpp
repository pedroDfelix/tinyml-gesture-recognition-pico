#include "gesture_model.h"

#include <cmath>

#include "weights.h"

namespace gesture {

static_assert(N_INPUT == WINDOW * AXES, "weights do not match the window size");
static_assert(N_OUTPUT == 4, "expected 4 classes");

const char* class_name(int c) {
    switch (c) {
        case CIRCLE: return "Circle";
        case SQUARE: return "Square";
        case REST: return "Rest";
        case TRIANGLE: return "Triangle";
        default: return "None";
    }
}

void preprocess(const Sample window[WINDOW], float out[WINDOW * AXES]) {
    const Sample& ref = window[0];
    for (int i = 0; i < WINDOW; ++i) {
        out[i * AXES + 0] = (window[i].x - ref.x) / 500.0f;
        out[i * AXES + 1] = (window[i].y - ref.y) / 500.0f;
        out[i * AXES + 2] = (window[i].z - ref.z) / 500.0f;
    }
}

// out[j] = activation(sum_i in[i] * W[i][j] + b[j]); W is stored [inputs][outputs] as in Keras.
template <int IN, int OUT>
static void dense(const float in[IN], const float (&W)[IN][OUT], const float (&b)[OUT],
                  float out[OUT], bool relu) {
    for (int j = 0; j < OUT; ++j) {
        float acc = b[j];
        for (int i = 0; i < IN; ++i) acc += in[i] * W[i][j];
        out[j] = (relu && acc < 0.0f) ? 0.0f : acc;
    }
}

int predict(const float input[N_INPUT], float probs[N_OUTPUT]) {
    float h1[N_HIDDEN1];
    float h2[N_HIDDEN2];
    float logits[N_OUTPUT];
    dense<N_INPUT, N_HIDDEN1>(input, W1, B1, h1, true);
    dense<N_HIDDEN1, N_HIDDEN2>(h1, W2, B2, h2, true);
    dense<N_HIDDEN2, N_OUTPUT>(h2, W3, B3, logits, false);

    // Softmax, shifted by the maximum for numerical stability (as in ai_puro.py).
    int best = 0;
    for (int k = 1; k < N_OUTPUT; ++k)
        if (logits[k] > logits[best]) best = k;
    float sum = 0.0f;
    for (int k = 0; k < N_OUTPUT; ++k) {
        probs[k] = std::exp(logits[k] - logits[best]);
        sum += probs[k];
    }
    for (int k = 0; k < N_OUTPUT; ++k) probs[k] /= sum;
    return best;
}

void Classifier::drop_oldest(int n) {
    for (int i = n; i < count_; ++i) buffer_[i - n] = buffer_[i];
    count_ -= n;
}

int Classifier::push(Sample s, float* confidence) {
    buffer_[count_++] = s;
    if (count_ < WINDOW) return NONE;

    // Rest filter from main.py: if x and y barely move, skip the window.
    int16_t min_x = buffer_[0].x, max_x = min_x, min_y = buffer_[0].y, max_y = min_y;
    for (int i = 1; i < WINDOW; ++i) {
        if (buffer_[i].x < min_x) min_x = buffer_[i].x;
        if (buffer_[i].x > max_x) max_x = buffer_[i].x;
        if (buffer_[i].y < min_y) min_y = buffer_[i].y;
        if (buffer_[i].y > max_y) max_y = buffer_[i].y;
    }
    if (max_x - min_x < REST_RANGE && max_y - min_y < REST_RANGE) {
        drop_oldest(STEP);
        return NONE;
    }

    float input[N_INPUT];
    float probs[N_OUTPUT];
    preprocess(buffer_, input);
    int c = predict(input, probs);
    drop_oldest(STEP);

    if (probs[c] > MIN_CONFIDENCE && c != REST) {
        if (confidence) *confidence = probs[c];
        return c;
    }
    return NONE;
}

}  // namespace gesture
