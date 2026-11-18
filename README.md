# Gesture recognition on a Raspberry Pi Pico (TinyML)

A small neural network that recognises hand-drawn shapes (circle, square, triangle) from accelerometer data, running directly on a Raspberry Pi Pico. No PC or internet connection is needed once the model is on the board: the Pico reads the sensor, runs the network and shows the result on an LCD.

Course project for Advanced Computing Systems (SISTCA), BSc in Electrical and Computer Engineering, ISEP – Polytechnic of Porto, 2026.

https://github.com/user-attachments/assets/c80502e5-68c4-4328-9ffe-27a85c7bfb63

Full report: [TinyML_Gesture_Recognition_Report.pdf](TinyML_Gesture_Recognition_Report.pdf)

## How it works

An ADXL345 accelerometer is attached to a pencil and read by the Pico over I2C. While you draw a shape, the Pico collects windows of 20 readings (X, Y, Z), moving forward 10 readings each time. If X and Y barely change within a window, the pencil is not moving and the window is skipped. Otherwise each reading is taken relative to the first one in the window and divided by 500, and the 60 values go through the network. A gesture is reported only when the network is more than 75% confident. The result is printed on the serial terminal and shown on the LCD, e.g. `RECONHECIDO: Circulo (100%)`.

The model has four classes: circle, square, triangle and rest (sensor not moving). The rest class lets the system ignore the pauses between gestures.

Training happens off the board, in Google Colab, with TensorFlow/Keras. Instead of using TensorFlow Lite for Microcontrollers, the trained weights and biases are exported as plain Python arrays and the forward pass (dense layers, ReLU, softmax) is implemented by hand in MicroPython (`ai_puro.py`). It is slower than an optimised C/C++ runtime, but every step of the inference is visible and easy to change.

The network is small: 60 inputs (20 readings × 3 axes), two hidden layers of 24 and 12 neurons with ReLU, and a softmax output over the 4 classes. That is about 1,800 weights and biases in total.

## Hardware

- Raspberry Pi Pico (RP2040)
- ADXL345 3-axis accelerometer (I2C)
- LCD display
- Breadboard and jumper wires

No hardware? The I2C part can be tested in the [Wokwi](https://wokwi.com) simulator. Wokwi has no ADXL345, so use an MPU6050 and only read its accelerometer registers.

## Pipeline

1. **Check the sensor** – `monitor_gestos.py` (runs on the PC) reads the serial port and plots the X, Y, Z trajectory live.
2. **Record data** – `gravar_dataset.py` saves the readings to one CSV per class while you repeat the gesture for about two minutes: `dados_circulo.csv`, `dados_quadrado.csv`, `dados_triangulo.csv`, `dados_repouso.csv`.
3. **Train** – upload the CSVs to Google Colab and train the network with TensorFlow/Keras, using the same preprocessing as the Pico (each reading relative to the first one in the window, divided by 500). The weights in `ai_puro.py` come from windows of 20 readings taken every 5 readings, with 20% of them kept aside for testing. `treino_colab.txt` holds a later version of the training script that cuts the recordings into whole gestures instead.
4. **Export** – run the code in `exportacao_colab.txt`, which prints the weights and biases as Python code. Paste the output into `ai_puro.py`.
5. **Deploy** – copy `main.py`, `ai_puro.py`, `lcd_api.py` and `machine_i2c_lcd.py` to the Pico and run `main.py`. The LCD shows "Pronto!" when it is ready.

Other files: `touchpad_gesture_capture.py` and `challenge.py` are solutions to the exercises in the report; the remaining scripts and `modelo_gestos.tflite` are from earlier iterations of the project.

## Results

Confusion matrix on the test set: 785 windows (20% of the recorded data) that were not used for training. Rows are the real gesture, columns the prediction:

|               | Circle | Square | Rest | Triangle |
|---------------|-------:|-------:|-----:|---------:|
| **Circle**    | 173    | 2      | 0    | 1        |
| **Square**    | 5      | 143    | 4    | 36       |
| **Rest**      | 1      | 4      | 212  | 16       |
| **Triangle**  | 0      | 20     | 8    | 160      |

Overall accuracy is 87.6% (688/785). Circle is almost always right (98%). The weak spot is square vs. triangle (76% and 85%): both are made of straight strokes with sharp corners, so their acceleration patterns look alike, especially when the drawing speed changes.

## C++ port

The `cpp/` folder has the same classifier written in C++, as a first step towards running it with the Pico SDK instead of MicroPython:

- `gesture_model.h/.cpp` – preprocessing, the forward pass (dense layers, ReLU, softmax) and a `Classifier` class that reproduces the loop in `main.py` (rest filter, step of 10 readings, 75% threshold). No dynamic memory and no libraries beyond `<cmath>`, so it can be built for a microcontroller.
- `weights.h` – generated from `ai_puro.py` by `gen_weights.py`, so the C++ and MicroPython versions always use the same weights.
- `evaluate.cpp` – runs the model on the recorded CSVs, prints a confusion matrix and writes every prediction to a file.
- `compare_with_python.py` – runs the original Python model on the same windows and checks that both give the same answers.

```
cd cpp
python gen_weights.py ../ai_puro.py weights.h
g++ -std=c++17 -O2 -Wall -o evaluate evaluate.cpp gesture_model.cpp
./evaluate .. cpp_predictions.csv
python compare_with_python.py .. cpp_predictions.csv
```

Result: the C++ version predicts the same class as the Python model in all 3,922 windows, with probabilities that differ by at most 1e-6 (the C++ code uses 32-bit floats). On the test set it gets the same 87.6% (688/785).

Fed reading by reading through the same logic as `main.py`, the circle recording produces 437 circles and nothing else, while the square and triangle recordings show the same mix-ups as the confusion matrix.

This has been tested on a PC only, not yet on the Pico.

## Limitations and next steps

- Square and triangle get mixed up. More data, drawn at different speeds, would probably help, and so would features such as per-axis variance instead of raw values only.
- Pure MicroPython inference uses more memory and time than compiled code. Next step: build the C++ classifier with the Pico SDK, read the ADXL345 over I2C in C++, and measure inference time and memory on the RP2040.

## Credits

Developed by Pedro Félix and João Patrão. Supervised by Prof. Rui Nuno Castro.
Original repository: [jonnypatron/SISTCA](https://github.com/jonnypatron/SISTCA).
