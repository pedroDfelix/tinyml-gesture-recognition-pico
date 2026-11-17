# Gesture recognition on a Raspberry Pi Pico (TinyML)

A small neural network that recognises hand-drawn shapes (circle, square, triangle) from accelerometer data, running directly on a Raspberry Pi Pico. No PC or internet connection is needed once the model is on the board: the Pico reads the sensor, runs the network and shows the result on an LCD.

Course project for Advanced Computing Systems (SISTCA), BSc in Electrical and Computer Engineering, ISEP – Polytechnic of Porto, 2026.

https://github.com/user-attachments/assets/2ca5425a-0a4a-4f93-a68b-f0c31b212cd2

## How it works

An ADXL345 accelerometer is attached to a pencil and read by the Pico over I2C. While you draw a shape, the Pico collects windows of 20 readings (X, Y, Z), normalises them and passes each window through the network. The predicted class and its confidence are printed on the serial terminal and shown on the LCD, e.g. `RECONHECIDO: Circulo (100%)`.

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
3. **Train** – upload the CSVs to Google Colab and run the training code from `treino_colab.txt`. Values are min-max normalised to [0, 1] before training.
4. **Export** – run the code in `exportacao_colab.txt`, which prints the weights and biases as Python code. Paste the output into `ai_puro.py`.
5. **Deploy** – copy `main.py`, `ai_puro.py`, `lcd_api.py` and `machine_i2c_lcd.py` to the Pico and run `main.py`. The LCD shows "Pronto!" when it is ready.

Other files: `touchpad_gesture_capture.py` and `challenge.py` are solutions to the exercises in the report; the remaining scripts and `modelo_gestos.tflite` are from earlier iterations of the project.

## Results

Confusion matrix on the recorded dataset, 785 windows (rows: real gesture, columns: prediction):

|               | Circle | Square | Rest | Triangle |
|---------------|-------:|-------:|-----:|---------:|
| **Circle**    | 173    | 2      | 0    | 1        |
| **Square**    | 5      | 143    | 4    | 36       |
| **Rest**      | 1      | 4      | 212  | 16       |
| **Triangle**  | 0      | 20     | 8    | 160      |

Overall accuracy is 87.6% (688/785). Circle is almost always right (98%). The weak spot is square vs. triangle (76% and 85%): both are made of straight strokes with sharp corners, so their acceleration patterns look alike, especially when the drawing speed changes.

## Limitations and next steps

- Square and triangle get mixed up. More data, drawn at different speeds, would probably help, and so would features such as per-axis variance instead of raw values only.
- Pure MicroPython inference uses more memory and time than a compiled runtime. Next step: port the inference to C++ (TensorFlow Lite for Microcontrollers or a hand-written version) and measure inference time and memory on the RP2040.

## Credits

Team: Igor Ávila, Pedro Félix, João Patrão, Rodrigo Teixeira. Supervised by Prof. Rui Nuno Castro.
Original repository: [jonnypatron/SISTCA](https://github.com/jonnypatron/SISTCA).
