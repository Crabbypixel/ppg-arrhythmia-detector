# PPG Arrhythmia Detector (AF Detection)

A real-time arrhythmia detection system using PPG signals and a lightweight 1D CNN model, designed for edge deployment on Raspberry Pi.

---

## Overview

This project detects **Atrial Fibrillation (AF)** from PPG signals using:

- 1D Convolutional Neural Network
- UMass Simband Dataset
- Real-time inference on Raspberry Pi
- MAX30102 sensor

---

## Key Features

- Real-time AF detection
- Lightweight model (~3 MB)
- Low latency (~5–10 ms on Pi)
- High accuracy (~98%)
- Edge deployment ready

---

## Model Performance

| Metric | Value |
|------|------|
| Accuracy | 98% |
| Precision (AF) | 95% |
| Recall (AF) | 83% |
| F1-score | 0.89 |
| ROC-AUC | 0.993 |

---

## Confusion Matrix
[[4485 25]
[ 72 371]]


---

## Dataset

- Dataset: UMass Simband Dataset
- Signal type: PPG
- Window size: 30 seconds (~1500 samples)
- Subjects: ~57 individuals

---

## Pipeline

PPG Signal → Filtering → Normalization → CNN → AF Prediction

---

## Model Architecture

- Conv1D layers for temporal feature extraction
- Batch Normalization + ReLU
- MaxPooling
- Dense layers for classification

Total parameters: ~288K

---

## Running the Project

### 1. Preprocessing
python scripts/step1_preprocess.py

### 2. Train/Test Split
python scripts/step6_split_data.py

### 3. Evaluate Model
python scripts/step7_evaluate_test.py

### 4. Simulate Real-Time
python scripts/step8_realtime_simulation.py

### 5. Raspberry Pi Inference
python pi/step5_pi_inference.py

---

## Why 1D CNN?

- Learns temporal patterns directly from raw PPG
- No manual feature extraction required
- Captures RR interval irregularities

---

## Limitations

- Lower recall (~83%) → some AF cases missed
- Dataset imbalance (Normal >> AF)
- PPG is more noise-prone than ECG

---

## Future Work

- Improve recall using class balancing
- Add multi-class arrhythmia detection
- Deploy on wearable devices
- Optimize with quantization

---

## Conclusion

A lightweight and efficient system for real-time AF detection using PPG signals, suitable for edge deployment.

---

## Author
Jaisurya
