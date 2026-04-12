import numpy as np
import time
from collections import deque
from smbus2 import SMBus
from scipy.signal import butter, filtfilt
import tensorflow as tf
import psutil
import os
import csv

# =========================
# CONFIG
# =========================
FS = 50
BUFFER_SIZE = 600
WINDOW_SIZE = 1500

THRESHOLD = float(np.load("best_threshold.npy")[0])

PRINT_INTERVAL = 2.0

# =========================
# LOAD MODEL
# =========================
interpreter = tf.lite.Interpreter(model_path="model_pi.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded")

# =========================
# SYSTEM MONITOR
# =========================
process = psutil.Process(os.getpid())

# =========================
# MAX30102 SETUP
# =========================
I2C_ADDR = 0x57
REG_FIFO_DATA = 0x07
REG_MODE_CONFIG = 0x09
REG_SPO2_CONFIG = 0x0A
REG_LED1_PA = 0x0C
REG_LED2_PA = 0x0D

bus = SMBus(1)

def max30102_init():
    bus.write_byte_data(I2C_ADDR, REG_MODE_CONFIG, 0x03)
    bus.write_byte_data(I2C_ADDR, REG_SPO2_CONFIG, 0x27)
    bus.write_byte_data(I2C_ADDR, REG_LED1_PA, 0x1F)
    bus.write_byte_data(I2C_ADDR, REG_LED2_PA, 0x1F)

def read_ir():
    data = bus.read_i2c_block_data(I2C_ADDR, REG_FIFO_DATA, 6)
    ir = (data[3]<<16 | data[4]<<8 | data[5]) & 0x03FFFF
    return float(ir)

# =========================
# SIGNAL PROCESSING
# =========================
def bandpass(signal, fs):
    low = 0.5 / (fs/2)
    high = 4.0 / (fs/2)
    b, a = butter(2, [low, high], btype='band')
    return filtfilt(b, a, signal)

def detect_peaks(signal, fs):
    derivative = np.diff(signal)

    peaks = []
    last_peak = -1000
    min_gap = int(0.4 * fs)

    for i in range(1, len(derivative)):
        if derivative[i-1] > 0 and derivative[i] <= 0:
            threshold_local = np.mean(signal) + 0.5 * np.std(signal)

            if signal[i] > threshold_local:
                if i - last_peak > min_gap:
                    peaks.append(i)
                    last_peak = i

    return peaks

def get_rr(peaks, fs):
    if len(peaks) < 2:
        return []
    return np.diff(peaks) / fs

# =========================
# LOGGING
# =========================
log_file = open("run_log.csv", "w", newline="")
writer = csv.writer(log_file)
writer.writerow([
    "time", "bpm", "prob", "status",
    "inference_ms", "cpu_percent", "memory_mb"
])

# =========================
# MAIN
# =========================
def main():
    max30102_init()

    ir_buf = deque(maxlen=BUFFER_SIZE)
    long_buf = deque(maxlen=WINDOW_SIZE)

    last_print = time.time()

    print("Place finger...")

    while True:
        loop_start = time.time()

        ir = read_ir()

        ir_buf.append(ir)
        long_buf.append(ir)

        # buffer fill
        if len(ir_buf) < BUFFER_SIZE:
            print(f"Filling: {len(ir_buf)}/{BUFFER_SIZE}")
            time.sleep(1/FS)
            continue

        # =========================
        # FILTER + PEAK DETECTION
        # =========================
        ir_np = np.array(ir_buf, dtype=np.float32)

        try:
            filtered = bandpass(ir_np, FS).astype(np.float32)
        except:
            filtered = ir_np

        peaks = detect_peaks(filtered, FS)

        # =========================
        # SIGNAL VALIDITY
        # =========================
        if len(peaks) < 3:
            print("Finger not placed correctly")
            continue

        rr = get_rr(peaks, FS)

        if len(rr) < 2:
            continue

        bpm = 60 / np.mean(rr)

        if bpm < 40 or bpm > 180:
            print("Invalid BPM")
            continue

        # =========================
        # BREATHING DETECTION
        # =========================
        breathing_like = False
        if len(rr) > 5:
            if np.mean(np.abs(np.diff(rr))) < 0.05:
                breathing_like = True

        # =========================
        # ML MODEL
        # =========================
        if len(long_buf) < WINDOW_SIZE:
            continue

        signal = np.array(long_buf, dtype=np.float32)

        try:
            filtered_long = bandpass(signal, FS).astype(np.float32)
        except:
            filtered_long = signal

        mean = np.mean(filtered_long)
        std = np.std(filtered_long)

        if std < 1e-3:
            continue

        signal = (filtered_long - mean) / std
        signal = signal.astype(np.float32)
        signal = signal.reshape(1, WINDOW_SIZE, 1)

        # =========================
        # INFERENCE TIMING
        # =========================
        t0 = time.time()

        interpreter.set_tensor(input_details[0]['index'], signal)
        interpreter.invoke()

        t1 = time.time()
        inference_ms = (t1 - t0) * 1000

        output = interpreter.get_tensor(output_details[0]['index'])[0]
        prob = float(output[1])

        # =========================
        # SYSTEM METRICS
        # =========================
        cpu = psutil.cpu_percent()
        mem = process.memory_info().rss / (1024 * 1024)

        # =========================
        # STATUS LOGIC
        # =========================
        if prob > 0.65 and not breathing_like:
            status = "ARRHYTHMIA"
        elif prob > 0.45:
            status = "IRREGULAR"
        else:
            status = "NORMAL"

        # =========================
        # PRINT
        # =========================
        if time.time() - last_print > PRINT_INTERVAL:
            last_print = time.time()

            print("\n----------------------------")
            print(f"BPM: {bpm:.1f}")
            print(f"Prob: {prob:.3f}")
            print(f"Status: {status}")
            print(f"Inference: {inference_ms:.2f} ms")
            print(f"CPU: {cpu:.1f}%")
            print(f"Memory: {mem:.1f} MB")

        # =========================
        # LOG TO CSV
        # =========================
        writer.writerow([
            time.time(),
            bpm,
            prob,
            status,
            inference_ms,
            cpu,
            mem
        ])

        # maintain sampling rate
        elapsed = time.time() - loop_start
        sleep = (1/FS) - elapsed
        if sleep > 0:
            time.sleep(sleep)

# =========================
if __name__ == "__main__":
    main()