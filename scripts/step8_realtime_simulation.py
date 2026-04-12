import numpy as np
import tensorflow as tf
import time

X_test = np.load("processed/X_test.npy")
y_test = np.load("processed/y_test.npy")

X_test = X_test[..., np.newaxis]

model = tf.keras.models.load_model("model.keras", compile=False)

THRESHOLD = 0.52

correct = 0
total_time = 0

for i in range(len(X_test)):
    sample = X_test[i:i+1]

    t0 = time.time()
    pred = model.predict(sample, verbose=0)[0]
    t1 = time.time()

    total_time += (t1 - t0)

    prob = pred[1]
    pred_class = 1 if prob > THRESHOLD else 0

    if pred_class == y_test[i]:
        correct += 1

accuracy = correct / len(X_test)
avg_time = total_time / len(X_test)

print("Realtime Accuracy:", round(accuracy, 4))
print("Avg Inference Time:", round(avg_time*1000, 2), "ms")
