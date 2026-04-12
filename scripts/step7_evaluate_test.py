import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

X_test = np.load("processed/X_test.npy")
y_test = np.load("processed/y_test.npy")

X_test = X_test[..., np.newaxis]

model = tf.keras.models.load_model("model.keras", compile=False)

y_prob = model.predict(X_test, batch_size=64)
y_pred = np.argmax(y_prob, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

auc = roc_auc_score(y_test, y_prob[:,1])
print("\nROC AUC:", round(auc, 4))

THRESHOLD = 0.52
y_pred_thresh = (y_prob[:,1] > THRESHOLD).astype(int)

print("\nThreshold-based Report:\n")
print(classification_report(y_test, y_pred_thresh))
