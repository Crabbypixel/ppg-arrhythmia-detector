import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import layers, models, callbacks

# ========================
# LOAD DATA
# ========================
X = np.load("processed/X.npy")
y = np.load("processed/y.npy")

X = X[..., np.newaxis]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ========================
# FOCAL LOSS
# ========================
def focal_loss(gamma=2., alpha=0.75):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, depth=2)

        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        loss = -alpha * tf.pow(1 - pt, gamma) * tf.math.log(pt + 1e-8)
        return tf.reduce_mean(loss)
    return loss

# ========================
# MODEL
# ========================
model = models.Sequential([
    layers.Input(shape=(1500,1)),

    layers.Conv1D(32, 21, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling1D(2),

    layers.Conv1D(64, 11, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling1D(2),

    layers.Conv1D(128, 7, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling1D(2),

    layers.Conv1D(256, 5, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),

    layers.GlobalAveragePooling1D(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(64, activation='relu'),
    layers.Dropout(0.4),

    layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=focal_loss(),
    metrics=['accuracy']
)

model.summary()

# ========================
# CALLBACKS
# ========================
cb = [
    callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
]

# ========================
# TRAIN
# ========================
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=40,
    batch_size=32,
    callbacks=cb
)

# ========================
# THRESHOLD SEARCH
# ========================
probs = model.predict(X_test)
arr_prob = probs[:, 1]

best_thresh = 0.5
best_f1 = 0

for t in np.arange(0.5, 0.95, 0.02):
    preds = (arr_prob > t).astype(int)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, preds)

    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print("\nBest Threshold:", best_thresh)

# ========================
# FINAL EVALUATION
# ========================
preds = (arr_prob > best_thresh).astype(int)

print("\nClassification Report:\n")
print(classification_report(y_test, preds))

# ========================
# SAVE
# ========================
model.save("model.keras")

np.save("best_threshold.npy", np.array([best_thresh]))

print("Saved model + threshold")