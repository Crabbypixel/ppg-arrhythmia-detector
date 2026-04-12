import numpy as np
from sklearn.model_selection import train_test_split

X = np.load("processed/X.npy")
y = np.load("processed/y.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

np.save("processed/X_train.npy", X_train)
np.save("processed/X_test.npy", X_test)
np.save("processed/y_train.npy", y_train)
np.save("processed/y_test.npy", y_test)

print("Train:", X_train.shape)
print("Test:", X_test.shape)
