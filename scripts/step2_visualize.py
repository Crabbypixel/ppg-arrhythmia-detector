from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True)
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir)
    X_path = processed_dir / "X_ppg.npy"
    y_path = processed_dir / "y_labels.npy"

    if not X_path.exists() or not y_path.exists():
        raise SystemExit("Missing X_ppg.npy or y_labels.npy. Run step1_preprocess.py first.")

    X = np.load(X_path)
    y = np.load(y_path)

    fs = 50
    t = np.arange(X.shape[1]) / fs

    print("Dataset shape:", X.shape)
    print("Class counts:")
    for cls, name in [(0, "Normal"), (1, "Arrhythmia")]:
        count = int(np.sum(y == cls))
        print(f"  {cls} {name}: {count}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("PPG Samples", fontsize=13)

    labels = {0: "Normal", 1: "Arrhythmia"}

    for row, cls in enumerate([0, 1]):
        idx = np.where(y == cls)[0]
        for col in range(3):
            ax = axes[row, col]
            if col < len(idx):
                ax.plot(t, X[idx[col]], linewidth=1)
                ax.set_title(f"{labels[cls]} sample {col + 1}")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Norm amp")
                ax.grid(True, alpha=0.3)
            else:
                ax.axis("off")
                ax.text(0.5, 0.5, "No sample", ha="center", va="center")

    plt.tight_layout()
    out1 = processed_dir / "ppg_samples.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", out1)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    counts = [int(np.sum(y == 0)), int(np.sum(y == 1))]
    bars = ax2.bar(["Normal", "Arrhythmia"], counts)
    ax2.set_ylabel("Segments")
    ax2.set_title("Class distribution")
    ax2.grid(axis="y", alpha=0.3)
    for bar, cnt in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(cnt),
                 ha="center", va="bottom")
    plt.tight_layout()
    out2 = processed_dir / "class_distribution.png"
    plt.savefig(out2, dpi=150)
    plt.show()
    print("Saved:", out2)


if __name__ == "__main__":
    main()
