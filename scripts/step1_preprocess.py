import pandas as pd
import numpy as np
import tarfile
import torch
import io
from pathlib import Path

DATA_ROOT = r"C:\Users\Administrator\Downloads\UMass\Data"

LABEL_DIR = Path(DATA_ROOT) / "Synapse-20260407T172924Z-3-002" / "Synapse" / "03_Labels"
TAR_DIR = Path(DATA_ROOT) / "GitHub-20260407T161501Z-3-008" / "GitHub" / "tar_PT_1D_PPG_HR_Pulsewatch"

OUT_DIR = Path("processed")
OUT_DIR.mkdir(exist_ok=True)

def get_subject_id(name):
    return int(name[:3])

def find_tar(subject_id):
    return TAR_DIR / f"{subject_id:03d}.tar"

def load_signal_from_pt(raw_bytes):
    buffer = io.BytesIO(raw_bytes)
    data = torch.load(buffer, weights_only=False)

    # extract correct signal
    signal = data["PPG_filt_WEPD"]

    if signal is None or len(signal) != 1500:
        return None

    return signal.astype(np.float32)

def main():
    X = []
    y = []

    for csv_file in LABEL_DIR.glob("*.csv"):
        df = pd.read_csv(csv_file)
        subject_id = get_subject_id(csv_file.name)

        tar_path = find_tar(subject_id)
        if not tar_path.exists():
            continue

        with tarfile.open(tar_path, "r") as tar:
            members = tar.getmembers()

            for _, row in df.iterrows():
                val = float(row["final_AF_GT_20230921"])

                # drop invalid
                if val < 0:
                    continue

                label = 0 if val == 0 else 1
                fname = row["table_file_name"]

                # find matching file
                target = None
                for m in members:
                    if fname in m.name:
                        target = m
                        break

                if target is None:
                    continue

                f = tar.extractfile(target)
                raw = f.read()

                try:
                    signal = load_signal_from_pt(raw)
                except:
                    continue

                if signal is None:
                    continue

                # normalize
                signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

                X.append(signal)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    np.save(OUT_DIR / "X.npy", X)
    np.save(OUT_DIR / "y.npy", y)

    print("Saved dataset")
    print("Shape:", X.shape)
    print("Labels:", np.bincount(y))

if __name__ == "__main__":
    main()