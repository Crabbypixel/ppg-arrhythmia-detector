from __future__ import annotations

import argparse
import io
import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


def find_dirs(data_root: Path):
    labels_dir = None
    tar_dirs = []

    for p in [data_root, *data_root.rglob("*")]:
        if p.is_dir() and p.name.lower() == "03_labels":
            labels_dir = p
        if p.is_dir() and "tar" in p.name.lower():
            # Catch both tar folders used by the dataset tree.
            if any(x in p.name.lower() for x in ("ppg", "linearinterp", "pulsewatch")):
                tar_dirs.append(p)

    tar_dirs = sorted(set(tar_dirs))
    return labels_dir, tar_dirs


def try_read_table_from_bytes(raw_bytes: bytes):
    # Try CSV/TSV/space-separated text first.
    try:
        text = raw_bytes.decode("utf-8", errors="replace")
    except Exception:
        text = None

    if text is not None:
        for sep in [",", "\t", ";", " "]:
            try:
                df = pd.read_csv(io.StringIO(text), sep=sep)
                if df.shape[1] > 1:
                    return ("table", df)
            except Exception:
                pass

    # Try numpy.
    try:
        arr = np.load(io.BytesIO(raw_bytes), allow_pickle=True)
        return ("npy", arr)
    except Exception:
        pass

    # Try mat.
    try:
        mat = loadmat(io.BytesIO(raw_bytes), squeeze_me=True, struct_as_record=False)
        return ("mat", mat)
    except Exception:
        pass

    return ("unknown", None)


def inspect_tar(tar_path: Path, max_files: int = 10):
    print("")
    print("=" * 72)
    print(f"TAR: {tar_path.name}")
    print(f"Size: {tar_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("=" * 72)

    with tarfile.open(tar_path, "r:*") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        print(f"Files inside: {len(members)}")
        for m in members[:max_files]:
            print(f"  {m.name}  ({m.size / 1024:.1f} KB)")

        if not members:
            return

        first = members[0]
        print("")
        print(f"Reading first file: {first.name}")
        raw = tf.extractfile(first).read()
        kind, obj = try_read_table_from_bytes(raw)

        print(f"Detected kind: {kind}")
        if kind == "table":
            df = obj
            print(f"Shape: {df.shape}")
            print("Columns:", list(df.columns))
            print(df.head(3).to_string(index=False))
        elif kind == "npy":
            arr = obj

            if isinstance(arr, np.lib.npyio.NpzFile):
                print("NPZ keys:", list(arr.keys()))
                for k in arr.keys():
                    try:
                        print(f"  {k}: shape={arr[k].shape} dtype={arr[k].dtype}")
                    except:
                        print(f"  {k}: could not read")
            else:
                print("Array shape:", arr.shape)
                print("Dtype:", arr.dtype)
        elif kind == "mat":
            mat = obj
            keys = [k for k in mat.keys() if not k.startswith("__")]
            print("MAT keys:", keys)
            for k in keys[:8]:
                v = mat[k]
                try:
                    arr = np.asarray(v)
                    print(f"  {k}: shape={arr.shape} dtype={arr.dtype}")
                except Exception:
                    print(f"  {k}: type={type(v)}")
        else:
            print("Could not parse first file as table, npy, or mat.")


def inspect_csv(csv_path: Path):
    print("")
    print("=" * 72)
    print(f"CSV: {csv_path.name}")
    print("=" * 72)

    df = pd.read_csv(csv_path)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head(5).to_string(index=False))

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            vals = df[col].dropna()
            print(f"Numeric column {col}: min={vals.min()} max={vals.max()} unique={vals.nunique()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    labels_dir, tar_dirs = find_dirs(data_root)

    print("Data root:", data_root)
    print("Labels dir:", labels_dir)
    print("Tar dirs:", tar_dirs)

    if labels_dir and labels_dir.exists():
        csvs = sorted(labels_dir.glob("*.csv"))
        print(f"\nLabel CSV files: {len(csvs)}")
        for p in csvs[:2]:
            inspect_csv(p)

    for d in tar_dirs[:2]:
        tars = sorted(d.glob("*.tar"))
        print(f"\nTar files in {d.name}: {len(tars)}")
        if tars:
            inspect_tar(tars[0])


if __name__ == "__main__":
    main()
