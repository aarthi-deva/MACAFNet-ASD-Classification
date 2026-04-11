import numpy as np
import h5py
import os
from sklearn.model_selection import train_test_split

data_path = "data/abide.hdf5"   # change if needed

file = h5py.File(data_path, "r")
patients = file["patients"]
keys = list(patients.keys())

labels = np.array([patients[k].attrs["y"] for k in keys])

indices = np.arange(len(labels))

print("Total samples:", len(indices))

# -------------------------
# 70 / 30 split
# -------------------------
train_idx, temp_idx, y_train, y_temp = train_test_split(
    indices,
    labels,
    test_size=0.30,
    stratify=labels,
    random_state=42
)

# -------------------------
# 15 / 15 split
# -------------------------
val_idx, test_idx, _, _ = train_test_split(
    temp_idx,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print("Train:", len(train_idx))
print("Val:", len(val_idx))
print("Test:", len(test_idx))

# save
os.makedirs("splits", exist_ok=True)

np.savez(
    "splits/data_split.npz",
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx
)

print("Saved splits/data_split.npz")
