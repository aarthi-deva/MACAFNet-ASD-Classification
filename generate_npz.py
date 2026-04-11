import numpy as np
from dataset_loader import load_dataset

# Load dataset using your existing loader
X, y, sites = load_dataset()

print("Dataset loaded")
print("Total samples:", len(y))

# Save labels + sites
np.savez(
    "dataset_labels_sites.npz",
    labels=y,
    sites=sites
)

print("Saved dataset_labels_sites.npz")
