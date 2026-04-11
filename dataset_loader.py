import torch
from torch.utils.data import Dataset
import h5py

class AutismDataset(Dataset):
    def __init__(self, file_path, atlases=None):
        self.file = h5py.File(file_path, "r")
        self.keys = list(self.file["patients"].keys())
        self.atlases = atlases if atlases is not None else ["aal", "cc200", "dosenbach160", "ho", "ez", "tt"]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        patient = self.file["patients"][self.keys[idx]]
        # Load all atlas data
        data = [torch.tensor(patient[atlas][()]) for atlas in self.atlases]
        # Load label
        label = torch.tensor(patient.attrs["y"])
        return (*data, label)
