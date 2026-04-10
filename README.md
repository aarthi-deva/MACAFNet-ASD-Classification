# MACAFNet: Multi-Atlas Context-Aware Fusion Network for ASD Classification

## Overview
This repository contains the implementation of **MACAFNet**, a deep learning framework for **Autism Spectrum Disorder (ASD) classification** using multi-atlas functional connectivity features derived from resting-state fMRI (rs-fMRI) data.

The model leverages:
- Multi-atlas feature extraction
- Transformer-based attention mechanism
- Context-aware fusion for improved classification performance

Note: This model is intended for **research purposes only** and is **not a clinical diagnostic tool**.

---

## Dataset

This study uses the **ABIDE I dataset** from the Preprocessed Connectomes Project.

- **Dataset**: ABIDE I  
- **Preprocessing Pipeline**: CPAC  
- **Data Type**: Resting-state fMRI  
- **Source**: http://preprocessed-connectomes-project.org/abide/

### Atlases Used
The following atlases were used to extract ROI-based features:
- CC200
- CC400
- AAL
- EZ
- TT
- HO

### Feature Extraction
1. ROI time-series extracted from each atlas
2. Pearson correlation computed → Functional Connectivity (FC) matrix
3. Upper triangular elements vectorized → feature vector

---

## Installation

### Requirements
- Python 3.8+
- PyTorch
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

### Install dependencies
```bash
pip install -r requirements.txt
