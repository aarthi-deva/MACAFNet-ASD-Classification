# MACAFNet: Multi-Atlas Context-Aware Fusion Network for ASD Classification

## Overview
This repository contains the implementation of **MACAFNet**, a deep learning framework for **Autism Spectrum Disorder (ASD) classification** using multi-atlas functional connectivity features derived from resting-state fMRI (rs-fMRI) data.

The model leverages:
- Multi-atlas feature extraction
- Transformer-based attention mechanism
- Context-aware fusion for improved classification performance

> **Note:** This model is intended for research purposes only and is **not a clinical diagnostic tool**.

---

## Dataset

This study uses the **ABIDE I dataset** from the Preprocessed Connectomes Project (PCP).

- **Dataset**: ABIDE I
- **Preprocessing Pipeline**: CPAC
- **Strategy**: filt_global (with global signal regression)
- **Data Type**: Resting-state fMRI
- **Source**: http://preprocessed-connectomes-project.org/abide/

### Atlases Used
The following atlases were used to extract ROI-based features:
- CC200
- Dosenbach160
- AAL
- EZ
- TT
- HO

### Feature Extraction
1. ROI time-series extracted from each atlas  
2. Pearson correlation computed → Functional Connectivity (FC) matrix  
3. Upper triangular elements vectorized → feature vector  

---

## Data Preparation

This repository includes scripts to download and preprocess ABIDE I data for multi-atlas ASD classification.

### Description
The data preparation pipeline:
- Downloads preprocessed ABIDE data
- Extracts ROI time-series
- Computes functional connectivity matrices
- Stores features in HDF5 format
- Generates train/validation/test splits for experiments

### Code Attribution
Parts of the data download and preprocessing pipeline are adapted from the MADE-for-ASD repository:  
https://github.com/hasan-rakibul/MADE-for-ASD  

Authors: Xuehan Liu et al.  

---

## Novelty of This Work
The original pipeline has been extended and adapted for the proposed MACAFNet framework:
- Integrated multi-atlas feature extraction (CC200, AAL, HO, EZ, TT, Dosenbach160)  
- Adapted preprocessing pipeline for MACAFNet architecture  
- Added structured HDF5 storage for multi-atlas data  
- Modified fold preparation for experimental consistency  
- Improved handling of missing ROI values and normalization  

---

## Installation

### Requirements
- Python 3.8+
- PyTorch
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- h5py

### Install dependencies
```bash
pip install -r requirements.txt
