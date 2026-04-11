# MACAFNet – Data Preparation Script for Multi-Atlas ASD Classification
This script prepares ABIDE I preprocessed functional connectivity datasets for training and evaluation of the MACAFNet model. It supports multi-atlas ROI-based feature extraction and fold generation for cross-validation.

### Original Source:
The dataset download proceudre and computation of functional connectivity matrix is adapted from the MADE-for-ASD repository:
https://github.com/hasan-rakibul/MADE-for-ASD

### Original Authors: Xuehan Liu et al.

## Modifications by Aarthi Devaraj:
- Integrated support for multi-atlas feature extraction (CC200, AAL, HO, EZ, TT, Dosenbach160)
- Adapted preprocessing pipeline for MACAFNet architecture
- Added structured HDF5 storage for multi-atlas data
- Modified fold preparation for experimental consistency
- Improved handling of missing ROI values and normalization

### License:
This file is a derivative work licensed under the Apache License, Version 2.0. You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

### Disclaimer:
This code is intended for research purposes only and is not a clinical diagnostic tool.
