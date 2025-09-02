# NSR-SingleleadECG-AFib-Classification

Identification of **Atrial Fibrillation** with **single-lead mobile ECG** during **Normal Sinus Rhythm** using **Deep Learning**

## Introduction
This repository contains the code implementation accompanying the study on identifying atrial fibrillation (AFib) from single-lead mobile ECG recordings during normal sinus rhythm (NSR) using deep learning, as detailed in the associated publication. The focus is on developing and validating a deep learning framework for detecting AFib in mobile ECG data.

The provided code covers data preprocessing, model architecture, training, and evaluation procedures based on the methodologies described in the paper.

Please note that the raw single-lead ECG datasets and corresponding labels used for training are derived from patient data, which is sensitive and confidential. Therefore, these datasets are not publicly available through this repository.

## Features
- Deep learning based classification model for AFib detection from single-lead ECG
- Preprocessing pipelines for raw ECG signals
- Utilization of Neurokit2 for ECG signal processing
- Implementation with PyTorch for flexible and scalable model training
- Designed for mobile ECG recordings, enabling real-world applicability

## Dependensies
- [Pytorch](https://pytorch.org/)
- [Neurokit2](https://neuropsychology.github.io/NeuroKit/)

For further details, please refer to **[_Identification of Atrial Fibrillation With Single-Lead Mobile ECG During Normal Sinus Rhythm Using Deep Learning_](https://doi.org/10.3346/jkms.2024.39.e56)** by Kim JW et al.
