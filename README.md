# Wearable-Based Artificial Intelligence for Freezing of Gait Detection in Parkinson’s Disease

This repository contains the code developed during my Ph.D. research on the automatic detection of **Freezing of Gait (FoG)** in Parkinson’s disease using wearable sensors and artificial intelligence.

The work explores FoG detection from complementary perspectives, including:
- human activity recognition (HAR) as a structural layer  
- subject-specific adaptation through fine-tuning  
- mixture-of-experts (MoE) architectures for heterogeneous sensor configurations  

---

##  Repository Structure

The repository is organized into three main folders, each corresponding to a core contribution of the thesis:

### 1. Human Activity Recognition (HAR)
Code for recognizing motor activities from wearable inertial data.

Includes:
- deep learning models (CNN, LSTM, GRU, etc.)  
- preprocessing and feature extraction pipelines  
- training and evaluation scripts  

---

### 2. Fine-Tuning for FoG Detection
Code for subject-specific adaptation of FoG detection models.

Includes:
- baseline FoG detection models  
- transfer learning and fine-tuning pipelines  
- subject-wise evaluation procedures  

---

### 3. Mixture-of-Experts (MoE)
Experimental framework for handling heterogeneous sensor configurations across datasets.

Includes:
- expert-specific architectures  
- gating network implementation  
- Leave-One-Dataset-Out (LODO) evaluation  

*Note: This component is exploratory and under ongoing development.*

---

## Requirements

Main dependencies include:

- Python 3.9+
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- TSFEL
