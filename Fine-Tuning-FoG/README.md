# Fine-Tuning for Freezing of Gait (FoG) Detection

This module implements a pipeline for Freezing of Gait (FoG) detection using wearable inertial sensor data, with a focus on **subject-specific adaptation through fine-tuning**.

The approach combines feature-based and deep learning models, and investigates how a model trained on population-level data can be adapted to individual patients.

---

## Overview

FoG detection is challenged by:
- strong inter-subject variability  
- limited availability of labeled freezing events  
- class imbalance  

This module addresses these issues through a two-stage approach:
1. training a general model on a cohort  
2. adapting the model to a target subject via fine-tuning  

---

## Pipeline

The workflow consists of the following stages:

### 1. Data Preprocessing
- signal cleaning (filtering, handling missing data)  
- segmentation into fixed-length temporal windows  
- window labeling using majority voting  

---

### 2. Feature Extraction
Feature extraction is performed using TSFEL, covering:

- statistical features  
- temporal features  
- spectral features  

Multiple window configurations (length and overlap) are explored to capture different temporal dynamics.

---

### 3. Classification and Evaluation

Both machine learning and deep learning models are implemented.

Models are trained on population-level data and evaluated using subject-independent splits.

### 4. Fine-Tuning

Fine-tuning is applied to adapt pretrained models to individual subjects.

- a model is first trained on the full training cohort  
- subject-specific data are then used for adaptation  
- only the added part of the model is updated during fine-tuning  

This approach allows:
- personalization with limited data  
- improved performance on target subjects  
- better alignment with individual motor patterns  

