# Human Activity Recognition (HAR) for Parkinson’s Disease

This module contains the implementation of Human Activity Recognition (HAR) models based on wearable inertial sensor data.  
The objective is to recognize motor activities relevant to Parkinson’s disease and provide a contextual layer for subsequent Freezing of Gait (FoG) detection.

---

## Overview

Human Activity Recognition is treated as a foundational component for modeling motor behavior.  
This module focuses on:

- classification of clinically relevant activities (e.g., walking, turning, sit-to-stand)  
- learning robust representations of movement patterns  
- supporting downstream FoG detection models  

Only the best-performing models from the experimental analysis are included.

---

## Implemented Models

### Random Forest (RF)
- trained on extracted features  
- robust baseline with strong performance  
- interpretable and computationally efficient  

### Long Short-Term Memory (LSTM)
- operates on time-series segments  
- captures temporal dependencies in movement signals  
- achieves high performance under subject-independent evaluation  
