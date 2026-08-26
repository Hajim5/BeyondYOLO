# Model-Assisted Dataset Collection

This module collects new dataset candidates from raw videos using an existing trained image classification model.

It is intended for iterative dataset expansion.

Instead of manually reviewing every frame from every video, the trained model first analyzes ROI images and groups them according to its predictions. The collected images can then be manually verified and added back into the dataset for further training.

---

## Overview

The model-assisted collection pipeline works as follows:

```text
Raw Videos
    ↓
Load ROI Configuration
    ↓
Sample Frames
    ↓
Crop Machine / Object ROIs
    ↓
Batch Model Inference
    ↓
Prediction + Confidence Score
    ↓
┌───────────────────────────────┐
│                               │
▼                               ▼
High Confidence             Low Confidence
Prediction                  Prediction
│                               │
▼                               ▼
Predicted Class             Uncertain
│                               │
└───────────────┬───────────────┘
                ↓
        Save Candidate Images
                ↓
        Manual Verification
                ↓
       Add Verified Images
                ↓
          Expanded Dataset
                ↓
           Retrain Model
