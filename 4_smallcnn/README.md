# 04 — SmallCNN

This module provides a lightweight Convolutional Neural Network (CNN) for image classification when classical computer vision methods such as template matching are not sufficiently reliable.

The objective is to maintain a balance between:

- High classification accuracy
- Low computational cost
- Fast inference
- Small model size
- Simple deployment

This stage is part of the broader BeyondYOLO pipeline, where lightweight classical computer vision methods are evaluated before using a neural network.

---

## Overview

The general workflow is:

```text
Image / Video
      │
      ▼
ROI Extraction
      │
      ▼
Classical Computer Vision
Template Matching / Baseline Analysis
      │
      ├── Reliable
      │      │
      │      ▼
      │   Final Decision
      │
      └── Not Reliable
             │
             ▼
      Dataset Preparation
             │
             ▼
          Roboflow
      Label and Organize Data
             │
             ▼
        Train SmallCNN
             │
             ▼
        Model Evaluation
             │
             ▼
       Lightweight Inference
