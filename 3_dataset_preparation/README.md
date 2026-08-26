# Dataset Preparation

This module contains the dataset preparation workflow used to build and expand training datasets from raw videos.

The goal is to efficiently collect useful images from large numbers of videos, reduce unnecessary manual searching, and progressively improve the dataset through model-assisted collection.

The workflow uses two collection approaches:

1. Template matching-assisted candidate collection
2. Model-assisted candidate collection using a trained SmallCNN

After images are collected, they are manually reviewed and labeled using Roboflow before being prepared for model training.

---

## Overview

The dataset preparation workflow is designed as an iterative process:

```text
Raw Videos
    ↓
Dataset Candidate Collection
    │
    ├── Template Matching
    │
    └── Model-Assisted Collection
    ↓
Collected Candidate Images
    ↓
Manual Review
    ↓
Roboflow Labeling
    ↓
Dataset Preparation
    ↓
Train SmallCNN
    ↓
Model-Assisted Collection
    ↓
Collect New Candidates
    ↓
Roboflow Labeling
    ↓
Updated Dataset
    ↓
Retrain Model
