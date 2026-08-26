# Template Matching Candidate Collection

This module collects representative image candidates from raw videos using template matching and baseline score analysis.

It is part of the dataset preparation stage of the BeyondYOLO pipeline.

The purpose is not to automatically assign final object labels. Instead, template matching is used as a lightweight filtering mechanism to identify different visual regions and collect useful candidate images for manual labeling and model training.

---

## Pipeline Position

```text
Raw Videos
    ↓
01_setup
    ↓
ROI Configuration
    ↓
02_template_matching
    ↓
Baseline Calculation
    ↓
03_dataset_preparation
    ↓
Template Matching Candidate Collection
    ↓
Manual Labeling
    ↓
Training Dataset
    ↓
Small CNN
