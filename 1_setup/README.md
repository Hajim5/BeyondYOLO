# Setup and ROI Configuration

This module prepares a new environment for the BeyondYOLO pipeline.

Its main purpose is to identify the relevant detection regions from input images or video and store them in a reusable configuration file.

The output of this stage is a confirmed configuration that can be used by later detection methods.

The general principle is:

> Configure the environment once, then reuse the configuration for lightweight detection.

---

# Pipeline Position

`01_setup` is the first stage of the BeyondYOLO pipeline.

```text
Input Video / Images
        ↓
01_setup
        ↓
Detect Candidate ROI
        ↓
Review and Confirm ROI
        ↓
config.json
        ↓
02_template_matching
