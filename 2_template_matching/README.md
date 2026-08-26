# Template Matching

This module implements the lightweight classical computer vision branch of the BeyondYOLO pipeline.

The objective is to determine whether a target object or condition can be detected reliably using **template matching and baseline comparison** before using a deep learning model.

The module is designed for environments where:

- Camera position is relatively stable
- The target appears in a known Region of Interest (ROI)
- The background and machine structure are relatively consistent
- Low computational cost is important
- A deep learning model may be unnecessary for simple or stable scenarios

The main principle is:

> Use the lowest-computation method that can achieve the required accuracy.

---

# Pipeline Overview

The template matching pipeline works as follows:

```text
config.json
    ↓
Load ROI Configuration
    ↓
2.1 Template Matching
    ↓
Generate Matching Score
    ↓
2.2 Create Baseline
    ↓
Learn Normal Reference Behaviour
    ↓
2.3 Baseline Decision
    ↓
Compare Current Score Against Baseline
    ↓
2.4 Evaluate Results
    ↓
Performance Metrics
    ↓
┌───────────────────────────────┐
│ Does the method meet target   │
│ performance requirements?     │
└───────────────┬───────────────┘
                │
         YES    │    NO
          ↓     │     ↓
   Keep Template │  Review method
   Matching      │  or use fallback
