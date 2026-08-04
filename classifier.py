"""
classifier.py

Cup detection using:
- ROI crop
- CLAHE
- Template Matching
- Similarity Score

Author: Ahmad Hazim
Project: Coffee Station Interactive Tester
"""

import cv2
import time
import numpy as np
from dataclasses import dataclass
from typing import List


# ==========================================================
# CONFIGURATION
# ==========================================================

TEMPLATE_PATH = "cup_template_ph4.jpg"

MATCH_METHOD = cv2.TM_CCORR_NORMED

CLAHE_CLIP_LIMIT = 2.0

CLAHE_TILE_GRID_SIZE = (8, 8)


# ==========================================================
# RESULT
# ==========================================================

@dataclass
class DetectionResult:

    roi_name: str

    status: str

    threshold: float

    white_percentage: float      # kept for compatibility

    execution_time: float

    match_location: tuple

    similarity: float

    debug_image: np.ndarray

    timestamp: float


# ==========================================================
# CLASSIFIER
# ==========================================================

class CupClassifier:

    def __init__(self):

        # CLAHE
        self.clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID_SIZE
        )

        # Load template
        template = cv2.imread(TEMPLATE_PATH)

        if template is None:
            raise FileNotFoundError(
                f"Cannot load template: {TEMPLATE_PATH}"
            )

        template = cv2.cvtColor(
            template,
            cv2.COLOR_BGR2GRAY
        )

        template = cv2.medianBlur(template, 3)

        self.template = self.clahe.apply(template)

        self.template = template

    # --------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------

    def preprocess(self, crop):

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY
        )
        # Median Blur
        gray = cv2.medianBlur(gray, 3)

        gray = self.clahe.apply(gray)

        return gray

    # --------------------------------------------------
    # TEMPLATE MATCH
    # --------------------------------------------------

    def match_template(self, crop):

        roi_gray = self.preprocess(crop)

        result = cv2.matchTemplate(
            roi_gray,
            self.template,
            MATCH_METHOD
        )

        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        similarity = max_val * 100

        return similarity, max_loc, roi_gray

    # --------------------------------------------------
    # DETECT ONE ROI
    # --------------------------------------------------

    def detect(self, crop, roi):

        if crop is None or crop.size == 0:
            raise ValueError("Empty ROI image.")

        start = time.perf_counter()

        similarity, location, debug = self.match_template(crop)

        threshold = roi["threshold"]

        status = (
            "CUP"
            if similarity >= threshold
            else "EMPTY"
        )

        elapsed = (time.perf_counter() - start) * 1000

        return DetectionResult(

            roi_name=roi["id"],

            status=status,

            threshold=threshold,

            # compatibility with existing UI
            white_percentage=similarity,

            execution_time=elapsed,

            match_location=location,

            similarity=similarity,

            debug_image=debug,

            timestamp=time.time()

        )

    # --------------------------------------------------
    # DETECT ROI
    # --------------------------------------------------

    def detect_roi(self, frame, roi):

        crop = frame[
            roi["y1"]:roi["y2"],
            roi["x1"]:roi["x2"]
        ]

        return self.detect(
            crop,
            roi
        )

    # --------------------------------------------------
    # DETECT MULTIPLE
    # --------------------------------------------------

    def detect_multiple(self, frame, rois) -> List[DetectionResult]:

        results = []

        for roi_name, roi in rois.items():

            results.append(
                self.detect_roi(frame, roi)
            )

        return results