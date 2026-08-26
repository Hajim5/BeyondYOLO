"""
2.1_template_matching.py

GENERAL TEMPLATE MATCHING MODULE

Pipeline:

    config.json
        ↓
    Load ROI
        ↓
    Load Template
        ↓
    Extract ROI
        ↓
    Preprocess Image
        ↓
    Template Matching
        ↓
    Matching Score

This script does NOT make the final decision.

The score can later be used by:

    2.2_baseline_decision.py
"""

from pathlib import Path
import json

import cv2
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_DIR / "config.json"


# ============================================================
# DEFAULT TEMPLATE MATCHING SETTINGS
# ============================================================

DEFAULT_SETTINGS = {

    "gaussian_kernel": [5, 5],

    "canny_low": 75,

    "canny_high": 175,

    "match_method": "TM_CCOEFF_NORMED",

}


# ============================================================
# LOAD CONFIG
# ============================================================

def load_config(config_path=CONFIG_PATH):

    config_path = Path(config_path)

    if not config_path.exists():

        raise FileNotFoundError(
            f"\nConfig file not found:\n"
            f"{config_path}\n"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as file:

        config = json.load(file)

    return config


# ============================================================
# GET MACHINES
# ============================================================

def get_machines(config):

    machines = config.get("machines")

    if machines is None:

        raise KeyError(
            "\n'machines' was not found in config.json\n"
        )

    if not isinstance(
        machines,
        dict
    ):

        raise TypeError(
            "\n'machines' must be a dictionary.\n"
        )

    return machines


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"\nImage not found:\n"
            f"{image_path}\n"
        )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            f"\nCould not read image:\n"
            f"{image_path}\n"
        )

    return image


# ============================================================
# LOAD TEMPLATE
# ============================================================

def load_template(template_path):

    template_path = Path(template_path)

    if not template_path.exists():

        raise FileNotFoundError(
            f"\nTemplate not found:\n"
            f"{template_path}\n"
        )

    template = cv2.imread(
        str(template_path)
    )

    if template is None:

        raise RuntimeError(
            f"\nCould not read template:\n"
            f"{template_path}\n"
        )

    return template


# ============================================================
# EXTRACT ROI
# ============================================================

def extract_roi(
    frame,
    roi
):

    if frame is None:

        return None

    if not isinstance(
        roi,
        (list, tuple)
    ):

        raise TypeError(
            "\nROI must be a list or tuple:\n"
            "[x1, y1, x2, y2]\n"
        )

    if len(roi) != 4:

        raise ValueError(
            "\nROI must contain exactly 4 values:\n"
            "[x1, y1, x2, y2]\n"
        )

    x1, y1, x2, y2 = [

        int(value)

        for value in roi

    ]

    height, width = frame.shape[:2]

    x1 = max(
        0,
        min(x1, width)
    )

    y1 = max(
        0,
        min(y1, height)
    )

    x2 = max(
        0,
        min(x2, width)
    )

    y2 = max(
        0,
        min(y2, height)
    )

    if x2 <= x1:

        return None

    if y2 <= y1:

        return None

    roi_image = frame[
        y1:y2,
        x1:x2
    ]

    if roi_image.size == 0:

        return None

    return roi_image


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(
    image,
    settings=None
):

    if image is None:

        return None

    if settings is None:

        settings = DEFAULT_SETTINGS

    gaussian_kernel = settings.get(
        "gaussian_kernel",
        DEFAULT_SETTINGS["gaussian_kernel"]
    )

    canny_low = settings.get(
        "canny_low",
        DEFAULT_SETTINGS["canny_low"]
    )

    canny_high = settings.get(
        "canny_high",
        DEFAULT_SETTINGS["canny_high"]
    )

    gaussian_kernel = tuple(
        gaussian_kernel
    )

    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    if len(image.shape) == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray = image.copy()

    # --------------------------------------------------------
    # GAUSSIAN BLUR
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        gaussian_kernel,
        0
    )

    # --------------------------------------------------------
    # CANNY EDGE
    # --------------------------------------------------------

    edges = cv2.Canny(
        blurred,
        canny_low,
        canny_high
    )

    return edges


# ============================================================
# PREPARE TEMPLATE
# ============================================================

def prepare_template(
    template,
    settings=None
):

    return preprocess_image(
        template,
        settings
    )


# ============================================================
# CHECK TEMPLATE COMPATIBILITY
# ============================================================

def template_fits(
    roi_image,
    template_image
):

    if roi_image is None:

        return False

    if template_image is None:

        return False

    roi_height, roi_width = (
        roi_image.shape[:2]
    )

    template_height, template_width = (
        template_image.shape[:2]
    )

    if template_width > roi_width:

        return False

    if template_height > roi_height:

        return False

    return True


# ============================================================
# TEMPLATE MATCHING
# ============================================================

def calculate_matching_score(
    roi_image,
    template_edges,
    settings=None
):

    if roi_image is None:

        return None

    if roi_image.size == 0:

        return None

    if template_edges is None:

        return None

    # --------------------------------------------------------
    # PREPROCESS ROI
    # --------------------------------------------------------

    roi_edges = preprocess_image(
        roi_image,
        settings
    )

    if roi_edges is None:

        return None

    # --------------------------------------------------------
    # CHECK SIZE
    # --------------------------------------------------------

    if not template_fits(
        roi_edges,
        template_edges
    ):

        return None

    # --------------------------------------------------------
    # TEMPLATE MATCHING
    # --------------------------------------------------------

    result = cv2.matchTemplate(
        roi_edges,
        template_edges,
        cv2.TM_CCOEFF_NORMED
    )

    # --------------------------------------------------------
    # BEST MATCH
    # --------------------------------------------------------

    min_value, max_value, min_location, max_location = (

        cv2.minMaxLoc(
            result
        )

    )

    return {

        "score": float(max_value),

        "location": [

            int(max_location[0]),

            int(max_location[1])

        ],

        "method": "TM_CCOEFF_NORMED"

    }


# ============================================================
# MATCH SINGLE MACHINE
# ============================================================

def match_machine(
    frame,
    machine_id,
    machine_config,
    template_edges,
    settings=None
):

    roi = machine_config.get(
        "roi"
    )

    if roi is None:

        raise KeyError(
            f"\nROI not found for machine:\n"
            f"{machine_id}\n"
        )

    # --------------------------------------------------------
    # EXTRACT ROI
    # --------------------------------------------------------

    roi_image = extract_roi(
        frame,
        roi
    )

    if roi_image is None:

        return {

            "machine_id": machine_id,

            "score": None,

            "location": None,

            "status": "INVALID_ROI"

        }

    # --------------------------------------------------------
    # TEMPLATE MATCHING
    # --------------------------------------------------------

    result = calculate_matching_score(
        roi_image,
        template_edges,
        settings
    )

    if result is None:

        return {

            "machine_id": machine_id,

            "score": None,

            "location": None,

            "status": "TEMPLATE_DOES_NOT_FIT"

        }

    return {

        "machine_id": machine_id,

        "score": result["score"],

        "location": result["location"],

        "status": "OK"

    }


# ============================================================
# MATCH ALL MACHINES
# ============================================================

def match_all_machines(
    frame,
    config,
    template_path
):

    machines = get_machines(
        config
    )

    settings = config.get(
        "template_matching",
        DEFAULT_SETTINGS
    )

    template = load_template(
        template_path
    )

    template_edges = prepare_template(
        template,
        settings
    )

    results = {}

    for machine_id, machine_config in machines.items():

        result = match_machine(

            frame=frame,

            machine_id=machine_id,

            machine_config=machine_config,

            template_edges=template_edges,

            settings=settings

        )

        results[
            machine_id
        ] = result

    return results


# ============================================================
# MATCH IMAGE
# ============================================================

def match_image(
    image_path,
    config_path,
    template_path
):

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    config = load_config(
        config_path
    )

    frame = load_image(
        image_path
    )

    # --------------------------------------------------------
    # MATCH
    # --------------------------------------------------------

    results = match_all_machines(

        frame=frame,

        config=config,

        template_path=template_path

    )

    return results


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results):

    print()

    print("=" * 70)

    print("TEMPLATE MATCHING RESULTS")

    print("=" * 70)

    for machine_id, result in results.items():

        print()

        print(
            f"Machine : {machine_id}"
        )

        print(
            f"Status  : {result['status']}"
        )

        score = result["score"]

        if score is not None:

            print(
                f"Score   : {score:.6f}"
            )

        else:

            print(
                "Score   : None"
            )


# ============================================================
# EXAMPLE
# ============================================================

def main():

    print()

    print("=" * 70)

    print("2.1 TEMPLATE MATCHING")

    print("=" * 70)

    print()

    print(
        "This module performs template matching only."
    )

    print(
        "Final classification is handled separately."
    )

    print()

    # --------------------------------------------------------
    # CHANGE THESE PATHS
    # --------------------------------------------------------

    image_path = PROJECT_DIR / "input" / "test.jpg"

    template_path = PROJECT_DIR / "templates" / "template.jpg"

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    results = match_image(

        image_path=image_path,

        config_path=CONFIG_PATH,

        template_path=template_path

    )

    print_results(
        results
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    main()
