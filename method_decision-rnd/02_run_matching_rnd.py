from pathlib import Path
import json
import math
import warnings

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as structural_similarity


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\Users\PC-1\Downloads\sensory\full_rnd"
)

CONFIG_PATH = BASE_DIR / "config.json"

TEMPLATE_PATH = BASE_DIR / "cup_template.jpg"

GROUND_TRUTH_PATH = (
    BASE_DIR
    / "results"
    / "ground_truth"
    / "annotations.csv"
)

RESULTS_DIR = BASE_DIR / "results"


METHOD_DIRS = {
    "template_matching":
        RESULTS_DIR / "01_template_matching",

    "white_pixel_percentage":
        RESULTS_DIR / "02_white_pixel_percentage",

    "edge_matching":
        RESULTS_DIR / "03_edge_matching",

    "gradient_matching":
        RESULTS_DIR / "04_gradient_matching",

    "ssim":
        RESULTS_DIR / "05_ssim",

    "histogram_similarity":
        RESULTS_DIR / "06_histogram_similarity",

    "hybrid_matching":
        RESULTS_DIR / "07_hybrid_matching",

    "overall":
        RESULTS_DIR / "overall_comparison",
}


# ============================================================
# VIDEO MAPPING
# ============================================================

VIDEO_MACHINES = {
    "normal-op1_E2_S1.mp4":
        ["E2", "S1"],

    "normal-op2_E1_S3_I1.mp4":
        ["E1", "S3", "I1"],

    "normal-op3_E1_E2_S2_S3_G1.mp4":
        ["E1", "E2", "S2", "S3", "G1"],

    "normal-op4_E1_S4_G1.mp4":
        ["E1", "S4", "G1"],

    "normal-op5_E1_E2_S5_S6.mp4":
        ["E1", "E2", "S5", "S6"],

    "normal_op7_E2_G1_S6_I1.mp4":
        ["E2", "G1", "S6", "I1"],
}


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

FRAME_STEP = 3


# Template scales
TEMPLATE_SCALES = [
    0.75,
    1.00,
    1.25,
]


# Binary fixed thresholds
BINARY_THRESHOLDS = [
    100,
    125,
    150,
    175,
]


# Canny experiments
CANNY_CONFIGS = [
    (50, 150),
    (75, 175),
    (100, 200),
]


# ============================================================
# PREPROCESSING PIPELINES
# ============================================================

PREPROCESSING_PIPELINES = [
    "original",
    "gray",
    "gray_median",
    "gray_gaussian",
    "gray_clahe",
    "gray_median_clahe",
    "gray_gaussian_clahe",
]


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

def create_output_directories():

    for path in METHOD_DIRS.values():

        path.mkdir(
            parents=True,
            exist_ok=True
        )

        (path / "csv").mkdir(
            parents=True,
            exist_ok=True
        )

        (path / "graphs").mkdir(
            parents=True,
            exist_ok=True
        )

        (path / "summary").mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# LOAD CONFIG
# ============================================================

def load_config():

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    result = {}

    for item in data:

        result[item["id"]] = {
            "id": item["id"],
            "class_name":
                item.get("class_name", ""),

            "x1": int(item["x1"]),
            "y1": int(item["y1"]),
            "x2": int(item["x2"]),
            "y2": int(item["y2"]),
        }

    return result


# ============================================================
# ROI
# ============================================================

def crop_roi(frame, roi):

    h, w = frame.shape[:2]

    x1 = max(
        0,
        min(roi["x1"], w - 1)
    )

    y1 = max(
        0,
        min(roi["y1"], h - 1)
    )

    x2 = max(
        x1 + 1,
        min(roi["x2"], w)
    )

    y2 = max(
        y1 + 1,
        min(roi["y2"], h)
    )

    return frame[
        y1:y2,
        x1:x2
    ]


# ============================================================
# BASIC IMAGE PROCESSING
# ============================================================

def to_gray(image):

    if image.ndim == 2:
        return image

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


def clahe_image(gray):

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(gray)


def preprocess(image, pipeline):

    if pipeline == "original":
        return image.copy()

    gray = to_gray(image)

    if pipeline == "gray":
        return gray

    if pipeline == "gray_median":

        return cv2.medianBlur(
            gray,
            5
        )

    if pipeline == "gray_gaussian":

        return cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

    if pipeline == "gray_clahe":

        return clahe_image(gray)

    if pipeline == "gray_median_clahe":

        temp = cv2.medianBlur(
            gray,
            5
        )

        return clahe_image(temp)

    if pipeline == "gray_gaussian_clahe":

        temp = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        return clahe_image(temp)

    raise ValueError(
        f"Unknown preprocessing: {pipeline}"
    )


# ============================================================
# TEMPLATE UTILITIES
# ============================================================

def resize_template(template, scale):

    h, w = template.shape[:2]

    new_w = max(
        5,
        int(w * scale)
    )

    new_h = max(
        5,
        int(h * scale)
    )

    return cv2.resize(
        template,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )


def compatible_template(
    roi,
    template
):

    rh, rw = roi.shape[:2]
    th, tw = template.shape[:2]

    if th > rh or tw > rw:
        return False

    return True


# ============================================================
# 1. TEMPLATE MATCHING
# ============================================================

def template_matching_score(
    roi,
    template,
    pipeline,
    scale
):

    proc_roi = preprocess(
        roi,
        pipeline
    )

    proc_template = preprocess(
        template,
        pipeline
    )

    proc_template = resize_template(
        proc_template,
        scale
    )

    if not compatible_template(
        proc_roi,
        proc_template
    ):
        return np.nan

    result = cv2.matchTemplate(
        proc_roi,
        proc_template,
        cv2.TM_CCOEFF_NORMED
    )

    _, max_val, _, max_loc = cv2.minMaxLoc(
        result
    )

    return float(max_val)


# ============================================================
# 2. WHITE PIXEL PERCENTAGE
# ============================================================

def white_percentage_fixed(
    roi,
    pipeline,
    threshold
):

    proc = preprocess(
        roi,
        pipeline
    )

    gray = to_gray(proc)

    _, binary = cv2.threshold(
        gray,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    white = np.count_nonzero(
        binary == 255
    )

    total = binary.size

    return (
        white / total
    ) * 100.0


def white_percentage_otsu(
    roi,
    pipeline
):

    proc = preprocess(
        roi,
        pipeline
    )

    gray = to_gray(proc)

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )

    white = np.count_nonzero(
        binary == 255
    )

    total = binary.size

    return (
        white / total
    ) * 100.0


# ============================================================
# 3. EDGE MATCHING
# ============================================================

def edge_matching_score(
    roi,
    template,
    pipeline,
    scale,
    canny_low,
    canny_high
):

    proc_roi = preprocess(
        roi,
        pipeline
    )

    proc_template = preprocess(
        template,
        pipeline
    )

    roi_gray = to_gray(proc_roi)
    temp_gray = to_gray(proc_template)

    roi_gray = cv2.GaussianBlur(
        roi_gray,
        (5, 5),
        0
    )

    temp_gray = cv2.GaussianBlur(
        temp_gray,
        (5, 5),
        0
    )

    roi_edge = cv2.Canny(
        roi_gray,
        canny_low,
        canny_high
    )

    temp_edge = cv2.Canny(
        temp_gray,
        canny_low,
        canny_high
    )

    temp_edge = resize_template(
        temp_edge,
        scale
    )

    if not compatible_template(
        roi_edge,
        temp_edge
    ):
        return np.nan

    result = cv2.matchTemplate(
        roi_edge,
        temp_edge,
        cv2.TM_CCOEFF_NORMED
    )

    _, max_val, _, _ = cv2.minMaxLoc(
        result
    )

    return float(max_val)


# ============================================================
# 4. GRADIENT MATCHING
# ============================================================

def gradient_magnitude(image):

    gray = to_gray(image)

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(
        gx,
        gy
    )

    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return magnitude.astype(
        np.uint8
    )


def gradient_matching_score(
    roi,
    template,
    pipeline,
    scale
):

    proc_roi = preprocess(
        roi,
        pipeline
    )

    proc_template = preprocess(
        template,
        pipeline
    )

    grad_roi = gradient_magnitude(
        proc_roi
    )

    grad_template = gradient_magnitude(
        proc_template
    )

    grad_template = resize_template(
        grad_template,
        scale
    )

    if not compatible_template(
        grad_roi,
        grad_template
    ):
        return np.nan

    result = cv2.matchTemplate(
        grad_roi,
        grad_template,
        cv2.TM_CCOEFF_NORMED
    )

    _, max_val, _, _ = cv2.minMaxLoc(
        result
    )

    return float(max_val)


# ============================================================
# 5. SSIM
# ============================================================

def ssim_score(
    roi,
    template,
    pipeline
):

    proc_roi = preprocess(
        roi,
        pipeline
    )

    proc_template = preprocess(
        template,
        pipeline
    )

    roi_gray = to_gray(proc_roi)
    temp_gray = to_gray(proc_template)

    # Resize template to ROI size for global
    # structural comparison.
    temp_resized = cv2.resize(
        temp_gray,
        (
            roi_gray.shape[1],
            roi_gray.shape[0]
        ),
        interpolation=cv2.INTER_AREA
    )

    score = structural_similarity(
        roi_gray,
        temp_resized,
        data_range=255
    )

    return float(score)


# ============================================================
# 6. HISTOGRAM SIMILARITY
# ============================================================

def histogram_score(
    roi,
    template,
    pipeline
):

    proc_roi = preprocess(
        roi,
        pipeline
    )

    proc_template = preprocess(
        template,
        pipeline
    )

    roi_gray = to_gray(proc_roi)
    temp_gray = to_gray(proc_template)

    hist_roi = cv2.calcHist(
        [roi_gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    hist_template = cv2.calcHist(
        [temp_gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    cv2.normalize(
        hist_roi,
        hist_roi
    )

    cv2.normalize(
        hist_template,
        hist_template
    )

    score = cv2.compareHist(
        hist_roi,
        hist_template,
        cv2.HISTCMP_CORREL
    )

    return float(score)


# ============================================================
# STATISTICS
# ============================================================

def pooled_effect_size(
    empty,
    cup
):

    empty = np.asarray(
        empty,
        dtype=float
    )

    cup = np.asarray(
        cup,
        dtype=float
    )

    if (
        len(empty) < 2
        or len(cup) < 2
    ):
        return np.nan

    mean_diff = (
        np.mean(cup)
        - np.mean(empty)
    )

    var1 = np.var(
        empty,
        ddof=1
    )

    var2 = np.var(
        cup,
        ddof=1
    )

    pooled = math.sqrt(
        (
            (len(empty) - 1) * var1
            +
            (len(cup) - 1) * var2
        )
        /
        (
            len(empty)
            + len(cup)
            - 2
        )
    )

    if pooled == 0:

        if mean_diff == 0:
            return 0.0

        return np.inf

    return (
        abs(mean_diff)
        / pooled
    )


def distribution_overlap(
    empty,
    cup,
    bins=50
):

    empty = np.asarray(
        empty,
        dtype=float
    )

    cup = np.asarray(
        cup,
        dtype=float
    )

    if (
        len(empty) == 0
        or len(cup) == 0
    ):
        return np.nan

    minimum = min(
        empty.min(),
        cup.min()
    )

    maximum = max(
        empty.max(),
        cup.max()
    )

    if minimum == maximum:
        return 1.0

    edges = np.linspace(
        minimum,
        maximum,
        bins + 1
    )

    h1, _ = np.histogram(
        empty,
        bins=edges,
        density=True
    )

    h2, _ = np.histogram(
        cup,
        bins=edges,
        density=True
    )

    bin_width = edges[1] - edges[0]

    overlap = np.sum(
        np.minimum(h1, h2)
    ) * bin_width

    return float(
        np.clip(
            overlap,
            0,
            1
        )
    )


def calculate_summary(group):

    empty = group[
        group["label"] == "EMPTY"
    ]["score"].dropna().values

    cup = group[
        group["label"] == "CUP"
    ]["score"].dropna().values

    if (
        len(empty) == 0
        or len(cup) == 0
    ):
        return None

    empty_mean = np.mean(empty)
    cup_mean = np.mean(cup)

    difference = (
        cup_mean
        - empty_mean
    )

    direction = (
        "INCREASE"
        if difference > 0
        else "DECREASE"
        if difference < 0
        else "NO_CHANGE"
    )

    effect = pooled_effect_size(
        empty,
        cup
    )

    overlap = distribution_overlap(
        empty,
        cup
    )

    wasserstein = wasserstein_distance(
        empty,
        cup
    )

    return {
        "empty_count":
            len(empty),

        "cup_count":
            len(cup),

        "empty_mean":
            empty_mean,

        "empty_median":
            np.median(empty),

        "empty_std":
            np.std(empty),

        "empty_min":
            np.min(empty),

        "empty_max":
            np.max(empty),

        "empty_range":
            np.ptp(empty),

        "cup_mean":
            cup_mean,

        "cup_median":
            np.median(cup),

        "cup_std":
            np.std(cup),

        "cup_min":
            np.min(cup),

        "cup_max":
            np.max(cup),

        "cup_range":
            np.ptp(cup),

        "mean_difference":
            difference,

        "absolute_mean_difference":
            abs(difference),

        "change_direction":
            direction,

        "effect_size":
            effect,

        "distribution_overlap":
            overlap,

        "wasserstein_distance":
            wasserstein,
    }


# ============================================================
# GRAPHING
# ============================================================

def safe_filename(text):

    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def plot_distribution(
    group,
    output_path,
    title
):

    empty = group[
        group["label"] == "EMPTY"
    ]["score"].dropna()

    cup = group[
        group["label"] == "CUP"
    ]["score"].dropna()

    if (
        len(empty) == 0
        or len(cup) == 0
    ):
        return

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        empty,
        bins=30,
        alpha=0.6,
        label="EMPTY"
    )

    plt.hist(
        cup,
        bins=30,
        alpha=0.6,
        label="CUP"
    )

    plt.xlabel("Feature / Similarity Score")
    plt.ylabel("Frequency")

    plt.title(title)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()


def plot_boxplot(
    group,
    output_path,
    title
):

    empty = group[
        group["label"] == "EMPTY"
    ]["score"].dropna()

    cup = group[
        group["label"] == "CUP"
    ]["score"].dropna()

    if (
        len(empty) == 0
        or len(cup) == 0
    ):
        return

    plt.figure(
        figsize=(8, 6)
    )

    plt.boxplot(
        [empty, cup],
        labels=[
            "EMPTY",
            "CUP"
        ]
    )

    plt.ylabel(
        "Feature / Similarity Score"
    )

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()


def plot_timeline(
    group,
    output_path,
    title
):

    if group.empty:
        return

    group = group.sort_values(
        "frame"
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        group["frame"],
        group["score"],
        linewidth=1
    )

    cup = group[
        group["label"] == "CUP"
    ]

    empty = group[
        group["label"] == "EMPTY"
    ]

    if not empty.empty:

        plt.scatter(
            empty["frame"],
            empty["score"],
            s=8,
            label="EMPTY"
        )

    if not cup.empty:

        plt.scatter(
            cup["frame"],
            cup["score"],
            s=8,
            label="CUP"
        )

    plt.xlabel("Frame")
    plt.ylabel("Score")

    plt.title(title)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()


# ============================================================
# RESULT WRITER
# ============================================================

def save_method_results(
    method_name,
    dataframe
):

    method_dir = METHOD_DIRS[
        method_name
    ]

    csv_dir = method_dir / "csv"
    graph_dir = method_dir / "graphs"
    summary_dir = method_dir / "summary"

    raw_path = (
        csv_dir
        / "all_frames.csv"
    )

    dataframe.to_csv(
        raw_path,
        index=False
    )

    summary_rows = []

    grouping_columns = [
        "video",
        "machine_id",
        "configuration"
    ]

    for keys, group in dataframe.groupby(
        grouping_columns
    ):

        video, machine, config = keys

        stats = calculate_summary(
            group
        )

        if stats is None:
            continue

        row = {
            "video": video,
            "machine_id": machine,
            "configuration": config,
        }

        row.update(stats)

        summary_rows.append(row)

        filename = safe_filename(
            f"{video}_{machine}_{config}"
        )

        plot_distribution(
            group,
            graph_dir
            / f"{filename}_distribution.png",

            f"{method_name} | "
            f"{machine} | {config}"
        )

        plot_boxplot(
            group,
            graph_dir
            / f"{filename}_boxplot.png",

            f"{method_name} | "
            f"{machine} | {config}"
        )

        plot_timeline(
            group,
            graph_dir
            / f"{filename}_timeline.png",

            f"{method_name} | "
            f"{machine} | {config}"
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        summary_dir
        / "per_video_machine_summary.csv",
        index=False
    )

    if not summary_df.empty:

        # ----------------------------------------------------
        # Cross-video machine/configuration summary
        # ----------------------------------------------------

        machine_summary = (
            summary_df
            .groupby(
                [
                    "machine_id",
                    "configuration"
                ],
                as_index=False
            )
            .agg(
                mean_effect_size=(
                    "effect_size",
                    "mean"
                ),

                mean_overlap=(
                    "distribution_overlap",
                    "mean"
                ),

                mean_abs_difference=(
                    "absolute_mean_difference",
                    "mean"
                ),

                mean_wasserstein=(
                    "wasserstein_distance",
                    "mean"
                ),

                videos_tested=(
                    "video",
                    "nunique"
                ),
            )
        )

        machine_summary[
            "quality_score"
        ] = (
            machine_summary[
                "mean_effect_size"
            ]
            *
            (
                1.0
                -
                machine_summary[
                    "mean_overlap"
                ]
            )
        )

        machine_summary = (
            machine_summary
            .sort_values(
                "quality_score",
                ascending=False
            )
        )

        machine_summary.to_csv(
            summary_dir
            / "per_machine_configuration_summary.csv",
            index=False
        )

        # ----------------------------------------------------
        # Overall preprocessing/configuration ranking
        # ----------------------------------------------------

        config_summary = (
            machine_summary
            .groupby(
                "configuration",
                as_index=False
            )
            .agg(
                mean_effect_size=(
                    "mean_effect_size",
                    "mean"
                ),

                mean_overlap=(
                    "mean_overlap",
                    "mean"
                ),

                mean_abs_difference=(
                    "mean_abs_difference",
                    "mean"
                ),

                machines_tested=(
                    "machine_id",
                    "nunique"
                ),
            )
        )

        config_summary[
            "quality_score"
        ] = (
            config_summary[
                "mean_effect_size"
            ]
            *
            (
                1.0
                -
                config_summary[
                    "mean_overlap"
                ]
            )
        )

        config_summary = (
            config_summary
            .sort_values(
                "quality_score",
                ascending=False
            )
        )

        config_summary.to_csv(
            summary_dir
            / "configuration_ranking.csv",
            index=False
        )

    print(
        f"[SAVED] {method_name}"
    )


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiment():

    create_output_directories()

    if not GROUND_TRUTH_PATH.exists():

        raise FileNotFoundError(
            "\nGround truth not found:\n"
            f"{GROUND_TRUTH_PATH}\n\n"
            "Run 01_ground_truth.py first."
        )

    if not TEMPLATE_PATH.exists():

        raise FileNotFoundError(
            f"Template not found:\n"
            f"{TEMPLATE_PATH}"
        )

    roi_map = load_config()

    template = cv2.imread(
        str(TEMPLATE_PATH)
    )

    if template is None:

        raise RuntimeError(
            "Failed to load cup template."
        )

    ground_truth = pd.read_csv(
        GROUND_TRUTH_PATH
    )

    # Only clean states are used for matching evaluation.
    clean_gt = ground_truth[
        ground_truth["label"].isin(
            [
                "EMPTY",
                "CUP"
            ]
        )
    ].copy()

    # Fast lookup:
    #
    # (video, machine, frame) -> label

    gt_lookup = {
        (
            row.video,
            row.machine_id,
            int(row.frame)
        ):
        row.label

        for row in clean_gt.itertuples()
    }

    results = {
        "template_matching": [],
        "white_pixel_percentage": [],
        "edge_matching": [],
        "gradient_matching": [],
        "ssim": [],
        "histogram_similarity": [],
    }

    print("=" * 80)
    print("MATCHING METHOD R&D")
    print("=" * 80)

    # ========================================================
    # PROCESS VIDEOS
    # ========================================================

    for video_name, machines in VIDEO_MACHINES.items():

        video_path = (
            BASE_DIR / video_name
        )

        if not video_path.exists():

            print(
                f"[WARNING] Missing: "
                f"{video_name}"
            )

            continue

        print()
        print("=" * 80)
        print(f"VIDEO: {video_name}")
        print("=" * 80)

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():

            print(
                f"Cannot open {video_name}"
            )

            continue

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        frame_idx = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if frame_idx % FRAME_STEP != 0:

                frame_idx += 1
                continue

            if frame_idx % 300 == 0:

                print(
                    f"Frame "
                    f"{frame_idx}/{total_frames}"
                )

            for machine_id in machines:

                lookup_key = (
                    video_name,
                    machine_id,
                    frame_idx
                )

                if lookup_key not in gt_lookup:
                    continue

                if machine_id not in roi_map:
                    continue

                label = gt_lookup[
                    lookup_key
                ]

                roi = crop_roi(
                    frame,
                    roi_map[machine_id]
                )

                if roi.size == 0:
                    continue

                timestamp = (
                    frame_idx / fps
                    if fps > 0
                    else 0
                )

                base_row = {
                    "video":
                        video_name,

                    "machine_id":
                        machine_id,

                    "frame":
                        frame_idx,

                    "time_seconds":
                        timestamp,

                    "label":
                        label,
                }

                # ============================================
                # TEMPLATE MATCHING
                # ============================================

                for pipeline in PREPROCESSING_PIPELINES:

                    for scale in TEMPLATE_SCALES:

                        score = (
                            template_matching_score(
                                roi,
                                template,
                                pipeline,
                                scale
                            )
                        )

                        results[
                            "template_matching"
                        ].append({
                            **base_row,

                            "configuration":
                                (
                                    f"{pipeline}"
                                    f"_scale_{scale:.2f}"
                                ),

                            "score":
                                score,
                        })

                # ============================================
                # WHITE PIXEL %
                # ============================================

                white_pipelines = [
                    "gray",
                    "gray_median",
                    "gray_gaussian",
                    "gray_clahe",
                    "gray_median_clahe",
                    "gray_gaussian_clahe",
                ]

                for pipeline in white_pipelines:

                    for threshold in BINARY_THRESHOLDS:

                        score = (
                            white_percentage_fixed(
                                roi,
                                pipeline,
                                threshold
                            )
                        )

                        results[
                            "white_pixel_percentage"
                        ].append({
                            **base_row,

                            "configuration":
                                (
                                    f"{pipeline}"
                                    f"_threshold_{threshold}"
                                ),

                            "score":
                                score,
                        })

                    score = (
                        white_percentage_otsu(
                            roi,
                            pipeline
                        )
                    )

                    results[
                        "white_pixel_percentage"
                    ].append({
                        **base_row,

                        "configuration":
                            (
                                f"{pipeline}"
                                f"_otsu"
                            ),

                        "score":
                            score,
                    })

                # ============================================
                # EDGE MATCHING
                # ============================================

                edge_pipelines = [
                    "gray",
                    "gray_gaussian",
                    "gray_clahe",
                    "gray_gaussian_clahe",
                ]

                for pipeline in edge_pipelines:

                    for scale in TEMPLATE_SCALES:

                        for (
                            canny_low,
                            canny_high
                        ) in CANNY_CONFIGS:

                            score = (
                                edge_matching_score(
                                    roi,
                                    template,
                                    pipeline,
                                    scale,
                                    canny_low,
                                    canny_high
                                )
                            )

                            config_name = (
                                f"{pipeline}"
                                f"_scale_{scale:.2f}"
                                f"_canny_"
                                f"{canny_low}_"
                                f"{canny_high}"
                            )

                            results[
                                "edge_matching"
                            ].append({
                                **base_row,

                                "configuration":
                                    config_name,

                                "score":
                                    score,
                            })

                # ============================================
                # GRADIENT MATCHING
                # ============================================

                gradient_pipelines = [
                    "gray",
                    "gray_median",
                    "gray_gaussian",
                    "gray_clahe",
                    "gray_median_clahe",
                    "gray_gaussian_clahe",
                ]

                for pipeline in gradient_pipelines:

                    for scale in TEMPLATE_SCALES:

                        score = (
                            gradient_matching_score(
                                roi,
                                template,
                                pipeline,
                                scale
                            )
                        )

                        results[
                            "gradient_matching"
                        ].append({
                            **base_row,

                            "configuration":
                                (
                                    f"{pipeline}"
                                    f"_scale_{scale:.2f}"
                                ),

                            "score":
                                score,
                        })

                # ============================================
                # SSIM
                # ============================================

                for pipeline in PREPROCESSING_PIPELINES:

                    score = ssim_score(
                        roi,
                        template,
                        pipeline
                    )

                    results[
                        "ssim"
                    ].append({
                        **base_row,

                        "configuration":
                            pipeline,

                        "score":
                            score,
                    })

                # ============================================
                # HISTOGRAM
                # ============================================

                for pipeline in PREPROCESSING_PIPELINES:

                    score = histogram_score(
                        roi,
                        template,
                        pipeline
                    )

                    results[
                        "histogram_similarity"
                    ].append({
                        **base_row,

                        "configuration":
                            pipeline,

                        "score":
                            score,
                    })

            frame_idx += 1

        cap.release()

    # ========================================================
    # SAVE INDIVIDUAL METHODS
    # ========================================================

    method_dataframes = {}

    for method_name, rows in results.items():

        df = pd.DataFrame(rows)

        method_dataframes[
            method_name
        ] = df

        if df.empty:

            print(
                f"[WARNING] No results: "
                f"{method_name}"
            )

            continue

        save_method_results(
            method_name,
            df
        )

    # ========================================================
    # HYBRID
    # ========================================================

    build_hybrid_results(
        method_dataframes
    )

    # ========================================================
    # OVERALL COMPARISON
    # ========================================================

    build_overall_comparison()

    print()
    print("=" * 80)
    print("R&D COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Results saved to:\n"
        f"{RESULTS_DIR}"
    )


# ============================================================
# FIND BEST CONFIGURATION FOR METHOD
# ============================================================

def get_best_configuration(
    method_name
):

    path = (
        METHOD_DIRS[method_name]
        / "summary"
        / "configuration_ranking.csv"
    )

    if not path.exists():
        return None

    df = pd.read_csv(path)

    if df.empty:
        return None

    return str(
        df.iloc[0][
            "configuration"
        ]
    )


# ============================================================
# HYBRID
# ============================================================

def build_hybrid_results(
    method_dataframes
):

    print()
    print("=" * 80)
    print("BUILDING HYBRID FEATURES")
    print("=" * 80)

    candidate_methods = [
        "template_matching",
        "white_pixel_percentage",
        "edge_matching",
        "gradient_matching",
        "ssim",
        "histogram_similarity",
    ]

    selected = {}

    for method in candidate_methods:

        config = get_best_configuration(
            method
        )

        if config is None:
            continue

        df = method_dataframes.get(
            method
        )

        if (
            df is None
            or df.empty
        ):
            continue

        subset = df[
            df["configuration"] == config
        ].copy()

        subset = subset[
            [
                "video",
                "machine_id",
                "frame",
                "time_seconds",
                "label",
                "score",
            ]
        ]

        subset = subset.rename(
            columns={
                "score": method
            }
        )

        selected[method] = subset

        print(
            f"{method}: {config}"
        )

    if len(selected) < 2:

        print(
            "Not enough methods for hybrid."
        )

        return

    merged = None

    for method, df in selected.items():

        if merged is None:

            merged = df.copy()

        else:

            merged = merged.merge(
                df,
                on=[
                    "video",
                    "machine_id",
                    "frame",
                    "time_seconds",
                    "label",
                ],
                how="inner"
            )

    if (
        merged is None
        or merged.empty
    ):
        return

    # --------------------------------------------------------
    # Normalize each feature per machine.
    #
    # Z-score normalization is used here only to put features
    # on comparable scales for feature fusion.
    # --------------------------------------------------------

    normalized_columns = []

    for method in selected.keys():

        norm_col = (
            method + "_normalized"
        )

        normalized_columns.append(
            norm_col
        )

        merged[norm_col] = (
            merged
            .groupby("machine_id")[method]
            .transform(
                lambda x:
                    (
                        x - x.mean()
                    )
                    /
                    (
                        x.std()
                        if x.std() > 1e-12
                        else 1.0
                    )
            )
        )

    # --------------------------------------------------------
    # Absolute standardized feature fusion.
    #
    # At this stage this is an R&D feature, NOT the final
    # CUP/EMPTY decision.
    # --------------------------------------------------------

    merged[
        "hybrid_all_features"
    ] = merged[
        normalized_columns
    ].abs().mean(axis=1)

    hybrid_rows = []

    for row in merged.itertuples():

        hybrid_rows.append({
            "video":
                row.video,

            "machine_id":
                row.machine_id,

            "frame":
                row.frame,

            "time_seconds":
                row.time_seconds,

            "label":
                row.label,

            "configuration":
                "all_best_features_equal_weight",

            "score":
                row.hybrid_all_features,
        })

    # --------------------------------------------------------
    # Pairwise hybrids
    # --------------------------------------------------------

    methods = list(
        selected.keys()
    )

    for i in range(len(methods)):

        for j in range(
            i + 1,
            len(methods)
        ):

            m1 = methods[i]
            m2 = methods[j]

            c1 = (
                m1
                + "_normalized"
            )

            c2 = (
                m2
                + "_normalized"
            )

            config_name = (
                f"{m1}+{m2}"
            )

            score_series = (
                merged[
                    [c1, c2]
                ]
                .abs()
                .mean(axis=1)
            )

            for idx, row in merged.iterrows():

                hybrid_rows.append({
                    "video":
                        row["video"],

                    "machine_id":
                        row["machine_id"],

                    "frame":
                        row["frame"],

                    "time_seconds":
                        row["time_seconds"],

                    "label":
                        row["label"],

                    "configuration":
                        config_name,

                    "score":
                        score_series.loc[idx],
                })

    hybrid_df = pd.DataFrame(
        hybrid_rows
    )

    save_method_results(
        "hybrid_matching",
        hybrid_df
    )


# ============================================================
# OVERALL METHOD COMPARISON
# ============================================================

def build_overall_comparison():

    print()
    print("=" * 80)
    print("OVERALL METHOD COMPARISON")
    print("=" * 80)

    rows = []

    methods = [
        "template_matching",
        "white_pixel_percentage",
        "edge_matching",
        "gradient_matching",
        "ssim",
        "histogram_similarity",
        "hybrid_matching",
    ]

    for method in methods:

        ranking_path = (
            METHOD_DIRS[method]
            / "summary"
            / "configuration_ranking.csv"
        )

        if not ranking_path.exists():
            continue

        df = pd.read_csv(
            ranking_path
        )

        if df.empty:
            continue

        best = df.iloc[0]

        rows.append({
            "method":
                method,

            "best_configuration":
                best["configuration"],

            "mean_effect_size":
                best[
                    "mean_effect_size"
                ],

            "mean_overlap":
                best[
                    "mean_overlap"
                ],

            "mean_abs_difference":
                best[
                    "mean_abs_difference"
                ],

            "machines_tested":
                best[
                    "machines_tested"
                ],

            "quality_score":
                best[
                    "quality_score"
                ],
        })

    comparison = pd.DataFrame(
        rows
    )

    if comparison.empty:
        return

    comparison = comparison.sort_values(
        "quality_score",
        ascending=False
    )

    output_dir = METHOD_DIRS[
        "overall"
    ]

    comparison.to_csv(
        output_dir
        / "method_ranking.csv",
        index=False
    )

    # --------------------------------------------------------
    # Overall ranking graph
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        comparison["method"],
        comparison["quality_score"]
    )

    plt.xticks(
        rotation=35,
        ha="right"
    )

    plt.ylabel(
        "Quality Score"
    )

    plt.title(
        "Overall Matching Method Comparison"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "graphs"
        / "overall_method_ranking.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Effect size
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        comparison["method"],
        comparison["mean_effect_size"]
    )

    plt.xticks(
        rotation=35,
        ha="right"
    )

    plt.ylabel(
        "Mean Standardized Separation"
    )

    plt.title(
        "EMPTY vs CUP Separation"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "graphs"
        / "separation_comparison.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Overlap
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        comparison["method"],
        comparison["mean_overlap"]
    )

    plt.xticks(
        rotation=35,
        ha="right"
    )

    plt.ylabel(
        "Distribution Overlap"
    )

    plt.title(
        "EMPTY vs CUP Distribution Overlap "
        "(Lower is Better)"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "graphs"
        / "overlap_comparison.png",
        dpi=150
    )

    plt.close()

    print()
    print(comparison.to_string(
        index=False
    ))


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_experiment()