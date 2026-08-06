from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\Users\PC-1\Downloads\sensory\full_rnd"
)

EDGE_CSV = (
    BASE_DIR
    / "results"
    / "03_edge_matching"
    / "csv"
    / "all_frames.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "decision_rnd_v2"
)


# ============================================================
# 2. SELECTED EDGE CONFIGURATION
# ============================================================

SELECTED_CONFIGURATION = (
    "gray_gaussian_scale_1.00_canny_75_175"
)


# ============================================================
# 3. VIDEOS
# ============================================================

ALL_VIDEOS = [
    "normal-op1_E2_S1.mp4",
    "normal_op7_E2_G1_S6_I1.mp4",
    "normal-op4_E1_S4_G1.mp4",
    "normal-op5_E1_E2_S5_S6.mp4",
    "normal-op3_E1_E2_S2_S3_G1.mp4",
    "normal-op2_E1_S3_I1.mp4",
]


# ============================================================
# 4. BASELINE VIDEOS
# ============================================================

BASELINE_VIDEOS = [
    "normal-op1_E2_S1.mp4",
    "normal-op4_E1_S4_G1.mp4",
]


# ============================================================
# 5. EXPECTED MACHINES
# ============================================================

EXPECTED_MACHINES = [
    "E1",
    "E2",
    "G1",
    "I1",
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
]


# ============================================================
# 6. SETTINGS
# ============================================================

MIN_EMPTY_SAMPLES = 3
MIN_CUP_SAMPLES = 3

THRESHOLD_SEARCH_POINTS = 1000
BASELINE_MARGIN_SEARCH_POINTS = 500

EPS = 1e-12


# ============================================================
# 7. OUTPUT DIRECTORIES
# ============================================================

DIRS = {
    "audit":
        OUTPUT_DIR / "00_data_audit",

    "baseline":
        OUTPUT_DIR / "01_baseline_only",

    "threshold":
        OUTPUT_DIR / "02_threshold_only",

    "combined":
        OUTPUT_DIR / "03_baseline_threshold",

    "comparison":
        OUTPUT_DIR / "04_fair_comparison",
}


def create_directories():

    for path in DIRS.values():

        path.mkdir(
            parents=True,
            exist_ok=True
        )

        (path / "graphs").mkdir(
            parents=True,
            exist_ok=True
        )

        (path / "confusion_matrices").mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# 8. SAFE DIVIDE
# ============================================================

def safe_divide(a, b):

    if b == 0:
        return 0.0

    return a / b


# ============================================================
# 9. METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum(
        (y_true == "CUP")
        &
        (y_pred == "CUP")
    ))

    tn = int(np.sum(
        (y_true == "EMPTY")
        &
        (y_pred == "EMPTY")
    ))

    fp = int(np.sum(
        (y_true == "EMPTY")
        &
        (y_pred == "CUP")
    ))

    fn = int(np.sum(
        (y_true == "CUP")
        &
        (y_pred == "EMPTY")
    ))

    total = tp + tn + fp + fn

    accuracy = safe_divide(
        tp + tn,
        total
    )

    precision = safe_divide(
        tp,
        tp + fp
    )

    recall = safe_divide(
        tp,
        tp + fn
    )

    specificity = safe_divide(
        tn,
        tn + fp
    )

    f1 = safe_divide(
        2 * precision * recall,
        precision + recall
    )

    balanced_accuracy = (
        recall + specificity
    ) / 2

    fpr = safe_divide(
        fp,
        fp + tn
    )

    fnr = safe_divide(
        fn,
        fn + tp
    )

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,

        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,

        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,

        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
    }


# ============================================================
# 10. BINARY BALANCED ACCURACY
# ============================================================

def binary_balanced_accuracy(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum(
        (y_true == 1)
        &
        (y_pred == 1)
    )

    tn = np.sum(
        (y_true == 0)
        &
        (y_pred == 0)
    )

    fp = np.sum(
        (y_true == 0)
        &
        (y_pred == 1)
    )

    fn = np.sum(
        (y_true == 1)
        &
        (y_pred == 0)
    )

    sensitivity = safe_divide(
        tp,
        tp + fn
    )

    specificity = safe_divide(
        tn,
        tn + fp
    )

    return (
        sensitivity
        + specificity
    ) / 2


# ============================================================
# 11. LOAD EDGE DATA
# ============================================================

def load_edge_data():

    print()
    print("=" * 80)
    print("STEP 1: LOAD SELECTED EDGE RESULTS")
    print("=" * 80)

    if not EDGE_CSV.exists():

        raise FileNotFoundError(
            f"\nEdge CSV not found:\n{EDGE_CSV}"
        )

    df = pd.read_csv(
        EDGE_CSV
    )

    print()
    print(
        f"Original rows: {len(df):,}"
    )

    required = {
        "video",
        "machine_id",
        "frame",
        "label",
        "configuration",
        "score",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Missing columns in Edge CSV: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Show configurations
    # --------------------------------------------------------

    configurations = sorted(
        df[
            "configuration"
        ].dropna().unique()
    )

    print(
        f"Edge configurations found: "
        f"{len(configurations)}"
    )

    # --------------------------------------------------------
    # Select winner ONLY
    # --------------------------------------------------------

    df = df[
        df["configuration"]
        == SELECTED_CONFIGURATION
    ].copy()

    if df.empty:

        raise ValueError(
            "\nSelected configuration not found:\n"
            f"{SELECTED_CONFIGURATION}"
        )

    print()
    print("Selected:")
    print(SELECTED_CONFIGURATION)

    print(
        f"Rows after configuration filter: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Video filter
    # --------------------------------------------------------

    df = df[
        df["video"].isin(
            ALL_VIDEOS
        )
    ].copy()

    # --------------------------------------------------------
    # Clean machine ID
    # --------------------------------------------------------

    df["machine_id"] = (
        df["machine_id"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # --------------------------------------------------------
    # Clean labels
    # --------------------------------------------------------

    df["label"] = (
        df["label"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # Only CUP/EMPTY.
    #
    # SKIP/UNKNOWN/transition etc are removed.
    # --------------------------------------------------------

    df = df[
        df["label"].isin(
            [
                "CUP",
                "EMPTY",
            ]
        )
    ].copy()

    df["score"] = pd.to_numeric(
        df["score"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["score"]
    )

    # --------------------------------------------------------
    # Remove exact duplicate measurements
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=[
            "video",
            "machine_id",
            "frame",
            "configuration",
        ],
        keep="first"
    )

    df = df.sort_values(
        [
            "video",
            "machine_id",
            "frame",
        ]
    ).reset_index(drop=True)

    print(
        f"Final valid CUP/EMPTY rows: "
        f"{len(df):,}"
    )

    print()
    print("Machines actually present:")

    print(
        sorted(
            df["machine_id"].unique()
        )
    )

    return df


# ============================================================
# 12. DATA AUDIT
# ============================================================

def audit_data(df):

    print()
    print("=" * 80)
    print("STEP 2: DATA AUDIT")
    print("=" * 80)

    rows = []

    for machine in EXPECTED_MACHINES:

        machine_df = df[
            df["machine_id"]
            == machine
        ]

        total = len(
            machine_df
        )

        empty_count = int(
            np.sum(
                machine_df["label"]
                == "EMPTY"
            )
        )

        cup_count = int(
            np.sum(
                machine_df["label"]
                == "CUP"
            )
        )

        videos = sorted(
            machine_df[
                "video"
            ].unique()
        )

        baseline_df = machine_df[
            (
                machine_df["video"]
                .isin(BASELINE_VIDEOS)
            )
            &
            (
                machine_df["label"]
                == "EMPTY"
            )
        ]

        baseline_count = len(
            baseline_df
        )

        baseline_videos = sorted(
            baseline_df[
                "video"
            ].unique()
        )

        edge_available = (
            total > 0
        )

        baseline_available = (
            baseline_count > 0
        )

        threshold_available = (
            empty_count
            >= MIN_EMPTY_SAMPLES
            and
            cup_count
            >= MIN_CUP_SAMPLES
        )

        reasons = []

        if not edge_available:

            reasons.append(
                "No selected Edge data"
            )

        if edge_available and empty_count == 0:

            reasons.append(
                "No EMPTY labels"
            )

        if edge_available and cup_count == 0:

            reasons.append(
                "No CUP labels"
            )

        if not baseline_available:

            reasons.append(
                "No EMPTY data in baseline videos"
            )

        if (
            empty_count
            < MIN_EMPTY_SAMPLES
        ):

            reasons.append(
                f"EMPTY samples < "
                f"{MIN_EMPTY_SAMPLES}"
            )

        if (
            cup_count
            < MIN_CUP_SAMPLES
        ):

            reasons.append(
                f"CUP samples < "
                f"{MIN_CUP_SAMPLES}"
            )

        if not reasons:

            reason_text = "OK"

        else:

            reason_text = "; ".join(
                reasons
            )

        rows.append({
            "machine_id":
                machine,

            "edge_data_available":
                edge_available,

            "total_samples":
                total,

            "empty_samples":
                empty_count,

            "cup_samples":
                cup_count,

            "videos_with_data":
                " | ".join(videos),

            "baseline_empty_samples":
                baseline_count,

            "baseline_videos":
                " | ".join(
                    baseline_videos
                ),

            "baseline_available":
                baseline_available,

            "threshold_available":
                threshold_available,

            "diagnostic":
                reason_text,
        })

        print()
        print(machine)
        print("-" * 40)

        print(
            f"Total Edge samples : "
            f"{total}"
        )

        print(
            f"EMPTY              : "
            f"{empty_count}"
        )

        print(
            f"CUP                : "
            f"{cup_count}"
        )

        print(
            f"Baseline EMPTY     : "
            f"{baseline_count}"
        )

        print(
            f"Baseline available : "
            f"{baseline_available}"
        )

        print(
            f"Threshold possible : "
            f"{threshold_available}"
        )

        print(
            f"Status             : "
            f"{reason_text}"
        )

    audit_df = pd.DataFrame(
        rows
    )

    audit_df.to_csv(
        DIRS["audit"]
        / "machine_data_audit.csv",
        index=False
    )

    # --------------------------------------------------------
    # Per machine/video audit
    # --------------------------------------------------------

    video_rows = []

    for machine in EXPECTED_MACHINES:

        for video in ALL_VIDEOS:

            subset = df[
                (
                    df["machine_id"]
                    == machine
                )
                &
                (
                    df["video"]
                    == video
                )
            ]

            video_rows.append({
                "machine_id":
                    machine,

                "video":
                    video,

                "total_samples":
                    len(subset),

                "empty_samples":
                    int(
                        np.sum(
                            subset["label"]
                            == "EMPTY"
                        )
                    ),

                "cup_samples":
                    int(
                        np.sum(
                            subset["label"]
                            == "CUP"
                        )
                    ),

                "is_baseline_video":
                    video
                    in BASELINE_VIDEOS,
            })

    pd.DataFrame(
        video_rows
    ).to_csv(
        DIRS["audit"]
        / "machine_video_audit.csv",
        index=False
    )

    return audit_df


# ============================================================
# 13. LEARN BASELINES
# ============================================================

def learn_baselines(df):

    print()
    print("=" * 80)
    print("STEP 3: LEARN EMPTY BASELINES")
    print("=" * 80)

    rows = []

    for machine in EXPECTED_MACHINES:

        machine_baseline = df[
            (
                df["machine_id"]
                == machine
            )
            &
            (
                df["video"]
                .isin(BASELINE_VIDEOS)
            )
            &
            (
                df["label"]
                == "EMPTY"
            )
        ].copy()

        if machine_baseline.empty:

            print(
                f"{machine}: "
                f"NO BASELINE"
            )

            continue

        scores = (
            machine_baseline[
                "score"
            ].to_numpy()
        )

        row = {
            "machine_id":
                machine,

            "baseline_samples":
                len(scores),

            "baseline_mean":
                float(
                    np.mean(scores)
                ),

            "baseline_median":
                float(
                    np.median(scores)
                ),

            "baseline_std":
                float(
                    np.std(scores)
                ),

            "baseline_min":
                float(
                    np.min(scores)
                ),

            "baseline_max":
                float(
                    np.max(scores)
                ),

            "baseline_range":
                float(
                    np.max(scores)
                    -
                    np.min(scores)
                ),
        }

        # ----------------------------------------------------
        # Separate baseline video statistics
        # ----------------------------------------------------

        video_medians = []

        for i, video in enumerate(
            BASELINE_VIDEOS,
            start=1
        ):

            subset = machine_baseline[
                machine_baseline["video"]
                == video
            ]

            row[
                f"baseline_video_{i}"
            ] = video

            row[
                f"baseline_video_{i}_samples"
            ] = len(subset)

            if subset.empty:

                row[
                    f"baseline_video_{i}_median"
                ] = np.nan

                row[
                    f"baseline_video_{i}_mean"
                ] = np.nan

            else:

                median_value = float(
                    subset["score"].median()
                )

                row[
                    f"baseline_video_{i}_median"
                ] = median_value

                row[
                    f"baseline_video_{i}_mean"
                ] = float(
                    subset["score"].mean()
                )

                video_medians.append(
                    median_value
                )

        if len(video_medians) >= 2:

            row[
                "baseline_video_difference"
            ] = float(
                max(video_medians)
                -
                min(video_medians)
            )

        else:

            row[
                "baseline_video_difference"
            ] = np.nan

        rows.append(row)

        print(
            f"{machine}: "
            f"baseline="
            f"{row['baseline_median']:.6f} | "
            f"std="
            f"{row['baseline_std']:.6f} | "
            f"samples="
            f"{row['baseline_samples']}"
        )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        DIRS["baseline"]
        / "baseline_statistics.csv",
        index=False
    )

    return result


# ============================================================
# 14. LEARN DIRECTION
# ============================================================

def determine_direction(machine_df):

    empty = machine_df[
        machine_df["label"]
        == "EMPTY"
    ]["score"].to_numpy()

    cup = machine_df[
        machine_df["label"]
        == "CUP"
    ]["score"].to_numpy()

    if (
        len(empty) == 0
        or
        len(cup) == 0
    ):

        return None

    empty_median = float(
        np.median(empty)
    )

    cup_median = float(
        np.median(cup)
    )

    if cup_median > empty_median:

        return "INCREASE"

    if cup_median < empty_median:

        return "DECREASE"

    return "NO_CHANGE"


# ============================================================
# 15. SEARCH THRESHOLD
# ============================================================

def search_threshold(machine_df):

    empty = machine_df[
        machine_df["label"]
        == "EMPTY"
    ]["score"].to_numpy()

    cup = machine_df[
        machine_df["label"]
        == "CUP"
    ]["score"].to_numpy()

    if (
        len(empty)
        < MIN_EMPTY_SAMPLES
        or
        len(cup)
        < MIN_CUP_SAMPLES
    ):

        return None

    direction = determine_direction(
        machine_df
    )

    if (
        direction is None
        or
        direction == "NO_CHANGE"
    ):

        return None

    scores = machine_df[
        "score"
    ].to_numpy()

    labels = machine_df[
        "label"
    ].to_numpy()

    y_true = np.where(
        labels == "CUP",
        1,
        0
    )

    score_min = float(
        np.min(scores)
    )

    score_max = float(
        np.max(scores)
    )

    if (
        score_max
        - score_min
        <= EPS
    ):

        return None

    candidates = np.linspace(
        score_min,
        score_max,
        THRESHOLD_SEARCH_POINTS
    )

    best = None

    for threshold in candidates:

        if direction == "INCREASE":

            y_pred = (
                scores >= threshold
            ).astype(int)

        else:

            y_pred = (
                scores <= threshold
            ).astype(int)

        balanced = (
            binary_balanced_accuracy(
                y_true,
                y_pred
            )
        )

        tp = np.sum(
            (y_true == 1)
            &
            (y_pred == 1)
        )

        fp = np.sum(
            (y_true == 0)
            &
            (y_pred == 1)
        )

        fn = np.sum(
            (y_true == 1)
            &
            (y_pred == 0)
        )

        precision = safe_divide(
            tp,
            tp + fp
        )

        recall = safe_divide(
            tp,
            tp + fn
        )

        f1 = safe_divide(
            2 * precision * recall,
            precision + recall
        )

        if best is None:

            best = {
                "threshold":
                    float(threshold),

                "balanced_accuracy":
                    balanced,

                "f1":
                    f1,
            }

            continue

        if (
            balanced
            >
            best["balanced_accuracy"]
        ):

            best = {
                "threshold":
                    float(threshold),

                "balanced_accuracy":
                    balanced,

                "f1":
                    f1,
            }

        elif (
            abs(
                balanced
                -
                best[
                    "balanced_accuracy"
                ]
            )
            < EPS
            and
            f1 > best["f1"]
        ):

            best = {
                "threshold":
                    float(threshold),

                "balanced_accuracy":
                    balanced,

                "f1":
                    f1,
            }

    return {
        "threshold":
            best["threshold"],

        "direction":
            direction,

        "training_balanced_accuracy":
            best["balanced_accuracy"],

        "training_f1":
            best["f1"],

        "empty_samples":
            len(empty),

        "cup_samples":
            len(cup),

        "empty_mean":
            float(
                np.mean(empty)
            ),

        "cup_mean":
            float(
                np.mean(cup)
            ),

        "empty_median":
            float(
                np.median(empty)
            ),

        "cup_median":
            float(
                np.median(cup)
            ),

        "median_difference":
            float(
                np.median(cup)
                -
                np.median(empty)
            ),
    }


# ============================================================
# 16. LEARN THRESHOLDS
# ============================================================

def learn_thresholds(df):

    print()
    print("=" * 80)
    print("STEP 4: LEARN THRESHOLDS")
    print("=" * 80)

    rows = []

    for machine in EXPECTED_MACHINES:

        machine_df = df[
            df["machine_id"]
            == machine
        ]

        result = search_threshold(
            machine_df
        )

        if result is None:

            print(
                f"{machine}: "
                f"THRESHOLD UNAVAILABLE"
            )

            continue

        row = {
            "machine_id":
                machine,
        }

        row.update(result)

        rows.append(row)

        print(
            f"{machine}: "
            f"{result['direction']} | "
            f"threshold="
            f"{result['threshold']:.6f} | "
            f"balanced="
            f"{result['training_balanced_accuracy']:.3f}"
        )

    result_df = pd.DataFrame(
        rows
    )

    result_df.to_csv(
        DIRS["threshold"]
        / "learned_thresholds.csv",
        index=False
    )

    return result_df


# ============================================================
# 17. SEARCH BASELINE MARGIN
# ============================================================

def search_baseline_margin(
    machine_df,
    baseline,
    direction
):

    scores = machine_df[
        "score"
    ].to_numpy()

    labels = machine_df[
        "label"
    ].to_numpy()

    y_true = np.where(
        labels == "CUP",
        1,
        0
    )

    if direction == "INCREASE":

        signed_change = (
            scores - baseline
        )

    elif direction == "DECREASE":

        signed_change = (
            baseline - scores
        )

    else:

        return None

    max_margin = float(
        max(
            np.max(
                np.maximum(
                    signed_change,
                    0
                )
            ),
            EPS
        )
    )

    candidates = np.linspace(
        0,
        max_margin,
        BASELINE_MARGIN_SEARCH_POINTS
    )

    best = None

    for margin in candidates:

        y_pred = (
            signed_change
            >= margin
        ).astype(int)

        balanced = (
            binary_balanced_accuracy(
                y_true,
                y_pred
            )
        )

        tp = np.sum(
            (y_true == 1)
            &
            (y_pred == 1)
        )

        fp = np.sum(
            (y_true == 0)
            &
            (y_pred == 1)
        )

        fn = np.sum(
            (y_true == 1)
            &
            (y_pred == 0)
        )

        precision = safe_divide(
            tp,
            tp + fp
        )

        recall = safe_divide(
            tp,
            tp + fn
        )

        f1 = safe_divide(
            2 * precision * recall,
            precision + recall
        )

        if best is None:

            best = {
                "margin":
                    float(margin),

                "balanced_accuracy":
                    balanced,

                "f1":
                    f1,
            }

            continue

        if (
            balanced
            >
            best[
                "balanced_accuracy"
            ]
        ):

            best = {
                "margin":
                    float(margin),

                "balanced_accuracy":
                    balanced,

                "f1":
                    f1,
            }

        elif (
            abs(
                balanced
                -
                best[
                    "balanced_accuracy"
                ]
            )
            < EPS
            and
            f1 > best["f1"]
        ):

            best = {
                "margin":
                    float(margin),

                "balanced_accuracy":
                    balanced,

                "f1":
                    f1,
            }

    return best


# ============================================================
# 18. LEARN BASELINE MARGINS
# ============================================================

def learn_baseline_margins(
    df,
    baseline_df,
    threshold_df
):

    print()
    print("=" * 80)
    print("STEP 5: LEARN BASELINE MARGINS")
    print("=" * 80)

    if baseline_df.empty:

        return pd.DataFrame()

    if threshold_df.empty:

        return pd.DataFrame()

    baseline_lookup = (
        baseline_df
        .set_index(
            "machine_id"
        )
        .to_dict(
            "index"
        )
    )

    threshold_lookup = (
        threshold_df
        .set_index(
            "machine_id"
        )
        .to_dict(
            "index"
        )
    )

    rows = []

    for machine in EXPECTED_MACHINES:

        if machine not in baseline_lookup:

            print(
                f"{machine}: "
                f"NO BASELINE"
            )

            continue

        if machine not in threshold_lookup:

            print(
                f"{machine}: "
                f"NO CUP DIRECTION"
            )

            continue

        baseline = float(
            baseline_lookup[
                machine
            ][
                "baseline_median"
            ]
        )

        direction = (
            threshold_lookup[
                machine
            ][
                "direction"
            ]
        )

        machine_df = df[
            df["machine_id"]
            == machine
        ]

        result = (
            search_baseline_margin(
                machine_df,
                baseline,
                direction
            )
        )

        if result is None:

            continue

        if direction == "INCREASE":

            boundary = (
                baseline
                +
                result["margin"]
            )

        else:

            boundary = (
                baseline
                -
                result["margin"]
            )

        rows.append({
            "machine_id":
                machine,

            "baseline":
                baseline,

            "direction":
                direction,

            "baseline_margin":
                result["margin"],

            "baseline_boundary":
                boundary,

            "training_balanced_accuracy":
                result[
                    "balanced_accuracy"
                ],

            "training_f1":
                result["f1"],
        })

        print(
            f"{machine}: "
            f"baseline="
            f"{baseline:.6f} | "
            f"margin="
            f"{result['margin']:.6f} | "
            f"boundary="
            f"{boundary:.6f} | "
            f"{direction}"
        )

    result_df = pd.DataFrame(
        rows
    )

    result_df.to_csv(
        DIRS["baseline"]
        / "learned_baseline_margins.csv",
        index=False
    )

    return result_df


# ============================================================
# 19. PREDICTION FUNCTIONS
# ============================================================

def baseline_predict(
    score,
    baseline,
    margin,
    direction
):

    if direction == "INCREASE":

        cup = (
            score
            >=
            baseline + margin
        )

    elif direction == "DECREASE":

        cup = (
            score
            <=
            baseline - margin
        )

    else:

        cup = False

    return (
        "CUP"
        if cup
        else "EMPTY"
    )


def threshold_predict(
    score,
    threshold,
    direction
):

    if direction == "INCREASE":

        cup = (
            score >= threshold
        )

    elif direction == "DECREASE":

        cup = (
            score <= threshold
        )

    else:

        cup = False

    return (
        "CUP"
        if cup
        else "EMPTY"
    )


def combined_predict(
    score,
    baseline,
    margin,
    threshold,
    direction
):

    baseline_result = (
        baseline_predict(
            score,
            baseline,
            margin,
            direction
        )
    )

    threshold_result = (
        threshold_predict(
            score,
            threshold,
            direction
        )
    )

    # BOTH must say CUP.
    if (
        baseline_result == "CUP"
        and
        threshold_result == "CUP"
    ):

        return "CUP"

    return "EMPTY"


# ============================================================
# 20. GENERATE ALL PREDICTIONS
# ============================================================

def generate_predictions(
    df,
    baseline_df,
    threshold_df,
    margin_df
):

    print()
    print("=" * 80)
    print("STEP 6: GENERATE RAW PREDICTIONS")
    print("=" * 80)

    baseline_lookup = {}

    threshold_lookup = {}

    margin_lookup = {}

    if not baseline_df.empty:

        baseline_lookup = (
            baseline_df
            .set_index("machine_id")
            .to_dict("index")
        )

    if not threshold_df.empty:

        threshold_lookup = (
            threshold_df
            .set_index("machine_id")
            .to_dict("index")
        )

    if not margin_df.empty:

        margin_lookup = (
            margin_df
            .set_index("machine_id")
            .to_dict("index")
        )

    rows = []

    for row in df.itertuples():

        machine = row.machine_id

        record = {
            "video":
                row.video,

            "machine_id":
                machine,

            "frame":
                int(row.frame),

            "ground_truth":
                row.label,

            "score":
                float(row.score),

            "baseline_available":
                False,

            "threshold_available":
                False,

            "combined_available":
                False,

            "baseline_prediction":
                None,

            "threshold_prediction":
                None,

            "combined_prediction":
                None,
        }

        # ----------------------------------------------------
        # Threshold
        # ----------------------------------------------------

        if machine in threshold_lookup:

            t = threshold_lookup[
                machine
            ]

            record[
                "threshold_available"
            ] = True

            record[
                "threshold"
            ] = float(
                t["threshold"]
            )

            record[
                "direction"
            ] = t["direction"]

            record[
                "threshold_prediction"
            ] = threshold_predict(
                float(row.score),
                float(
                    t["threshold"]
                ),
                t["direction"]
            )

        # ----------------------------------------------------
        # Baseline
        # ----------------------------------------------------

        if (
            machine in baseline_lookup
            and
            machine in margin_lookup
        ):

            b = baseline_lookup[
                machine
            ]

            m = margin_lookup[
                machine
            ]

            record[
                "baseline_available"
            ] = True

            record[
                "baseline"
            ] = float(
                b[
                    "baseline_median"
                ]
            )

            record[
                "baseline_margin"
            ] = float(
                m[
                    "baseline_margin"
                ]
            )

            record[
                "baseline_boundary"
            ] = float(
                m[
                    "baseline_boundary"
                ]
            )

            record[
                "baseline_prediction"
            ] = baseline_predict(
                float(row.score),

                float(
                    b[
                        "baseline_median"
                    ]
                ),

                float(
                    m[
                        "baseline_margin"
                    ]
                ),

                m["direction"]
            )

        # ----------------------------------------------------
        # Combined
        # ----------------------------------------------------

        if (
            machine in baseline_lookup
            and
            machine in margin_lookup
            and
            machine in threshold_lookup
        ):

            b = baseline_lookup[
                machine
            ]

            m = margin_lookup[
                machine
            ]

            t = threshold_lookup[
                machine
            ]

            record[
                "combined_available"
            ] = True

            record[
                "combined_prediction"
            ] = combined_predict(
                float(row.score),

                float(
                    b[
                        "baseline_median"
                    ]
                ),

                float(
                    m[
                        "baseline_margin"
                    ]
                ),

                float(
                    t[
                        "threshold"
                    ]
                ),

                t[
                    "direction"
                ]
            )

        rows.append(record)

    prediction_df = pd.DataFrame(
        rows
    )

    prediction_df.to_csv(
        OUTPUT_DIR
        / "all_decision_predictions.csv",
        index=False
    )

    return prediction_df


# ============================================================
# 21. FULL-COVERAGE EVALUATION
# ============================================================

def evaluate_full_coverage(
    prediction_df
):

    print()
    print("=" * 80)
    print("STEP 7: FULL-COVERAGE RESULTS")
    print("=" * 80)

    methods = {
        "Baseline Only":
            (
                "baseline_available",
                "baseline_prediction"
            ),

        "Threshold Only":
            (
                "threshold_available",
                "threshold_prediction"
            ),

        "Baseline + Threshold":
            (
                "combined_available",
                "combined_prediction"
            ),
    }

    overall_rows = []
    machine_rows = []

    for method, (
        available_column,
        prediction_column
    ) in methods.items():

        subset = prediction_df[
            prediction_df[
                available_column
            ] == True
        ].copy()

        subset = subset[
            subset[
                prediction_column
            ].notna()
        ]

        if subset.empty:
            continue

        metrics = calculate_metrics(
            subset["ground_truth"],
            subset[prediction_column]
        )

        overall_rows.append({
            "method":
                method,

            "samples":
                len(subset),

            "machines":
                subset[
                    "machine_id"
                ].nunique(),

            **metrics
        })

        print()
        print(method)

        print(
            f"Samples           : "
            f"{len(subset)}"
        )

        print(
            f"Machines          : "
            f"{subset['machine_id'].nunique()}"
        )

        print(
            f"Accuracy          : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Balanced Accuracy : "
            f"{metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"F1                : "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"FP                : "
            f"{metrics['FP']}"
        )

        print(
            f"FN                : "
            f"{metrics['FN']}"
        )

        for machine, group in subset.groupby(
            "machine_id"
        ):

            m = calculate_metrics(
                group[
                    "ground_truth"
                ],
                group[
                    prediction_column
                ]
            )

            machine_rows.append({
                "method":
                    method,

                "machine_id":
                    machine,

                "samples":
                    len(group),

                "empty_samples":
                    int(
                        np.sum(
                            group[
                                "ground_truth"
                            ] == "EMPTY"
                        )
                    ),

                "cup_samples":
                    int(
                        np.sum(
                            group[
                                "ground_truth"
                            ] == "CUP"
                        )
                    ),

                **m
            })

    overall_df = pd.DataFrame(
        overall_rows
    )

    machine_df = pd.DataFrame(
        machine_rows
    )

    overall_df.to_csv(
        DIRS["comparison"]
        / "full_coverage_overall.csv",
        index=False
    )

    machine_df.to_csv(
        DIRS["comparison"]
        / "full_coverage_per_machine.csv",
        index=False
    )

    return overall_df, machine_df


# ============================================================
# 22. BUILD COMMON SUBSET
# ============================================================

def build_common_subset(
    prediction_df
):

    print()
    print("=" * 80)
    print("STEP 8: BUILD FAIR COMMON SUBSET")
    print("=" * 80)

    common = prediction_df[
        (
            prediction_df[
                "baseline_available"
            ] == True
        )
        &
        (
            prediction_df[
                "threshold_available"
            ] == True
        )
        &
        (
            prediction_df[
                "combined_available"
            ] == True
        )
        &
        (
            prediction_df[
                "baseline_prediction"
            ].notna()
        )
        &
        (
            prediction_df[
                "threshold_prediction"
            ].notna()
        )
        &
        (
            prediction_df[
                "combined_prediction"
            ].notna()
        )
    ].copy()

    common.to_csv(
        DIRS["comparison"]
        / "common_evaluation_subset.csv",
        index=False
    )

    print()
    print(
        f"Original labelled samples : "
        f"{len(prediction_df)}"
    )

    print(
        f"Common samples            : "
        f"{len(common)}"
    )

    print(
        f"Common machines           : "
        f"{common['machine_id'].nunique()}"
    )

    print()

    if not common.empty:

        print(
            "Machines in fair comparison:"
        )

        for machine in sorted(
            common[
                "machine_id"
            ].unique()
        ):

            subset = common[
                common["machine_id"]
                == machine
            ]

            empty_count = int(
                np.sum(
                    subset[
                        "ground_truth"
                    ] == "EMPTY"
                )
            )

            cup_count = int(
                np.sum(
                    subset[
                        "ground_truth"
                    ] == "CUP"
                )
            )

            print(
                f"  {machine}: "
                f"{len(subset)} samples "
                f"(EMPTY={empty_count}, "
                f"CUP={cup_count})"
            )

    return common


# ============================================================
# 23. CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    metrics,
    title,
    path
):

    matrix = np.array([
        [
            metrics["TN"],
            metrics["FP"]
        ],

        [
            metrics["FN"],
            metrics["TP"]
        ],
    ])

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    image = ax.imshow(
        matrix
    )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_xticklabels([
        "Pred EMPTY",
        "Pred CUP",
    ])

    ax.set_yticklabels([
        "True EMPTY",
        "True CUP",
    ])

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                str(
                    matrix[i, j]
                ),
                ha="center",
                va="center"
            )

    ax.set_title(title)

    fig.colorbar(image)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150
    )

    plt.close()


# ============================================================
# 24. FAIR COMMON-SUBSET EVALUATION
# ============================================================

def evaluate_common_subset(
    common
):

    print()
    print("=" * 80)
    print("STEP 9: FAIR DECISION METHOD COMPARISON")
    print("=" * 80)

    if common.empty:

        print()
        print(
            "ERROR: No common samples."
        )

        print(
            "Check machine_data_audit.csv."
        )

        return (
            pd.DataFrame(),
            pd.DataFrame()
        )

    methods = {
        "Baseline Only":
            "baseline_prediction",

        "Threshold Only":
            "threshold_prediction",

        "Baseline + Threshold":
            "combined_prediction",
    }

    overall_rows = []
    machine_rows = []

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    for method, column in methods.items():

        metrics = calculate_metrics(
            common[
                "ground_truth"
            ],
            common[column]
        )

        overall_rows.append({
            "decision_method":
                method,

            "samples":
                len(common),

            "machines":
                common[
                    "machine_id"
                ].nunique(),

            **metrics
        })

        print()
        print(method)
        print("-" * 50)

        print(
            f"Samples           : "
            f"{len(common)}"
        )

        print(
            f"Accuracy          : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Balanced Accuracy : "
            f"{metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"Precision         : "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall            : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"Specificity       : "
            f"{metrics['specificity']:.4f}"
        )

        print(
            f"F1                : "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"FP                : "
            f"{metrics['FP']}"
        )

        print(
            f"FN                : "
            f"{metrics['FN']}"
        )

        safe_name = (
            method
            .lower()
            .replace(
                " + ",
                "_"
            )
            .replace(
                " ",
                "_"
            )
        )

        save_confusion_matrix(
            metrics,

            (
                f"{method}\n"
                f"Common Evaluation Subset"
            ),

            DIRS["comparison"]
            / "confusion_matrices"
            / f"{safe_name}.png"
        )

    # --------------------------------------------------------
    # Per machine
    # --------------------------------------------------------

    for machine, group in common.groupby(
        "machine_id"
    ):

        for method, column in methods.items():

            metrics = calculate_metrics(
                group[
                    "ground_truth"
                ],
                group[column]
            )

            machine_rows.append({
                "machine_id":
                    machine,

                "decision_method":
                    method,

                "samples":
                    len(group),

                "empty_samples":
                    int(
                        np.sum(
                            group[
                                "ground_truth"
                            ] == "EMPTY"
                        )
                    ),

                "cup_samples":
                    int(
                        np.sum(
                            group[
                                "ground_truth"
                            ] == "CUP"
                        )
                    ),

                **metrics
            })

    overall_df = pd.DataFrame(
        overall_rows
    )

    machine_df = pd.DataFrame(
        machine_rows
    )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    overall_df = overall_df.sort_values(
        [
            "balanced_accuracy",
            "f1",
            "false_positive_rate",
            "accuracy",
        ],

        ascending=[
            False,
            False,
            True,
            False,
        ]
    ).reset_index(
        drop=True
    )

    overall_df.insert(
        0,
        "rank",
        np.arange(
            1,
            len(overall_df) + 1
        )
    )

    overall_df.to_csv(
        DIRS["comparison"]
        / "FAIR_decision_method_ranking.csv",
        index=False
    )

    machine_df.to_csv(
        DIRS["comparison"]
        / "FAIR_per_machine_results.csv",
        index=False
    )

    # --------------------------------------------------------
    # Per-machine winner
    # --------------------------------------------------------

    winners = (
        machine_df
        .sort_values(
            [
                "machine_id",
                "balanced_accuracy",
                "f1",
                "false_positive_rate",
            ],

            ascending=[
                True,
                False,
                False,
                True,
            ]
        )
        .groupby(
            "machine_id",
            as_index=False
        )
        .first()
    )

    winners.to_csv(
        DIRS["comparison"]
        / "FAIR_per_machine_winner.csv",
        index=False
    )

    # --------------------------------------------------------
    # Graph: Balanced accuracy
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        overall_df[
            "decision_method"
        ],
        overall_df[
            "balanced_accuracy"
        ]
    )

    plt.ylim(0, 1)

    plt.ylabel(
        "Balanced Accuracy"
    )

    plt.title(
        "Fair Decision Method Comparison\n"
        "Same Machines + Same Frames"
    )

    plt.tight_layout()

    plt.savefig(
        DIRS["comparison"]
        / "graphs"
        / "fair_balanced_accuracy.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Graph: F1
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        overall_df[
            "decision_method"
        ],
        overall_df[
            "f1"
        ]
    )

    plt.ylim(0, 1)

    plt.ylabel(
        "F1 Score"
    )

    plt.title(
        "Fair Decision Method F1 Comparison\n"
        "Same Machines + Same Frames"
    )

    plt.tight_layout()

    plt.savefig(
        DIRS["comparison"]
        / "graphs"
        / "fair_f1.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Graph: FP/FN
    # --------------------------------------------------------

    x = np.arange(
        len(overall_df)
    )

    width = 0.35

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        x - width / 2,
        overall_df["FP"],
        width,
        label="False Positive"
    )

    plt.bar(
        x + width / 2,
        overall_df["FN"],
        width,
        label="False Negative"
    )

    plt.xticks(
        x,
        overall_df[
            "decision_method"
        ]
    )

    plt.ylabel(
        "Number of Errors"
    )

    plt.title(
        "False Positive / False Negative\n"
        "Fair Common Subset"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        DIRS["comparison"]
        / "graphs"
        / "fair_fp_fn.png",
        dpi=150
    )

    plt.close()

    return (
        overall_df,
        machine_df
    )


# ============================================================
# 25. PER-MACHINE COMPARISON GRAPH
# ============================================================

def plot_per_machine_comparison(
    machine_results
):

    if machine_results.empty:
        return

    pivot = machine_results.pivot(
        index="machine_id",
        columns="decision_method",
        values="balanced_accuracy"
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(12, 7)
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_ylabel(
        "Balanced Accuracy"
    )

    ax.set_xlabel(
        "Machine"
    )

    ax.set_title(
        "Decision Method Performance Per Machine\n"
        "Fair Common Subset"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        DIRS["comparison"]
        / "graphs"
        / "fair_per_machine_balanced_accuracy.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 26. BUILD FINAL PARAMETER TABLE
# ============================================================

def build_parameter_table(
    baseline_df,
    threshold_df,
    margin_df
):

    machines = pd.DataFrame({
        "machine_id":
            EXPECTED_MACHINES
    })

    result = machines.copy()

    if not baseline_df.empty:

        keep = [
            "machine_id",
            "baseline_samples",
            "baseline_mean",
            "baseline_median",
            "baseline_std",
            "baseline_min",
            "baseline_max",
            "baseline_range",
            "baseline_video_difference",
        ]

        result = result.merge(
            baseline_df[keep],
            on="machine_id",
            how="left"
        )

    if not threshold_df.empty:

        keep = [
            "machine_id",
            "threshold",
            "direction",
            "empty_samples",
            "cup_samples",
            "empty_median",
            "cup_median",
            "median_difference",
            "training_balanced_accuracy",
            "training_f1",
        ]

        temp = (
            threshold_df[
                keep
            ].copy()
        )

        temp = temp.rename(
            columns={
                "training_balanced_accuracy":
                    "threshold_training_balanced_accuracy",

                "training_f1":
                    "threshold_training_f1",
            }
        )

        result = result.merge(
            temp,
            on="machine_id",
            how="left"
        )

    if not margin_df.empty:

        keep = [
            "machine_id",
            "baseline_margin",
            "baseline_boundary",
            "training_balanced_accuracy",
            "training_f1",
        ]

        temp = (
            margin_df[
                keep
            ].copy()
        )

        temp = temp.rename(
            columns={
                "training_balanced_accuracy":
                    "baseline_training_balanced_accuracy",

                "training_f1":
                    "baseline_training_f1",
            }
        )

        result = result.merge(
            temp,
            on="machine_id",
            how="left"
        )

    result.to_csv(
        OUTPUT_DIR
        / "learned_decision_parameters.csv",
        index=False
    )

    return result


# ============================================================
# 27. SAVE JSON
# ============================================================

def save_json(parameters):

    output = {
        "selected_matching_method": {
            "method":
                "edge_matching",

            "configuration":
                SELECTED_CONFIGURATION,

            "representation":
                "grayscale",

            "preprocessing":
                "gaussian_blur",

            "template_scale":
                1.00,

            "canny_low":
                75,

            "canny_high":
                175,

            "matching":
                "TM_CCOEFF_NORMED",
        },

        "decision_rnd": {
            "baseline_videos":
                BASELINE_VIDEOS,

            "threshold_videos":
                ALL_VIDEOS,

            "temporal_confirmation":
                False,

            "note":
                (
                    "Temporal stability is deliberately "
                    "excluded from this experiment."
                ),
        },

        "machines": {}
    }

    for row in parameters.to_dict(
        "records"
    ):

        machine = row[
            "machine_id"
        ]

        machine_data = {}

        for key, value in row.items():

            if key == "machine_id":
                continue

            if pd.isna(value):

                machine_data[
                    key
                ] = None

            elif isinstance(
                value,
                (
                    np.integer,
                    int
                )
            ):

                machine_data[
                    key
                ] = int(value)

            elif isinstance(
                value,
                (
                    np.floating,
                    float
                )
            ):

                machine_data[
                    key
                ] = float(value)

            else:

                machine_data[
                    key
                ] = value

        output[
            "machines"
        ][machine] = machine_data

    with open(
        OUTPUT_DIR
        / "decision_parameters.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=4
        )


# ============================================================
# 28. FINAL SUMMARY
# ============================================================

def print_final_summary(
    audit_df,
    ranking_df
):

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print()
    print(
        "Selected matching method:"
    )

    print(
        SELECTED_CONFIGURATION
    )

    print()
    print(
        "Machine audit:"
    )

    for row in audit_df.itertuples():

        print(
            f"{row.machine_id:3} | "
            f"Total={row.total_samples:4} | "
            f"EMPTY={row.empty_samples:4} | "
            f"CUP={row.cup_samples:4} | "
            f"Baseline={str(row.baseline_available):5} | "
            f"Threshold={str(row.threshold_available):5} | "
            f"{row.diagnostic}"
        )

    if (
        ranking_df is not None
        and
        not ranking_df.empty
    ):

        print()
        print("=" * 80)
        print(
            "FAIR RANKING "
            "(SAME MACHINES + SAME FRAMES)"
        )
        print("=" * 80)

        columns = [
            "rank",
            "decision_method",
            "samples",
            "machines",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "FP",
            "FN",
        ]

        print()
        print(
            ranking_df[
                columns
            ].to_string(
                index=False
            )
        )

        winner = (
            ranking_df.iloc[0]
        )

        print()
        print("=" * 80)

        print(
            "CURRENT DECISION METHOD WINNER:"
        )

        print(
            winner[
                "decision_method"
            ]
        )

        print(
            f"Balanced Accuracy: "
            f"{winner['balanced_accuracy']:.4f}"
        )

        print(
            f"F1: "
            f"{winner['f1']:.4f}"
        )

        print(
            f"FP: "
            f"{winner['FP']}"
        )

        print(
            f"FN: "
            f"{winner['FN']}"
        )

        print("=" * 80)


# ============================================================
# 29. MAIN
# ============================================================

def main():

    create_directories()

    print()
    print("=" * 80)
    print("DECISION METHOD R&D V2")
    print("=" * 80)

    print()
    print(
        "Matching method is FIXED."
    )

    print(
        "No matching-method comparison "
        "is performed here."
    )

    print()

    # --------------------------------------------------------
    # Load Edge scores
    # --------------------------------------------------------

    df = load_edge_data()

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    audit_df = audit_data(
        df
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline_df = (
        learn_baselines(
            df
        )
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    threshold_df = (
        learn_thresholds(
            df
        )
    )

    # --------------------------------------------------------
    # Baseline margin
    # --------------------------------------------------------

    margin_df = (
        learn_baseline_margins(
            df,
            baseline_df,
            threshold_df
        )
    )

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    parameters = (
        build_parameter_table(
            baseline_df,
            threshold_df,
            margin_df
        )
    )

    save_json(
        parameters
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = (
        generate_predictions(
            df,
            baseline_df,
            threshold_df,
            margin_df
        )
    )

    # --------------------------------------------------------
    # Full coverage
    #
    # Useful diagnostically,
    # but NOT used for winner selection.
    # --------------------------------------------------------

    evaluate_full_coverage(
        predictions
    )

    # --------------------------------------------------------
    # FAIR common subset
    # --------------------------------------------------------

    common = (
        build_common_subset(
            predictions
        )
    )

    ranking_df, machine_results = (
        evaluate_common_subset(
            common
        )
    )

    # --------------------------------------------------------
    # Per-machine graph
    # --------------------------------------------------------

    plot_per_machine_comparison(
        machine_results
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_final_summary(
        audit_df,
        ranking_df
    )

    print()
    print("=" * 80)
    print("DECISION R&D V2 COMPLETE")
    print("=" * 80)

    print()
    print(
        "Results:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "IMPORTANT FILE #1:"
    )

    print(
        DIRS["audit"]
        / "machine_data_audit.csv"
    )

    print()
    print(
        "IMPORTANT FILE #2:"
    )

    print(
        DIRS["comparison"]
        / "FAIR_decision_method_ranking.csv"
    )

    print()
    print(
        "IMPORTANT FILE #3:"
    )

    print(
        DIRS["comparison"]
        / "FAIR_per_machine_results.csv"
    )

    print()
    print(
        "IMPORTANT FILE #4:"
    )

    print(
        DIRS["comparison"]
        / "FAIR_per_machine_winner.csv"
    )

    print()
    print(
        "IMPORTANT FILE #5:"
    )

    print(
        OUTPUT_DIR
        / "learned_decision_parameters.csv"
    )


if __name__ == "__main__":
    main()