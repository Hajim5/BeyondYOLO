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
    r"OWN_INPUT"
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
    / "04_baseline_rnd"
)

# ============================================================
# 2. LOCKED EDGE CONFIGURATION
# ============================================================

SELECTED_CONFIGURATION = (
    "OWN_RESULT"
)

# ============================================================
# 3. VIDEOS
# ============================================================

VIDEOS = ["OWN_INPUT"]

# ============================================================
# 4. MACHINES
# ============================================================

EXPECTED_MACHINES = ["OWN_MACHINE"]

# ============================================================
# 5. BASELINE SETTINGS
# ============================================================

# IQR multiplier for suspicious EMPTY detection.
IQR_MULTIPLIER = 1.5

# If there are very few EMPTY samples, IQR filtering can be
# unreliable. We therefore do not automatically remove
# suspicious samples unless there are at least this many.
MIN_SAMPLES_FOR_FILTERING = 5

# Quality assessment.
MIN_BASELINE_SAMPLES = 3
RECOMMENDED_BASELINE_SAMPLES = 5
RECOMMENDED_EMPTY_VIDEOS = 2

# Suspicious samples close together in time will be grouped
# into one suspicious region.
#
# Example sampled frames:
# 100, 103, 106, 109
#
# If frame spacing <= 10 they belong to the same region.
SUSPICIOUS_REGION_MAX_FRAME_GAP = 10

# ============================================================
# 6. OUTPUT DIRECTORIES
# ============================================================

DIRS = {
    "audit":
        OUTPUT_DIR / "00_audit",

    "samples":
        OUTPUT_DIR / "01_baseline_samples",

    "per_video":
        OUTPUT_DIR / "02_per_video",

    "final":
        OUTPUT_DIR / "03_final_baseline",

    "graphs":
        OUTPUT_DIR / "04_graphs",

    "suspicious_graphs":
        OUTPUT_DIR
        / "04_graphs"
        / "suspicious_regions",
}


def create_directories():

    for directory in DIRS.values():

        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# 7. LOAD EDGE DATA
# ============================================================

def load_edge_data():

    print()
    print("=" * 80)
    print("STEP 1: LOAD EDGE MATCHING DATA")
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

    required_columns = {
        "video",
        "machine_id",
        "frame",
        "configuration",
        "score",
        "label",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "\nMissing required columns:\n"
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Standardize text
    # --------------------------------------------------------

    df["video"] = (
        df["video"]
        .astype(str)
        .str.strip()
    )

    df["machine_id"] = (
        df["machine_id"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["configuration"] = (
        df["configuration"]
        .astype(str)
        .str.strip()
    )

    df["label"] = (
        df["label"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # --------------------------------------------------------
    # Numerical conversion
    # --------------------------------------------------------

    df["score"] = pd.to_numeric(
        df["score"],
        errors="coerce"
    )

    df["frame"] = pd.to_numeric(
        df["frame"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "score",
            "frame",
        ]
    )

    # --------------------------------------------------------
    # Keep selected Edge configuration
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

    # --------------------------------------------------------
    # Keep experiment videos
    # --------------------------------------------------------

    df = df[
        df["video"].isin(
            VIDEOS
        )
    ].copy()

    # --------------------------------------------------------
    # Keep valid labels
    # --------------------------------------------------------

    df = df[
        df["label"].isin(
            [
                "EMPTY",
                "CUP",
            ]
        )
    ].copy()

    # --------------------------------------------------------
    # Remove duplicates
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
            "machine_id",
            "video",
            "frame",
        ]
    ).reset_index(
        drop=True
    )

    print()
    print(
        f"Usable rows: {len(df):,}"
    )

    print()
    print(
        "Labels:"
    )

    print(
        df["label"].value_counts()
    )

    print()
    print(
        "Machines:"
    )

    print(
        sorted(
            df["machine_id"].unique()
        )
    )

    return df


# ============================================================
# 8. GROUND TRUTH AUDIT
# ============================================================

def create_ground_truth_audit(
    df
):

    print()
    print("=" * 80)
    print("STEP 2: GROUND-TRUTH AUDIT")
    print("=" * 80)

    rows = []

    for machine in EXPECTED_MACHINES:

        machine_df = df[
            df["machine_id"]
            == machine
        ]

        empty_df = machine_df[
            machine_df["label"]
            == "EMPTY"
        ]

        cup_df = machine_df[
            machine_df["label"]
            == "CUP"
        ]

        rows.append({

            "machine_id":
                machine,

            "total_samples":
                len(machine_df),

            "empty_samples":
                len(empty_df),

            "cup_samples":
                len(cup_df),

            "empty_videos":
                empty_df[
                    "video"
                ].nunique(),

            "cup_videos":
                cup_df[
                    "video"
                ].nunique(),
        })

    audit = pd.DataFrame(
        rows
    )

    audit.to_csv(
        DIRS["audit"]
        / "ground_truth_audit.csv",
        index=False
    )

    print()
    print(
        audit.to_string(
            index=False
        )
    )

    return audit


# ============================================================
# 9. EXTRACT RAW EMPTY / CUP
# ============================================================

def extract_raw_samples(
    df
):

    print()
    print("=" * 80)
    print("STEP 3: EXTRACT RAW EMPTY SAMPLES")
    print("=" * 80)

    empty_df = df[
        df["label"]
        == "EMPTY"
    ].copy()

    cup_df = df[
        df["label"]
        == "CUP"
    ].copy()

    empty_df[
        "baseline_status"
    ] = "RAW_EMPTY"

    cup_df[
        "baseline_status"
    ] = "EXCLUDED_CUP"

    empty_df.to_csv(
        DIRS["samples"]
        / "raw_empty_samples.csv",
        index=False
    )

    cup_df.to_csv(
        DIRS["samples"]
        / "excluded_cup_samples.csv",
        index=False
    )

    print()
    print(
        f"Raw EMPTY samples: {len(empty_df)}"
    )

    print(
        f"CUP samples excluded: {len(cup_df)}"
    )

    return (
        empty_df,
        cup_df
    )


# ============================================================
# 10. DETECT SUSPICIOUS EMPTY SAMPLES
# ============================================================

def detect_suspicious_empty_samples(
    empty_df
):

    print()
    print("=" * 80)
    print("STEP 4: DETECT SUSPICIOUS EMPTY SAMPLES")
    print("=" * 80)

    processed_groups = []

    statistics = []

    for machine in EXPECTED_MACHINES:

        machine_df = empty_df[
            empty_df["machine_id"]
            == machine
        ].copy()

        if machine_df.empty:

            statistics.append({

                "machine_id":
                    machine,

                "raw_empty_samples":
                    0,

                "q1":
                    np.nan,

                "q3":
                    np.nan,

                "iqr":
                    np.nan,

                "lower_bound":
                    np.nan,

                "upper_bound":
                    np.nan,

                "suspicious_samples":
                    0,

                "clean_samples":
                    0,

                "filter_applied":
                    False,
            })

            continue

        scores = machine_df[
            "score"
        ].to_numpy(
            dtype=float
        )

        q1 = float(
            np.percentile(
                scores,
                25
            )
        )

        q3 = float(
            np.percentile(
                scores,
                75
            )
        )

        iqr = (
            q3 - q1
        )

        lower_bound = (
            q1
            -
            IQR_MULTIPLIER
            * iqr
        )

        upper_bound = (
            q3
            +
            IQR_MULTIPLIER
            * iqr
        )

        # ----------------------------------------------------
        # If very few samples exist, we keep all samples.
        #
        # We still calculate the IQR statistics for reference,
        # but do not automatically reject anything.
        # ----------------------------------------------------

        if (
            len(machine_df)
            >= MIN_SAMPLES_FOR_FILTERING
            and iqr > 0
        ):

            suspicious_mask = (
                (
                    machine_df["score"]
                    < lower_bound
                )
                |
                (
                    machine_df["score"]
                    > upper_bound
                )
            )

            filter_applied = True

        else:

            suspicious_mask = pd.Series(
                False,
                index=machine_df.index
            )

            filter_applied = False

        machine_df[
            "is_suspicious"
        ] = suspicious_mask

        machine_df[
            "iqr_q1"
        ] = q1

        machine_df[
            "iqr_q3"
        ] = q3

        machine_df[
            "iqr"
        ] = iqr

        machine_df[
            "iqr_lower_bound"
        ] = lower_bound

        machine_df[
            "iqr_upper_bound"
        ] = upper_bound

        machine_df[
            "baseline_status"
        ] = np.where(
            machine_df[
                "is_suspicious"
            ],
            "SUSPICIOUS_EMPTY",
            "CLEAN_EMPTY"
        )

        suspicious_count = int(
            machine_df[
                "is_suspicious"
            ].sum()
        )

        clean_count = (
            len(machine_df)
            -
            suspicious_count
        )

        statistics.append({

            "machine_id":
                machine,

            "raw_empty_samples":
                len(machine_df),

            "q1":
                q1,

            "q3":
                q3,

            "iqr":
                iqr,

            "lower_bound":
                lower_bound,

            "upper_bound":
                upper_bound,

            "suspicious_samples":
                suspicious_count,

            "clean_samples":
                clean_count,

            "filter_applied":
                filter_applied,
        })

        processed_groups.append(
            machine_df
        )

        print()
        print(
            f"{machine}"
        )

        print(
            f"  Raw EMPTY       : "
            f"{len(machine_df)}"
        )

        print(
            f"  Q1              : "
            f"{q1:.8f}"
        )

        print(
            f"  Q3              : "
            f"{q3:.8f}"
        )

        print(
            f"  IQR             : "
            f"{iqr:.8f}"
        )

        print(
            f"  Lower bound     : "
            f"{lower_bound:.8f}"
        )

        print(
            f"  Upper bound     : "
            f"{upper_bound:.8f}"
        )

        print(
            f"  Suspicious      : "
            f"{suspicious_count}"
        )

        print(
            f"  Clean           : "
            f"{clean_count}"
        )

        print(
            f"  Filter applied  : "
            f"{filter_applied}"
        )

    if processed_groups:

        processed = pd.concat(
            processed_groups,
            ignore_index=True
        )

    else:

        processed = empty_df.copy()

        processed[
            "is_suspicious"
        ] = False

    stats_df = pd.DataFrame(
        statistics
    )

    clean_df = processed[
        processed[
            "is_suspicious"
        ]
        == False
    ].copy()

    suspicious_df = processed[
        processed[
            "is_suspicious"
        ]
        == True
    ].copy()

    # --------------------------------------------------------
    # Save all three datasets
    # --------------------------------------------------------

    processed.to_csv(
        DIRS["samples"]
        / "all_empty_samples_with_flags.csv",
        index=False
    )

    clean_df.to_csv(
        DIRS["samples"]
        / "clean_empty_samples.csv",
        index=False
    )

    suspicious_df.to_csv(
        DIRS["samples"]
        / "suspicious_empty_samples.csv",
        index=False
    )

    stats_df.to_csv(
        DIRS["audit"]
        / "outlier_detection_summary.csv",
        index=False
    )

    return (
        processed,
        clean_df,
        suspicious_df,
        stats_df
    )


# ============================================================
# 11. GROUP SUSPICIOUS SAMPLES INTO REGIONS
# ============================================================

def create_suspicious_regions(
    suspicious_df
):

    print()
    print("=" * 80)
    print("STEP 5: GROUP SUSPICIOUS EMPTY REGIONS")
    print("=" * 80)

    if suspicious_df.empty:

        print()
        print(
            "No suspicious EMPTY samples detected."
        )

        result = pd.DataFrame(
            columns=[
                "machine_id",
                "video",
                "region_id",
                "start_frame",
                "end_frame",
                "samples",
                "median_score",
                "mean_score",
                "min_score",
                "max_score",
            ]
        )

        result.to_csv(
            DIRS["audit"]
            / "suspicious_regions.csv",
            index=False
        )

        return result

    rows = []

    for (
        machine,
        video
    ), group in suspicious_df.groupby(
        [
            "machine_id",
            "video",
        ]
    ):

        group = group.sort_values(
            "frame"
        ).copy()

        frames = group[
            "frame"
        ].to_numpy()

        region_numbers = []

        region = 1

        previous_frame = None

        for frame in frames:

            if previous_frame is None:

                region_numbers.append(
                    region
                )

            else:

                gap = (
                    frame
                    -
                    previous_frame
                )

                if (
                    gap
                    >
                    SUSPICIOUS_REGION_MAX_FRAME_GAP
                ):

                    region += 1

                region_numbers.append(
                    region
                )

            previous_frame = frame

        group[
            "region_id"
        ] = region_numbers

        for (
            region_id,
            region_df
        ) in group.groupby(
            "region_id"
        ):

            scores = region_df[
                "score"
            ].to_numpy(
                dtype=float
            )

            rows.append({

                "machine_id":
                    machine,

                "video":
                    video,

                "region_id":
                    int(region_id),

                "start_frame":
                    int(
                        region_df[
                            "frame"
                        ].min()
                    ),

                "end_frame":
                    int(
                        region_df[
                            "frame"
                        ].max()
                    ),

                "samples":
                    len(region_df),

                "median_score":
                    float(
                        np.median(scores)
                    ),

                "mean_score":
                    float(
                        np.mean(scores)
                    ),

                "min_score":
                    float(
                        np.min(scores)
                    ),

                "max_score":
                    float(
                        np.max(scores)
                    ),
            })

    regions_df = pd.DataFrame(
        rows
    )

    regions_df.to_csv(
        DIRS["audit"]
        / "suspicious_regions.csv",
        index=False
    )

    print()

    if not regions_df.empty:

        print(
            regions_df.to_string(
                index=False
            )
        )

    return regions_df


# ============================================================
# 12. CALCULATE PER-VIDEO BASELINES
# ============================================================

def calculate_per_video_baselines(
    raw_empty_df,
    clean_empty_df
):

    print()
    print("=" * 80)
    print("STEP 6: PER-VIDEO BASELINE ANALYSIS")
    print("=" * 80)

    rows = []

    combinations = (
        raw_empty_df[
            [
                "machine_id",
                "video",
            ]
        ]
        .drop_duplicates()
    )

    for combination in combinations.itertuples():

        machine = (
            combination.machine_id
        )

        video = (
            combination.video
        )

        raw_subset = raw_empty_df[
            (
                raw_empty_df[
                    "machine_id"
                ]
                == machine
            )
            &
            (
                raw_empty_df[
                    "video"
                ]
                == video
            )
        ]

        clean_subset = clean_empty_df[
            (
                clean_empty_df[
                    "machine_id"
                ]
                == machine
            )
            &
            (
                clean_empty_df[
                    "video"
                ]
                == video
            )
        ]

        raw_scores = raw_subset[
            "score"
        ].to_numpy(
            dtype=float
        )

        clean_scores = clean_subset[
            "score"
        ].to_numpy(
            dtype=float
        )

        raw_median = float(
            np.median(
                raw_scores
            )
        )

        if len(clean_scores) > 0:

            clean_median = float(
                np.median(
                    clean_scores
                )
            )

            clean_mean = float(
                np.mean(
                    clean_scores
                )
            )

            clean_std = float(
                np.std(
                    clean_scores
                )
            )

        else:

            clean_median = np.nan
            clean_mean = np.nan
            clean_std = np.nan

        rows.append({

            "machine_id":
                machine,

            "video":
                video,

            "raw_empty_samples":
                len(raw_scores),

            "clean_empty_samples":
                len(clean_scores),

            "removed_suspicious":
                (
                    len(raw_scores)
                    -
                    len(clean_scores)
                ),

            "raw_median":
                raw_median,

            "clean_median":
                clean_median,

            "median_change_after_cleaning":
                (
                    clean_median
                    -
                    raw_median
                    if not np.isnan(
                        clean_median
                    )
                    else np.nan
                ),

            "clean_mean":
                clean_mean,

            "clean_std":
                clean_std,
        })

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        DIRS["per_video"]
        / "baseline_per_video.csv",
        index=False
    )

    return result


# ============================================================
# 13. QUALITY ASSESSMENT
# ============================================================

def determine_quality(
    clean_samples,
    empty_videos
):

    if clean_samples == 0:

        return (
            "UNAVAILABLE",
            "No clean EMPTY samples available."
        )

    if clean_samples < MIN_BASELINE_SAMPLES:

        return (
            "LOW",
            (
                f"Only {clean_samples} clean "
                "EMPTY samples available."
            )
        )

    if (
        clean_samples
        < RECOMMENDED_BASELINE_SAMPLES
    ):

        return (
            "ACCEPTABLE",
            (
                "Baseline available, but more "
                "EMPTY observations are recommended."
            )
        )

    if (
        empty_videos
        < RECOMMENDED_EMPTY_VIDEOS
    ):

        return (
            "ACCEPTABLE",
            (
                "Enough EMPTY observations but "
                "limited cross-video coverage."
            )
        )

    return (
        "GOOD",
        (
            "Sufficient clean EMPTY observations "
            "and cross-video coverage."
        )
    )


# ============================================================
# 14. FINAL BASELINES
# ============================================================

def calculate_final_baselines(
    raw_empty_df,
    clean_empty_df,
    suspicious_df,
    per_video_df
):

    print()
    print("=" * 80)
    print("STEP 7: CALCULATE FINAL CLEAN BASELINES")
    print("=" * 80)

    rows = []

    for machine in EXPECTED_MACHINES:

        raw = raw_empty_df[
            raw_empty_df[
                "machine_id"
            ]
            == machine
        ]

        clean = clean_empty_df[
            clean_empty_df[
                "machine_id"
            ]
            == machine
        ]

        suspicious = suspicious_df[
            suspicious_df[
                "machine_id"
            ]
            == machine
        ]

        raw_scores = raw[
            "score"
        ].to_numpy(
            dtype=float
        )

        clean_scores = clean[
            "score"
        ].to_numpy(
            dtype=float
        )

        # ----------------------------------------------------
        # No baseline
        # ----------------------------------------------------

        if len(raw_scores) == 0:

            rows.append({

                "machine_id":
                    machine,

                "baseline_available":
                    False,

                "raw_empty_samples":
                    0,

                "clean_empty_samples":
                    0,

                "suspicious_samples":
                    0,

                "suspicious_percentage":
                    np.nan,

                "empty_videos":
                    0,

                "raw_baseline_median":
                    np.nan,

                "clean_baseline_median":
                    np.nan,

                "baseline_change":
                    np.nan,

                "baseline_change_abs":
                    np.nan,

                "clean_mean":
                    np.nan,

                "clean_std":
                    np.nan,

                "clean_min":
                    np.nan,

                "clean_max":
                    np.nan,

                "clean_range":
                    np.nan,

                "clean_q1":
                    np.nan,

                "clean_q3":
                    np.nan,

                "clean_iqr":
                    np.nan,

                "video_median_std":
                    np.nan,

                "video_median_range":
                    np.nan,

                "quality":
                    "UNAVAILABLE",

                "quality_note":
                    "No EMPTY observations.",
            })

            continue

        # ----------------------------------------------------
        # Raw baseline
        # ----------------------------------------------------

        raw_median = float(
            np.median(
                raw_scores
            )
        )

        # ----------------------------------------------------
        # Clean baseline
        # ----------------------------------------------------

        if len(clean_scores) > 0:

            clean_median = float(
                np.median(
                    clean_scores
                )
            )

            clean_mean = float(
                np.mean(
                    clean_scores
                )
            )

            clean_std = float(
                np.std(
                    clean_scores
                )
            )

            clean_min = float(
                np.min(
                    clean_scores
                )
            )

            clean_max = float(
                np.max(
                    clean_scores
                )
            )

            clean_range = (
                clean_max
                -
                clean_min
            )

            clean_q1 = float(
                np.percentile(
                    clean_scores,
                    25
                )
            )

            clean_q3 = float(
                np.percentile(
                    clean_scores,
                    75
                )
            )

            clean_iqr = (
                clean_q3
                -
                clean_q1
            )

        else:

            clean_median = np.nan
            clean_mean = np.nan
            clean_std = np.nan
            clean_min = np.nan
            clean_max = np.nan
            clean_range = np.nan
            clean_q1 = np.nan
            clean_q3 = np.nan
            clean_iqr = np.nan

        # ----------------------------------------------------
        # Cross-video clean medians
        # ----------------------------------------------------

        video_stats = per_video_df[
            per_video_df[
                "machine_id"
            ]
            == machine
        ]

        video_medians = (
            video_stats[
                "clean_median"
            ]
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        if len(video_medians) > 0:

            video_median_std = float(
                np.std(
                    video_medians
                )
            )

            video_median_range = float(
                np.max(
                    video_medians
                )
                -
                np.min(
                    video_medians
                )
            )

        else:

            video_median_std = np.nan
            video_median_range = np.nan

        empty_videos = (
            clean[
                "video"
            ].nunique()
        )

        quality, note = (
            determine_quality(
                len(clean_scores),
                empty_videos
            )
        )

        suspicious_percentage = (
            len(suspicious)
            /
            len(raw)
            * 100
        )

        baseline_change = (
            clean_median
            -
            raw_median
            if not np.isnan(
                clean_median
            )
            else np.nan
        )

        rows.append({

            "machine_id":
                machine,

            "baseline_available":
                len(clean_scores) > 0,

            "raw_empty_samples":
                len(raw_scores),

            "clean_empty_samples":
                len(clean_scores),

            "suspicious_samples":
                len(suspicious),

            "suspicious_percentage":
                suspicious_percentage,

            "empty_videos":
                empty_videos,

            "raw_baseline_median":
                raw_median,

            "clean_baseline_median":
                clean_median,

            "baseline_change":
                baseline_change,

            "baseline_change_abs":
                (
                    abs(
                        baseline_change
                    )
                    if not np.isnan(
                        baseline_change
                    )
                    else np.nan
                ),

            "clean_mean":
                clean_mean,

            "clean_std":
                clean_std,

            "clean_min":
                clean_min,

            "clean_max":
                clean_max,

            "clean_range":
                clean_range,

            "clean_q1":
                clean_q1,

            "clean_q3":
                clean_q3,

            "clean_iqr":
                clean_iqr,

            "video_median_std":
                video_median_std,

            "video_median_range":
                video_median_range,

            "quality":
                quality,

            "quality_note":
                note,
        })

    baseline_df = pd.DataFrame(
        rows
    )

    baseline_df.to_csv(
        DIRS["final"]
        / "machine_baselines.csv",
        index=False
    )

    return baseline_df


# ============================================================
# 15. CREATE BASELINE COVERAGE
# ============================================================

def create_baseline_coverage(
    baseline_df
):

    columns = [
        "machine_id",
        "raw_empty_samples",
        "clean_empty_samples",
        "suspicious_samples",
        "suspicious_percentage",
        "empty_videos",
        "baseline_available",
        "quality",
        "quality_note",
    ]

    coverage = baseline_df[
        columns
    ].copy()

    coverage.to_csv(
        DIRS["audit"]
        / "baseline_coverage.csv",
        index=False
    )

    return coverage


# ============================================================
# 16. GRAPH:
# RAW / CLEAN / SUSPICIOUS
# ============================================================

def plot_machine_samples(
    machine,
    processed_empty_df,
    baseline_df
):

    subset = processed_empty_df[
        processed_empty_df[
            "machine_id"
        ]
        == machine
    ].copy()

    if subset.empty:
        return

    baseline_row = baseline_df[
        baseline_df[
            "machine_id"
        ]
        == machine
    ].iloc[0]

    raw_baseline = (
        baseline_row[
            "raw_baseline_median"
        ]
    )

    clean_baseline = (
        baseline_row[
            "clean_baseline_median"
        ]
    )

    plt.figure(
        figsize=(14, 7)
    )

    clean = subset[
        subset[
            "is_suspicious"
        ]
        == False
    ]

    suspicious = subset[
        subset[
            "is_suspicious"
        ]
        == True
    ]

    plt.scatter(
        clean["frame"],
        clean["score"],
        label="Clean EMPTY",
        alpha=0.75
    )

    if not suspicious.empty:

        plt.scatter(
            suspicious["frame"],
            suspicious["score"],
            marker="x",
            s=90,
            label="Suspicious EMPTY"
        )

    plt.axhline(
        raw_baseline,
        linestyle=":",
        linewidth=2,
        label=(
            f"Raw baseline "
            f"{raw_baseline:.6f}"
        )
    )

    if not pd.isna(
        clean_baseline
    ):

        plt.axhline(
            clean_baseline,
            linestyle="--",
            linewidth=2,
            label=(
                f"Clean baseline "
                f"{clean_baseline:.6f}"
            )
        )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Edge Similarity Score"
    )

    plt.title(
        f"{machine} - EMPTY Baseline Outlier Analysis"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        DIRS["graphs"]
        / f"{machine}_baseline_cleaning.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 17. GRAPH:
# RAW VS CLEAN BASELINE
# ============================================================

def plot_raw_vs_clean_baselines(
    baseline_df
):

    available = baseline_df[
        baseline_df[
            "baseline_available"
        ]
        == True
    ].copy()

    if available.empty:
        return

    x = np.arange(
        len(available)
    )

    width = 0.35

    plt.figure(
        figsize=(13, 7)
    )

    plt.bar(
        x - width / 2,
        available[
            "raw_baseline_median"
        ],
        width,
        label="Raw baseline"
    )

    plt.bar(
        x + width / 2,
        available[
            "clean_baseline_median"
        ],
        width,
        label="Clean baseline"
    )

    plt.xticks(
        x,
        available[
            "machine_id"
        ]
    )

    plt.xlabel(
        "Machine"
    )

    plt.ylabel(
        "Edge Similarity"
    )

    plt.title(
        "Raw vs Clean EMPTY Baseline"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        DIRS["graphs"]
        / "raw_vs_clean_baselines.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 18. GRAPH:
# SUSPICIOUS SAMPLE COUNT
# ============================================================

def plot_suspicious_counts(
    baseline_df
):

    x = np.arange(
        len(baseline_df)
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        x,
        baseline_df[
            "suspicious_samples"
        ]
    )

    plt.xticks(
        x,
        baseline_df[
            "machine_id"
        ]
    )

    plt.xlabel(
        "Machine"
    )

    plt.ylabel(
        "Suspicious EMPTY Samples"
    )

    plt.title(
        "Suspicious EMPTY Samples by Machine"
    )

    plt.tight_layout()

    plt.savefig(
        DIRS["graphs"]
        / "suspicious_sample_count.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 19. GRAPH:
# SUSPICIOUS PERCENTAGE
# ============================================================

def plot_suspicious_percentage(
    baseline_df
):

    x = np.arange(
        len(baseline_df)
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        x,
        baseline_df[
            "suspicious_percentage"
        ].fillna(0)
    )

    plt.xticks(
        x,
        baseline_df[
            "machine_id"
        ]
    )

    plt.xlabel(
        "Machine"
    )

    plt.ylabel(
        "Suspicious EMPTY (%)"
    )

    plt.title(
        "Percentage of EMPTY Samples Flagged as Suspicious"
    )

    plt.tight_layout()

    plt.savefig(
        DIRS["graphs"]
        / "suspicious_percentage.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 20. GRAPH:
# BASELINE CHANGE AFTER CLEANING
# ============================================================

def plot_baseline_change(
    baseline_df
):

    available = baseline_df[
        baseline_df[
            "baseline_available"
        ]
        == True
    ].copy()

    if available.empty:
        return

    x = np.arange(
        len(available)
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        x,
        available[
            "baseline_change_abs"
        ]
    )

    plt.xticks(
        x,
        available[
            "machine_id"
        ]
    )

    plt.xlabel(
        "Machine"
    )

    plt.ylabel(
        "Absolute Baseline Change"
    )

    plt.title(
        "Effect of Suspicious Sample Removal on Baseline"
    )

    plt.tight_layout()

    plt.savefig(
        DIRS["graphs"]
        / "baseline_change_after_cleaning.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 21. GRAPH:
# PER VIDEO CLEAN BASELINE
# ============================================================

def plot_per_video_baseline(
    machine,
    per_video_df,
    baseline_df
):

    subset = per_video_df[
        per_video_df[
            "machine_id"
        ]
        == machine
    ].copy()

    if subset.empty:
        return

    baseline_row = baseline_df[
        baseline_df[
            "machine_id"
        ]
        == machine
    ].iloc[0]

    clean_baseline = (
        baseline_row[
            "clean_baseline_median"
        ]
    )

    labels = [
        Path(video).stem
        for video
        in subset[
            "video"
        ]
    ]

    x = np.arange(
        len(subset)
    )

    plt.figure(
        figsize=(13, 6)
    )

    plt.bar(
        x,
        subset[
            "clean_median"
        ]
    )

    if not pd.isna(
        clean_baseline
    ):

        plt.axhline(
            clean_baseline,
            linestyle="--",
            label=(
                f"Final clean baseline "
                f"{clean_baseline:.6f}"
            )
        )

    plt.xticks(
        x,
        labels,
        rotation=30,
        ha="right"
    )

    plt.ylabel(
        "Clean EMPTY Median"
    )

    plt.title(
        f"{machine} - Clean EMPTY Baseline by Video"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        DIRS["graphs"]
        / f"{machine}_per_video_baseline.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 22. GRAPH:
# COMPLETE LABEL TIMELINE
#
# This is useful for finding possible manual ground-truth
# mistakes.
# ============================================================

def plot_label_timeline(
    df
):

    print()
    print("=" * 80)
    print("STEP 8: GENERATE LABEL TIMELINE GRAPHS")
    print("=" * 80)

    for (
        machine,
        video
    ), subset in df.groupby(
        [
            "machine_id",
            "video",
        ]
    ):

        subset = subset.sort_values(
            "frame"
        )

        empty = subset[
            subset["label"]
            == "EMPTY"
        ]

        cup = subset[
            subset["label"]
            == "CUP"
        ]

        plt.figure(
            figsize=(14, 6)
        )

        if not empty.empty:

            plt.scatter(
                empty["frame"],
                empty["score"],
                label="Label = EMPTY",
                alpha=0.75
            )

        if not cup.empty:

            plt.scatter(
                cup["frame"],
                cup["score"],
                label="Label = CUP",
                marker="x",
                s=80
            )

        plt.xlabel(
            "Frame"
        )

        plt.ylabel(
            "Edge Similarity Score"
        )

        plt.title(
            f"{machine} | {Path(video).stem}"
        )

        plt.legend()

        plt.tight_layout()

        filename = (
            f"{machine}_"
            f"{Path(video).stem}_timeline.png"
        )

        plt.savefig(
            DIRS["suspicious_graphs"]
            / filename,
            dpi=150
        )

        plt.close()


# ============================================================
# 23. SAVE FINAL JSON
# ============================================================

def save_baseline_config(
    baseline_df
):

    output = {

        "experiment":
            "04_baseline_rnd",

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
                1.0,

            "canny_low":
                75,

            "canny_high":
                175,

            "template_matching":
                "TM_CCOEFF_NORMED",
        },

        "baseline_method": {

            "source":
                (
                    "Manual ground-truth EMPTY "
                    "observations"
                ),

            "raw_baseline":
                (
                    "Median of all manually-labelled "
                    "EMPTY observations"
                ),

            "outlier_detection":
                "IQR",

            "iqr_multiplier":
                IQR_MULTIPLIER,

            "minimum_samples_for_filtering":
                MIN_SAMPLES_FOR_FILTERING,

            "clean_baseline":
                (
                    "Median after removing "
                    "IQR-flagged suspicious EMPTY "
                    "observations"
                ),

            "important_note":
                (
                    "Suspicious EMPTY observations "
                    "are not automatically classified "
                    "as CUP. They may represent manual "
                    "labelling errors, occlusion, "
                    "lighting variation, reflection, "
                    "ROI variation or other anomalies."
                ),
        },

        "machines": {}
    }

    for row in baseline_df.to_dict(
        orient="records"
    ):

        machine = row[
            "machine_id"
        ]

        data = {}

        for key, value in row.items():

            if key == "machine_id":
                continue

            if pd.isna(value):

                data[key] = None

            elif isinstance(
                value,
                (
                    np.bool_,
                    bool
                )
            ):

                data[key] = bool(
                    value
                )

            elif isinstance(
                value,
                (
                    np.integer,
                    int
                )
            ):

                data[key] = int(
                    value
                )

            elif isinstance(
                value,
                (
                    np.floating,
                    float
                )
            ):

                data[key] = float(
                    value
                )

            else:

                data[key] = value

        output[
            "machines"
        ][machine] = data

    with open(
        DIRS["final"]
        / "baseline_config.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )


# ============================================================
# 24. SAVE TEXT SUMMARY
# ============================================================

def save_summary(
    baseline_df,
    regions_df
):

    lines = []

    lines.append(
        "04 - BASELINE R&D"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")

    lines.append(
        "Selected matching configuration:"
    )

    lines.append(
        SELECTED_CONFIGURATION
    )

    lines.append("")

    lines.append(
        "Baseline methodology:"
    )

    lines.append(
        "1. Select manually-labelled EMPTY observations."
    )

    lines.append(
        "2. Calculate raw EMPTY baseline."
    )

    lines.append(
        "3. Detect statistically suspicious EMPTY "
        "observations using IQR."
    )

    lines.append(
        "4. Remove suspicious observations from the "
        "clean baseline calculation."
    )

    lines.append(
        "5. Calculate final clean median baseline."
    )

    lines.append(
        "6. Compare raw and clean baseline."
    )

    lines.append("")

    lines.append(
        "IMPORTANT:"
    )

    lines.append(
        "A suspicious EMPTY observation is NOT "
        "automatically considered CUP."
    )

    lines.append("")

    lines.append(
        "=" * 80
    )

    lines.append(
        "MACHINE RESULTS"
    )

    lines.append(
        "=" * 80
    )

    for row in baseline_df.itertuples():

        lines.append("")

        lines.append(
            f"Machine: {row.machine_id}"
        )

        lines.append(
            f"Raw EMPTY samples: "
            f"{row.raw_empty_samples}"
        )

        lines.append(
            f"Clean EMPTY samples: "
            f"{row.clean_empty_samples}"
        )

        lines.append(
            f"Suspicious samples: "
            f"{row.suspicious_samples}"
        )

        if not pd.isna(
            row.suspicious_percentage
        ):

            lines.append(
                f"Suspicious percentage: "
                f"{row.suspicious_percentage:.2f}%"
            )

        if row.baseline_available:

            lines.append(
                f"Raw baseline median: "
                f"{row.raw_baseline_median:.8f}"
            )

            lines.append(
                f"Clean baseline median: "
                f"{row.clean_baseline_median:.8f}"
            )

            lines.append(
                f"Baseline change: "
                f"{row.baseline_change:.8f}"
            )

            lines.append(
                f"Clean standard deviation: "
                f"{row.clean_std:.8f}"
            )

            lines.append(
                f"Cross-video median range: "
                f"{row.video_median_range:.8f}"
            )

        lines.append(
            f"Quality: {row.quality}"
        )

        lines.append(
            f"Quality note: "
            f"{row.quality_note}"
        )

    lines.append("")

    lines.append(
        "=" * 80
    )

    lines.append(
        "SUSPICIOUS REGIONS"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")

    lines.append(
        f"Total suspicious regions: "
        f"{len(regions_df)}"
    )

    for row in regions_df.itertuples():

        lines.append("")

        lines.append(
            f"{row.machine_id} | "
            f"{row.video} | "
            f"Frames "
            f"{row.start_frame}-{row.end_frame} | "
            f"Samples {row.samples}"
        )

    with open(
        OUTPUT_DIR
        / "baseline_rnd_summary.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(
                lines
            )
        )


# ============================================================
# 25. PRINT FINAL RESULTS
# ============================================================

def print_final_results(
    baseline_df,
    regions_df
):

    print()
    print("=" * 100)
    print("FINAL BASELINE R&D RESULTS")
    print("=" * 100)

    columns = [
        "machine_id",
        "raw_empty_samples",
        "clean_empty_samples",
        "suspicious_samples",
        "raw_baseline_median",
        "clean_baseline_median",
        "baseline_change_abs",
        "quality",
    ]

    print()

    print(
        baseline_df[
            columns
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "=" * 100
    )

    print(
        "SUSPICIOUS REGIONS FOUND:"
    )

    print(
        len(
            regions_df
        )
    )

    print(
        "=" * 100
    )

    if not regions_df.empty:

        print()

        print(
            regions_df.to_string(
                index=False
            )
        )


# ============================================================
# 26. MAIN
# ============================================================

def main():

    create_directories()

    print()
    print("=" * 80)
    print("04 - ROBUST BASELINE RESEARCH AND DEVELOPMENT")
    print("=" * 80)

    print()
    print(
        "Edge configuration:"
    )

    print(
        SELECTED_CONFIGURATION
    )

    print()
    print(
        "Baseline:"
    )

    print(
        "Manual EMPTY labels + IQR suspicious "
        "sample analysis"
    )

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    df = load_edge_data()

    # --------------------------------------------------------
    # 2. Audit
    # --------------------------------------------------------

    create_ground_truth_audit(
        df
    )

    # --------------------------------------------------------
    # 3. Raw EMPTY
    # --------------------------------------------------------

    raw_empty_df, cup_df = (
        extract_raw_samples(
            df
        )
    )

    # --------------------------------------------------------
    # 4. Suspicious detection
    # --------------------------------------------------------

    (
        processed_empty_df,
        clean_empty_df,
        suspicious_df,
        outlier_stats_df
    ) = detect_suspicious_empty_samples(
        raw_empty_df
    )

    # --------------------------------------------------------
    # 5. Suspicious regions
    # --------------------------------------------------------

    regions_df = (
        create_suspicious_regions(
            suspicious_df
        )
    )

    # --------------------------------------------------------
    # 6. Per-video analysis
    # --------------------------------------------------------

    per_video_df = (
        calculate_per_video_baselines(
            raw_empty_df,
            clean_empty_df
        )
    )

    # --------------------------------------------------------
    # 7. Final baseline
    # --------------------------------------------------------

    baseline_df = (
        calculate_final_baselines(
            raw_empty_df,
            clean_empty_df,
            suspicious_df,
            per_video_df
        )
    )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    create_baseline_coverage(
        baseline_df
    )

    # --------------------------------------------------------
    # 8. Graphs
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 8: GENERATE GRAPHS")
    print("=" * 80)

    for machine in EXPECTED_MACHINES:

        plot_machine_samples(
            machine,
            processed_empty_df,
            baseline_df
        )

        plot_per_video_baseline(
            machine,
            per_video_df,
            baseline_df
        )

    plot_raw_vs_clean_baselines(
        baseline_df
    )

    plot_suspicious_counts(
        baseline_df
    )

    plot_suspicious_percentage(
        baseline_df
    )

    plot_baseline_change(
        baseline_df
    )

    # Full EMPTY/CUP timeline
    plot_label_timeline(
        df
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    save_baseline_config(
        baseline_df
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    save_summary(
        baseline_df,
        regions_df
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print_final_results(
        baseline_df,
        regions_df
    )

    print()
    print("=" * 80)
    print("04 BASELINE R&D COMPLETE")
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
        "IMPORTANT FILES"
    )

    print("-" * 80)

    important_files = [

        DIRS["samples"]
        / "raw_empty_samples.csv",

        DIRS["samples"]
        / "clean_empty_samples.csv",

        DIRS["samples"]
        / "suspicious_empty_samples.csv",

        DIRS["audit"]
        / "outlier_detection_summary.csv",

        DIRS["audit"]
        / "suspicious_regions.csv",

        DIRS["per_video"]
        / "baseline_per_video.csv",

        DIRS["final"]
        / "machine_baselines.csv",

        DIRS["final"]
        / "baseline_config.json",

        OUTPUT_DIR
        / "baseline_rnd_summary.txt",
    ]

    for file in important_files:

        print(
            file
        )


if __name__ == "__main__":

    main()
