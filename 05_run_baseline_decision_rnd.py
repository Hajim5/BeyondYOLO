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

BASELINE_CSV = (
    BASE_DIR
    / "results"
    / "04_baseline_rnd"
    / "03_final_baseline"
    / "machine_baselines.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "05_baseline_decision_rnd"
)

# ============================================================
# 2. LOCKED MATCHING CONFIGURATION
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
# 5. SETTINGS
# ============================================================

# Minimum number of CUP and EMPTY samples preferred before
# treating a learned decision threshold as reasonably supported.
MIN_CLASS_SAMPLES = 3

# Number of candidate thresholds tested between observed
# delta values.
THRESHOLD_GRID_POINTS = 1000

# ============================================================
# 6. OUTPUT DIRECTORIES
# ============================================================

DIRS = {

    "audit":
        OUTPUT_DIR / "00_audit",

    "delta":
        OUTPUT_DIR / "01_delta_analysis",

    "threshold":
        OUTPUT_DIR / "02_threshold_search",

    "evaluation":
        OUTPUT_DIR / "03_evaluation",

    "final":
        OUTPUT_DIR / "04_final_decision",

    "graphs":
        OUTPUT_DIR / "05_graphs",

    "machine_graphs":
        OUTPUT_DIR
        / "05_graphs"
        / "machines",
}


def create_directories():

    for directory in DIRS.values():

        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# 7. LOAD EDGE RESULTS
# ============================================================

def load_edge_data():

    print()
    print("=" * 90)
    print("STEP 1: LOAD EDGE MATCHING RESULTS")
    print("=" * 90)

    if not EDGE_CSV.exists():

        raise FileNotFoundError(
            f"\nEdge CSV not found:\n{EDGE_CSV}"
        )

    df = pd.read_csv(
        EDGE_CSV
    )

    required = {
        "video",
        "machine_id",
        "frame",
        "configuration",
        "score",
        "label",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"\nMissing columns:\n{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Clean strings
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
    # Numeric conversion
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
    # Selected matching configuration
    # --------------------------------------------------------

    df = df[
        df["configuration"]
        == SELECTED_CONFIGURATION
    ].copy()

    # --------------------------------------------------------
    # Selected videos
    # --------------------------------------------------------

    df = df[
        df["video"].isin(
            VIDEOS
        )
    ].copy()

    # --------------------------------------------------------
    # Valid ground truth
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

    if df.empty:

        raise ValueError(
            "\nNo usable Edge matching rows found."
        )

    print()
    print(
        f"Usable observations: {len(df)}"
    )

    print()
    print(
        df["label"].value_counts()
    )

    return df


# ============================================================
# 8. LOAD BASELINES FROM 04
# ============================================================

def load_baselines():

    print()
    print("=" * 90)
    print("STEP 2: LOAD CLEAN BASELINES FROM 04")
    print("=" * 90)

    if not BASELINE_CSV.exists():

        raise FileNotFoundError(
            "\nBaseline CSV not found:\n"
            f"{BASELINE_CSV}"
        )

    baseline_df = pd.read_csv(
        BASELINE_CSV
    )

    required = {
        "machine_id",
        "baseline_available",
        "clean_baseline_median",
        "quality",
    }

    missing = (
        required
        - set(baseline_df.columns)
    )

    if missing:

        raise ValueError(
            "\nMissing baseline columns:\n"
            f"{sorted(missing)}"
        )

    baseline_df["machine_id"] = (
        baseline_df["machine_id"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    baseline_df[
        "clean_baseline_median"
    ] = pd.to_numeric(
        baseline_df[
            "clean_baseline_median"
        ],
        errors="coerce"
    )

    print()

    print(
        baseline_df[
            [
                "machine_id",
                "clean_baseline_median",
                "quality",
            ]
        ].to_string(
            index=False
        )
    )

    return baseline_df


# ============================================================
# 9. ADD MACHINE BASELINE + DELTA
#
# delta = current score - EMPTY baseline
#
# Positive:
# current score is above EMPTY baseline
#
# Negative:
# current score is below EMPTY baseline
# ============================================================

def calculate_delta(
    df,
    baseline_df
):

    print()
    print("=" * 90)
    print("STEP 3: CALCULATE BASELINE-RELATIVE DELTA")
    print("=" * 90)

    baseline_lookup = (
        baseline_df[
            [
                "machine_id",
                "clean_baseline_median",
                "quality",
            ]
        ]
        .rename(
            columns={
                "clean_baseline_median":
                    "empty_baseline",

                "quality":
                    "baseline_quality",
            }
        )
    )

    result = df.merge(
        baseline_lookup,
        on="machine_id",
        how="left"
    )

    result["delta"] = (
        result["score"]
        -
        result["empty_baseline"]
    )

    result["absolute_delta"] = (
        result["delta"].abs()
    )

    result = result.dropna(
        subset=[
            "empty_baseline",
            "delta",
        ]
    )

    result.to_csv(
        DIRS["delta"]
        / "all_samples_with_delta.csv",
        index=False
    )

    print()
    print(
        f"Samples with valid baseline: "
        f"{len(result)}"
    )

    return result


# ============================================================
# 10. DELTA DISTRIBUTION ANALYSIS
# ============================================================

def analyze_delta_distribution(
    df
):

    print()
    print("=" * 90)
    print("STEP 4: EMPTY VS CUP DELTA ANALYSIS")
    print("=" * 90)

    rows = []

    for machine in EXPECTED_MACHINES:

        subset = df[
            df["machine_id"]
            == machine
        ]

        empty = subset[
            subset["label"]
            == "EMPTY"
        ]

        cup = subset[
            subset["label"]
            == "CUP"
        ]

        empty_delta = (
            empty["delta"]
            .to_numpy(
                dtype=float
            )
        )

        cup_delta = (
            cup["delta"]
            .to_numpy(
                dtype=float
            )
        )

        if len(empty_delta) > 0:

            empty_median = float(
                np.median(
                    empty_delta
                )
            )

            empty_mean = float(
                np.mean(
                    empty_delta
                )
            )

            empty_std = float(
                np.std(
                    empty_delta
                )
            )

            empty_min = float(
                np.min(
                    empty_delta
                )
            )

            empty_max = float(
                np.max(
                    empty_delta
                )
            )

        else:

            empty_median = np.nan
            empty_mean = np.nan
            empty_std = np.nan
            empty_min = np.nan
            empty_max = np.nan

        if len(cup_delta) > 0:

            cup_median = float(
                np.median(
                    cup_delta
                )
            )

            cup_mean = float(
                np.mean(
                    cup_delta
                )
            )

            cup_std = float(
                np.std(
                    cup_delta
                )
            )

            cup_min = float(
                np.min(
                    cup_delta
                )
            )

            cup_max = float(
                np.max(
                    cup_delta
                )
            )

        else:

            cup_median = np.nan
            cup_mean = np.nan
            cup_std = np.nan
            cup_min = np.nan
            cup_max = np.nan

        # ----------------------------------------------------
        # Determine expected CUP direction
        # ----------------------------------------------------

        if (
            len(empty_delta) > 0
            and
            len(cup_delta) > 0
        ):

            median_difference = (
                cup_median
                -
                empty_median
            )

            if median_difference > 0:

                cup_direction = "ABOVE"

            elif median_difference < 0:

                cup_direction = "BELOW"

            else:

                cup_direction = "NONE"

        else:

            median_difference = np.nan
            cup_direction = "UNAVAILABLE"

        # ----------------------------------------------------
        # Simple median separation
        # ----------------------------------------------------

        if (
            not np.isnan(
                median_difference
            )
        ):

            median_separation = abs(
                median_difference
            )

        else:

            median_separation = np.nan

        rows.append({

            "machine_id":
                machine,

            "empty_samples":
                len(empty_delta),

            "cup_samples":
                len(cup_delta),

            "empty_delta_median":
                empty_median,

            "empty_delta_mean":
                empty_mean,

            "empty_delta_std":
                empty_std,

            "empty_delta_min":
                empty_min,

            "empty_delta_max":
                empty_max,

            "cup_delta_median":
                cup_median,

            "cup_delta_mean":
                cup_mean,

            "cup_delta_std":
                cup_std,

            "cup_delta_min":
                cup_min,

            "cup_delta_max":
                cup_max,

            "cup_minus_empty_median":
                median_difference,

            "median_separation":
                median_separation,

            "cup_direction":
                cup_direction,
        })

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        DIRS["delta"]
        / "delta_distribution_summary.csv",
        index=False
    )

    print()
    print(
        result[
            [
                "machine_id",
                "empty_samples",
                "cup_samples",
                "empty_delta_median",
                "cup_delta_median",
                "median_separation",
                "cup_direction",
            ]
        ].to_string(
            index=False
        )
    )

    return result


# ============================================================
# 11. CLASSIFICATION METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    tp = int(
        np.sum(
            (y_true == 1)
            &
            (y_pred == 1)
        )
    )

    tn = int(
        np.sum(
            (y_true == 0)
            &
            (y_pred == 0)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0)
            &
            (y_pred == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1)
            &
            (y_pred == 0)
        )
    )

    total = (
        tp + tn + fp + fn
    )

    accuracy = (
        (tp + tn)
        / total
        if total
        else 0
    )

    precision = (
        tp
        / (tp + fp)
        if (tp + fp)
        else 0
    )

    recall = (
        tp
        / (tp + fn)
        if (tp + fn)
        else 0
    )

    specificity = (
        tn
        / (tn + fp)
        if (tn + fp)
        else 0
    )

    if (
        precision + recall
    ):

        f1 = (
            2
            * precision
            * recall
            /
            (
                precision
                +
                recall
            )
        )

    else:

        f1 = 0

    balanced_accuracy = (
        (
            recall
            +
            specificity
        )
        / 2
    )

    return {

        "tp":
            tp,

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "specificity":
            specificity,

        "f1":
            f1,

        "balanced_accuracy":
            balanced_accuracy,
    }


# ============================================================
# 12. FIND BEST THRESHOLD
#
# We test BOTH directions:
#
# ABOVE:
# CUP if delta >= threshold
#
# BELOW:
# CUP if delta <= threshold
#
# This avoids assuming every machine behaves the same way.
# ============================================================

def find_best_thresholds(
    df
):

    print()
    print("=" * 90)
    print("STEP 5: SEARCH BEST BASELINE MARGIN")
    print("=" * 90)

    best_rows = []

    all_search_rows = []

    for machine in EXPECTED_MACHINES:

        subset = df[
            df["machine_id"]
            == machine
        ].copy()

        if subset.empty:
            continue

        if (
            subset["label"].nunique()
            < 2
        ):

            print()
            print(
                f"{machine}: skipped - "
                "both EMPTY and CUP are required."
            )

            continue

        delta = subset[
            "delta"
        ].to_numpy(
            dtype=float
        )

        y_true = (
            subset["label"]
            == "CUP"
        ).astype(
            int
        ).to_numpy()

        delta_min = float(
            np.min(
                delta
            )
        )

        delta_max = float(
            np.max(
                delta
            )
        )

        # ----------------------------------------------------
        # Candidate thresholds
        # ----------------------------------------------------

        if (
            delta_min
            ==
            delta_max
        ):

            thresholds = np.array(
                [
                    delta_min
                ]
            )

        else:

            thresholds = np.linspace(
                delta_min,
                delta_max,
                THRESHOLD_GRID_POINTS
            )

        machine_candidates = []

        # ----------------------------------------------------
        # Test both possible directions
        # ----------------------------------------------------

        for direction in [
            "ABOVE",
            "BELOW",
        ]:

            for threshold in thresholds:

                if direction == "ABOVE":

                    y_pred = (
                        delta
                        >= threshold
                    ).astype(
                        int
                    )

                else:

                    y_pred = (
                        delta
                        <= threshold
                    ).astype(
                        int
                    )

                metrics = (
                    calculate_metrics(
                        y_true,
                        y_pred
                    )
                )

                row = {

                    "machine_id":
                        machine,

                    "direction":
                        direction,

                    "threshold":
                        float(
                            threshold
                        ),

                    **metrics
                }

                machine_candidates.append(
                    row
                )

                all_search_rows.append(
                    row
                )

        candidates_df = pd.DataFrame(
            machine_candidates
        )

        # ----------------------------------------------------
        # Ranking
        #
        # Primary:
        # balanced accuracy
        #
        # Secondary:
        # F1
        #
        # Third:
        # accuracy
        # ----------------------------------------------------

        candidates_df = (
            candidates_df
            .sort_values(
                [
                    "balanced_accuracy",
                    "f1",
                    "accuracy",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ]
            )
            .reset_index(
                drop=True
            )
        )

        best = (
            candidates_df.iloc[0]
        )

        empty_count = int(
            np.sum(
                subset["label"]
                == "EMPTY"
            )
        )

        cup_count = int(
            np.sum(
                subset["label"]
                == "CUP"
            )
        )

        support_ok = (
            empty_count
            >= MIN_CLASS_SAMPLES
            and
            cup_count
            >= MIN_CLASS_SAMPLES
        )

        best_rows.append({

            "machine_id":
                machine,

            "empty_samples":
                empty_count,

            "cup_samples":
                cup_count,

            "support_ok":
                support_ok,

            "direction":
                best[
                    "direction"
                ],

            "best_delta_threshold":
                float(
                    best[
                        "threshold"
                    ]
                ),

            "accuracy":
                float(
                    best[
                        "accuracy"
                    ]
                ),

            "balanced_accuracy":
                float(
                    best[
                        "balanced_accuracy"
                    ]
                ),

            "precision":
                float(
                    best[
                        "precision"
                    ]
                ),

            "recall":
                float(
                    best[
                        "recall"
                    ]
                ),

            "specificity":
                float(
                    best[
                        "specificity"
                    ]
                ),

            "f1":
                float(
                    best[
                        "f1"
                    ]
                ),

            "tp":
                int(
                    best[
                        "tp"
                    ]
                ),

            "tn":
                int(
                    best[
                        "tn"
                    ]
                ),

            "fp":
                int(
                    best[
                        "fp"
                    ]
                ),

            "fn":
                int(
                    best[
                        "fn"
                    ]
                ),
        })

        print()
        print(
            f"{machine}"
        )

        print(
            f"  Direction         : "
            f"{best['direction']}"
        )

        print(
            f"  Delta threshold   : "
            f"{best['threshold']:.8f}"
        )

        print(
            f"  Balanced accuracy : "
            f"{best['balanced_accuracy']:.4f}"
        )

        print(
            f"  F1                : "
            f"{best['f1']:.4f}"
        )

    best_df = pd.DataFrame(
        best_rows
    )

    search_df = pd.DataFrame(
        all_search_rows
    )

    best_df.to_csv(
        DIRS["threshold"]
        / "best_threshold_per_machine.csv",
        index=False
    )

    search_df.to_csv(
        DIRS["threshold"]
        / "all_threshold_search.csv",
        index=False
    )

    return (
        best_df,
        search_df
    )


# ============================================================
# 13. APPLY BEST DECISION
# ============================================================

def apply_best_decisions(
    df,
    threshold_df
):

    print()
    print("=" * 90)
    print("STEP 6: APPLY LEARNED DECISION RULES")
    print("=" * 90)

    result_groups = []

    for machine in EXPECTED_MACHINES:

        subset = df[
            df["machine_id"]
            == machine
        ].copy()

        rule = threshold_df[
            threshold_df[
                "machine_id"
            ]
            == machine
        ]

        if (
            subset.empty
            or
            rule.empty
        ):

            continue

        rule = rule.iloc[0]

        direction = (
            rule["direction"]
        )

        threshold = float(
            rule[
                "best_delta_threshold"
            ]
        )

        if direction == "ABOVE":

            subset[
                "predicted_label"
            ] = np.where(
                subset["delta"]
                >= threshold,
                "CUP",
                "EMPTY"
            )

        else:

            subset[
                "predicted_label"
            ] = np.where(
                subset["delta"]
                <= threshold,
                "CUP",
                "EMPTY"
            )

        subset[
            "decision_direction"
        ] = direction

        subset[
            "decision_threshold"
        ] = threshold

        subset[
            "correct"
        ] = (
            subset[
                "predicted_label"
            ]
            ==
            subset[
                "label"
            ]
        )

        result_groups.append(
            subset
        )

    if result_groups:

        result = pd.concat(
            result_groups,
            ignore_index=True
        )

    else:

        result = pd.DataFrame()

    result.to_csv(
        DIRS["evaluation"]
        / "all_predictions.csv",
        index=False
    )

    return result


# ============================================================
# 14. PER-VIDEO EVALUATION
# ============================================================

def evaluate_per_video(
    predictions_df
):

    print()
    print("=" * 90)
    print("STEP 7: PER-VIDEO EVALUATION")
    print("=" * 90)

    rows = []

    if predictions_df.empty:

        result = pd.DataFrame()

        result.to_csv(
            DIRS["evaluation"]
            / "per_video_evaluation.csv",
            index=False
        )

        return result

    for (
        machine,
        video
    ), subset in predictions_df.groupby(
        [
            "machine_id",
            "video",
        ]
    ):

        y_true = (
            subset["label"]
            == "CUP"
        ).astype(
            int
        )

        y_pred = (
            subset[
                "predicted_label"
            ]
            == "CUP"
        ).astype(
            int
        )

        metrics = (
            calculate_metrics(
                y_true,
                y_pred
            )
        )

        rows.append({

            "machine_id":
                machine,

            "video":
                video,

            "samples":
                len(
                    subset
                ),

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

            **metrics
        })

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        DIRS["evaluation"]
        / "per_video_evaluation.csv",
        index=False
    )

    return result


# ============================================================
# 15. DECISION QUALITY
# ============================================================

def assign_decision_quality(
    row
):

    if not row[
        "support_ok"
    ]:

        return (
            "LOW_SUPPORT",
            (
                "Too few EMPTY or CUP samples "
                "for a strongly supported decision."
            )
        )

    balanced = (
        row[
            "balanced_accuracy"
        ]
    )

    if balanced >= 0.90:

        return (
            "STRONG",
            "Strong EMPTY/CUP separation."
        )

    elif balanced >= 0.80:

        return (
            "GOOD",
            "Good EMPTY/CUP separation."
        )

    elif balanced >= 0.70:

        return (
            "MODERATE",
            (
                "Usable separation, but errors "
                "remain."
            )
        )

    elif balanced >= 0.60:

        return (
            "WEAK",
            (
                "Weak separation. Additional "
                "decision logic may be required."
            )
        )

    else:

        return (
            "POOR",
            (
                "Baseline-relative Edge score "
                "does not provide reliable "
                "separation."
            )
        )


# ============================================================
# 16. CREATE FINAL DECISION TABLE
# ============================================================

def create_final_decision_table(
    baseline_df,
    delta_summary,
    threshold_df
):

    print()
    print("=" * 90)
    print("STEP 8: CREATE FINAL DECISION TABLE")
    print("=" * 90)

    baseline_columns = baseline_df[
        [
            "machine_id",
            "clean_baseline_median",
            "quality",
        ]
    ].rename(
        columns={
            "clean_baseline_median":
                "empty_baseline",

            "quality":
                "baseline_quality",
        }
    )

    delta_columns = delta_summary[
        [
            "machine_id",
            "empty_delta_median",
            "cup_delta_median",
            "median_separation",
            "cup_direction",
        ]
    ]

    final = threshold_df.merge(
        baseline_columns,
        on="machine_id",
        how="left"
    )

    final = final.merge(
        delta_columns,
        on="machine_id",
        how="left"
    )

    qualities = []

    notes = []

    for _, row in final.iterrows():

        quality, note = (
            assign_decision_quality(
                row
            )
        )

        qualities.append(
            quality
        )

        notes.append(
            note
        )

    final[
        "decision_quality"
    ] = qualities

    final[
        "decision_note"
    ] = notes

    # --------------------------------------------------------
    # Absolute score threshold equivalent
    #
    # delta = score - baseline
    #
    # Therefore:
    #
    # score threshold =
    # baseline + delta threshold
    # --------------------------------------------------------

    final[
        "equivalent_score_threshold"
    ] = (
        final[
            "empty_baseline"
        ]
        +
        final[
            "best_delta_threshold"
        ]
    )

    final.to_csv(
        DIRS["final"]
        / "final_machine_decisions.csv",
        index=False
    )

    print()

    display_columns = [
        "machine_id",
        "empty_baseline",
        "cup_direction",
        "best_delta_threshold",
        "equivalent_score_threshold",
        "balanced_accuracy",
        "f1",
        "decision_quality",
    ]

    print(
        final[
            display_columns
        ].to_string(
            index=False
        )
    )

    return final


# ============================================================
# 17. GRAPH: DELTA DISTRIBUTION
# ============================================================

def plot_delta_distribution(
    machine,
    df,
    final_df
):

    subset = df[
        df["machine_id"]
        == machine
    ]

    rule = final_df[
        final_df["machine_id"]
        == machine
    ]

    if (
        subset.empty
        or
        rule.empty
    ):

        return

    rule = rule.iloc[0]

    empty = subset[
        subset["label"]
        == "EMPTY"
    ]["delta"]

    cup = subset[
        subset["label"]
        == "CUP"
    ]["delta"]

    threshold = float(
        rule[
            "best_delta_threshold"
        ]
    )

    direction = (
        rule[
            "direction"
        ]
    )

    plt.figure(
        figsize=(11, 6)
    )

    if len(empty):

        plt.hist(
            empty,
            bins=min(
                15,
                max(
                    5,
                    len(empty)
                )
            ),
            alpha=0.6,
            label="EMPTY"
        )

    if len(cup):

        plt.hist(
            cup,
            bins=min(
                15,
                max(
                    5,
                    len(cup)
                )
            ),
            alpha=0.6,
            label="CUP"
        )

    plt.axvline(
        0,
        linestyle=":",
        linewidth=2,
        label="EMPTY baseline"
    )

    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=(
            f"Decision threshold "
            f"{threshold:.6f}"
        )
    )

    plt.xlabel(
        "Delta = Edge Score - EMPTY Baseline"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        f"{machine} - EMPTY vs CUP Baseline Delta\n"
        f"CUP direction: {direction}"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        DIRS["machine_graphs"]
        / f"{machine}_delta_distribution.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 18. GRAPH: DELTA TIMELINE
# ============================================================

def plot_delta_timeline(
    machine,
    video,
    df,
    final_df
):

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
    ].sort_values(
        "frame"
    )

    rule = final_df[
        final_df[
            "machine_id"
        ]
        == machine
    ]

    if (
        subset.empty
        or
        rule.empty
    ):

        return

    threshold = float(
        rule.iloc[0][
            "best_delta_threshold"
        ]
    )

    direction = (
        rule.iloc[0][
            "direction"
        ]
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
            empty["delta"],
            label="EMPTY",
            alpha=0.75
        )

    if not cup.empty:

        plt.scatter(
            cup["frame"],
            cup["delta"],
            label="CUP",
            marker="x",
            s=80
        )

    plt.axhline(
        0,
        linestyle=":",
        linewidth=2,
        label="EMPTY baseline"
    )

    plt.axhline(
        threshold,
        linestyle="--",
        linewidth=2,
        label="Decision threshold"
    )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Delta"
    )

    plt.title(
        f"{machine} | {Path(video).stem}\n"
        f"CUP direction = {direction}"
    )

    plt.legend()

    plt.tight_layout()

    filename = (
        f"{machine}_"
        f"{Path(video).stem}_delta_timeline.png"
    )

    plt.savefig(
        DIRS["machine_graphs"]
        / filename,
        dpi=150
    )

    plt.close()


# ============================================================
# 19. GRAPH: BALANCED ACCURACY
# ============================================================

def plot_machine_accuracy(
    final_df
):

    if final_df.empty:
        return

    x = np.arange(
        len(final_df)
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        x,
        final_df[
            "balanced_accuracy"
        ]
    )

    plt.axhline(
        0.90,
        linestyle="--",
        label="Strong = 0.90"
    )

    plt.axhline(
        0.80,
        linestyle=":",
        label="Good = 0.80"
    )

    plt.xticks(
        x,
        final_df[
            "machine_id"
        ]
    )

    plt.ylim(
        0,
        1.05
    )

    plt.ylabel(
        "Balanced Accuracy"
    )

    plt.xlabel(
        "Machine"
    )

    plt.title(
        "Baseline Decision Performance by Machine"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        DIRS["graphs"]
        / "machine_balanced_accuracy.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 20. GRAPH: MEDIAN SEPARATION
# ============================================================

def plot_median_separation(
    final_df
):

    if final_df.empty:
        return

    x = np.arange(
        len(final_df)
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        x,
        final_df[
            "median_separation"
        ]
    )

    plt.xticks(
        x,
        final_df[
            "machine_id"
        ]
    )

    plt.ylabel(
        "|CUP Median Delta - EMPTY Median Delta|"
    )

    plt.xlabel(
        "Machine"
    )

    plt.title(
        "EMPTY vs CUP Delta Separation"
    )

    plt.tight_layout()

    plt.savefig(
        DIRS["graphs"]
        / "machine_delta_separation.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 21. SAVE FINAL JSON
# ============================================================

def save_final_config(
    final_df
):

    output = {

        "experiment":
            "05_baseline_decision_rnd",

        "matching_method": {

            "method":
                "edge_matching",

            "configuration":
                SELECTED_CONFIGURATION,
        },

        "decision_method": {

            "baseline_source":
                (
                    "04_baseline_rnd "
                    "clean_baseline_median"
                ),

            "feature":
                (
                    "delta = current_edge_score "
                    "- empty_baseline"
                ),

            "threshold_selection":
                (
                    "Machine-specific threshold "
                    "optimized using balanced "
                    "accuracy, F1 and accuracy"
                ),

            "direction":
                (
                    "Machine-specific ABOVE or BELOW"
                ),

            "important_note":
                (
                    "These thresholds are R&D results "
                    "estimated using the same labelled "
                    "dataset. They should be validated "
                    "on independent video before being "
                    "treated as final production "
                    "thresholds."
                ),
        },

        "machines": {}
    }

    for row in final_df.to_dict(
        orient="records"
    ):

        machine = (
            row[
                "machine_id"
            ]
        )

        machine_data = {}

        for key, value in row.items():

            if key == "machine_id":
                continue

            if pd.isna(
                value
            ):

                machine_data[
                    key
                ] = None

            elif isinstance(
                value,
                (
                    np.bool_,
                    bool
                )
            ):

                machine_data[
                    key
                ] = bool(
                    value
                )

            elif isinstance(
                value,
                (
                    np.integer,
                    int
                )
            ):

                machine_data[
                    key
                ] = int(
                    value
                )

            elif isinstance(
                value,
                (
                    np.floating,
                    float
                )
            ):

                machine_data[
                    key
                ] = float(
                    value
                )

            else:

                machine_data[
                    key
                ] = value

        output[
            "machines"
        ][machine] = machine_data

    with open(
        DIRS["final"]
        / "baseline_decision_config.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )


# ============================================================
# 22. SAVE SUMMARY
# ============================================================

def save_summary(
    final_df
):

    lines = []

    lines.append(
        "05 - BASELINE DECISION R&D"
    )

    lines.append(
        "=" * 90
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
        "Decision feature:"
    )

    lines.append(
        "delta = Edge similarity score "
        "- machine EMPTY baseline"
    )

    lines.append("")

    lines.append(
        "The CUP direction is learned independently "
        "for every machine."
    )

    lines.append("")

    lines.append(
        "ABOVE:"
    )

    lines.append(
        "CUP if delta >= threshold"
    )

    lines.append("")

    lines.append(
        "BELOW:"
    )

    lines.append(
        "CUP if delta <= threshold"
    )

    lines.append("")

    lines.append(
        "=" * 90
    )

    lines.append(
        "MACHINE RESULTS"
    )

    lines.append(
        "=" * 90
    )

    for row in final_df.itertuples():

        lines.append("")

        lines.append(
            f"Machine: {row.machine_id}"
        )

        lines.append(
            f"EMPTY baseline: "
            f"{row.empty_baseline:.8f}"
        )

        lines.append(
            f"Baseline quality: "
            f"{row.baseline_quality}"
        )

        lines.append(
            f"EMPTY samples: "
            f"{row.empty_samples}"
        )

        lines.append(
            f"CUP samples: "
            f"{row.cup_samples}"
        )

        lines.append(
            f"CUP direction: "
            f"{row.direction}"
        )

        lines.append(
            f"Delta threshold: "
            f"{row.best_delta_threshold:.8f}"
        )

        lines.append(
            f"Equivalent score threshold: "
            f"{row.equivalent_score_threshold:.8f}"
        )

        lines.append(
            f"EMPTY delta median: "
            f"{row.empty_delta_median:.8f}"
        )

        lines.append(
            f"CUP delta median: "
            f"{row.cup_delta_median:.8f}"
        )

        lines.append(
            f"Median separation: "
            f"{row.median_separation:.8f}"
        )

        lines.append(
            f"Accuracy: "
            f"{row.accuracy:.4f}"
        )

        lines.append(
            f"Balanced accuracy: "
            f"{row.balanced_accuracy:.4f}"
        )

        lines.append(
            f"Precision: "
            f"{row.precision:.4f}"
        )

        lines.append(
            f"Recall: "
            f"{row.recall:.4f}"
        )

        lines.append(
            f"Specificity: "
            f"{row.specificity:.4f}"
        )

        lines.append(
            f"F1: "
            f"{row.f1:.4f}"
        )

        lines.append(
            f"Decision quality: "
            f"{row.decision_quality}"
        )

        lines.append(
            f"Decision note: "
            f"{row.decision_note}"
        )

    with open(
        OUTPUT_DIR
        / "baseline_decision_rnd_summary.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(
                lines
            )
        )


# ============================================================
# 23. PRINT FINAL RESULTS
# ============================================================

def print_final_results(
    final_df
):

    print()
    print("=" * 110)
    print("FINAL BASELINE DECISION RESULTS")
    print("=" * 110)

    if final_df.empty:

        print(
            "No valid machine decisions generated."
        )

        return

    columns = [
        "machine_id",
        "empty_baseline",
        "direction",
        "best_delta_threshold",
        "balanced_accuracy",
        "f1",
        "decision_quality",
    ]

    print()

    print(
        final_df[
            columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 110)


# ============================================================
# 24. MAIN
# ============================================================

def main():

    create_directories()

    print()
    print("=" * 90)
    print("05 - BASELINE DECISION RESEARCH AND DEVELOPMENT")
    print("=" * 90)

    print()
    print(
        "Matching configuration:"
    )

    print(
        SELECTED_CONFIGURATION
    )

    print()
    print(
        "Decision variable:"
    )

    print(
        "delta = Edge score - clean EMPTY baseline"
    )

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    df = load_edge_data()

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    baseline_df = (
        load_baselines()
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    delta_df = (
        calculate_delta(
            df,
            baseline_df
        )
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    delta_summary = (
        analyze_delta_distribution(
            delta_df
        )
    )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    (
        threshold_df,
        threshold_search_df
    ) = find_best_thresholds(
        delta_df
    )

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    predictions_df = (
        apply_best_decisions(
            delta_df,
            threshold_df
        )
    )

    # --------------------------------------------------------
    # STEP 7
    # --------------------------------------------------------

    evaluate_per_video(
        predictions_df
    )

    # --------------------------------------------------------
    # STEP 8
    # --------------------------------------------------------

    final_df = (
        create_final_decision_table(
            baseline_df,
            delta_summary,
            threshold_df
        )
    )

    # --------------------------------------------------------
    # STEP 9 - GRAPHS
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("STEP 9: GENERATE GRAPHS")
    print("=" * 90)

    for machine in EXPECTED_MACHINES:

        plot_delta_distribution(
            machine,
            delta_df,
            final_df
        )

    for (
        machine,
        video
    ) in (
        delta_df[
            [
                "machine_id",
                "video",
            ]
        ]
        .drop_duplicates()
        .itertuples(
            index=False,
            name=None
        )
    ):

        plot_delta_timeline(
            machine,
            video,
            delta_df,
            final_df
        )

    plot_machine_accuracy(
        final_df
    )

    plot_median_separation(
        final_df
    )

    # --------------------------------------------------------
    # STEP 10 - CONFIG
    # --------------------------------------------------------

    save_final_config(
        final_df
    )

    # --------------------------------------------------------
    # STEP 11 - SUMMARY
    # --------------------------------------------------------

    save_summary(
        final_df
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print_final_results(
        final_df
    )

    print()
    print("=" * 90)
    print("05 BASELINE DECISION R&D COMPLETE")
    print("=" * 90)

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "IMPORTANT OUTPUTS"
    )

    print("-" * 90)

    important_files = [

        DIRS["delta"]
        / "all_samples_with_delta.csv",

        DIRS["delta"]
        / "delta_distribution_summary.csv",

        DIRS["threshold"]
        / "best_threshold_per_machine.csv",

        DIRS["threshold"]
        / "all_threshold_search.csv",

        DIRS["evaluation"]
        / "all_predictions.csv",

        DIRS["evaluation"]
        / "per_video_evaluation.csv",

        DIRS["final"]
        / "final_machine_decisions.csv",

        DIRS["final"]
        / "baseline_decision_config.json",

        OUTPUT_DIR
        / "baseline_decision_rnd_summary.txt",
    ]

    for file in important_files:

        print(
            file
        )


if __name__ == "__main__":

    main()
