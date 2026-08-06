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

PREDICTIONS_CSV = (
    BASE_DIR
    / "results"
    / "05_baseline_decision_rnd"
    / "03_evaluation"
    / "all_predictions.csv"
)

DECISION_CSV = (
    BASE_DIR
    / "results"
    / "05_baseline_decision_rnd"
    / "04_final_decision"
    / "final_machine_decisions.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "06_temporal_stability_rnd"
)


# ============================================================
# 2. MACHINES
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
# 3. TEMPORAL STRATEGIES
# ============================================================

TEMPORAL_STRATEGIES = [

    {
        "name": "raw_1_of_1",
        "window_size": 1,
        "votes_required": 1,
    },

    {
        "name": "2_of_3",
        "window_size": 3,
        "votes_required": 2,
    },

    {
        "name": "3_of_3",
        "window_size": 3,
        "votes_required": 3,
    },

    {
        "name": "3_of_5",
        "window_size": 5,
        "votes_required": 3,
    },

    {
        "name": "4_of_5",
        "window_size": 5,
        "votes_required": 4,
    },

    {
        "name": "5_of_5",
        "window_size": 5,
        "votes_required": 5,
    },

    {
        "name": "4_of_7",
        "window_size": 7,
        "votes_required": 4,
    },

    {
        "name": "5_of_7",
        "window_size": 7,
        "votes_required": 5,
    },

    {
        "name": "6_of_7",
        "window_size": 7,
        "votes_required": 6,
    },
]


# ============================================================
# 4. TEMPORAL MODES
# ============================================================

TEMPORAL_MODES = [
    "trailing",
    "centered",
]

RUNTIME_MODE = "trailing"

MIN_MACHINE_SAMPLES = 6


# ============================================================
# 5. OUTPUT DIRECTORIES
# ============================================================

DIRS = {

    "audit":
        OUTPUT_DIR / "00_audit",

    "predictions":
        OUTPUT_DIR / "01_temporal_predictions",

    "evaluation":
        OUTPUT_DIR / "02_evaluation",

    "comparison":
        OUTPUT_DIR / "03_strategy_comparison",

    "final":
        OUTPUT_DIR / "04_final_temporal_config",

    "graphs":
        OUTPUT_DIR / "05_graphs",

    "machine_graphs":
        OUTPUT_DIR
        / "05_graphs"
        / "machines",

    "timeline_graphs":
        OUTPUT_DIR
        / "05_graphs"
        / "timelines",
}


def create_directories():

    for directory in DIRS.values():

        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# 6. LOAD 05 PREDICTIONS
# ============================================================

def load_predictions():

    print()
    print("=" * 100)
    print("STEP 1: LOAD RAW DECISIONS FROM 05")
    print("=" * 100)

    if not PREDICTIONS_CSV.exists():

        raise FileNotFoundError(
            "\n05 predictions not found:\n"
            f"{PREDICTIONS_CSV}"
        )

    df = pd.read_csv(
        PREDICTIONS_CSV
    )

    required_columns = {
        "video",
        "machine_id",
        "frame",
        "label",
        "predicted_label",
        "score",
        "delta",
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

    df["label"] = (
        df["label"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["predicted_label"] = (
        df["predicted_label"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    df["frame"] = pd.to_numeric(
        df["frame"],
        errors="coerce"
    )

    df["score"] = pd.to_numeric(
        df["score"],
        errors="coerce"
    )

    df["delta"] = pd.to_numeric(
        df["delta"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "frame",
            "score",
            "delta",
        ]
    )

    # --------------------------------------------------------
    # Valid labels
    # --------------------------------------------------------

    df = df[
        df["label"].isin(
            [
                "EMPTY",
                "CUP",
            ]
        )
    ].copy()

    df = df[
        df["predicted_label"].isin(
            [
                "EMPTY",
                "CUP",
            ]
        )
    ].copy()

    # --------------------------------------------------------
    # Remove duplicate observations
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=[
            "video",
            "machine_id",
            "frame",
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

    # --------------------------------------------------------
    # Binary representation
    # --------------------------------------------------------

    df["ground_truth_binary"] = (
        df["label"]
        == "CUP"
    ).astype(int)

    df["raw_prediction_binary"] = (
        df["predicted_label"]
        == "CUP"
    ).astype(int)

    print()
    print(
        f"Loaded observations: {len(df)}"
    )

    print()
    print("Available machines in 05 predictions:")

    print(
        df.groupby(
            "machine_id"
        ).size()
    )

    # --------------------------------------------------------
    # Audit machine availability
    # --------------------------------------------------------

    audit_rows = []

    available_machines = set(
        df["machine_id"].unique()
    )

    for machine in EXPECTED_MACHINES:

        if machine in available_machines:

            machine_df = df[
                df["machine_id"]
                == machine
            ]

            audit_rows.append({

                "machine_id":
                    machine,

                "available_in_05_predictions":
                    True,

                "samples":
                    len(machine_df),

                "empty_samples":
                    int(
                        np.sum(
                            machine_df[
                                "ground_truth_binary"
                            ]
                            == 0
                        )
                    ),

                "cup_samples":
                    int(
                        np.sum(
                            machine_df[
                                "ground_truth_binary"
                            ]
                            == 1
                        )
                    ),

                "status":
                    "AVAILABLE",
            })

        else:

            audit_rows.append({

                "machine_id":
                    machine,

                "available_in_05_predictions":
                    False,

                "samples":
                    0,

                "empty_samples":
                    0,

                "cup_samples":
                    0,

                "status":
                    "NOT_AVAILABLE_FROM_05",
            })

    audit_df = pd.DataFrame(
        audit_rows
    )

    audit_df.to_csv(
        DIRS["audit"]
        / "machine_input_availability.csv",
        index=False
    )

    print()
    print("Complete machine availability:")

    print(
        audit_df.to_string(
            index=False
        )
    )

    return df


# ============================================================
# 7. LOAD 05 DECISION CONFIGURATION
# ============================================================

def load_decision_config():

    print()
    print("=" * 100)
    print("STEP 2: LOAD MACHINE DECISIONS FROM 05")
    print("=" * 100)

    if not DECISION_CSV.exists():

        raise FileNotFoundError(
            "\n05 final decision file not found:\n"
            f"{DECISION_CSV}"
        )

    df = pd.read_csv(
        DECISION_CSV
    )

    df["machine_id"] = (
        df["machine_id"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    print()

    columns = [
        column
        for column in [
            "machine_id",
            "empty_baseline",
            "direction",
            "best_delta_threshold",
            "balanced_accuracy",
            "decision_quality",
        ]
        if column in df.columns
    ]

    print(
        df[
            columns
        ].to_string(
            index=False
        )
    )

    return df


# ============================================================
# 8. CLASSIFICATION METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true,
        dtype=int
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int
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
        if total > 0
        else 0
    )

    precision = (
        tp
        / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp
        / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    specificity = (
        tn
        / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    if (
        precision
        +
        recall
    ) > 0:

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
        recall
        +
        specificity
    ) / 2

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
# 9. TRAILING TEMPORAL VOTING
# ============================================================

def temporal_vote_trailing(
    values,
    window_size,
    votes_required
):

    values = np.asarray(
        values,
        dtype=int
    )

    output = np.zeros(
        len(values),
        dtype=int
    )

    vote_counts = np.zeros(
        len(values),
        dtype=int
    )

    actual_window_sizes = np.zeros(
        len(values),
        dtype=int
    )

    for i in range(
        len(values)
    ):

        start = max(
            0,
            i - window_size + 1
        )

        window = values[
            start:
            i + 1
        ]

        actual_size = len(
            window
        )

        cup_votes = int(
            np.sum(
                window
            )
        )

        # ----------------------------------------------------
        # Warm-up proportional threshold
        # --------------------------------------------------------

        scaled_required = int(
            np.ceil(
                (
                    votes_required
                    /
                    window_size
                )
                *
                actual_size
            )
        )

        scaled_required = max(
            1,
            scaled_required
        )

        output[i] = int(
            cup_votes
            >= scaled_required
        )

        vote_counts[i] = (
            cup_votes
        )

        actual_window_sizes[i] = (
            actual_size
        )

    return (
        output,
        vote_counts,
        actual_window_sizes
    )


# ============================================================
# 10. CENTERED TEMPORAL VOTING
#
# Offline analysis only.
# ============================================================

def temporal_vote_centered(
    values,
    window_size,
    votes_required
):

    values = np.asarray(
        values,
        dtype=int
    )

    output = np.zeros(
        len(values),
        dtype=int
    )

    vote_counts = np.zeros(
        len(values),
        dtype=int
    )

    actual_window_sizes = np.zeros(
        len(values),
        dtype=int
    )

    half_window = (
        window_size // 2
    )

    for i in range(
        len(values)
    ):

        start = max(
            0,
            i - half_window
        )

        end = min(
            len(values),
            i + half_window + 1
        )

        window = values[
            start:end
        ]

        actual_size = len(
            window
        )

        cup_votes = int(
            np.sum(
                window
            )
        )

        scaled_required = int(
            np.ceil(
                (
                    votes_required
                    /
                    window_size
                )
                *
                actual_size
            )
        )

        scaled_required = max(
            1,
            scaled_required
        )

        output[i] = int(
            cup_votes
            >= scaled_required
        )

        vote_counts[i] = (
            cup_votes
        )

        actual_window_sizes[i] = (
            actual_size
        )

    return (
        output,
        vote_counts,
        actual_window_sizes
    )


# ============================================================
# 11. RUN TEMPORAL STRATEGIES
# ============================================================

def run_temporal_strategies(
    df
):

    print()
    print("=" * 100)
    print("STEP 3: RUN TEMPORAL STRATEGIES")
    print("=" * 100)

    all_results = []

    groups = df.groupby(
        [
            "machine_id",
            "video",
        ],
        sort=False
    )

    for (
        machine,
        video
    ), subset in groups:

        subset = subset.sort_values(
            "frame"
        ).copy()

        raw_values = (
            subset[
                "raw_prediction_binary"
            ]
            .to_numpy(
                dtype=int
            )
        )

        for strategy in TEMPORAL_STRATEGIES:

            strategy_name = (
                strategy[
                    "name"
                ]
            )

            window_size = int(
                strategy[
                    "window_size"
                ]
            )

            votes_required = int(
                strategy[
                    "votes_required"
                ]
            )

            for mode in TEMPORAL_MODES:

                if mode == "trailing":

                    (
                        temporal_prediction,
                        vote_counts,
                        actual_window_sizes
                    ) = temporal_vote_trailing(
                        raw_values,
                        window_size,
                        votes_required
                    )

                else:

                    (
                        temporal_prediction,
                        vote_counts,
                        actual_window_sizes
                    ) = temporal_vote_centered(
                        raw_values,
                        window_size,
                        votes_required
                    )

                result = (
                    subset.copy()
                )

                result[
                    "temporal_mode"
                ] = mode

                result[
                    "strategy"
                ] = strategy_name

                result[
                    "window_size"
                ] = window_size

                result[
                    "votes_required"
                ] = votes_required

                result[
                    "cup_votes"
                ] = vote_counts

                result[
                    "actual_window_size"
                ] = actual_window_sizes

                result[
                    "temporal_prediction_binary"
                ] = temporal_prediction

                result[
                    "temporal_prediction"
                ] = np.where(
                    temporal_prediction == 1,
                    "CUP",
                    "EMPTY"
                )

                result[
                    "temporal_correct"
                ] = (
                    result[
                        "temporal_prediction_binary"
                    ]
                    ==
                    result[
                        "ground_truth_binary"
                    ]
                )

                all_results.append(
                    result
                )

    if not all_results:

        raise ValueError(
            "No temporal results were generated."
        )

    result_df = pd.concat(
        all_results,
        ignore_index=True
    )

    result_df.to_csv(
        DIRS["predictions"]
        / "all_temporal_predictions.csv",
        index=False
    )

    print()
    print(
        f"Generated temporal rows: "
        f"{len(result_df):,}"
    )

    return result_df


# ============================================================
# 12. EVALUATE MACHINE STRATEGIES
# ============================================================

def evaluate_machine_strategies(
    temporal_df
):

    print()
    print("=" * 100)
    print("STEP 4: EVALUATE EACH TEMPORAL STRATEGY")
    print("=" * 100)

    rows = []

    grouped = temporal_df.groupby(
        [
            "machine_id",
            "temporal_mode",
            "strategy",
            "window_size",
            "votes_required",
        ]
    )

    for (
        machine,
        mode,
        strategy,
        window_size,
        votes_required
    ), subset in grouped:

        metrics = calculate_metrics(
            subset[
                "ground_truth_binary"
            ],
            subset[
                "temporal_prediction_binary"
            ]
        )

        raw_metrics = calculate_metrics(
            subset[
                "ground_truth_binary"
            ],
            subset[
                "raw_prediction_binary"
            ]
        )

        rows.append({

            "machine_id":
                machine,

            "temporal_mode":
                mode,

            "strategy":
                strategy,

            "window_size":
                window_size,

            "votes_required":
                votes_required,

            "samples":
                len(subset),

            "empty_samples":
                int(
                    np.sum(
                        subset[
                            "ground_truth_binary"
                        ]
                        == 0
                    )
                ),

            "cup_samples":
                int(
                    np.sum(
                        subset[
                            "ground_truth_binary"
                        ]
                        == 1
                    )
                ),

            "accuracy":
                metrics["accuracy"],

            "balanced_accuracy":
                metrics[
                    "balanced_accuracy"
                ],

            "precision":
                metrics["precision"],

            "recall":
                metrics["recall"],

            "specificity":
                metrics["specificity"],

            "f1":
                metrics["f1"],

            "tp":
                metrics["tp"],

            "tn":
                metrics["tn"],

            "fp":
                metrics["fp"],

            "fn":
                metrics["fn"],

            "raw_accuracy":
                raw_metrics["accuracy"],

            "raw_balanced_accuracy":
                raw_metrics[
                    "balanced_accuracy"
                ],

            "raw_f1":
                raw_metrics["f1"],

            "balanced_accuracy_change":
                (
                    metrics[
                        "balanced_accuracy"
                    ]
                    -
                    raw_metrics[
                        "balanced_accuracy"
                    ]
                ),

            "f1_change":
                (
                    metrics["f1"]
                    -
                    raw_metrics["f1"]
                ),

            "accuracy_change":
                (
                    metrics["accuracy"]
                    -
                    raw_metrics["accuracy"]
                ),
        })

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        DIRS["evaluation"]
        / "machine_strategy_evaluation.csv",
        index=False
    )

    return result


# ============================================================
# 13. PER-VIDEO EVALUATION
# ============================================================

def evaluate_per_video(
    temporal_df
):

    print()
    print("=" * 100)
    print("STEP 5: PER-VIDEO EVALUATION")
    print("=" * 100)

    rows = []

    grouped = temporal_df.groupby(
        [
            "machine_id",
            "video",
            "temporal_mode",
            "strategy",
            "window_size",
            "votes_required",
        ]
    )

    for (
        machine,
        video,
        mode,
        strategy,
        window_size,
        votes_required
    ), subset in grouped:

        metrics = calculate_metrics(
            subset[
                "ground_truth_binary"
            ],
            subset[
                "temporal_prediction_binary"
            ]
        )

        rows.append({

            "machine_id":
                machine,

            "video":
                video,

            "temporal_mode":
                mode,

            "strategy":
                strategy,

            "window_size":
                window_size,

            "votes_required":
                votes_required,

            "samples":
                len(subset),

            "empty_samples":
                int(
                    np.sum(
                        subset[
                            "ground_truth_binary"
                        ]
                        == 0
                    )
                ),

            "cup_samples":
                int(
                    np.sum(
                        subset[
                            "ground_truth_binary"
                        ]
                        == 1
                    )
                ),

            **metrics
        })

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        DIRS["evaluation"]
        / "per_video_strategy_evaluation.csv",
        index=False
    )

    return result


# ============================================================
# 14. FIND BEST RUNTIME STRATEGY
# ============================================================

def find_best_runtime_strategy(
    evaluation_df
):

    print()
    print("=" * 100)
    print("STEP 6: SELECT BEST RUNTIME TEMPORAL STRATEGY")
    print("=" * 100)

    trailing = evaluation_df[
        evaluation_df[
            "temporal_mode"
        ]
        == RUNTIME_MODE
    ].copy()

    rows = []

    for machine in EXPECTED_MACHINES:

        subset = trailing[
            trailing[
                "machine_id"
            ]
            == machine
        ].copy()

        if subset.empty:

            print(
                f"{machine}: NOT TESTABLE"
            )

            continue

        subset = subset.sort_values(
            [
                "balanced_accuracy",
                "f1",
                "accuracy",
                "window_size",
                "votes_required",
            ],
            ascending=[
                False,
                False,
                False,
                True,
                True,
            ]
        ).reset_index(
            drop=True
        )

        best = subset.iloc[0]

        support_ok = (
            int(
                best["samples"]
            )
            >= MIN_MACHINE_SAMPLES
        )

        improvement = float(
            best[
                "balanced_accuracy_change"
            ]
        )

        if improvement > 0.05:

            temporal_effect = (
                "CLEAR_IMPROVEMENT"
            )

        elif improvement > 0:

            temporal_effect = (
                "SMALL_IMPROVEMENT"
            )

        elif np.isclose(
            improvement,
            0
        ):

            temporal_effect = (
                "NO_CHANGE"
            )

        else:

            temporal_effect = (
                "WORSE"
            )

        rows.append({

            "machine_id":
                machine,

            "best_strategy":
                best["strategy"],

            "temporal_mode":
                best["temporal_mode"],

            "window_size":
                int(
                    best["window_size"]
                ),

            "votes_required":
                int(
                    best["votes_required"]
                ),

            "samples":
                int(
                    best["samples"]
                ),

            "empty_samples":
                int(
                    best["empty_samples"]
                ),

            "cup_samples":
                int(
                    best["cup_samples"]
                ),

            "support_ok":
                support_ok,

            "raw_accuracy":
                float(
                    best["raw_accuracy"]
                ),

            "raw_balanced_accuracy":
                float(
                    best[
                        "raw_balanced_accuracy"
                    ]
                ),

            "raw_f1":
                float(
                    best["raw_f1"]
                ),

            "temporal_accuracy":
                float(
                    best["accuracy"]
                ),

            "temporal_balanced_accuracy":
                float(
                    best[
                        "balanced_accuracy"
                    ]
                ),

            "temporal_f1":
                float(
                    best["f1"]
                ),

            "balanced_accuracy_change":
                float(
                    best[
                        "balanced_accuracy_change"
                    ]
                ),

            "f1_change":
                float(
                    best["f1_change"]
                ),

            "accuracy_change":
                float(
                    best["accuracy_change"]
                ),

            "temporal_effect":
                temporal_effect,

            "tp":
                int(
                    best["tp"]
                ),

            "tn":
                int(
                    best["tn"]
                ),

            "fp":
                int(
                    best["fp"]
                ),

            "fn":
                int(
                    best["fn"]
                ),
        })

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        DIRS["final"]
        / "best_temporal_strategy_per_machine.csv",
        index=False
    )

    return result


# ============================================================
# 15. GLOBAL STRATEGY COMPARISON
# ============================================================

def evaluate_global_strategies(
    temporal_df
):

    print()
    print("=" * 100)
    print("STEP 7: GLOBAL TEMPORAL STRATEGY COMPARISON")
    print("=" * 100)

    rows = []

    trailing = temporal_df[
        temporal_df[
            "temporal_mode"
        ]
        == RUNTIME_MODE
    ]

    grouped = trailing.groupby(
        [
            "strategy",
            "window_size",
            "votes_required",
        ]
    )

    for (
        strategy,
        window_size,
        votes_required
    ), subset in grouped:

        metrics = calculate_metrics(
            subset[
                "ground_truth_binary"
            ],
            subset[
                "temporal_prediction_binary"
            ]
        )

        raw_metrics = calculate_metrics(
            subset[
                "ground_truth_binary"
            ],
            subset[
                "raw_prediction_binary"
            ]
        )

        machine_balanced = []
        machine_f1 = []

        for machine, machine_df in subset.groupby(
            "machine_id"
        ):

            machine_metrics = (
                calculate_metrics(
                    machine_df[
                        "ground_truth_binary"
                    ],
                    machine_df[
                        "temporal_prediction_binary"
                    ]
                )
            )

            machine_balanced.append(
                machine_metrics[
                    "balanced_accuracy"
                ]
            )

            machine_f1.append(
                machine_metrics[
                    "f1"
                ]
            )

        rows.append({

            "strategy":
                strategy,

            "window_size":
                window_size,

            "votes_required":
                votes_required,

            "samples":
                len(subset),

            "accuracy":
                metrics["accuracy"],

            "balanced_accuracy":
                metrics[
                    "balanced_accuracy"
                ],

            "f1":
                metrics["f1"],

            "precision":
                metrics["precision"],

            "recall":
                metrics["recall"],

            "specificity":
                metrics["specificity"],

            "macro_machine_balanced_accuracy":
                float(
                    np.mean(
                        machine_balanced
                    )
                )
                if machine_balanced
                else np.nan,

            "macro_machine_f1":
                float(
                    np.mean(
                        machine_f1
                    )
                )
                if machine_f1
                else np.nan,

            "raw_balanced_accuracy":
                raw_metrics[
                    "balanced_accuracy"
                ],

            "raw_f1":
                raw_metrics["f1"],

            "balanced_accuracy_change":
                (
                    metrics[
                        "balanced_accuracy"
                    ]
                    -
                    raw_metrics[
                        "balanced_accuracy"
                    ]
                ),
        })

    result = pd.DataFrame(
        rows
    )

    result = result.sort_values(
        [
            "macro_machine_balanced_accuracy",
            "macro_machine_f1",
            "balanced_accuracy",
            "window_size",
            "votes_required",
        ],
        ascending=[
            False,
            False,
            False,
            True,
            True,
        ]
    ).reset_index(
        drop=True
    )

    result.to_csv(
        DIRS["comparison"]
        / "global_strategy_comparison.csv",
        index=False
    )

    print()

    print(
        result[
            [
                "strategy",
                "macro_machine_balanced_accuracy",
                "macro_machine_f1",
                "balanced_accuracy",
                "f1",
            ]
        ].to_string(
            index=False
        )
    )

    return result


# ============================================================
# 16. TRAILING VS CENTERED
# ============================================================

def compare_temporal_modes(
    evaluation_df
):

    print()
    print("=" * 100)
    print("STEP 8: TRAILING VS CENTERED")
    print("=" * 100)

    result = (
        evaluation_df.groupby(
            [
                "temporal_mode",
                "strategy",
                "window_size",
                "votes_required",
            ]
        )
        .agg(
            mean_balanced_accuracy=(
                "balanced_accuracy",
                "mean"
            ),

            mean_f1=(
                "f1",
                "mean"
            ),

            mean_accuracy=(
                "accuracy",
                "mean"
            ),
        )
        .reset_index()
    )

    result.to_csv(
        DIRS["comparison"]
        / "trailing_vs_centered.csv",
        index=False
    )

    return result


# ============================================================
# 17. TRANSITION DELAY ANALYSIS
# ============================================================

def analyze_transition_delay(
    temporal_df,
    best_df
):

    print()
    print("=" * 100)
    print("STEP 9: TRANSITION DELAY ANALYSIS")
    print("=" * 100)

    rows = []

    if best_df.empty:

        result = pd.DataFrame()

        result.to_csv(
            DIRS["evaluation"]
            / "transition_delay_analysis.csv",
            index=False
        )

        return result

    for best in best_df.itertuples():

        machine = (
            best.machine_id
        )

        strategy = (
            best.best_strategy
        )

        subset = temporal_df[
            (
                temporal_df[
                    "machine_id"
                ]
                == machine
            )
            &
            (
                temporal_df[
                    "temporal_mode"
                ]
                == RUNTIME_MODE
            )
            &
            (
                temporal_df[
                    "strategy"
                ]
                == strategy
            )
        ].copy()

        for video, video_df in subset.groupby(
            "video"
        ):

            video_df = video_df.sort_values(
                "frame"
            ).reset_index(
                drop=True
            )

            gt = video_df[
                "ground_truth_binary"
            ].to_numpy()

            pred = video_df[
                "temporal_prediction_binary"
            ].to_numpy()

            frames = video_df[
                "frame"
            ].to_numpy()

            for i in range(
                1,
                len(video_df)
            ):

                if (
                    gt[i]
                    ==
                    gt[i - 1]
                ):

                    continue

                new_state = (
                    gt[i]
                )

                transition_frame = (
                    frames[i]
                )

                delay_readings = None
                detection_frame = None

                for j in range(
                    i,
                    len(video_df)
                ):

                    if (
                        pred[j]
                        ==
                        new_state
                    ):

                        delay_readings = (
                            j - i
                        )

                        detection_frame = (
                            frames[j]
                        )

                        break

                rows.append({

                    "machine_id":
                        machine,

                    "video":
                        video,

                    "strategy":
                        strategy,

                    "transition_frame":
                        transition_frame,

                    "new_state":
                        (
                            "CUP"
                            if new_state == 1
                            else "EMPTY"
                        ),

                    "detection_frame":
                        detection_frame,

                    "delay_readings":
                        delay_readings,

                    "delay_frames":
                        (
                            detection_frame
                            -
                            transition_frame
                            if detection_frame
                            is not None
                            else np.nan
                        ),
                })

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        DIRS["evaluation"]
        / "transition_delay_analysis.csv",
        index=False
    )

    return result


# ============================================================
# 18. COMPLETE MACHINE SUMMARY
#
# IMPORTANT:
#
# ALL expected machines are included.
# ============================================================

def create_machine_summary(
    best_df,
    decision_df
):

    print()
    print("=" * 100)
    print("STEP 10: CREATE COMPLETE MACHINE SUMMARY")
    print("=" * 100)

    rows = []

    for machine in EXPECTED_MACHINES:

        decision_row = decision_df[
            decision_df[
                "machine_id"
            ]
            == machine
        ]

        temporal_row = best_df[
            best_df[
                "machine_id"
            ]
            == machine
        ]

        # ====================================================
        # TESTED MACHINE
        # ====================================================

        if not temporal_row.empty:

            temporal = (
                temporal_row.iloc[0]
            )

            if not decision_row.empty:

                decision = (
                    decision_row.iloc[0]
                )

                empty_baseline = (
                    decision.get(
                        "empty_baseline",
                        np.nan
                    )
                )

                baseline_quality = (
                    decision.get(
                        "baseline_quality",
                        None
                    )
                )

                direction = (
                    decision.get(
                        "direction",
                        None
                    )
                )

                delta_threshold = (
                    decision.get(
                        "best_delta_threshold",
                        np.nan
                    )
                )

                decision_quality = (
                    decision.get(
                        "decision_quality",
                        None
                    )
                )

            else:

                empty_baseline = np.nan
                baseline_quality = None
                direction = None
                delta_threshold = np.nan
                decision_quality = None

            improvement = float(
                temporal[
                    "balanced_accuracy_change"
                ]
            )

            use_temporal = (
                improvement > 0
            )

            if use_temporal:

                recommended_strategy = (
                    temporal[
                        "best_strategy"
                    ]
                )

                final_runtime_stage = (
                    "TEMPORAL_CONFIRMATION"
                )

            else:

                recommended_strategy = (
                    "raw_1_of_1"
                )

                final_runtime_stage = (
                    "RAW_DECISION"
                )

            rows.append({

                "machine_id":
                    machine,

                "decision_rnd_status":
                    "AVAILABLE",

                "temporal_rnd_status":
                    "TESTED",

                "data_status":
                    "AVAILABLE",

                "empty_baseline":
                    empty_baseline,

                "baseline_quality":
                    baseline_quality,

                "direction":
                    direction,

                "best_delta_threshold":
                    delta_threshold,

                "decision_quality":
                    decision_quality,

                "samples":
                    int(
                        temporal[
                            "samples"
                        ]
                    ),

                "empty_samples":
                    int(
                        temporal[
                            "empty_samples"
                        ]
                    ),

                "cup_samples":
                    int(
                        temporal[
                            "cup_samples"
                        ]
                    ),

                "raw_accuracy":
                    float(
                        temporal[
                            "raw_accuracy"
                        ]
                    ),

                "raw_balanced_accuracy":
                    float(
                        temporal[
                            "raw_balanced_accuracy"
                        ]
                    ),

                "raw_f1":
                    float(
                        temporal[
                            "raw_f1"
                        ]
                    ),

                "best_temporal_strategy":
                    temporal[
                        "best_strategy"
                    ],

                "window_size":
                    int(
                        temporal[
                            "window_size"
                        ]
                    ),

                "votes_required":
                    int(
                        temporal[
                            "votes_required"
                        ]
                    ),

                "temporal_accuracy":
                    float(
                        temporal[
                            "temporal_accuracy"
                        ]
                    ),

                "temporal_balanced_accuracy":
                    float(
                        temporal[
                            "temporal_balanced_accuracy"
                        ]
                    ),

                "temporal_f1":
                    float(
                        temporal[
                            "temporal_f1"
                        ]
                    ),

                "balanced_accuracy_change":
                    float(
                        temporal[
                            "balanced_accuracy_change"
                        ]
                    ),

                "f1_change":
                    float(
                        temporal[
                            "f1_change"
                        ]
                    ),

                "accuracy_change":
                    float(
                        temporal[
                            "accuracy_change"
                        ]
                    ),

                "temporal_effect":
                    temporal[
                        "temporal_effect"
                    ],

                "recommended_use_temporal":
                    use_temporal,

                "recommended_runtime_strategy":
                    recommended_strategy,

                "final_runtime_stage":
                    final_runtime_stage,

                "reason":
                    (
                        "Temporal evaluation available."
                    ),
            })

        # ====================================================
        # NOT TESTABLE
        # ====================================================

        else:

            if not decision_row.empty:

                decision = (
                    decision_row.iloc[0]
                )

                empty_baseline = (
                    decision.get(
                        "empty_baseline",
                        np.nan
                    )
                )

                baseline_quality = (
                    decision.get(
                        "baseline_quality",
                        None
                    )
                )

            else:

                empty_baseline = np.nan
                baseline_quality = None

            rows.append({

                "machine_id":
                    machine,

                "decision_rnd_status":
                    "INSUFFICIENT_DATA",

                "temporal_rnd_status":
                    "NOT_TESTABLE",

                "data_status":
                    "INSUFFICIENT_EMPTY_CUP_DATA",

                "empty_baseline":
                    empty_baseline,

                "baseline_quality":
                    baseline_quality,

                "direction":
                    None,

                "best_delta_threshold":
                    np.nan,

                "decision_quality":
                    "INSUFFICIENT_DATA",

                "samples":
                    0,

                "empty_samples":
                    np.nan,

                "cup_samples":
                    np.nan,

                "raw_accuracy":
                    np.nan,

                "raw_balanced_accuracy":
                    np.nan,

                "raw_f1":
                    np.nan,

                "best_temporal_strategy":
                    None,

                "window_size":
                    np.nan,

                "votes_required":
                    np.nan,

                "temporal_accuracy":
                    np.nan,

                "temporal_balanced_accuracy":
                    np.nan,

                "temporal_f1":
                    np.nan,

                "balanced_accuracy_change":
                    np.nan,

                "f1_change":
                    np.nan,

                "accuracy_change":
                    np.nan,

                "temporal_effect":
                    "NOT_TESTABLE",

                "recommended_use_temporal":
                    False,

                "recommended_runtime_strategy":
                    "NOT_AVAILABLE",

                "final_runtime_stage":
                    "INSUFFICIENT_DATA",

                "reason":
                    (
                        "Machine does not have sufficient "
                        "EMPTY and CUP labelled observations "
                        "to learn and evaluate a decision rule."
                    ),
            })

    final = pd.DataFrame(
        rows
    )

    final.to_csv(
        DIRS["final"]
        / "final_temporal_decisions.csv",
        index=False
    )

    print()

    print(
        final[
            [
                "machine_id",
                "decision_rnd_status",
                "temporal_rnd_status",
                "raw_balanced_accuracy",
                "best_temporal_strategy",
                "temporal_balanced_accuracy",
                "recommended_runtime_strategy",
            ]
        ].to_string(
            index=False
        )
    )

    return final


# ============================================================
# 19. MACHINE STRATEGY GRAPH
# ============================================================

def plot_machine_strategy_performance(
    machine,
    evaluation_df
):

    subset = evaluation_df[
        (
            evaluation_df[
                "machine_id"
            ]
            == machine
        )
        &
        (
            evaluation_df[
                "temporal_mode"
            ]
            == RUNTIME_MODE
        )
    ].copy()

    if subset.empty:

        return

    order = [
        strategy["name"]
        for strategy
        in TEMPORAL_STRATEGIES
    ]

    subset["strategy"] = pd.Categorical(
        subset["strategy"],
        categories=order,
        ordered=True
    )

    subset = subset.sort_values(
        "strategy"
    )

    x = np.arange(
        len(subset)
    )

    width = 0.35

    plt.figure(
        figsize=(13, 6)
    )

    plt.bar(
        x - width / 2,
        subset[
            "balanced_accuracy"
        ],
        width,
        label="Balanced Accuracy"
    )

    plt.bar(
        x + width / 2,
        subset["f1"],
        width,
        label="F1"
    )

    plt.xticks(
        x,
        subset["strategy"],
        rotation=35,
        ha="right"
    )

    plt.ylim(
        0,
        1.05
    )

    plt.xlabel(
        "Temporal Strategy"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        f"{machine} - Temporal Strategy Performance"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        DIRS["machine_graphs"]
        / f"{machine}_strategy_performance.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 20. RAW VS TEMPORAL GRAPH
# ============================================================

def plot_raw_vs_temporal(
    final_df
):

    tested = final_df[
        final_df[
            "temporal_rnd_status"
        ]
        == "TESTED"
    ].copy()

    if tested.empty:

        return

    x = np.arange(
        len(tested)
    )

    width = 0.35

    plt.figure(
        figsize=(13, 6)
    )

    plt.bar(
        x - width / 2,
        tested[
            "raw_balanced_accuracy"
        ],
        width,
        label="Raw Decision"
    )

    plt.bar(
        x + width / 2,
        tested[
            "temporal_balanced_accuracy"
        ],
        width,
        label="Best Temporal"
    )

    plt.xticks(
        x,
        tested["machine_id"]
    )

    plt.ylim(
        0,
        1.05
    )

    plt.xlabel(
        "Machine"
    )

    plt.ylabel(
        "Balanced Accuracy"
    )

    plt.title(
        "Raw Decision vs Best Temporal Confirmation"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        DIRS["graphs"]
        / "raw_vs_best_temporal.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 21. TEMPORAL IMPROVEMENT GRAPH
# ============================================================

def plot_temporal_improvement(
    final_df
):

    tested = final_df[
        final_df[
            "temporal_rnd_status"
        ]
        == "TESTED"
    ].copy()

    if tested.empty:

        return

    x = np.arange(
        len(tested)
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        x,
        tested[
            "balanced_accuracy_change"
        ]
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xticks(
        x,
        tested["machine_id"]
    )

    plt.xlabel(
        "Machine"
    )

    plt.ylabel(
        "Balanced Accuracy Change"
    )

    plt.title(
        "Effect of Temporal Confirmation"
    )

    plt.tight_layout()

    plt.savefig(
        DIRS["graphs"]
        / "temporal_improvement.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 22. GLOBAL STRATEGY GRAPH
# ============================================================

def plot_global_strategy_comparison(
    global_df
):

    if global_df.empty:

        return

    x = np.arange(
        len(global_df)
    )

    plt.figure(
        figsize=(13, 6)
    )

    plt.bar(
        x,
        global_df[
            "macro_machine_balanced_accuracy"
        ]
    )

    plt.xticks(
        x,
        global_df["strategy"],
        rotation=35,
        ha="right"
    )

    plt.ylim(
        0,
        1.05
    )

    plt.ylabel(
        "Macro Machine Balanced Accuracy"
    )

    plt.xlabel(
        "Temporal Strategy"
    )

    plt.title(
        "Global Temporal Strategy Comparison"
    )

    plt.tight_layout()

    plt.savefig(
        DIRS["graphs"]
        / "global_strategy_comparison.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 23. BEST STRATEGY TIMELINES
# ============================================================

def plot_best_strategy_timelines(
    temporal_df,
    final_df
):

    tested = final_df[
        final_df[
            "temporal_rnd_status"
        ]
        == "TESTED"
    ]

    for row in tested.itertuples():

        machine = (
            row.machine_id
        )

        strategy = (
            row.best_temporal_strategy
        )

        subset = temporal_df[
            (
                temporal_df[
                    "machine_id"
                ]
                == machine
            )
            &
            (
                temporal_df[
                    "temporal_mode"
                ]
                == RUNTIME_MODE
            )
            &
            (
                temporal_df[
                    "strategy"
                ]
                == strategy
            )
        ].copy()

        for video, video_df in subset.groupby(
            "video"
        ):

            video_df = video_df.sort_values(
                "frame"
            )

            plt.figure(
                figsize=(15, 7)
            )

            plt.step(
                video_df["frame"],
                video_df[
                    "ground_truth_binary"
                ],
                where="post",
                label="Ground Truth"
            )

            plt.step(
                video_df["frame"],
                video_df[
                    "raw_prediction_binary"
                ],
                where="post",
                label="Raw Prediction"
            )

            plt.step(
                video_df["frame"],
                video_df[
                    "temporal_prediction_binary"
                ],
                where="post",
                label=(
                    f"Temporal ({strategy})"
                )
            )

            plt.yticks(
                [
                    0,
                    1,
                ],
                [
                    "EMPTY",
                    "CUP",
                ]
            )

            plt.xlabel(
                "Frame"
            )

            plt.ylabel(
                "State"
            )

            plt.title(
                f"{machine} | "
                f"{Path(video).stem}\n"
                f"Temporal strategy: "
                f"{strategy}"
            )

            plt.legend()

            plt.tight_layout()

            filename = (
                f"{machine}_"
                f"{Path(video).stem}_"
                f"{strategy}.png"
            )

            plt.savefig(
                DIRS["timeline_graphs"]
                / filename,
                dpi=150
            )

            plt.close()


# ============================================================
# 24. SAVE JSON CONFIGURATION
# ============================================================

def save_temporal_config(
    final_df,
    global_df
):

    output = {

        "experiment":
            "06_temporal_stability_rnd",

        "runtime_mode":
            "trailing",

        "all_expected_machines":
            EXPECTED_MACHINES,

        "temporal_principle":
            (
                "Confirm CUP when the required "
                "number of CUP votes occurs within "
                "the recent reading window."
            ),

        "strategies_tested":
            TEMPORAL_STRATEGIES,

        "global_best_strategy":
            None,

        "machines":
            {},
    }

    if not global_df.empty:

        global_best = (
            global_df.iloc[0]
        )

        output[
            "global_best_strategy"
        ] = {

            "strategy":
                global_best[
                    "strategy"
                ],

            "window_size":
                int(
                    global_best[
                        "window_size"
                    ]
                ),

            "votes_required":
                int(
                    global_best[
                        "votes_required"
                    ]
                ),

            "macro_machine_balanced_accuracy":
                float(
                    global_best[
                        "macro_machine_balanced_accuracy"
                    ]
                ),

            "macro_machine_f1":
                float(
                    global_best[
                        "macro_machine_f1"
                    ]
                ),
        }

    for row in final_df.to_dict(
        orient="records"
    ):

        machine = (
            row["machine_id"]
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
                    bool,
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
                    int,
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
                    float,
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
        ][machine] = (
            machine_data
        )

    with open(
        DIRS["final"]
        / "temporal_stability_config.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )


# ============================================================
# 25. SAVE SUMMARY
# ============================================================

def save_summary(
    final_df,
    global_df,
    transition_df
):

    lines = []

    lines.append(
        "06 - TEMPORAL STABILITY R&D"
    )

    lines.append(
        "=" * 100
    )

    lines.append("")

    lines.append(
        "Purpose:"
    )

    lines.append(
        "Evaluate whether multiple-reading confirmation "
        "improves the raw baseline decision."
    )

    lines.append("")

    lines.append(
        "Runtime temporal mode:"
    )

    lines.append(
        "TRAILING"
    )

    lines.append("")

    lines.append(
        "All expected machines:"
    )

    lines.append(
        ", ".join(
            EXPECTED_MACHINES
        )
    )

    lines.append("")

    lines.append(
        "=" * 100
    )

    lines.append(
        "GLOBAL RESULT"
    )

    lines.append(
        "=" * 100
    )

    if not global_df.empty:

        best = (
            global_df.iloc[0]
        )

        lines.append("")

        lines.append(
            "NOTE:"
        )

        lines.append(
            "Global strategy result includes "
            "TESTABLE machines only."
        )

        lines.append("")

        lines.append(
            f"Best global strategy: "
            f"{best['strategy']}"
        )

        lines.append(
            "Macro machine balanced accuracy: "
            f"{best['macro_machine_balanced_accuracy']:.4f}"
        )

        lines.append(
            f"Macro machine F1: "
            f"{best['macro_machine_f1']:.4f}"
        )

    lines.append("")

    lines.append(
        "=" * 100
    )

    lines.append(
        "COMPLETE MACHINE RESULTS"
    )

    lines.append(
        "=" * 100
    )

    for row in final_df.itertuples():

        lines.append("")

        lines.append(
            f"Machine: "
            f"{row.machine_id}"
        )

        lines.append(
            f"Decision R&D status: "
            f"{row.decision_rnd_status}"
        )

        lines.append(
            f"Temporal R&D status: "
            f"{row.temporal_rnd_status}"
        )

        # ----------------------------------------------------
        # TESTED
        # ----------------------------------------------------

        if (
            row.temporal_rnd_status
            == "TESTED"
        ):

            lines.append(
                f"Raw balanced accuracy: "
                f"{row.raw_balanced_accuracy:.4f}"
            )

            lines.append(
                f"Raw F1: "
                f"{row.raw_f1:.4f}"
            )

            lines.append(
                f"Best temporal strategy: "
                f"{row.best_temporal_strategy}"
            )

            lines.append(
                f"Window size: "
                f"{int(row.window_size)}"
            )

            lines.append(
                f"Votes required: "
                f"{int(row.votes_required)}"
            )

            lines.append(
                "Temporal balanced accuracy: "
                f"{row.temporal_balanced_accuracy:.4f}"
            )

            lines.append(
                f"Temporal F1: "
                f"{row.temporal_f1:.4f}"
            )

            lines.append(
                "Balanced accuracy change: "
                f"{row.balanced_accuracy_change:+.4f}"
            )

            lines.append(
                f"Temporal effect: "
                f"{row.temporal_effect}"
            )

            lines.append(
                "Recommended use temporal: "
                f"{row.recommended_use_temporal}"
            )

            lines.append(
                "Recommended runtime strategy: "
                f"{row.recommended_runtime_strategy}"
            )

        # ----------------------------------------------------
        # NOT TESTABLE
        # ----------------------------------------------------

        else:

            lines.append(
                "Raw balanced accuracy: N/A"
            )

            lines.append(
                "Best temporal strategy: N/A"
            )

            lines.append(
                "Temporal balanced accuracy: N/A"
            )

            lines.append(
                "Recommended runtime strategy: "
                "NOT_AVAILABLE"
            )

            lines.append(
                f"Reason: "
                f"{row.reason}"
            )

    lines.append("")

    lines.append(
        "=" * 100
    )

    lines.append(
        "TRANSITION DELAY"
    )

    lines.append(
        "=" * 100
    )

    if transition_df.empty:

        lines.append("")

        lines.append(
            "No labelled state transitions available."
        )

    else:

        valid_delay = (
            transition_df[
                "delay_readings"
            ]
            .dropna()
        )

        if len(valid_delay):

            lines.append("")

            lines.append(
                "Mean temporal transition delay: "
                f"{valid_delay.mean():.2f} readings"
            )

            lines.append(
                "Median temporal transition delay: "
                f"{valid_delay.median():.2f} readings"
            )

    lines.append("")

    lines.append(
        "=" * 100
    )

    lines.append(
        "INTERPRETATION"
    )

    lines.append(
        "=" * 100
    )

    lines.append("")

    lines.append(
        "Machines marked NOT_TESTABLE are not "
        "considered failed machines."
    )

    lines.append(
        "They could not be evaluated because sufficient "
        "EMPTY and CUP labelled observations were not "
        "available for the preceding decision R&D stage."
    )

    with open(
        OUTPUT_DIR
        / "temporal_stability_rnd_summary.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(
                lines
            )
        )


# ============================================================
# 26. PRINT FINAL RESULTS
# ============================================================

def print_final_results(
    final_df,
    global_df
):

    print()
    print("=" * 120)
    print("FINAL TEMPORAL STABILITY RESULTS")
    print("=" * 120)

    if final_df.empty:

        print(
            "\nNo machine results available."
        )

        return

    print()

    for _, row in final_df.iterrows():

        machine = (
            row["machine_id"]
        )

        print("-" * 120)

        print(
            f"MACHINE: {machine}"
        )

        print(
            "Decision R&D status : "
            f"{row['decision_rnd_status']}"
        )

        print(
            "Temporal R&D status : "
            f"{row['temporal_rnd_status']}"
        )

        if (
            row[
                "temporal_rnd_status"
            ]
            == "TESTED"
        ):

            print(
                "Raw balanced acc.   : "
                f"{row['raw_balanced_accuracy']:.4f}"
            )

            print(
                "Best temporal       : "
                f"{row['best_temporal_strategy']}"
            )

            print(
                "Temporal bal. acc.  : "
                f"{row['temporal_balanced_accuracy']:.4f}"
            )

            print(
                "Change              : "
                f"{row['balanced_accuracy_change']:+.4f}"
            )

            print(
                "Effect              : "
                f"{row['temporal_effect']}"
            )

            print(
                "Runtime strategy    : "
                f"{row['recommended_runtime_strategy']}"
            )

        else:

            print(
                "Raw balanced acc.   : N/A"
            )

            print(
                "Best temporal       : N/A"
            )

            print(
                "Temporal bal. acc.  : N/A"
            )

            print(
                "Runtime strategy    : NOT_AVAILABLE"
            )

            print(
                "Reason              : "
                f"{row['reason']}"
            )

    if not global_df.empty:

        print()
        print("=" * 120)

        print(
            "BEST GLOBAL STRATEGY "
            "(TESTABLE MACHINES ONLY)"
        )

        print("=" * 120)

        print(
            "Strategy: "
            f"{global_df.iloc[0]['strategy']}"
        )

        print(
            "Macro machine balanced accuracy: "
            f"{global_df.iloc[0]['macro_machine_balanced_accuracy']:.4f}"
        )

    print()
    print("=" * 120)


# ============================================================
# 27. MAIN
# ============================================================

def main():

    create_directories()

    print()
    print("=" * 100)
    print(
        "06 - TEMPORAL STABILITY "
        "RESEARCH AND DEVELOPMENT"
    )
    print("=" * 100)

    print()
    print(
        "Expected machines:"
    )

    print(
        ", ".join(
            EXPECTED_MACHINES
        )
    )

    print()
    print(
        "Runtime mode:"
    )

    print(
        "TRAILING"
    )

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    predictions_df = (
        load_predictions()
    )

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    decision_df = (
        load_decision_config()
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    temporal_df = (
        run_temporal_strategies(
            predictions_df
        )
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    evaluation_df = (
        evaluate_machine_strategies(
            temporal_df
        )
    )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    evaluate_per_video(
        temporal_df
    )

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    best_df = (
        find_best_runtime_strategy(
            evaluation_df
        )
    )

    # --------------------------------------------------------
    # STEP 7
    # --------------------------------------------------------

    global_df = (
        evaluate_global_strategies(
            temporal_df
        )
    )

    # --------------------------------------------------------
    # STEP 8
    # --------------------------------------------------------

    compare_temporal_modes(
        evaluation_df
    )

    # --------------------------------------------------------
    # STEP 9
    # --------------------------------------------------------

    transition_df = (
        analyze_transition_delay(
            temporal_df,
            best_df
        )
    )

    # --------------------------------------------------------
    # STEP 10
    # --------------------------------------------------------

    final_df = (
        create_machine_summary(
            best_df,
            decision_df
        )
    )

    # --------------------------------------------------------
    # STEP 11
    # GRAPHS
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("STEP 11: GENERATE GRAPHS")
    print("=" * 100)

    for machine in EXPECTED_MACHINES:

        plot_machine_strategy_performance(
            machine,
            evaluation_df
        )

    plot_raw_vs_temporal(
        final_df
    )

    plot_temporal_improvement(
        final_df
    )

    plot_global_strategy_comparison(
        global_df
    )

    plot_best_strategy_timelines(
        temporal_df,
        final_df
    )

    # --------------------------------------------------------
    # STEP 12
    # JSON
    # --------------------------------------------------------

    save_temporal_config(
        final_df,
        global_df
    )

    # --------------------------------------------------------
    # STEP 13
    # SUMMARY
    # --------------------------------------------------------

    save_summary(
        final_df,
        global_df,
        transition_df
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print_final_results(
        final_df,
        global_df
    )

    print()
    print("=" * 100)
    print(
        "06 TEMPORAL STABILITY R&D COMPLETE"
    )
    print("=" * 100)

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

    print("-" * 100)

    important_files = [

        DIRS["audit"]
        / "machine_input_availability.csv",

        DIRS["predictions"]
        / "all_temporal_predictions.csv",

        DIRS["evaluation"]
        / "machine_strategy_evaluation.csv",

        DIRS["evaluation"]
        / "per_video_strategy_evaluation.csv",

        DIRS["evaluation"]
        / "transition_delay_analysis.csv",

        DIRS["comparison"]
        / "global_strategy_comparison.csv",

        DIRS["comparison"]
        / "trailing_vs_centered.csv",

        DIRS["final"]
        / "best_temporal_strategy_per_machine.csv",

        DIRS["final"]
        / "final_temporal_decisions.csv",

        DIRS["final"]
        / "temporal_stability_config.json",

        OUTPUT_DIR
        / "temporal_stability_rnd_summary.txt",
    ]

    for file in important_files:

        print(
            file
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()