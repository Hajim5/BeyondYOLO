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

RESULTS_DIR = (
    BASE_DIR
    / "results"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "07_final_system_integration"
)

# ------------------------------------------------------------
# Original machine / ROI configuration
# ------------------------------------------------------------

ORIGINAL_CONFIG_JSON = (
    BASE_DIR
    / "config.json"
)

# ------------------------------------------------------------
# 05 - Baseline + Decision R&D
# ------------------------------------------------------------

DECISION_CSV = (
    RESULTS_DIR
    / "05_baseline_decision_rnd"
    / "04_final_decision"
    / "final_machine_decisions.csv"
)

# ------------------------------------------------------------
# 06 - Temporal Stability R&D
# ------------------------------------------------------------

TEMPORAL_CSV = (
    RESULTS_DIR
    / "06_temporal_stability_rnd"
    / "04_final_temporal_config"
    / "final_temporal_decisions.csv"
)

TEMPORAL_JSON = (
    RESULTS_DIR
    / "06_temporal_stability_rnd"
    / "04_final_temporal_config"
    / "temporal_stability_config.json"
)

# ============================================================
# 2. EXPECTED MACHINES
# ============================================================

EXPECTED_MACHINES = ["OWN_INPUT"]

# ============================================================
# 3. SELECTED MATCHING METHOD
#
# Winner from Edge Matching R&D
# ============================================================

MATCHING_CONFIG = {"OWN_RESULT"}

# ============================================================
# 4. OUTPUT DIRECTORIES
# ============================================================

DIRS = {

    "audit":
        OUTPUT_DIR
        / "00_audit",

    "machine_config":
        OUTPUT_DIR
        / "01_machine_configuration",

    "validation":
        OUTPUT_DIR
        / "02_integration_validation",

    "graphs":
        OUTPUT_DIR
        / "03_graphs",

    "final":
        OUTPUT_DIR
        / "04_final_config",
}


def create_directories():

    for directory in DIRS.values():

        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# 5. HELPER
# ============================================================

def clean_machine_id(value):

    return (
        str(value)
        .upper()
        .strip()
    )


def safe_float(value):

    try:

        if pd.isna(value):

            return None

        return float(value)

    except Exception:

        return None


def safe_int(value):

    try:

        if pd.isna(value):

            return None

        return int(value)

    except Exception:

        return None


def safe_string(value):

    if value is None:

        return None

    try:

        if pd.isna(value):

            return None

    except Exception:

        pass

    value = str(value).strip()

    if value.lower() in [
        "",
        "nan",
        "none",
        "null",
    ]:

        return None

    return value


def safe_bool(value):

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        )
    ):

        return bool(value)

    if isinstance(
        value,
        str
    ):

        value = (
            value
            .strip()
            .lower()
        )

        if value in [
            "true",
            "1",
            "yes",
        ]:

            return True

        if value in [
            "false",
            "0",
            "no",
        ]:

            return False

    try:

        if pd.isna(value):

            return False

    except Exception:

        pass

    return bool(value)


# ============================================================
# 6. INPUT FILE AUDIT
# ============================================================

def audit_input_files():

    print()
    print("=" * 100)
    print("STEP 1: INPUT FILE AUDIT")
    print("=" * 100)

    files = {

        "original_config":
            ORIGINAL_CONFIG_JSON,

        "decision_config":
            DECISION_CSV,

        "temporal_config_csv":
            TEMPORAL_CSV,

        "temporal_config_json":
            TEMPORAL_JSON,
    }

    rows = []

    for name, path in files.items():

        exists = path.exists()

        rows.append({

            "input_name":
                name,

            "path":
                str(path),

            "exists":
                exists,
        })

        print(
            f"{name:<25} : "
            f"{'FOUND' if exists else 'MISSING'}"
        )

        print(
            f"    {path}"
        )

    audit_df = pd.DataFrame(
        rows
    )

    audit_df.to_csv(
        DIRS["audit"]
        / "input_files_audit.csv",
        index=False
    )

    # --------------------------------------------------------
    # Required inputs
    # --------------------------------------------------------

    required = [
        ORIGINAL_CONFIG_JSON,
        DECISION_CSV,
        TEMPORAL_CSV,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:

        print()
        print(
            "WARNING: Some required integration "
            "inputs are missing."
        )

        for path in missing:

            print(
                f"  - {path}"
            )

    return audit_df


# ============================================================
# 7. LOAD ORIGINAL CONFIG
# ============================================================

def load_original_config():

    print()
    print("=" * 100)
    print("STEP 2: LOAD MACHINE / ROI CONFIGURATION")
    print("=" * 100)

    if not ORIGINAL_CONFIG_JSON.exists():

        print()
        print(
            "Original config.json not found."
        )

        return {}

    with open(
        ORIGINAL_CONFIG_JSON,
        "r",
        encoding="utf-8"
    ) as file:

        config = json.load(
            file
        )

    print()
    print(
        "config.json loaded successfully."
    )

    return config


# ============================================================
# 8. EXTRACT MACHINE CONFIGURATIONS
#
# Supports the actual config.json structure:
#
# [
#     {
#         "id": "E1",
#         "class_name": "eversys",
#         "x1": 597,
#         "y1": 1304,
#         "x2": 895,
#         "y2": 1542
#     },
#     ...
# ]
# ============================================================

def extract_machine_configs(config):

    machine_configs = {}

    # ========================================================
    # FORMAT 1
    # YOUR CURRENT CONFIG FORMAT
    #
    # [
    #     {
    #         "id": "E1",
    #         "class_name": "eversys",
    #         "x1": ...,
    #         "y1": ...,
    #         "x2": ...,
    #         "y2": ...
    #     }
    # ]
    # ========================================================

    if isinstance(config, list):

        for item in config:

            if not isinstance(item, dict):
                continue

            machine = (
                item.get("id")
                or
                item.get("machine_id")
                or
                item.get("name")
            )

            if machine is None:
                continue

            machine = clean_machine_id(
                machine
            )

            machine_configs[
                machine
            ] = item

        return machine_configs

    # ========================================================
    # OTHER POSSIBLE FORMATS
    # ========================================================

    if not isinstance(config, dict):

        return machine_configs

    # --------------------------------------------------------
    # FORMAT 2
    #
    # {
    #     "machines": {
    #         "E1": {...}
    #     }
    # }
    # --------------------------------------------------------

    if (
        "machines" in config
        and
        isinstance(
            config["machines"],
            dict
        )
    ):

        for machine, data in (
            config["machines"].items()
        ):

            machine_configs[
                clean_machine_id(
                    machine
                )
            ] = data

        return machine_configs

    # --------------------------------------------------------
    # FORMAT 3
    #
    # {
    #     "machines": [
    #         {
    #             "id": "E1",
    #             ...
    #         }
    #     ]
    # }
    # --------------------------------------------------------

    if (
        "machines" in config
        and
        isinstance(
            config["machines"],
            list
        )
    ):

        for item in config["machines"]:

            if not isinstance(item, dict):
                continue

            machine = (
                item.get("id")
                or
                item.get("machine_id")
                or
                item.get("name")
            )

            if machine is None:
                continue

            machine_configs[
                clean_machine_id(
                    machine
                )
            ] = item

        return machine_configs

    # --------------------------------------------------------
    # FORMAT 4
    #
    # {
    #     "E1": {...},
    #     "E2": {...}
    # }
    # --------------------------------------------------------

    for machine in EXPECTED_MACHINES:

        if machine in config:

            machine_configs[
                machine
            ] = config[
                machine
            ]

    return machine_configs


# ============================================================
# 9. LOAD DECISION RESULTS
# ============================================================

def load_decision_results():

    print()
    print("=" * 100)
    print("STEP 3: LOAD 05 DECISION RESULTS")
    print("=" * 100)

    if not DECISION_CSV.exists():

        print()
        print(
            "05 decision CSV not found."
        )

        return pd.DataFrame()

    df = pd.read_csv(
        DECISION_CSV
    )

    if "machine_id" not in df.columns:

        raise ValueError(
            "05 decision CSV does not contain "
            "'machine_id'."
        )

    df["machine_id"] = (
        df["machine_id"]
        .apply(
            clean_machine_id
        )
    )

    print()
    print(
        f"Decision rows loaded: {len(df)}"
    )

    print()
    print(
        "Machines available:"
    )

    print(
        ", ".join(
            sorted(
                df[
                    "machine_id"
                ].unique()
            )
        )
    )

    return df


# ============================================================
# 10. LOAD TEMPORAL RESULTS
# ============================================================

def load_temporal_results():

    print()
    print("=" * 100)
    print("STEP 4: LOAD 06 TEMPORAL RESULTS")
    print("=" * 100)

    if not TEMPORAL_CSV.exists():

        print()
        print(
            "06 temporal CSV not found."
        )

        return pd.DataFrame()

    df = pd.read_csv(
        TEMPORAL_CSV
    )

    if "machine_id" not in df.columns:

        raise ValueError(
            "06 temporal CSV does not contain "
            "'machine_id'."
        )

    df["machine_id"] = (
        df["machine_id"]
        .apply(
            clean_machine_id
        )
    )

    print()
    print(
        f"Temporal rows loaded: {len(df)}"
    )

    print()
    print(
        df[
            [
                column
                for column in [
                    "machine_id",
                    "decision_rnd_status",
                    "temporal_rnd_status",
                    "recommended_runtime_strategy",
                ]
                if column in df.columns
            ]
        ].to_string(
            index=False
        )
    )

    return df


# ============================================================
# 11. EXTRACT ROI
# ============================================================

def extract_roi(
    machine_config
):

    if not isinstance(
        machine_config,
        dict
    ):

        return None

    # --------------------------------------------------------
    # Direct ROI
    # --------------------------------------------------------

    for key in [
        "roi",
        "bbox",
        "bounding_box",
        "coordinates",
    ]:

        if key in machine_config:

            return machine_config[
                key
            ]

    # --------------------------------------------------------
    # x1 y1 x2 y2
    # --------------------------------------------------------

    keys = set(
        machine_config.keys()
    )

    if {
        "x1",
        "y1",
        "x2",
        "y2",
    }.issubset(
        keys
    ):

        return {

            "x1":
                machine_config[
                    "x1"
                ],

            "y1":
                machine_config[
                    "y1"
                ],

            "x2":
                machine_config[
                    "x2"
                ],

            "y2":
                machine_config[
                    "y2"
                ],
        }

    return None


# ============================================================
# 12. EXTRACT TEMPLATE INFORMATION
# ============================================================

def extract_template_info(
    machine_config
):

    if not isinstance(
        machine_config,
        dict
    ):

        return None

    for key in [
        "template",
        "template_path",
        "cup_template",
        "cup_template_path",
        "reference",
        "reference_image",
    ]:

        if key in machine_config:

            return machine_config[
                key
            ]

    return None


# ============================================================
# 13. GET DECISION ROW
# ============================================================

def get_decision_row(
    machine,
    decision_df
):

    if decision_df.empty:

        return None

    subset = decision_df[
        decision_df[
            "machine_id"
        ]
        == machine
    ]

    if subset.empty:

        return None

    return subset.iloc[0]


# ============================================================
# 14. GET TEMPORAL ROW
# ============================================================

def get_temporal_row(
    machine,
    temporal_df
):

    if temporal_df.empty:

        return None

    subset = temporal_df[
        temporal_df[
            "machine_id"
        ]
        == machine
    ]

    if subset.empty:

        return None

    return subset.iloc[0]


# ============================================================
# 15. BUILD MACHINE CONFIGURATION
# ============================================================

def build_machine_configurations(
    machine_configs,
    decision_df,
    temporal_df
):

    print()
    print("=" * 100)
    print("STEP 5: BUILD INTEGRATED MACHINE CONFIGURATION")
    print("=" * 100)

    rows = []

    full_configs = {}

    for machine in EXPECTED_MACHINES:

        print()
        print("-" * 100)
        print(
            f"MACHINE: {machine}"
        )

        original = (
            machine_configs.get(
                machine,
                {}
            )
        )

        roi = extract_roi(
            original
        )

        template_info = (
            extract_template_info(
                original
            )
        )

        decision = get_decision_row(
            machine,
            decision_df
        )

        temporal = get_temporal_row(
            machine,
            temporal_df
        )

        # ====================================================
        # ROI STATUS
        # ====================================================

        roi_available = (
            roi is not None
        )

        # ====================================================
        # DECISION CONFIG
        # ====================================================

        empty_baseline = None
        direction = None
        delta_threshold = None
        decision_quality = None

        decision_available = False

        if decision is not None:

            empty_baseline = (
                safe_float(
                    decision.get(
                        "empty_baseline",
                        np.nan
                    )
                )
            )

            direction = (
                safe_string(
                    decision.get(
                        "direction",
                        None
                    )
                )
            )

            delta_threshold = (
                safe_float(
                    decision.get(
                        "best_delta_threshold",
                        np.nan
                    )
                )
            )

            decision_quality = (
                safe_string(
                    decision.get(
                        "decision_quality",
                        None
                    )
                )
            )

            decision_available = (
                empty_baseline is not None
                and
                direction is not None
                and
                delta_threshold is not None
            )

        # ====================================================
        # TEMPORAL CONFIG
        # ====================================================

        temporal_status = None
        temporal_strategy = None
        window_size = None
        votes_required = None

        temporal_available = False

        if temporal is not None:

            temporal_status = (
                safe_string(
                    temporal.get(
                        "temporal_rnd_status",
                        None
                    )
                )
            )

            temporal_strategy = (
                safe_string(
                    temporal.get(
                        "recommended_runtime_strategy",
                        None
                    )
                )
            )

            # ------------------------------------------------
            # Use selected temporal parameters
            # ------------------------------------------------

            if (
                temporal_strategy
                and
                temporal_strategy
                != "NOT_AVAILABLE"
            ):

                if (
                    temporal_strategy
                    == "raw_1_of_1"
                ):

                    window_size = 1
                    votes_required = 1

                else:

                    window_size = (
                        safe_int(
                            temporal.get(
                                "window_size",
                                np.nan
                            )
                        )
                    )

                    votes_required = (
                        safe_int(
                            temporal.get(
                                "votes_required",
                                np.nan
                            )
                        )
                    )

                temporal_available = (
                    window_size is not None
                    and
                    votes_required is not None
                )

        # ====================================================
        # CONFIGURATION READINESS
        # ====================================================

        missing_components = []

        if not roi_available:

            missing_components.append(
                "ROI"
            )

        if not decision_available:

            missing_components.append(
                "DECISION_PARAMETERS"
            )

        if not temporal_available:

            missing_components.append(
                "TEMPORAL_PARAMETERS"
            )

        # ----------------------------------------------------
        # Matching configuration is global and known
        # ----------------------------------------------------

        matching_available = True

        # ----------------------------------------------------
        # Runtime readiness
        # ----------------------------------------------------

        runtime_ready = (
            roi_available
            and
            matching_available
            and
            decision_available
            and
            temporal_available
        )

        if runtime_ready:

            readiness_status = (
                "READY"
            )

            reason = (
                "All required runtime parameters "
                "are available."
            )

        else:

            readiness_status = (
                "NOT_READY"
            )

            reason = (
                "Missing: "
                +
                ", ".join(
                    missing_components
                )
            )

        # ====================================================
        # MACHINE CONFIG
        # ====================================================

        machine_config = {

            "machine_id":
                machine,

            # ------------------------------------------------
            # Original setup
            # ------------------------------------------------

            "roi":
                roi,

            "template_reference":
                template_info,

            # ------------------------------------------------
            # Selected matching method
            # ------------------------------------------------

            "matching": {

                "method":
                    MATCHING_CONFIG[
                        "method"
                    ],

                "variant":
                    MATCHING_CONFIG[
                        "variant"
                    ],

                "image_representation":
                    MATCHING_CONFIG[
                        "image_representation"
                    ],

                "preprocessing":
                    MATCHING_CONFIG[
                        "preprocessing"
                    ],

                "scale":
                    MATCHING_CONFIG[
                        "scale"
                    ],

                "canny_low":
                    MATCHING_CONFIG[
                        "canny_low"
                    ],

                "canny_high":
                    MATCHING_CONFIG[
                        "canny_high"
                    ],

                "template_matching_method":
                    MATCHING_CONFIG[
                        "template_matching_method"
                    ],
            },

            # ------------------------------------------------
            # Decision
            # ------------------------------------------------

            "decision": {

                "empty_baseline":
                    empty_baseline,

                "direction":
                    direction,

                "delta_threshold":
                    delta_threshold,

                "decision_quality":
                    decision_quality,
            },

            # ------------------------------------------------
            # Temporal
            # ------------------------------------------------

            "temporal": {

                "rnd_status":
                    temporal_status,

                "strategy":
                    temporal_strategy,

                "window_size":
                    window_size,

                "votes_required":
                    votes_required,

                "mode":
                    "trailing",
            },

            # ------------------------------------------------
            # Runtime
            # ------------------------------------------------

            "runtime": {

                "ready":
                    runtime_ready,

                "status":
                    readiness_status,

                "missing_components":
                    missing_components,

                "reason":
                    reason,
            },
        }

        full_configs[
            machine
        ] = machine_config

        # ====================================================
        # CSV ROW
        # ====================================================

        rows.append({

            "machine_id":
                machine,

            "roi_available":
                roi_available,

            "template_reference_available":
                (
                    template_info
                    is not None
                ),

            "matching_method":
                MATCHING_CONFIG[
                    "method"
                ],

            "matching_variant":
                MATCHING_CONFIG[
                    "variant"
                ],

            "image_representation":
                MATCHING_CONFIG[
                    "image_representation"
                ],

            "preprocessing":
                MATCHING_CONFIG[
                    "preprocessing"
                ],

            "scale":
                MATCHING_CONFIG[
                    "scale"
                ],

            "canny_low":
                MATCHING_CONFIG[
                    "canny_low"
                ],

            "canny_high":
                MATCHING_CONFIG[
                    "canny_high"
                ],

            "empty_baseline":
                empty_baseline,

            "direction":
                direction,

            "delta_threshold":
                delta_threshold,

            "decision_quality":
                decision_quality,

            "decision_available":
                decision_available,

            "temporal_rnd_status":
                temporal_status,

            "temporal_strategy":
                temporal_strategy,

            "window_size":
                window_size,

            "votes_required":
                votes_required,

            "temporal_available":
                temporal_available,

            "runtime_ready":
                runtime_ready,

            "readiness_status":
                readiness_status,

            "missing_components":
                (
                    "; ".join(
                        missing_components
                    )
                ),

            "reason":
                reason,
        })

        print(
            f"ROI                 : "
            f"{'OK' if roi_available else 'MISSING'}"
        )

        print(
            f"Decision parameters : "
            f"{'OK' if decision_available else 'MISSING'}"
        )

        print(
            f"Temporal parameters : "
            f"{'OK' if temporal_available else 'MISSING'}"
        )

        print(
            f"Runtime ready       : "
            f"{runtime_ready}"
        )

        if temporal_strategy:

            print(
                f"Temporal strategy   : "
                f"{temporal_strategy}"
            )

    result_df = pd.DataFrame(
        rows
    )

    result_df.to_csv(
        DIRS["machine_config"]
        / "all_machine_configuration.csv",
        index=False
    )

    return (
        result_df,
        full_configs
    )


# ============================================================
# 16. VALIDATE CONFIGURATION
# ============================================================

def validate_configuration(
    config_df
):

    print()
    print("=" * 100)
    print("STEP 6: VALIDATE INTEGRATED CONFIGURATION")
    print("=" * 100)

    validation_rows = []

    for _, row in config_df.iterrows():

        machine = (
            row["machine_id"]
        )

        checks = {

            "roi":
                bool(
                    row[
                        "roi_available"
                    ]
                ),

            "matching_method":
                bool(
                    row[
                        "matching_method"
                    ]
                ),

            "baseline":
                pd.notna(
                    row[
                        "empty_baseline"
                    ]
                ),

            "direction":
                pd.notna(
                    row[
                        "direction"
                    ]
                ),

            "threshold":
                pd.notna(
                    row[
                        "delta_threshold"
                    ]
                ),

            "temporal_strategy":
                pd.notna(
                    row[
                        "temporal_strategy"
                    ]
                )
                and
                row[
                    "temporal_strategy"
                ]
                != "NOT_AVAILABLE",

            "window_size":
                pd.notna(
                    row[
                        "window_size"
                    ]
                ),

            "votes_required":
                pd.notna(
                    row[
                        "votes_required"
                    ]
                ),
        }

        for component, passed in (
            checks.items()
        ):

            validation_rows.append({

                "machine_id":
                    machine,

                "component":
                    component,

                "passed":
                    passed,

                "status":
                    (
                        "PASS"
                        if passed
                        else "FAIL"
                    ),
            })

    validation_df = pd.DataFrame(
        validation_rows
    )

    validation_df.to_csv(
        DIRS["validation"]
        / "validation_results.csv",
        index=False
    )

    return validation_df


# ============================================================
# 17. MACHINE READINESS
# ============================================================

def create_machine_readiness(
    config_df
):

    print()
    print("=" * 100)
    print("STEP 7: MACHINE READINESS")
    print("=" * 100)

    readiness = config_df[
        [
            "machine_id",
            "roi_available",
            "decision_available",
            "temporal_available",
            "runtime_ready",
            "readiness_status",
            "missing_components",
        ]
    ].copy()

    readiness.to_csv(
        DIRS["validation"]
        / "machine_readiness.csv",
        index=False
    )

    print()

    print(
        readiness.to_string(
            index=False
        )
    )

    return readiness


# ============================================================
# 18. MISSING CONFIGURATION
# ============================================================

def create_missing_configuration(
    config_df
):

    missing = config_df[
        config_df[
            "runtime_ready"
        ]
        == False
    ].copy()

    columns = [
        "machine_id",
        "roi_available",
        "decision_available",
        "temporal_available",
        "missing_components",
        "reason",
    ]

    missing[
        columns
    ].to_csv(
        DIRS["audit"]
        / "missing_configuration.csv",
        index=False
    )

    return missing


# ============================================================
# 19. MACHINE PIPELINE STATUS GRAPH
# ============================================================

def plot_machine_pipeline_status(
    config_df
):

    machines = (
        config_df[
            "machine_id"
        ].tolist()
    )

    roi = (
        config_df[
            "roi_available"
        ]
        .astype(int)
        .to_numpy()
    )

    decision = (
        config_df[
            "decision_available"
        ]
        .astype(int)
        .to_numpy()
    )

    temporal = (
        config_df[
            "temporal_available"
        ]
        .astype(int)
        .to_numpy()
    )

    x = np.arange(
        len(machines)
    )

    width = 0.25

    plt.figure(
        figsize=(14, 6)
    )

    plt.bar(
        x - width,
        roi,
        width,
        label="ROI"
    )

    plt.bar(
        x,
        decision,
        width,
        label="Decision"
    )

    plt.bar(
        x + width,
        temporal,
        width,
        label="Temporal"
    )

    plt.xticks(
        x,
        machines
    )

    plt.yticks(
        [
            0,
            1,
        ],
        [
            "Missing",
            "Available",
        ]
    )

    plt.ylim(
        0,
        1.2
    )

    plt.xlabel(
        "Machine"
    )

    plt.ylabel(
        "Configuration Status"
    )

    plt.title(
        "Final Integration - Machine Pipeline Status"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        DIRS["graphs"]
        / "machine_pipeline_status.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 20. TEMPORAL STRATEGY GRAPH
# ============================================================

def plot_temporal_strategy_summary(
    config_df
):

    subset = config_df[
        config_df[
            "temporal_available"
        ]
        == True
    ].copy()

    if subset.empty:

        return

    strategy_counts = (
        subset[
            "temporal_strategy"
        ]
        .value_counts()
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        strategy_counts.index,
        strategy_counts.values
    )

    plt.xlabel(
        "Temporal Strategy"
    )

    plt.ylabel(
        "Number of Machines"
    )

    plt.title(
        "Selected Temporal Strategy per Machine"
    )

    plt.tight_layout()

    plt.savefig(
        DIRS["graphs"]
        / "temporal_strategy_summary.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 21. SAVE FINAL CSV
# ============================================================

def save_final_csv(
    config_df
):

    config_df.to_csv(
        DIRS["final"]
        / "final_system_config.csv",
        index=False
    )


# ============================================================
# 22. SAVE FINAL JSON
# ============================================================

def save_final_json(
    full_configs
):

    ready_machines = []

    not_ready_machines = []

    for machine, config in (
        full_configs.items()
    ):

        if config[
            "runtime"
        ][
            "ready"
        ]:

            ready_machines.append(
                machine
            )

        else:

            not_ready_machines.append(
                machine
            )

    final_config = {

        "system":
            {

                "name":
                    "Cup Presence Detection System",

                "output":
                    {

                        "cup":
                            True,

                        "empty":
                            False,
                    },

                "matching_method":
                    MATCHING_CONFIG,

                "decision_method":
                    (
                        "Per-machine EMPTY baseline "
                        "+ baseline-relative delta threshold"
                    ),

                "temporal_mode":
                    "trailing",
            },

        "integration":
            {

                "expected_machines":
                    EXPECTED_MACHINES,

                "ready_machines":
                    ready_machines,

                "not_ready_machines":
                    not_ready_machines,

                "ready_count":
                    len(
                        ready_machines
                    ),

                "not_ready_count":
                    len(
                        not_ready_machines
                    ),
            },

        "machines":
            full_configs,
    }

    with open(
        DIRS["final"]
        / "final_system_config.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            final_config,
            file,
            indent=4
        )

    return final_config


# ============================================================
# 23. SAVE INTEGRATION SUMMARY
# ============================================================

def save_integration_summary(
    config_df,
    missing_df
):

    ready = config_df[
        config_df[
            "runtime_ready"
        ]
        == True
    ]

    not_ready = config_df[
        config_df[
            "runtime_ready"
        ]
        == False
    ]

    lines = []

    lines.append(
        "07 - FINAL SYSTEM CONFIGURATION / INTEGRATION"
    )

    lines.append(
        "=" * 100
    )

    lines.append("")

    lines.append(
        "PURPOSE"
    )

    lines.append(
        "-" * 100
    )

    lines.append(
        "Combine the selected outputs from the "
        "previous R&D stages into one final "
        "runtime-ready configuration."
    )

    lines.append("")

    lines.append(
        "SELECTED MATCHING METHOD"
    )

    lines.append(
        "-" * 100
    )

    lines.append(
        "Method: Edge Template Matching"
    )

    lines.append(
        "Variant: "
        "gray_gaussian_scale_1.00_canny_75_175"
    )

    lines.append(
        "Image representation: Grayscale"
    )

    lines.append(
        "Preprocessing: Gaussian Blur"
    )

    lines.append(
        "Scale: 1.00"
    )

    lines.append(
        "Canny thresholds: 75 / 175"
    )

    lines.append(
        "Template matching: TM_CCOEFF_NORMED"
    )

    lines.append("")

    lines.append(
        "DECISION METHOD"
    )

    lines.append(
        "-" * 100
    )

    lines.append(
        "Per-machine EMPTY baseline"
    )

    lines.append(
        "+"
    )

    lines.append(
        "Baseline-relative similarity difference"
    )

    lines.append(
        "+"
    )

    lines.append(
        "Per-machine learned direction and threshold"
    )

    lines.append("")

    lines.append(
        "TEMPORAL METHOD"
    )

    lines.append(
        "-" * 100
    )

    lines.append(
        "Per-machine strategy selected from "
        "06 Temporal Stability R&D."
    )

    lines.append(
        "Runtime implementation uses trailing "
        "temporal voting."
    )

    lines.append("")

    lines.append(
        "MACHINE READINESS"
    )

    lines.append(
        "-" * 100
    )

    lines.append(
        f"Expected machines: "
        f"{len(EXPECTED_MACHINES)}"
    )

    lines.append(
        f"Runtime-ready machines: "
        f"{len(ready)}"
    )

    lines.append(
        f"Not-ready machines: "
        f"{len(not_ready)}"
    )

    lines.append("")

    for _, row in (
        config_df.iterrows()
    ):

        lines.append(
            f"{row['machine_id']}: "
            f"{row['readiness_status']}"
        )

        if not row[
            "runtime_ready"
        ]:

            lines.append(
                f"    Missing: "
                f"{row['missing_components']}"
            )

    lines.append("")

    lines.append(
        "FINAL INTEGRATED PIPELINE"
    )

    lines.append(
        "-" * 100
    )

    lines.append(
        "Video / Camera Frame"
    )

    lines.append(
        "    -> Machine ROI"
    )

    lines.append(
        "    -> Grayscale"
    )

    lines.append(
        "    -> Gaussian Blur"
    )

    lines.append(
        "    -> Canny Edge Detection (75, 175)"
    )

    lines.append(
        "    -> Edge Template Matching"
    )

    lines.append(
        "    -> Similarity Score"
    )

    lines.append(
        "    -> EMPTY Baseline"
    )

    lines.append(
        "    -> Baseline-relative Delta"
    )

    lines.append(
        "    -> Machine-specific Direction + Threshold"
    )

    lines.append(
        "    -> Raw CUP / EMPTY"
    )

    lines.append(
        "    -> Temporal Confirmation"
    )

    lines.append(
        "    -> Final CUP / EMPTY"
    )

    lines.append("")

    lines.append(
        "IMPORTANT"
    )

    lines.append(
        "-" * 100
    )

    lines.append(
        "Machines marked NOT_READY are not "
        "considered failed."
    )

    lines.append(
        "They are missing one or more required "
        "configuration parameters from the "
        "preceding R&D stages."
    )

    with open(
        DIRS["final"]
        / "integration_summary.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(
                lines
            )
        )


# ============================================================
# 24. PRINT FINAL RESULT
# ============================================================

def print_final_result(
    config_df
):

    print()
    print("=" * 120)
    print("FINAL SYSTEM INTEGRATION RESULT")
    print("=" * 120)

    print()

    display = config_df[
        [
            "machine_id",
            "roi_available",
            "decision_available",
            "temporal_strategy",
            "runtime_ready",
            "readiness_status",
        ]
    ]

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print("-" * 120)

    ready = config_df[
        config_df[
            "runtime_ready"
        ]
        == True
    ][
        "machine_id"
    ].tolist()

    not_ready = config_df[
        config_df[
            "runtime_ready"
        ]
        == False
    ][
        "machine_id"
    ].tolist()

    print()
    print(
        f"READY ({len(ready)}):"
    )

    if ready:

        print(
            ", ".join(
                ready
            )
        )

    else:

        print(
            "None"
        )

    print()
    print(
        f"NOT READY ({len(not_ready)}):"
    )

    if not_ready:

        print(
            ", ".join(
                not_ready
            )
        )

    else:

        print(
            "None"
        )

    print()
    print("=" * 120)


# ============================================================
# 25. MAIN
# ============================================================

def main():

    create_directories()

    print()
    print("=" * 100)
    print(
        "07 - FINAL SYSTEM CONFIGURATION "
        "/ INTEGRATION R&D"
    )
    print("=" * 100)

    print()
    print(
        "Selected matching winner:"
    )

    print(
        MATCHING_CONFIG[
            "variant"
        ]
    )

    # ========================================================
    # STEP 1
    # ========================================================

    audit_input_files()

    # ========================================================
    # STEP 2
    # ========================================================

    original_config = (
        load_original_config()
    )

    machine_configs = (
        extract_machine_configs(
            original_config
        )
    )

    print()
    print(
        "Machines found in original config:"
    )

    if machine_configs:

        print(
            ", ".join(
                sorted(
                    machine_configs.keys()
                )
            )
        )

    else:

        print(
            "None detected."
        )

    # ========================================================
    # STEP 3
    # ========================================================

    decision_df = (
        load_decision_results()
    )

    # ========================================================
    # STEP 4
    # ========================================================

    temporal_df = (
        load_temporal_results()
    )

    # ========================================================
    # STEP 5
    # ========================================================

    (
        config_df,
        full_configs
    ) = build_machine_configurations(

        machine_configs,
        decision_df,
        temporal_df
    )

    # ========================================================
    # STEP 6
    # ========================================================

    validate_configuration(
        config_df
    )

    # ========================================================
    # STEP 7
    # ========================================================

    create_machine_readiness(
        config_df
    )

    # ========================================================
    # STEP 8
    # ========================================================

    missing_df = (
        create_missing_configuration(
            config_df
        )
    )

    # ========================================================
    # STEP 9
    # ========================================================

    print()
    print("=" * 100)
    print("STEP 8: GENERATE GRAPHS")
    print("=" * 100)

    plot_machine_pipeline_status(
        config_df
    )

    plot_temporal_strategy_summary(
        config_df
    )

    # ========================================================
    # STEP 10
    # ========================================================

    save_final_csv(
        config_df
    )

    # ========================================================
    # STEP 11
    # ========================================================

    save_final_json(
        full_configs
    )

    # ========================================================
    # STEP 12
    # ========================================================

    save_integration_summary(
        config_df,
        missing_df
    )

    # ========================================================
    # FINAL
    # ========================================================

    print_final_result(
        config_df
    )

    print()
    print("=" * 100)
    print(
        "07 FINAL SYSTEM INTEGRATION COMPLETE"
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
        / "input_files_audit.csv",

        DIRS["audit"]
        / "missing_configuration.csv",

        DIRS["machine_config"]
        / "all_machine_configuration.csv",

        DIRS["validation"]
        / "validation_results.csv",

        DIRS["validation"]
        / "machine_readiness.csv",

        DIRS["final"]
        / "final_system_config.csv",

        DIRS["final"]
        / "final_system_config.json",

        DIRS["final"]
        / "integration_summary.txt",
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
