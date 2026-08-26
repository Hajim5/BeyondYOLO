"""
2.2_create_baseline.py

GENERAL BASELINE CREATION MODULE

Purpose:
    Learn the normal reference state from EMPTY / normal data.

Pipeline:

    EMPTY images or video
            ↓
    2.1_template_matching.py
            ↓
    Collect matching scores
            ↓
    Remove invalid scores
            ↓
    Optional outlier filtering
            ↓
    Calculate baseline
            ↓
    Save baseline results

This script does NOT make the final object decision.

The baseline can later be used by:

    2.3_baseline_decision.py
"""

from pathlib import Path
import json
import statistics
import sys


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_DIR / "config.json"

OUTPUT_DIR = PROJECT_DIR / "output"

BASELINE_OUTPUT_PATH = (
    OUTPUT_DIR
    / "baseline.json"
)


# ============================================================
# IMPORT TEMPLATE MATCHING MODULE
# ============================================================

MODULE_DIR = Path(__file__).resolve().parent

if str(MODULE_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(MODULE_DIR)
    )


# IMPORTANT:
#
# Python filenames beginning with numbers cannot be imported
# normally using:
#
# import 2.1_template_matching
#
# Therefore we load the file dynamically.


import importlib.util


TEMPLATE_MATCHING_PATH = (
    MODULE_DIR
    / "2.1_template_matching.py"
)


def load_template_matching_module():

    if not TEMPLATE_MATCHING_PATH.exists():

        raise FileNotFoundError(
            "\n2.1_template_matching.py not found:\n"
            f"{TEMPLATE_MATCHING_PATH}\n"
        )

    spec = importlib.util.spec_from_file_location(
        "template_matching",
        TEMPLATE_MATCHING_PATH
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


template_matching = (
    load_template_matching_module()
)


# ============================================================
# LOAD CONFIG
# ============================================================

def load_config(
    config_path=CONFIG_PATH
):

    config_path = Path(
        config_path
    )

    if not config_path.exists():

        raise FileNotFoundError(
            "\nConfig file not found:\n"
            f"{config_path}\n"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


# ============================================================
# VALIDATE MACHINE
# ============================================================

def get_machine_config(
    config,
    machine_id
):

    machines = config.get(
        "machines"
    )

    if machines is None:

        raise KeyError(
            "\n'machines' not found "
            "in config.json\n"
        )

    if machine_id not in machines:

        raise KeyError(
            "\nMachine not found:\n"
            f"{machine_id}\n"
        )

    return machines[
        machine_id
    ]


# ============================================================
# GET IMAGE FILES
# ============================================================

def get_image_files(
    input_dir
):

    input_dir = Path(
        input_dir
    )

    if not input_dir.exists():

        raise FileNotFoundError(
            "\nInput directory not found:\n"
            f"{input_dir}\n"
        )

    extensions = {

        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"

    }

    image_files = []

    for path in input_dir.rglob("*"):

        if (
            path.is_file()
            and
            path.suffix.lower()
            in extensions
        ):

            image_files.append(
                path
            )

    return sorted(
        image_files
    )


# ============================================================
# CALCULATE MEDIAN
# ============================================================

def calculate_median(
    scores
):

    if not scores:

        return None

    return float(
        statistics.median(
            scores
        )
    )


# ============================================================
# CALCULATE MEAN
# ============================================================

def calculate_mean(
    scores
):

    if not scores:

        return None

    return float(
        statistics.mean(
            scores
        )
    )


# ============================================================
# CALCULATE STANDARD DEVIATION
# ============================================================

def calculate_std(
    scores
):

    if len(scores) < 2:

        return 0.0

    return float(
        statistics.stdev(
            scores
        )
    )


# ============================================================
# REMOVE OUTLIERS USING IQR
# ============================================================

def remove_outliers_iqr(
    scores,
    multiplier=1.5
):

    if len(scores) < 4:

        return scores.copy()

    sorted_scores = sorted(
        scores
    )

    count = len(
        sorted_scores
    )

    q1_index = (
        (count - 1)
        * 0.25
    )

    q3_index = (
        (count - 1)
        * 0.75
    )

    def interpolate(
        values,
        index
    ):

        lower = int(index)

        upper = min(
            lower + 1,
            len(values) - 1
        )

        fraction = (
            index - lower
        )

        return (

            values[lower]
            +
            (
                values[upper]
                - values[lower]
            )
            * fraction

        )

    q1 = interpolate(
        sorted_scores,
        q1_index
    )

    q3 = interpolate(
        sorted_scores,
        q3_index
    )

    iqr = (
        q3 - q1
    )

    lower_bound = (
        q1
        -
        multiplier * iqr
    )

    upper_bound = (
        q3
        +
        multiplier * iqr
    )

    filtered_scores = [

        score

        for score in scores

        if (
            lower_bound
            <= score
            <= upper_bound
        )

    ]

    return filtered_scores


# ============================================================
# CALCULATE BASELINE STATISTICS
# ============================================================

def calculate_baseline_statistics(
    scores,
    remove_outliers=True
):

    if not scores:

        return {

            "baseline": None,

            "raw_samples": 0,

            "clean_samples": 0,

            "median": None,

            "mean": None,

            "std": None

        }

    raw_scores = scores.copy()

    if remove_outliers:

        clean_scores = remove_outliers_iqr(
            raw_scores
        )

    else:

        clean_scores = raw_scores.copy()

    if not clean_scores:

        clean_scores = raw_scores.copy()

    median_score = calculate_median(
        clean_scores
    )

    mean_score = calculate_mean(
        clean_scores
    )

    std_score = calculate_std(
        clean_scores
    )

    return {

        # Main baseline
        "baseline": median_score,

        # Statistics
        "median": median_score,

        "mean": mean_score,

        "std": std_score,

        # Sample counts
        "raw_samples": len(
            raw_scores
        ),

        "clean_samples": len(
            clean_scores
        ),

        "outliers_removed": (
            len(raw_scores)
            -
            len(clean_scores)
        )

    }


# ============================================================
# PROCESS SINGLE IMAGE
# ============================================================

def process_image(
    image_path,
    machine_id,
    config,
    template_path
):

    frame = (
        template_matching.load_image(
            image_path
        )
    )

    machine_config = (
        get_machine_config(
            config,
            machine_id
        )
    )

    settings = config.get(
        "template_matching",
        template_matching.DEFAULT_SETTINGS
    )

    template = (
        template_matching.load_template(
            template_path
        )
    )

    template_processed = (
        template_matching.prepare_template(
            template,
            settings
        )
    )

    result = (
        template_matching.match_machine(

            frame=frame,

            machine_id=machine_id,

            machine_config=machine_config,

            template_edges=template_processed,

            settings=settings

        )
    )

    return result


# ============================================================
# CREATE BASELINE FOR ONE MACHINE
# ============================================================

def create_baseline(
    input_dir,
    machine_id,
    template_path,
    config_path=CONFIG_PATH,
    remove_outliers=True
):

    print()

    print("=" * 70)

    print(
        "CREATE EMPTY BASELINE"
    )

    print("=" * 70)

    print()

    print(
        f"Machine ID : {machine_id}"
    )

    print(
        f"Input      : {input_dir}"
    )

    print()

    # --------------------------------------------------------
    # LOAD CONFIG
    # --------------------------------------------------------

    config = load_config(
        config_path
    )

    # Validate machine
    get_machine_config(
        config,
        machine_id
    )

    # --------------------------------------------------------
    # GET INPUT FILES
    # --------------------------------------------------------

    image_files = get_image_files(
        input_dir
    )

    if not image_files:

        raise RuntimeError(
            "\nNo images found "
            "in input directory.\n"
        )

    print(
        f"Images found: "
        f"{len(image_files)}"
    )

    print()

    # --------------------------------------------------------
    # PROCESS IMAGES
    # --------------------------------------------------------

    scores = []

    failed_files = []

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        try:

            result = process_image(

                image_path=image_path,

                machine_id=machine_id,

                config=config,

                template_path=template_path

            )

            score = result.get(
                "score"
            )

            if score is not None:

                scores.append(
                    float(score)
                )

                print(
                    f"[{index}/{len(image_files)}] "
                    f"{image_path.name} "
                    f"Score: {score:.6f}"
                )

            else:

                failed_files.append(
                    str(image_path)
                )

                print(
                    f"[{index}/{len(image_files)}] "
                    f"{image_path.name} "
                    f"INVALID SCORE"
                )

        except Exception as error:

            failed_files.append(
                str(image_path)
            )

            print(
                f"[{index}/{len(image_files)}] "
                f"{image_path.name} "
                f"FAILED"
            )

            print(
                f"Error: {error}"
            )

    # --------------------------------------------------------
    # CHECK RESULTS
    # --------------------------------------------------------

    if not scores:

        raise RuntimeError(
            "\nNo valid matching scores "
            "were produced.\n"
        )

    # --------------------------------------------------------
    # CALCULATE BASELINE
    # --------------------------------------------------------

    statistics_result = (
        calculate_baseline_statistics(

            scores=scores,

            remove_outliers=remove_outliers

        )
    )

    result = {

        "machine_id": machine_id,

        "baseline_type": "empty_reference",

        "baseline": (
            statistics_result[
                "baseline"
            ]
        ),

        "statistics": {

            "median": (
                statistics_result[
                    "median"
                ]
            ),

            "mean": (
                statistics_result[
                    "mean"
                ]
            ),

            "std": (
                statistics_result[
                    "std"
                ]
            ),

            "raw_samples": (
                statistics_result[
                    "raw_samples"
                ]
            ),

            "clean_samples": (
                statistics_result[
                    "clean_samples"
                ]
            ),

            "outliers_removed": (
                statistics_result[
                    "outliers_removed"
                ]
            )

        },

        "failed_samples": (
            len(failed_files)
        )

    }

    return result


# ============================================================
# SAVE BASELINE
# ============================================================

def save_baseline(
    result,
    output_path=BASELINE_OUTPUT_PATH
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD EXISTING FILE
    # --------------------------------------------------------

    if output_path.exists():

        with open(
            output_path,
            "r",
            encoding="utf-8"
        ) as file:

            output = json.load(
                file
            )

    else:

        output = {

            "baseline_method": {
                "description": (
                    "Normal reference score "
                    "calculated from EMPTY "
                    "or reference data"
                ),

                "method": (
                    "median_after_iqr_filtering"
                )

            },

            "machines": {}

        }

    # --------------------------------------------------------
    # SAVE MACHINE BASELINE
    # --------------------------------------------------------

    machine_id = result[
        "machine_id"
    ]

    output[
        "machines"
    ][
        machine_id
    ] = result

    # --------------------------------------------------------
    # WRITE FILE
    # --------------------------------------------------------

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(

            output,

            file,

            indent=4

        )

    print()

    print(
        "Baseline saved:"
    )

    print(
        output_path
    )


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    result
):

    print()

    print("=" * 70)

    print(
        "BASELINE RESULT"
    )

    print("=" * 70)

    print()

    print(
        f"Machine ID        : "
        f"{result['machine_id']}"
    )

    print(
        f"Baseline          : "
        f"{result['baseline']:.6f}"
    )

    print()

    print(
        "Statistics:"
    )

    for key, value in (
        result[
            "statistics"
        ].items()
    ):

        print(
            f"  {key}: {value}"
        )

    print()

    print(
        f"Failed samples    : "
        f"{result['failed_samples']}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)

    print(
        "2.2 CREATE BASELINE"
    )

    print("=" * 70)

    print()

    print(
        "This module creates a normal "
        "reference baseline from EMPTY data."
    )

    print()

    # --------------------------------------------------------
    # EXAMPLE PATHS
    # --------------------------------------------------------

    EMPTY_DATA_DIR = (
        PROJECT_DIR
        / "data"
        / "empty"
    )

    TEMPLATE_PATH = (
        PROJECT_DIR
        / "templates"
        / "template.jpg"
    )

    MACHINE_ID = (
        "machine_01"
    )

    # --------------------------------------------------------
    # CREATE BASELINE
    # --------------------------------------------------------

    result = create_baseline(

        input_dir=EMPTY_DATA_DIR,

        machine_id=MACHINE_ID,

        template_path=TEMPLATE_PATH,

        config_path=CONFIG_PATH,

        remove_outliers=True

    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print_result(
        result
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_baseline(
        result
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
