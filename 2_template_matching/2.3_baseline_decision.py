"""
2.3_baseline_decision.py

GENERAL BASELINE DECISION MODULE

Purpose:
    Compare a current template-matching score against the
    learned EMPTY baseline.

Pipeline:

    New image
        ↓
    2.1_template_matching.py
        ↓
    Current matching score
        ↓
    Load baseline.json
        ↓
    Calculate delta
        ↓
    Compare against threshold
        ↓
    DETECTED / NOT DETECTED


Decision:

    delta = current_score - empty_baseline

The object may cause the score to:

    ABOVE:
        current_score > baseline

    BELOW:
        current_score < baseline

This script supports both directions.

The direction can be:

    AUTO
    ABOVE
    BELOW
"""


from pathlib import Path
import json
import sys
import importlib.util


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_DIR / "config.json"

OUTPUT_DIR = PROJECT_DIR / "output"

BASELINE_PATH = (
    OUTPUT_DIR
    / "baseline.json"
)


# ============================================================
# LOAD TEMPLATE MATCHING MODULE
# ============================================================

MODULE_DIR = Path(__file__).resolve().parent

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
# LOAD BASELINES
# ============================================================

def load_baselines(
    baseline_path=BASELINE_PATH
):

    baseline_path = Path(
        baseline_path
    )

    if not baseline_path.exists():

        raise FileNotFoundError(
            "\nBaseline file not found:\n"
            f"{baseline_path}\n\n"
            "Run 2.2_create_baseline.py first.\n"
        )

    with open(
        baseline_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


# ============================================================
# GET MACHINE BASELINE
# ============================================================

def get_machine_baseline(
    baseline_data,
    machine_id
):

    machines = baseline_data.get(
        "machines",
        {}
    )

    if machine_id not in machines:

        raise KeyError(
            "\nBaseline not found for machine:\n"
            f"{machine_id}\n"
        )

    machine_data = (
        machines[machine_id]
    )

    baseline = machine_data.get(
        "baseline"
    )

    if baseline is None:

        raise ValueError(
            "\nInvalid baseline for machine:\n"
            f"{machine_id}\n"
        )

    return float(
        baseline
    )


# ============================================================
# GET THRESHOLD
# ============================================================

def get_threshold(
    baseline_data,
    machine_id,
    default_threshold=None
):

    machines = baseline_data.get(
        "machines",
        {}
    )

    machine_data = (
        machines.get(
            machine_id,
            {}
        )
    )

    # If threshold was previously saved
    threshold = machine_data.get(
        "threshold"
    )

    if threshold is not None:

        return abs(
            float(threshold)
        )

    # Optional default
    if default_threshold is not None:

        return abs(
            float(default_threshold)
        )

    # Automatic fallback using baseline std
    statistics_data = (
        machine_data.get(
            "statistics",
            {}
        )
    )

    std = statistics_data.get(
        "std"
    )

    if std is not None:

        std = abs(
            float(std)
        )

        # Minimum threshold protection
        return max(
            std * 3.0,
            0.01
        )

    raise ValueError(
        "\nNo threshold found.\n"
        "Provide a threshold or save one in baseline.json.\n"
    )


# ============================================================
# GET DIRECTION
# ============================================================

def get_direction(
    baseline_data,
    machine_id,
    direction="AUTO"
):

    direction = (
        direction
        .strip()
        .upper()
    )

    valid_directions = {

        "AUTO",

        "ABOVE",

        "BELOW"

    }

    if direction not in valid_directions:

        raise ValueError(
            "\nInvalid direction:\n"
            f"{direction}\n\n"
            "Valid directions:\n"
            "AUTO\n"
            "ABOVE\n"
            "BELOW\n"
        )

    # User explicitly selected direction
    if direction != "AUTO":

        return direction

    # Try loading saved direction
    machines = baseline_data.get(
        "machines",
        {}
    )

    machine_data = (
        machines.get(
            machine_id,
            {}
        )
    )

    saved_direction = (
        machine_data.get(
            "direction"
        )
    )

    if saved_direction:

        saved_direction = (
            str(saved_direction)
            .upper()
        )

        if saved_direction in {

            "ABOVE",

            "BELOW"

        }:

            return saved_direction

    # Default when direction has not been learned yet
    return "BOTH"


# ============================================================
# CALCULATE DELTA
# ============================================================

def calculate_delta(
    current_score,
    baseline
):

    return (
        float(current_score)
        -
        float(baseline)
    )


# ============================================================
# MAKE DECISION
# ============================================================

def make_decision(
    current_score,
    baseline,
    threshold,
    direction="BOTH"
):

    delta = calculate_delta(
        current_score,
        baseline
    )

    direction = (
        direction
        .strip()
        .upper()
    )

    threshold = abs(
        float(threshold)
    )

    # --------------------------------------------------------
    # ABOVE
    # --------------------------------------------------------

    if direction == "ABOVE":

        detected = (
            delta >= threshold
        )

    # --------------------------------------------------------
    # BELOW
    # --------------------------------------------------------

    elif direction == "BELOW":

        detected = (
            delta <= -threshold
        )

    # --------------------------------------------------------
    # BOTH
    #
    # Detect significant score change in either direction.
    # --------------------------------------------------------

    elif direction == "BOTH":

        detected = (
            abs(delta)
            >= threshold
        )

    else:

        raise ValueError(
            "\nInvalid decision direction:\n"
            f"{direction}\n"
        )

    if detected:

        status = "DETECTED"

    else:

        status = "NOT_DETECTED"

    return {

        "status": status,

        "detected": detected,

        "current_score": float(
            current_score
        ),

        "baseline": float(
            baseline
        ),

        "delta": float(
            delta
        ),

        "absolute_delta": float(
            abs(delta)
        ),

        "threshold": float(
            threshold
        ),

        "direction": direction

    }


# ============================================================
# RUN DECISION FROM SCORE
# ============================================================

def decide_from_score(
    current_score,
    machine_id,
    baseline_path=BASELINE_PATH,
    threshold=None,
    direction="AUTO"
):

    # --------------------------------------------------------
    # LOAD BASELINE DATA
    # --------------------------------------------------------

    baseline_data = load_baselines(
        baseline_path
    )

    # --------------------------------------------------------
    # GET EMPTY BASELINE
    # --------------------------------------------------------

    baseline = get_machine_baseline(
        baseline_data,
        machine_id
    )

    # --------------------------------------------------------
    # GET THRESHOLD
    # --------------------------------------------------------

    decision_threshold = get_threshold(

        baseline_data=baseline_data,

        machine_id=machine_id,

        default_threshold=threshold

    )

    # --------------------------------------------------------
    # GET DIRECTION
    # --------------------------------------------------------

    decision_direction = get_direction(

        baseline_data=baseline_data,

        machine_id=machine_id,

        direction=direction

    )

    # --------------------------------------------------------
    # MAKE DECISION
    # --------------------------------------------------------

    result = make_decision(

        current_score=current_score,

        baseline=baseline,

        threshold=decision_threshold,

        direction=decision_direction

    )

    result[
        "machine_id"
    ] = machine_id

    return result


# ============================================================
# PROCESS IMAGE
# ============================================================

def decide_from_image(
    image_path,
    machine_id,
    template_path,
    config_path=CONFIG_PATH,
    baseline_path=BASELINE_PATH,
    threshold=None,
    direction="AUTO"
):

    # --------------------------------------------------------
    # LOAD CONFIG
    # --------------------------------------------------------

    config = load_config(
        config_path
    )

    machines = config.get(
        "machines",
        {}
    )

    if machine_id not in machines:

        raise KeyError(
            "\nMachine not found in config:\n"
            f"{machine_id}\n"
        )

    machine_config = (
        machines[machine_id]
    )

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    frame = (
        template_matching.load_image(
            image_path
        )
    )

    # --------------------------------------------------------
    # LOAD TEMPLATE
    # --------------------------------------------------------

    template = (
        template_matching.load_template(
            template_path
        )
    )

    # --------------------------------------------------------
    # GET SETTINGS
    # --------------------------------------------------------

    settings = config.get(
        "template_matching",
        template_matching.DEFAULT_SETTINGS
    )

    # --------------------------------------------------------
    # PREPARE TEMPLATE
    # --------------------------------------------------------

    template_processed = (
        template_matching.prepare_template(

            template,

            settings

        )
    )

    # --------------------------------------------------------
    # RUN TEMPLATE MATCHING
    # --------------------------------------------------------

    matching_result = (
        template_matching.match_machine(

            frame=frame,

            machine_id=machine_id,

            machine_config=machine_config,

            template_edges=template_processed,

            settings=settings

        )
    )

    current_score = (
        matching_result.get(
            "score"
        )
    )

    if current_score is None:

        raise RuntimeError(
            "\nTemplate matching did not "
            "produce a valid score.\n"
        )

    # --------------------------------------------------------
    # BASELINE DECISION
    # --------------------------------------------------------

    decision_result = (
        decide_from_score(

            current_score=current_score,

            machine_id=machine_id,

            baseline_path=baseline_path,

            threshold=threshold,

            direction=direction

        )
    )

    # Add template matching information
    decision_result[
        "template_matching"
    ] = matching_result

    return decision_result


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    result
):

    print()

    print("=" * 70)

    print(
        "BASELINE DECISION"
    )

    print("=" * 70)

    print()

    print(
        f"Machine ID     : "
        f"{result['machine_id']}"
    )

    print(
        f"Status         : "
        f"{result['status']}"
    )

    print()

    print(
        f"Current Score  : "
        f"{result['current_score']:.6f}"
    )

    print(
        f"Baseline       : "
        f"{result['baseline']:.6f}"
    )

    print(
        f"Delta          : "
        f"{result['delta']:+.6f}"
    )

    print(
        f"Threshold      : "
        f"{result['threshold']:.6f}"
    )

    print(
        f"Direction      : "
        f"{result['direction']}"
    )

    print()

    print(
        f"Detected       : "
        f"{result['detected']}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)

    print(
        "2.3 BASELINE DECISION"
    )

    print("=" * 70)

    print()

    # --------------------------------------------------------
    # EXAMPLE INPUT
    # --------------------------------------------------------

    IMAGE_PATH = (
        PROJECT_DIR
        / "data"
        / "test"
        / "test_image.jpg"
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
    # OPTIONAL SETTINGS
    # --------------------------------------------------------

    # None = automatically use baseline.json
    THRESHOLD = None

    # AUTO:
    #   Use saved machine direction.
    #
    # ABOVE:
    #   Detect when score increases.
    #
    # BELOW:
    #   Detect when score decreases.
    #
    # If AUTO has no learned direction,
    # BOTH is used automatically.

    DIRECTION = "AUTO"

    # --------------------------------------------------------
    # RUN DECISION
    # --------------------------------------------------------

    result = decide_from_image(

        image_path=IMAGE_PATH,

        machine_id=MACHINE_ID,

        template_path=TEMPLATE_PATH,

        config_path=CONFIG_PATH,

        baseline_path=BASELINE_PATH,

        threshold=THRESHOLD,

        direction=DIRECTION

    )

    # --------------------------------------------------------
    # PRINT RESULT
    # --------------------------------------------------------

    print_result(
        result
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
