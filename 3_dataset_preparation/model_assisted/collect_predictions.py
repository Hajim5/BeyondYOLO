"""
collect_predictions.py

General model-assisted dataset collection.

Purpose:
    Use an existing trained classification model to collect new candidate
    images from raw videos.

Pipeline:

    Raw Videos
        ↓
    ROI Configuration
        ↓
    Extract ROI Images
        ↓
    Batch Model Inference
        ↓
    Predicted Class + Confidence
        ↓
    Predicted Class
        or
    Uncertain
        ↓
    Save Candidate Images
        ↓
    Manual Review
        ↓
    Expanded Dataset

Important:

    Model predictions are NOT ground truth labels.

    predicted_class != manual_label

The collected images should be reviewed before being added to the
training dataset.
"""

from pathlib import Path
import argparse
import csv
import importlib
import json
import math
import time

import cv2
import torch
from torchvision import transforms


# ============================================================
# VIDEO EXTENSIONS
# ============================================================

VIDEO_EXTENSIONS = {
    ".avi",
    ".mp4",
    ".mov",
    ".mkv",
    ".m4v",
}


# ============================================================
# CONFIGURATION
# ============================================================

def load_json(path):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def load_config(config_path):

    config = load_json(
        config_path
    )

    if not isinstance(
        config,
        dict
    ):

        raise ValueError(
            "Configuration must contain a JSON object."
        )

    return config


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_path(
    path_value,
    config_path
):

    path = Path(path_value)

    if path.is_absolute():
        return path

    return (
        Path(config_path)
        .parent
        / path
    ).resolve()


# ============================================================
# VIDEO DISCOVERY
# ============================================================

def find_videos(
    video_directories,
    config_path
):

    videos = []

    for directory_value in video_directories:

        directory = resolve_path(
            directory_value,
            config_path
        )

        if not directory.exists():

            print(
                "[WARNING] Video directory not found:"
            )

            print(
                f"  {directory}"
            )

            continue

        for path in directory.rglob("*"):

            if (
                path.is_file()
                and path.suffix.lower()
                in VIDEO_EXTENSIONS
            ):

                videos.append(
                    path
                )

    videos = sorted(
        set(videos)
    )

    return videos


# ============================================================
# ROI
# ============================================================

def crop_roi(
    frame,
    roi
):

    if frame is None:
        return None

    if not roi:
        return None

    x = int(
        roi.get("x", 0)
    )

    y = int(
        roi.get("y", 0)
    )

    width = int(
        roi.get("width", 0)
    )

    height = int(
        roi.get("height", 0)
    )

    if (
        width <= 0
        or height <= 0
    ):

        return None

    frame_height, frame_width = (
        frame.shape[:2]
    )

    x1 = max(
        0,
        min(x, frame_width)
    )

    y1 = max(
        0,
        min(y, frame_height)
    )

    x2 = max(
        x1,
        min(
            x + width,
            frame_width
        )
    )

    y2 = max(
        y1,
        min(
            y + height,
            frame_height
        )
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):

        return None

    crop = frame[
        y1:y2,
        x1:x2
    ].copy()

    if crop.size == 0:
        return None

    return crop


# ============================================================
# FRAME SAMPLING
# ============================================================

def generate_sample_frames(
    total_frames,
    frames_per_video
):

    if total_frames <= 0:
        return []

    if frames_per_video <= 0:
        return []

    if total_frames <= frames_per_video:

        return list(
            range(total_frames)
        )

    positions = []

    for index in range(
        frames_per_video
    ):

        position = int(
            round(
                index
                * (
                    total_frames - 1
                )
                /
                (
                    frames_per_video - 1
                )
            )
        )

        positions.append(
            position
        )

    return sorted(
        set(positions)
    )


# ============================================================
# MODEL LOADING
# ============================================================

def import_model_class(
    module_name,
    class_name
):

    try:

        module = importlib.import_module(
            module_name
        )

    except ImportError as error:

        raise ImportError(
            "Could not import model module:\n"
            f"{module_name}\n\n"
            f"Error: {error}"
        )

    if not hasattr(
        module,
        class_name
    ):

        raise AttributeError(
            f"Model class '{class_name}' "
            f"was not found in module "
            f"'{module_name}'."
        )

    return getattr(
        module,
        class_name
    )


def load_model(
    model_config,
    config_path
):

    print()
    print("=" * 80)
    print("LOADING MODEL")
    print("=" * 80)

    model_path_value = (
        model_config.get("path")
    )

    if not model_path_value:

        raise ValueError(
            "Model configuration requires "
            "'model.path'."
        )

    model_path = resolve_path(
        model_path_value,
        config_path
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    device_name = (
        model_config.get(
            "device",
            "auto"
        )
    )

    if device_name == "auto":

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:

        device = torch.device(
            device_name
        )

    architecture = model_config.get(
        "architecture",
        {}
    )

    module_name = architecture.get(
        "module"
    )

    class_name = architecture.get(
        "class"
    )

    init_kwargs = architecture.get(
        "kwargs",
        {}
    )

    if (
        not module_name
        or not class_name
    ):

        raise ValueError(
            "Model architecture requires:\n"
            "model.architecture.module\n"
            "model.architecture.class"
        )

    model_class = import_model_class(
        module_name,
        class_name
    )

    model = model_class(
        **init_kwargs
    )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(
        device
    )

    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print()

    print(
        f"Model: {model_path}"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )

    return model, device


# ============================================================
# IMAGE TRANSFORM
# ============================================================

def create_transform(
    model_config
):

    input_config = (
        model_config.get(
            "input",
            {}
        )
    )

    image_size = int(
        input_config.get(
            "image_size",
            224
        )
    )

    mean = input_config.get(
        "mean",
        [
            0.485,
            0.456,
            0.406,
        ]
    )

    std = input_config.get(
        "std",
        [
            0.229,
            0.224,
            0.225,
        ]
    )

    return transforms.Compose([
        transforms.ToPILImage(),

        transforms.Resize(
            (
                image_size,
                image_size
            )
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=mean,
            std=std
        ),
    ])


# ============================================================
# OUTPUT INTERPRETATION
# ============================================================

def get_probabilities(
    logits,
    output_type
):

    if output_type == "sigmoid":

        return torch.sigmoid(
            logits
        )

    if output_type == "softmax":

        return torch.softmax(
            logits,
            dim=1
        )

    raise ValueError(
        "Unsupported output_type:\n"
        f"{output_type}\n\n"
        "Supported:\n"
        "sigmoid\n"
        "softmax"
    )


def classify_prediction(
    probabilities,
    class_names,
    prediction_config
):

    probabilities = [
        float(value)
        for value
        in probabilities
    ]

    if len(probabilities) != len(
        class_names
    ):

        raise ValueError(
            "Number of probabilities does not "
            "match number of classes."
        )

    uncertain_threshold = float(
        prediction_config.get(
            "uncertain_threshold",
            0.60
        )
    )

    class_thresholds = (
        prediction_config.get(
            "class_thresholds",
            {}
        )
    )

    best_index = max(
        range(
            len(probabilities)
        ),
        key=lambda index:
            probabilities[index]
    )

    best_class = class_names[
        best_index
    ]

    confidence = probabilities[
        best_index
    ]

    threshold = float(
        class_thresholds.get(
            best_class,
            prediction_config.get(
                "default_threshold",
                0.50
            )
        )
    )

    if confidence < uncertain_threshold:

        return {
            "collection_category":
                "uncertain",

            "predicted_class":
                best_class,

            "confidence":
                confidence,

            "accepted":
                False,
        }

    if confidence < threshold:

        return {
            "collection_category":
                "uncertain",

            "predicted_class":
                best_class,

            "confidence":
                confidence,

            "accepted":
                False,
        }

    return {
        "collection_category":
            f"predicted_{best_class}",

        "predicted_class":
            best_class,

        "confidence":
            confidence,

        "accepted":
            True,
    }


# ============================================================
# BATCH PREDICTION
# ============================================================

def predict_batch(
    roi_images,
    model,
    device,
    transform,
    model_config,
    class_names
):

    if not roi_images:
        return []

    tensors = []

    for image in roi_images:

        if image is None:

            raise ValueError(
                "ROI image is None."
            )

        if image.size == 0:

            raise ValueError(
                "ROI image is empty."
            )

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        tensor = transform(
            image_rgb
        )

        tensors.append(
            tensor
        )

    batch = torch.stack(
        tensors,
        dim=0
    ).to(
        device
    )

    output_type = (
        model_config.get(
            "output_type",
            "softmax"
        )
    )

    prediction_config = (
        model_config.get(
            "prediction",
            {}
        )
    )

    with torch.no_grad():

        logits = model(
            batch
        )

        probabilities = get_probabilities(
            logits,
            output_type
        )

    predictions = []

    for index in range(
        len(roi_images)
    ):

        probability_values = (
            probabilities[index]
            .detach()
            .cpu()
            .tolist()
        )

        classification = (
            classify_prediction(
                probabilities=
                    probability_values,

                class_names=
                    class_names,

                prediction_config=
                    prediction_config
            )
        )

        probability_dict = {}

        for class_index, class_name in enumerate(
            class_names
        ):

            probability_dict[
                class_name
            ] = float(
                probability_values[
                    class_index
                ]
            )

        predictions.append({

            "probabilities":
                probability_dict,

            "collection_category":
                classification[
                    "collection_category"
                ],

            "predicted_class":
                classification[
                    "predicted_class"
                ],

            "confidence":
                classification[
                    "confidence"
                ],

            "accepted":
                classification[
                    "accepted"
                ],
        })

    return predictions


# ============================================================
# TEMPORAL DIVERSITY
# ============================================================

def is_temporally_diverse(
    selected_records,
    candidate,
    minimum_gap_seconds
):

    for record in selected_records:

        if (
            record["source_video"]
            ==
            candidate["source_video"]
        ):

            difference = abs(
                record["timestamp"]
                -
                candidate["timestamp"]
            )

            if (
                difference
                <
                minimum_gap_seconds
            ):

                return False

    return True


# ============================================================
# CANDIDATE SELECTION
# ============================================================

def select_candidates(
    candidates,
    categories,
    target_per_category,
    minimum_gap_seconds
):

    selected = []

    grouped = {
        category: []
        for category
        in categories
    }

    for candidate in candidates:

        category = candidate[
            "collection_category"
        ]

        if category in grouped:

            grouped[
                category
            ].append(
                candidate
            )

    for category in categories:

        category_candidates = (
            grouped[category]
        )

        # High confidence first.
        category_candidates.sort(
            key=lambda item:
                item["confidence"],
            reverse=True
        )

        category_selected = []

        for candidate in category_candidates:

            if (
                len(category_selected)
                >=
                target_per_category
            ):

                break

            if is_temporally_diverse(
                category_selected,
                candidate,
                minimum_gap_seconds
            ):

                category_selected.append(
                    candidate
                )

        selected.extend(
            category_selected
        )

    return selected


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(
    image,
    output_path,
    jpeg_quality
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    success = cv2.imwrite(
        str(output_path),
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            jpeg_quality,
        ]
    )

    return success


# ============================================================
# PROCESS VIDEO
# ============================================================

def process_video(
    video_path,
    machines,
    model,
    device,
    transform,
    model_config,
    class_names,
    frames_per_video,
    batch_size
):

    candidates = []

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            "[WARNING] Could not open:"
        )

        print(
            f"  {video_path}"
        )

        return candidates

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    if fps <= 0:
        fps = 30.0

    sample_frames = (
        generate_sample_frames(
            total_frames,
            frames_per_video
        )
    )

    frame_items = []

    for frame_index in sample_frames:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index
        )

        success, frame = cap.read()

        if not success:
            continue

        timestamp = (
            frame_index / fps
        )

        for machine_id, machine in (
            machines.items()
        ):

            roi = machine.get(
                "roi"
            )

            crop = crop_roi(
                frame,
                roi
            )

            if crop is None:
                continue

            frame_items.append({

                "machine_id":
                    machine_id,

                "source_video":
                    str(video_path),

                "video_name":
                    video_path.name,

                "frame_index":
                    int(frame_index),

                "timestamp":
                    float(timestamp),

                "image":
                    crop,
            })

    cap.release()

    for start in range(
        0,
        len(frame_items),
        batch_size
    ):

        batch_items = frame_items[
            start:
            start + batch_size
        ]

        batch_images = [
            item["image"]
            for item
            in batch_items
        ]

        predictions = predict_batch(
            roi_images=batch_images,
            model=model,
            device=device,
            transform=transform,
            model_config=model_config,
            class_names=class_names
        )

        for item, prediction in zip(
            batch_items,
            predictions
        ):

            candidates.append({

                **item,

                **prediction,
            })

    return candidates


# ============================================================
# MANIFEST
# ============================================================

def write_manifest(
    rows,
    manifest_path,
    class_names
):

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [

        "image_path",

        "source_video",

        "video_name",

        "machine_id",

        "frame_index",

        "timestamp",

        "collection_category",

        "predicted_class",

        "confidence",
    ]

    for class_name in class_names:

        fieldnames.append(
            f"probability_{class_name}"
        )

    fieldnames.extend([

        "manual_label",

        "review_status",
    ])

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# COLLECTION INFO
# ============================================================

def write_collection_info(
    output_path,
    config_path,
    model_config,
    videos,
    total_saved,
    categories
):

    info = {

        "purpose":
            (
                "Model-assisted candidate "
                "dataset collection"
            ),

        "config_path":
            str(config_path),

        "model_path":
            model_config.get(
                "path"
            ),

        "classes":
            model_config.get(
                "classes",
                []
            ),

        "video_count":
            len(videos),

        "total_saved_images":
            total_saved,

        "categories":
            categories,

        "important_note":
            (
                "Model predictions are candidate "
                "predictions and are not final "
                "ground truth labels."
            ),
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            info,
            file,
            indent=4
        )


# ============================================================
# MAIN COLLECTION
# ============================================================

def collect_predictions(
    config_path,
    output_directory,
    target_per_category=None,
    frames_per_video=None,
    minimum_gap_seconds=None,
    batch_size=None
):

    print()
    print("=" * 80)
    print("MODEL-ASSISTED DATASET COLLECTION")
    print("=" * 80)

    config_path = Path(
        config_path
    ).resolve()

    config = load_config(
        config_path
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model_config = config.get(
        "model"
    )

    if not model_config:

        raise ValueError(
            "Configuration requires a 'model' section."
        )

    class_names = model_config.get(
        "classes",
        []
    )

    if not class_names:

        raise ValueError(
            "model.classes cannot be empty."
        )

    # --------------------------------------------------------
    # DATASET COLLECTION SETTINGS
    # --------------------------------------------------------

    collection_config = config.get(
        "dataset_collection",
        {}
    )

    if target_per_category is None:

        target_per_category = int(
            collection_config.get(
                "target_per_category",
                100
            )
        )

    if frames_per_video is None:

        frames_per_video = int(
            collection_config.get(
                "frames_per_video",
                120
            )
        )

    if minimum_gap_seconds is None:

        minimum_gap_seconds = float(
            collection_config.get(
                "minimum_gap_seconds",
                10.0
            )
        )

    if batch_size is None:

        batch_size = int(
            collection_config.get(
                "batch_size",
                32
            )
        )

    jpeg_quality = int(
        collection_config.get(
            "jpeg_quality",
            95
        )
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output_directory = Path(
        output_directory
    ).resolve()

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest_path = (
        output_directory
        / "manifest.csv"
    )

    info_path = (
        output_directory
        / "collection_info.json"
    )

    # --------------------------------------------------------
    # VIDEO DIRECTORIES
    # --------------------------------------------------------

    video_directories = config.get(
        "video_directories",
        []
    )

    if not video_directories:

        raise ValueError(
            "No video_directories configured."
        )

    videos = find_videos(
        video_directories,
        config_path
    )

    if not videos:

        raise RuntimeError(
            "No videos found."
        )

    # --------------------------------------------------------
    # MACHINES / ROIS
    # --------------------------------------------------------

    machines = config.get(
        "machines",
        {}
    )

    if not machines:

        raise ValueError(
            "No machines configured."
        )

    if not isinstance(
        machines,
        dict
    ):

        raise ValueError(
            "'machines' must be a dictionary."
        )

    valid_machines = {}

    for machine_id, machine in (
        machines.items()
    ):

        if not machine.get("roi"):

            print(
                f"[WARNING] "
                f"{machine_id}: "
                f"missing ROI"
            )

            continue

        valid_machines[
            machine_id
        ] = machine

    machines = valid_machines

    if not machines:

        raise RuntimeError(
            "No valid machine ROIs found."
        )

    # --------------------------------------------------------
    # CATEGORIES
    # --------------------------------------------------------

    categories = [

        f"predicted_{class_name}"

        for class_name
        in class_names
    ]

    categories.append(
        "uncertain"
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model, device = load_model(
        model_config,
        config_path
    )

    transform = create_transform(
        model_config
    )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    print()

    print(
        f"Videos found: "
        f"{len(videos)}"
    )

    print(
        f"ROIs found: "
        f"{len(machines)}"
    )

    print(
        f"Classes: "
        f"{', '.join(class_names)}"
    )

    print(
        f"Frames per video: "
        f"{frames_per_video}"
    )

    print(
        f"Target per category: "
        f"{target_per_category}"
    )

    print(
        f"Batch size: "
        f"{batch_size}"
    )

    # --------------------------------------------------------
    # COLLECTION
    # --------------------------------------------------------

    all_candidates = []

    start_time = time.perf_counter()

    for index, video_path in enumerate(
        videos,
        start=1
    ):

        print()

        print(
            f"[{index}/{len(videos)}] "
            f"{video_path.name}"
        )

        video_candidates = process_video(

            video_path=video_path,

            machines=machines,

            model=model,

            device=device,

            transform=transform,

            model_config=model_config,

            class_names=class_names,

            frames_per_video=
                frames_per_video,

            batch_size=
                batch_size,
        )

        all_candidates.extend(
            video_candidates
        )

    # --------------------------------------------------------
    # SELECT PER MACHINE
    # --------------------------------------------------------

    selected_candidates = []

    for machine_id in machines:

        machine_candidates = [

            candidate

            for candidate
            in all_candidates

            if candidate[
                "machine_id"
            ]
            ==
            machine_id
        ]

        selected = select_candidates(

            candidates=
                machine_candidates,

            categories=
                categories,

            target_per_category=
                target_per_category,

            minimum_gap_seconds=
                minimum_gap_seconds,
        )

        selected_candidates.extend(
            selected
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    manifest_rows = []

    for index, candidate in enumerate(
        selected_candidates,
        start=1
    ):

        machine_id = candidate[
            "machine_id"
        ]

        category = candidate[
            "collection_category"
        ]

        filename = (

            f"{machine_id}_"

            f"{category}_"

            f"{index:06d}.jpg"
        )

        output_path = (

            output_directory

            / machine_id

            / category

            / filename
        )

        success = save_image(

            image=
                candidate["image"],

            output_path=
                output_path,

            jpeg_quality=
                jpeg_quality,
        )

        if not success:

            print(
                "[WARNING] Failed to save:"
            )

            print(
                f"  {output_path}"
            )

            continue

        row = {

            "image_path":
                str(output_path),

            "source_video":
                candidate[
                    "source_video"
                ],

            "video_name":
                candidate[
                    "video_name"
                ],

            "machine_id":
                machine_id,

            "frame_index":
                candidate[
                    "frame_index"
                ],

            "timestamp":
                (
                    f"{candidate['timestamp']:.6f}"
                ),

            "collection_category":
                category,

            "predicted_class":
                candidate[
                    "predicted_class"
                ],

            "confidence":
                (
                    f"{candidate['confidence']:.6f}"
                ),

            "manual_label":
                "",

            "review_status":
                "pending",
        }

        for class_name in class_names:

            row[
                f"probability_{class_name}"
            ] = (

                f"{candidate['probabilities'][class_name]:.6f}"
            )

        manifest_rows.append(
            row
        )

    # --------------------------------------------------------
    # MANIFEST
    # --------------------------------------------------------

    write_manifest(

        rows=
            manifest_rows,

        manifest_path=
            manifest_path,

        class_names=
            class_names,
    )

    # --------------------------------------------------------
    # INFO
    # --------------------------------------------------------

    write_collection_info(

        output_path=
            info_path,

        config_path=
            config_path,

        model_config=
            model_config,

        videos=
            videos,

        total_saved=
            len(manifest_rows),

        categories=
            categories,
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    elapsed = (

        time.perf_counter()

        -

        start_time
    )

    print()
    print("=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)

    print()

    print(
        f"Videos processed: "
        f"{len(videos)}"
    )

    print(
        f"Images saved: "
        f"{len(manifest_rows)}"
    )

    print(
        f"Runtime: "
        f"{elapsed:.2f} seconds"
    )

    print()

    print(
        "Output:"
    )

    print(
        output_directory
    )

    print()

    print(
        "Manifest:"
    )

    print(
        manifest_path
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=
            (
                "General model-assisted "
                "dataset collector"
            )
    )

    parser.add_argument(

        "--config",

        required=True,

        help=
            (
                "Path to configuration "
                "JSON file"
            )
    )

    parser.add_argument(

        "--output",

        default=
            "model_assisted_dataset",

        help=
            (
                "Output dataset "
                "directory"
            )
    )

    parser.add_argument(

        "--target",

        type=int,

        default=None,

        help=
            (
                "Maximum images per "
                "category per ROI"
            )
    )

    parser.add_argument(

        "--frames-per-video",

        type=int,

        default=None,

        help=
            (
                "Frames sampled "
                "per video"
            )
    )

    parser.add_argument(

        "--minimum-gap",

        type=float,

        default=None,

        help=
            (
                "Minimum time gap "
                "between selected images"
            )
    )

    parser.add_argument(

        "--batch-size",

        type=int,

        default=None,

        help=
            (
                "Model inference "
                "batch size"
            )
    )

    args = parser.parse_args()

    collect_predictions(

        config_path=
            args.config,

        output_directory=
            args.output,

        target_per_category=
            args.target,

        frames_per_video=
            args.frames_per_video,

        minimum_gap_seconds=
            args.minimum_gap,

        batch_size=
            args.batch_size,
    )


if __name__ == "__main__":
    main()
