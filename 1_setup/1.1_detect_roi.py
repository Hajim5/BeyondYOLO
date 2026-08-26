"""
detect_roi.py

GENERAL ROI DISCOVERY
=====================

One-time Region of Interest (ROI) discovery.

An optional object detection model is used ONLY during the
setup stage to automatically propose regions of interest.

The detected regions are NOT automatically treated as final.

Pipeline:

    Input Data
        ↓
    Sample Frames
        ↓
    Detection Model
        ↓
    Repeated Detections
        ↓
    Stable ROI Proposals
        ↓
    roi_config.json
        ↓
    review_roi.py
        ↓
    Confirmed Configuration


The detection model is only used to help discover relevant
regions during setup.

Runtime methods can use the confirmed ROI configuration without
running the detection model again.
"""

from pathlib import Path
from collections import defaultdict
from statistics import median
import argparse
import csv
import json
import math
import time

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT_DIR = PROJECT_DIR / "input" / "OWN_INPUT"

DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output"

DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "YOUR_MODEL.pt"


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
# YOLO SETTINGS
# ============================================================

YOLO_CONFIDENCE = 0.25

YOLO_IOU = 0.45


# ============================================================
# DISCOVERY SETTINGS
# ============================================================

# Number of videos distributed across the dataset.
DISCOVERY_VIDEO_COUNT = 8


# Number of timestamps sampled from each discovery video.
FRAMES_PER_VIDEO = 5


# Avoid beginning/end of video.
VIDEO_EDGE_MARGIN_SECONDS = 20.0


# ============================================================
# STABLE DETECTION SETTINGS
# ============================================================

# A target must appear in at least this many sampled frames.
MIN_DETECTION_COUNT = 3


# Remove obvious positional outliers before calculating final ROI.
POSITION_OUTLIER_FACTOR = 3.0


# ============================================================
# ROI PADDING
# ============================================================

ROI_PADDING_X = 0

ROI_PADDING_Y = 0


# ============================================================
# PREVIEW
# ============================================================

PREVIEW_FONT_SCALE = 0.65

PREVIEW_THICKNESS = 2


# ============================================================
# HELPERS
# ============================================================

def ensure_directory(path):

    path.mkdir(
        parents=True,
        exist_ok=True
    )


def save_json(
    data,
    path
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )


# ============================================================
# FIND VIDEOS
# ============================================================

def find_videos(
    input_dir
):

    videos = []

    for path in input_dir.rglob("*"):

        if not path.is_file():
            continue

        if (
            path.suffix.lower()
            not in
            VIDEO_EXTENSIONS
        ):
            continue

        videos.append(
            path
        )

    return sorted(
        videos
    )


# ============================================================
# SELECT DISCOVERY VIDEOS
# ============================================================

def select_discovery_videos(
    videos,
    count
):

    if len(videos) <= count:

        return videos

    indices = np.linspace(
        0,
        len(videos) - 1,
        count
    )

    indices = np.round(
        indices
    ).astype(int)

    selected = []

    used = set()

    for index in indices:

        index = int(index)

        if index in used:
            continue

        used.add(
            index
        )

        selected.append(
            videos[index]
        )

    return selected


# ============================================================
# VIDEO INFORMATION
# ============================================================

def get_video_info(
    video_path
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        cap.release()

        return None

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    if (
        fps is None
        or
        fps <= 0
    ):

        duration = 0.0

    else:

        duration = (
            frame_count
            /
            fps
        )

    cap.release()

    return {
        "fps":
            float(fps),

        "frame_count":
            int(frame_count),

        "duration":
            float(duration),

        "width":
            width,

        "height":
            height,
    }


# ============================================================
# SAMPLE TIMESTAMPS
# ============================================================

def get_sample_timestamps(
    duration,
    count
):

    if duration <= 0:

        return []

    margin = min(
        VIDEO_EDGE_MARGIN_SECONDS,
        duration * 0.10
    )

    start = margin

    end = (
        duration
        -
        margin
    )

    if end <= start:

        start = 0.0

        end = duration

    timestamps = np.linspace(
        start,
        end,
        count
    )

    return [
        float(value)
        for value
        in timestamps
    ]


# ============================================================
# LOAD YOLO
# ============================================================

def load_detection_model(
    model_path
):

    if YOLO is None:

        raise ImportError(
            "\nUltralytics is not installed.\n"
            "Install it using:\n\n"
            "    pip install ultralytics\n"
        )

    if not model_path.exists():

        raise FileNotFoundError(
            "\nDetection model not found:\n"
            f"{model_path}\n"
        )

    print()
    print("Loading detection model:")
    print(
        f"  {model_path}"
    )

    return YOLO(
        str(model_path)
    )


# ============================================================
# READ FRAME
# ============================================================

def read_frame_at_time(
    video_path,
    timestamp
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        cap.release()

        return None

    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        timestamp * 1000.0
    )

    success, frame = cap.read()

    cap.release()

    if not success:

        return None

    return frame


# ============================================================
# RUN DETECTION
# ============================================================

def detect_objects(
    model,
    frame
):

    results = model(
        frame,
        conf=YOLO_CONFIDENCE,
        iou=YOLO_IOU,
        verbose=False
    )

    detections = []

    if not results:

        return detections

    result = results[0]

    if result.boxes is None:

        return detections

    boxes = result.boxes

    for index in range(
        len(boxes)
    ):

        xyxy = (
            boxes.xyxy[index]
            .cpu()
            .numpy()
        )

        confidence = float(
            boxes.conf[index]
            .cpu()
            .item()
        )

        class_id = int(
            boxes.cls[index]
            .cpu()
            .item()
        )

        x1, y1, x2, y2 = [
            float(value)
            for value
            in xyxy
        ]

        detections.append({

            "class_id":
                class_id,

            "confidence":
                confidence,

            "x1":
                x1,

            "y1":
                y1,

            "x2":
                x2,

            "y2":
                y2,

        })

    return detections


# ============================================================
# CENTER DISTANCE
# ============================================================

def get_center(
    detection
):

    center_x = (
        detection["x1"]
        +
        detection["x2"]
    ) / 2.0

    center_y = (
        detection["y1"]
        +
        detection["y2"]
    ) / 2.0

    return (
        center_x,
        center_y
    )


# ============================================================
# GROUP REPEATED DETECTIONS
# ============================================================

def group_detections(
    detections
):

    groups = []

    for detection in detections:

        center_x, center_y = (
            get_center(
                detection
            )
        )

        matched_group = None

        for group in groups:

            group_x = median(
                [
                    get_center(item)[0]
                    for item
                    in group
                ]
            )

            group_y = median(
                [
                    get_center(item)[1]
                    for item
                    in group
                ]
            )

            distances = []

            widths = []

            heights = []

            for item in group:

                item_x, item_y = (
                    get_center(
                        item
                    )
                )

                distance = math.sqrt(
                    (item_x - group_x) ** 2
                    +
                    (item_y - group_y) ** 2
                )

                distances.append(
                    distance
                )

                widths.append(
                    item["x2"]
                    -
                    item["x1"]
                )

                heights.append(
                    item["y2"]
                    -
                    item["y1"]
                )

            typical_size = max(
                median(widths),
                median(heights),
                1.0
            )

            distance_to_group = math.sqrt(
                (center_x - group_x) ** 2
                +
                (center_y - group_y) ** 2
            )

            if (
                distance_to_group
                <=
                typical_size
            ):

                matched_group = group

                break

        if matched_group is None:

            groups.append(
                [detection]
            )

        else:

            matched_group.append(
                detection
            )

    return groups


# ============================================================
# REMOVE POSITION OUTLIERS
# ============================================================

def remove_position_outliers(
    detections
):

    if len(detections) < 3:

        return detections

    centers = np.array(
        [
            get_center(item)
            for item
            in detections
        ],
        dtype=np.float32
    )

    median_center = np.median(
        centers,
        axis=0
    )

    distances = np.linalg.norm(
        centers
        -
        median_center,
        axis=1
    )

    median_distance = np.median(
        distances
    )

    mad = np.median(
        np.abs(
            distances
            -
            median_distance
        )
    )

    if mad <= 0:

        return detections

    limit = (
        median_distance
        +
        POSITION_OUTLIER_FACTOR
        *
        mad
    )

    filtered = []

    for detection, distance in zip(
        detections,
        distances
    ):

        if distance <= limit:

            filtered.append(
                detection
            )

    return filtered


# ============================================================
# BUILD STABLE ROIS
# ============================================================

def build_stable_rois(
    detections_by_class,
    frame_width,
    frame_height
):

    rois = []

    roi_counter = 1

    for class_id in sorted(
        detections_by_class.keys()
    ):

        groups = group_detections(
            detections_by_class[class_id]
        )

        for group in groups:

            if (
                len(group)
                <
                MIN_DETECTION_COUNT
            ):

                continue

            filtered = (
                remove_position_outliers(
                    group
                )
            )

            if not filtered:

                continue

            x1 = median(
                [
                    item["x1"]
                    for item
                    in filtered
                ]
            )

            y1 = median(
                [
                    item["y1"]
                    for item
                    in filtered
                ]
            )

            x2 = median(
                [
                    item["x2"]
                    for item
                    in filtered
                ]
            )

            y2 = median(
                [
                    item["y2"]
                    for item
                    in filtered
                ]
            )

            x1 -= ROI_PADDING_X
            y1 -= ROI_PADDING_Y

            x2 += ROI_PADDING_X
            y2 += ROI_PADDING_Y

            x1 = max(
                0,
                int(round(x1))
            )

            y1 = max(
                0,
                int(round(y1))
            )

            x2 = min(
                frame_width,
                int(round(x2))
            )

            y2 = min(
                frame_height,
                int(round(y2))
            )

            if (
                x2 <= x1
                or
                y2 <= y1
            ):

                continue

            confidences = [
                item["confidence"]
                for item
                in filtered
            ]

            rois.append({

                "id":
                    f"roi_{roi_counter:03d}",

                "class_id":
                    int(class_id),

                "roi": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },

                "detection_count":
                    len(group),

                "used_detection_count":
                    len(filtered),

                "mean_confidence":
                    float(
                        np.mean(
                            confidences
                        )
                    ),

                "median_confidence":
                    float(
                        np.median(
                            confidences
                        )
                    ),

                "min_confidence":
                    float(
                        np.min(
                            confidences
                        )
                    ),

                "max_confidence":
                    float(
                        np.max(
                            confidences
                        )
                    ),

                "review_status":
                    "PENDING",

            })

            roi_counter += 1

    return rois


# ============================================================
# DRAW PREVIEW
# ============================================================

def draw_roi_preview(
    frame,
    rois
):

    preview = frame.copy()

    for roi in rois:

        coordinates = roi["roi"]

        x1 = coordinates["x1"]
        y1 = coordinates["y1"]

        x2 = coordinates["x2"]
        y2 = coordinates["y2"]

        label = roi["id"]

        cv2.rectangle(
            preview,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            PREVIEW_THICKNESS
        )

        cv2.putText(
            preview,
            label,
            (
                x1,
                max(
                    25,
                    y1 - 8
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            PREVIEW_FONT_SCALE,
            (0, 255, 0),
            PREVIEW_THICKNESS,
            cv2.LINE_AA
        )

    return preview


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary_csv(
    rois,
    summary_path
):

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "class_id",
                "x1",
                "y1",
                "x2",
                "y2",
                "detection_count",
                "used_detection_count",
                "mean_confidence",
                "median_confidence",
                "min_confidence",
                "max_confidence",
                "review_status",
            ]
        )

        writer.writeheader()

        for roi in rois:

            coordinates = roi["roi"]

            writer.writerow({

                "id":
                    roi["id"],

                "class_id":
                    roi["class_id"],

                "x1":
                    coordinates["x1"],

                "y1":
                    coordinates["y1"],

                "x2":
                    coordinates["x2"],

                "y2":
                    coordinates["y2"],

                "detection_count":
                    roi["detection_count"],

                "used_detection_count":
                    roi[
                        "used_detection_count"
                    ],

                "mean_confidence":
                    f"{roi['mean_confidence']:.6f}",

                "median_confidence":
                    f"{roi['median_confidence']:.6f}",

                "min_confidence":
                    f"{roi['min_confidence']:.6f}",

                "max_confidence":
                    f"{roi['max_confidence']:.6f}",

                "review_status":
                    roi["review_status"],

            })


# ============================================================
# MAIN ROI DISCOVERY
# ============================================================

def detect_roi(
    input_dir,
    output_dir,
    model_path
):

    start_time = time.time()

    input_dir = Path(
        input_dir
    ).resolve()

    output_dir = Path(
        output_dir
    ).resolve()

    model_path = Path(
        model_path
    ).resolve()

    if not input_dir.exists():

        raise FileNotFoundError(
            "\nInput directory not found:\n"
            f"{input_dir}\n"
        )

    samples_dir = (
        output_dir
        /
        "samples"
    )

    ensure_directory(
        output_dir
    )

    ensure_directory(
        samples_dir
    )

    config_path = (
        output_dir
        /
        "roi_config.json"
    )

    summary_path = (
        output_dir
        /
        "roi_summary.csv"
    )

    preview_path = (
        output_dir
        /
        "roi_preview.jpg"
    )

    # ========================================================
    # FIND VIDEOS
    # ========================================================

    videos = find_videos(
        input_dir
    )

    if not videos:

        raise FileNotFoundError(
            "\nNo supported videos found in:\n"
            f"{input_dir}\n"
        )

    discovery_videos = (
        select_discovery_videos(
            videos,
            DISCOVERY_VIDEO_COUNT
        )
    )

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 80)
    print("ROI DISCOVERY")
    print("=" * 80)

    print()
    print(
        f"Input directory : {input_dir}"
    )

    print(
        f"Output directory: {output_dir}"
    )

    print(
        f"Total videos    : {len(videos)}"
    )

    print(
        f"Discovery videos: "
        f"{len(discovery_videos)}"
    )

    print(
        f"Frames per video: "
        f"{FRAMES_PER_VIDEO}"
    )

    print(
        f"Maximum model frames: "
        f"{len(discovery_videos) * FRAMES_PER_VIDEO}"
    )

    print()
    print(
        "The detection model is used only for "
        "initial ROI discovery."
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = load_detection_model(
        model_path
    )

    # ========================================================
    # COLLECTION
    # ========================================================

    detections_by_class = defaultdict(
        list
    )

    sampled_frames = []

    total_processed_frames = 0

    reference_width = None
    reference_height = None

    sample_counter = 0

    print()
    print("=" * 80)
    print("RUN DISCOVERY")
    print("=" * 80)

    # ========================================================
    # PROCESS VIDEOS
    # ========================================================

    for video_index, video_path in enumerate(
        discovery_videos,
        start=1
    ):

        print()
        print(
            f"[{video_index}/"
            f"{len(discovery_videos)}] "
            f"{video_path.name}"
        )

        info = get_video_info(
            video_path
        )

        if info is None:

            print(
                "  Could not open video."
            )

            continue

        timestamps = (
            get_sample_timestamps(
                info["duration"],
                FRAMES_PER_VIDEO
            )
        )

        for timestamp in timestamps:

            frame = (
                read_frame_at_time(
                    video_path,
                    timestamp
                )
            )

            if frame is None:

                print(
                    f"  Failed frame at "
                    f"{timestamp:.2f}s"
                )

                continue

            frame_height, frame_width = (
                frame.shape[:2]
            )

            if reference_width is None:

                reference_width = frame_width
                reference_height = frame_height

            sample_counter += 1

            sample_path = (
                samples_dir
                /
                f"sample_{sample_counter:03d}.jpg"
            )

            cv2.imwrite(
                str(sample_path),
                frame
            )

            sampled_frames.append(
                frame.copy()
            )

            detections = (
                detect_objects(
                    model,
                    frame
                )
            )

            for detection in detections:

                class_id = (
                    detection["class_id"]
                )

                detections_by_class[
                    class_id
                ].append(
                    detection
                )

            total_processed_frames += 1

    # ========================================================
    # VALIDATE
    # ========================================================

    if total_processed_frames == 0:

        raise RuntimeError(
            "\nNo frames were successfully processed.\n"
            "\nCheck:\n"
            "  1. Video codec compatibility\n"
            "  2. Video files\n"
            "  3. OpenCV video support\n"
        )

    if (
        reference_width is None
        or
        reference_height is None
    ):

        raise RuntimeError(
            "\nCould not determine frame size."
        )

    # ========================================================
    # BUILD ROIS
    # ========================================================

    rois = build_stable_rois(
        detections_by_class,
        reference_width,
        reference_height
    )

    if not rois:

        print()
        print(
            "No stable ROIs were found."
        )

    # ========================================================
    # CONFIGURATION
    # ========================================================

    config = {

        "status":
            "PENDING_REVIEW",

        "input": {
            "directory":
                str(input_dir),
        },

        "reference_frame": {
            "width":
                reference_width,

            "height":
                reference_height,
        },

        "discovery": {

            "method":
                "object_detection",

            "model":
                model_path.name,

            "confidence_threshold":
                YOLO_CONFIDENCE,

            "iou_threshold":
                YOLO_IOU,

            "discovery_video_count":
                len(discovery_videos),

            "frames_per_video":
                FRAMES_PER_VIDEO,

            "minimum_detection_count":
                MIN_DETECTION_COUNT,

        },

        "regions":
            rois,

    }

    save_json(
        config,
        config_path
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    save_summary_csv(
        rois,
        summary_path
    )

    # ========================================================
    # SAVE PREVIEW
    # ========================================================

    if sampled_frames:

        preview = draw_roi_preview(
            sampled_frames[0],
            rois
        )

        cv2.imwrite(
            str(preview_path),
            preview
        )

    # ========================================================
    # FINISH
    # ========================================================

    elapsed = (
        time.time()
        -
        start_time
    )

    print()
    print("=" * 80)
    print("ROI DISCOVERY COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Processed frames : "
        f"{total_processed_frames}"
    )

    print(
        f"Stable ROIs      : "
        f"{len(rois)}"
    )

    print(
        f"Configuration    : "
        f"{config_path}"
    )

    print(
        f"Summary          : "
        f"{summary_path}"
    )

    print(
        f"Preview          : "
        f"{preview_path}"
    )

    print(
        f"Elapsed time     : "
        f"{elapsed:.2f} seconds"
    )

    print()
    print(
        "Next step:"
    )

    print(
        "  Review the generated ROI configuration "
        "before using it in later stages."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Automatically discover candidate "
            "Regions of Interest from video data."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
            "Directory containing input videos."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory where discovery results "
            "will be saved."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=(
            "Path to the detection model."
        )
    )

    args = parser.parse_args()

    detect_roi(
        input_dir=args.input,
        output_dir=args.output,
        model_path=args.model
    )


if __name__ == "__main__":

    main()
