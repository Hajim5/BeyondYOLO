from pathlib import Path
import json
import csv
import argparse
import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

CATEGORIES = [
    "normal_baseline",
    "high_baseline",
    "low_baseline",
    "high",
    "low",
]


# ============================================================
# LOAD CONFIG
# ============================================================

def load_config(config_path):

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found:\n{config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# VIDEO DISCOVERY
# ============================================================

def find_videos(video_directories):

    extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
    }

    videos = []

    for directory in video_directories:

        directory = Path(directory)

        if not directory.exists():
            print(
                f"[WARNING] Video directory not found: {directory}"
            )
            continue

        for path in directory.rglob("*"):

            if (
                path.is_file()
                and path.suffix.lower() in extensions
            ):
                videos.append(path)

    return sorted(videos)


# ============================================================
# ROI
# ============================================================

def crop_roi(frame, roi):

    x = int(roi["x"])
    y = int(roi["y"])
    w = int(roi["width"])
    h = int(roi["height"])

    return frame[y:y + h, x:x + w]


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(image, settings):

    if image is None:
        return None

    result = image.copy()

    grayscale = settings.get(
        "grayscale",
        True
    )

    if grayscale:
        result = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2GRAY
        )

    blur = settings.get("blur")

    if blur:

        kernel = int(
            blur.get("kernel", 5)
        )

        sigma = float(
            blur.get("sigma", 0)
        )

        if kernel % 2 == 0:
            kernel += 1

        result = cv2.GaussianBlur(
            result,
            (kernel, kernel),
            sigma
        )

    canny = settings.get("canny")

    if canny:

        threshold1 = int(
            canny.get("threshold1", 75)
        )

        threshold2 = int(
            canny.get("threshold2", 175)
        )

        result = cv2.Canny(
            result,
            threshold1,
            threshold2
        )

    return result


# ============================================================
# TEMPLATE MATCHING
# ============================================================

def run_template_matching(
    image,
    template,
    method_name
):

    methods = {

        "TM_CCOEFF_NORMED":
            cv2.TM_CCOEFF_NORMED,

        "TM_CCORR_NORMED":
            cv2.TM_CCORR_NORMED,

        "TM_SQDIFF_NORMED":
            cv2.TM_SQDIFF_NORMED,

    }

    if method_name not in methods:

        raise ValueError(
            f"Unsupported template matching method:\n"
            f"{method_name}"
        )

    method = methods[method_name]

    result = cv2.matchTemplate(
        image,
        template,
        method
    )

    min_val, max_val, _, _ = cv2.minMaxLoc(
        result
    )

    if method == cv2.TM_SQDIFF_NORMED:

        score = 1.0 - min_val

    else:

        score = max_val

    return float(score)


# ============================================================
# CANDIDATE CATEGORY
# ============================================================

def classify_candidate(
    score,
    baseline,
    thresholds
):

    delta = score - baseline

    normal_range = float(
        thresholds.get(
            "normal_range",
            0.02
        )
    )

    baseline_range = float(
        thresholds.get(
            "baseline_range",
            0.05
        )
    )

    if abs(delta) <= normal_range:

        return (
            "normal_baseline",
            delta
        )

    if delta > 0:

        if delta <= baseline_range:

            return (
                "high_baseline",
                delta
            )

        return (
            "high",
            delta
        )

    if abs(delta) <= baseline_range:

        return (
            "low_baseline",
            delta
        )

    return (
        "low",
        delta
    )


# ============================================================
# TEMPORAL DIVERSITY
# ============================================================

def is_temporally_diverse(
    selected,
    candidate,
    minimum_gap_seconds
):

    for item in selected:

        if (
            item["video"]
            == candidate["video"]
        ):

            difference = abs(
                item["timestamp"]
                - candidate["timestamp"]
            )

            if difference < minimum_gap_seconds:
                return False

    return True


# ============================================================
# SELECT CANDIDATES
# ============================================================

def select_candidates(
    candidates,
    target_per_category,
    minimum_gap_seconds
):

    selected = []

    grouped = {
        category: []
        for category in CATEGORIES
    }

    for candidate in candidates:

        category = candidate["category"]

        if category in grouped:

            grouped[category].append(
                candidate
            )

    for category in CATEGORIES:

        category_candidates = grouped[
            category
        ]

        # Sort by strongest deviation
        category_candidates.sort(
            key=lambda x: abs(x["delta"]),
            reverse=True
        )

        category_selected = []

        for candidate in category_candidates:

            if len(category_selected) >= target_per_category:
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

def save_candidate_image(
    image,
    output_path
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    success = cv2.imwrite(
        str(output_path),
        image
    )

    return success


# ============================================================
# PROCESS VIDEO
# ============================================================

def process_video(
    video_path,
    roi_id,
    roi,
    template,
    matching_settings,
    baseline,
    thresholds,
    sample_interval_frames
):

    candidates = []

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            f"[WARNING] Could not open:\n"
            f"{video_path}"
        )

        return candidates

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    frame_index = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        if (
            frame_index
            % sample_interval_frames
            != 0
        ):

            frame_index += 1
            continue

        roi_image = crop_roi(
            frame,
            roi
        )

        if roi_image.size == 0:

            frame_index += 1
            continue

        processed_roi = preprocess_image(
            roi_image,
            matching_settings
        )

        score = run_template_matching(
            processed_roi,
            template,
            matching_settings.get(
                "method",
                "TM_CCOEFF_NORMED"
            )
        )

        category, delta = classify_candidate(
            score,
            baseline,
            thresholds
        )

        timestamp = (
            frame_index / fps
        )

        candidates.append({

            "video":
                str(video_path),

            "roi_id":
                roi_id,

            "frame_index":
                frame_index,

            "timestamp":
                timestamp,

            "score":
                score,

            "baseline":
                baseline,

            "delta":
                delta,

            "category":
                category,

            "image":
                roi_image.copy(),

        })

        frame_index += 1

    cap.release()

    return candidates


# ============================================================
# MAIN COLLECTION
# ============================================================

def collect_candidates(
    config_path,
    output_directory,
    target_per_category=100,
    sample_interval_frames=30,
    minimum_gap_seconds=5
):

    print("=" * 80)
    print("TEMPLATE MATCHING CANDIDATE COLLECTION")
    print("=" * 80)

    config = load_config(
        config_path
    )

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    video_directories = config.get(
        "video_directories",
        []
    )

    videos = find_videos(
        video_directories
    )

    if not videos:

        raise RuntimeError(
            "No videos found."
        )

    print(
        f"\nVideos found: {len(videos)}"
    )

    matching_settings = config.get(
        "template_matching",
        {}
    )

    preprocessing = matching_settings.get(
        "preprocessing",
        {}
    )

    matching_settings = {
        **matching_settings,
        **preprocessing,
    }

    thresholds = config.get(
        "candidate_thresholds",
        {}
    )

    machines = config.get(
        "machines",
        []
    )

    if not machines:

        raise RuntimeError(
            "No ROI or machine configuration found."
        )

    all_manifest_rows = []

    for machine in machines:

        roi_id = machine.get(
            "id",
            machine.get(
                "name",
                "unknown_roi"
            )
        )

        roi = machine.get("roi")

        if not roi:

            print(
                f"[WARNING] No ROI for {roi_id}"
            )

            continue

        baseline = machine.get(
            "baseline"
        )

        if baseline is None:

            print(
                f"[WARNING] No baseline for {roi_id}"
            )

            continue

        template_path = machine.get(
            "template_path"
        )

        if not template_path:

            print(
                f"[WARNING] No template for {roi_id}"
            )

            continue

        template_path = Path(
            template_path
        )

        if not template_path.exists():

            print(
                f"[WARNING] Template not found:\n"
                f"{template_path}"
            )

            continue

        template = cv2.imread(
            str(template_path)
        )

        if template is None:

            print(
                f"[WARNING] Could not load template:\n"
                f"{template_path}"
            )

            continue

        template = preprocess_image(
            template,
            matching_settings
        )

        print("\n" + "-" * 80)
        print(
            f"Processing ROI: {roi_id}"
        )
        print("-" * 80)

        roi_candidates = []

        for index, video_path in enumerate(
            videos,
            start=1
        ):

            print(
                f"[{index}/{len(videos)}] "
                f"{video_path.name}"
            )

            video_candidates = process_video(
                video_path=video_path,
                roi_id=roi_id,
                roi=roi,
                template=template,
                matching_settings=matching_settings,
                baseline=float(baseline),
                thresholds=thresholds,
                sample_interval_frames=
                    sample_interval_frames
            )

            roi_candidates.extend(
                video_candidates
            )

        selected = select_candidates(
            candidates=roi_candidates,
            target_per_category=
                target_per_category,
            minimum_gap_seconds=
                minimum_gap_seconds
        )

        print(
            f"Candidates selected: "
            f"{len(selected)}"
        )

        for index, candidate in enumerate(
            selected,
            start=1
        ):

            category = candidate[
                "category"
            ]

            filename = (
                f"{roi_id}_"
                f"{category}_"
                f"{index:05d}.jpg"
            )

            image_path = (
                output_directory
                / roi_id
                / category
                / filename
            )

            success = save_candidate_image(
                candidate["image"],
                image_path
            )

            if not success:

                continue

            all_manifest_rows.append({

                "image_path":
                    str(image_path),

                "source_video":
                    candidate["video"],

                "roi_id":
                    roi_id,

                "frame_index":
                    candidate["frame_index"],

                "timestamp":
                    candidate["timestamp"],

                "matching_score":
                    candidate["score"],

                "baseline":
                    candidate["baseline"],

                "delta":
                    candidate["delta"],

                "candidate_category":
                    category,

                "manual_label":
                    "",

            })

    # ========================================================
    # SAVE MANIFEST
    # ========================================================

    manifest_path = (
        output_directory
        / "manifest.csv"
    )

    if all_manifest_rows:

        with open(
            manifest_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=
                    all_manifest_rows[0].keys()
            )

            writer.writeheader()

            writer.writerows(
                all_manifest_rows
            )

    # ========================================================
    # COLLECTION INFO
    # ========================================================

    info = {

        "config_path":
            str(config_path),

        "target_per_category":
            target_per_category,

        "sample_interval_frames":
            sample_interval_frames,

        "minimum_gap_seconds":
            minimum_gap_seconds,

        "total_images":
            len(all_manifest_rows),

        "categories":
            CATEGORIES,

    }

    info_path = (
        output_directory
        / "collection_info.json"
    )

    with open(
        info_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            info,
            f,
            indent=4
        )

    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)

    print(
        f"Output:\n{output_directory}"
    )

    print(
        f"Manifest:\n{manifest_path}"
    )

    print(
        f"Total images: "
        f"{len(all_manifest_rows)}"
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=
        "General template matching candidate collector"
    )

    parser.add_argument(
        "--config",
        required=True,
        help=
        "Path to configuration JSON file"
    )

    parser.add_argument(
        "--output",
        default=
        "candidate_dataset",
        help=
        "Output directory"
    )

    parser.add_argument(
        "--target",
        type=int,
        default=100,
        help=
        "Target images per category"
    )

    parser.add_argument(
        "--sample-interval",
        type=int,
        default=30,
        help=
        "Process every N frames"
    )

    parser.add_argument(
        "--minimum-gap",
        type=float,
        default=5.0,
        help=
        "Minimum seconds between selected samples"
    )

    args = parser.parse_args()

    collect_candidates(
        config_path=args.config,
        output_directory=args.output,
        target_per_category=args.target,
        sample_interval_frames=
            args.sample_interval,
        minimum_gap_seconds=
            args.minimum_gap
    )


if __name__ == "__main__":
    main()
