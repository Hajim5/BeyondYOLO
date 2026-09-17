"""
detect_station.py

PURPOSE:
--------
One-time machine / ROI discovery for a station.
Runs YOLO to discover stable machine regions and exports:
    station_layout_<station>.json
"""

from pathlib import Path
from collections import defaultdict
import csv
import json
import time

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ============================================================
# PROJECT & DIRECTORY CONFIGURATION
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent
STATIONS_DIR = Path(r"C:\Users\PC-1\Downloads\sensory\1_station_video")
YOLO_MODEL_PATH = PROJECT_DIR / "best.pt"

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".m4v"}

# ============================================================
# DETECTION & DISCOVERY TUNING
# ============================================================
YOLO_CONFIDENCE = 0.20  # Lowered slightly to ensure initial proposal recall
YOLO_IOU = 0.45

DISCOVERY_VIDEO_COUNT = 8
FRAMES_PER_VIDEO = 5
VIDEO_EDGE_MARGIN_SECONDS = 2.0  # Kept lower to handle short clips

# Minimum detections needed across all sampled frames to form an ROI
MIN_DETECTION_COUNT = 1
POSITION_OUTLIER_FACTOR = 3.0

ROI_PADDING_X = 0
ROI_PADDING_Y = 0

PREVIEW_FONT_SCALE = 0.65
PREVIEW_THICKNESS = 2

# ============================================================
# HELPERS
# ============================================================
def ensure_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def save_json(data: dict, path: Path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

def find_station_directory(station_name: str) -> Path:
    direct = STATIONS_DIR / station_name
    if direct.exists() and direct.is_dir():
        return direct

    if STATIONS_DIR.exists():
        target = station_name.strip().lower()
        for path in STATIONS_DIR.iterdir():
            if path.is_dir() and path.name.lower() == target:
                return path

    raise FileNotFoundError(f"\nStation directory not found: {direct}\n")

def find_station_videos(station_dir: Path):
    videos = [
        path for path in station_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(videos)

def select_discovery_videos(videos, count):
    if len(videos) <= count:
        return videos
    indices = np.round(np.linspace(0, len(videos) - 1, count)).astype(int)
    selected = []
    used = set()
    for index in indices:
        if index not in used:
            used.add(index)
            selected.append(videos[index])
    return selected

def get_video_info(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = (frame_count / fps) if fps and fps > 0 else 0.0

    cap.release()
    return {
        "fps": float(fps) if fps else 0.0,
        "frame_count": int(frame_count),
        "duration": float(duration),
        "width": width,
        "height": height,
    }

def get_sample_timestamps(duration, count):
    if duration <= 0:
        return [0.0]

    margin = min(VIDEO_EDGE_MARGIN_SECONDS, duration * 0.10)
    start = margin
    end = duration - margin

    if end <= start:
        start = 0.0
        end = duration

    if count <= 1:
        return [(start + end) / 2.0]

    return [float(v) for v in np.linspace(start, end, count)]

def read_frame_at_time(video_path: Path, timestamp: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def load_yolo_model():
    if YOLO is None:
        raise ImportError("ultralytics is not installed. Run: pip install ultralytics")
    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(f"YOLO model not found: {YOLO_MODEL_PATH}")
    return YOLO(str(YOLO_MODEL_PATH))

def normalize_machine_label(class_name):
    return str(class_name).strip().upper()

def detect_frame(model, frame):
    results = model.predict(
        source=frame,
        conf=YOLO_CONFIDENCE,
        iou=YOLO_IOU,
        verbose=False
    )
    detections = []
    if not results or results[0].boxes is None:
        return detections

    result = results[0]
    names = result.names

    for box in result.boxes:
        xyxy = box.xyxy[0].detach().cpu().numpy()
        confidence = float(box.conf[0].detach().cpu().item())
        class_id = int(box.cls[0].detach().cpu().item())

        class_name = names.get(class_id, class_id) if isinstance(names, dict) else names[class_id]
        machine_id = normalize_machine_label(class_name)

        x1, y1, x2, y2 = [float(v) for v in xyxy]
        detections.append({
            "machine_id": machine_id,
            "class_name": str(class_name),
            "class_id": class_id,
            "confidence": confidence,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": x2 - x1,
            "height": y2 - y1,
            "center_x": (x1 + x2) / 2.0,
            "center_y": (y1 + y2) / 2.0,
        })
    return detections

def draw_raw_detections(frame, detections):
    output = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = int(round(d["x1"])), int(round(d["y1"])), int(round(d["x2"])), int(round(d["y2"]))
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 255), 2)
        label = f"{d['machine_id']} {d['confidence']:.2f}"
        cv2.putText(output, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return output

def build_stable_roi(detections, frame_width, frame_height):
    x1_values = [d["x1"] for d in detections]
    y1_values = [d["y1"] for d in detections]
    x2_values = [d["x2"] for d in detections]
    y2_values = [d["y2"] for d in detections]

    x1 = max(0, min(frame_width - 1, int(round(float(np.median(x1_values)))) - ROI_PADDING_X))
    y1 = max(0, min(frame_height - 1, int(round(float(np.median(y1_values)))) - ROI_PADDING_Y))
    x2 = max(0, min(frame_width, int(round(float(np.median(x2_values)))) + ROI_PADDING_X))
    y2 = max(0, min(frame_height, int(round(float(np.median(y2_values)))) + ROI_PADDING_Y))

    confs = [d["confidence"] for d in detections]
    return {
        "roi": [x1, y1, x2, y2],
        "raw_detection_count": len(detections),
        "used_detection_count": len(detections),
        "mean_confidence": float(np.mean(confs)),
        "median_confidence": float(np.median(confs)),
        "min_confidence": float(np.min(confs)),
        "max_confidence": float(np.max(confs)),
    }

def draw_final_layout(frame, machines):
    output = frame.copy()
    for machine_id, machine in machines.items():
        x1, y1, x2, y2 = machine["roi"]
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 255), 3)
        label = f"{machine_id} [{machine['used_detection_count']}]"
        cv2.putText(output, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, PREVIEW_FONT_SCALE, (255, 255, 255), PREVIEW_THICKNESS, cv2.LINE_AA)
    return output

def save_detection_summary(machines, output_path: Path):
    fieldnames = [
        "machine_id", "x1", "y1", "x2", "y2",
        "raw_detection_count", "used_detection_count",
        "mean_confidence", "median_confidence",
        "min_confidence", "max_confidence", "review_status",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for machine_id, machine in machines.items():
            x1, y1, x2, y2 = machine["roi"]
            writer.writerow({
                "machine_id": machine_id,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "raw_detection_count": machine["raw_detection_count"],
                "used_detection_count": machine["used_detection_count"],
                "mean_confidence": f"{machine['mean_confidence']:.6f}",
                "median_confidence": f"{machine['median_confidence']:.6f}",
                "min_confidence": f"{machine['min_confidence']:.6f}",
                "max_confidence": f"{machine['max_confidence']:.6f}",
                "review_status": machine["review_status"],
            })

# ============================================================
# MAIN DETECTION PIPELINE
# ============================================================
def detect_station(station_name: str):
    start_time = time.time()
    station_dir = find_station_directory(station_name)
    detection_dir = station_dir / "detection"
    samples_dir = detection_dir / "samples"

    ensure_directory(detection_dir)
    ensure_directory(samples_dir)

    layout_path = detection_dir / f"station_layout_{station_name}.json"
    summary_path = detection_dir / "detection_summary.csv"
    preview_path = detection_dir / "detection_preview.jpg"

    videos = find_station_videos(station_dir)
    if not videos:
        raise FileNotFoundError(f"No video files found in station directory: {station_dir}")

    discovery_videos = select_discovery_videos(videos, DISCOVERY_VIDEO_COUNT)
    model = load_yolo_model()

    detections_by_machine = defaultdict(list)
    sampled_frames = []
    total_yolo_frames = 0
    ref_w, ref_h = None, None

    sample_counter = 0
    for v_idx, v_path in enumerate(discovery_videos, start=1):
        info = get_video_info(v_path)
        if info is None:
            continue

        if ref_w is None:
            ref_w, ref_h = info["width"], info["height"]

        timestamps = get_sample_timestamps(info["duration"], FRAMES_PER_VIDEO)
        for ts in timestamps:
            frame = read_frame_at_time(v_path, ts)
            if frame is None:
                continue

            total_yolo_frames += 1
            dets = detect_frame(model, frame)

            for d in dets:
                item = dict(d)
                item.update({"video": v_path.name, "timestamp_seconds": float(ts)})
                detections_by_machine[d["machine_id"]].append(item)

            sample_counter += 1
            annotated = draw_raw_detections(frame, dets)
            sample_name = f"sample_{sample_counter:03d}_{v_path.stem}_{ts:.0f}s.jpg"
            cv2.imwrite(str(samples_dir / sample_name), annotated)

            sampled_frames.append({
                "video": v_path.name,
                "timestamp_seconds": float(ts),
                "frame": frame,
                "detections": dets,
            })

    if total_yolo_frames == 0 or not sampled_frames:
        raise RuntimeError("Failed to extract any readable video frames.")

    # Build stable machines
    machines = {}
    for m_id, dets in detections_by_machine.items():
        if len(dets) >= MIN_DETECTION_COUNT:
            stable = build_stable_roi(dets, ref_w, ref_h)
            machines[m_id] = {
                "id": m_id,
                "roi": stable["roi"],
                "raw_detection_count": stable["raw_detection_count"],
                "used_detection_count": stable["used_detection_count"],
                "mean_confidence": stable["mean_confidence"],
                "median_confidence": stable["median_confidence"],
                "min_confidence": stable["min_confidence"],
                "max_confidence": stable["max_confidence"],
                "review_status": "PENDING",
            }

    # Fallback placeholder so downstream review script NEVER crashes with missing layout
    if not machines:
        print("[WARNING] YOLO found no machines. Generating placeholder template so layout file exists.")
        fallback_roi = [int(ref_w * 0.25), int(ref_h * 0.25), int(ref_w * 0.75), int(ref_h * 0.75)]
        machines["M1"] = {
            "id": "M1",
            "roi": fallback_roi,
            "raw_detection_count": 0,
            "used_detection_count": 0,
            "mean_confidence": 0.0,
            "median_confidence": 0.0,
            "min_confidence": 0.0,
            "max_confidence": 0.0,
            "review_status": "MANUAL_ADJUSTMENT_REQUIRED",
        }

    # Render Preview
    best_sample = sampled_frames[0]
    final_preview = draw_final_layout(best_sample["frame"], machines)
    cv2.imwrite(str(preview_path), final_preview)

    # Save Layout JSON
    layout_data = {
        "station": station_name,
        "layout_version": 1,
        "status": "PENDING_REVIEW",
        "source": "YOLO_DISCOVERY",
        "frame_width": ref_w,
        "frame_height": ref_h,
        "preview_source": {
            "video": best_sample["video"],
            "timestamp_seconds": best_sample["timestamp_seconds"],
        },
        "machines": machines,
        "setup_time_seconds": time.time() - start_time,
    }
    save_json(layout_data, layout_path)
    save_detection_summary(machines, summary_path)

    print(f"\n[SUCCESS] Station layout generated at:\n  {layout_path}")

def main():
    station_name = input("Station name: ").strip()
    if not station_name:
        raise ValueError("Station name cannot be empty.")
    detect_station(station_name)

if __name__ == "__main__":
    main()