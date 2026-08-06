from pathlib import Path
import json
import csv
import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"C:\Users\PC-1\Downloads\sensory\full_rnd")

CONFIG_PATH = BASE_DIR / "config.json"

OUTPUT_DIR = BASE_DIR / "results" / "ground_truth"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANNOTATION_CSV = OUTPUT_DIR / "annotations.csv"
INTERVAL_CSV = OUTPUT_DIR / "annotation_intervals.csv"


VIDEO_MACHINES = {
    "normal-op1_E2_S1.mp4": [
        "E2", "S1"
    ],

    "normal-op2_E1_S3_I1.mp4": [
        "E1", "S3", "I1"
    ],

    "normal-op3_E1_E2_S2_S3_G1.mp4": [
        "E1", "E2", "S2", "S3", "G1"
    ],

    "normal-op4_E1_S4_G1.mp4": [
        "E1", "S4", "G1"
    ],

    "normal-op5_E1_E2_S5_S6.mp4": [
        "E1", "E2", "S5", "S6"
    ],

    "normal_op7_E2_G1_S6_I1.mp4": [
        "E2", "G1", "S6", "I1"
    ],
}


# ============================================================
# ANNOTATION SETTINGS
# ============================================================

# ------------------------------------------------------------
# Playback speed
#
# Larger number = slower playback.
#
# ~30 ms  = approximately normal speed
# ~60 ms  = approximately 0.5x
# ~80 ms  = approximately 0.4x
# ~100 ms = approximately 0.3x
# ------------------------------------------------------------

DEFAULT_PLAY_DELAY_MS = 80

MIN_PLAY_DELAY_MS = 20
MAX_PLAY_DELAY_MS = 300

SPEED_CHANGE_MS = 20


# Navigation
SMALL_SEEK_FRAMES = 5
LARGE_SEEK_FRAMES = 30


# Display
DISPLAY_MAX_WIDTH = 1200
ROI_DISPLAY_SIZE = 550


# ============================================================
# LOAD CONFIG
# ============================================================

def load_config():

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config not found:\n{CONFIG_PATH}"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    roi_map = {}

    for item in data:

        machine_id = item["id"]

        roi_map[machine_id] = {
            "id": machine_id,
            "class_name": item.get("class_name", ""),
            "x1": int(item["x1"]),
            "y1": int(item["y1"]),
            "x2": int(item["x2"]),
            "y2": int(item["y2"]),
        }

    return roi_map


# ============================================================
# LOAD EXISTING ANNOTATIONS
# ============================================================

def load_existing_annotations():

    annotations = {}

    if not ANNOTATION_CSV.exists():
        return annotations

    print()
    print("Loading existing annotations:")
    print(ANNOTATION_CSV)

    with open(
        ANNOTATION_CSV,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            key = (
                row["video"],
                row["machine_id"],
                int(row["frame"])
            )

            annotations[key] = row["label"]

    print(
        f"Loaded {len(annotations):,} existing frame labels."
    )

    return annotations


# ============================================================
# SAVE ANNOTATIONS
# ============================================================

def save_annotations(annotation_dict):

    rows = []

    for (video, machine, frame), label in annotation_dict.items():

        rows.append({
            "video": video,
            "machine_id": machine,
            "frame": frame,
            "label": label,
        })

    rows.sort(
        key=lambda x: (
            x["video"],
            x["machine_id"],
            x["frame"]
        )
    )

    with open(
        ANNOTATION_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "machine_id",
                "frame",
                "label",
            ]
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# BUILD INTERVALS
# ============================================================

def build_intervals(annotation_dict):

    grouped = {}

    for (video, machine, frame), label in annotation_dict.items():

        key = (
            video,
            machine
        )

        grouped.setdefault(
            key,
            []
        )

        grouped[key].append(
            (
                frame,
                label
            )
        )

    intervals = []

    for (
        video,
        machine
    ), items in grouped.items():

        items.sort(
            key=lambda x: x[0]
        )

        if not items:
            continue

        start_frame = items[0][0]
        previous_frame = items[0][0]
        current_label = items[0][1]

        for frame, label in items[1:]:

            if (
                label != current_label
                or frame != previous_frame + 1
            ):

                intervals.append({
                    "video": video,
                    "machine_id": machine,
                    "start_frame": start_frame,
                    "end_frame": previous_frame,
                    "label": current_label,
                })

                start_frame = frame
                current_label = label

            previous_frame = frame

        intervals.append({
            "video": video,
            "machine_id": machine,
            "start_frame": start_frame,
            "end_frame": previous_frame,
            "label": current_label,
        })

    intervals.sort(
        key=lambda x: (
            x["video"],
            x["machine_id"],
            x["start_frame"]
        )
    )

    return intervals


def save_intervals(annotation_dict):

    intervals = build_intervals(
        annotation_dict
    )

    with open(
        INTERVAL_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "machine_id",
                "start_frame",
                "end_frame",
                "label",
            ]
        )

        writer.writeheader()
        writer.writerows(intervals)


def save_all(annotation_dict):

    save_annotations(
        annotation_dict
    )

    save_intervals(
        annotation_dict
    )


# ============================================================
# DISPLAY UTILITIES
# ============================================================

def resize_keep_aspect(
    image,
    max_width
):

    h, w = image.shape[:2]

    if w <= max_width:
        return image

    scale = (
        max_width / w
    )

    new_w = int(
        w * scale
    )

    new_h = int(
        h * scale
    )

    return cv2.resize(
        image,
        (
            new_w,
            new_h
        ),
        interpolation=cv2.INTER_AREA
    )


def crop_roi(
    frame,
    roi
):

    h, w = frame.shape[:2]

    x1 = max(
        0,
        min(
            roi["x1"],
            w - 1
        )
    )

    y1 = max(
        0,
        min(
            roi["y1"],
            h - 1
        )
    )

    x2 = max(
        x1 + 1,
        min(
            roi["x2"],
            w
        )
    )

    y2 = max(
        y1 + 1,
        min(
            roi["y2"],
            h
        )
    )

    return frame[
        y1:y2,
        x1:x2
    ]


# ============================================================
# LABEL DISPLAY
# ============================================================

def label_color(label):

    if label == "EMPTY":
        return (0, 255, 0)

    if label == "CUP":
        return (0, 0, 255)

    if label == "SKIP":
        return (0, 255, 255)

    return (255, 255, 255)


def make_roi_display(
    roi_image,
    machine,
    label,
    frame_number
):

    canvas = np.zeros(
        (
            ROI_DISPLAY_SIZE,
            ROI_DISPLAY_SIZE,
            3
        ),
        dtype=np.uint8
    )

    if roi_image.size == 0:

        cv2.putText(
            canvas,
            "INVALID ROI",
            (40, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        return canvas

    # Reserve top area for information
    info_height = 100

    available_height = (
        ROI_DISPLAY_SIZE
        - info_height
    )

    h, w = roi_image.shape[:2]

    scale = min(
        ROI_DISPLAY_SIZE / w,
        available_height / h
    )

    new_w = max(
        1,
        int(w * scale)
    )

    new_h = max(
        1,
        int(h * scale)
    )

    resized = cv2.resize(
        roi_image,
        (
            new_w,
            new_h
        ),
        interpolation=cv2.INTER_NEAREST
    )

    x_offset = (
        ROI_DISPLAY_SIZE
        - new_w
    ) // 2

    y_offset = (
        info_height
        +
        (
            available_height
            - new_h
        ) // 2
    )

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    color = label_color(
        label
    )

    cv2.putText(
        canvas,
        f"TARGET: {machine}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        canvas,
        f"STATE: {label}",
        (10, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2
    )

    cv2.putText(
        canvas,
        f"FRAME: {frame_number}",
        (10, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    return canvas


# ============================================================
# DELETE EXISTING LABELS FROM FRAME FORWARD
# ============================================================

def clear_annotations_from_frame(
    annotations,
    video_name,
    machine_id,
    start_frame
):

    keys_to_delete = []

    for key in annotations:

        video, machine, frame = key

        if (
            video == video_name
            and machine == machine_id
            and frame >= start_frame
        ):
            keys_to_delete.append(
                key
            )

    for key in keys_to_delete:
        del annotations[key]

    return len(
        keys_to_delete
    )


# ============================================================
# GET EXISTING MACHINE PROGRESS
# ============================================================

def get_machine_progress(
    annotations,
    video_name,
    machine_id
):

    frames = []

    for (
        video,
        machine,
        frame
    ), label in annotations.items():

        if (
            video == video_name
            and machine == machine_id
        ):
            frames.append(
                frame
            )

    if not frames:
        return None

    return max(
        frames
    )


# ============================================================
# ANNOTATION SESSION
# ============================================================

def annotate_machine(
    video_path,
    machine_id,
    roi,
    annotations,
    start_frame=0
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            f"Cannot open video:\n"
            f"{video_path}"
        )

        return False

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    play_delay = (
        DEFAULT_PLAY_DELAY_MS
    )

    current_frame = max(
        0,
        min(
            start_frame,
            total_frames - 1
        )
    )

    paused = True

    current_label = (
        "UNLABELED"
    )

    # --------------------------------------------------------
    # Recover label immediately before start position
    # --------------------------------------------------------

    for search_frame in range(
        current_frame,
        -1,
        -1
    ):

        existing_key = (
            video_path.name,
            machine_id,
            search_frame
        )

        if existing_key in annotations:

            current_label = (
                annotations[
                    existing_key
                ]
            )

            break

    print()
    print("=" * 80)
    print(
        f"VIDEO   : {video_path.name}"
    )
    print(
        f"MACHINE : {machine_id}"
    )
    print(
        f"FRAMES  : {total_frames}"
    )
    print(
        f"FPS     : {fps:.2f}"
    )
    print(
        f"START   : {current_frame}"
    )
    print("=" * 80)

    print("""
LABELS
------------------------------------------------
E = EMPTY
C = CUP
S = SKIP / TRANSITION / ROBOT / OCCLUDED


PLAYBACK
------------------------------------------------
SPACE = Pause / Play

+     = Slower
-     = Faster


NAVIGATION
------------------------------------------------
A = Back 5 frames
D = Forward 5 frames

J = Back 1 frame
L = Forward 1 frame

Z = Back 30 frames
X = Forward 30 frames


EDITING
------------------------------------------------
E/C/S changes the state from the CURRENT frame.

BACK UP FIRST if you noticed the state change late.

K = Clear annotations from current frame forward


SESSION
------------------------------------------------
N = Finish current machine / next machine
Q = Save and quit entire program
""")

    while True:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            current_frame
        )

        ret, frame = cap.read()

        if not ret:
            print(
                "\nReached end of video."
            )
            break

        annotation_key = (
            video_path.name,
            machine_id,
            current_frame
        )

        # ----------------------------------------------------
        # Determine displayed state
        # ----------------------------------------------------

        existing_label = (
            annotations.get(
                annotation_key
            )
        )

        if existing_label is not None:

            display_label = (
                existing_label
            )

        else:

            display_label = (
                current_label
            )

        # ----------------------------------------------------
        # Main video
        # ----------------------------------------------------

        display = frame.copy()

        x1 = roi["x1"]
        y1 = roi["y1"]
        x2 = roi["x2"]
        y2 = roi["y2"]

        roi_color = (
            label_color(
                display_label
            )
        )

        cv2.rectangle(
            display,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            roi_color,
            5
        )

        cv2.putText(
            display,
            f"TARGET: {machine_id}",
            (
                x1,
                max(
                    40,
                    y1 - 15
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            roi_color,
            3
        )

        timestamp = (
            current_frame / fps
            if fps > 0
            else 0
        )

        playback_estimate = (
            30.0 / play_delay
        )

        status = (
            "PAUSED"
            if paused
            else "PLAYING"
        )

        info1 = (
            f"Frame "
            f"{current_frame}/"
            f"{total_frames - 1}"
        )

        info2 = (
            f"Time: "
            f"{timestamp:.2f}s"
        )

        info3 = (
            f"State: "
            f"{display_label}"
        )

        info4 = (
            f"{status} | "
            f"Delay: {play_delay} ms "
            f"| Approx speed: "
            f"{playback_estimate:.2f}x"
        )

        cv2.putText(
            display,
            info1,
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            info2,
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            info3,
            (30, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            roi_color,
            2
        )

        cv2.putText(
            display,
            info4,
            (30, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        display = resize_keep_aspect(
            display,
            DISPLAY_MAX_WIDTH
        )

        # ----------------------------------------------------
        # Enlarged ROI
        # ----------------------------------------------------

        roi_image = crop_roi(
            frame,
            roi
        )

        roi_display = make_roi_display(
            roi_image,
            machine_id,
            display_label,
            current_frame
        )

        cv2.imshow(
            "Ground Truth - Full Video",
            display
        )

        cv2.imshow(
            "Ground Truth - Target ROI",
            roi_display
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Only write current active state when playing.
        #
        # This prevents simply navigating backward from
        # accidentally overwriting previous annotations.
        # ----------------------------------------------------

        if (
            not paused
            and current_label
            != "UNLABELED"
        ):

            annotations[
                annotation_key
            ] = current_label

        # ----------------------------------------------------
        # Keyboard
        # ----------------------------------------------------

        delay = (
            0
            if paused
            else play_delay
        )

        key_code = (
            cv2.waitKey(delay)
            & 0xFF
        )

        # ====================================================
        # LABEL: EMPTY
        # ====================================================

        if key_code in (
            ord("e"),
            ord("E")
        ):

            current_label = "EMPTY"

            annotations[
                annotation_key
            ] = current_label

            print(
                f"[{current_frame}] "
                f"{machine_id} -> EMPTY"
            )

        # ====================================================
        # LABEL: CUP
        # ====================================================

        elif key_code in (
            ord("c"),
            ord("C")
        ):

            current_label = "CUP"

            annotations[
                annotation_key
            ] = current_label

            print(
                f"[{current_frame}] "
                f"{machine_id} -> CUP"
            )

        # ====================================================
        # LABEL: SKIP
        # ====================================================

        elif key_code in (
            ord("s"),
            ord("S")
        ):

            current_label = "SKIP"

            annotations[
                annotation_key
            ] = current_label

            print(
                f"[{current_frame}] "
                f"{machine_id} -> SKIP"
            )

        # ====================================================
        # PAUSE / PLAY
        # ====================================================

        elif key_code == 32:

            paused = not paused

            print(
                "PAUSED"
                if paused
                else "PLAYING"
            )

        # ====================================================
        # SLOWER
        # ====================================================

        elif key_code in (
            ord("+"),
            ord("=")
        ):

            play_delay = min(
                MAX_PLAY_DELAY_MS,
                play_delay
                + SPEED_CHANGE_MS
            )

            print(
                f"Playback slower: "
                f"{play_delay} ms"
            )

        # ====================================================
        # FASTER
        # ====================================================

        elif key_code in (
            ord("-"),
            ord("_")
        ):

            play_delay = max(
                MIN_PLAY_DELAY_MS,
                play_delay
                - SPEED_CHANGE_MS
            )

            print(
                f"Playback faster: "
                f"{play_delay} ms"
            )

        # ====================================================
        # BACK 5
        # ====================================================

        elif key_code in (
            ord("a"),
            ord("A")
        ):

            paused = True

            current_frame = max(
                0,
                current_frame
                - SMALL_SEEK_FRAMES
            )

            continue

        # ====================================================
        # FORWARD 5
        # ====================================================

        elif key_code in (
            ord("d"),
            ord("D")
        ):

            paused = True

            current_frame = min(
                total_frames - 1,
                current_frame
                + SMALL_SEEK_FRAMES
            )

            continue

        # ====================================================
        # BACK 1
        # ====================================================

        elif key_code in (
            ord("j"),
            ord("J")
        ):

            paused = True

            current_frame = max(
                0,
                current_frame - 1
            )

            continue

        # ====================================================
        # FORWARD 1
        # ====================================================

        elif key_code in (
            ord("l"),
            ord("L")
        ):

            paused = True

            current_frame = min(
                total_frames - 1,
                current_frame + 1
            )

            continue

        # ====================================================
        # BACK 30
        # ====================================================

        elif key_code in (
            ord("z"),
            ord("Z")
        ):

            paused = True

            current_frame = max(
                0,
                current_frame
                - LARGE_SEEK_FRAMES
            )

            continue

        # ====================================================
        # FORWARD 30
        # ====================================================

        elif key_code in (
            ord("x"),
            ord("X")
        ):

            paused = True

            current_frame = min(
                total_frames - 1,
                current_frame
                + LARGE_SEEK_FRAMES
            )

            continue

        # ====================================================
        # CLEAR FROM CURRENT FRAME FORWARD
        # ====================================================

        elif key_code in (
            ord("k"),
            ord("K")
        ):

            paused = True

            removed = (
                clear_annotations_from_frame(
                    annotations,
                    video_path.name,
                    machine_id,
                    current_frame
                )
            )

            current_label = (
                "UNLABELED"
            )

            save_all(
                annotations
            )

            print(
                f"Cleared {removed} "
                f"annotations from "
                f"frame {current_frame} forward."
            )

        # ====================================================
        # NEXT MACHINE
        # ====================================================

        elif key_code in (
            ord("n"),
            ord("N")
        ):

            save_all(
                annotations
            )

            print(
                f"Saved {machine_id}."
            )

            cap.release()

            cv2.destroyAllWindows()

            return True

        # ====================================================
        # QUIT
        # ====================================================

        elif key_code in (
            ord("q"),
            ord("Q")
        ):

            save_all(
                annotations
            )

            print(
                "Annotations saved."
            )

            cap.release()

            cv2.destroyAllWindows()

            return False

        # ----------------------------------------------------
        # Advance
        # ----------------------------------------------------

        if not paused:

            current_frame += 1

            if (
                current_frame
                >= total_frames
            ):

                break

    save_all(
        annotations
    )

    cap.release()

    cv2.destroyAllWindows()

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    roi_map = load_config()

    annotations = (
        load_existing_annotations()
    )

    print()
    print("=" * 80)
    print("GROUND TRUTH ANNOTATION")
    print("=" * 80)

    print()
    print("Config:")
    print(CONFIG_PATH)

    print()
    print("Output:")
    print(OUTPUT_DIR)

    for (
        video_name,
        machines
    ) in VIDEO_MACHINES.items():

        video_path = (
            BASE_DIR
            / video_name
        )

        if not video_path.exists():

            print()
            print(
                "[WARNING] "
                "Missing video:"
            )

            print(
                video_path
            )

            continue

        for machine_id in machines:

            if machine_id not in roi_map:

                print(
                    f"[WARNING] "
                    f"{machine_id} not "
                    f"found in config.json"
                )

                continue

            progress = (
                get_machine_progress(
                    annotations,
                    video_name,
                    machine_id
                )
            )

            print()
            print("#" * 80)

            print(
                f"TARGET: "
                f"{video_name} "
                f"-> {machine_id}"
            )

            if progress is not None:

                print(
                    f"Existing annotation "
                    f"up to frame: "
                    f"{progress}"
                )

            else:

                print(
                    "No existing annotations."
                )

            print("#" * 80)

            if progress is None:

                user_input = input(
                    "\n"
                    "ENTER = Start annotation\n"
                    "skip  = Skip this machine\n"
                    "quit  = Save and quit\n"
                    "\nChoice: "
                ).strip().lower()

                start_frame = 0

            else:

                user_input = input(
                    "\n"
                    "ENTER    = Start from beginning\n"
                    "resume   = Resume near last annotated frame\n"
                    "skip     = Skip this machine\n"
                    "quit     = Save and quit\n"
                    "\nChoice: "
                ).strip().lower()

                if user_input == "resume":

                    # Go back a little so the user
                    # can visually verify the context.
                    start_frame = max(
                        0,
                        progress - 60
                    )

                else:

                    start_frame = 0

            if user_input == "quit":

                save_all(
                    annotations
                )

                print(
                    "Saved."
                )

                return

            if user_input == "skip":
                continue

            continue_program = (
                annotate_machine(
                    video_path,
                    machine_id,
                    roi_map[
                        machine_id
                    ],
                    annotations,
                    start_frame
                )
            )

            if not continue_program:

                print(
                    "Annotations saved."
                )

                return

    save_all(
        annotations
    )

    print()
    print("=" * 80)
    print("ANNOTATION COMPLETE")
    print("=" * 80)

    print()
    print(
        "Frame annotations:"
    )

    print(
        ANNOTATION_CSV
    )

    print()
    print(
        "Annotation intervals:"
    )

    print(
        INTERVAL_CSV
    )


if __name__ == "__main__":
    main()