"""
main_v4.py

Coffee Station Interactive Tester V4

Architecture:
- Independent ROI State Machine
- ROIManager V2
- Classical CV Cup Detection
- No global InspectionSession

Author: Ahmad Hazim
"""
import os
import time
import cv2

from roi import ROIManager
from classifier import CupClassifier

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

VIDEO_PATH = r"C:\Users\PC-1\Downloads\sensory\template_method\testvideo_ph4\ph4.Avi"
CONFIG_PATH = "config_ph4_thresh.json"

DISPLAY_WIDTH = 1800
DISPLAY_HEIGHT = 1013

DETECTION_INTERVAL = 6

CLEAR_DELAY = 5
SUCCESS_DISPLAY_TIME = 2

DETECTION_INTERVAL = 6

CONFIRM_CUP = 5
CONFIRM_EMPTY = 5

CONFIRM_HOLD_TIME = 1.0    # seconds

CLEAR_DELAY = 5
SUCCESS_DISPLAY_TIME = 2

# --------------------------------------------------
# UI COLORS
# --------------------------------------------------

COLOR = {

    "IDLE": (255, 0, 0),

    "MONITORING": (0, 255, 255),

    "FAILED": (0, 0, 255),

    "SUCCESS": (0, 255, 0),

}

# --------------------------------------------------
# INITIALIZATION
# --------------------------------------------------

roi_manager = ROIManager(CONFIG_PATH)

classifier = CupClassifier()

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    raise FileNotFoundError(

        f"Cannot open video:\n{VIDEO_PATH}"

    )

paused = False

frame_counter = 0

last_results = {}

print()

print("=" * 60)

print("Coffee Station Interactive Tester V4")

print("=" * 60)

print()

print("SPACE : Pause / Resume")

print("C     : Start Inspection")

print("X     : Reset")

print("Q     : Quit")

print("=" * 60)

print()


# --------------------------------------------------
# COMMAND PARSER
# --------------------------------------------------

def parse_command(command: str):

    tokens = [

        token.strip().upper()

        for token in command.split(":")

        if token.strip()

    ]

    if len(tokens) < 2:

        print("[ERROR] Invalid command.")

        return None

    roi_names = []

    for token in tokens:

        if token.isdigit():

            continue

        if roi_manager.has_roi(token):

            roi_names.append(token)

        else:

            print(f"[WARNING] Unknown ROI: {token}")

    if not roi_names:

        print("[ERROR] No valid ROI selected.")

        return None

    return roi_names


# --------------------------------------------------
# START INSPECTION
# --------------------------------------------------

def start_inspection(roi_names):

    global last_results

    roi_manager.activate_rois(roi_names)

    last_results.clear()

    print()

    print("=" * 60)

    print("Inspection Started")

    print("=" * 60)

    for roi_name in roi_names:

        roi = roi_manager.get_roi(roi_name)

        roi["ui_state"] = "MONITORING"

        roi["inspection"]["cup_seen"] = False

        roi["inspection"]["clear_start"] = None

        roi["inspection"]["success_start"] = None

        roi["inspection"]["finished"] = False

        roi["inspection"]["cup_counter"] = 0
        roi["inspection"]["empty_counter"] = 0
        roi["inspection"]["confirm_start"] = None 

        print(f"Monitoring : {roi_name}")

    print()


# --------------------------------------------------
# FINISH SINGLE ROI
# --------------------------------------------------

def finish_roi(roi_name):

    roi = roi_manager.get_roi(roi_name)

    roi["enabled"] = False

    roi["ui_state"] = "IDLE"

    roi["inspection"]["finished"] = True

    print(f"[DONE] {roi_name} completed.")


# --------------------------------------------------
# RESET EVERYTHING
# --------------------------------------------------

def reset_all():

    global last_results

    roi_manager.reset_states()

    last_results.clear()

    print()

    print("=" * 60)

    print("Reset Complete")

    print("=" * 60)

    print()


# --------------------------------------------------
# CHECK WHETHER ALL ROI ARE FINISHED
# --------------------------------------------------

def all_finished():

    active = roi_manager.get_active_rois()

    if active:

        return False

    for roi in roi_manager.get_all_rois().values():

        if roi["inspection"]["finished"]:

            continue

        if roi["enabled"]:

            return False

    return True

# --------------------------------------------------
# UPDATE SINGLE ROI
# --------------------------------------------------

def update_single_roi(roi_name):

    roi = roi_manager.get_roi(roi_name)

    if roi is None:
        return

    if not roi["enabled"]:
        return

    inspection = roi["inspection"]
    status = roi["status"]

    # --------------------------------------------------
    # UPDATE CONSECUTIVE COUNTERS
    # --------------------------------------------------

    if status == "CUP":

        inspection["cup_counter"] += 1
        inspection["empty_counter"] = 0

    elif status == "EMPTY":

        inspection["empty_counter"] += 1
        inspection["cup_counter"] = 0

        # Cancel pending confirmation
        inspection["confirm_start"] = None

    else:

        # Unknown detection
        inspection["cup_counter"] = 0
        inspection["empty_counter"] = 0
        return

    print(
        f"{roi_name} | "
        f"Status={status} | "
        f"CUP={inspection['cup_counter']} | "
        f"EMPTY={inspection['empty_counter']} | "
        f"CUP_SEEN={inspection['cup_seen']}"
    )

    # --------------------------------------------------
    # WAITING FOR FIRST CONFIRMED CUP
    # --------------------------------------------------

    if not inspection["cup_seen"]:

        roi["ui_state"] = "MONITORING"

        # Not enough consecutive CUP detections yet
        if inspection["cup_counter"] < CONFIRM_CUP:

            inspection["confirm_start"] = None
            return

        # First time reaching CONFIRM_CUP
        if inspection["confirm_start"] is None:

            inspection["confirm_start"] = time.time()

            # <<< ADD HERE >>>
            print(f"[HOLD] {roi_name} Hold timer started")

            return

        # <<< ADD HERE >>>
        elapsed = time.time() - inspection["confirm_start"]

        print(
            f"[HOLD] {roi_name} "
            f"{elapsed:.2f}/{CONFIRM_HOLD_TIME:.2f}s"
        )

        # CUP disappeared before hold time completed
        if status != "CUP":

            inspection["confirm_start"] = None
            inspection["cup_counter"] = 0
            return

        # Hold timer still running
        if elapsed < CONFIRM_HOLD_TIME:
            return

        # Confirm cup
        print(f"[CONFIRMED] Cup detected : {roi_name}")

        inspection["cup_seen"] = True
        inspection["confirm_start"] = None
        inspection["clear_start"] = None
        inspection["success_start"] = None

        roi["ui_state"] = "FAILED"

        return

    # --------------------------------------------------
    # CUP STILL PRESENT
    # --------------------------------------------------

    if inspection["cup_counter"] >= CONFIRM_CUP:

        inspection["clear_start"] = None
        inspection["success_start"] = None

        roi["ui_state"] = "FAILED"

        return

    # --------------------------------------------------
    # WAITING FOR CUP REMOVAL
    # --------------------------------------------------

    roi["ui_state"] = "MONITORING"

    if inspection["empty_counter"] < CONFIRM_EMPTY:
        return

    # --------------------------------------------------
    # START CLEAR TIMER
    # --------------------------------------------------

    if inspection["clear_start"] is None:

        print(f"[INFO] Cup removed : {roi_name}")

        inspection["clear_start"] = time.time()

        return

    # --------------------------------------------------
    # WAITING 5 SECONDS
    # --------------------------------------------------

    elapsed = time.time() - inspection["clear_start"]

    if elapsed < CLEAR_DELAY:
        return

    # --------------------------------------------------
    # SUCCESS
    # --------------------------------------------------

    roi["ui_state"] = "SUCCESS"

    if inspection["success_start"] is None:

        inspection["success_start"] = time.time()

        return

    # --------------------------------------------------
    # FINISH ROI
    # --------------------------------------------------

    if time.time() - inspection["success_start"] >= SUCCESS_DISPLAY_TIME:

        finish_roi(roi_name)

# --------------------------------------------------
# UPDATE ALL ACTIVE ROI
# --------------------------------------------------

def update_all_rois():

    active = roi_manager.get_active_rois()

    if not active:
        return

    for roi_name in list(active.keys()):

        update_single_roi(roi_name)


# --------------------------------------------------
# PROCESS DETECTION
# --------------------------------------------------

def process_detection(frame):

    global last_results

    active = roi_manager.get_active_rois()

    if not active:
        return

    results = classifier.detect_multiple(

        frame,

        active

    )

    for result in results:

        roi_manager.update_result(result)

        last_results[result.roi_name] = result

# --------------------------------------------------
# DRAW ROI BOXES
# --------------------------------------------------

def draw_rois(display):

    for roi_name in roi_manager.get_all_roi_names():

        roi = roi_manager.get_roi(roi_name)

        color = COLOR.get(
            roi["ui_state"],
            (255, 255, 255)
        )

        cv2.rectangle(
            display,
            (roi["x1"], roi["y1"]),
            (roi["x2"], roi["y2"]),
            color,
            3
        )

        label = roi_name

        if roi["status"] is not None:

            label += f" : {roi['status']}"

        cv2.putText(
            display,
            label,
            (roi["x1"], roi["y1"] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )


# --------------------------------------------------
# DRAW RESULT PANEL
# --------------------------------------------------

def draw_results(display):

    y = 35

    for roi_name in sorted(last_results):

        result = last_results[roi_name]

        roi = roi_manager.get_roi(roi_name)

        text = (
            f"{roi_name:<3} | "
            f"{result.status:<5} | "
            f"{result.white_percentage:5.1f}% | "
            f"T:{roi['threshold']:.2f}"
        )

        cv2.putText(
            display,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        y += 28


# --------------------------------------------------
# DRAW ROI STATE PANEL
# --------------------------------------------------

def draw_roi_states(display):

    x = display.shape[1] - 300
    y = 35

    cv2.putText(
        display,
        "Inspection Status",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    y += 35

    for roi_name in roi_manager.get_all_roi_names():

        roi = roi_manager.get_roi(roi_name)

        state = roi["ui_state"]

        color = COLOR.get(
            state,
            (255, 255, 255)
        )

        text = f"{roi_name:<3} : {state}"

        cv2.putText(
            display,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            color,
            2
        )

        y += 28


# --------------------------------------------------
# DRAW PAUSE
# --------------------------------------------------

def draw_pause(display):

    if not paused:
        return

    cv2.putText(
        display,
        "PAUSED",
        (20, display.shape[0] - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )


# --------------------------------------------------
# DRAW COMPLETE UI
# --------------------------------------------------

def draw_ui(display):

    draw_rois(display)

    draw_results(display)

    draw_roi_states(display)

    draw_pause(display)

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

while True:

    # ----------------------------------------------
    # Read Frame
    # ----------------------------------------------

    if not paused:

        ret, frame = cap.read()

        if not ret:

            print("[INFO] End of video.")

            break

    display = frame.copy()

    frame_counter += 1

    # ----------------------------------------------
    # Detection
    # ----------------------------------------------

    active = roi_manager.get_active_rois()

    if active and frame_counter % DETECTION_INTERVAL == 0:

        process_detection(frame)

    # ----------------------------------------------
    # Update Independent ROI State Machines
    # ----------------------------------------------

    update_all_rois()

    # ----------------------------------------------
    # Draw UI
    # ----------------------------------------------

    draw_ui(display)

    display = cv2.resize(
        display,
        (DISPLAY_WIDTH, DISPLAY_HEIGHT)
    )

    cv2.imshow(
        "Coffee Station Interactive Tester V4",
        display
    )

    # ----------------------------------------------
    # Keyboard
    # ----------------------------------------------

    key = cv2.waitKey(1) 

    if key != -1:
     print("Key:", key)

    # Pause / Resume
    if key == ord(" "):

        paused = not paused

    # Start Inspection
    elif key == ord("c"):

        paused = True

        command = input(
            "Command (Example: 1:E2:S1): "
        )

        roi_names = parse_command(command)

        if roi_names is not None:

            start_inspection(roi_names)

        paused = False

    # Reset
    elif key == ord("x"):

        reset_all()

    # Forward 1 second
    elif key == ord("d"):

        current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cap.set(cv2.CAP_PROP_POS_FRAMES, current + 30)

    # Backward 1 second
    elif key == ord("a"):

        current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current - 30))

        current = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current - 30))

    # Quit
    elif key == ord("q"):

        break

# --------------------------------------------------
# CLEANUP
# --------------------------------------------------

cap.release()

cv2.destroyAllWindows()