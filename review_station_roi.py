"""
review_station_roi.py

UPGRADED: Flexible 4-Corner / Oriented Quadrilateral Annotator

CONTROLS
--------

MOUSE:
    Left Click inside ROI
        Select machine

    Left Drag inside ROI
        Move entire ROI (all 4 vertices)

    Left Drag Corner Handle (Red squares)
        Independently adjust that corner point (perspective/slant)

    Left Drag Rotation Handle (Cyan circle)
        Rotate ROI freely around its center

    Left Click x4 (in ADD MODE 'A')
        Place 4 consecutive corner points to form an arbitrary quad

KEYBOARD:
    A
        Add machine by clicking 4 corners sequentially
    R
        Rename selected machine
    D / Delete
        Delete selected machine
    S
        Save layout
    C
        Confirm layout
    ESC
        Cancel active mode / Deselect
    Q
        Quit
"""

from pathlib import Path
import json
import copy
import math

import cv2
import numpy as np


# ============================================================
# PROJECT & CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
STATIONS_DIR = Path(r"C:\Users\PC-1\Downloads\sensory\1_station_video")

WINDOW_NAME = "Station ROI Review (Flexible Quadrilateral)"

MAX_DISPLAY_WIDTH = 1600
MAX_DISPLAY_HEIGHT = 900

BOX_THICKNESS = 2
SELECTED_BOX_THICKNESS = 3
CORNER_HANDLE_SIZE = 6
ROT_HANDLE_OFFSET = 30

COLOR_NORMAL = (0, 255, 0)
COLOR_SELECTED = (0, 255, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_HANDLE = (0, 0, 255)
COLOR_ROT_HANDLE = (255, 255, 0)
COLOR_NEW = (255, 0, 255)
COLOR_INFO_BG = (20, 20, 20)


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def clamp(val, min_v, max_v):
    return max(min_v, min(max_v, val))


def ensure_quad_points(roi):
    """
    Normalizes both legacy [x1, y1, x2, y2] and 4-point [[x,y],...]
    into a list of 4 float coordinates: [[x0, y0], [x1, y1], [x2, y2], [x3, y3]].
    """
    if not roi:
        return None
    if len(roi) == 4 and not isinstance(roi[0], (list, tuple)):
        x1, y1, x2, y2 = roi
        return [
            [float(x1), float(y1)],
            [float(x2), float(y1)],
            [float(x2), float(y2)],
            [float(x1), float(y2)],
        ]
    elif len(roi) == 4 and isinstance(roi[0], (list, tuple)):
        return [[float(p[0]), float(p[1])] for p in roi]
    return None


def get_quad_center(points):
    pts = np.array(points, dtype=np.float32)
    return np.mean(pts, axis=0)


def get_rotation_handle_pos(points):
    """Calculates a projection handle positioned above the top edge center."""
    p0, p1 = np.array(points[0]), np.array(points[1])
    top_mid = (p0 + p1) / 2.0
    center = get_quad_center(points)
    direction = top_mid - center
    norm = np.linalg.norm(direction)
    if norm < 1e-4:
        return top_mid - np.array([0, ROT_HANDLE_OFFSET])
    return top_mid + (direction / norm) * ROT_HANDLE_OFFSET


def rotate_points_around(points, angle_rad, center):
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    cx, cy = center
    rotated = []
    for x, y in points:
        tx = x - cx
        ty = y - cy
        rx = tx * cos_a - ty * sin_a + cx
        ry = tx * sin_a + ty * cos_a + cy
        rotated.append([rx, ry])
    return rotated


# ============================================================
# DISK & SOURCE HELPERS
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def find_station_directory(station_name):
    direct = STATIONS_DIR / station_name
    if direct.exists():
        return direct
    if STATIONS_DIR.exists():
        target = station_name.strip().lower()
        for path in STATIONS_DIR.iterdir():
            if path.is_dir() and path.name.lower() == target:
                return path
    raise FileNotFoundError(f"Station directory not found: {direct}")


def find_video_from_layout(station_dir, layout):
    video_name = layout.get("preview_source", {}).get("video")
    if not video_name:
        raise RuntimeError("Layout missing 'preview_source.video'.")
    matches = list(station_dir.rglob(video_name))
    if matches:
        return matches[0]
    target = video_name.lower()
    for path in station_dir.rglob("*"):
        if path.is_file() and path.name.lower() == target:
            return path
    raise FileNotFoundError(f"Preview video not found: {video_name}")


def read_reference_frame(video_path, timestamp):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame at {timestamp}s from {video_path}")
    return frame


def machine_sort_key(machine_id):
    mid = str(machine_id).upper()
    prefix = "".join(c for c in mid if not c.isdigit())
    digits = "".join(c for c in mid if c.isdigit())
    return (prefix, int(digits) if digits else 9999, mid)


# ============================================================
# ROI EDITOR
# ============================================================

class ROIEditor:
    def __init__(self, frame, layout, layout_path):
        self.original_frame = frame.copy()
        self.frame_height, self.frame_width = frame.shape[:2]
        self.layout = layout
        self.layout_path = layout_path
        self.machines = copy.deepcopy(layout.get("machines", {}))

        # Standardize all existing ROIs to 4-point quadrilaterals
        for m_id, data in self.machines.items():
            pts = ensure_quad_points(data.get("roi"))
            if pts:
                data["roi"] = pts

        self.selected_id = None
        self.mouse_down = False
        self.operation = None  # "MOVE", "CORNER_DRAG", "ROTATE"
        self.active_corner_idx = None

        self.start_pos = (0, 0)
        self.original_roi_snapshot = None
        self.initial_angle = 0.0

        # Add Mode state (sequential 4 clicks)
        self.add_mode = False
        self.new_quad_points = []
        self.cursor_pos = (0, 0)

        # Scaling calculations
        scale_w = MAX_DISPLAY_WIDTH / self.frame_width
        scale_h = MAX_DISPLAY_HEIGHT / self.frame_height
        self.display_scale = min(1.0, scale_w, scale_h)
        self.display_width = int(round(self.frame_width * self.display_scale))
        self.display_height = int(round(self.frame_height * self.display_scale))
        self.modified = False

    def display_to_original(self, dx, dy):
        ox = clamp(int(round(dx / self.display_scale)), 0, self.frame_width - 1)
        oy = clamp(int(round(dy / self.display_scale)), 0, self.frame_height - 1)
        return ox, oy

    # --------------------------------------------------------
    # HIT TESTING
    # --------------------------------------------------------

    def find_machine_at(self, x, y):
        candidates = []
        point = (float(x), float(y))
        for mid, data in self.machines.items():
            pts = data.get("roi")
            if not pts or len(pts) != 4:
                continue
            cnt = np.array(pts, dtype=np.int32)
            if cv2.pointPolygonTest(cnt, point, False) >= 0:
                area = cv2.contourArea(cnt)
                candidates.append((area, mid))
        if not candidates:
            return None
        candidates.sort()  # prioritize smaller overlapping boxes
        return candidates[0][1]

    def hit_test_handles(self, machine_id, x, y):
        pts = self.machines[machine_id].get("roi")
        if not pts:
            return None, None
        tol = (CORNER_HANDLE_SIZE + 4) / self.display_scale

        # 1. Corner handles
        for idx, (px, py) in enumerate(pts):
            if math.hypot(x - px, y - py) <= tol:
                return "CORNER", idx

        # 2. Rotation handle
        rot_pos = get_rotation_handle_pos(pts)
        if math.hypot(x - rot_pos[0], y - rot_pos[1]) <= tol * 1.5:
            return "ROTATE", None

        return None, None

    # --------------------------------------------------------
    # MOUSE CALLBACK
    # --------------------------------------------------------

    def mouse_callback(self, event, dx, dy, flags, param):
        x, y = self.display_to_original(dx, dy)
        self.cursor_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_down = True
            self.start_pos = (x, y)

            # Manual 4-point placement
            if self.add_mode:
                self.new_quad_points.append([x, y])
                if len(self.new_quad_points) == 4:
                    self.finish_new_quad()
                return

            # Check handles on current selection first
            if self.selected_id:
                handle_type, idx = self.hit_test_handles(self.selected_id, x, y)
                if handle_type == "CORNER":
                    self.operation = "CORNER_DRAG"
                    self.active_corner_idx = idx
                    self.original_roi_snapshot = copy.deepcopy(self.machines[self.selected_id]["roi"])
                    return
                elif handle_type == "ROTATE":
                    self.operation = "ROTATE"
                    center = get_quad_center(self.machines[self.selected_id]["roi"])
                    self.initial_angle = math.atan2(y - center[1], x - center[0])
                    self.original_roi_snapshot = copy.deepcopy(self.machines[self.selected_id]["roi"])
                    return

            # Selection or Move hit test
            hit_id = self.find_machine_at(x, y)
            if hit_id:
                self.selected_id = hit_id
                self.original_roi_snapshot = copy.deepcopy(self.machines[hit_id]["roi"])
                self.operation = "MOVE"
            else:
                self.selected_id = None
                self.operation = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if not self.mouse_down or not self.selected_id:
                return

            roi_ref = self.original_roi_snapshot
            if not roi_ref:
                return

            if self.operation == "MOVE":
                dx_val = x - self.start_pos[0]
                dy_val = y - self.start_pos[1]
                new_pts = [[p[0] + dx_val, p[1] + dy_val] for p in roi_ref]
                self.machines[self.selected_id]["roi"] = new_pts
                self.machines[self.selected_id]["review_status"] = "MANUALLY_EDITED"
                self.modified = True

            elif self.operation == "CORNER_DRAG":
                new_pts = copy.deepcopy(roi_ref)
                new_pts[self.active_corner_idx] = [float(x), float(y)]
                self.machines[self.selected_id]["roi"] = new_pts
                self.machines[self.selected_id]["review_status"] = "MANUALLY_EDITED"
                self.modified = True

            elif self.operation == "ROTATE":
                center = get_quad_center(roi_ref)
                curr_angle = math.atan2(y - center[1], x - center[0])
                delta_angle = curr_angle - self.initial_angle
                rotated_pts = rotate_points_around(roi_ref, delta_angle, center)
                self.machines[self.selected_id]["roi"] = rotated_pts
                self.machines[self.selected_id]["review_status"] = "MANUALLY_EDITED"
                self.modified = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_down = False
            self.operation = None
            self.active_corner_idx = None
            self.original_roi_snapshot = None

    # --------------------------------------------------------
    # ACTIONS & ADD MODE
    # --------------------------------------------------------

    def finish_new_quad(self):
        pts = self.new_quad_points
        self.new_quad_points = []
        self.add_mode = False

        print("\n" + "=" * 60 + "\nADD MACHINE VIA 4 CORNERS\n" + "=" * 60)
        mid = input("Machine ID (e.g. G1, I1, S6): ").strip().upper()
        if not mid:
            print("Cancelled.")
            return

        self.machines[mid] = {
            "id": mid,
            "roi": pts,
            "source": "MANUAL",
            "review_status": "MANUALLY_ADDED",
            "raw_detection_count": 0,
            "used_detection_count": 0,
            "mean_confidence": 0.0,
        }
        self.selected_id = mid
        self.modified = True
        print(f"Added {mid} with custom 4-point bounds.")

    def rename_selected(self):
        if not self.selected_id:
            return
        old_id = self.selected_id
        new_id = input(f"\nRename '{old_id}' to: ").strip().upper()
        if not new_id or new_id == old_id or new_id in self.machines:
            return
        entry = self.machines.pop(old_id)
        entry["id"] = new_id
        entry["review_status"] = "MANUALLY_RENAMED"
        self.machines[new_id] = entry
        self.selected_id = new_id
        self.modified = True

    def delete_selected(self):
        if not self.selected_id:
            return
        mid = self.selected_id
        if input(f"Delete machine {mid}? [y/N]: ").strip().lower() == "y":
            del self.machines[mid]
            self.selected_id = None
            self.modified = True

    # --------------------------------------------------------
    # RENDERING
    # --------------------------------------------------------

    def render(self):
        img = self.original_frame.copy()

        for mid, data in sorted(self.machines.items(), key=lambda kv: machine_sort_key(kv[0])):
            pts = data.get("roi")
            if not pts or len(pts) != 4:
                continue

            np_pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            is_sel = (mid == self.selected_id)
            color = COLOR_SELECTED if is_sel else COLOR_NORMAL
            thickness = SELECTED_BOX_THICKNESS if is_sel else BOX_THICKNESS

            cv2.polylines(img, [np_pts], isClosed=True, color=color, thickness=thickness)

            # Draw labels
            label = f"{mid} [{data.get('review_status', 'PENDING')}]"
            lx, ly = int(pts[0][0]), int(pts[0][1])
            cv2.putText(img, label, (lx, max(20, ly - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 2, cv2.LINE_AA)

            # Draw handles if selected
            if is_sel:
                sz = CORNER_HANDLE_SIZE
                for c_idx, (px, py) in enumerate(pts):
                    ix, iy = int(round(px)), int(round(py))
                    cv2.rectangle(img, (ix - sz, iy - sz), (ix + sz, iy + sz), COLOR_HANDLE, -1)
                    cv2.putText(img, str(c_idx + 1), (ix + sz + 2, iy + sz), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT, 1)

                rot_pos = get_rotation_handle_pos(pts)
                rx, ry = int(round(rot_pos[0])), int(round(rot_pos[1]))
                top_mid = (np.array(pts[0]) + np.array(pts[1])) / 2.0
                cv2.line(img, (int(top_mid[0]), int(top_mid[1])), (rx, ry), COLOR_ROT_HANDLE, 1, cv2.LINE_AA)
                cv2.circle(img, (rx, ry), 6, COLOR_ROT_HANDLE, -1)

        # Draw Add Mode interactive lines
        if self.add_mode:
            for i, p in enumerate(self.new_quad_points):
                cv2.circle(img, (int(p[0]), int(p[1])), 4, COLOR_NEW, -1)
                if i > 0:
                    prev = self.new_quad_points[i - 1]
                    cv2.line(img, (int(prev[0]), int(prev[1])), (int(p[0]), int(p[1])), COLOR_NEW, 2)
            if self.new_quad_points:
                last_p = self.new_quad_points[-1]
                cv2.line(img, (int(last_p[0]), int(last_p[1])), self.cursor_pos, COLOR_NEW, 1, cv2.LINE_AA)

        # Status overlay banner
        cv2.rectangle(img, (0, 0), (self.frame_width, 40), COLOR_INFO_BG, -1)
        if self.add_mode:
            st = f"ADD MODE: Click 4 corner points ({len(self.new_quad_points)}/4 clicked) | ESC=Cancel"
        elif self.selected_id:
            st = f"[{self.selected_id}] Drag Inside=Move | Drag Red Corners=Perspective | Drag Cyan=Rotate | R=Rename | D=Del"
        else:
            st = "Click inside ROI to select | A=Add (4 corners) | S=Save | C=Confirm | Q=Quit"
        cv2.putText(img, st, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1, cv2.LINE_AA)

        if self.display_scale < 1.0:
            return cv2.resize(img, (self.display_width, self.display_height), interpolation=cv2.INTER_AREA)
        return img

    def save(self, confirmed=False):
        if confirmed:
            for d in self.machines.values():
                d["review_status"] = "CONFIRMED"
            self.layout["status"] = "CONFIRMED"
            self.layout["reviewed"] = True
        elif self.layout.get("status") != "CONFIRMED":
            self.layout["status"] = "PENDING_REVIEW"

        self.layout["machines"] = self.machines
        self.layout["machine_count"] = len(self.machines)
        save_json(self.layout, self.layout_path)
        self.modified = False
        print(f"\nLayout saved to: {self.layout_path} (Status: {self.layout.get('status')})")

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, self.display_width, self.display_height)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

        while True:
            cv2.imshow(WINDOW_NAME, self.render())
            key = cv2.waitKey(20) & 0xFF

            if key == 255:
                continue
            elif key in (ord("a"), ord("A")):
                self.add_mode = True
                self.new_quad_points = []
                self.selected_id = None
            elif key in (ord("r"), ord("R")):
                self.rename_selected()
            elif key in (ord("d"), ord("D"), 127):
                self.delete_selected()
            elif key in (ord("s"), ord("S")):
                self.save(confirmed=False)
            elif key in (ord("c"), ord("C")):
                if input("\nConfirm layout? [y/N]: ").strip().lower() == "y":
                    self.save(confirmed=True)
            elif key == 27:  # ESC
                self.add_mode = False
                self.new_quad_points = []
                self.selected_id = None
            elif key in (ord("q"), ord("Q")):
                if self.modified and input("\nSave changes before quitting? [Y/n]: ").strip().lower() != "n":
                    self.save(confirmed=False)
                break

        cv2.destroyAllWindows()


# ============================================================
# ENTRY
# ============================================================

def review_station(station_name):
    station_dir = find_station_directory(station_name)
    detection_dir = station_dir / "detection"
    layout_path = detection_dir / f"station_layout_{station_name}.json"

    if not layout_path.exists():
        candidates = list(detection_dir.glob("station_layout_*.json"))
        if len(candidates) == 1:
            layout_path = candidates[0]
        else:
            raise FileNotFoundError(f"Station layout not found at {layout_path}")

    layout = load_json(layout_path)
    video_path = find_video_from_layout(station_dir, layout)
    timestamp = layout.get("preview_source", {}).get("timestamp_seconds", 0.0)

    frame = read_reference_frame(video_path, timestamp)
    editor = ROIEditor(frame, layout, layout_path)
    editor.run()


def main():
    name = input("Station name: ").strip()
    if name:
        review_station(name)


if __name__ == "__main__":
    main()