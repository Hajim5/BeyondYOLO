
import json
import copy

class ROIManager:
    """
    ROI Manager V2

    Supports independent inspection state for every ROI.
    """

    def __init__(self, config_path):
        self.rois = {}
        self.load_config(config_path)

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            data = json.load(f)

        for item in data:
            self.rois[item["id"]] = {
                "id": item["id"],
                "class_name": item.get("class_name"),

                "x1": item["x1"],
                "y1": item["y1"],
                "x2": item["x2"],
                "y2": item["y2"],

                "threshold": item.get("threshold"),
                "empty_mean": item.get("empty_mean"),
                "empty_std": item.get("empty_std"),
                "empty_min": item.get("empty_min"),
                "empty_max": item.get("empty_max"),

                "cup_mean": item.get("cup_mean"),
                "cup_std": item.get("cup_std"),
                "cup_min": item.get("cup_min"),
                "cup_max": item.get("cup_max"),

                "confidence": item.get("confidence"),
                "sample_count": item.get("sample_count"),

                "enabled": False,
                "ui_state": "IDLE",
                "status": None,
                "result": None,

                "inspection": {
                    "cup_seen": False,
                    "clear_start": None,
                    "success_start": None,
                    "finished": False,
                }
            }

    # -----------------------------
    # Basic ROI
    # -----------------------------

    def has_roi(self, name):
        return name in self.rois

    def get_roi(self, name):
        return self.rois.get(name)

    def get_all_rois(self):
        return self.rois

    def get_all_roi_names(self):
        return list(self.rois.keys())

    def get_active_rois(self):
        return {
            k: v
            for k, v in self.rois.items()
            if v["enabled"]
        }

    # -----------------------------
    # Enable / Disable
    # -----------------------------

    def activate_rois(self, roi_names):
        self.reset_states()

        for name in roi_names:
            if name in self.rois:
                self.rois[name]["enabled"] = True
                self.start_roi(name)

    def start_roi(self, name):
        roi = self.rois[name]

        roi["enabled"] = True
        roi["ui_state"] = "MONITORING"
        roi["status"] = None
        roi["result"] = None

        roi["inspection"] = {
            "cup_seen": False,
            "clear_start": None,
            "success_start": None,
            "finished": False,
        }

    def finish_roi(self, name):
        roi = self.rois[name]

        roi["enabled"] = False
        roi["ui_state"] = "IDLE"

        roi["inspection"]["finished"] = True

    def is_finished(self, name):
        return self.rois[name]["inspection"]["finished"]

    # -----------------------------
    # Result
    # -----------------------------

    def update_result(self, result):
        roi = self.rois[result.roi_name]
        roi["result"] = result
        roi["status"] = result.status

    def get_result(self, name):
        return self.rois[name]["result"]

    def get_status(self, name):
        return self.rois[name]["status"]

    # -----------------------------
    # UI
    # -----------------------------

    def set_ui_state(self, name, state):
        self.rois[name]["ui_state"] = state

    def get_ui_state(self, name):
        return self.rois[name]["ui_state"]

    # -----------------------------
    # Inspection State
    # -----------------------------

    def get_inspection(self, name):
        return self.rois[name]["inspection"]

    def set_cup_seen(self, name, value=True):
        self.rois[name]["inspection"]["cup_seen"] = value

    def cup_seen(self, name):
        return self.rois[name]["inspection"]["cup_seen"]

    def set_clear_start(self, name, value):
        self.rois[name]["inspection"]["clear_start"] = value

    def get_clear_start(self, name):
        return self.rois[name]["inspection"]["clear_start"]

    def set_success_start(self, name, value):
        self.rois[name]["inspection"]["success_start"] = value

    def get_success_start(self, name):
        return self.rois[name]["inspection"]["success_start"]

    # -----------------------------
    # Reset
    # -----------------------------

    def reset_roi_state(self, name):
        roi = self.rois[name]

        roi["enabled"] = False
        roi["ui_state"] = "IDLE"
        roi["status"] = None
        roi["result"] = None

        roi["inspection"] = {
            "cup_seen": False,
            "clear_start": None,
            "success_start": None,
            "finished": False,
        }

    def reset_states(self):
        for name in self.rois:
            self.reset_roi_state(name)

    # -----------------------------
    # Helpers
    # -----------------------------

    def all_finished(self):
        active = self.get_active_rois()

        if not active:
            return False

        return all(
            roi["inspection"]["finished"]
            for roi in active.values()
        )

    def active_count(self):
        return len(self.get_active_rois())

    def __len__(self):
        return len(self.rois)

    def __contains__(self, name):
        return name in self.rois

    def __getitem__(self, name):
        return self.rois[name]

    def copy(self):
        return copy.deepcopy(self.rois)
