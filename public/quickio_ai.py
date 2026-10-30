#!/usr/bin/env python3
"""
Quickio AI — Multimodal Desktop Application (Windows .exe Ready)
Built with CustomTkinter, Quickio Cloud & Local Ollama GPU Tunnel (Gemma 3 / Gemma 4),
Live Open-Meteo Network Weather, Interactive Quickio Quiz, Flowchart Generator,
and OpenCV + YOLOv8 30 FPS Clamped Live Camera Bounding Box Tracker.
"""

import os
import sys
import time
import json
import uuid
import base64
import tempfile
import webbrowser
import threading
from typing import Optional, Dict, List

import requests
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageOps

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from google import genai
except ImportError:
    genai = None


# =====================================================================
# 1. CENTRALIZED BACKEND ROUTER: get_ai_response()
# =====================================================================

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_VISION_MODEL = "gemma3:4b"
DEFAULT_OLLAMA_TEXT_MODEL = "gemma3:4b"
DEFAULT_CLOUD_MODEL = "gemini-2.5-flash"


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def get_ai_response(
    prompt: str,
    image: Optional[str] = None,
    mode: str = "Network",
    system_instruction: Optional[str] = None,
    ollama_model_override: Optional[str] = None,
    api_key_override: Optional[str] = None,
) -> str:
    """
    Centralized Request Handler routing prompts and optional image paths between:
      - "Network": Quickio Cloud API
      - "Offline": Local GPU Tunnel via Ollama (http://localhost:11434 -> Gemma 4B / 8B / 12B)
    """
    full_prompt = f"{system_instruction}\n\nUser Input: {prompt}" if system_instruction else prompt

    if mode == "Network":
        if genai is None:
            return "[Dependency Missing] Run: pip install google-genai"

        api_key = api_key_override or os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return (
                "Quickio Cloud API key is not configured.\n"
                "Switch to 'Offline mode' in the top-right corner to use your local Gemma GPU, or open 'Setting ⚙' to add a key."
            )

        try:
            client = genai.Client(api_key=api_key)
            contents: List = []
            if image and os.path.exists(image):
                contents.append(Image.open(image))
            contents.append(full_prompt)

            fallback_models = [DEFAULT_CLOUD_MODEL, "gemini-2.5-flash-lite", "gemini-flash-latest"]
            last_err = None
            for model_name in fallback_models:
                try:
                    response = client.models.generateContent(model=model_name, contents=contents)
                    return response.text or "No response text returned."
                except Exception as model_exc:
                    last_err = model_exc
                    err_str = str(model_exc).lower()
                    if "503" in err_str or "unavailable" in err_str or "429" in err_str or "quota" in err_str:
                        continue
                    raise model_exc
            raise last_err
        except Exception as exc:
            err_lower = str(exc).lower()
            if "429" in err_lower or "quota" in err_lower or "resource_exhausted" in err_lower or "limit" in err_lower:
                return "Limit reached. Please switch to Offline mode in the top-right corner to continue using Quickio on your local GPU."
            return "Limit reached or cloud busy. Please switch to Offline mode in the top-right corner."

    elif mode == "Offline":
        selected_model = ollama_model_override or (
            DEFAULT_OLLAMA_VISION_MODEL if image else DEFAULT_OLLAMA_TEXT_MODEL
        )
        payload: Dict = {
            "model": selected_model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"num_gpu": 99, "temperature": 0.3},
        }
        if image and os.path.exists(image):
            payload["images"] = [encode_image_to_base64(image)]

        try:
            res = requests.post(DEFAULT_OLLAMA_URL, json=payload, timeout=90)
            res.raise_for_status()
            return res.json().get("response", "Empty response from local Ollama tunnel.")
        except Exception as exc:
            return f"[Offline Mode Error — http://localhost:11434]\n{str(exc)}"

    return f"Unknown mode: {mode}"


# =====================================================================
# 2. LIVE NETWORK WEATHER FETCHER (Open-Meteo Real-Time API)
# =====================================================================

def fetch_live_network_weather(city_name: str = "New Delhi") -> str:
    """Fetches real-time temperature, humidity, wind speed, and 5-day forecast."""
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={requests.utils.quote(city_name)}&count=1"
        geo_res = requests.get(geo_url, timeout=8).json()
        results = geo_res.get("results", [])
        if not results:
            return f"Could not find coordinates for '{city_name}'."

        loc = results[0]
        lat, lon = loc["latitude"], loc["longitude"]
        resolved_name = f"{loc.get('name')}, {loc.get('country', '')}"

        w_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,wind_speed_10m"
            "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max&timezone=auto"
        )
        w_data = requests.get(w_url, timeout=8).json()
        curr = w_data.get("current", {})
        daily = w_data.get("daily", {})

        report = [
            f"🌐 LIVE NETWORK WEATHER — {resolved_name}",
            f"• Current Temp: {curr.get('temperature_2m')}°C (Feels like {curr.get('apparent_temperature')}°C)",
            f"• Humidity: {curr.get('relative_humidity_2m')}% | Wind: {curr.get('wind_speed_10m')} km/h | Precip: {curr.get('precipitation')} mm",
            "\n📅 5-Day Agronomy Forecast:",
        ]
        dates = daily.get("time", [])[:5]
        max_t = daily.get("temperature_2m_max", [])[:5]
        min_t = daily.get("temperature_2m_min", [])[:5]
        rain_p = daily.get("precipitation_probability_max", [])[:5]
        for d, hi, lo, rp in zip(dates, max_t, min_t, rain_p):
            report.append(f"  {d}: High {hi}°C / Low {lo}°C · Rain Prob: {rp}%")
        return "\n".join(report)
    except Exception as exc:
        return f"[Live Network Weather Error] {str(exc)}"


# =====================================================================
# 3. WEBCAM HELPER & CLICK-AND-SEND PHOTO DIALOG
# =====================================================================

def open_camera_capture(camera_index: int = 0):
    """Safely opens laptop (0) or external USB webcam (1, 2) with multi-backend fallback."""
    if cv2 is None:
        return None
    if sys.platform.startswith("win"):
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            cap = cv2.VideoCapture(camera_index, backend)
            if cap is not None and cap.isOpened():
                return cap
            if cap is not None:
                cap.release()
    return cv2.VideoCapture(camera_index)


class WebcamPhotoCaptureDialog(ctk.CTkToplevel):
    """
    Live Webcam Photo Capture Dialog with BOTH:
      1. '📸 Click & Send Now' (Captures photo and immediately sends to AI)
      2. '📎 Click & Attach to Chat' (Attaches photo to chat box so you can add a question)
    """

    def __init__(self, parent, initial_camera_index: int = 0, on_photo_captured=None, on_photo_send_now=None):
        super().__init__(parent)
        self.parent = parent
        self.camera_index = initial_camera_index
        self.on_photo_captured = on_photo_captured
        self.on_photo_send_now = on_photo_send_now
        self.cap = None
        self.current_frame_bgr = None
        self._running = True

        self.title("Quickio AI — Click Photo from Webcam")
        self.geometry("760x580")
        self.configure(fg_color="#121216")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._close_dialog)

        top_bar = ctk.CTkFrame(self, fg_color="transparent")
        top_bar.pack(fill="x", padx=20, pady=(16, 8))

        ctk.CTkLabel(
            top_bar,
            text="📷 Live Webcam Capture",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#f4f4f5",
        ).pack(side="left")

        self.cam_selector = ctk.CTkOptionMenu(
            top_bar,
            values=[
                "Camera 0 (Laptop Webcam)",
                "Camera 1 (External USB Webcam)",
                "Camera 2 (Secondary External)",
            ],
            width=240,
            height=32,
            fg_color="#23252b",
            button_color="#7e22ce",
            button_hover_color="#a855f7",
            command=self._on_switch_camera,
        )
        self.cam_selector.pack(side="right")
        self.cam_selector.set(
            "Camera 0 (Laptop Webcam)"
            if self.camera_index == 0
            else f"Camera {self.camera_index} (External USB Webcam)"
        )

        self.preview_container = ctk.CTkFrame(
            self, width=720, height=420, fg_color="#050507", corner_radius=16, border_width=2, border_color="#3b1d5e"
        )
        self.preview_container.pack(padx=20, pady=8)
        self.preview_container.pack_propagate(False)

        self.video_label = ctk.CTkLabel(
            self.preview_container,
            text="Opening camera stream...",
            font=ctk.CTkFont(size=14),
            text_color="#9ca3af",
        )
        self.video_label.pack(fill="both", expand=True)

        bottom_bar = ctk.CTkFrame(self, fg_color="transparent")
        bottom_bar.pack(fill="x", padx=20, pady=(8, 16))

        ctk.CTkButton(
            bottom_bar,
            text="Cancel",
            width=85,
            height=38,
            fg_color="#23252b",
            hover_color="#32353e",
            corner_radius=10,
            command=self._close_dialog,
        ).pack(side="right", padx=(8, 0))

        ctk.CTkButton(
            bottom_bar,
            text="📸 Click & Send Now ➔",
            width=180,
            height=38,
            fg_color="#7e22ce",
            hover_color="#a855f7",
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=10,
            command=lambda: self._capture_snapshot(send_immediately=True),
        ).pack(side="right", padx=(8, 0))

        ctk.CTkButton(
            bottom_bar,
            text="📎 Click & Attach to Chat",
            width=175,
            height=38,
            fg_color="#3b0764",
            hover_color="#581c87",
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=10,
            command=lambda: self._capture_snapshot(send_immediately=False),
        ).pack(side="right")

        self._init_camera(self.camera_index)
        self._update_preview_loop()

    def _init_camera(self, idx: int):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
            time.sleep(0.15)  # Allow Windows driver to release hardware handle before switching

        if cv2 is None:
            self.video_label.configure(text="[Missing Dependency] Run: pip install opencv-python")
            return

        self.cap = open_camera_capture(idx)
        if self.cap is None or not self.cap.isOpened():
            self.video_label.configure(
                image=None,
                text=f"Camera {idx} is not connected or is busy.\nSwitch camera above or check Windows privacy settings.",
            )

    def _on_switch_camera(self, choice: str):
        idx = 1 if "Camera 1" in choice else (2 if "Camera 2" in choice else 0)
        self.camera_index = idx
        self.parent.camera_device_index = idx
        self._init_camera(idx)

    def _update_preview_loop(self):
        if not self._running:
            return
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.current_frame_bgr = frame.copy()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = ImageOps.contain(Image.fromarray(rgb), (708, 408))
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
                self.video_label.configure(image=ctk_img, text="")
                self.video_label.image = ctk_img
        self.after(30, self._update_preview_loop)

    def _capture_snapshot(self, send_immediately: bool = False):
        if self.current_frame_bgr is None or cv2 is None:
            return
        save_path = os.path.join(tempfile.gettempdir(), f"quickio_webcam_{int(time.time())}.jpg")
        cv2.imwrite(save_path, self.current_frame_bgr)
        if send_immediately and self.on_photo_send_now:
            self.on_photo_send_now(save_path)
        elif self.on_photo_captured:
            self.on_photo_captured(save_path)
        self._close_dialog()

    def _close_dialog(self):
        self._running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.destroy()


# =====================================================================
# 4. LIVE CAMERA BOUNDING BOX TRACKER (OpenCV + YOLOv8n)
# =====================================================================

class LiveBoundingBoxTracker:
    PALETTE_BGR = [
        (234, 51, 147),
        (16, 185, 129),
        (245, 158, 11),
        (239, 68, 68),
        (59, 130, 246),
        (168, 85, 247),
    ]

    def __init__(self, camera_index: int = 0, model_weights: str = "yolov8n.pt", conf_threshold: float = 0.40):
        self.camera_index = camera_index
        self.model_weights = model_weights
        self.conf_threshold = conf_threshold
        self._running = False

    def start(self, status_callback=None):
        if self._running:
            return
        if cv2 is None or YOLO is None:
            if status_callback:
                status_callback("Install opencv-python and ultralytics: pip install opencv-python ultralytics")
            return
        self._running = True
        threading.Thread(target=self._webcam_loop, args=(status_callback,), daemon=True).start()

    def _webcam_loop(self, status_callback=None):
        try:
            model = YOLO(self.model_weights)
            cap = open_camera_capture(self.camera_index)
            if cap is None or not cap.isOpened():
                self._running = False
                if status_callback:
                    status_callback(f"Could not open Camera {self.camera_index}.")
                return

            window_title = f"Quickio Live — 30 FPS Bounding Box Tracker [Cam {self.camera_index}] (Press 'Q' to Exit)"
            while self._running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                h_frame, w_frame = frame.shape[:2]
                results = model.predict(source=frame, conf=self.conf_threshold, max_det=15, verbose=False)
                for result in results:
                    if result.boxes is None:
                        continue
                    for box in result.boxes:
                        raw_x1, raw_y1, raw_x2, raw_y2 = map(int, box.xyxy[0].tolist())
                        # Strictly clamp coordinates inside video frame bounds so boxes NEVER go out of frame
                        x1 = max(2, min(w_frame - 20, raw_x1))
                        y1 = max(22, min(h_frame - 20, raw_y1))
                        x2 = max(x1 + 16, min(w_frame - 2, raw_x2))
                        y2 = max(y1 + 16, min(h_frame - 2, raw_y2))

                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = model.names.get(cls_id, f"Class {cls_id}")
                        color = self.PALETTE_BGR[cls_id % len(self.PALETTE_BGR)]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{cls_name} {conf * 100:.0f}%"
                        cv2.putText(frame, label, (x1, max(y1 - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2)

                cv2.imshow(window_title, frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
                    break
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                    break

            cap.release()
            cv2.destroyAllWindows()
            self._running = False
        except Exception as exc:
            self._running = False
            if status_callback:
                status_callback(f"[Tracker Error] {str(exc)}")


# =====================================================================
# 5. UPDATED CATEGORIZED HIERARCHY
# =====================================================================

CATEGORIES_HIERARCHY = [
    {
        "category": "✨ QUICKIO CORE",
        "items": [
            {
                "name": "Quickio Pulse",
                "vision": False,
                "is_camera_tracker": False,
                "desc": "What shall we build, solve, or explore right now?",
                "system_prompt": "You are Quickio Pulse, a versatile multimodal AI companion.",
            },
        ],
    },
    {
        "category": "👁️ VISION & ECOLOGY",
        "items": [
            {
                "name": "Waste Sorter",
                "vision": True,
                "is_camera_tracker": False,
                "desc": "Classify waste items into Recyclable, Compostable, E-Waste, or Landfill with disposal instructions.",
                "system_prompt": "You are Quickio AI Waste Sorter. Classify waste material and provide recycling instructions.",
            },
            {
                "name": "Plant Species Identifier",
                "vision": True,
                "is_camera_tracker": False,
                "desc": "Identify plant species, scientific name, botanical care guide, and leaf health from photos.",
                "system_prompt": "You are Quickio AI Plant Species Identifier. Identify the exact plant species, care requirements, and leaf health.",
            },
            {
                "name": "Fruit Freshness",
                "vision": True,
                "is_camera_tracker": False,
                "desc": "Evaluate ripeness index, surface bruising, mold indicators, and remaining shelf life.",
                "system_prompt": "You are Quickio AI Fruit Freshness Inspector. Evaluate ripeness stage and remaining shelf life.",
            },
            {
                "name": "Live Camera Bounding Box Tracker",
                "vision": True,
                "is_camera_tracker": True,
                "desc": "Launch real-time OpenCV webcam loop with YOLOv8 bounding boxes & class labels.",
                "system_prompt": "You are Quickio AI Object Tracker. Describe objects and spatial positions.",
            },
        ],
    },
    {
        "category": "🛡️ SAFETY & BIOMETRICS",
        "items": [
            {
                "name": "Posture Ergonomics",
                "vision": True,
                "is_camera_tracker": False,
                "desc": "Assess desk seating angles, cervical spine alignment, and wrist positioning.",
                "system_prompt": "You are Quickio AI Ergonomics Specialist. Evaluate seating posture and give corrective steps.",
            },
        ],
    },
    {
        "category": "📚 PRODUCTIVITY & TECH",
        "items": [
            {
                "name": "Flowchart Generator",
                "vision": False,
                "is_camera_tracker": False,
                "desc": "Convert plain text topics or processes into a structured visual ASCII/Step flowchart.",
                "system_prompt": (
                    "You are Quickio AI Flowchart Generator. Convert the user's plain text into a clean ASCII/Unicode "
                    "box-and-arrow flowchart diagram followed by a numbered breakdown of each decision and process node."
                ),
            },
            {
                "name": "Study Planner",
                "vision": False,
                "is_camera_tracker": False,
                "desc": "Default Instructions: Enter Subject, Exam Date, and Daily Hours to build a Pomodoro & Spaced Repetition plan.",
                "system_prompt": "You are Quickio AI Study Planner. Build a day-by-day Pomodoro and Spaced Repetition timetable.",
            },
            {
                "name": "Bilingual Quiz",
                "vision": False,
                "is_camera_tracker": False,
                "desc": "Interactive Quickio multiple-choice bilingual quiz with hints, grading, and explanations.",
                "system_prompt": "You are Quickio AI Bilingual Quiz Engine. Create a 5-question interactive bilingual quiz with hints and explanations.",
            },
            {
                "name": "Spam & Fact Audit",
                "vision": True,
                "is_camera_tracker": False,
                "desc": "Audit suspicious links/messages + Official Cyber Crime Reporting steps (1930 / cybercrime.gov.in / 7726).",
                "system_prompt": (
                    "You are Quickio AI Spam & Fact Auditor. Audit the message for phishing/scam indicators and provide "
                    "reporting instructions (National Cyber Crime Portal cybercrime.gov.in, Helpline 1930, SMS 7726)."
                ),
            },
            {
                "name": "Crop & Weather",
                "vision": True,
                "is_camera_tracker": False,
                "desc": "Live Network Weather (Open-Meteo) + Interactive AI Agronomy Advisor based on your farm details.",
                "system_prompt": "You are Quickio AI Crop & Weather Agronomist. Provide tailored crop and weather guidance.",
            },
        ],
    },
]


# =====================================================================
# 6. MAIN CUSTOMTKINTER WINDOW
# =====================================================================

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

BG_ROOT = "#050507"
BG_HEADER = "#141418"
BG_SIDEBAR = "#121216"
BG_WORKSPACE = "#08080c"
BG_INPUT_CARD = "#141418"
BG_PILL = "#23252b"
PURPLE_ACCENT = "#7e22ce"
PURPLE_GLOW = "#a855f7"
CYAN_DOT = "#0ea5e9"
TEXT_PRIMARY = "#f4f4f5"
TEXT_MUTED = "#9ca3af"


class CommercialSettingsDialog(ctk.CTkToplevel):
    """Commercial AI Settings Modal with Google Account ID, Delete Account, Model Preferences & Confidence Threshold Guide."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Quickio AI — Account & Settings")
        self.geometry("600x620")
        self.configure(fg_color=BG_SIDEBAR)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        ctk.CTkLabel(
            self,
            text="Quickio AI — Settings & Account",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=TEXT_PRIMARY,
        ).pack(anchor="w", padx=24, pady=(20, 10))

        # Account & Google Login Card
        acc_card = ctk.CTkFrame(self, fg_color="#191922", corner_radius=14, border_width=1, border_color="#3b1d5e")
        acc_card.pack(fill="x", padx=24, pady=(0, 14))

        ctk.CTkLabel(
            acc_card,
            text=f"Permanent User Account ID: {self.parent.user_account_id}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#d8b4fe",
        ).pack(anchor="w", padx=16, pady=(12, 4))

        btn_row = ctk.CTkFrame(acc_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(4, 12))

        ctk.CTkButton(
            btn_row,
            text="🔗 Sign in with Google (Cloud Sync)",
            fg_color=PURPLE_ACCENT,
            hover_color=PURPLE_GLOW,
            height=32,
            command=lambda: webbrowser.open("https://accounts.google.com"),
        ).pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            btn_row,
            text="🗑 Delete Account & All Data",
            fg_color="#7f1d1d",
            hover_color="#991b1b",
            height=32,
            command=self._delete_account,
        ).pack(side="left")

        # API Key
        ctk.CTkLabel(
            self,
            text="Quickio Cloud API Key:",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY,
        ).pack(anchor="w", padx=24, pady=(4, 4))

        self.api_key_entry = ctk.CTkEntry(
            self, width=550, height=34, placeholder_text="Enter Cloud API Key...", show="*", fg_color=BG_INPUT_CARD, border_color=PURPLE_ACCENT
        )
        self.api_key_entry.pack(padx=24, pady=(0, 12))
        self.api_key_entry.insert(0, self.parent.gemini_api_key)

        # Offline Model
        ctk.CTkLabel(
            self,
            text="Offline Mode Local GPU Model (http://localhost:11434):",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY,
        ).pack(anchor="w", padx=24, pady=(4, 4))

        self.ollama_model_menu = ctk.CTkOptionMenu(
            self,
            values=["gemma3:4b", "gemma3:12b", "gemma4:8b", "gemma4:12b", "llava", "moondream", "llama3.2"],
            width=550,
            height=34,
            fg_color=BG_PILL,
            button_color=PURPLE_ACCENT,
            button_hover_color=PURPLE_GLOW,
        )
        self.ollama_model_menu.pack(padx=24, pady=(0, 12))
        self.ollama_model_menu.set(self.parent.ollama_model)

        # Confidence Threshold + Plain-English Explanation
        ctk.CTkLabel(
            self,
            text=f"Vision Bounding Box Confidence Threshold ({int(self.parent.yolo_conf * 100)}%):",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY,
        ).pack(anchor="w", padx=24, pady=(4, 2))

        ctk.CTkLabel(
            self,
            text=(
                "What is Confidence Threshold? It sets the minimum certainty score (20%–85%) required before the AI "
                "draws a bounding box around an object. Higher values (70%+) prevent false detections; lower values (30%) "
                "detect more objects even in dim lighting."
            ),
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
            wraplength=550,
            justify="left",
        ).pack(anchor="w", padx=24, pady=(0, 6))

        self.conf_slider = ctk.CTkSlider(
            self, from_=0.20, to=0.85, number_of_steps=13, width=550, progress_color=PURPLE_ACCENT, button_color=PURPLE_GLOW
        )
        self.conf_slider.set(self.parent.yolo_conf)
        self.conf_slider.pack(padx=24, pady=(2, 16))

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", padx=24, pady=(10, 16))

        ctk.CTkButton(footer, text="Save Settings", fg_color=PURPLE_ACCENT, hover_color=PURPLE_GLOW, command=self._save).pack(
            side="right", padx=(8, 0)
        )
        ctk.CTkButton(footer, text="Close", fg_color=BG_PILL, command=self.destroy).pack(side="right")

    def _delete_account(self):
        if messagebox.askyesno("Delete Account", "Permanently delete your account ID and all saved chat history?"):
            self.parent.chat_sessions.clear()
            self.parent.user_account_id = f"usr_{uuid.uuid4().hex[:10]}"
            self.parent.chat_display.configure(state="normal")
            self.parent.chat_display.delete("1.0", "end")
            self.parent.chat_display.configure(state="disabled")
            self.destroy()

    def _save(self):
        self.parent.gemini_api_key = self.api_key_entry.get().strip()
        if self.parent.gemini_api_key:
            os.environ["GEMINI_API_KEY"] = self.parent.gemini_api_key
        self.parent.ollama_model = self.ollama_model_menu.get()
        self.parent.yolo_conf = float(self.conf_slider.get())
        self.parent.yolo_tracker.conf_threshold = self.parent.yolo_conf
        self.destroy()


class QuickioAIApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Quickio AI")
        self.geometry("1380x860")
        self.minsize(1100, 720)
        self.configure(fg_color=BG_ROOT)

        self.user_account_id = f"usr_{uuid.uuid4().hex[:10]}"
        self.mode_var = ctk.StringVar(value="Network")
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        self.ollama_model = "gemma3:4b"
        self.yolo_conf = 0.40
        self.camera_device_index = 0
        self.selected_image_path: Optional[str] = None
        self.active_feature = CATEGORIES_HIERARCHY[0]["items"][0]
        self.category_frames: Dict[str, ctk.CTkFrame] = {}
        self.category_States: Dict[str, bool] = {}
        self.feature_buttons: Dict[str, ctk.CTkButton] = {}
        self.chat_sessions: Dict[str, str] = {"Session 1 (Current)": ""}
        self.current_session_name = "Session 1 (Current)"
        self.yolo_tracker = LiveBoundingBoxTracker(camera_index=0, conf_threshold=self.yolo_conf)

        self._build_header_bar()
        self._build_main_body()
        self._select_feature(self.active_feature)

    def _build_header_bar(self):
        self.header_bar = ctk.CTkFrame(self, height=62, fg_color=BG_HEADER, corner_radius=0)
        self.header_bar.pack(side="top", fill="x")
        self.header_bar.pack_propagate(False)

        # Chat History Dropdown (Quickio-style)
        hist_frame = ctk.CTkFrame(self.header_bar, fg_color="transparent")
        hist_frame.pack(side="left", padx=20)

        ctk.CTkLabel(hist_frame, text="Chat History:", font=ctk.CTkFont(size=12, weight="bold"), text_color=TEXT_MUTED).pack(
            side="left", padx=(0, 8)
        )
        self.history_dropdown = ctk.CTkOptionMenu(
            hist_frame,
            values=list(self.chat_sessions.keys()),
            width=200,
            height=30,
            fg_color=BG_PILL,
            button_color=PURPLE_ACCENT,
            command=self._switch_chat_session,
        )
        self.history_dropdown.pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            hist_frame,
            text="+ New Chat",
            width=95,
            height=30,
            fg_color="#231d33",
            hover_color=PURPLE_ACCENT,
            command=self._new_chat_session,
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            hist_frame,
            text="↻ Quick Reload",
            width=110,
            height=30,
            fg_color="#1f2028",
            hover_color="#3b0764",
            command=self._quick_reload_chat,
        ).pack(side="left")

        mode_container = ctk.CTkFrame(self.header_bar, fg_color="transparent")
        mode_container.pack(side="right", padx=24, pady=12)

        self.net_pill = ctk.CTkFrame(mode_container, fg_color=BG_PILL, corner_radius=18, border_width=1, border_color=PURPLE_GLOW)
        self.net_pill.pack(side="left", padx=(0, 14))
        ctk.CTkRadioButton(
            self.net_pill, text="Network mode", variable=self.mode_var, value="Network", fg_color=CYAN_DOT
        ).pack(padx=16, pady=7)

        self.off_pill = ctk.CTkFrame(mode_container, fg_color=BG_PILL, corner_radius=18, border_width=1, border_color="#2e3038")
        self.off_pill.pack(side="left")
        ctk.CTkRadioButton(
            self.off_pill, text="Offline mode", variable=self.mode_var, value="Offline", fg_color=CYAN_DOT
        ).pack(padx=16, pady=7)

    def _new_chat_session(self):
        self.chat_sessions[self.current_session_name] = self.chat_display.get("1.0", "end")
        feat_name = self.active_feature["name"]
        new_name = f"{feat_name} #{len(self.chat_sessions) + 1}"
        self.chat_sessions[new_name] = ""
        self.current_session_name = new_name
        self.history_dropdown.configure(values=list(self.chat_sessions.keys()))
        self.history_dropdown.set(new_name)
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.configure(state="disabled")

    def _quick_reload_chat(self):
        self.chat_sessions[self.current_session_name] = ""
        self.selected_image_path = None
        self.attached_lbl.configure(text="")
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.configure(state="disabled")

    def _switch_chat_session(self, name: str):
        self.chat_sessions[self.current_session_name] = self.chat_display.get("1.0", "end")
        self.current_session_name = name
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.insert("end", self.chat_sessions.get(name, ""))
        self.chat_display.configure(state="disabled")

    def _build_main_body(self):
        self.body_frame = ctk.CTkFrame(self, fg_color=BG_ROOT, corner_radius=0)
        self.body_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        self.sidebar_card = ctk.CTkFrame(
            self.body_frame, width=285, fg_color=BG_SIDEBAR, corner_radius=24, border_width=2, border_color="#3b1d5e"
        )
        self.sidebar_card.pack(side="left", fill="y", padx=(4, 14), pady=4)
        self.sidebar_card.pack_propagate(False)

        ctk.CTkLabel(self.sidebar_card, text="Quickio AI", font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT_PRIMARY).pack(
            anchor="w", padx=22, pady=(20, 10)
        )
        ctk.CTkFrame(self.sidebar_card, height=2, fg_color=PURPLE_ACCENT, corner_radius=1).pack(fill="x", padx=16, pady=(0, 10))

        self.nav_scroll = ctk.CTkScrollableFrame(self.sidebar_card, fg_color="transparent")
        self.nav_scroll.pack(side="top", fill="both", expand=True, padx=8, pady=4)

        for cat_group in CATEGORIES_HIERARCHY:
            cat_title = cat_group["category"]
            self.category_States[cat_title] = True
            cat_header_btn = ctk.CTkButton(
                self.nav_scroll,
                text=f"▾  {cat_title}",
                anchor="w",
                fg_color="#1b1b22",
                hover_color="#262633",
                text_color="#d8b4fe",
                height=34,
                corner_radius=10,
                font=ctk.CTkFont(size=12, weight="bold"),
            )
            cat_header_btn.pack(fill="x", padx=4, pady=(8, 3))

            items_container = ctk.CTkFrame(self.nav_scroll, fg_color="transparent")
            items_container.pack(fill="x", padx=4, pady=(0, 4))
            self.category_frames[cat_title] = items_container
            cat_header_btn.configure(command=lambda c=cat_title, b=cat_header_btn: self._toggle_category_dropdown(c, b))

            for item in cat_group["items"]:
                row_frame = ctk.CTkFrame(items_container, fg_color="transparent")
                row_frame.pack(fill="x", pady=2)
                btn = ctk.CTkButton(
                    row_frame,
                    text=item["name"],
                    anchor="w",
                    fg_color="transparent",
                    hover_color="#231d33",
                    text_color=TEXT_PRIMARY,
                    height=32,
                    corner_radius=8,
                    font=ctk.CTkFont(size=12),
                    command=lambda feat=item: self._select_feature(feat),
                )
                btn.pack(side="left", fill="x", expand=True)
                self.feature_buttons[item["name"]] = btn
                if item["vision"]:
                    ctk.CTkLabel(
                        row_frame, text="Vision", font=ctk.CTkFont(size=10, weight="bold"), fg_color="#2e1065", text_color="#e9d5ff", corner_radius=6, padx=6, pady=1
                    ).pack(side="right", padx=(4, 2))

        setting_footer = ctk.CTkFrame(self.sidebar_card, fg_color="transparent")
        setting_footer.pack(side="bottom", fill="x", padx=16, pady=16)
        ctk.CTkButton(
            setting_footer, text="Setting", anchor="w", fg_color="transparent", hover_color="#1f1f29", font=ctk.CTkFont(size=20), command=lambda: CommercialSettingsDialog(self)
        ).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(
            setting_footer, text="⚙", width=40, height=40, fg_color="transparent", hover_color="#1f1f29", font=ctk.CTkFont(size=22), command=lambda: CommercialSettingsDialog(self)
        ).pack(side="right")

        # RIGHT WORKSPACE
        self.workspace_panel = ctk.CTkFrame(self.body_frame, fg_color=BG_WORKSPACE, corner_radius=20)
        self.workspace_panel.pack(side="right", fill="both", expand=True, padx=4, pady=4)

        self.feature_header = ctk.CTkFrame(self.workspace_panel, fg_color="transparent")
        self.feature_header.pack(side="top", fill="x", padx=28, pady=(20, 4))

        self.feature_title_lbl = ctk.CTkLabel(self.feature_header, text="", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT_PRIMARY)
        self.feature_title_lbl.pack(side="left")

        self.yolo_launch_btn = ctk.CTkButton(
            self.feature_header,
            text="🎥 Launch Live YOLO Webcam Tracker",
            fg_color=PURPLE_ACCENT,
            hover_color=PURPLE_GLOW,
            height=34,
            command=lambda: self.yolo_tracker.start(status_callback=lambda m: self._append_chat_message("YOLO Tracker", m)),
        )

        self.weather_btn = ctk.CTkButton(
            self.feature_header,
            text="🌦 Fetch Live Network Weather",
            fg_color="#0284c7",
            hover_color="#0369a1",
            height=34,
            command=self._load_live_weather,
        )

        self.feature_desc_lbl = ctk.CTkLabel(self.workspace_panel, text="", font=ctk.CTkFont(size=13), text_color=TEXT_MUTED)
        self.feature_desc_lbl.pack(anchor="w", padx=28, pady=(0, 8))

        self.chat_display = ctk.CTkTextbox(
            self.workspace_panel, fg_color="#0c0c10", text_color=TEXT_PRIMARY, border_width=1, border_color="#1f1f29", corner_radius=16, font=ctk.CTkFont(size=14), wrap="word"
        )
        self.chat_display.pack(fill="both", expand=True, padx=28, pady=(0, 14))
        self.chat_display.configure(state="disabled")

        # BOTTOM CHAT INPUT BOX WITH '+' MENU, '📷' CAMERA BUTTON, AND 'SEND PHOTO NOW'
        self.input_card = ctk.CTkFrame(
            self.workspace_panel, height=145, fg_color=BG_INPUT_CARD, corner_radius=28, border_width=2, border_color=PURPLE_ACCENT
        )
        self.input_card.pack(side="bottom", fill="x", padx=64, pady=(0, 20))
        self.input_card.pack_propagate(False)

        self.prompt_input = ctk.CTkTextbox(self.input_card, fg_color="transparent", text_color=TEXT_PRIMARY, font=ctk.CTkFont(size=14), height=68, wrap="word")
        self.prompt_input.pack(side="top", fill="both", expand=True, padx=20, pady=(10, 2))
        self.prompt_input.bind("<Control-Return>", lambda _e: self._send_prompt())

        bottom_input_bar = ctk.CTkFrame(self.input_card, fg_color="transparent")
        bottom_input_bar.pack(side="bottom", fill="x", padx=16, pady=(0, 10))

        # '+' Plus Button for Upload / Extra Input Tools
        ctk.CTkButton(
            bottom_input_bar,
            text="＋",
            width=36,
            height=32,
            corner_radius=16,
            fg_color="#231d33",
            hover_color=PURPLE_ACCENT,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self._choose_image,
        ).pack(side="left", padx=(0, 6))

        # '📷' Camera Button directly inside chat box
        ctk.CTkButton(
            bottom_input_bar,
            text="📷 Camera",
            width=96,
            height=32,
            corner_radius=16,
            fg_color="#231d33",
            hover_color=PURPLE_ACCENT,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._open_webcam_photo_dialog,
        ).pack(side="left", padx=(0, 8))

        self.attached_lbl = ctk.CTkLabel(bottom_input_bar, text="", font=ctk.CTkFont(size=11), text_color="#d8b4fe")
        self.attached_lbl.pack(side="left", padx=4)

        self.send_btn = ctk.CTkButton(
            bottom_input_bar,
            text="Send ➔",
            fg_color=PURPLE_ACCENT,
            hover_color=PURPLE_GLOW,
            font=ctk.CTkFont(size=13, weight="bold"),
            width=120,
            height=32,
            corner_radius=16,
            command=self._send_prompt,
        )
        self.send_btn.pack(side="right")

    def _toggle_category_dropdown(self, category_title: str, header_btn: ctk.CTkButton):
        currently_open = self.category_States.get(category_title, True)
        frame = self.category_frames[category_title]
        if currently_open:
            frame.pack_forget()
            header_btn.configure(text=f"▸  {category_title}")
            self.category_States[category_title] = False
        else:
            frame.pack(fill="x", padx=4, pady=(0, 4), after=header_btn)
            header_btn.configure(text=f"▾  {category_title}")
            self.category_States[category_title] = True

    def _select_feature(self, feature: Dict):
        # Save current feature's chat before switching so each feature tab has its own isolated chat
        if hasattr(self, "chat_display"):
            self.chat_sessions[self.current_session_name] = self.chat_display.get("1.0", "end").strip()

        self.active_feature = feature
        for name, btn in self.feature_buttons.items():
            btn.configure(fg_color="#3b0764" if name == feature["name"] else "transparent")

        self.feature_title_lbl.configure(text=feature["name"])
        self.feature_desc_lbl.configure(text=feature["desc"])

        # Switch to or create this feature's dedicated chat session
        target_session_key = f"{feature['name']} — Chat"
        if target_session_key not in self.chat_sessions:
            self.chat_sessions[target_session_key] = ""
        self.current_session_name = target_session_key
        self.history_dropdown.configure(values=list(self.chat_sessions.keys()))
        self.history_dropdown.set(target_session_key)

        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        saved_txt = self.chat_sessions.get(target_session_key, "")
        if saved_txt:
            self.chat_display.insert("end", saved_txt + "\n\n")
        self.chat_display.configure(state="disabled")

        if feature.get("is_camera_tracker"):
            self.yolo_launch_btn.pack(side="right", padx=4)
        else:
            self.yolo_launch_btn.pack_forget()

        if feature["name"] == "Crop & Weather":
            self.weather_btn.pack(side="right", padx=4)
            self._load_live_weather()
        else:
            self.weather_btn.pack_forget()

    def _load_live_weather(self):
        def worker():
            report = fetch_live_network_weather("New Delhi")
            self.after(0, lambda: self._append_chat_message("Live Network Weather Service", report))
        threading.Thread(target=worker, daemon=True).start()

    def _open_webcam_photo_dialog(self):
        WebcamPhotoCaptureDialog(
            self,
            initial_camera_index=self.camera_device_index,
            on_photo_captured=self._attach_image_path,
            on_photo_send_now=self._attach_and_send_immediately,
        )

    def _choose_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image for Quickio AI",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.webp *.bmp")],
        )
        if file_path:
            self._attach_image_path(file_path)

    def _attach_image_path(self, file_path: str):
        self.selected_image_path = file_path
        self.attached_lbl.configure(text=f"📎 {os.path.basename(file_path)} (Click Send ➔)")

    def _attach_and_send_immediately(self, file_path: str):
        self.selected_image_path = file_path
        self.attached_lbl.configure(text=f"📎 {os.path.basename(file_path)}")
        self._send_prompt()

    def _append_chat_message(self, sender: str, text: str):
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", f"{sender}:\n{text}\n" + ("─" * 68) + "\n\n")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def _send_prompt(self):
        user_text = self.prompt_input.get("1.0", "end").strip()
        if not user_text and not self.selected_image_path:
            return
        if not user_text and self.selected_image_path:
            user_text = f"Analyze this photo using {self.active_feature['name']}."

        self.prompt_input.delete("1.0", "end")
        mode = self.mode_var.get()
        feature_name = self.active_feature["name"]
        sys_prompt = self.active_feature["system_prompt"]
        img_path = self.selected_image_path

        header_tag = "You"
        if img_path:
            header_tag += f" · 📎 {os.path.basename(img_path)}"

        self._append_chat_message(header_tag, user_text)
        self.selected_image_path = None
        self.attached_lbl.configure(text="")

        # Start animated pulsing dots indicator on Send button while waiting for AI
        self._thinking_active = True
        self._thinking_step = 0
        self.send_btn.configure(state="disabled", text="Thinking •")
        self._animate_thinking_dots()

        def worker():
            reply = get_ai_response(
                prompt=user_text,
                image=img_path,
                mode=mode,
                system_instruction=sys_prompt,
                ollama_model_override=self.ollama_model,
                api_key_override=self.gemini_api_key,
            )
            self.after(0, lambda: self._on_ai_complete(reply, mode))

        threading.Thread(target=worker, daemon=True).start()

    def _animate_thinking_dots(self):
        if not getattr(self, "_thinking_active", False):
            return
        dots = ["Thinking •", "Thinking • •", "Thinking • • •"]
        self.send_btn.configure(text=dots[self._thinking_step % len(dots)])
        self._thinking_step += 1
        self.after(280, self._animate_thinking_dots)

    def _on_ai_complete(self, reply: str, mode: str):
        self._thinking_active = False
        endpoint_tag = "Quickio AI"
        # Soft Typewriter Reveal Effect in CustomTkinter Chat Display
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", f"{endpoint_tag}:\n")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

        words = reply.split(" ")
        self._typewriter_words(words, 0)

    def _typewriter_words(self, words: List[str], idx: int):
        if idx < len(words):
            chunk = words[idx] + (" " if idx < len(words) - 1 else "")
            self.chat_display.configure(state="normal")
            self.chat_display.insert("end", chunk)
            self.chat_display.see("end")
            self.chat_display.configure(state="disabled")
            self.after(18, lambda: self._typewriter_words(words, idx + 1))
        else:
            self.chat_display.configure(state="normal")
            self.chat_display.insert("end", "\n" + ("─" * 68) + "\n\n")
            self.chat_display.see("end")
            self.chat_display.configure(state="disabled")
            self.send_btn.configure(state="normal", text="Send ➔")


if __name__ == "__main__":
    app = QuickioAIApp()
    app.mainloop()
