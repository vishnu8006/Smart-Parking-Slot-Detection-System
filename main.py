#!/usr/bin/env python3
"""
Smart Parking Detection System — UI/UX Theme & Controls Refinement
Provides separate Dark/Light UI designs, automatic reactive inference, custom video play/pause loops,
split available/occupied columns, and bold typography.
"""
import streamlit as st
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import cv2
import tempfile
import os
import torch
import plotly.graph_objects as go


# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Parking System",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme Session State Initialization ─────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False
if "play_speed" not in st.session_state:
    st.session_state.play_speed = 1.0
if "page" not in st.session_state:
    st.session_state.page = "home"
if "detection_result" not in st.session_state:
    st.session_state.detection_result = None
if "conf_thr" not in st.session_state:
    st.session_state.conf_thr = 0.25
if "iou_thr" not in st.session_state:
    st.session_state.iou_thr = 0.45
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "best.pt"
if "video_frame_idx" not in st.session_state:
    st.session_state.video_frame_idx = 0
if "active_cache_key" not in st.session_state:
    st.session_state.active_cache_key = None

# CCTV session state variables
if "cctv_connected" not in st.session_state:
    st.session_state.cctv_connected = False
if "cctv_config" not in st.session_state:
    st.session_state.cctv_config = {}
if "cctv_warning" not in st.session_state:
    st.session_state.cctv_warning = None


# ── CSS Styles Injection (Bold White Typography in Dark Mode) ──────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Orbitron:wght@400;600;700;900&display=swap');

/* Hide Streamlit header branding but keep sidebar recovery arrow functional */
[data-testid="stHeader"] {
    background-color: transparent !important;
    background: transparent !important;
}
.stDeployButton, [data-testid="stDeployButton"], #MainMenu, footer {
    display: none !important;
}
.block-container { padding-top: 1.5rem !important; padding-bottom: 3rem !important; }

/* ── Default Dark Theme Styles ── */
html, body, .stApp {
    background-color: #060913 !important;
    color: #c4cfec;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background: #090e1a !important;
    border-right: 1px solid #142340;
    min-width: 250px !important;
    max-width: 250px !important;
}
[data-testid="stSidebar"] section { padding: 0 !important; }

/* Bold White Text for Sidebar links in Dark Mode */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    padding: 12px 16px !important;
    text-align: left !important;
    width: 100% !important;
    justify-content: flex-start !important;
    transition: all 0.3s ease !important;
    margin-bottom: 4px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(0, 240, 255, 0.08) !important;
    color: #00f0ff !important;
    text-shadow: 0 0 10px rgba(0, 240, 255, 0.4);
    padding-left: 20px !important;
}
.nav-active [data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(90deg, rgba(0, 240, 255, 0.15), transparent) !important;
    color: #00f0ff !important;
    border-left: 3px solid #00f0ff !important;
    border-radius: 0 8px 8px 0 !important;
    font-weight: 800 !important;
    padding-left: 18px !important;
    text-shadow: 0 0 8px rgba(0, 240, 255, 0.5);
}

/* Cards with bold White Typography inside Dark Mode */
.premium-card {
    background: rgba(13, 20, 38, 0.65);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(0, 240, 255, 0.12);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 240, 255, 0.03);
    color: #ffffff;
    transition: all 0.3s ease;
}
.premium-card:hover {
    border-color: rgba(0, 240, 255, 0.25);
    box-shadow: 0 8px 32px 0 rgba(0, 240, 255, 0.08);
}

.glowing-green-card {
    background: rgba(0, 230, 118, 0.04) !important;
    border: 2px solid #00e676 !important;
    box-shadow: 0 0 18px rgba(0, 230, 118, 0.25) !important;
}

/* Titles that adapt to Dark/Light theme */
.card-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: #ffffff !important;
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.sidebar-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    color: #ffffff !important;
    line-height: 1.1;
    letter-spacing: 0.5px;
}

/* Metric text visibility explicitly set to bold White in Dark Mode */
.premium-metric-title {
    font-size: 0.75rem;
    color: #ffffff !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
    margin-bottom: 6px;
}
.premium-metric-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.3rem;
    font-weight: 800;
    line-height: 1.1;
    color: #ffffff !important;
}
.premium-metric-icon {
    position: absolute;
    top: 20px;
    right: 20px;
    font-size: 1.5rem;
    color: #ffffff !important;
    opacity: 0.85;
}

/* Headers explicitly set to bold White in Dark Mode */
.premium-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(0, 240, 255, 0.12);
    padding-bottom: 12px;
    margin-bottom: 24px;
}
.premium-header-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.65rem;
    font-weight: 800;
    color: #ffffff !important;
}

/* Surveillance preview camera wrapper */
.surveillance-hud {
    position: relative;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(0, 240, 255, 0.15);
}
.surveillance-tag {
    position: absolute;
    top: 12px;
    left: 12px;
    background: rgba(6, 9, 19, 0.85);
    border: 1px solid rgba(0, 240, 255, 0.25);
    color: #00f0ff;
    padding: 4px 10px;
    border-radius: 4px;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.7rem;
    z-index: 10;
}
.surveillance-rec {
    position: absolute;
    top: 16px;
    right: 16px;
    background: #ff1744;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    z-index: 10;
    box-shadow: 0 0 10px #ff1744;
    animation: flash-rec 1.5s infinite;
}
@keyframes flash-rec {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.2; }
}

.premium-uploader-box {
    border: 2px dashed rgba(0, 240, 255, 0.25);
    background: rgba(9, 14, 26, 0.5);
    border-radius: 12px;
    padding: 40px 15px;
    text-align: center;
    transition: all 0.3s;
}

.stButton > button {
    background: linear-gradient(135deg, #00b0ff, #0088cc) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 10px 24px !important;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00c3ff, #009ce6) !important;
    box-shadow: 0 0 15px rgba(0, 176, 255, 0.4) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: rgba(13, 20, 38, 0.6) !important;
    border: 1px solid rgba(0, 240, 255, 0.2) !important;
    color: #798ea8 !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #00f0ff !important;
    color: #00f0ff !important;
}

/* Pill Badges */
.slot-badge-grid {
    display: grid;
    gap: 8px;
    max-height: 420px;
    overflow-y: auto;
    padding-right: 5px;
}
.slot-badge {
    text-align: center;
    padding: 6px 4px;
    font-size: 0.76rem;
    font-weight: 700;
    border-radius: 6px;
    font-family: 'Orbitron', sans-serif;
}
.badge-vacant {
    background: rgba(0, 230, 118, 0.05);
    border: 1px solid #00e676;
    color: #00e676;
}
.badge-occupied {
    background: rgba(255, 23, 68, 0.04);
    border: 1px solid rgba(255, 23, 68, 0.4);
    color: rgba(255, 23, 68, 0.85);
}

.play-btn-box button {
    background: linear-gradient(135deg, #00e676, #00b359) !important;
}
.play-btn-box button:hover {
    background: linear-gradient(135deg, #26ff8d, #00cc66) !important;
    box-shadow: 0 0 18px rgba(0,230,118,0.45) !important;
}
.pause-btn-box button {
    background: linear-gradient(135deg, #ffd600, #cca300) !important;
}
.pause-btn-box button:hover {
    background: linear-gradient(135deg, #ffea4d, #e6b800) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Separate Light UI Theme Styles Injection (Fidelity UI/UX separation) ─────────
if st.session_state.theme == "light":
    st.markdown("""
    <style>
    /* Clean minimal Light UI overrides */
    html, body, .stApp {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #cbd5e1 !important;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label {
        color: #334155 !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        color: #334155 !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(14, 165, 233, 0.08) !important;
        color: #0ea5e9 !important;
        text-shadow: none !important;
    }
    .nav-active [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, rgba(14, 165, 233, 0.1), transparent) !important;
        color: #0284c7 !important;
        border-left: 3px solid #0284c7 !important;
        text-shadow: none !important;
    }
    
    /* Light Cards styling */
    .premium-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
        color: #0f172a !important;
    }
    .premium-card:hover {
        border-color: rgba(14, 165, 233, 0.3) !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08) !important;
    }
    
    .glowing-green-card {
        background: rgba(0, 200, 83, 0.02) !important;
        border: 2px solid #00c853 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 200, 83, 0.08) !important;
    }
    
    /* Dynamic Title Overrides */
    .card-title {
        color: #0f172a !important;
    }
    .sidebar-title {
        color: #0f172a !important;
    }
    
    /* Dark contrast labels inside Light Mode */
    .premium-header-title {
        color: #0f172a !important;
    }
    .premium-metric-title {
        color: #475569 !important;
    }
    .premium-metric-value {
        color: #0f172a !important;
    }
    .premium-metric-icon {
        color: #475569 !important;
    }
    .premium-header {
        border-bottom: 1px solid #cbd5e1 !important;
    }
    
    /* Ensure visibility for text descriptions */
    div[data-testid="stMarkdownContainer"] p, 
    div[data-testid="stMarkdownContainer"] span, 
    div[data-testid="stMarkdownContainer"] strong {
        color: #0f172a !important;
    }
    .stSlider label, .stSelectbox label, .stRadio label, .stSlider div {
        color: #0f172a !important;
    }
    
    .stButton > button[kind="secondary"] {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #334155 !important;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #0284c7 !important;
        color: #0284c7 !important;
    }
    
    /* Input label / text visibility in light mode */
    input {
        color: #0f172a !important;
        background-color: #ffffff !important;
    }
    div[data-baseweb="input"] {
        background-color: #ffffff !important;
        border-color: #cbd5e1 !important;
    }
    </style>
    """, unsafe_allow_html=True)



# ── AI Model Loader (Invisibly defaults to root best.pt) ───────────────────────
@st.cache_resource(show_spinner=False)
def load_yolo_model(model_name="best.pt"):
    from ultralytics import YOLO
    root_path = Path(__file__).resolve().parent / model_name
    if root_path.exists():
        return YOLO(str(root_path))
    elif Path("yolov8m.pt").exists():
        return YOLO("yolov8m.pt")
    else:
        return YOLO("yolov8n.pt")


# ── Row Clustering Sequential Sorter (A1 next to A2, ... A10) ──────────────────
def cluster_and_sort_slots(slots_raw):
    """
    1D threshold-based grouping algorithm to cluster slot center-y values into rows.
    Sorts each row left-to-right and sequentially labels them: A1, A2... B1, B2...
    """
    if not slots_raw:
        return []
    
    # Sort initially by cy to process row layers top-down
    slots_raw.sort(key=lambda s: s["cy"])
    
    rows = []
    current_row = [slots_raw[0]]
    
    # Cluster y coordinates within 65 pixels
    for s in slots_raw[1:]:
        avg_cy = sum(item["cy"] for item in current_row) / len(current_row)
        if abs(s["cy"] - avg_cy) < 65:
            current_row.append(s)
        else:
            rows.append(current_row)
            current_row = [s]
    rows.append(current_row)
    
    # Sort completed rows top-to-bottom by average cy
    rows.sort(key=lambda r: sum(item["cy"] for item in r) / len(r))
    
    sorted_slots = []
    row_letters = ["A", "B", "C", "D", "E", "F", "G"]
    
    for row_idx, r in enumerate(rows):
        # Sort current row strictly left-to-right
        r.sort(key=lambda s: s["cx"])
        row_letter = row_letters[row_idx] if row_idx < len(row_letters) else chr(ord("A") + row_idx)
        
        for col_idx, s in enumerate(r):
            s["id"] = f"{row_letter}{col_idx + 1}"
            sorted_slots.append(s)
            
    return sorted_slots


# ── Annotation Utilities (Double concentric outline HUD lock) ─────────────────
def draw_bounding_boxes(img_pil, results):
    """
    Annotates image with concentric double-line target locks for vacant slots.
    Improves line thicknesses and sequential label drawing.
    """
    draw = ImageDraw.Draw(img_pil)
    COLOR_VACANT = (0, 230, 118)   # Glowing Neon Green
    COLOR_OCCUPIED = (255, 23, 68)  # Neon Red
    COLOR_TXT = (255, 255, 255)
    
    try:
        font_lbl = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font_lbl = ImageFont.load_default()

    # Collect raw detections
    slots_raw = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xy)
            slots_raw.append({
                "cls": cls, "conf": conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2
            })

    # Sort slots using our DB-clustering row sorter
    slots_sorted = cluster_and_sort_slots(slots_raw)

    annotated_slots = []
    for s in slots_sorted:
        x1, y1, x2, y2 = s["x1"], s["y1"], s["x2"], s["y2"]
        is_vacant = (s["cls"] == 0)
        color = COLOR_VACANT if is_vacant else COLOR_OCCUPIED
        slot_id = s["id"]

        # Double concentric border for vacant target slots
        if is_vacant:
            # 1st line: Outer rectangle (3px thickness offset)
            draw.rectangle([x1 - 2, y1 - 2, x2 + 2, y2 + 2], outline=color, width=2)
            # 2nd line: Inner rectangle (drawn right inside with 2px gap)
            draw.rectangle([x1 + 1, y1 + 1, x2 - 1, y2 - 1], outline=color, width=1)
        else:
            # Single solid target box for occupied slots
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw sequential label text pill on top-left of the box
        lbl_box = draw.textbbox((x1 + 4, y1 - 16), slot_id, font=font_lbl)
        draw.rectangle([lbl_box[0] - 3, lbl_box[1] - 2, lbl_box[2] + 3, lbl_box[3] + 2], fill=color)
        draw.text((x1 + 4, y1 - 16), slot_id, fill=COLOR_TXT, font=font_lbl)

        annotated_slots.append({
            "id": slot_id,
            "cls": s["cls"],
            "conf": s["conf"],
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
        
    return img_pil, annotated_slots


def perform_inference(img_pil, model_name="best.pt", conf=0.25, iou=0.45):
    orig_w, orig_h = img_pil.size
    if orig_w > 800:
        scale = 800 / orig_w
        img_pil = img_pil.resize((800, int(orig_h * scale)), Image.Resampling.BILINEAR)

    model = load_yolo_model(model_name)
    img_np = np.array(img_pil)
    
    device_val = 0 if torch.cuda.is_available() else "cpu"
    results = model.predict(source=img_np, conf=conf, iou=iou, device=device_val, verbose=False, save=False)
    annotated_img, slots = draw_bounding_boxes(img_pil.copy(), results)
    
    vacant = [s for s in slots if s["cls"] == 0]
    occupied = [s for s in slots if s["cls"] == 1]
    
    return annotated_img, slots, vacant, occupied


def process_video(video_path, model_name, conf, iou, frame_stride=30):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_results = []
    frame_count = 0
    model = load_yolo_model(model_name)
    device_val = 0 if torch.cuda.is_available() else "cpu"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_stride == 0:
            h, w = frame.shape[:2]
            if w > 800:
                scale = 800 / w
                frame = cv2.resize(frame, (800, int(h * scale)))
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            res = model.predict(source=np.array(pil_img), conf=conf, iou=iou, device=device_val, verbose=False)
            annotated_img, slots = draw_bounding_boxes(pil_img.copy(), res)
            
            vacant = [s for s in slots if s["cls"] == 0]
            occupied = [s for s in slots if s["cls"] == 1]
            
            frame_results.append({
                "frame": frame_count,
                "time_sec": round(frame_count / fps, 2),
                "image": annotated_img,
                "slots": slots,
                "empty": vacant,
                "occupied": occupied,
                "total": len(slots),
                "width": pil_img.width,
                "height": pil_img.height,
            })
        frame_count += 1
    cap.release()
    return frame_results, total_frames, fps


# ── Render dynamic 1:1 Pixel Coordinate Accurate SVG Map ──────────────────────
def render_pixel_svg_map(slots=None, img_w=None, img_h=None, theme="dark"):
    """
    Renders an extremely premium 1:1 pixel coordinate map.
    If theme is light, generates a clean white "blueprint sheet" layout interface.
    Includes solid dark/light badges behind slot labels (A1, B1) for high visibility.
    """
    parts = []  # Initialize parts list for SVG elements

    if not img_w or not img_h:
        img_w = 800
        img_h = 450
        
    if not slots:
        mock_slots = []
        for i in range(10):
            mock_slots.append({
                "id": f"A{i+1}", "cls": 0 if i not in [2, 5, 8] else 1,
                "x1": 45 + i * 70, "y1": 50,
                "x2": 95 + i * 70, "y2": 140
            })
        for i in range(10):
            mock_slots.append({
                "id": f"B{i+1}", "cls": 0 if i not in [1, 4, 7, 9] else 1,
                "x1": 45 + i * 70, "y1": 270,
                "x2": 95 + i * 70, "y2": 360
            })
        slots = mock_slots
        img_w = 800
        img_h = 450

    # Auto-crop the map to the parking area based on slots bounding box
    if slots:
        min_x = min(s.get("x1", 0) for s in slots)
        min_y = min(s.get("y1", 0) for s in slots)
        max_x = max(s.get("x2", 0) for s in slots)
        max_y = max(s.get("y2", 0) for s in slots)
        
        pad_x = (max_x - min_x) * 0.08
        pad_y = (max_y - min_y) * 0.08
        
        crop_x = max(0, min_x - pad_x)
        crop_y = max(0, min_y - pad_y)
        crop_w = min(img_w - crop_x, (max_x - min_x) + pad_x * 2)
        crop_h = min(img_h - crop_y, (max_y - min_y) + pad_y * 2)
        
        # Shift slot coordinates to the new cropped origin
        new_slots = []
        for s in slots:
            ns = dict(s)
            ns["x1"] -= crop_x
            ns["x2"] -= crop_x
            ns["y1"] -= crop_y
            ns["y2"] -= crop_y
            new_slots.append(ns)
        slots = new_slots
        img_w = crop_w
        img_h = crop_h

    # Calculate target SVG dimensions with relaxed constraints to avoid squishing
    if img_w:
        SVG_W = int(min(max(img_w, 800), 1200))
    else:
        SVG_W = 800
        
    if img_w and img_h:
        aspect_ratio = img_h / img_w
        SVG_H = int(SVG_W * aspect_ratio)
        SVG_H = int(max(200, min(SVG_H, 650)))
    else:
        SVG_H = 450

    sx = SVG_W / img_w
    sy = SVG_H / img_h

    # Layout colors based on theme
    if theme == "light":
        canvas_bg = "#ffffff"
        grid_col = "rgba(0, 0, 0, 0.04)"
        border_col = "#cbd5e1"
        road_bg = "#e2e8f0"
        lane_col = "#e2b714" # Yellow line
        road_txt_col = "rgba(15, 23, 42, 0.5)"
        
        vacant_outline = "#0284c7"  # Sky Blue for Light Mode vacant
        vacant_fill = "rgba(14, 165, 233, 0.05)"
        occupied_outline = "#94a3b8"  # Slate Grey for Light Mode occupied
        occupied_fill = "#f1f5f9"
        badge_bg = "#f8fafc"
    else:
        canvas_bg = "#080d1a"
        grid_col = "rgba(0, 240, 255, 0.02)"
        border_col = "rgba(0, 240, 255, 0.1)"
        road_bg = "#111827" # Very dark asphalt
        lane_col = "#f59e0b" # Yellow dashed line
        road_txt_col = "rgba(0, 240, 255, 0.35)"
        
        vacant_outline = "#00e676"  # Glowing Green for Dark Mode vacant
        vacant_fill = "rgba(0, 230, 118, 0.04)"
        occupied_outline = "rgba(255, 23, 68, 0.4)"  # Faded Red for Dark Mode occupied
        occupied_fill = "rgba(255, 23, 68, 0.02)"
        badge_bg = "#090e1a"

    parts = [
        f'<svg width="100%" height="100%" viewBox="0 0 {SVG_W} {SVG_H}" xmlns="http://www.w3.org/2000/svg">',
        '<defs>',
        '  <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">',
        '    <feGaussianBlur stdDeviation="3" result="blur" />',
        '    <feComposite in="SourceGraphic" in2="blur" operator="over" />',
        '  </filter>',
        '  <style>',
        '    @keyframes pulse {',
        '      0%, 100% { opacity: 0.85; stroke-width: 1.8; }',
        '      50% { opacity: 0.45; stroke-width: 1.0; }',
        '    }',
        '    .pulse-vacant { animation: pulse 2.5s infinite ease-in-out; }',
        '  </style>',
        '</defs>',
        
        # Grid Background
        f'<rect width="{SVG_W}" height="{SVG_H}" fill="{canvas_bg}" rx="12" stroke="{border_col}" stroke-width="1.5"/>',
    ]

    # Add background grid lines
    for gy in range(50, SVG_H, 50):
        parts.append(f'<line x1="0" y1="{gy}" x2="{SVG_W}" y2="{gy}" stroke="{grid_col}" stroke-width="1"/>')
    for gx in range(50, SVG_W, 50):
        parts.append(f'<line x1="{gx}" y1="0" x2="{gx}" y2="{SVG_H}" stroke="{grid_col}" stroke-width="1"/>')

    # Draw Driveways (Roads) dynamically in the spaces between rows of slots
    # First, let's group slots into rows based on Y coordinates
    if slots:
        slots_copied = [dict(s) for s in slots]
        slots_copied.sort(key=lambda s: s.get("y1", 0) + s.get("y2", 0))
        
        rows = []
        current_row = []
        for s in slots_copied:
            cy = (s.get("y1", 0) + s.get("y2", 0)) / 2
            if not current_row:
                current_row.append(s)
            else:
                avg_cy = sum((item.get("y1", 0) + item.get("y2", 0)) / 2 for item in current_row) / len(current_row)
                if abs(cy - avg_cy) < (img_h * 0.15):
                    current_row.append(s)
                else:
                    rows.append(current_row)
                    current_row = [s]
        if current_row:
            rows.append(current_row)

        # Draw driveway between rows
        for i in range(len(rows) - 1):
            row_i_bottom = max(s.get("y2", 0) for s in rows[i])
            row_next_top = min(s.get("y1", 0) for s in rows[i+1])
            
            # Draw the road in between the slots rows
            y_top_scaled = int(row_i_bottom * sy)
            y_bot_scaled = int(row_next_top * sy)
            
            # Provide some padding so the roadway doesn't touch/overlap the slots directly
            y_top_scaled += 4
            y_bot_scaled -= 4
            
            road_height = y_bot_scaled - y_top_scaled
            if road_height > 15:
                y_center_scaled = (y_top_scaled + y_bot_scaled) // 2
                parts.append(f'<rect x="10" y="{y_top_scaled}" width="{SVG_W-20}" height="{road_height}" fill="{road_bg}" rx="6" />')
                parts.append(f'<line x1="20" y1="{y_center_scaled}" x2="{SVG_W-20}" y2="{y_center_scaled}" stroke="{lane_col}" stroke-width="2" stroke-dasharray="10 8" />')
                parts.append(f'<text x="{SVG_W//2}" y="{y_center_scaled+3}" font-size="8.5" fill="{road_txt_col}" font-family="Orbitron" font-weight="800" text-anchor="middle" letter-spacing="1.5">DRIVEWAY / ROADWAY</text>')
        
        # If there is only one row, draw a roadway below it
        if len(rows) == 1:
            row_bottom = max(s.get("y2", 0) for s in rows[0])
            y_top_scaled = int(row_bottom * sy) + 10
            road_height = 50
            if y_top_scaled + road_height < SVG_H:
                y_center_scaled = y_top_scaled + (road_height // 2)
                parts.append(f'<rect x="10" y="{y_top_scaled}" width="{SVG_W-20}" height="{road_height}" fill="{road_bg}" rx="6" />')
                parts.append(f'<line x1="20" y1="{y_center_scaled}" x2="{SVG_W-20}" y2="{y_center_scaled}" stroke="{lane_col}" stroke-width="2" stroke-dasharray="10 8" />')
                parts.append(f'<text x="{SVG_W//2}" y="{y_center_scaled+3}" font-size="8.5" fill="{road_txt_col}" font-family="Orbitron" font-weight="800" text-anchor="middle" letter-spacing="1.5">DRIVEWAY / ROADWAY</text>')
    else:
        # Default mock roadway
        parts.append(f'<rect x="10" y="160" width="{SVG_W-20}" height="90" fill="{road_bg}" rx="6" />')
        parts.append(f'<line x1="20" y1="205" x2="{SVG_W-20}" y2="205" stroke="{lane_col}" stroke-width="2" stroke-dasharray="10 8" />')
        parts.append(f'<text x="{SVG_W//2}" y="{y_center_scaled+3}" font-size="8.5" fill="{road_txt_col}" font-family="Orbitron" font-weight="800" text-anchor="middle" letter-spacing="1.5">DRIVEWAY / ROADWAY</text>')

    for s in slots:
        bx = int(s["x1"] * sx)
        by = int(s["y1"] * sy)
        bw = int((s["x2"] - s["x1"]) * sx)
        bh = int((s["y2"] - s["y1"]) * sy)
        
        # Ensure slots are sufficiently large for clear display
        bw = max(bw, 24)
        bh = max(bh, 24)
        
        is_vacant = (s["cls"] == 0)
        
        if is_vacant:
            # Concentric double outline target for vacant spaces
            parts.append(
                f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="{vacant_fill}" stroke="{vacant_outline}" stroke-width="2" rx="3" class="pulse-vacant" />'
                f'<rect x="{bx+2}" y="{by+2}" width="{bw-4}" height="{bh-4}" fill="none" stroke="{vacant_outline}" stroke-width="1" rx="2" opacity="0.75" />'
            )
            parts.append(f'<circle cx="{bx+bw//2}" cy="{by+bh//2}" r="2" fill="{vacant_outline}" opacity="0.8"/>')
        else:
            # Solid bounding line for occupied spaces
            parts.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="{occupied_fill}" stroke="{occupied_outline}" stroke-width="1.2" rx="3" />')
            
            # Draw vector car outline
            car_w = int(bw * 0.7)
            car_h = int(bh * 0.8)
            cx = bx + (bw - car_w) // 2
            cy = by + (bh - car_h) // 2
            parts.append(
                f'<rect x="{cx}" y="{cy}" width="{car_w}" height="{car_h}" fill="rgba(100,116,139,0.05)" stroke="{occupied_outline}" stroke-width="1" rx="2" />'
                f'<line x1="{cx+2}" y1="{cy+3}" x2="{cx+car_w-2}" y2="{cy+3}" stroke="{occupied_outline}" stroke-width="1"/>'
            )

        # Centered slot ID text label with high contrast background badge
        tx = bx + bw // 2
        ty = by + bh // 2
        text_color = vacant_outline if is_vacant else occupied_outline
        
        # Responsive badge dimensions based on slot size
        badge_w = min(60, max(26, int(bw * 0.4)))
        badge_h = min(28, max(14, int(bh * 0.25)))
        font_sz = min(24, max(10, int(bw * 0.22)))
        parts.append(
            f'<rect x="{tx - badge_w//2}" y="{ty - badge_h//2}" width="{badge_w}" height="{badge_h}" fill="{badge_bg}" rx="3" stroke="{text_color}" stroke-width="0.8"/>'
            f'<text x="{tx}" y="{ty + font_sz//3}" text-anchor="middle" font-size="{font_sz}" font-family="Orbitron, monospace" font-weight="900" fill="{text_color}">{s["id"]}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


# ── Sidebar Navigation & Layout Swaps ──────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 24px 16px 14px; display:flex; align-items:center; gap:12px;">
        <div style="background: linear-gradient(135deg, #00f0ff, #0072ff);
                    border-radius:10px; width:38px; height:38px;
                    display:flex; align-items:center; justify-content:center;
                    font-size:1.25rem; flex-shrink:0; box-shadow: 0 0 12px rgba(0, 240, 255, 0.4);">🅿️</div>
        <div>
            <div class="sidebar-title" style="font-size: 1.1rem; line-height: 1.1;">SMART PARK</div>
        </div>
    </div>
    <div style="height:1px; background:rgba(255,255,255,0.12); margin: 0 16px 16px;"></div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.65rem;font-weight:700;letter-spacing:0.12em;color:#5b6e85;text-transform:uppercase;padding:0 20px 8px;">Navigation</div>', unsafe_allow_html=True)

    pages = [
        ("home",      "🏠 Home"),
        ("result",    "📊 Result"),
        ("analytics", "📈 Analytics"),
    ]
    for pid, label in pages:
        is_active = (st.session_state.page == pid)
        btn_type = "primary" if is_active else "secondary"
        if st.button(label, key=f"nav_{pid}", use_container_width=True, type=btn_type):
            st.session_state.page = pid
            st.rerun()

    st.markdown('<div style="height:1px;background:rgba(255,255,255,0.12);margin:18px 16px;"></div>', unsafe_allow_html=True)


# ── Global Diagnostic calculation ──────────────────────────────────────────────
res = st.session_state.detection_result
if res:
    if res["type"] in ["image", "cctv"]:
        total_slots = res["total"]
        vacant_slots = len(res["empty"])
        occupied_slots = len(res["occupied"])
        occupancy_rate = int(res["rate"] * 100) if total_slots else 0
        current_slots_list = res["slots"]
    else:
        frames = res["frame_results"]
        idx = st.session_state.video_frame_idx
        if idx >= len(frames): idx = 0
        cur_f = frames[idx]
        
        total_slots = cur_f["total"]
        vacant_slots = len(cur_f["empty"])
        occupied_slots = len(cur_f["occupied"])
        occupancy_rate = int((occupied_slots / total_slots) * 100) if total_slots else 0
        current_slots_list = cur_f["slots"]
else:
    total_slots = 250
    vacant_slots = 85
    occupied_slots = 165
    occupancy_rate = 66
    current_slots_list = []



# ── Dynamic Plotly Layout config based on Active Theme ────────────────────────
def apply_plotly_theme(fig):
    if st.session_state.theme == "light":
        fig.update_layout(
            paper_bgcolor='rgba(255,255,255,0)',
            plot_bgcolor='rgba(255,255,255,0)',
            font=dict(color='#0f172a', family='Plus Jakarta Sans'),
            xaxis=dict(gridcolor='#cbd5e1', linecolor='#cbd5e1', tickfont=dict(color='#475569')),
            yaxis=dict(gridcolor='#cbd5e1', linecolor='#cbd5e1', tickfont=dict(color='#475569')),
            legend=dict(font=dict(color='#0f172a'))
        )
    else:
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', family='Plus Jakarta Sans'),
            xaxis=dict(gridcolor='rgba(0, 240, 255, 0.04)', linecolor='rgba(0, 240, 255, 0.1)', tickfont=dict(color='#ffffff')),
            yaxis=dict(gridcolor='rgba(0, 240, 255, 0.04)', linecolor='rgba(0, 240, 255, 0.1)', tickfont=dict(color='#ffffff')),
            legend=dict(font=dict(color='#ffffff'))
        )


# ══════════════════════════════════════════════════════════════════════════════
# TOP HEADER BAR WITH DUAL THEME PICKER (RIGHT SIDE TOP)
# ══════════════════════════════════════════════════════════════════════════════
page_titles = {
    "home": "Smart Parking System",
    "result": "Diagnostic Results Overview",
    "analytics": "Analytical Overview",
}
page_title = page_titles.get(st.session_state.page, "Smart Parking System")

head_col1, head_col2 = st.columns([5, 1.2])
with head_col1:
    st.markdown(f'<div class="premium-header-title" style="font-size:1.65rem; font-weight:800; margin-bottom: 20px;">{page_title}</div>', unsafe_allow_html=True)
with head_col2:
    theme_opt = st.selectbox(
        "Theme Selection",
        ["Dark Mode", "Light Mode"],
        index=0 if st.session_state.theme == "dark" else 1,
        label_visibility="collapsed",
        key="theme_picker_select"
    )
    new_theme = "dark" if "Dark" in theme_opt else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: 🏠 HOME (UPLOAD PORTAL & AUTOMATIC DETECTIONS)
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: 🏠 HOME (UPLOAD PORTAL & AUTOMATIC DETECTIONS)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "home":
    col_uploader, col_processed = st.columns([1, 1.25], gap="large")

    with col_uploader:
        st.markdown('<div class="premium-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">File Upload & Feed Portal</div>', unsafe_allow_html=True)
        
        src_tab = st.radio("Input Source:", ["📷 Image", "🎬 Video", "🎥 CCTV"], horizontal=True, label_visibility="collapsed", key="home_source_radio")
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        
        uploaded_file = None
        if "Image" in src_tab:
            st.session_state.cctv_connected = False
            uploaded_file = st.file_uploader(
                "Upload frames",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                label_visibility="collapsed",
                key="home_img_picker"
            )
        elif "Video" in src_tab:
            st.session_state.cctv_connected = False
            uploaded_file = st.file_uploader(
                "Upload videos",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                label_visibility="collapsed",
                key="home_vid_picker"
            )
        elif "CCTV" in src_tab:
            st.markdown('<div class="card-title" style="font-size:0.75rem; margin-top:10px;">CCTV Stream Settings</div>', unsafe_allow_html=True)
            cctv_name = st.text_input("Camera Name", value="Front Lot Camera", key="cctv_name_input")
            cctv_ip = st.text_input("IP Address / RTSP URL / WebCam Index", value="0", help="For webcams put 0 or 1. Otherwise RTSP or HTTP URL.", key="cctv_ip_input")
            cctv_port = st.text_input("Port (Optional)", value="554", key="cctv_port_input")
            
            cc_col1, cc_col2 = st.columns(2)
            with cc_col1:
                cctv_user = st.text_input("Username (Optional)", value="", key="cctv_user_input")
            with cc_col2:
                cctv_pass = st.text_input("Password (Optional)", value="", type="password", key="cctv_pass_input")
                
            connect_label = "⚡ DISCONNECT CCTV" if st.session_state.cctv_connected else "🔌 CONNECT CCTV"
            connect_type = "secondary" if st.session_state.cctv_connected else "primary"
            
            if st.button(connect_label, use_container_width=True, type=connect_type, key="cctv_connect_toggle"):
                if st.session_state.cctv_connected:
                    st.session_state.cctv_connected = False
                    st.session_state.detection_result = None
                    st.session_state.active_cache_key = None
                    st.rerun()
                else:
                    url = cctv_ip.strip()
                    if url.isdigit():
                        stream_source = int(url)
                    else:
                        if cctv_user and cctv_pass:
                            clean_url = url.replace("rtsp://", "").replace("http://", "")
                            if cctv_port:
                                stream_source = f"rtsp://{cctv_user}:{cctv_pass}@{clean_url}:{cctv_port}"
                            else:
                                stream_source = f"rtsp://{cctv_user}:{cctv_pass}@{clean_url}"
                        else:
                            stream_source = url
                            
                    cap_test = cv2.VideoCapture(stream_source)
                    if cap_test.isOpened():
                        ret, test_frame = cap_test.read()
                        cap_test.release()
                        if ret:
                            st.session_state.cctv_connected = True
                            st.session_state.cctv_warning = None
                            st.session_state.cctv_config = {
                                "source": stream_source,
                                "name": cctv_name,
                                "port": cctv_port
                            }
                            # Initialize detection_result
                            st.session_state.detection_result = {
                                "type": "cctv",
                                "source_name": cctv_name,
                                "slots": [],
                                "empty": [],
                                "occupied": [],
                                "total": 0,
                                "rate": 0.0,
                                "width": 640,
                                "height": 480
                            }
                            st.session_state.active_cache_key = f"cctv_{time.time()}"
                            st.rerun()
                        else:
                            st.session_state.cctv_warning = "Connected but failed to retrieve frame. Please check credentials or camera status."
                    else:
                        st.session_state.cctv_warning = "Failed to open connection source. Please check IP/URL, port, or status."
            
            if st.session_state.cctv_warning:
                st.error(st.session_state.cctv_warning)
            
            if st.session_state.cctv_connected:
                st.success(f"🟢 Connected to CCTV: {st.session_state.cctv_config.get('name')}")
            
        # Reset button positioned cleanly directly below upload portal
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ RESET INFRASTRUCTURE DATA", use_container_width=True, type="secondary"):
            st.session_state.detection_result = None
            st.session_state.active_cache_key = None
            st.session_state.video_frame_idx = 0
            st.session_state.is_playing = False
            st.session_state.cctv_connected = False
            st.rerun()
            
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        
        if uploaded_file:
            st.markdown(f"""
            <div style="background: rgba(0,230,118,0.06); border: 1px solid #00e676; border-radius:8px; padding:10px; font-size:0.75rem; color:#00e676; margin-bottom:14px; text-align:center;">
                📄 {uploaded_file.name} ({uploaded_file.size//1024} KB) loaded!
            </div>
            """, unsafe_allow_html=True)
            
            if "Image" in src_tab:
                pil_raw = Image.open(uploaded_file).convert("RGB")
                st.image(pil_raw, use_container_width=True, caption="Original Input Preview")
            elif "Video" in src_tab:
                st.video(uploaded_file)
        elif "CCTV" not in src_tab:
            st.markdown("""
            <div class="premium-uploader-box">
                <div style="font-size:2.3rem; margin-bottom:10px; filter: drop-shadow(0 0 10px rgba(0,240,255,0.25));">🖼️</div>
                <div style="font-weight:600; font-size:0.85rem; margin-bottom:4px;">Drag and drop file here</div>
                <div style="font-size:0.72rem; color:#5b6e85;">JPG, PNG, MP4 up to 50MB</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

    # ── COLUMN 2: PROCESSED OUTPUT VIEW ──
    with col_processed:
        st.markdown('<div class="premium-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Processed Output Feed</div>', unsafe_allow_html=True)
        
        slider_placeholder = st.empty()
        
        # ── AUTOMATIC REACTIVE DETECTION RESOLVER ──
        if uploaded_file:
            cache_key = f"{uploaded_file.name}_{st.session_state.conf_thr}_{st.session_state.iou_thr}_{st.session_state.selected_model}"
            
            if st.session_state.active_cache_key != cache_key:
                with st.spinner("Analyzing feed automatically..."):
                    if "Image" in src_tab:
                        pil_raw = Image.open(uploaded_file).convert("RGB")
                        ann_img, slots, vacant, occupied = perform_inference(
                            pil_raw,
                            st.session_state.selected_model,
                            st.session_state.conf_thr,
                            st.session_state.iou_thr
                        )
                        total = len(slots)
                        rate = len(occupied) / total if total else 0.0
                        
                        st.session_state.detection_result = {
                            "type": "image",
                            "source_name": uploaded_file.name,
                            "out_img": ann_img,
                            "slots": slots,
                            "empty": vacant,
                            "occupied": occupied,
                            "total": total,
                            "rate": rate,
                            "width": pil_raw.width,
                            "height": pil_raw.height,
                        }
                    else:
                        with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix, delete=False) as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name
                        try:
                            # Stride 15 for smoother playback, default 30
                            f_res, tot_f, fps = process_video(
                                tmp_path,
                                st.session_state.selected_model,
                                st.session_state.conf_thr,
                                st.session_state.iou_thr,
                                frame_stride=15
                            )
                            if f_res:
                                last = f_res[-1]
                                total = last["total"]
                                rate = len(last["occupied"]) / total if total else 0.0
                                
                                st.session_state.detection_result = {
                                    "type": "video",
                                    "source_name": uploaded_file.name,
                                    "frame_results": f_res,
                                    "frame_stride": 15,
                                    "out_img": last["image"],
                                    "slots": last["slots"],
                                    "empty": last["empty"],
                                    "occupied": last["occupied"],
                                    "total": total,
                                    "rate": rate,
                                    "fps": fps,
                                    "total_frames": tot_f,
                                    "width": last["width"],
                                    "height": last["height"],
                                }
                                st.session_state.video_frame_idx = 0
                        finally:
                            try: os.unlink(tmp_path)
                            except: pass
                            
                    st.session_state.active_cache_key = cache_key
                    st.rerun()

        res = st.session_state.detection_result
        if res:
            if res["type"] == "image":
                st.image(res["out_img"], use_container_width=True)
                st.markdown(f"""
                <div style="display:flex; justify-content:center; gap:20px; font-size:0.8rem; margin-top:14px; background: rgba(9,14,26,0.5); padding:10px; border-radius:6px; border:1px solid rgba(0,240,255,0.06);">
                    <span>🟢 <strong style="color:#00e676;">VACANT BAYS:</strong> {len(res['empty'])}</span>
                    <span>🔴 <strong style="color:#ff1744;">OCCUPIED BAYS:</strong> {len(res['occupied'])}</span>
                </div>
                """, unsafe_allow_html=True)
            elif res["type"] == "video":
                frames = res["frame_results"]
                
                # Image/Stats placeholder for smooth in-place updating
                image_placeholder = st.empty()
                info_placeholder = st.empty()
                time_placeholder = st.empty()
                
                # Play/Pause and Speed adjust controls
                st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
                play_col, speed_col = st.columns([1.5, 2], gap="small")
                
                with play_col:
                    play_lbl = "⏸ Pause Video" if st.session_state.is_playing else "▶ Play Video"
                    btn_class = "pause-btn-box" if st.session_state.is_playing else "play-btn-box"
                    
                    st.markdown(f'<div class="{btn_class}">', unsafe_allow_html=True)
                    if st.button(play_lbl, use_container_width=True, key="video_toggle_play"):
                        st.session_state.is_playing = not st.session_state.is_playing
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with speed_col:
                    speed_mult = st.select_slider(
                        "Playback Speed",
                        options=[0.5, 1.0, 1.5, 2.0],
                        value=st.session_state.play_speed,
                        format_func=lambda x: f"{x}x",
                        key="video_speed_slider"
                    )
                    if speed_mult != st.session_state.play_speed:
                        st.session_state.play_speed = speed_mult
                        st.rerun()

                # Handle continuous playback loop in-place without page reloads
                if st.session_state.is_playing:
                    idx = st.session_state.video_frame_idx
                    while st.session_state.is_playing:
                        if idx >= len(frames):
                            idx = 0
                        cur_f = frames[idx]
                        st.session_state.video_frame_idx = idx
                        
                        image_placeholder.image(cur_f["image"], use_container_width=True)
                        info_placeholder.markdown(f"""
                        <div style="display:flex; justify-content:center; gap:20px; font-size:0.8rem; margin-top:14px; background: rgba(9,14,26,0.5); padding:10px; border-radius:6px; border:1px solid rgba(0,240,255,0.06);">
                            <span>🟢 <strong style="color:#00e676;">VACANT BAYS:</strong> {len(cur_f['empty'])}</span>
                            <span>🔴 <strong style="color:#ff1744;">OCCUPIED BAYS:</strong> {len(cur_f['occupied'])}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        time_placeholder.markdown(f"""
                        <div style="text-align:center; font-size:0.75rem; color:#798ea8; font-family:'Orbitron'; margin-top:8px;">
                            Frame index: {idx+1} / {len(frames)} &nbsp;·&nbsp; Playback time: {cur_f['time_sec']:.1f}s / {frames[-1]['time_sec']:.1f}s
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Real delay slow calculations (stride / fps) / speed
                        stride = res.get("frame_stride", 15)
                        fps = res.get("fps", 30) or 30
                        base_delay = stride / fps
                        time.sleep(base_delay / st.session_state.play_speed)
                        idx = (idx + 1) % len(frames)
                else:
                    # Render single static frame
                    idx = st.session_state.video_frame_idx
                    if idx >= len(frames): idx = 0
                    cur_f = frames[idx]
                    image_placeholder.image(cur_f["image"], use_container_width=True)
                    info_placeholder.markdown(f"""
                    <div style="display:flex; justify-content:center; gap:20px; font-size:0.8rem; margin-top:14px; background: rgba(9,14,26,0.5); padding:10px; border-radius:6px; border:1px solid rgba(0,240,255,0.06);">
                        <span>🟢 <strong style="color:#00e676;">VACANT BAYS:</strong> {len(cur_f['empty'])}</span>
                        <span>🔴 <strong style="color:#ff1744;">OCCUPIED BAYS:</strong> {len(cur_f['occupied'])}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    time_placeholder.markdown(f"""
                    <div style="text-align:center; font-size:0.75rem; color:#798ea8; font-family:'Orbitron'; margin-top:8px;">
                        Frame index: {idx+1} / {len(frames)} &nbsp;·&nbsp; Playback time: {cur_f['time_sec']:.1f}s / {frames[-1]['time_sec']:.1f}s
                    </div>
                    """, unsafe_allow_html=True)
            elif res["type"] == "cctv":
                # Real-time CCTV analysis loop
                image_placeholder = st.empty()
                info_placeholder = st.empty()
                
                if st.session_state.cctv_connected:
                    source = st.session_state.cctv_config.get("source")
                    cap = cv2.VideoCapture(source)
                    
                    # Live Stream Loop
                    while st.session_state.cctv_connected:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Lost connection to CCTV. Reconnecting...")
                            time.sleep(2)
                            cap = cv2.VideoCapture(source)
                            continue
                            
                        # Resize for inference
                        h, w = frame.shape[:2]
                        if w > 800:
                            scale = 800 / w
                            frame = cv2.resize(frame, (800, int(h * scale)))
                        
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(rgb)
                        
                        # Inference
                        ann_img, slots, vacant, occupied = perform_inference(
                            pil_img,
                            st.session_state.selected_model,
                            st.session_state.conf_thr,
                            st.session_state.iou_thr
                        )
                        
                        # Update global results dict
                        total = len(slots)
                        rate = len(occupied) / total if total else 0.0
                        st.session_state.detection_result = {
                            "type": "cctv",
                            "source_name": st.session_state.cctv_config.get("name"),
                            "out_img": ann_img,
                            "slots": slots,
                            "empty": vacant,
                            "occupied": occupied,
                            "total": total,
                            "rate": rate,
                            "width": pil_img.width,
                            "height": pil_img.height
                        }
                        
                        image_placeholder.image(ann_img, use_container_width=True)
                        info_placeholder.markdown(f"""
                        <div style="display:flex; justify-content:center; gap:20px; font-size:0.8rem; margin-top:14px; background: rgba(9,14,26,0.5); padding:10px; border-radius:6px; border:1px solid rgba(0,240,255,0.06);">
                            <span>🟢 <strong style="color:#00e676;">VACANT BAYS:</strong> {len(vacant)}</span>
                            <span>🔴 <strong style="color:#ff1744;">OCCUPIED BAYS:</strong> {len(occupied)}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        time.sleep(0.05)
                    cap.release()
            
            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
            if st.button("📊 DISPLAY SYSTEM DIAGNOSTICS & MAPS →", use_container_width=True, type="primary", key="home_redirect_results_btn"):
                st.session_state.page = "result"
                st.rerun()
        else:
            st.markdown("""
            <div style="border: 1px solid rgba(0, 240, 255, 0.12); border-radius:12px; background:rgba(9,14,26,0.3); height:300px; display:flex; flex-direction:column; align-items:center; justify-content:center; color:#5b6e85;">
                <div style="font-size:3rem; margin-bottom:12px; text-shadow:0 0 10px rgba(0,240,255,0.25);">📡</div>
                <div style="font-family:'Orbitron'; font-weight:600; font-size:0.9rem;">NO INFERENCE FEED DETECTED</div>
                <div style="font-size:0.75rem; color:#5b6e85; margin-top:4px;">Upload an image or video file, or connect CCTV to trigger predictions.</div>
            </div>
            """, unsafe_allow_html=True)

        with slider_placeholder.container():
            st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
            sc1, sc2 = st.columns(2)
            with sc1:
                conf_val = st.slider("Prediction Confidence", 0.05, 0.95, st.session_state.conf_thr, 0.05, key="live_conf_slider")
                if conf_val != st.session_state.conf_thr:
                    st.session_state.conf_thr = conf_val
                    st.session_state.active_cache_key = None
                    st.rerun()
            with sc2:
                iou_val = st.slider("Overlap cutoff (IoU)", 0.05, 0.95, st.session_state.iou_thr, 0.05, key="live_iou_slider")
                if iou_val != st.session_state.iou_thr:
                    st.session_state.iou_thr = iou_val
                    st.session_state.active_cache_key = None
                    st.rerun()
                    
            # ── FRAME ADJUSTING OPTION (ONLY FOR VIDEO) ──
            if res and res["type"] == "video":
                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                frames = res["frame_results"]
                idx_val = st.slider("Frame Adjust", 0, len(frames) - 1, st.session_state.video_frame_idx, step=1, key="live_frame_adjust_slider")
                if idx_val != st.session_state.video_frame_idx:
                    st.session_state.video_frame_idx = idx_val
                    st.session_state.is_playing = False # Pause when manually scrubbing
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: 📊 RESULT (METRICS, 1:1 MAP TWIN, BADGES GRID)
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "result":
    # ── METRIC ROW ──
    met_col1, met_col2, met_col3 = st.columns(3, gap="medium")
    
    with met_col1:
        st.markdown(f"""
        <div class="premium-card" style="position:relative;">
            <div class="premium-metric-title">Monitored Capacity</div>
            <div class="premium-metric-value">{total_slots}</div>
            <span class="premium-metric-icon">🅿️</span>
            <div style="font-size:0.7rem; color:#798ea8; margin-top:6px;">Infrastructure capacity</div>
        </div>
        """, unsafe_allow_html=True)
        
    with met_col2:
        st.markdown(f"""
        <div class="premium-card glowing-green-card" style="position:relative;">
            <div class="premium-metric-title" style="color:#00e676 !important;">Available Slots</div>
            <div class="premium-metric-value" style="color: #00e676 !important;">{vacant_slots}</div>
            <span class="premium-metric-icon" style="color:#00e676 !important;">🟢</span>
            <div style="font-size:0.7rem; color:#00e676; opacity:0.8; margin-top:6px;">Dynamic vacancy ready</div>
        </div>
        """, unsafe_allow_html=True)
        
    with met_col3:
        st.markdown(f"""
        <div class="premium-card" style="position:relative;">
            <div class="premium-metric-title">Occupancy loading</div>
            <div class="premium-metric-value">{occupancy_rate}%</div>
            <span class="premium-metric-icon">🚗</span>
            <div style="font-size:0.7rem; color:#798ea8; margin-top:6px;">Current parking density</div>
        </div>
        """, unsafe_allow_html=True)

    # ── MAP TWIN & BADGES GRID ──
    col_map_twin, col_badges_grid = st.columns([1.6, 1], gap="large")
    
    with col_map_twin:
        st.markdown('<div class="premium-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Surveillance Grid Twin</div>', unsafe_allow_html=True)
        
        # Render map using accurate pixel positions scaled to width/height
        if res:
            svg_code = render_pixel_svg_map(current_slots_list, res.get("width"), res.get("height"), st.session_state.theme)
        else:
            svg_code = render_pixel_svg_map(theme=st.session_state.theme)
            
        st.markdown(svg_code, unsafe_allow_html=True)
        
        # Video controls if the result is video so they can play/pause right next to the map
        if res and res["type"] == "video":
            st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
            v_play, v_speed = st.columns([1, 1.5])
            with v_play:
                lbl = "⏸ Pause Video" if st.session_state.is_playing else "▶ Play Video"
                b_cls = "pause-btn-box" if st.session_state.is_playing else "play-btn-box"
                st.markdown(f'<div class="{b_cls}">', unsafe_allow_html=True)
                if st.button(lbl, key="res_play_toggle", use_container_width=True):
                    st.session_state.is_playing = not st.session_state.is_playing
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            with v_speed:
                speed = st.select_slider(
                    "Speed",
                    options=[0.5, 1.0, 1.5, 2.0],
                    value=st.session_state.play_speed,
                    format_func=lambda x: f"{x}x",
                    key="res_speed_slider"
                )
                if speed != st.session_state.play_speed:
                    st.session_state.play_speed = speed
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_badges_grid:
        st.markdown('<div class="premium-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Active Space allocations</div>', unsafe_allow_html=True)
        
        diag_slots = []
        if res:
            diag_slots = current_slots_list
        else:
            for i in range(20):
                is_vac = (i not in [2, 5, 8, 9, 11, 14, 17])
                diag_slots.append({
                    "id": f"A{i+1}" if i < 10 else f"B{i-9}",
                    "cls": 0 if is_vac else 1
                })
                
        # Split Space Badges layout in half: Free on Left, Occupied on Right
        col_badge_free, col_badge_occ = st.columns(2)
        
        with col_badge_free:
            st.markdown('<div style="font-family:\'Orbitron\',sans-serif;font-size:0.75rem;font-weight:700;color:#00e676;margin-bottom:10px;text-transform:uppercase;letter-spacing:0.5px;">Free Slots</div>', unsafe_allow_html=True)
            st.markdown('<div class="slot-badge-grid" style="grid-template-columns: 1fr 1fr;">', unsafe_allow_html=True)
            for s in diag_slots:
                if s["cls"] == 0:
                    st.markdown(f'<div class="slot-badge badge-vacant">🟢 {s["id"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_badge_occ:
            st.markdown('<div style="font-family:\'Orbitron\',sans-serif;font-size:0.75rem;font-weight:700;color:#ff1744;margin-bottom:10px;text-transform:uppercase;letter-spacing:0.5px;">Occupied</div>', unsafe_allow_html=True)
            st.markdown('<div class="slot-badge-grid" style="grid-template-columns: 1fr 1fr;">', unsafe_allow_html=True)
            for s in diag_slots:
                if s["cls"] == 1:
                    st.markdown(f'<div class="slot-badge badge-occupied">🔴 {s["id"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: 📈 ANALYTICS (PLOTLY THEMED GRAPHS)
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: 📈 ANALYTICS (PLOTLY THEMED GRAPHS)
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "analytics":
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Parking Occupancy Trend Timeline</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.8rem; color:#798ea8; margin-top:-8px;">This timeline tracks the overall parking space occupancy percentage over time to analyze peak usage hours.</p>', unsafe_allow_html=True)
    
    if res and res["type"] == "video" and len(res["frame_results"]) > 1:
        fr = res["frame_results"]
        times = [f["time_sec"] for f in fr]
        rates = [(f["total"] - len(f["empty"])) / f["total"] * 100 if f["total"] else 0.0 for f in fr]
        x_title = "Video Duration Timeline (seconds)"
    else:
        times = np.linspace(0, 24, 25)
        rates = [22, 18, 15, 12, 19, 35, 62, 75, 78, 68, 62, 65, 71, 74, 66, 60, 58, 67, 82, 88, 76, 54, 42, 30, 24]
        x_title = "Time of Day (24-Hour System clock)"
        
    fig_wave = go.Figure()
    
    line_col = '#0284c7' if st.session_state.theme == "light" else '#00f0ff'
    fill_col = 'rgba(14, 165, 233, 0.08)' if st.session_state.theme == "light" else 'rgba(0, 240, 255, 0.08)'
    
    fig_wave.add_trace(go.Scatter(
        x=times, y=rates,
        mode='lines+markers' if len(times) < 30 else 'lines',
        line=dict(width=3, color=line_col, shape='spline'),
        fill='tozeroy',
        fillcolor=fill_col,
        name="Occupancy Rate %",
        hovertemplate="Time: %{x}<br>Occupancy: %{y:.1f}%<extra></extra>"
    ))
    # Threshold Line at 75%
    fig_wave.add_shape(
        type="line", x0=min(times), y0=75, x1=max(times), y1=75,
        line=dict(color="#ff1744", width=1.5, dash="dash")
    )
    # Add text label for the threshold line
    fig_wave.add_annotation(
        x=max(times) * 0.8, y=78,
        text="Threshold: High Occupancy (75%)",
        showarrow=False,
        font=dict(size=9, color="#ff1744")
    )
    
    fig_wave.update_layout(
        margin=dict(l=40, r=20, t=10, b=30),
        height=220,
        showlegend=False,
        hovermode="x unified",
        xaxis=dict(
            zeroline=False,
            title=dict(text=x_title, font=dict(size=9))
        ),
        yaxis=dict(
            zeroline=False,
            title=dict(text="Occupancy Rate (%)", font=dict(size=9)),
            range=[0, 100]
        )
    )
    apply_plotly_theme(fig_wave)
    st.plotly_chart(fig_wave, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ROW 2: BAR GRAPH & DONUT CHART
    col_bar, col_donut = st.columns(2, gap="medium")
    
    # Process dynamic zones data
    if res:
        if res["type"] == "video":
            idx = st.session_state.video_frame_idx
            frames = res["frame_results"]
            if idx >= len(frames): idx = 0
            cur_slots = frames[idx]["slots"]
        else:
            cur_slots = res["slots"]
    else:
        # Mock data
        cur_slots = []
        for i in range(10):
            cur_slots.append({"id": f"A{i+1}", "cls": 0 if i not in [2, 5, 8] else 1})
        for i in range(10):
            cur_slots.append({"id": f"B{i+1}", "cls": 0 if i not in [1, 4, 7, 9] else 1})

    zones_dict = {}
    for s in cur_slots:
        slot_id = s.get("id", "A1")
        zone_letter = slot_id[0]
        zone_name = f"Zone {zone_letter}"
        if zone_name not in zones_dict:
            zones_dict[zone_name] = {"total": 0, "available": 0, "occupied": 0}
        zones_dict[zone_name]["total"] += 1
        if s.get("cls", 0) == 0:
            zones_dict[zone_name]["available"] += 1
        else:
            zones_dict[zone_name]["occupied"] += 1

    sorted_zones = sorted(zones_dict.keys())
    zone_totals = [zones_dict[z]["total"] for z in sorted_zones]
    zone_avail = [zones_dict[z]["available"] for z in sorted_zones]
    zone_occ = [zones_dict[z]["occupied"] for z in sorted_zones]

    with col_bar:
        st.markdown('<div class="premium-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Zone-by-Zone Space Capacity</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.75rem; color:#798ea8; margin-top:-8px;">Comparison of available and occupied spaces in each parking zone.</p>', unsafe_allow_html=True)
        
        fig_bar = go.Figure()
        
        # Add Available spaces bar
        fig_bar.add_trace(go.Bar(
            name='Available Space',
            x=sorted_zones, y=zone_avail,
            marker_color='#00e676',
            width=0.3
        ))
        
        # Add Occupied spaces bar
        fig_bar.add_trace(go.Bar(
            name='Occupied Space',
            x=sorted_zones, y=zone_occ,
            marker_color='rgba(255, 23, 68, 0.8)',
            width=0.3
        ))
        
        fig_bar.update_layout(
            margin=dict(l=40, r=20, t=10, b=30),
            height=200,
            barmode='group',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.99,
                xanchor="right",
                x=1
            ),
            xaxis=dict(zeroline=False),
            yaxis=dict(zeroline=False, title=dict(text="Number of Slots", font=dict(size=9)))
        )
        apply_plotly_theme(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_donut:
        st.markdown('<div class="premium-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Overall Space Distribution</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.75rem; color:#798ea8; margin-top:-8px;">Total system-wide vacancy ratio index.</p>', unsafe_allow_html=True)
        
        fig_donut = go.Figure()
        text_color = '#0f172a' if st.session_state.theme == "light" else '#ffffff'
        
        fig_donut.add_trace(go.Pie(
            labels=['Available Spaces', 'Occupied spaces'],
            values=[vacant_slots, occupied_slots],
            hole=0.6,
            marker=dict(
                colors=['#00e676', 'rgba(255, 23, 68, 0.8)'],
                line=dict(color='#060913' if st.session_state.theme == "dark" else '#ffffff', width=3)
            ),
            hoverinfo='label+percent',
            textinfo='none'
        ))
        
        fig_donut.update_layout(
            margin=dict(l=20, r=20, t=10, b=20),
            height=200,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            annotations=[
                dict(
                    text=f"{occupancy_rate}%",
                    x=0.5, y=0.5,
                    font=dict(size=20, color=text_color, family='Orbitron', weight='bold'),
                    showarrow=False
                )
            ]
        )
        apply_plotly_theme(fig_donut)
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ── DYNAMIC VIDEO STREAM ANIMATION THREAD EXECUTION BLOCK ──────────────────────
if st.session_state.is_playing and res and res["type"] == "video":
    # Compute real playback delay to sync with video timeline
    stride = res.get("frame_stride", 15)
    fps = res.get("fps", 30) or 30
    delay = (stride / fps) / st.session_state.play_speed
    time.sleep(max(0.1, delay)) # cap min delay to avoid infinite fast loops
    st.session_state.video_frame_idx = (st.session_state.video_frame_idx + 1) % len(res["frame_results"])
    st.rerun()

