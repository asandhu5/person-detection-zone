import os
import tempfile

import cv2
import streamlit as st

from detector import MODEL_OPTIONS, detect_persons, load_model
from zone import (
    draw_alert_banner, draw_detection,
    draw_zone, get_foot_point, is_point_in_zone,
)

st.set_page_config(
    page_title="Restricted Zone Monitor",
    page_icon="🔒",
    layout="wide"
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0d0d1a; color: #e0e0e0; }
    [data-testid="stSidebar"]          { background-color: #12122a; }
    [data-testid="stSidebar"] *        { color: #d0d0d0 !important; }
    h1, h2, h3 { color: #e8e8ff; }
    .stButton > button {
        background-color: #b22222;
        color: white; border: none;
        border-radius: 6px; font-weight: 600;
    }
    .stButton > button:hover { background-color: #8b0000; }
    .alert-banner {
        background: #b22222; color: white; border-radius: 8px;
        padding: 0.7rem 1.1rem; font-weight: bold;
        font-size: 1.05rem; margin: 0.4rem 0;
    }
    .safe-banner {
        background: #1a5c2a; color: white; border-radius: 8px;
        padding: 0.7rem 1.1rem; font-weight: bold;
        font-size: 1.05rem; margin: 0.4rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔒 Person Detection in Restricted Zones")
st.caption("Upload a video → YOLOv10 detects persons → alerts when someone enters the restricted zone")
st.divider()


DEFAULT_ZONE = [
    (200, 120),  # ↖ top-left
    (600, 120),  # ↗ top-right
    (600, 370),  # ↘ bottom-right
    (200, 370),  # ↙ bottom-left
]
CANVAS_W, CANVAS_H = 800, 450


with st.sidebar:
    st.header("⚙️ Settings")

    model_name = st.selectbox(
        "Detection Model",
        list(MODEL_OPTIONS.keys()),
        help="Larger = more accurate but slower. Use 'n' on free cloud tier."
    )
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10, max_value=0.90, value=0.45, step=0.05,
        help="Detections below this score are ignored."
    )
    frame_skip = st.slider(
        "Process every N frames",
        min_value=1, max_value=5, value=1,
        help="Higher = faster but choppier."
    )
    show_conf = st.checkbox("Show confidence on boxes", value=True)
    show_foot = st.checkbox("Show foot-point marker",   value=False)

    st.divider()
    st.markdown("🟢 **Green** — outside zone")
    st.markdown("🔴 **Red** — inside zone (alert)")
    st.markdown("🔵 **Blue outline** — restricted boundary")
    st.divider()

with st.spinner(f"Loading {model_name.split('—')[0].strip()}…"):
    model = load_model(MODEL_OPTIONS[model_name])
st.success("✅ Model loaded.", icon="✅")

col_video, col_stats = st.columns([3, 1], gap="large")

with col_video:
    st.subheader("📹 Video Feed")
    uploaded         = st.file_uploader(
        "Upload video", type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed"
    )
    video_placeholder = st.empty()

with col_stats:
    st.subheader("📊 Live Stats")
    ph_total  = st.empty()
    ph_alert  = st.empty()
    ph_status = st.empty()
    ph_frame  = st.empty()

    st.divider()
    st.subheader("🗺️ Zone Points")
    st.caption(f"Canvas: {CANVAS_W}×{CANVAS_H} px")
    labels = ["↖ TL", "↗ TR", "↘ BR", "↙ BL"]
    for lbl, pt in zip(labels, DEFAULT_ZONE):
        st.code(f"{lbl}: {pt}", language=None)

if not uploaded:
    video_placeholder.info("👆 Upload an MP4, AVI, MOV, or MKV video to start.")
else:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    cap          = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress     = st.progress(0.0, text="Starting…")
    stop_btn     = st.button("⏹️ Stop Processing", type="secondary")
    frame_idx    = 0

    try:
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            frame      = cv2.resize(frame, (CANVAS_W, CANVAS_H))
            detections = detect_persons(model, frame, confidence)

            draw_zone(frame, DEFAULT_ZONE)

            in_zone = 0
            for det in detections:
                foot   = get_foot_point(det["box"])
                inside = is_point_in_zone(foot, DEFAULT_ZONE)
                if inside:
                    in_zone += 1
                draw_detection(
                    frame, det["box"], inside, det["confidence"],
                    show_conf=show_conf, show_foot=show_foot
                )

            draw_alert_banner(frame, in_zone)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb, use_column_width=True)

            ph_total.metric("👤 Persons Detected", len(detections))
            ph_alert.metric("🚨 In Restricted Zone", in_zone)
            ph_frame.metric("🎞️ Frame", f"{frame_idx} / {total_frames}")

            if in_zone > 0:
                ph_status.markdown(
                    f'<div class="alert-banner">⚠️ ALERT: {in_zone} in zone!</div>',
                    unsafe_allow_html=True
                )
            else:
                ph_status.markdown(
                    '<div class="safe-banner">✅ Zone clear</div>',
                    unsafe_allow_html=True
                )

            progress.progress(
                min(frame_idx / max(total_frames, 1), 1.0),
                text=f"Frame {frame_idx} / {total_frames}"
            )

    finally:
        cap.release()
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not stop_btn:
        progress.progress(1.0, text="Complete!")
        st.success("✅ Video processing complete!")
