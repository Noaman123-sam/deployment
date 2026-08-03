# ==========================================
# PART 1: IMPORTS + PAGE CONFIG + CSS
# ==========================================

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import time

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Traffic Analysis System",
    page_icon="🚗",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background-color: #0f172a;
}

.block-container {
    padding-top: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

h1 {
    text-align: center;
    color: #38bdf8;
    font-size: 45px;
    font-weight: 800;
}

div[data-testid="stMetric"] {
    background: #1e293b;
    padding: 20px;
    border-radius: 15px;
    border: 2px solid #38bdf8;
}

div[data-testid="stMetric"] label {
    color: white !important;
    font-size: 20px !important;
    font-weight: bold;
}

div[data-testid="stMetric"] div {
    color: #38bdf8 !important;
    font-size: 35px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# PART 2: LOAD YOLO MODEL & SETUP
# ==========================================

@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# Class mapping setup
CLASS_NAMES = model.names
CAR_CLASS = []
HEAVY_CLASS = []

for idx, name in CLASS_NAMES.items():
    if name.lower() == "car":
        CAR_CLASS.append(idx)
    elif name.lower() == "heavy":
        HEAVY_CLASS.append(idx)

TRACKER = "bytetrack.yaml"

# ==========================================
# PART 3: SIDEBAR & HEADER
# ==========================================

st.title("🚗 AI Traffic Detection & Analysis System")
st.divider()

st.sidebar.title("⚙ Settings")

confidence = st.sidebar.slider(
    "🎯 Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.30,
    step=0.05
)

imgsz = st.sidebar.selectbox(
    "📐 Image Size",
    options=[320, 416, 640],
    index=2
)

mode = st.sidebar.radio(
    "🚦 Detection Mode",
    options=[
        "🖼 Image Detection",
        "🎥 Video Detection"
    ]
)

st.sidebar.divider()
st.sidebar.info(
"""
### 🚗 Classes
✅ Car  
✅ Heavy Vehicle  

### Tracking
ByteTrack  

### Model
YOLO
"""
)

# ==========================================
# PART 4: IMAGE DETECTION
# ==========================================

if mode == "🖼 Image Detection":
    st.header("🖼 Vehicle Image Detection")

    uploaded_image = st.file_uploader(
        "Upload Traffic Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        image_np = np.array(image)

        if st.button("🚀 Detect Vehicles", use_container_width=True):
            with st.spinner("Running YOLO Detection..."):
                results = model.predict(
                    image_np,
                    conf=confidence,
                    imgsz=imgsz,
                    verbose=False
                )

            annotated = results[0].plot()
            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📷 Original Image")
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("🚗 Detection Result")
                st.image(annotated, use_container_width=True)

            # Count vehicles
            car_count = 0
            heavy_count = 0

            for box in results[0].boxes:
                cls = int(box.cls[0])
                label = model.names[cls].lower()

                if label == "car":
                    car_count += 1
                elif label == "heavy":
                    heavy_count += 1

            total = car_count + heavy_count
            st.divider()

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("🚗 Cars", car_count)
            with c2:
                st.metric("🚚 Heavy Vehicles", heavy_count)
            with c3:
                st.metric("📦 Total Vehicles", total)

            # Save and download result
            output_image = Image.fromarray(annotated)
            output_image.save("traffic_detection.jpg")

            with open("traffic_detection.jpg", "rb") as file:
                st.download_button(
                    label="⬇ Download Result Image",
                    data=file,
                    file_name="traffic_detection.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )

# ==========================================
# PART 5 & 6: VIDEO DETECTION & ANALYSIS
# ==========================================

if mode == "🎥 Video Detection":
    st.header("🎥 Traffic Video Analysis")

    uploaded_video = st.file_uploader(
        "Upload Traffic Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:
        st.subheader("📹 Original Video")
        st.video(uploaded_video)
        st.divider()

        start = st.button("🚀 Start Tracking", use_container_width=True)

        if start:
            # Save uploaded video to temporary file
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_video.write(uploaded_video.read())
            temp_video.close()

            cap = cv2.VideoCapture(temp_video.name)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or np.isnan(fps):
                fps = 30.0

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            output_path = "traffic_result.mp4"

            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

            progress = st.progress(0)
            status = st.empty()

            tracked_ids = set()
            car_ids = set()
            heavy_ids = set()

            frame_number = 0
            start_time = time.time()
            prev_frame_time = time.time()

            # Process Video Loop
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.track(
                    frame,
                    conf=confidence,
                    imgsz=imgsz,
                    tracker=TRACKER,
                    persist=True,
                    verbose=False
                )

                annotated = results[0].plot()

                # Extract and Save IDs
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    ids = results[0].boxes.id.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()

                    for obj_id, cls in zip(ids, classes):
                        obj_id = int(obj_id)
                        cls = int(cls)

                        tracked_ids.add(obj_id)

                        if cls in CAR_CLASS:
                            car_ids.add(obj_id)
                        elif cls in HEAVY_CLASS:
                            heavy_ids.add(obj_id)

                current_total = len(tracked_ids)
                current_cars = len(car_ids)
                current_heavy = len(heavy_ids)

                # Live Frame-by-Frame FPS Calculation
                new_frame_time = time.time()
                current_fps = 1 / (new_frame_time - prev_frame_time + 1e-5)
                prev_frame_time = new_frame_time

                # Draw overlays
                cv2.putText(annotated, f"Vehicles: {current_total}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated, f"Cars: {current_cars}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(annotated, f"Heavy: {current_heavy}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(annotated, f"FPS: {current_fps:.2f}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                writer.write(annotated)
                frame_number += 1

                if total_frames > 0:
                    progress.progress(min(frame_number / total_frames, 1.0))
                status.info(f"Processing Frame {frame_number}/{total_frames}")

            cap.release()
            writer.release()

            elapsed = round(time.time() - start_time, 2)
            st.success("✅ Tracking Finished")

            # ==========================================
            # TRAFFIC ANALYSIS & DISPLAY RESULTS
            # ==========================================
            total_vehicles = len(tracked_ids)
            total_cars = len(car_ids)
            total_heavy = len(heavy_ids)

            st.divider()
            st.subheader("📊 Traffic Analysis")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🚗 Cars", total_cars)
            with col2:
                st.metric("🚚 Heavy Vehicles", total_heavy)
            with col3:
                st.metric("📦 Total Vehicles", total_vehicles)
            with col4:
                st.metric("⏱ Processing Time", f"{elapsed} sec")

            st.divider()
            st.subheader("🚦 Traffic Density")

            if total_vehicles < 10:
                density = "🟢 Low Traffic"
            elif total_vehicles < 30:
                density = "🟡 Medium Traffic"
            else:
                density = "🔴 High Traffic"

            st.success(density)
            st.divider()

            st.subheader("🎬 Detection Result Video")
            st.video(output_path)

            st.divider()

            with open(output_path, "rb") as file:
                st.download_button(
                    label="⬇ Download Detection Video",
                    data=file,
                    file_name="traffic_analysis_result.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )

            # Cleanup Temp File
            try:
                os.remove(temp_video.name)
            except Exception:
                pass