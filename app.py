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
    page_title="Traffic Vehicle Detection",
    page_icon="🚗",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

.stApp{
background-color:#0f172a;
}

.block-container{
padding-top:1rem;
padding-left:2rem;
padding-right:2rem;
}

h1{
text-align:center;
color:#38bdf8;
font-size:48px;
font-weight:bold;
}

div[data-testid="stMetric"]{

background:#1e293b;

padding:20px;

border-radius:15px;

border:2px solid #38bdf8;

box-shadow:0 0 10px rgba(56,189,248,.25);

}

div[data-testid="stMetric"] label{

color:white !important;

font-size:22px !important;

font-weight:bold;

}

div[data-testid="stMetric"] div{

color:#38bdf8 !important;

font-size:42px !important;

font-weight:bold;

}

</style>
""",unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    return YOLO("best.pt")

model = load_model()

# ==========================================
# HEADER
# ==========================================

st.title("🚗 Traffic Vehicle Detection")

st.divider()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("⚙ Settings")

confidence = st.sidebar.slider(

"Confidence",

0.10,

1.00,

0.30,

0.05

)

imgsz = st.selectbox if False else st.sidebar.selectbox

imgsz = st.sidebar.selectbox(

"Image Size",

[320,416,640],

index=2

)

page = st.sidebar.radio(

"Mode",

(

"Image Detection",

"Video Detection"

)

)

st.sidebar.divider()

st.sidebar.success("""

Classes

🚗 Car

🚚 Heavy

""")
# ==========================================
# IMAGE DETECTION
# ==========================================

if page == "Image Detection":

    st.header("🖼 Image Detection")

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image).convert("RGB")
        image_np = np.array(image)

        if st.button(
            "🚀 Detect Vehicles",
            use_container_width=True
        ):

            with st.spinner("Running Detection..."):

                results = model.predict(
                    image_np,
                    conf=confidence,
                    imgsz=imgsz,
                    verbose=False
                )

            annotated = results[0].plot()

            st.divider()

            # ==============================
            # BIG IMAGES
            # ==============================

            col1, col2 = st.columns([1, 1])

            with col1:

                st.subheader("📷 Original Image")

                st.image(
                    image,
                    use_container_width=True
                )

            with col2:

                st.subheader("🚗 Detection Result")

                st.image(
                    annotated,
                    use_container_width=True
                )

            # ==============================
            # COUNT OBJECTS
            # ==============================

            car_count = 0
            heavy_count = 0

            for box in results[0].boxes:

                cls = int(box.cls[0])

                label = model.names[cls]

                if label == "car":
                    car_count += 1

                elif label == "heavy":
                    heavy_count += 1

            total = car_count + heavy_count

            st.divider()

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "🚗 Cars",
                    car_count
                )

            with c2:
                st.metric(
                    "🚚 Heavy",
                    heavy_count
                )

            with c3:
                st.metric(
                    "📦 Total",
                    total
                )

            # ==============================
            # DOWNLOAD IMAGE
            # ==============================

            result_image = Image.fromarray(annotated)

            result_image.save("detected_image.jpg")

            with open("detected_image.jpg", "rb") as file:

                st.download_button(
                    "⬇ Download Detection",
                    data=file,
                    file_name="detected_image.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
# ==========================================
# VIDEO DETECTION
# ==========================================

if page == "Video Detection":

    st.header("🎥 Video Detection")

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        st.subheader("📹 Original Video")
        st.video(uploaded_video)

        st.divider()

        start_detection = st.button(
            "🚀 Start Detection",
            use_container_width=True
        )

        if start_detection:

            # Save uploaded video
            temp_video = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            temp_video.write(uploaded_video.read())
            temp_video.close()

            cap = cv2.VideoCapture(temp_video.name)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps == 0:
                fps = 30

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            output_path = "detected_video.mp4"

            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

            progress = st.progress(0)
            status = st.empty()

            frame_count = 0
            car_count = 0
            heavy_count = 0

            start_time = time.time()

            while True:

                success, frame = cap.read()

                if not success:
                    break

                results = model.predict(
                    frame,
                    conf=confidence,
                    imgsz=imgsz,
                    verbose=False
                )

                plotted = results[0].plot()

                writer.write(plotted)

                # Count objects
                for box in results[0].boxes:

                    cls = int(box.cls[0])

                    label = model.names[cls]

                    if label == "car":
                        car_count += 1

                    elif label == "heavy":
                        heavy_count += 1

                frame_count += 1

                progress.progress(
                    min(frame_count / total_frames, 1.0)
                )

                status.info(
                    f"Processing Frame {frame_count}/{total_frames}"
                )

            cap.release()
            writer.release()

            elapsed = round(
                time.time() - start_time,
                2
            )

            st.success("✅ Detection Finished")

            st.divider()

            st.subheader("🎬 Detection Result")

            st.video(output_path)

            st.divider()

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "🎞 Frames",
                frame_count
            )

            m2.metric(
                "🚗 Cars",
                car_count
            )

            m3.metric(
                "🚚 Heavy",
                heavy_count
            )

            m4.metric(
                "⏱ Time",
                f"{elapsed} s"
            )

            with open(output_path, "rb") as file:

                st.download_button(
                    "⬇ Download Detected Video",
                    data=file,
                    file_name="detected_video.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )

            try:
                os.remove(temp_video.name)
            except:
                pass
                            