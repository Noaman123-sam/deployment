# 🚗 AI Traffic Detection & Analysis System

A real-time traffic detection and analysis system built with a custom-trained **YOLO** model, using **Python** and **OpenCV**, deployed as an interactive web application.

🔗 **Live Demo:** https://deployment-pwhhtmy2hyaq6lenvxfwew.streamlit.app/


---

## 📌 Overview

This project detects and analyzes vehicles and traffic patterns in real time from video or webcam feeds. A custom-trained YOLO object detection model identifies vehicles, and the system processes the detections to provide traffic-related insights (e.g. vehicle counts, flow, congestion) through a simple web interface.

---

## 🎯 Problem Statement

Manual traffic monitoring is slow and doesn't scale. Automated, real-time vehicle detection can support traffic management, congestion analysis, and smart-city applications by turning raw video feeds into structured, actionable data.

---

## 🗂️ Dataset

- **Source:** [dataset name / link — e.g. Roboflow / custom-collected footage]
- **Classes detected:** [e.g. car, truck, bus, motorcycle]
- **Annotation format:** [e.g. YOLO format via Roboflow]

---

## ⚙️ Approach

1. **Data Preparation & Annotation**
   - Collected/sourced traffic footage and images
   - Annotated vehicle classes for object detection training

2. **Model Training**
   - Custom-trained a YOLO model on the annotated dataset
   - Tuned hyperparameters to balance detection speed and accuracy

3. **Detection Pipeline**
   - Built a real-time inference pipeline using OpenCV to process video/webcam frames
   - Applied the trained YOLO model to detect and track vehicles frame-by-frame

4. **Evaluation**
-mAP@50–95: 80.83%
-Precision: 94.22%
-Recall: 89.72%

5. **Deployment**
   - Wrapped the detection pipeline into a web application for real-time traffic analysis
---

## 🛠️ Tech Stack

- **Language:** Python
- **Object Detection:** YOLO
- **Computer Vision:** OpenCV
- **Deployment:** [Add framework — e.g. Streamlit / Flask]

---

## 🚀 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/Noaman123-sam/[repo-name].git
cd [repo-name]

# Install dependencies
pip install -r requirements.txt

# Run the application
[add run command, e.g. streamlit run app.py]
```

---

## 📁 Project Structure

```
[repo-name]/
│
├── app.py                # Web application entry point
├── model/                 # Trained YOLO weights
├── data/                   # Sample images/videos for testing
├── notebooks/            # Training & experimentation notebooks
├── requirements.txt       # Project dependencies
└── README.md
```

*(Update this to match your actual repo structure)*

---

## 📈 Results & Insights

- [Add 2–3 bullet points — e.g. detection accuracy achieved, real-time performance, challenges with occlusion/lighting, etc.]

---

## 🔮 Future Improvements

- Add vehicle tracking (e.g. DeepSORT) for counting and flow analysis, not just detection
- Improve performance in low-light/night conditions
- Add traffic density heatmaps and analytics dashboard
- Deploy on a permanent hosting platform

---
