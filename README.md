# 👁️ FaceFlow · AI Focus Detector

> **Real-time focus detection powered by computer vision and machine learning — no wearables, no special hardware, just your webcam.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-00897B?style=flat-square)](https://mediapipe.dev)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 📌 Overview

**FaceFlow** is a real-time AI-powered focus detection system that uses computer vision and machine learning to determine whether a student or professional is mentally engaged or distracted. It detects focus state through subtle eye and facial cues — providing instant feedback and detailed session analytics to help you stay productive.

| Metric | Value |
|---|---|
| 🎯 Model Accuracy | **89.8%** |
| 🖼️ Training Images | **84,000+** |
| 🗺️ Facial Landmarks | **468** |
| ⚡ Inference Time | **~30ms** |
| 🔒 Data Privacy | **100% local — nothing sent to any server** |

---

## ✨ Features

### 📸 Photo Analysis Mode
Upload any front-facing photo to instantly detect the focus state. The system overlays MediaPipe face mesh landmarks and shows detailed confidence scores. Every result is auto-logged to History.

### 🎥 Live Webcam Mode
Real-time focus detection at ~20 FPS using your webcam. A live HUD displays focus state and session timer. Distraction alerts are counted and the full session timeline is auto-saved to Reports when you stop.

### 📊 Session Analytics & Reports
Every session produces a **focus score**, a timeline bar chart, and a focus/distraction pie chart. Full PNG reports can be exported for personal productivity tracking.

### 🕐 History Log
Every photo analysis is automatically logged with timestamp, filename, prediction, and confidence score. Browse and clear your history at any time.

### 🔒 Privacy First
All processing happens **100% locally** on your device. No images, video, or personal data are ever sent to any server. FaceFlow works entirely offline after installation — your face data never leaves your machine.

---

## 🛠️ Technology Stack

| Technology | Role |
|---|---|
| 🐍 **Python 3.10+** | Core language |
| 🤖 **MediaPipe** | Google's real-time 468-point face mesh detection |
| 📷 **OpenCV** | Video capture, image processing & frame annotation |
| 🌲 **scikit-learn** | Random Forest classifier trained on 84K+ eye images |
| 🔢 **NumPy** | Fast numerical computation for feature extraction |
| 🌊 **Streamlit** | Interactive web UI with real-time webcam streaming |
| 📊 **Matplotlib** | Focus timeline and pie chart report generation |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- A webcam (for Live Mode)
- `focus_model.pkl` trained model file (see [Model Training](#-model-training) below)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/faceflow.git
cd faceflow
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Place the trained model**

Copy `focus_model.pkl` into the project root directory (same folder as `app.py`).

**5. Run the app**
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📦 Requirements

Create a `requirements.txt` with the following:

```
streamlit>=1.28.0
opencv-python>=4.8.0
mediapipe>=0.10.0
scikit-learn>=1.3.0
numpy>=1.24.0
matplotlib>=3.7.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🔬 Detection Pipeline

```
Frame / Image
      │
      ▼
┌─────────────────────────┐
│   1. Frame Capture      │  OpenCV VideoCapture — flipped for mirror view
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  2. Face Mesh Detection │  MediaPipe FaceMesh — 468 3D landmarks
│     (MediaPipe)         │  Eye indices: 33, 263, 159, 386
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  3. Eye Region Crop     │  Precise bilateral eye crop with 10–20 px margin
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  4. Feature Extraction  │  Resize to 64×32 px → grayscale → 6 features:
│                         │  mean brightness, upper/lower split brightness,
│                         │  std deviation, bright pixels, dark pixels
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  5. Random Forest       │  focus_model.pkl → binary prediction
│     Classification      │  focused = 1 / distracted = 0 + confidence scores
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  6. Result Display      │  Confidence bars, landmark overlays, history log,
│     & Logging           │  session charts, exportable PNG reports
└─────────────────────────┘
```

---

## 🧠 Model Training

The classifier is a **Random Forest** model trained on 84,000+ labeled eye images.

### Feature Vector (6 features per sample)

| # | Feature | Description |
|---|---|---|
| 1 | Mean brightness | Average pixel intensity of the eye region |
| 2 | Upper brightness | Mean intensity of the top half (eyelid area) |
| 3 | Lower brightness | Mean intensity of the bottom half |
| 4 | Standard deviation | Pixel intensity variance (texture measure) |
| 5 | Bright pixel count | Pixels with intensity > 150 (open eye indicator) |
| 6 | Dark pixel count | Pixels with intensity < 80 (closed eye / shadow) |

### Training the Model

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# X = feature matrix (n_samples, 6)
# y = labels (1 = focused, 0 = distracted)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Accuracy: {model.score(X_test, y_test):.2%}")  # 89.8%

with open("focus_model.pkl", "wb") as f:
    pickle.dump(model, f)
```

---

## 📁 Project Structure

```
faceflow/
│
├── app.py                  # Main Streamlit application
├── focus_model.pkl         # Trained Random Forest model (required)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .streamlit/ 
│    └── config.toml
└── assets/                 # Optional: screenshots, demo GIFs
    └── demo.png
```

---

## 🖥️ App Pages

| Page | Description |
|---|---|
| **📸 Photo Analysis** | Upload a photo → detect focus state → view landmarks and confidence |
| **🎥 Live Webcam** | Start a real-time session → get live HUD → auto-save session to Reports |
| **📊 Reports** | View all webcam session reports and photo analysis summaries with charts |
| **🕐 History** | Browse the full log of all photo analyses with timestamps and confidence |
| **ℹ️ About** | Project overview, pipeline explanation, tech stack, and creator info |

---

## ⚠️ Troubleshooting

**`focus_model.pkl` not found**
> Place the model file in the same directory as `app.py` and restart the app.

**No face detected**
> Ensure the photo is a clear, front-facing image with adequate lighting. Avoid extreme angles or occlusions.

**Camera not accessible (Live Mode)**
> Grant browser/OS camera permissions. Close other apps that may be using the webcam. Try a different browser if the issue persists.

**MediaPipe installation fails**
> Try: `pip install mediapipe --upgrade`. On Apple Silicon Macs, use `pip install mediapipe-silicon` if available.

**Low detection accuracy**
> Ensure good, even lighting on your face. Avoid backlit environments or strong shadows across the eye region.

---

## 🗺️ Roadmap

- [ ] Drowsiness detection via eye aspect ratio (EAR)
- [ ] Multi-face support for classroom environments
- [ ] Exportable session history as CSV
- [ ] Email/Slack distraction alert integration
- [ ] Mobile-friendly responsive UI
- [ ] Docker deployment support

---

## 👩‍💻 About the Creator

**Hamna Munir** — AI/ML Engineer · Software Engineering Student

Built FaceFlow from scratch over 4 weeks as a passion project exploring real-world computer vision. The system demonstrates a full end-to-end ML pipeline: dataset curation, model training, feature engineering, and production deployment via Streamlit.

Hamna is a Software Engineering student from Pakistan specializing in AI/ML systems.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [MediaPipe](https://mediapipe.dev) by Google — for the incredible real-time face mesh solution
- [Streamlit](https://streamlit.io) — for making ML app deployment effortless
- [scikit-learn](https://scikit-learn.org) — for the robust Random Forest implementation
- The open-source computer vision community for datasets and research

---

<div align="center">
  <strong>FaceFlow · v1.0 · 2026</strong><br/>
  Built with ❤️ by <strong>Hamna Munir</strong> · Python · OpenCV · MediaPipe · scikit-learn · Streamlit
</div>
