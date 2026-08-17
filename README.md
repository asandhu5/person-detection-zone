# 🔒 Person Detection in Restricted Zones

> Upload a video → YOLOv10 detects every person → alerts when anyone enters the restricted zone.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white)
![YOLOv10](https://img.shields.io/badge/YOLOv10-Ultralytics-00BFFF)
![Tests](https://img.shields.io/badge/Tests-59%20passed-brightgreen)


---

## Demo


### Tracking outside the zone

Selecting the Model and uploading the video.
<img src="assets/model_selection.gif" width="650" >

### Zone crossing triggers the alert

The moment a detection falls inside the polygon, the box switches to red and the alert banner fires.

<img src="assets/crossing_alert.gif" width="650">


---

## ▶️ HOW TO RUN (Step by Step)

### Step 1 — Clone this repo

```bash

git clone git@github.com:asandhu5/person-detection-zone.git
cd person-detection-zones
```

### Step 2 — Create a virtual environment

```bash
# Mac / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install all dependencies

```bash
pip install -r requirements.txt
```

This installs Streamlit, Ultralytics (YOLOv10), OpenCV, NumPy, Pillow, and pytest.


### Step 4 — Run the app

```bash
streamlit run app.py
```

Your browser opens at http://localhost:8501

**On first launch:** Ultralytics automatically downloads the YOLOv10 weights
(~6 MB for nano). This only happens once — they are cached locally.

### Step 5 — Upload a video and test it

Upload any MP4 video that contains people walking around.
The blue rectangular zone is drawn on screen. Anyone stepping inside
gets a red box + alert banner. Anyone outside gets a green box.

---

## 🗺️ How to Move the Restricted Zone

Open `app.py` and find this near the top:

```python
DEFAULT_ZONE = [
    (200, 120),  # top-left
    (600, 120),  # top-right
    (600, 370),  # bottom-right
    (200, 370),  # bottom-left
]
```

These are pixel coordinates on an **800×450** canvas.
Change the numbers, save the file, and the app reloads automatically.
The zone can have any number of points — not just four.



## ⚠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: ultralytics` | Run `pip install -r requirements.txt` |
| `ModuleNotFoundError: cv2` | Run `pip install opencv-python-headless` |
| No boxes appearing on video | Lower the confidence threshold in sidebar |
| App slow on Streamlit Cloud | Set "Process every N frames" to 3 in sidebar |
| Zone in wrong position | Edit `DEFAULT_ZONE` coordinates in `app.py` |
| Video won't open | Convert to H.264 MP4 and try again |