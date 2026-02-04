# The "Looksmaxxing" Face Scanner

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-00BFFF?style=flat&logo=google&logoColor=white)

Hey! This is a simple Python tool that uses your webcam to analyze your face and give you some "beauty stats." 

**Let’s be real:** It’s definitely **not** a professional tool—it’s just a fun experiment with computer vision.

## 🕹 What does it do?
The script tracks 468 points on your face and tries to calculate:
* **Symmetry:** Are you balanced or a bit lopsided?
* **Eye Stats:** Your "Canthal Tilt" (Hunter vs. Negative) and spacing.
* **Jawline:** It measures your Gonial Angle and E-line in profile view.
* **Archetypes:** It'll guess if you're a "Hunter," "Model," "Royal," or just "Balanced."
* **Best Side:** It tracks which side of your face looks better in profile.

## 🚀 Quick Start

1. **Clone the repo or download the script.**
2. **Install the dependencies:**

   * **Using requirements file (recommended):**
     ```bash
     pip install -r requirements.txt
     ```
   * **Or manually:**
     ```bash
     pip install opencv-python numpy mediapipe
     ```
3.  **Launch the magic:**
    ```bash
    python main.py
    ```
    *(Note: The script will automatically download the `face_landmarker.task` model file on the first run.)*

## ⌨️ Controls
* **Press `m`** to toggle the "cyberpunk" face mask.
* **Press `q`** to quit before the AI judges you too hard.

## ⚠️ Disclaimer
This project was made **for fun and out of boredom**. The "score" and the "archetypes" are based on arbitrary numbers I plugged into the code. Don't go booking a face-lift because a Python script told you your symmetry is 70%. You're beautiful (probably).

## 💻 Tech used
* **Python**
* **MediaPipe** — The "brain" that finds your face landmarks.
* **OpenCV** — The "eyes" that handle the video stream and draw the HUD.
* **NumPy** — For all the math and coordinate geometry.
