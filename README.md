# Real-Time Safety Helmet Detection with MediaPipe Pose

This project implements a real-time safety helmet detection system that leverages MediaPipe Pose for accurate head keypoint detection. By combining pose landmarks with HSV color-based helmet detection (yellow and red helmets), it identifies people wearing safety helmets in a live webcam feed.

# Features

- Head and body keypoint detection using MediaPipe Pose
- Yellow and red helmet detection via HSV color filtering
- Visual bounding boxes highlighting heads and helmets
- Automatic screenshot capture with helmet count overlay
- Dark-themed GUI built with Tkinter and PIL
- Event logging and screenshot management
- Optimized for real-time performance with GPU support when available

# Dependencies

Python 3.8+
mediapipe
opencv-python
torch
pillow
tkinter (usually included with Python)

#Installation
pip install mediapipe opencv-python torch pillow
