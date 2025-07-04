# Real-Time Safety Helmet Detection System

This is a real-time safety monitoring system that uses computer vision and AI to detect whether people are wearing safety helmets in a live webcam feed. It combines YOLO object detection, custom color-based helmet detection, and real-time edge enhancement via Canny filter (implemented in PyTorch). A modern GUI is provided using Tkinter for visualization, logging, and screenshot review.

# Features

- Detects persons using YOLOv8
- Detects yellow/red safety helmets using HSV color range
- Canny-based edge enhancement using PyTorch
- Takes automatic screenshots every few seconds
- Displays logs and a screenshot gallery in the GUI
- Dark theme GUI with tkinter and PIL
- Real-time performance with multithreading

# Dependencies

- Python 3.8+
- ultralytics (for YOLOv8)
- opencv-python
- torch , torchvision
- Pillow
- tkinter (comes with most Python installations)

# Installation

pip install ultralytics opencv-python torch torchvision Pillow
