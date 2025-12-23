# Bird Counting – FastAPI Project

This project is a FastAPI-based backend application that detects and counts birds in uploaded video files using a YOLO object detection model.

## What the project does
- Accepts a video upload via FastAPI
- Extracts frames from the video
- Runs object detection on each frame using YOLO
- Counts detected birds
- Saves processed frames with bounding boxes
- Returns total bird count as API response

## Tech Stack
- Python
- FastAPI
- OpenCV
- YOLO (Ultralytics)
- NumPy

## How to run the project

1. Create virtual environment  
```bash
python -m venv venv
source venv/bin/activate
