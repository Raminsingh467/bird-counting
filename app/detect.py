from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")

FRAMES_DIR = "frames"
OUTPUT_DIR = "detected_frames"

def detect_birds():
    for img_name in os.listdir(FRAMES_DIR):
        img_path = os.path.join(FRAMES_DIR, img_name)
        img = cv2.imread(img_path)

        results = model(img)

        for r in results:
            img = r.plot()

        out_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(out_path, img)
