import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    total_detections = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.4)
        boxes = results[0].boxes

        if boxes is not None:
            total_detections += len(boxes)

        annotated = results[0].plot()
        out.write(annotated)

    cap.release()
    out.release()

    return {
        "total_detections": total_detections
    }
