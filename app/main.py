from fastapi import FastAPI, UploadFile, File
import cv2
import os

app = FastAPI()

# Directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "detected_frames"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    video_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save uploaded video
    with open(video_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Save every 10th frame
        if frame_count % 10 == 0:
            frame_filename = f"frame_{frame_count:04d}.jpg"
            frame_path = os.path.join(OUTPUT_DIR, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_frames += 1

        frame_count += 1

    cap.release()

    return {
        "filename": file.filename,
        "total_frames": frame_count,
        "saved_frames": saved_frames
    }
