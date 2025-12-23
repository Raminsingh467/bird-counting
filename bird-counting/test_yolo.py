from ultralytics import YOLO

model = YOLO ("yolov8n.pt")
result = model("https://ultralytics.com/images/bus.jpg")

print("yolo working!")
