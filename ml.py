from ultralytics import YOLO
import torch
try:
   model = YOLO('yolov8m.pt')
   results = model.train(
   data='dataset\imgs\data.yaml',
   imgsz=640,
   epochs=10,
   batch=2,
   name='yolov8',
   device=0)
except Exception as e:
   print(f"Caught an exception: {e}")

