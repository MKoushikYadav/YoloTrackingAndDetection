from ultralytics import YOLO
import torch

data_loader = torch.d
model = YOLO('yolov8m.pt')

results=model.track()