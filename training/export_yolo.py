"""Export the COCO-pretrained YOLOv8n detector to ONNX (no training).
COCO ids used by the app: 46 banana, 47 apple, 49 orange.
Usage: python export_yolo.py --out ../models
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="../models")
args = ap.parse_args()

model = YOLO("yolov8n.pt")  # downloads the official COCO weights
onnx_path = model.export(format="onnx", opset=17, imgsz=640, dynamic=False, simplify=True)
assert model.names[46] == "banana" and model.names[47] == "apple" and model.names[49] == "orange"
Path(args.out).mkdir(parents=True, exist_ok=True)
shutil.copy(onnx_path, Path(args.out) / "yolov8n.onnx")
print("saved", Path(args.out) / "yolov8n.onnx")
