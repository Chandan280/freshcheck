# models/

- `yolov8n.onnx` - COCO-pretrained YOLOv8n exported with `training/export_yolo.py` (opset 17, 640x640, output [1,84,8400]). Not fine-tuned.
- `resnet18_fresh.onnx` - **create it** with `training/train_classifier.ipynb` in Colab and copy it here. Input `input` [N,3,224,224], output `logits` [N,6] in the label order of `app/config.json`.

The server refuses to start if either file is missing.
