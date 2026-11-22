"""Builds train_classifier.ipynb from train_classifier.py + export_yolo.py (keeps them in sync).
Run: python make_notebook.py"""
import json
from pathlib import Path

here = Path(__file__).parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


cells = [
    md("""
# FreshCheck - train the fresh/rotten classifier (Colab)
UPES CSAI3027P - Project 26. **Runtime -> Change runtime type -> T4 GPU**, then *Runtime -> Run all*.

Outputs (downloaded as `freshcheck_models.zip` at the end):
`resnet18_fresh.onnx`, `yolov8n.onnx`, `metrics.json`, `labels.json`, `confusion_matrix.png`, `training_curves.png`.
Copy the two `.onnx` files into the repo's `freshcheck/models/` folder.
"""),
    code("!pip -q install imagehash kagglehub onnx onnxruntime ultralytics"),
    md("""
## 1. Download the dataset
Kaggle: *Fruits fresh and rotten for classification* (`sriramr/fruits-fresh-and-rotten-for-classification`).
If `kagglehub` asks for credentials: Kaggle -> Settings -> API -> *Create New Token*, then upload `kaggle.json`
when prompted by the cell below (or set `KAGGLE_USERNAME` / `KAGGLE_KEY` in Colab secrets).
"""),
    code("""
import os
try:
    import kagglehub
    DATA = kagglehub.dataset_download("sriramr/fruits-fresh-and-rotten-for-classification")
except Exception as e:
    print("kagglehub failed:", e, "-> upload kaggle.json")
    from google.colab import files
    files.upload()  # choose kaggle.json
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    os.system("mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json")
    os.system("kaggle datasets download -d sriramr/fruits-fresh-and-rotten-for-classification -p /content/data --unzip")
    DATA = "/content/data"
print("DATA =", DATA)
!find "$DATA" -maxdepth 4 -type d | head -20
"""),
    md("## 2. Training script (same file as `training/train_classifier.py` in the repo)"),
    code("%%writefile train_classifier.py\n" + (here / "train_classifier.py").read_text()),
    md("## 3. Train (~5 epochs, a few minutes on a T4), evaluate on the test split, export ONNX"),
    code('!python train_classifier.py --data "$DATA" --out out --epochs 5'),
    code("""
from IPython.display import Image, display
display(Image("out/training_curves.png")); display(Image("out/confusion_matrix.png"))
import json; m = json.load(open("out/metrics.json"))
print({k: m[k] for k in ["test_macro_f1", "test_accuracy", "rotten_shown_green_rate", "onnx_max_abs_diff"]})
"""),
    md("## 4. Export COCO-pretrained YOLOv8n to ONNX (no training)"),
    code("%%writefile export_yolo.py\n" + (here / "export_yolo.py").read_text()),
    code("!python export_yolo.py --out out"),
    md("## 5. Download everything"),
    code("""
!cd out && zip -q ../freshcheck_models.zip resnet18_fresh.onnx yolov8n.onnx metrics.json labels.json split.json *.png
from google.colab import files
files.download("freshcheck_models.zip")
"""),
]

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
(here / "train_classifier.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote train_classifier.ipynb")
