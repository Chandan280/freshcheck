"""Puts the trained model in models/ automatically.
Looks for freshcheck_models.zip (from the Colab notebook) in this folder, its parent and ~/Downloads,
and extracts resnet18_fresh.onnx (+ yolov8n.onnx if missing). Exit code 1 = model still missing."""
import sys
import zipfile
from pathlib import Path

here = Path(__file__).resolve().parent
models = here / "models"
target = models / "resnet18_fresh.onnx"

if not target.exists():
    places = [here, here.parent, Path.home() / "Downloads", Path.home() / "Desktop"]
    zips = [z for p in places if p.exists() for z in p.glob("freshcheck_models*.zip")]
    loose = [p / "resnet18_fresh.onnx" for p in places if (p / "resnet18_fresh.onnx").exists()]
    if zips:
        z = max(zips, key=lambda f: f.stat().st_mtime)  # newest download
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                base = Path(name).name
                if base == "resnet18_fresh.onnx" or (base == "yolov8n.onnx" and not (models / base).exists()):
                    (models / base).write_bytes(zf.read(name))
        print(f"Found {z} -> extracted model(s) into models/")
    elif loose:
        target.write_bytes(loose[0].read_bytes())
        print(f"Copied {loose[0]} into models/")

if not target.exists():
    print("""
========================================================================
 FreshCheck needs the trained model:  models/resnet18_fresh.onnx
 1. Open training/train_classifier.ipynb in Google Colab (T4 GPU) -> Run all
 2. It downloads freshcheck_models.zip -> just leave it in your Downloads folder
 3. Run this script again (it finds the zip automatically)
========================================================================""")
    sys.exit(1)
print("Models OK:", ", ".join(sorted(p.name for p in models.glob("*.onnx"))))
