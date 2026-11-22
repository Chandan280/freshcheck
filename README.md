---
title: FreshCheck
emoji: 🍎
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# FreshCheck 🍎🍌🍊

Low-cost fruit freshness scanner for street vendors, households and food banks in India.
Point a phone or laptop camera at apples, bananas or oranges: FreshCheck draws a live box around each
fruit and shows **Fresh / Use soon / Rotten**, a big colour + icon + text action, a prototype
"use within ~X days" estimate, and the food saved this session (kg and ₹).

UPES CSAI3027P Deep Learning, Project 26.

## ⚡ Quick start (2 steps)

1. **Get the model once:** open `training/train_classifier.ipynb` in Google Colab (T4 GPU) → *Run all*.
   It downloads `freshcheck_models.zip`; leave it in your **Downloads** folder.
2. **Run:**
   - **Windows:** double-click **`run.bat`** (needs Python from python.org).
   - **Mac / Linux / WSL:** `./run.sh`

The launcher finds the zip, installs what it needs (first run only), starts the server and opens
**http://localhost:7860**. Press **Start camera**. `run.sh` builds the C++ server; if C++ tools are
missing it starts the Python server instead (same app). `./run.sh python` forces Python.

> Decision support only, not a food-safety certificate. When in doubt, throw it out.
> Shelf-life is a prototype estimate.

## How it works

```
camera frame (JPEG, ≤640 px, every 700 ms)
  └─ POST /api/grade ──► C++17 server (ONNX Runtime CPU + OpenCV)
                           1. YOLOv8n (COCO, pretrained)  keep banana(46) apple(47) orange(49), conf > 0.4, NMS
                           2. crop each fruit             (no fruit found → classify the whole image)
                           3. ResNet-18 (fine-tuned)      6 classes → p_rotten = Σ P(rotten*)
                           4. Grader (config.json)        p_rotten < 0.35 Fresh · 0.35–0.65 Medium · > 0.65 Rotten
  ◄─ JSON {fruits:[{name, box, grade, confidence, action, action_color, use_within_days, message_en, message_hi}], latency_ms}
```

| Folder | What |
|---|---|
| `training/` | `train_classifier.ipynb` (Colab), `train_classifier.py`, `export_yolo.py` |
| `models/` | `yolov8n.onnx` (included), `resnet18_fresh.onnx` (**you create it in Colab**) |
| `app/` | C++17 server: `Detector`, `Classifier`, `Grader`, `Pipeline`, `Server`; `config.json`; `web/index.html` |
| `app_fallback/` | SAFETY NET: Python FastAPI server, same API, same UI |
| `Dockerfile` | Hugging Face Spaces image (C++); `Dockerfile.fallback` = Python version |

## 1. Train the classifier (Google Colab, ~10 min)

1. Open `training/train_classifier.ipynb` in Colab (File → Upload notebook).
2. Runtime → Change runtime type → **T4 GPU**. Runtime → **Run all**.
3. If Kaggle asks for credentials, upload your `kaggle.json` (Kaggle → Settings → API → Create New Token).
4. It downloads `freshcheck_models.zip`. Put **`resnet18_fresh.onnx`** into `freshcheck/models/`
   (`yolov8n.onnx` is already there; the zip has a fresh copy too).
5. Write down the printed numbers (test macro F1, confusion matrix, "rotten shown GREEN" %) from
   `metrics.json` for the report. Do not use numbers from anywhere else.

The notebook: dedupes with `imagehash`, makes a stratified 70/15/15 split (seed 42), fine-tunes
ResNet-18 (ImageNet weights, 224×224, light augmentation, 5 epochs), prints macro F1 + confusion matrix,
saves plots, exports ONNX opset 17 and checks ONNX vs PyTorch outputs.

## 2. Build and run locally (Ubuntu / WSL)

Run everything from the `freshcheck/` folder.

```bash
sudo apt install -y build-essential cmake curl libopencv-dev
```

```bash
mkdir -p ~/onnxruntime && curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz | tar -xz -C ~/onnxruntime --strip-components=1
```

```bash
cmake -S app -B app/build -DONNXRUNTIME_DIR=$HOME/onnxruntime
```

```bash
cmake --build app/build -j
```

Start the server, then open **http://localhost:7860**:

```bash
./app/build/freshcheck
```

CLI (prints JSON):

```bash
./app/build/freshcheck --image path/to/apple.jpg
```

macOS: `brew install cmake opencv`, use `onnxruntime-osx-arm64-1.20.1.tgz` (Apple Silicon) in the curl
command; the rest is the same. Windows: use WSL or Docker.

### Or with Docker

```bash
docker build -t freshcheck .
```

```bash
docker run --rm -p 7860:7860 freshcheck
```

### Safety net: Python server (if the C++ build fails)

Same API, same `config.json`, same `index.html`, same models.

```bash
pip install -r app_fallback/requirements.txt
```

```bash
python app_fallback/server.py
```

## 3. Test the webcam

- **Laptop:** open http://localhost:7860 → *Start camera* → allow camera. Boxes appear on the video
  about every 0.7 s. Press **📸 Snapshot (log)** to log the fruit (this is what counts in the
  Vendor / Home / Food bank lists and in "Food saved"). Live frames are shown but not counted.
- **Phone:** browsers only allow the camera on **HTTPS** (or localhost). Use the Hugging Face URL
  (below), or tunnel your laptop: `cloudflared tunnel --url http://localhost:7860` and open the
  `https://…trycloudflare.com` link on the phone. The back camera is used by default; *Switch camera*
  flips it.
- No camera / permission denied: the page says so; use **Upload photo**.
- API check without a browser:

```bash
curl -F image=@apple.jpg "http://localhost:7860/api/grade?record=1"
```

```bash
curl http://localhost:7860/api/stats
```

## 4. Deploy to Hugging Face Spaces (free CPU)

1. huggingface.co → New Space → SDK **Docker** → Blank → CPU basic (free) → Create.
2. Push the contents of this `freshcheck/` folder (this README's header configures the Space):

```bash
git clone https://huggingface.co/spaces/YOUR_USER/freshcheck hf-space
```

```bash
cp -r freshcheck/. hf-space/ && cd hf-space
```

```bash
git lfs install && git lfs track "*.onnx" && git add .gitattributes . && git commit -m "FreshCheck" && git push
```

3. Wait for the build (~5 min, see the *Logs* tab). Open **https://YOUR_USER-freshcheck.hf.space**
   directly on the phone (the camera works there because it is HTTPS).
4. If the C++ build fails on HF: `mv Dockerfile.fallback Dockerfile`, commit, push.

## API

| Method | Path | |
|---|---|---|
| POST | `/api/grade` | multipart field `image`, or JSON `{"image": "<base64 or data URL>", "record": true}`, or raw image body. `?record=1` logs the fruit into the session totals. |
| GET | `/api/stats` | `{scans, fruits_logged, by_grade, by_fruit, kg_saved, inr_saved, avg_latency_ms}` |
| POST | `/api/reset` | reset session totals |
| GET | `/health` | `{"status":"ok"}` |
| GET | `/` | web UI |

Each fruit: `name, box [x,y,w,h], grade, confidence, p_rotten, action, action_color, use_within_days, message_en, message_hi`.
`fallback: true` means YOLO found no fruit and the whole image was classified (dashed box in the UI).

## Configuration (`app/config.json`)

Grade thresholds, shelf-life base days (Fresh 5 / Medium 2 / Rotten 0, scaled by
`1 − k + k·(1 − p_rotten)`), action texts (English + Hindi), detector thresholds, average fruit weight
and ₹/kg used for "food saved". Session totals are kept in server memory and shared by everyone using
the same server.

## Known limits (prototype)

- Only apple, banana, orange. Trained on a Kaggle studio-style dataset; lighting, unseen varieties and
  cluttered backgrounds will reduce accuracy. Internal rot that is not visible cannot be detected.
- YOLOv8n is COCO-pretrained, not fine-tuned: it sometimes misses very rotten or very close-up fruit;
  then the whole frame is classified. On a non-fruit scene that fallback can still produce a confident
  wrong label, so the UI marks it "whole photo (no box found)".
- "Use within X days" and "food saved" are rule-based estimates, not measurements.
