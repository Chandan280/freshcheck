"""
FreshCheck SAFETY NET: Python (FastAPI + onnxruntime) server with the SAME API as the C++ app.
Use it only if the C++ build fails. It reads the same app/config.json, models/ and app/web/index.html.

    pip install -r app_fallback/requirements.txt
    python app_fallback/server.py            # -> http://localhost:7860
    python app_fallback/server.py --image apple.jpg
"""
import argparse
import base64
import io
import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("FRESHCHECK_CONFIG", ROOT / "app" / "config.json"))
CFG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
BASE = CONFIG_PATH.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("freshcheck")


def session(path):
    log.info("Loading %s", path)
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


class Detector:
    def __init__(self, cfg):
        self.s = session(BASE / cfg["model"])
        self.inp = self.s.get_inputs()[0].name
        self.size, self.conf, self.iou = cfg["input_size"], cfg["conf_threshold"], cfg["nms_iou"]
        self.pad, self.max = cfg["box_padding"], cfg["max_fruits"]
        self.classes = {int(k): v for k, v in cfg["coco_classes"].items()}

    def detect(self, img):  # img: PIL RGB
        W, H = img.size
        sc = min(self.size / W, self.size / H)
        nw, nh = round(W * sc), round(H * sc)
        px, py = (self.size - nw) // 2, (self.size - nh) // 2
        canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
        canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
        x = (np.asarray(canvas, np.float32) / 255.0).transpose(2, 0, 1)[None]
        out = self.s.run(None, {self.inp: x})[0][0]  # [84, N]
        if out.shape[0] > out.shape[1]:
            out = out.T
        ids = list(self.classes)
        scores = out[[4 + i for i in ids]]            # [3, N]
        best, which = scores.max(0), scores.argmax(0)
        keep = best >= self.conf
        boxes = out[:4, keep].T
        best, which = best[keep], which[keep]
        # cx,cy,w,h (letterbox) -> x,y,w,h (original)
        xywh = np.stack([(boxes[:, 0] - boxes[:, 2] / 2 - px) / sc, (boxes[:, 1] - boxes[:, 3] / 2 - py) / sc,
                         boxes[:, 2] / sc, boxes[:, 3] / sc], 1)
        order, chosen = np.argsort(-best), []
        for i in order:  # class-agnostic NMS
            if all(iou(xywh[i], xywh[k]) <= self.iou for k in chosen):
                chosen.append(i)
            if len(chosen) >= self.max:
                break
        dets = []
        for i in chosen:
            x0, y0, w, h = xywh[i]
            ex, ey = w * self.pad, h * self.pad
            x1, y1 = max(0, int(x0 - ex)), max(0, int(y0 - ey))
            x2, y2 = min(W, int(x0 - ex) + int(w + 2 * ex)), min(H, int(y0 - ey) + int(h + 2 * ey))
            if x2 - x1 >= 8 and y2 - y1 >= 8:
                dets.append(([x1, y1, x2 - x1, y2 - y1], self.classes[ids[which[i]]], float(best[i])))
        return dets


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


class Classifier:
    def __init__(self, cfg):
        self.s = session(BASE / cfg["model"])
        self.inp = self.s.get_inputs()[0].name
        self.size, self.labels = cfg["input_size"], cfg["labels"]
        self.mean, self.std = np.array(cfg["mean"], np.float32), np.array(cfg["std"], np.float32)
        self.min_conf = cfg.get("fallback_min_confidence", 0.6)

    def classify(self, crops):
        x = np.stack([((np.asarray(c.resize((self.size, self.size), Image.BILINEAR), np.float32) / 255.0
                        - self.mean) / self.std).transpose(2, 0, 1) for c in crops])
        logits = self.s.run(None, {self.inp: x})[0]
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        rot = [i for i, l in enumerate(self.labels) if l.startswith("rotten")]
        res = []
        for row in p:
            top = int(row.argmax())
            fruit = next((f for f in ("apple", "banana", "orange") if f in self.labels[top]), "fruit")
            res.append({"p_rotten": float(row[rot].sum()), "top_label": self.labels[top],
                        "top_fruit": fruit, "top_prob": float(row[top])})
        return res


def grade(fruit, p_rotten):
    g = CFG["grading"]
    name = "Fresh" if p_rotten < g["fresh_below"] else "Rotten" if p_rotten > g["rotten_above"] else "Medium"
    sl = CFG["shelf_life"]
    k = sl["confidence_k"]
    days = int(round(sl["base_days"][name] * (1 - k + k * (1 - p_rotten))))
    if name == "Medium":
        days = max(days, 1)
    act = CFG["actions"][name]
    fruit_hi = CFG["fruit_names_hi"].get(fruit, fruit)
    advice = lambda a: a.split(" - ", 1)[-1]
    if name == "Rotten":
        en = f"Rotten {fruit}: don't sell or donate. Compost it."
        hi = f"{fruit_hi} सड़ा हुआ है: न बेचें, न दान करें। खाद बनाएँ।"
    else:
        en = f"{name} {fruit}: {advice(act['en'])}. Use within ~{days} day(s) (prototype estimate)."
        hi = f"{fruit_hi}: {advice(act['hi'])}। लगभग {days} दिन में इस्तेमाल करें (अनुमान)।"
    return {"grade": name, "confidence": round(max(p_rotten, 1 - p_rotten), 2), "action": act["en"],
            "action_color": act["color"], "use_within_days": days, "message_en": en, "message_hi": hi}


DET, CLS = Detector(CFG["detector"]), Classifier(CFG["classifier"])


def run_pipeline(img):
    t0 = time.perf_counter()
    dets = DET.detect(img)
    fallback = not dets
    crops = [img.crop((x, y, x + w, y + h)) for (x, y, w, h), _, _ in dets] or [img]
    fruits = []
    for i, c in enumerate(CLS.classify(crops)):
        if fallback and c["top_prob"] < CLS.min_conf:
            continue
        name = c["top_fruit"] if fallback else dets[i][1]
        box = [0, 0, img.width, img.height] if fallback else dets[i][0]
        g = grade(name, c["p_rotten"])
        fruits.append({"name": name, "box": box, "grade": g["grade"], "confidence": g["confidence"],
                       "p_rotten": round(c["p_rotten"], 2), "detector_score": 0.0 if fallback else round(dets[i][2], 2),
                       "classifier_label": c["top_label"], **{k: g[k] for k in
                       ("action", "action_color", "use_within_days", "message_en", "message_hi")}})
    return {"fruits": fruits, "fallback": fallback, "image_size": [img.width, img.height],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


def load_image(data: bytes):
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    if max(img.size) > 1280:
        img.thumbnail((1280, 1280))
    return img


# ---------------- session totals ----------------
LOCK = threading.Lock()


def empty_stats():
    return {"scans": 0, "fruits_logged": 0, "by_grade": {"Fresh": 0, "Medium": 0, "Rotten": 0},
            "by_fruit": {}, "kg": 0.0, "inr": 0.0, "lat": 0.0}


STATS = empty_stats()


def stats_json():
    s = STATS
    return {"scans": s["scans"], "fruits_logged": s["fruits_logged"], "by_grade": s["by_grade"],
            "by_fruit": s["by_fruit"], "kg_saved": round(s["kg"], 2), "inr_saved": round(s["inr"]),
            "avg_latency_ms": round(s["lat"] / s["scans"], 1) if s["scans"] else 0,
            "note": "Estimates from config.json values (avg weight, price per kg)."}


def record(result):
    sv = CFG["savings"]
    with LOCK:
        for f in result["fruits"]:
            STATS["fruits_logged"] += 1
            STATS["by_grade"][f["grade"]] += 1
            STATS["by_fruit"][f["name"]] = STATS["by_fruit"].get(f["name"], 0) + 1
            if f["grade"] in sv["saved_grades"]:
                kg = sv["avg_weight_kg"].get(f["name"], 0.15)
                STATS["kg"] += kg
                STATS["inr"] += kg * sv["price_inr_per_kg"].get(f["name"], 80)


# ---------------- web app ----------------
def make_app():
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="FreshCheck (Python fallback)")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    web = BASE / CFG.get("web_dir", "web")

    @app.get("/")
    def index():
        return FileResponse(web / "index.html")

    app.mount("/static", StaticFiles(directory=web), name="static")

    @app.get("/health")
    def health():
        return {"status": "ok", "engine": "python"}

    @app.get("/api/stats")
    def stats():
        with LOCK:
            return stats_json()

    @app.post("/api/reset")
    def reset():
        global STATS
        with LOCK:
            STATS = empty_stats()
            return stats_json()

    @app.post("/api/grade")
    async def grade_ep(request: Request):
        rec = request.query_params.get("record") in ("1", "true", "yes")
        ctype = request.headers.get("content-type", "")
        try:
            if "multipart" in ctype:
                form = await request.form()
                data = await form["image"].read()
                rec = rec or str(form.get("record", "")) in ("1", "true", "yes")
            elif "json" in ctype:
                body = await request.json()
                b64 = body.get("image", "")
                data = base64.b64decode(b64.split(",", 1)[1] if b64.startswith("data:") else b64)
                rec = rec or body.get("record") is True
            else:
                data = await request.body()
            img = load_image(data)
        except Exception as e:
            return JSONResponse({"error": f"could not decode image: {e}"}, status_code=400)
        out = run_pipeline(img)
        out["recorded"] = rec
        with LOCK:
            STATS["scans"] += 1
            STATS["lat"] += out["latency_ms"]
        if rec:
            record(out)
        log.info("grade %dx%d fruits=%d fallback=%d %.1fms%s", img.width, img.height, len(out["fruits"]),
                 out["fallback"], out["latency_ms"], " [logged]" if rec else "")
        return out

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", CFG.get("port", 7860))))
    a = ap.parse_args()
    if a.image:
        print(json.dumps(run_pipeline(load_image(open(a.image, "rb").read())), indent=2, ensure_ascii=False))
    else:
        import uvicorn
        uvicorn.run(make_app(), host="0.0.0.0", port=a.port)
