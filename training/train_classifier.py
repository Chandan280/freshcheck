"""
FreshCheck - train the fresh/rotten classifier and export it to ONNX.

Dataset: Kaggle "Fruits fresh and rotten for classification" (sriramr),
6 classes: freshapples, freshbanana, freshoranges, rottenapples, rottenbanana, rottenoranges.

Steps:
  1. collect all images (train+test folders of the Kaggle zip are merged)
  2. dedupe with perceptual hash (imagehash.phash)
  3. stratified train/val/test split (70/15/15) with a fixed seed
  4. fine-tune ResNet-18 (ImageNet weights), 224x224, light augmentation
  5. test set: macro F1 + confusion matrix (+ plots) + safety metric
  6. export ONNX (opset 17) and check ONNX vs PyTorch outputs match

Usage (Colab):  python train_classifier.py --data /path/to/dataset --out out --epochs 5
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CLASSES = ["freshapples", "freshbanana", "freshoranges",
           "rottenapples", "rottenbanana", "rottenoranges"]  # alphabetical = label index
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# Grade thresholds used by the app (config.json) - for the safety metric.
FRESH_BELOW = 0.35


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_images(root):
    """Finds every image inside folders named like one of CLASSES (any depth)."""
    items = []
    for dirpath, _, files in os.walk(root):
        name = Path(dirpath).name.lower()
        if name not in CLASSES:
            continue
        for f in sorted(files):
            if Path(f).suffix.lower() in IMG_EXT:
                items.append((os.path.join(dirpath, f), CLASSES.index(name)))
    return sorted(items)


def dedupe(items):
    """Drops images whose perceptual hash was already seen (exact + near-exact copies)."""
    import imagehash
    seen, kept, dropped = {}, [], 0
    for path, label in items:
        try:
            with Image.open(path) as im:
                h = str(imagehash.phash(im.convert("RGB")))
        except Exception:
            dropped += 1  # unreadable file
            continue
        if h in seen:
            dropped += 1
            continue
        seen[h] = path
        kept.append((path, label))
    print(f"dedupe: kept {len(kept)}, dropped {dropped} duplicate/unreadable")
    return kept


class ImageList(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        with Image.open(path) as im:
            return self.tf(im.convert("RGB")), label


def build_model(pretrained):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    return m


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), 1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def save_plots(history, cm, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"], label="train loss")
    ax[0].plot(history["val_loss"], label="val loss")
    ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].set_title("Loss")
    ax[1].plot(history["val_f1"], marker="o")
    ax[1].set_xlabel("epoch"); ax[1].set_title("Val macro F1")
    fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Test confusion matrix")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=120); plt.close(fig)


def export_onnx(model, out_path, device):
    """Exports with a dynamic batch axis and checks ONNX Runtime matches PyTorch."""
    import onnxruntime as ort
    model.eval().cpu()
    dummy = torch.randn(2, 3, 224, 224)
    torch.onnx.export(model, dummy, str(out_path), opset_version=17,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      dynamo=False)
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        ref = model(x).numpy()
    got = sess.run(None, {"input": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    print(f"ONNX check: max |pytorch - onnx| = {diff:.2e}  ->  {'OK' if diff < 1e-3 else 'MISMATCH'}")
    model.to(device)
    return diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder containing the Kaggle class folders")
    ap.add_argument("--out", default="out")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-pretrained", action="store_true", help="random init (smoke tests only)")
    ap.add_argument("--no-dedupe", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # 1-2) collect + dedupe
    items = collect_images(args.data)
    print(f"found {len(items)} images")
    if not items:
        raise SystemExit("No images found. --data must contain folders named like " + ", ".join(CLASSES))
    if not args.no_dedupe:
        items = dedupe(items)

    # 3) stratified 70/15/15 split
    labels = [l for _, l in items]
    train, rest = train_test_split(items, test_size=0.30, stratify=labels, random_state=args.seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=[l for _, l in rest], random_state=args.seed)
    counts = {c: [sum(1 for _, l in s if l == i) for s in (train, val, test)] for i, c in enumerate(CLASSES)}
    print("split sizes train/val/test:", len(train), len(val), len(test))
    for c, n in counts.items():
        print(f"  {c:14s} {n}")
    json.dump({"train": train, "val": val, "test": test}, open(out / "split.json", "w"))

    # 4) data + model
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # no hue: colour = rot cue
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    eval_tf = transforms.Compose([  # must match the C++ app: plain resize to 224x224
        transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    dl = lambda s, tf, sh: DataLoader(ImageList(s, tf), batch_size=args.batch, shuffle=sh,
                                      num_workers=args.workers, pin_memory=(device == "cuda"))
    train_dl, val_dl, test_dl = dl(train, train_tf, True), dl(val, eval_tf, False), dl(test, eval_tf, False)

    model = build_model(not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1, best_path = -1.0, out / "resnet18_fresh.pt"
    for epoch in range(args.epochs):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y); n += len(y)
        sched.step()
        p, yv = predict(model, val_dl, device)
        val_loss = float(nn.functional.nll_loss(torch.log(torch.tensor(p) + 1e-9), torch.tensor(yv)))
        val_f1 = f1_score(yv, p.argmax(1), average="macro")
        history["train_loss"].append(total / n); history["val_loss"].append(val_loss); history["val_f1"].append(val_f1)
        print(f"epoch {epoch + 1}/{args.epochs}  train_loss {total / n:.4f}  val_loss {val_loss:.4f}  "
              f"val_macroF1 {val_f1:.4f}  ({time.time() - t0:.0f}s)")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)

    # 5) test set with the best checkpoint
    model.load_state_dict(torch.load(best_path, map_location=device))
    p, yt = predict(model, test_dl, device)
    pred = p.argmax(1)
    macro_f1 = f1_score(yt, pred, average="macro")
    cm = confusion_matrix(yt, pred, labels=range(len(CLASSES)))
    print("\nTEST macro F1:", round(macro_f1, 4))
    print(classification_report(yt, pred, target_names=CLASSES, digits=4))
    print("confusion matrix (rows=true, cols=pred):\n", cm)

    # Safety metric: truly rotten fruit that the app would show GREEN (p_rotten < FRESH_BELOW).
    rotten_idx = [i for i, c in enumerate(CLASSES) if c.startswith("rotten")]
    p_rotten = p[:, rotten_idx].sum(1)
    is_rotten = np.isin(yt, rotten_idx)
    green_rotten = float((p_rotten[is_rotten] < FRESH_BELOW).mean()) if is_rotten.any() else float("nan")
    print(f"SAFETY: rotten shown GREEN = {green_rotten * 100:.2f}% of {int(is_rotten.sum())} rotten test images")

    save_plots(history, cm, out)

    # 6) ONNX export + check
    diff = export_onnx(model, out / "resnet18_fresh.onnx", device)

    metrics = {"test_macro_f1": macro_f1, "test_accuracy": float((pred == yt).mean()),
               "rotten_shown_green_rate": green_rotten, "n_train": len(train), "n_val": len(val),
               "n_test": len(test), "epochs": args.epochs, "seed": args.seed, "best_val_macro_f1": best_f1,
               "onnx_max_abs_diff": diff, "confusion_matrix": cm.tolist(), "labels": CLASSES,
               "pretrained": not args.no_pretrained}
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    json.dump(CLASSES, open(out / "labels.json", "w"))
    print("saved:", sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
