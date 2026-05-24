"""
Facial Expression Detector — CNN Version
==========================================
Detects faces in real-time and classifies expressions using a trained CNN.

Architecture:
  - Face detection:     OpenCV Haar Cascade
  - Classifier:         3-block CNN (Conv → BN → ReLU → Pool) trained on FER-2013
  - Augmentation:       Random flip, rotation, brightness jitter during training
  - Device:             Auto-selects Apple Metal (MPS) → CUDA → CPU
  - Output:             Live webcam feed with expression label + confidence bars

Expressions: happy, sad, angry, surprise, fear, disgust, neutral

Usage:
  python expression_detector.py --train        # train on fer2013/ then open webcam
  python expression_detector.py --train --force  # force retrain
  python expression_detector.py                # webcam (loads saved model)
  python expression_detector.py --image face.jpg

Requirements:
  pip install torch torchvision opencv-python scikit-learn
"""

import cv2
import numpy as np
import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MODEL_PATH  = "expression_cnn.pt"
IMG_SIZE    = (48, 48)
BATCH_SIZE  = 64
EPOCHS      = 50

LABEL_ALIASES = {
    "angry":    "angry",
    "disgust":  "disgust",
    "fear":     "fear",
    "happy":    "happy",
    "neutral":  "neutral",
    "sad":      "sad",
    "surprise": "surprise",
}

EXPRESSIONS = ["neutral", "happy", "sad", "angry", "surprise", "fear", "disgust"]

EXPR_COLORS = {
    "happy":    (0,   220, 255),
    "sad":      (200, 100,  50),
    "angry":    (30,   30, 220),
    "surprise": (0,   180, 255),
    "fear":     (180,  60, 200),
    "disgust":  (40,  160,  40),
    "neutral":  (180, 180, 180),
}


# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        print("[DEVICE] Apple Metal (MPS) — GPU acceleration enabled")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("[DEVICE] CUDA GPU detected")
        return torch.device("cuda")
    else:
        print("[DEVICE] CPU only — training will be slow (~2–3 hrs for 50 epochs)")
        return torch.device("cpu")


# ──────────────────────────────────────────────
# CNN Model
# ──────────────────────────────────────────────
class ExpressionCNN(nn.Module):
    """
    3-block convolutional network for 48×48 grayscale face images.

    Block structure: Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU
                     → MaxPool2d → Dropout2d

    Spatial progression: 48 → 24 → 12 → 6
    Final feature map:   128 × 6 × 6 = 4608 → FC 512 → FC num_classes
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1  (48 → 24)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Block 2  (24 → 12)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Block 3  (12 → 6)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class FERDataset(Dataset):
    """
    Loads FER-2013 images on-demand from disk.
    augment=True applies random flips / rotation / brightness jitter
    to regularise training and improve real-world generalisation.
    """
    def __init__(self, image_paths: list, labels: list,
                 label_to_idx: dict, augment: bool = False):
        self.paths        = image_paths
        self.labels       = labels
        self.label_to_idx = label_to_idx

        base = [
            transforms.ToPILImage(),
            transforms.Resize(IMG_SIZE),
            transforms.Grayscale(),
        ]
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ] if augment else []

        self.transform = transforms.Compose(base + aug + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros(IMG_SIZE, dtype=np.uint8)
        tensor = self.transform(img)
        label  = self.label_to_idx[self.labels[idx]]
        return tensor, label


# ──────────────────────────────────────────────
# Data Collection
# ──────────────────────────────────────────────
def collect_paths(data_dir: str) -> tuple[list, list, list, dict, dict]:
    """
    Walk fer2013/train/ and fer2013/test/, return all image paths + labels.
    Returns (all_paths, all_labels, classes, label_to_idx, idx_to_label)
    """
    all_paths, all_labels = [], []

    for split in ["train", "test"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            print(f"[WARN] Not found, skipping: {split_dir}")
            continue

        for emotion_folder in sorted(os.listdir(split_dir)):
            emotion_dir = os.path.join(split_dir, emotion_folder)
            if not os.path.isdir(emotion_dir):
                continue

            label = LABEL_ALIASES.get(emotion_folder.lower())
            if label is None:
                print(f"[WARN] Unknown folder '{emotion_folder}', skipping.")
                continue

            files = [f for f in os.listdir(emotion_dir)
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))]

            for fname in files:
                all_paths.append(os.path.join(emotion_dir, fname))
                all_labels.append(label)

            print(f"  [{split}/{emotion_folder}]  {len(files)} images  →  '{label}'")

    classes      = sorted(set(all_labels))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}
    return all_paths, all_labels, classes, label_to_idx, idx_to_label


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def train_model(data_dir: str, force: bool = False) -> tuple:
    """
    Train the CNN on FER-2013 and save to MODEL_PATH.
    Returns (model, idx_to_label, device).
    If a saved model exists and force=False, loads it instead.
    """
    device = get_device()

    # ── load existing model ──
    if not force and os.path.exists(MODEL_PATH):
        print(f"[INFO] Loading saved model from '{MODEL_PATH}'")
        checkpoint   = torch.load(MODEL_PATH, map_location=device)
        idx_to_label = checkpoint["idx_to_label"]
        model        = ExpressionCNN(num_classes=len(idx_to_label)).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model, idx_to_label, device

    # ── collect data ──
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"FER-2013 folder not found: '{data_dir}'\n"
            f"Make sure the fer2013/ folder is in the same directory as this script."
        )

    print(f"[TRAIN] Scanning FER-2013 at '{data_dir}' …")
    all_paths, all_labels, classes, label_to_idx, idx_to_label = collect_paths(data_dir)
    print(f"\n[TRAIN] Total images: {len(all_paths)}  |  Classes: {classes}")

    tr_paths, va_paths, tr_labels, va_labels = train_test_split(
        all_paths, all_labels, test_size=0.15, stratify=all_labels, random_state=42
    )
    print(f"[TRAIN] Train: {len(tr_paths)}  |  Val: {len(va_paths)}")

    train_ds = FERDataset(tr_paths, tr_labels, label_to_idx, augment=True)
    val_ds   = FERDataset(va_paths, va_labels, label_to_idx, augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

    model     = ExpressionCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    print(f"\n[TRAIN] Starting training for {EPOCHS} epochs …\n")
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── train ──
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for imgs, lbls in train_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * imgs.size(0)
            correct  += (out.argmax(1) == lbls).sum().item()
            total    += imgs.size(0)
        train_acc  = correct / total
        train_loss = run_loss / total

        # ── validate ──
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_dl:
                imgs, lbls = imgs.to(device), lbls.to(device)
                val_correct += (model(imgs).argmax(1) == lbls).sum().item()
                val_total   += imgs.size(0)
        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        saved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state":  model.state_dict(),
                "idx_to_label": idx_to_label,
                "classes":      classes,
            }, MODEL_PATH)
            saved = "  ✓ saved"

        print(f"Epoch {epoch:>3}/{EPOCHS}  "
              f"loss: {train_loss:.4f}  "
              f"train: {train_acc*100:.1f}%  "
              f"val: {val_acc*100:.1f}%"
              f"{saved}")

    print(f"\n[TRAIN] Best val accuracy: {best_val_acc*100:.1f}%")
    print(f"[TRAIN] Model saved → '{MODEL_PATH}'")

    # reload best checkpoint
    checkpoint   = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, idx_to_label, device


# ──────────────────────────────────────────────
# Inference transform (no augmentation)
# ──────────────────────────────────────────────
INFER_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def predict_expression(model, idx_to_label: dict, device,
                        face_roi: np.ndarray) -> tuple:
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) \
           if len(face_roi.shape) == 3 else face_roi

    tensor = INFER_TRANSFORM(gray).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    idx       = int(np.argmax(probs))
    label     = idx_to_label[idx]
    prob_dict = {idx_to_label[i]: float(p) for i, p in enumerate(probs)}
    return label, float(probs[idx]), prob_dict


# ──────────────────────────────────────────────
# Drawing
# ──────────────────────────────────────────────
def draw_expression_overlay(frame, x, y, w, h, label, conf, prob_dict):
    color = EXPR_COLORS.get(label, (200, 200, 200))
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"{label.upper()}  {conf*100:.0f}%"
    (tw, th), _ = cv2.getTextSize(text, font, 0.75, 2)
    pad = 6
    rx1, ry1 = x, y - th - pad*2
    rx2, ry2 = x + tw + pad*2, y
    if ry1 < 0:
        ry1, ry2 = y + h, y + h + th + pad*2
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, -1)
    cv2.putText(frame, text, (rx1+pad, ry2-pad), font, 0.75, (0,0,0), 2, cv2.LINE_AA)

    bar_x, bar_y            = frame.shape[1] - 220, 20
    bar_w_max, bar_h, gap   = 160, 18, 26
    overlay = frame.copy()
    cv2.rectangle(overlay, (bar_x-10, bar_y-10),
                  (frame.shape[1]-5, bar_y + gap*len(EXPRESSIONS)+5),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, (expr, prob) in enumerate(sorted(prob_dict.items(), key=lambda kv: -kv[1])):
        bx, by = bar_x, bar_y + i*gap
        c = EXPR_COLORS.get(expr, (150, 150, 150))
        cv2.rectangle(frame, (bx, by), (bx+bar_w_max, by+bar_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (bx, by), (bx+int(prob*bar_w_max), by+bar_h), c, -1)
        cv2.putText(frame, f"{expr[:7]:<7} {prob*100:4.1f}%",
                    (bx+bar_w_max+4, by+bar_h-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)


def draw_instructions(frame):
    fh = frame.shape[0]
    for i, line in enumerate(["Q / ESC  quit", "S        screenshot"]):
        cv2.putText(frame, line, (10, fh-20-i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────
# Face Detector
# ──────────────────────────────────────────────
def load_detector():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    det  = cv2.CascadeClassifier(path)
    if det.empty():
        raise RuntimeError(f"Could not load face cascade: {path}")
    return det


# ──────────────────────────────────────────────
# Webcam Loop
# ──────────────────────────────────────────────
def run_webcam(model, idx_to_label, device):
    detector = load_detector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Use --image <path> for a static image.")
        return

    print("[INFO] Webcam open — Q/ESC to quit, S to save screenshot.")
    model.eval()
    fps_timer, fps_count, fps_display = time.time(), 0, 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fps_count += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_count / (time.time() - fps_timer)
            fps_count   = 0
            fps_timer   = time.time()

        gray  = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 220), 2, cv2.LINE_AA)
        else:
            for (x, y, w, h) in faces:
                m   = int(0.15 * w)
                roi = frame[max(0, y-m):min(frame.shape[0], y+h+m),
                            max(0, x-m):min(frame.shape[1], x+w+m)]
                if roi.size == 0:
                    continue
                label, conf, prob_dict = predict_expression(
                    model, idx_to_label, device, roi)
                draw_expression_overlay(frame, x, y, w, h, label, conf, prob_dict)

        cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1, cv2.LINE_AA)
        draw_instructions(frame)
        cv2.imshow("Expression Detector — Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            fname = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[INFO] Screenshot saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# Single Image Mode
# ──────────────────────────────────────────────
def run_image(model, idx_to_label, device, path: str):
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] Could not read: {path}")
        return

    gray  = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    faces = load_detector().detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    if len(faces) == 0:
        print("[RESULT] No face detected.")
        cv2.putText(frame, "No face detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 220), 2, cv2.LINE_AA)
    else:
        model.eval()
        for (x, y, w, h) in faces:
            m   = int(0.15 * w)
            roi = frame[max(0, y-m):min(frame.shape[0], y+h+m),
                        max(0, x-m):min(frame.shape[1], x+w+m)]
            if roi.size == 0:
                continue
            label, conf, prob_dict = predict_expression(
                model, idx_to_label, device, roi)
            draw_expression_overlay(frame, x, y, w, h, label, conf, prob_dict)
            print(f"\n[RESULT] {label.upper()}  ({conf*100:.1f}% confidence)")
            for expr, p in sorted(prob_dict.items(), key=lambda kv: -kv[1]):
                print(f"  {'█'*int(p*30):<30} {expr:<10} {p*100:5.1f}%")

    out = "result_" + os.path.basename(path)
    cv2.imwrite(out, frame)
    print(f"[INFO] Annotated image saved: {out}")
    cv2.imshow("Result — any key to close", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────
def main():
    global EPOCHS
    script_dir       = os.path.dirname(os.path.abspath(__file__))
    default_fer_path = os.path.join(script_dir, "fer2013")

    parser = argparse.ArgumentParser(
        description="Facial Expression Detector — CNN trained on FER-2013."
    )
    parser.add_argument("--train",  action="store_true",
                        help="Train the CNN (auto-runs if no saved model found).")
    parser.add_argument("--force",  action="store_true",
                        help="Force retrain even if a saved model exists.")
    parser.add_argument("--data",   type=str, default=default_fer_path,
                        help=f"Path to fer2013 folder (default: {default_fer_path})")
    parser.add_argument("--image",  type=str, default=None,
                        help="Run on a single image file instead of webcam.")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of training epochs (default: {EPOCHS})")
    args = parser.parse_args()

    # Allow overriding epochs from CLI
    EPOCHS = args.epochs

    if args.train or args.force or not os.path.exists(MODEL_PATH):
        model, idx_to_label, device = train_model(
            data_dir=args.data, force=args.force
        )
    else:
        device = get_device()
        print(f"[INFO] Loading saved model from '{MODEL_PATH}'")
        checkpoint   = torch.load(MODEL_PATH, map_location=device)
        idx_to_label = checkpoint["idx_to_label"]
        model        = ExpressionCNN(num_classes=len(idx_to_label)).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

    if args.image:
        run_image(model, idx_to_label, device, args.image)
    else:
        run_webcam(model, idx_to_label, device)


if __name__ == "__main__":
    main()
