# stream_lit/model_utils.py
from __future__ import annotations

import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ---------- Device & constants ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAD_MULTIPLE = 32

# ---------- Paths (relative to this file) ----------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(THIS_DIR, "weights")
UD_MODEL_PATH = os.path.join(WEIGHTS_DIR, "UDnet.pth")
YOLO_MODEL_PATH = os.path.join(WEIGHTS_DIR, "YOLO.pt")


def get_project_root():
    """Repository root (parent of stream_lit/)."""
    return os.path.abspath(os.path.join(THIS_DIR, ".."))


def list_yolo_weights():
    """Return sorted basenames of *.pt files in stream_lit/weights (YOLO checkpoints)."""
    if not os.path.isdir(WEIGHTS_DIR):
        return []
    names = [
        f
        for f in os.listdir(WEIGHTS_DIR)
        if f.lower().endswith(".pt") and os.path.isfile(os.path.join(WEIGHTS_DIR, f))
    ]
    return sorted(names)


def yolo_weights_path(basename: str) -> str:
    """Resolve a weight filename inside weights dir (no path traversal)."""
    base = os.path.basename(basename)
    full = os.path.join(WEIGHTS_DIR, base)
    if not os.path.isfile(full):
        raise FileNotFoundError(f"YOLO weights not found: {base}")
    return full


# ---------- Load Models ----------
def load_ud_model():
    """
    Load UDNet enhancement model.
    Expects your UDNet code at: imageEnhancement/model_utils/UDnet.py
    with classes: mynet(opt), Opt(device)
    """
    from imageEnhancement.model_utils.UDnet import mynet, Opt

    device = torch.device(DEVICE)
    opt = Opt(device)
    model = mynet(opt)

    # Load weights
    state = torch.load(
        UD_MODEL_PATH, map_location=device, weights_only=False
    )
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def load_yolo_model(weight_basename: str | None = None):
    """
    Load YOLO from stream_lit/weights.
    If weight_basename is None, uses YOLO.pt when present, else first available .pt.
    """
    available = list_yolo_weights()
    if not available:
        raise FileNotFoundError(
            f"No .pt weights found in {WEIGHTS_DIR}. Add YOLO checkpoints (e.g. YOLO.pt)."
        )
    name = weight_basename or ("YOLO.pt" if "YOLO.pt" in available else available[0])
    if name not in available:
        name = available[0]
    return YOLO(yolo_weights_path(name))


# ---------- Enhancement Helpers ----------
def preprocess_np(img_np, multiple=32):
    """
    Pad image so H and W are multiples of `multiple`,
    convert to tensor [1,3,H,W] in RGB, normalized [0,1].
    Input: BGR numpy (cv2).
    """
    h, w = img_np.shape[:2]
    new_h = int(np.ceil(h / multiple) * multiple)
    new_w = int(np.ceil(w / multiple) * multiple)
    pad_bottom = new_h - h
    pad_right = new_w - w

    img_pad = cv2.copyMakeBorder(
        img_np, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT
    )

    rgb = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return tensor, (0, pad_bottom, 0, pad_right)


def tensor_to_bgr_image(tensor):
    """Convert model tensor output [1,3,H,W] or [3,H,W] in [0,1] RGB to BGR uint8."""
    tensor = tensor.detach().cpu()
    arr = tensor.squeeze(0).clamp(0, 1).numpy().transpose(1, 2, 0)  # H,W,3 RGB
    bgr = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return bgr


def crop_to_original(img_np, pads):
    """Remove padding to get back to original size."""
    top, bottom, left, right = pads
    h, w = img_np.shape[:2]
    y0, y1 = top, h - bottom if bottom != 0 else h
    x0, x1 = left, w - right if right != 0 else w
    return img_np[y0:y1, x0:x1]


# ---------- Full Enhancement Step ----------
def enhance_image(model, img_cv):
    """
    Run UDNet enhancement on a BGR image (cv2).
    Returns enhanced image in BGR.
    """
    tensor, pads = preprocess_np(img_cv, PAD_MULTIPLE)
    tensor = tensor.to(torch.device(DEVICE))

    with torch.no_grad():
        # Try different forward signatures, depending on your UDNet implementation
        try:
            model.forward(tensor, tensor, training=False)
        except Exception:
            try:
                model.forward(tensor, training=False)
            except Exception:
                pass

        out = model.sample(testing=True)
        out_tensor = out[0] if isinstance(out, (list, tuple)) else out
        enhanced_bgr = crop_to_original(tensor_to_bgr_image(out_tensor), pads)

    return enhanced_bgr


# ---------- YOLO Detection ----------
def _class_label(names, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(int(cls_id), f"class_{int(cls_id)}"))
    if isinstance(names, (list, tuple)) and 0 <= int(cls_id) < len(names):
        return str(names[int(cls_id)])
    return f"class_{int(cls_id)}"


def summarize_detection_results(result) -> dict:
    """
    Build stats from a single ultralytics Results object.
    Returns: counts (label -> n), total, mean_confidence in [0,1], per-class max conf.
    """
    boxes = result.boxes
    names = getattr(result, "names", None) or {}
    if boxes is None or len(boxes) == 0:
        return {
            "counts": {},
            "total": 0,
            "mean_confidence": 0.0,
            "max_confidence": 0.0,
        }
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    counts: dict[str, int] = {}
    for cls_id in np.unique(clss):
        n = int(np.sum(clss == cls_id))
        label = _class_label(names, int(cls_id))
        counts[label] = n
    return {
        "counts": counts,
        "total": int(len(clss)),
        "mean_confidence": float(np.mean(confs)) if len(confs) else 0.0,
        "max_confidence": float(np.max(confs)) if len(confs) else 0.0,
    }


def detect_image_full(yolo_model, bgr_image, conf: float = 0.4):
    """
    Run YOLO once; return (annotated_bgr, stats_dict).
    Annotated image is suitable for OpenCV / BGR display pipeline.
    """
    results = yolo_model.predict(
        source=bgr_image, conf=float(conf), save=False, verbose=False
    )
    r0 = results[0]
    annotated = r0.plot()
    stats = summarize_detection_results(r0)
    return annotated, stats


def run_detection(yolo_model, bgr_image, conf: float = 0.4):
    """
    Run YOLO detection on a BGR image and return an annotated image (BGR).
    """
    annotated, _ = detect_image_full(yolo_model, bgr_image, conf=conf)
    return annotated
