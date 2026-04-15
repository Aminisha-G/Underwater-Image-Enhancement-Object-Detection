import os
import sys
import math
import cv2
import torch
import numpy as np
from torch.autograd import Variable

# Project root must be on sys.path before package imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from imageEnhancement.model_utils.UDnet import mynet


# ----------------- Settings -----------------

INPUT_IMAGE = (
    "underwaterObjectDetection/aquarium_pretrain/valid/images/"
    "IMG_2277_jpeg_jpg.rf.86c72d6192da48d941ffa957f4780665.jpg"
)
MODEL_PATH = 'imageEnhancement/model/UDnet.pth'
OUTPUT_DIR = './output/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PAD_MULTIPLE = 32

# ----------------- Helper Classes & Functions -----------------
class Opt:
    """Simple object to mimic argparse.Namespace for mynet."""
    def __init__(self, device):
        self.device = device


def pad_to_multiple(img, multiple=32):
    h, w = img.shape[:2]
    new_h = int(math.ceil(h / multiple) * multiple)
    new_w = int(math.ceil(w / multiple) * multiple)

    pad_bottom = new_h - h
    pad_right = new_w - w
    pad_top = 0
    pad_left = 0

    if pad_bottom != 0 or pad_right != 0:
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)

    return img, (pad_top, pad_bottom, pad_left, pad_right)


def preprocess(img_path, multiple=32):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, pads = pad_to_multiple(img, multiple=multiple)
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

    return tensor, pads


def crop_to_original(img_np, pads):
    top, bottom, left, right = pads
    h, w = img_np.shape[:2]
    y0 = top
    y1 = h - bottom if bottom > 0 else h
    x0 = left
    x1 = w - right if right > 0 else w

    return img_np[y0:y1, x0:x1]


def save_output(tensor, output_path, pads=(0, 0, 0, 0)):
    if isinstance(tensor, (tuple, list)):
        tensor = tensor[0]

    if tensor.dim() == 4:
        img = tensor.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    elif tensor.dim() == 3:
        img = tensor.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

    img = (img * 255).astype(np.uint8)
    img = crop_to_original(img, pads)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Saved output to {output_path}")


# ----------------- Main Inference -----------------
def main():
    print("Loading model...")
    device = torch.device(DEVICE)
    opt = Opt(device)       # FIXED Opt class
    model = mynet(opt)

    state = torch.load(
        MODEL_PATH, map_location=device, weights_only=False
    )
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    print("Model loaded.")

    print("Processing image...")
    input_tensor, pads = preprocess(INPUT_IMAGE, multiple=PAD_MULTIPLE)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        try:
            model.forward(input_tensor, input_tensor, training=False)
        except:
            try:
                model.forward(input_tensor, training=False)
            except:
                pass

        output = model.sample(testing=True)

    if isinstance(output, (list, tuple)) and len(output) > 0:
        output_tensor = output[0]
    else:
        output_tensor = output

    print("Output tensor shape:", output_tensor.shape)

    out_path = os.path.join(OUTPUT_DIR, os.path.basename(INPUT_IMAGE))
    save_output(output_tensor, out_path, pads=pads)


# ----------------- FIXED MAIN GUARD -----------------
if __name__ == "__main__":
    main()