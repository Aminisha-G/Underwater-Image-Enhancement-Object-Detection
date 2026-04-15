import streamlit as st
import cv2
import os
import sys
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
import io



# ---------- Config ----------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

PAD_MULTIPLE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UD_MODEL_PATH = "stream_lit/weights/UDnet.pth"
YOLO_MODEL_PATH = "stream_lit/weights/YOLO.pt"

# ---------- Load models ----------
@st.cache_resource(show_spinner=False)
def load_ud_model():
    from imageEnhancement.model_utils.UDnet import mynet, Opt
    device = torch.device(DEVICE)
    opt = Opt(device)
    model = mynet(opt)
    state = torch.load(
        UD_MODEL_PATH, map_location=device, weights_only=False
    )
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_yolo_model():
    return YOLO(YOLO_MODEL_PATH)

ud_model = load_ud_model()
yolo_model = load_yolo_model()

# ---------- Helpers ----------
def preprocess_np(img_np, multiple=32):
    h, w = img_np.shape[:2]
    new_h = int(np.ceil(h/multiple)*multiple)
    new_w = int(np.ceil(w/multiple)*multiple)
    pad_bottom = new_h - h
    pad_right = new_w - w
    img_pad = cv2.copyMakeBorder(img_np, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
    tensor = torch.from_numpy(cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)
    return tensor, (0, pad_bottom, 0, pad_right)

def tensor_to_bgr_image(tensor):
    tensor = tensor.detach().cpu()
    arr = tensor.squeeze(0).clamp(0,1).numpy().transpose(1,2,0)
    return cv2.cvtColor((arr*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

def crop_to_original(img_np, pads):
    top, bottom, left, right = pads
    h, w = img_np.shape[:2]
    y0, y1 = top, h-bottom if bottom !=0 else h
    x0, x1 = left, w-right if right !=0 else w
    return img_np[y0:y1, x0:x1]

# ---------- Streamlit UI ----------
st.title("UDnet Enhancement + Object Detection")

uploaded = st.file_uploader("Upload image", type=["png","jpg","jpeg"])
enhance_btn = st.button("Enhance Image")

# Reset state if new image uploaded
if uploaded is not None and ('uploaded_name' not in st.session_state or st.session_state['uploaded_name'] != uploaded.name):
    st.session_state['enhanced_image'] = None
    st.session_state['detected_image'] = None
    st.session_state['uploaded_name'] = uploaded.name

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_cv is None:
        st.error("Could not read uploaded image.")
    else:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), use_container_width=True)

        # ---------- Step 1: Enhancement ----------
        if enhance_btn and st.session_state.get('enhanced_image') is None:
            inp_tensor, pads = preprocess_np(img_cv, PAD_MULTIPLE)
            inp_tensor = inp_tensor.to(torch.device(DEVICE))
            with torch.no_grad():
                try:
                    ud_model.forward(inp_tensor, inp_tensor, training=False)
                except TypeError:
                    ud_model.forward(inp_tensor, training=False)
                out = ud_model.sample(testing=True)
                out_tensor = out[0] if isinstance(out, (list,tuple)) else out
                enhanced_bgr = crop_to_original(tensor_to_bgr_image(out_tensor), pads)
            st.session_state['enhanced_image'] = enhanced_bgr

        # ---------- Show enhanced image ----------
        if st.session_state.get('enhanced_image') is not None:
            st.subheader("Enhanced Image")
            st.image(cv2.cvtColor(st.session_state['enhanced_image'], cv2.COLOR_BGR2RGB), use_container_width=True)

            # Download button for enhanced image
            _, buffer = cv2.imencode(".png", st.session_state['enhanced_image'])
            st.download_button("Download Enhanced Image", data=io.BytesIO(buffer.tobytes()),
                               file_name="enhanced.png", mime="image/png")

            # ---------- Step 2: Object Detection ----------
            if st.button("Run Object Detection"):
                if st.session_state.get('detected_image') is None:
                    enhanced_bgr = st.session_state['enhanced_image']
                    # YOLO supports np array input
                    results = yolo_model.predict(source=enhanced_bgr, conf=0.4, save=False)
                    detected_img = results[0].plot()
                    st.session_state['detected_image'] = detected_img

        # ---------- Show detected image ----------
        if st.session_state.get('detected_image') is not None:
            st.subheader("Object Detection Result")
            st.image(cv2.cvtColor(st.session_state['detected_image'], cv2.COLOR_BGR2RGB), use_container_width=True)
            _, buffer = cv2.imencode(".png", st.session_state['detected_image'])
            st.download_button("Download Detected Image", data=io.BytesIO(buffer.tobytes()),
                               file_name="detected.png", mime="image/png")