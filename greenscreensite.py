import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import tempfile
import os
import requests

# --- PAGE CONFIG ---
st.set_page_config(page_title="VantageBG", page_icon="🎬")
st.title("🎬 VantageBG: AI Background Remover")

uploaded_file = st.file_uploader("Upload a video (MP4/MOV)", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") 
    tfile.write(uploaded_file.read())
    tfile_path = tfile.name
    tfile.close() 
    
    output_path = os.path.join(tempfile.gettempdir(), "vantage_output.mp4")
    
    # --- MODEL DOWNLOAD & SETUP ---
    model_url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
    model_local_path = os.path.join(tempfile.gettempdir(), "selfie_segmenter.tflite")

    if not os.path.exists(model_local_path):
        with st.spinner("Initializing AI Engine..."):
            r = requests.get(model_url)
            with open(model_local_path, "wb") as f:
                f.write(r.content)

    base_options = mp_python.BaseOptions(model_asset_path=model_local_path)
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        output_category_mask=True
    )

    # --- PROCESSING ---
    with vision.ImageSegmenter.create_from_options(options) as segmentor:
        cap = cv2.VideoCapture(tfile_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        bar = st.progress(0)
        status = st.empty()

        frame_count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            segmentation_result = segmentor.segment(mp_image)
            category_mask = segmentation_result.category_mask.numpy_view()
            
            # --- LOGIC FLIP FIX ---
            # We want 'mask' to be True for the person (so they are NOT green)
            # In this model, high values = person, low values = background
            mask = category_mask.squeeze() > 0.1 
            
            green_bg = np.zeros(frame.shape, dtype=np.uint8)
            green_bg[:] = (0, 255, 0)
            
            # np.where(condition, if_true, if_false)
            # If mask is True (Person), keep original frame. Else (Background), use green_bg.
            condition = np.stack((mask,) * 3, axis=-1)
            output_frame = np.where(condition, frame, green_bg)
            
            out.write(output_frame.astype(np.uint8))
            
            frame_count += 1
            if total_frames > 0:
                bar.progress(min(frame_count / total_frames, 1.0))
            status.text(f"Processing: {frame_count}/{total_frames} frames")

        cap.release()
        out.release()

    if os.path.exists(output_path):
        st.divider()
        with open(output_path, "rb") as file:
            st.download_button(label="⬇️ Download VantageBG Result", data=file, file_name="vantage_output.mp4", mime="video/mp4")
        st.success("✅ Success! The background has been removed.")
    
    try:
        os.remove(tfile_path)
    except:
        pass
