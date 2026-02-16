import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os

# Explicitly import the solution to avoid AttributeError on Streamlit Cloud
from mediapipe.python.solutions import selfie_segmentation as mp_selfie_segmentation

# --- UI SETUP ---
st.set_page_config(page_title="VantageBG", page_icon="🎬")
st.title("🎬 VantageBG: AI Background Remover")
st.write("Professional-grade green screen generation.")

uploaded_file = st.file_uploader("Upload a video (MP4/MOV)", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") 
    tfile.write(uploaded_file.read())
    tfile_path = tfile.name
    tfile.close() 
    
    output_path = os.path.join(tempfile.gettempdir(), "vantage_output.mp4")
    
    # Use the explicitly imported module
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmentor:
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

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = segmentor.process(frame_rgb)
            
            # Ensure we have a valid segmentation mask
            if results.segmentation_mask is not None:
                mask = results.segmentation_mask > 0.5
                green_bg = np.zeros(frame.shape, dtype=np.uint8)
                green_bg[:] = (0, 255, 0)
                condition = np.stack((mask,) * 3, axis=-1)
                output_frame = np.where(condition, frame, green_bg)
                out.write(output_frame)
            
            frame_count += 1
            if total_frames > 0:
                bar.progress(frame_count / total_frames)
            status.text(f"Processing: {frame_count}/{total_frames} frames")

        cap.release()
        out.release()

    if os.path.exists(output_path):
        with open(output_path, "rb") as file:
            st.download_button(
                label="⬇️ Download VantageBG Result",
                data=file,
                file_name="vantage_greenscreen.mp4",
                mime="video/mp4"
            )
        st.success("Video processed successfully!")
    
    os.remove(tfile_path)
