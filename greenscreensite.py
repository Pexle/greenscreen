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

# 1. Initialize Session States
if 'processed_path' not in st.session_state:
    st.session_state.processed_path = None

def reset_app():
    # Cleanup file if it exists
    if st.session_state.processed_path and os.path.exists(st.session_state.processed_path):
        try:
            os.remove(st.session_state.processed_path)
        except:
            pass
    st.session_state.processed_path = None
    st.rerun()

# 2. File Uploader (Hidden if a video is already processed)
if st.session_state.processed_path is None:
    uploaded_file = st.file_uploader("Upload a video (MP4/MOV)", type=["mp4", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") 
        tfile.write(uploaded_file.read())
        tfile_path = tfile.name
        tfile.close() 
        
        output_path = os.path.join(tempfile.gettempdir(), "vantage_final.mp4")
        
        # --- AI ENGINE SETUP ---
        model_url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
        model_local_path = os.path.join(tempfile.gettempdir(), "selfie_segmenter.tflite")

        if not os.path.exists(model_local_path):
            with st.spinner("Initializing AI Engine..."):
                r = requests.get(model_url)
                with open(model_local_path, "wb") as f:
                    f.write(r.content)

        base_options = mp_python.BaseOptions(model_asset_path=model_local_path)
        options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)

        with vision.ImageSegmenter.create_from_options(options) as segmentor:
            cap = cv2.VideoCapture(tfile_path)
            width, height = int(cap.get(3)), int(cap.get(4))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            bar = st.progress(0)
            status = st.empty()
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            while cap.isOpened():
                success, frame = cap.read()
                if not success: break

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                segmentation_result = segmentor.segment(mp_image)
                category_mask = segmentation_result.category_mask.numpy_view()
                
                # Logic for keeping figure (invert if necessary)
                mask = ~(category_mask.squeeze() > 0.2) 
                
                green_bg = np.zeros(frame.shape, dtype=np.uint8)
                green_bg[:] = (0, 255, 0)
                
                condition = np.stack((mask,) * 3, axis=-1)
                output_frame = np.where(condition, frame, green_bg)
                out.write(output_frame.astype(np.uint8))
                
                frame_count += 1
                bar.progress(min(frame_count / total_frames, 1.0))
                status.text(f"AI Processing: {frame_count}/{total_frames} frames")

            cap.release()
            out.release()
            
            # Save to state and cleanup input
            st.session_state.processed_path = output_path
            os.remove(tfile_path)
            st.rerun() # Refresh to show the download UI

# 3. Download UI (Only shows when processing is done)
else:
    st.success("✅ Your video is ready!")
    
    with open(st.session_state.processed_path, "rb") as file:
        st.download_button(
            label="⬇️ Download Result", 
            data=file, 
            file_name="vantage_greenscreen.mp4", 
            mime="video/mp4"
        )
    
    st.divider()
    if st.button("🗑️ Start New Video"):
        reset_app()
