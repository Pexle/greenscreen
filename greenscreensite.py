import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
import tempfile
import os

# --- PAGE CONFIG (GENERIC) ---
st.set_page_config(page_title="AI Background Remover", page_icon="🎬")
st.title("🎬 Professional Video Background Remover")
st.write("Upload your video to automatically replace the background with a high-quality green screen.")

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload video file", type=["mp4", "mov", "avi", "mpeg4"])

if uploaded_file is not None:
    # 1. Save uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") 
    tfile.write(uploaded_file.read())
    tfile_path = tfile.name
    tfile.close() 
    
    # 2. Setup output path
    output_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
    
    # 3. Initialize MediaPipe
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    
    # --- START PROCESSING ---
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmentor:
        cap = cv2.VideoCapture(tfile_path)
        
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup Video Writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # UI Progress Elements
        status_text = st.empty()
        progress_bar = st.progress(0)
        status_text.text("🔄 Analyzing frames... Please keep this tab open.")

        frame_count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = segmentor.process(frame_rgb)
            
            # Create mask
            mask = results.segmentation_mask > 0.5
            
            # Create Green Background
            green_bg = np.zeros(frame.shape, dtype=np.uint8)
            green_bg[:] = (0, 255, 0)

            # Composite
            condition = np.stack((mask,) * 3, axis=-1)
            output_frame = np.where(condition, frame, green_bg)
            
            out.write(output_frame)
            
            frame_count += 1
            if total_frames > 0:
                progress_bar.progress(frame_count / total_frames)

        cap.release()
        out.release()

    # --- SHOW DOWNLOAD BUTTON ---
    if os.path.exists(output_path):
        st.divider()
        st.subheader("Process Complete")
        with open(output_path, "rb") as file:
            st.download_button(
                label="⬇️ Download Processed Video",
                data=file,
                file_name="background_removed.mp4",
                mime="video/mp4",
                use_container_width=True
            )
        st.success("Your video is ready for download.")
    
    # Cleanup
    try:
        os.remove(tfile_path)
    except:
        pass