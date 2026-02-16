import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="VantageBG", page_icon="🎬")
st.title("🎬 VantageBG: AI Background Remover")

uploaded_file = st.file_uploader("Upload a video (MP4/MOV)", type=["mp4", "mov"])

if uploaded_file is not None:
    # 1. Handle Temporary Files
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") 
    tfile.write(uploaded_file.read())
    tfile_path = tfile.name
    tfile.close() 
    
    output_path = os.path.join(tempfile.gettempdir(), "vantage_output.mp4")
    
    # 2. Run AI Segmentation
    # Using the standard solutions path for cloud stability
    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as segmentor:
        cap = cv2.VideoCapture(tfile_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 'mp4v' is the safest codec for Streamlit Cloud to Windows downloads
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
            
            if results.segmentation_mask is not None:
                mask = results.segmentation_mask > 0.5
                green_bg = np.zeros(frame.shape, dtype=np.uint8)
                green_bg[:] = (0, 255, 0) # Neon Green (BGR)
                
                condition = np.stack((mask,) * 3, axis=-1)
                output_frame = np.where(condition, frame, green_bg)
                out.write(output_frame)
            
            frame_count += 1
            if total_frames > 0:
                bar.progress(min(frame_count / total_frames, 1.0))
            status.text(f"Processing: {frame_count}/{total_frames} frames")

        cap.release()
        out.release()

    # 3. Final Download Link
    if os.path.exists(output_path):
        st.divider()
        with open(output_path, "rb") as file:
            st.download_button(
                label="⬇️ Download VantageBG Result",
                data=file,
                file_name="background_removed.mp4",
                mime="video/mp4"
            )
        st.success("Processing Complete!")
    
    # Cleanup temp file
    try:
        os.remove(tfile_path)
    except:
        pass
