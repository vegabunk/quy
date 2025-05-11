import streamlit as st
import cv2 as cv
import numpy as np
import joblib
from PIL import Image
import base64
import os
import time

# === Mở rộng: thêm suppress_stderr để tắt log OpenCV về camera ===
import sys
from contextlib import redirect_stderr

@st.cache_resource
def suppress_stderr():
    # Trả về file object /dev/null để redirect
    return open(os.devnull, "w")

# Main function
def main():
    # Load video as base64
    def get_base64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    video_path = "resources/videos/background_7.mp4"
    video_b64 = get_base64(video_path) if os.path.exists(video_path) else ""

    # CSS + HTML injection (only sidebar gradient and video background)
    css_html = f"""
    <style>
    /* Cover entire sidebar with gradient from light pink to dark pink */
    [data-testid="stSidebar"] {{
        background: linear-gradient(135deg, #ffc0cb, #ff69b4) !important;
    }}
    /* Video background behind content */
    .video-bg {{
        position: fixed;
        top: 0;
        left: 18rem;
        width: calc(100% - 18rem);
        height: 100vh;
        z-index: -2;
        object-fit: cover;
    }}
    /* Dark overlay above video */
    .overlay {{
        position: fixed;
        top: 0;
        left: 18rem;
        width: calc(100% - 18rem);
        height: 100vh;
        z-index: -1;
        background-color: rgba(0, 0, 0, 0.1);
    }}
    /* Transparent main container */
    [data-testid="stAppViewContainer"] {{
        background-color: transparent !important;
    }}
    </style>
    <video class="video-bg" autoplay muted loop>
      <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
    </video>
    <div class="overlay"></div>
    """
    st.markdown(css_html, unsafe_allow_html=True)

    # Load face detection and recognition models
    face_detector = cv.FaceDetectorYN.create(
        'face_detection_yunet_2023mar.onnx', '', (320, 320),
        score_threshold=0.9, nms_threshold=0.3, top_k=5000
    )
    face_recognizer = cv.FaceRecognizerSF.create(
        'face_recognition_sface_2021dec.onnx', ''
    )
    svc = joblib.load('svc.pkl')
    names_list = ['GIATHIEU', 'LEQUYEN', 'THANHQUY']

    # Function to draw rectangles and labels
    def visualize(img, faces, names):
        if faces[1] is not None:
            for i, face in enumerate(faces[1][:3]):
                x, y, w, h = map(int, face[:4])
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if i < len(names):
                    cv.putText(img, names[i], (x, y - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return img

    # Function to recognize faces in a frame
    def recognize(frame):
        bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        face_detector.setInputSize((w, h))
        faces = face_detector.detect(bgr)
        detected = []
        if faces[1] is not None:
            for face in faces[1][:3]:
                crop = face_recognizer.alignCrop(bgr, face)
                feat = face_recognizer.feature(crop)
                score = svc.decision_function(feat.reshape(1, -1))
                if np.max(score) > 0.5:
                    idx = int(svc.predict(feat.reshape(1, -1))[0])
                    detected.append(names_list[idx])
                else:
                    detected.append('Unknown')
        result = visualize(bgr, faces, detected)
        return cv.cvtColor(result, cv.COLOR_BGR2RGB)

    # Function to run webcam loop
    def webcam_loop():
        # === Mở rộng: mở camera trong context redirect_stderr ===
        with redirect_stderr(suppress_stderr()):
            cap = cv.VideoCapture(0)

        # Kiểm tra ngay sau khi mở camera
        if not cap.isOpened():
            st.sidebar.error("🔴 Không thể mở camera với index 0. Vui lòng kiểm tra kết nối hoặc chọn chế độ 'Ảnh tĩnh'.")
            return

        placeholder = st.empty()
        while st.session_state.get('webcam', False):
            ret, frame = cap.read()
            if not ret:
                st.error("⚠️ Không đọc được khung hình từ camera.")
                break
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            placeholder.image(recognize(rgb), channels='RGB', use_container_width=True)
        cap.release()

    # UI elements
    st.markdown(
        '<h1 style="text-align:left; color:#330000;">🔮 Ứng dụng nhận diện khuôn mặt</h1>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown('<h3>🎛️ Chọn chế độ</h3>', unsafe_allow_html=True)
    mode = st.sidebar.radio('', ['Ảnh tĩnh', 'Webcam'])

    if mode == 'Ảnh tĩnh':
        uploaded_file = st.sidebar.file_uploader('', type=['jpg', 'png', 'jpeg','bmp','tif'])
        if uploaded_file:
            img = np.array(Image.open(uploaded_file))
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, use_container_width=True)
            with col2:
                st.image(recognize(img), use_container_width=True)
    else:
        if 'webcam' not in st.session_state:
            st.session_state['webcam'] = False
        if st.button('Start Webcam'):
            st.session_state['webcam'] = True
        if st.button('Stop Webcam'):
            st.session_state['webcam'] = False
        if st.session_state['webcam']:
            webcam_loop()

if __name__ == '__main__':
    main()
