import streamlit as st
import cv2 as cv
import numpy as np
import joblib
from PIL import Image

# Load model
face_detector = cv.FaceDetectorYN.create(
    'face_detection_yunet_2023mar.onnx', "", (320, 320),
    score_threshold=0.9, nms_threshold=0.3, top_k=5000
)
face_recognizer = cv.FaceRecognizerSF.create('face_recognition_sface_2021dec.onnx', "")

svc = joblib.load('svc.pkl')
mydict = ['GIATHIEU','LEQUYEN','THANHQUY','UNKNOWN']

# Hàm visualize khuôn mặt
def visualize(input, faces, names, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1][:3]):
            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.putText(input, names[idx], (coords[0], coords[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return input

# Hàm nhận diện khuôn mặt
def recognize_faces(img):
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    frameHeight, frameWidth = img.shape[:2]
    face_detector.setInputSize((frameWidth, frameHeight))
    
    faces = face_detector.detect(img)
    names = []

    if faces[1] is not None:
        for face in faces[1][:3]:
            face_align = face_recognizer.alignCrop(img, face)
            face_feature = face_recognizer.feature(face_align)

            similarity_score = svc.decision_function(face_feature.reshape(1, -1))
            max_similarity = np.max(similarity_score)

            threshold = 0.5
            if max_similarity > threshold:
                test_predict = svc.predict(face_feature)
                names.append(mydict[test_predict[0]])
            else:
                names.append("Unknown")

    result = visualize(img, faces, names)
    result = cv.cvtColor(result, cv.COLOR_BGR2RGB)
    return result

# Giao diện Streamlit
st.set_page_config(page_title="Nhận diện khuôn mặt", layout="wide")
st.title("🔍 Hệ thống nhận diện khuôn mặt")

# Tạo 2 cột chọn chế độ
col1, col2 = st.columns(2)
with col1:
    mode = st.radio("Chọn chế độ:", ('Ảnh tĩnh', 'Webcam'), index=0, horizontal=True)

# Phần nội dung chính
if mode == 'Ảnh tĩnh':
    st.header("📷 Nhận diện từ ảnh tĩnh")
    uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Hiển thị 2 cột ảnh
        col1, col2 = st.columns(2)
        img = Image.open(uploaded_file)
        img = np.array(img)
        
        # Cột trái: Hiển thị ảnh tải lên
        with col1:
            st.image(img, caption="Ảnh tải lên", use_container_width=True)

        # Cột phải: Hiển thị ảnh kết quả nhận diện (cùng kích thước với ảnh tải lên)
        with col2:
            output_img = recognize_faces(img)
            st.image(output_img, caption="Kết quả nhận diện", use_container_width=True)  

elif mode == 'Webcam':
    st.header("🎥 Nhận diện từ Webcam")
    if 'webcam' not in st.session_state:
        st.session_state['webcam'] = False

    start_button = st.button('▶️ Start Webcam')
    stop_button = st.button('⏹️ Stop Webcam')

    FRAME_WINDOW = st.image([])

    cap = cv.VideoCapture(0)

    if start_button:
        st.session_state['webcam'] = True
    if stop_button:
        st.session_state['webcam'] = False

    if st.session_state['webcam']:
        while st.session_state['webcam']:
            ret, frame = cap.read()
            if not ret:
                st.warning("Không lấy được hình ảnh từ webcam.")
                break
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            result = recognize_faces(frame)
            FRAME_WINDOW.image(result, use_container_width=True)  
    cap.release()
