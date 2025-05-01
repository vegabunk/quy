import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
from PIL import Image

# Load annotation image
annotation_img = Image.open('resources/images/bang_ky_tu.jpg')

# Thiết lập Streamlit
st.title('Hand Detection with Start/Stop')
if 'running' not in st.session_state:
    st.session_state.running = False

# Nút Start và Stop
col1, col2 = st.columns(2)
with col1:
    if st.button('Start'):
        st.session_state.running = True
with col2:
    if st.button('Stop'):
        st.session_state.running = False

# Load model và khởi tạo MediaPipe
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
               8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P',
               15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W',
               22: 'X', 23: 'Y'}
DESIRED_ASPECT_RATIO = 1.3333
PADDING = 10
model_dict = pickle.load(open('./MLP_model.p', 'rb'))
model = model_dict['model']
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Hàm tính bounding box
def calculate_bounding_box(hand_landmarks, frame_shape):
    h, w, _ = frame_shape
    x_min, y_min, x_max, y_max = w, h, 0, 0
    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min, y_min = min(x, x_min), min(y, y_min)
        x_max, y_max = max(x, x_max), max(y, y_max)
    x_min, y_min = max(0, x_min - PADDING), max(0, y_min - PADDING)
    x_max, y_max = min(w, x_max + PADDING), min(h, y_max + PADDING)
    return x_min, y_min, x_max, y_max

# Hàm ép tỉ lệ khung
def enforce_aspect_ratio(x_min, y_min, x_max, y_max, frame_shape, desired_aspect_ratio):
    h, w, _ = frame_shape
    box_w, box_h = x_max - x_min, y_max - y_min
    current_ratio = box_h / box_w
    if current_ratio < desired_aspect_ratio:
        new_h = int(box_w * desired_aspect_ratio)
        yc = (y_min + y_max) // 2
        y_min, y_max = max(0, yc - new_h//2), min(h, yc + new_h//2)
    else:
        new_w = int(box_h / desired_aspect_ratio)
        xc = (x_min + x_max) // 2
        x_min, x_max = max(0, xc - new_w//2), min(w, xc + new_w//2)
    return x_min, y_min, x_max, y_max

# Placeholder hiển thị frame
frame_placeholder = st.empty()

# Mở webcam
cap = cv2.VideoCapture(0)

# Vòng lặp chính
while True:
    if not st.session_state.running:
        time.sleep(0.1)
        frame_placeholder.empty()
        continue

    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Hiển thị annotation image cố định góc phải, tăng kích thước lên 30%
    h_frame, w_frame, _ = frame.shape
    anno = np.array(annotation_img.resize((int(w_frame*0.3), int(h_frame*0.3))))
    x_off = w_frame - anno.shape[1] - 10
    y_off = 10
    frame[y_off:y_off+anno.shape[0], x_off:x_off+anno.shape[1]] = anno

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            data_aux, x_, y_ = [], [], []
            x_min, y_min, x_max, y_max = calculate_bounding_box(hand_landmarks, frame.shape)
            x_min, y_min, x_max, y_max = enforce_aspect_ratio(x_min, y_min, x_max, y_max, frame.shape, DESIRED_ASPECT_RATIO)
            label = handedness.classification[0].label
            color = (0, 255, 0) if label == 'Left' else (0, 0, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
            for lm in hand_landmarks.landmark:
                x_.append(lm.x); y_.append(lm.y)
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_)); data_aux.append(lm.y - min(y_))
            pred = model.predict([np.asarray(data_aux)])
            ch = labels_dict.get(int(pred[0]), "Unknown")
            cv2.putText(frame, f'{label} hand: {ch}', (x_min, y_min - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Hiển thị frame với kích thước lớn hơn
    frame_placeholder.image(frame, channels='BGR', use_container_width=True)
    if not st.session_state.running:
        break

# Giải phóng
cap.release()
cv2.destroyAllWindows()
