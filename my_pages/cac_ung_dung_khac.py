import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
import cv2
import tempfile
import numpy as np
import base64
import os
from PIL import Image
from io import BytesIO
import pickle
import mediapipe as mp
import time

# === WebRTC ===
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# === Base64 video background & CSS (giữ nguyên) ===
def get_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

video_path = os.path.join(os.path.dirname(__file__), "resources/videos/background_3.mp4")
video_b64 = get_base64(video_path) if os.path.exists(video_path) else ""
css = f"""
<style>
  [data-testid="stSidebar"], [data-testid="stSidebarNav"] {{
    background: linear-gradient(135deg,#ff9999,#ffcc99,#ffff99,#99ff99,#99ffcc,#99ffff) !important;
    height: 100vh; padding: 0;
  }}
  .video-bg {{
    position: fixed; top: 0; left: 0;
    width: 100%; height: 100vh;
    object-fit: cover; z-index: -1; opacity: 0.92;
  }}
  h1 {{ color: white !important; text-shadow: 1px 1px 2px black; }}
  .caption-text {{
    font-size: 18px; font-weight: bold;
    color: white !important; margin-bottom: 5px;
    text-shadow: 1px 1px 2px black;
  }}
</style>
<video class="video-bg" autoplay muted loop>
  <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
</video>
"""
st.markdown(css, unsafe_allow_html=True)

# === Common Configurations ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# === Style Transfer Components (giữ nguyên) ===
def test_transform(image_size=None):
    resize = [transforms.Resize(image_size)] if image_size else []
    return transforms.Compose(resize + [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

def denormalize(tensors):
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return tensors

def inference(content_image, checkpoint_model):
    transformer = TransformerNet().to(device)
    transformer.load_state_dict(torch.load(checkpoint_model, map_location=device))
    transformer.eval()
    img = Image.open(content_image)
    tensor = Variable(test_transform()(img)).to(device).unsqueeze(0)
    with torch.no_grad():
        out = denormalize(transformer(tensor)).cpu()
    out_img = np.clip(out.squeeze(0).permute(1,2,0).numpy(), 0, 1)
    return out_img

class ConvBlock(nn.Module):
    def __init__(self,in_c,out_c,kernel_size,stride=1,upsample=False,normalize=True,relu=True):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.ReflectionPad2d(kernel_size//2),
            nn.Conv2d(in_c,out_c,kernel_size,stride)
        )
        self.norm = nn.InstanceNorm2d(out_c, affine=True) if normalize else None
        self.relu = relu
    def forward(self,x):
        if self.upsample: x = F.interpolate(x, scale_factor=2)
        x = self.block(x)
        if self.norm: x = self.norm(x)
        return F.relu(x) if self.relu else x

class ResidualBlock(nn.Module):
    def __init__(self,ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(ch,ch,3,1,False,True,True),
            ConvBlock(ch,ch,3,1,False,True,False)
        )
    def forward(self,x): return self.block(x) + x

class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            ConvBlock(3,32,9,1),
            ConvBlock(32,64,3,2),
            ConvBlock(64,128,3,2),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ConvBlock(128,64,3,upsample=True),
            ConvBlock(64,32,3,upsample=True),
            ConvBlock(32,3,9,1,False,False)
        )
    def forward(self,x): return self.model(x)

# === Hand Detection Components (giữ nguyên) ===
labels_dict = {
    0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'K',
    10:'L',11:'M',12:'N',13:'O',14:'P',15:'Q',16:'R',17:'S',18:'T',
    19:'U',20:'V',21:'W',22:'X',23:'Y'
}
DESIRED_ASPECT_RATIO, PADDING = 1.3333, 10
model_dict = pickle.load(open('./MLP_model.p','rb'))
model = model_dict['model']
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
annotation_img = Image.open('resources/images/bang_ky_tu.jpg')

def calculate_bounding_box(hand_landmarks, frame_shape):
    h, w, _ = frame_shape
    x_min, y_min, x_max, y_max = w, h, 0, 0
    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min, y_min = min(x, x_min), min(y, y_min)
        x_max, y_max = max(x, x_max), max(y, y_max)
    return (max(0, x_min - PADDING), max(0, y_min - PADDING),
            min(w, x_max + PADDING), min(h, y_max + PADDING))

def enforce_aspect_ratio(x_min, y_min, x_max, y_max, frame_shape, ratio):
    h, w, _ = frame_shape
    bw, bh = x_max - x_min, y_max - y_min
    if bh / bw < ratio:
        new_h = int(bw * ratio)
        yc = (y_min + y_max) // 2
        y_min, y_max = max(0, yc - new_h//2), min(h, yc + new_h//2)
    else:
        new_w = int(bh / ratio)
        xc = (x_min + x_max) // 2
        x_min, x_max = max(0, xc - new_w//2), min(w, xc + new_w//2)
    return x_min, y_min, x_max, y_max

# === Main Application ===
def main():
    st.sidebar.title("Ứng dụng thị giác máy")
    mode = st.sidebar.radio("Chọn ứng dụng", [
        "Nhận diện ký tự tay",
        "Chuyển đổi phong cách ảnh"
    ])

    if mode == "Nhận diện ký tự tay":
        st.title("🔠 Nhận diện ký tự tay")
        # Start/Stop buttons (giữ nguyên logic nút)
        if 'cam_running' not in st.session_state:
            st.session_state.cam_running = False
        col1, col2 = st.columns(2)
        if col1.button('Start'):
            st.session_state.cam_running = True
        if col2.button('Stop'):
            st.session_state.cam_running = False

        if st.session_state.cam_running:
            # Live webcam stream via WebRTC
            class HandProcessor(VideoProcessorBase):
                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    img_bgr = frame.to_ndarray(format="bgr24")
                    frame = cv2.flip(img_bgr, 1)

                    # Overlay annotation image
                    h, w, _ = frame.shape
                    anno = np.array(annotation_img.resize((int(w*0.3), int(h*0.3))))
                    frame[10:10+anno.shape[0], w-10-anno.shape[1]:w-10] = anno

                    # Hand detection & classification
                    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results.multi_hand_landmarks and results.multi_handedness:
                        for hand_landmarks, handedness in zip(
                            results.multi_hand_landmarks, results.multi_handedness
                        ):
                            x_min, y_min, x_max, y_max = calculate_bounding_box(hand_landmarks, frame.shape)
                            x_min, y_min, x_max, y_max = enforce_aspect_ratio(
                                x_min, y_min, x_max, y_max, frame.shape, DESIRED_ASPECT_RATIO
                            )
                            label = handedness.classification[0].label
                            color = (0,255,0) if label == 'Left' else (0,0,255)
                            cv2.rectangle(frame, (x_min,y_min), (x_max,y_max), color, 3)
                            # normalize landmarks for model input
                            xs = [lm.x for lm in hand_landmarks.landmark]
                            ys = [lm.y for lm in hand_landmarks.landmark]
                            data_aux = []
                            for lm in hand_landmarks.landmark:
                                data_aux += [lm.x - min(xs), lm.y - min(ys)]
                            pred = model.predict([np.asarray(data_aux)])
                            ch = labels_dict.get(int(pred[0]), 'Unknown')
                            cv2.putText(
                                frame,
                                f'{label} hand: {ch}',
                                (x_min, y_min - 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0,
                                color,
                                3
                            )
                            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    return av.VideoFrame.from_ndarray(frame, format="bgr24")

            webrtc_ctx = webrtc_streamer(
                key="hand-stream",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=HandProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            if webrtc_ctx.state.playing:
                st.success("🎬 Đang livestream và nhận diện ký tự tay…")
            else:
                st.info("⏸️ Nhấn ▶️ để bắt đầu livestream")

    else:
        st.title("🎨 Chuyển đổi phong cách ảnh")
        # (giữ nguyên logic style transfer)
        sub1, space1, sub2, space2, sub3 = st.columns([10,0.2,10,0.2,10])
        with sub1:
            uploaded = st.sidebar.file_uploader('', type=['png','jpg','jpeg','tif'], label_visibility='collapsed')
            if uploaded:
                st.image(Image.open(uploaded), use_container_width=True)
        with sub2:
            chose_style = st.selectbox(
                '', options=[
                    'Phong cách Hậu ấn tượng',
                    'Phong cách Lập thể',
                    'Phong cách Trừu tượng',
                    'Phong cách Số hóa'
                ],
                label_visibility='collapsed'
            )
            style_btn = st.button('Chuyển đổi!')
            style_dict = {
                'Phong cách Hậu ấn tượng': 'weights/best_model-Post_Impressionism.pth',
                'Phong cách Lập thể':      'weights/best_model-Cubism.pth',
                'Phong cách Trừu tượng':   'weights/best_model-Abstract_Expressionism.pth',
                'Phong cách Số hóa':       'weights/best_model-Digital_Painting.pth'
            }
        with sub3:
            if uploaded and style_btn:
                result_img = inference(
                    content_image=uploaded,
                    checkpoint_model=style_dict[chose_style]
                )
                st.image(result_img, use_container_width=True)
                buf = BytesIO()
                Image.fromarray((result_img*255).astype(np.uint8)).save(buf, format="PNG")
                st.download_button(
                    label="📥 Tải ảnh về",
                    data=buf.getvalue(),
                    file_name='styled_image.png',
                    mime="image/png"
                )

if __name__ == '__main__':
    main()
