import streamlit as st
import cv2
import numpy as np
import io
import base64
import os
from PIL import Image, ImageDraw, ImageFont
from typing import Any
from ultralytics import YOLO
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator

# 💥 Chỉ gọi set_page_config một lần nếu cần
# st.set_page_config(layout="wide")

# === Hàm hỗ trợ phân loại hình học ===
@st.cache_data
def phan_nguong(imgin: np.ndarray) -> np.ndarray:
    if len(imgin.shape) == 3:
        gray = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
    else:
        gray = imgin.copy()
    M, N = gray.shape
    imgout = np.zeros((M, N), np.uint8)
    for x in range(M):
        for y in range(N):
            imgout[x, y] = 255 if gray[x, y] == 63 else 0
    imgout = cv2.medianBlur(imgout, 7)
    return imgout

@st.cache_data
def predict_shape(imgin: np.ndarray) -> (np.ndarray, str):
    bin_img = phan_nguong(imgin)
    m = cv2.moments(bin_img)
    hu = cv2.HuMoments(m).flatten()
    h0 = hu[0]
    if 0.000620 <= h0 <= 0.000632:
        label = 'Hình Tròn'
    elif 0.000644 <= h0 <= 0.000668:
        label = 'Hình Vuông'
    elif 0.000725 <= h0 <= 0.000751:
        label = 'Hình Tam Giác'
    else:
        label = 'Không xác định'
    return bin_img, label

# === Hàm lấy base64 cho video background ===
def get_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# === CSS & Video Background & Styles ===
video_path = os.path.join(os.path.dirname(__file__), "resources/videos/background_5.mp4")
video_b64 = get_base64(video_path) if os.path.exists(video_path) else ""
css = f"""
<style>
  /* Gradient sidebar */
  [data-testid="stSidebar"], [data-testid="stSidebarNav"] {{
    background: linear-gradient(135deg,#ccff99,#99ff99,#b2ff66,#66ff66,#99ff33, #33ff33, #80ff00, #00ff00) !important;
    height: 100vh;
    padding: 0;
  }}
  /* Video nền */
  .video-bg {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100vh;
    object-fit: cover;
    z-index: -1;
    opacity: 0.92;
  }}
  /* Title trắng, viền đen */
  h1 {{
    color: white !important;
    text-shadow: 1px 1px 2px black;
  }}
  /* Style cho nhãn ảnh */
  .caption-text {{
    font-size: 18px;
    font-weight: bold;
    color: white !important;
    margin-bottom: 5px;
    text-shadow: 1px 1px 2px black;
  }}
</style>
<video class="video-bg" autoplay muted loop>
  <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
</video>
"""
st.markdown(css, unsafe_allow_html=True)

# === Class Phát hiện đối tượng (video) ===
class Inference:
    def __init__(self, **kwargs: Any):
        check_requirements("streamlit>=1.29.0")
        st.title("🔍 Ứng dụng Phát hiện đối tượng")
        self.st = st
        self.conf = 0.25
        self.iou = 0.45
        self.source = None
        self.enable_trk = False
        self.org_frame = None
        self.ann_frame = None
        self.model = None
        self.vid_file_name = 0
        self.selected_ind = []
        self.temp = {"model": None, **kwargs}
        self.model_path = self.temp.get("model")

    def sidebar(self):
        with st.sidebar:
            st.title("Cấu hình")
            self.source = st.selectbox("Video từ", ("webcam", "video"))
            self.enable_trk = st.radio("Bật tracking", ("Yes", "No"))
            self.conf = float(st.slider("Ngưỡng confidence", 0.0, 1.0, self.conf, 0.01))
            self.iou = float(st.slider("Ngưỡng IoU", 0.0, 1.0, self.iou, 0.01))
        col1, col2 = st.columns(2)
        self.org_frame = col1.empty()
        self.ann_frame = col2.empty()

    def source_upload(self):
        self.vid_file_name = 0
        if self.source == "video":
            vid = st.sidebar.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])
            if vid is not None:
                content = io.BytesIO(vid.read())
                with open("ultra.mp4", "wb") as f:
                    f.write(content.read())
                self.vid_file_name = "ultra.mp4"

    def configure(self):
        models = [x.replace("yolo", "YOLO") for x in GITHUB_ASSETS_STEMS if x.startswith("yolo11")] 
        if self.model_path:
            models.insert(0, self.model_path.split(".pt")[0])
        sel = st.sidebar.selectbox("Chọn model", models)
        with st.spinner("Loading model..."):
            self.model = YOLO(f"{sel.lower()}.pt")
        names = list(self.model.names.values())
        cls = st.sidebar.multiselect("Chọn đối tượng", names, default=names[:3])
        self.selected_ind = [names.index(c) for c in cls]

    def run(self):
        self.sidebar()
        self.source_upload()
        self.configure()
        if st.button("Start"):
            stop = st.button("Stop")
            cap = cv2.VideoCapture(self.vid_file_name)
            if not cap.isOpened():
                st.error("Không mở nguồn.")
                return
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                if self.enable_trk == "Yes":
                    res = self.model.track(
                        frame, conf=self.conf, iou=self.iou,
                        classes=self.selected_ind, persist=True
                    )
                else:
                    res = self.model(
                        frame, conf=self.conf, iou=self.iou,
                        classes=self.selected_ind
                    )
                out = res[0].plot()
                if stop:
                    cap.release()
                    st.stop()
                self.org_frame.image(frame, channels="BGR")
                self.ann_frame.image(out, channels="BGR")
            cap.release()
            cv2.destroyAllWindows()

# === Hàm Nhận diện trái cây (ảnh) ===
def fruit_detection():
    st.title("🍎 Nhận diện và phân loại trái cây")
    model = YOLO("yolo11n_trai_cay.pt", task="detect")
    buf = st.sidebar.file_uploader("Upload ảnh", type=["bmp", "png", "jpg", "jpeg", "tif"])
    if buf is not None:
        img = Image.open(buf).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='caption-text'>Ảnh gốc</div>", unsafe_allow_html=True)
            st.image(img, use_container_width=True)
        if st.button("Process"):
            arr = np.array(img)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            res = model.predict(bgr, conf=0.5, verbose=False)
            boxes = res[0].boxes.xyxy.cpu()
            cls = res[0].boxes.cls.cpu().tolist()
            confs = res[0].boxes.conf.tolist()
            names = model.names
            ann = Annotator(bgr)
            for b, c, cf in zip(boxes, cls, confs):
                ann.box_label(b, f"{names[int(c)]} {cf:.2f}")
            out_bgr = ann.result()
            out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
            with col2:
                st.markdown("<div class='caption-text'>Ảnh sau nhận diện</div>", unsafe_allow_html=True)
                st.image(out_rgb, use_container_width=True)
                st.download_button(
                    "💾 Tải ảnh",
                    data=cv2.imencode('.jpg', cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))[1].tobytes(),
                    file_name="ketqua.jpg",
                    mime="image/jpeg"
                )

# === Hàm Phân loại hình học (ảnh) ===
def shape_detection():
    st.title("🔷  Nhận diện và phân loại hình học")
    uploaded = st.sidebar.file_uploader('Upload ảnh gốc', type=['png', 'jpg', 'jpeg', 'bmp'])
    if uploaded is not None:
        img = Image.open(uploaded).convert('RGB')
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='caption-text'>Ảnh gốc</div>", unsafe_allow_html=True)
            st.image(img, use_container_width=True)
        if st.button('Process'):
            bin_img, label = predict_shape(img_np)
            pil = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            try:
                font = ImageFont.truetype('resources/fonts/Arial.ttf', 40)
            except:
                font = ImageFont.load_default()
            draw.text((10, 30), label, font=font, fill=(0, 255, 0))
            with col2:
                st.markdown("<div class='caption-text'>Ảnh chú thích</div>", unsafe_allow_html=True)
                st.image(pil, use_container_width=True)
                st.markdown("<div class='caption-text'>Ảnh nhị phân</div>", unsafe_allow_html=True)
                st.image(bin_img, use_container_width=True)
   

# === Main ===
def main():
    mode = st.sidebar.radio('Chức năng', [
        'Phát hiện đối tượng (video)',
        'Nhận diện và phân loại trái cây (ảnh)',
        'Nhận diện và phân loại hình học (ảnh)'
    ])
    if mode == 'Phát hiện đối tượng (video)':
        Inference().run()
    elif mode == 'Nhận diện và phân loại trái cây (ảnh)':
        fruit_detection()
    else:
        shape_detection()

if __name__ == '__main__':
    main()
