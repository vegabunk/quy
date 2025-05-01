import streamlit as st
# cấu hình page must be first

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Hàm phân ngưỡng ảnh (thresholding)
@st.cache_data
def phan_nguong(imgin):
    # Chuyển sang grayscale nếu ảnh màu
    if len(imgin.shape) == 3:
        gray = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
    else:
        gray = imgin.copy()
    # Tạo ảnh nhị phân: pixel == 63 thành 255, ngược lại 0
    M, N = gray.shape
    imgout = np.zeros((M, N), np.uint8)
    for x in range(M):
        for y in range(N):
            imgout[x, y] = 255 if gray[x, y] == 63 else 0
    # Làm mịn biên bằng median blur
    imgout = cv2.medianBlur(imgout, 7)
    return imgout

# Hàm tính Hu Moments và phân loại hình học
def predict_shape(imgin):
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

# Giao diện Streamlit
#st.set_page_config(layout='wide')
st.title('Phân Loại Hình Học')
st.sidebar.header('Tùy chọn')

# Upload ảnh gốc
uploaded_file = st.sidebar.file_uploader('Upload ảnh gốc', type=['png', 'jpg', 'jpeg', 'bmp'])

if uploaded_file is not None:
    # Đọc ảnh gốc và hiển thị
    image = Image.open(uploaded_file).convert('RGB')
    img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    st.image(image, caption='Ảnh gốc',use_container_width=True)

    # Nút Process trên màn hình chính
    if st.button('Process'):
        # Phân loại
        bin_img, shape_label = predict_shape(img_np)
        # Chuyển ảnh gốc sang PIL để ghi text Unicode
        pil_img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        # Sử dụng font TTF hỗ trợ tiếng Việt (cần đặt Arial.ttf vào resources/fonts)
        try:
            font = ImageFont.truetype('resources/fonts/Arial.ttf', 40)
        except OSError:
            font = ImageFont.load_default()
        # Ghi nhãn lên ảnh
        draw.text((10, 30), shape_label, font=font, fill=(0, 255, 0))
        # Hiển thị ảnh chú thích và ảnh nhị phân
        st.image(pil_img, caption='Ảnh chú thích', use_container_width=True)
        st.image(bin_img, caption='Ảnh nhị phân', use_container_width=True)
else:
    st.info('Vui lòng tải ảnh gốc lên để phân loại hình học.')
