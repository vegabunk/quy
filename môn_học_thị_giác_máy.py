import streamlit as st
import time
import base64
import os
import cv2

# === Tắt log/cảnh báo của OpenCV một cách phòng vệ ===
try:
    # Nếu có module utils.logging (OpenCV ≥4.x)
    log = getattr(cv2.utils, "logging", None)
    if log:
        log.setLogLevel(log.ERROR)
    # Ngược lại thử dùng setLogLevel trực tiếp
    elif hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
except Exception:
    pass

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Nội dung môn học",
    page_icon="📸",
    layout="wide"
)

# Hàm đọc file và mã hóa Base64 (dùng cho video background)
def get_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Đường dẫn video nền & lấy Base64
video_path = "resources/videos/background_11.mp4"
video_b64 = get_base64(video_path) if os.path.exists(video_path) else ""

# CSS + HTML để hiển thị video nền và overlay
css = f"""
<style>
  /* --- GIỮ NGUYÊN TOÀN BỘ CSS GỐC CỦA BẠN --- */
  .video-bg {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    z-index: -1;
  }}
  .overlay {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.5);
    z-index: 0;
  }}
  .frame-common {{
    background: rgba(255,255,255,0.8);
    border-radius: 10px;
    padding: 30px;
    margin: 20px;
  }}
  .big-text {{ font-size: 2.5rem; font-weight: bold; }}
  .sub-text {{ font-size: 1.2rem; }}
  .main-text {{ font-size: 1.5rem; font-weight: 500; }}
  /* ... (các style khác nếu có) ... */
</style>
<video class="video-bg" autoplay muted loop>
  <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
</video>
<div class="overlay"></div>
"""
st.markdown(css, unsafe_allow_html=True)

# Định nghĩa các trang con và đường dẫn file
dict_pages = {
    "Nhận diện khuôn mặt": "my_pages/1_nhan_dien_khuon_mat.py",
    "Phát hiện và nhận dạng đối tượng": "my_pages/2_phat_hien_doi_tuong.py",
    "Thị giác máy": "my_pages/thi_giac_may.py",
    "Các ứng dụng ": "my_pages/cac_ung_dung_khac.py"
}

# Khởi tạo biến lưu trang hiện tại
if "page" not in st.session_state:
    st.session_state.page = None

# Sidebar để chọn trang
with st.sidebar:
    st.image("resources/images/hcmute_logo.png", use_container_width=True)
    st.markdown('<div class="big-text">📚 NỘI DUNG MÔN HỌC</div>', unsafe_allow_html=True)
    if st.session_state.page is None:
        for name, file in dict_pages.items():
            if st.button(name, label_visibility="visible"):
                st.session_state.page = file
                st.rerun()
    else:
        if st.button("⬅️ Quay về trang chủ", label_visibility="visible"):
            st.session_state.page = None
            st.rerun()

# Phần nội dung chính
if st.session_state.page is None:
    # Màn hình chào mừng
    st.markdown('''
      <div class="frame-common" style="text-align:center;">
        <div class="big-text">CHÀO MỪNG BẠN ĐẾN VỚI MÔN THỊ GIÁC MÁY</div>
        <div class="sub-text">Chúng mình tên là Nguyễn Thành Quý và Phạm Gia Thiều</div>
      </div>
    ''', unsafe_allow_html=True)

    # Giới thiệu nội dung môn học
    st.markdown('''
      <div class="frame-common">
        <div class="big-text">NỘI DUNG MÔN HỌC</div>
        <ul>
          <li class="sub-text">Nhận diện khuôn mặt</li>
          <li class="sub-text">Nhận dạng và phát hiện đối tượng</li>
          <li class="sub-text">Thị giác máy</li>
          <li class="sub-text">Các ứng dụng</li>
        </ul>
      </div>
    ''', unsafe_allow_html=True)

    # Hướng dẫn chọn ở sidebar
    st.markdown('''
      <div class="frame-common">
        <div class="main-text">👈 Hãy chọn nội dung học ở thanh bên trái nhé!</div>
      </div>
    ''', unsafe_allow_html=True)

else:
    # Khi đã chọn một trang con, load file bên ngoài
    with st.spinner("Đang tải nội dung..."):
        time.sleep(1)
        try:
            exec(open(st.session_state.page, "r", encoding="utf-8").read())
        except Exception as e:
            st.error(f"Đã xảy ra lỗi khi tải nội dung: {e}")
