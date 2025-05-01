import streamlit as st
import time
import base64
import os

# Cấu hình trang
st.set_page_config(page_title="Nội dung môn học", page_icon="📸", layout="wide")

# Hàm đọc và mã hóa file sang Base64
def get_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Video nền
video_path = "resources/videos/background_11.mp4"
video_b64 = get_base64(video_path) if os.path.exists(video_path) else ""

# CSS & HTML
css = f"""
<style>
  /* Ẩn navigation gốc */
  section[data-testid=\"stSidebarNav\"], header[data-testid=\"stHeader\"] {{ display: none; }}

  /* Sidebar nền pastel mint và viền rainbow pastel */
  [data-testid=\"stSidebar\"] {{
    background: #E6FFCC !important;
    border: 3px solid;
    border-image: linear-gradient(45deg, #FFDDEE, #FFF5CC, #E6FFCC, #CCEFFF, #E5CCFF, #FFE5CC, #FFCCE5) 1;
    box-shadow: 0 0 15px rgba(255,255,255,0.6);
    padding-top: 1rem;
    color: #000 !important;
  }}

  /* Nút bấm sidebar */
  [data-testid=\"stSidebar\"] .stButton>button {{
    background: linear-gradient(270deg, #FFF5CC, #E6FFCC, #CCEFFF, #E5CCFF, #FFE5CC);
    background-size: 400% 500%;
    animation: btnPastel 12s ease infinite;
    color: #000;
    padding: 16px;
    font-size: 26px;
    font-weight: bold;
    border: 4px solid rgba(102,255,255,1);
    border-radius: 14px;
    margin: 8px 0;
    width: 100%;
  }}
  [data-testid=\"stSidebar\"] .stButton>button:hover {{
    transform: scale(1.05);
    color: #333;
  }}
  @keyframes btnPastel {{
    0%,100% {{ background-position:0% 50%; }}
    50% {{ background-position:100% 50%; }}
  }}

  /* Video nền */
  .video-bg {{
    position: fixed;
    top: 0;
    left: 18rem;
    height: 110%;
    z-index: -3;
    object-fit: cover;
  }}
  .overlay {{
    position: fixed;
    top: 0;
    left: 260px;
    right: 0;
    bottom: 0;
    z-index: -4;
    background: rgba(255,255,255,0.1);
  }}

  /* Nội dung nổi lên */
  .block-container, .element-container {{
    position: relative;
    z-index: 1;
    color: #003366 !important;
  }}

  /* Kiểu chữ chung */
  .big-text {{ color: #003366 !important; font-weight: 700 !important; }}
  .sub-text {{ color: #003366 !important; font-weight: 700 !important; }}
  .main-text {{ color: #003366 !important; font-weight: 700 !important; }}

  /* Glow-border chung */
  @keyframes glow {{
    0%,100% {{ box-shadow: 0 0 10px rgba(221,221,221,0.7); }}
    50%  {{ box-shadow: 0 0 20px rgba(221,221,221,1); }}
  }}
  .frame-common {{
    border: 3px solid #DDD;
    animation: glow 2.5s ease-in-out infinite;
    background-color: rgba(255,255,255,0.4);
    backdrop-filter: blur(4px);
  }}

  /* Welcome frame */
  .welcome-frame {{
    display: block;
    margin: 24px auto 0 auto;
    padding: 24px 32px;
    border-radius: 16px;
    text-align: center;
  }}
  .welcome-frame .big-text {{
    font-size: 40px !important;
    text-transform: uppercase;
    margin-bottom: 12px;
  }}
  .welcome-frame .big-text::before {{ content: '🎉 '; }}
  .welcome-frame .sub-text {{
    font-size: 28px !important;
    margin-top: 8px;
  }}

  /* Nội dung môn học frame */
  .section-container {{
    margin:24px auto 0 auto;
    display: inline-block;
    
    text-align: left;
  }}
  .section-frame {{
    padding: 24px 32px;
    border-radius: 16px;
  }}
  .section-frame .big-text {{
    font-size: 40px !important;
    text-transform: uppercase;
    margin-bottom: 12px;
    display: inline;
  }}
  .section-frame .sub-text {{
    font-size: 28px !important;
    margin: 8px 0;
    display: block;
  }}

  /* Last frame */
  .last-frame {{
    display: inline-block;
    margin: 24px auto 0 auto;
    padding: 24px 32px;
    border-radius: 16px;
    text-align: left;
  }}
  .last-frame .main-text {{
    font-size: 32px !important;
  }}
</style>
<video class="video-bg" autoplay muted loop>
  <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
</video>
<div class="overlay"></div>
"""

st.markdown(css, unsafe_allow_html=True)

# Sidebar pages (giữ nguyên)
dict_pages = {
    "Nhận diện khuôn mặt": "my_pages/1_nhan_dien_khuon_mat.py",
    "Phát hiện và nhận dạng đối tượng": "my_pages/2_phat_hien_doi_tuong.py",
    "Thị giác máy": "my_pages/thi_giac_may.py",
    "Các ứng dụng ": "my_pages/cac_ung_dung_khac.py"
   
}

if "page" not in st.session_state:
    st.session_state.page = None

with st.sidebar:
    st.image("resources/images/hcmute_logo.png", use_container_width=True)
    st.markdown('<div class="big-text">📚 NỘI DUNG MÔN HỌC </div>', unsafe_allow_html=True)
    if st.session_state.page is None:
        for name, file in dict_pages.items():
            if st.button(name):
                st.session_state.page = file
                st.rerun()
    else:
        if st.button("⬅️ Quay về trang chủ"):
            st.session_state.page = None
            st.rerun()

# Nội dung chính
if st.session_state.page is None:
    # Welcome frame
    st.markdown(
      '''
      <div class="welcome-frame frame-common">
        <div class="big-text">CHÀO MỪNG BẠN ĐẾN VỚI MÔN THỊ GIÁC MÁY</div>
        <div class="sub-text">Chúng mình tên là Nguyễn Thành Quý và Phạm Gia Thiều</div>
      </div>
      ''', unsafe_allow_html=True
    )

    # Nội dung môn học frame
    st.markdown(
      '''
      <div class="section-container">
        <div class="section-frame frame-common">
          <div class="big-text">NỘI DUNG MÔN HỌC</div>
          <ul>
            <li class="sub-text">Nhận diện khuôn mặt</li>
            <li class="sub-text">Nhận dạng và phát hiện đối tượng</li>
            <li class="sub-text">Thị giác máy</li>
            <li class="sub-text">Các ứng dụng</li>
          </ul>
        </div>
      </div>
      ''', unsafe_allow_html=True
    )

    # Last frame
    st.markdown(
      '<div class="last-frame frame-common"><div class="main-text">👈 Hãy chọn nội dung học ở thanh bên trái nhé!</div></div>',
      unsafe_allow_html=True
    )

else:
    with st.spinner("Đang tải nội dung..."):
        time.sleep(1)
        exec(open(st.session_state.page, "r", encoding="utf-8").read())
