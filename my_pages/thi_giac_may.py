import streamlit as st
import numpy as np
from PIL import Image
import cv2
import base64
import os
# === Cấu hình chung ===
def get_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
video_path = os.path.join(os.path.dirname(__file__), "resources/videos/background_4.mp4")
video_b64 = get_base64(video_path) if os.path.exists(video_path) else ""
css = f"""
<style>
  /* Gradient sidebar */
  [data-testid="stSidebar"], [data-testid="stSidebarNav"] {{
    background: linear-gradient(135deg, #99ccff, #66b2ff, #3399ff) !important;
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

L = 256

# === Chương 3 – Spatial Domain ===

def HistEqual(imgin): return cv2.equalizeHist(imgin)
def SmoothBox(imgin): return cv2.boxFilter(imgin, cv2.CV_8UC1, (21,21))
def SmoothGauss(imgin): return cv2.GaussianBlur(imgin,  (43,43), 7.0)
def MedianFilter(imgin): return cv2.medianBlur(imgin, 9)
def Negative(imgin):
    M,N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    
    for x in range(0, M):
         for y in range(0, N):
              r = imgin[x,y]
              s = L - 1 - r
              imgout[x,y] = np.uint8(s)         
    # imgout[:, :, 0] = np.uint8(255) 
    # Anh mau cua opencv la BGR
    # Anh mau cua pillow la RGB
    return imgout

def HistStat(imgin):
     M,N  = imgin.shape
     imgout = np.zeros((M,N), np.uint8)

     mean, stddev = cv2.meanStdDev(imgin)
     mG = mean[0,0]
     sigmaG = stddev[0,0]
     
     m = 3
     n = 3
     a = m // 2
     b = m // 2

     C = 22.8
     k0 = 0.0; k1 = 0.1
     k2 = 0.0; k3 = 0.1
     for x in range(a, M-a):
          for y in range(b, M-b):
               w = imgin[x-a:x+a+1, y-b:y+b+1]
               mean, stddev = cv2.meanStdDev(w)
               msxy = mean[0,0]
               sigmasxy = stddev[0,0]
               if (k0*mG <= msxy <= k1*mG) and (k2*sigmaG <= sigmasxy <= k3*sigmaG):
                    imgout[x,y] = np.uint8(C*imgin[x,y])
               else:
                    imgout[x,y] = imgin[x,y]
     return imgout
def Logarit(imgin):
    M,N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)

    c = (L - 1)/np.log(1.0*L)

    for x in range(0, M):
         for y in range(0, N):
              r = imgin[x,y]
              if r == 0:
                   r = 1
              s = c*np.log(1.0 + r)
              imgout[x,y] = np.uint8(s) 

    return imgout

def Power(imgin):
    M,N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)

    gamma = 5.0
    c = np.power(L - 1.0, 1 - gamma)

    for x in range(0, M):
         for y in range(0, N):
              r = imgin[x,y]
              if r == 0:
                   r = 1
              s = c*np.power(1.0* r, gamma)
              imgout[x,y] = np.uint8(s) 

    return imgout
def LocalHist(imgin):
     M,N  = imgin.shape
     imgout = np.zeros((M,N), np.uint8)
     m = 3
     n = 3
     a = m // 2
     b = m // 2
     for x in range(a, M-a):
          for y in range(b, M-b):
               w = imgin[x-a:x+a+1, y-b:y+b+1]
               w = cv2.equalizeHist(w)
               imgout[x,y] = w[a,b]
     return imgout

def Sharp(imgin):
     w = np.array([[1,1,1],[1,-8,1],[1,1,1]], np.float32)
     # Laplacian la dao ham cap 2
     Laplacian = cv2.filter2D(imgin, cv2.CV_32FC1, w)
     imgout = imgin - Laplacian
     imgout = np.clip(imgout,  0, L-1)
     imgout = imgout.astype(np.uint8)
     return imgout 
def Gradient(imgin):
     #Gradient la dao ham cap 1 cua anh
     Sobel_x =  np.array([[-1,-2,-1],[0,0,0],[1,2,1]], np.float32)
     Sobel_y =  np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)
     gx = cv2.filter2D(imgin, cv2.CV_32FC1, Sobel_x)
     gy = cv2.filter2D(imgin, cv2.CV_32FC1, Sobel_y)
     imgout = abs(gx) + abs(gy)
     
     imgout = np.clip(imgout,  0, L-1)
     imgout = imgout.astype(np.uint8)
     return imgout

# === Chương 4 – Frequency Domain ===
def FrequencyFiltering(imgin,H):
    # Ta khong can mo rong anh  kich thuoc PxQ
    f = imgin.astype(np.float32)
    # Buoc 1: bien doi Fourier
    F = np.fft.fft2(f)
    # Buoc 2: Doi F vao chinh giua anh
    F = np.fft.fftshift(F)
    # Buoc 3: Nhan F voi H
    G = F*H
    # Buoc 4: Doi F ve vi tri ban dau
    G = np.fft.ifftshift(G)
    # Buoc 5: Tinh IDFT
    g = np.fft.ifft2(G)
    # Buoc 6: lay phan thuc
    gR = np.clip(g.real, 0 , L-1)
    imgout = gR.astype(np.uint8)
    return imgout

def  Spectrum(imgin):
    # Ta khong can mo rong anh  kich thuoc PxQ
    f = imgin.astype(np.float32)/(L-1)
    # Buoc 1: bien doi Fourier
    F = np.fft.fft2(f)
    # Buoc 2: Doi F vao chinh giua anh
    F = np.fft.fftshift(F)
    # Buoc 3: Tinh pho
    S = np.sqrt(F.real**2 + F.imag**2)
    S = np.clip(S, 0, L-1)
    imgout = S.astype(np.uint8) 
    return imgout

def CreateMoireFiler(M, N):
     H = np.ones((M,N), np.complex64)
     H.imag = 0
     u1, v1 = 44, 55
     u2, v2 = 85, 55
     u3, v3 = 41, 111
     u4, v4 = 81, 111

     u5, v5 = M - 44, N - 55
     u6, v6 = M - 85, N - 55
     u7, v7 = M - 41, N - 111
     u8, v8 = M - 81, N - 111

     D0 = 10 

     for u in range(0, M):
         for v in range(0, N):
             # u1, v1
             D = np.sqrt((1.0*u-u1)**2 + (1.0*v-v1)**2)
             if D <=D0:
                 H.real[u,v] = 0.0
             # u2, v2
             D = np.sqrt((1.0*u-u2)**2 + (1.0*v-v2)**2)
             if D <=D0:
                 H.real[u,v] = 0.0
             # u3, v3
             D = np.sqrt((1.0*u-u3)**2 + (1.0*v-v3)**2)
             if D <=D0:
                 H.real[u,v] = 0.0
             # u4, v4
             D = np.sqrt((1.0*u-u4)**2 + (1.0*v-v4)**2)
             if D <=D0:
                H.real[u,v] = 0.0
             # u5, v5
             D = np.sqrt((1.0*u-u5)**2 + (1.0*v-v5)**2)
             if D <=D0:
                 H.real[u,v] = 0.0
             # u6, v6
             D = np.sqrt((1.0*u-u6)**2 + (1.0*v-v6)**2)
             if D <=D0:
                 H.real[u,v] = 0.0
              # u7, v7
             D = np.sqrt((1.0*u-u7)**2 + (1.0*v-v7)**2)
             if D <=D0:
                 H.real[u,v] = 0.0
              # u8, v8
             D = np.sqrt((1.0*u-u8)**2 + (1.0*v-v8)**2)
             if D <=D0:
                 H.real[u,v] = 0.0
     
     return H

def CreateInterferenceFiler(M, N):
     H = np.ones((M,N), np.complex64)
     H.imag= 0
   

     D0 = 7 
     D1 = 7 
     for u in range(0, M):
         for v in range(0, N):
             if u not in range(M//2 - D0, M//2 + 1 + D0):
                D = abs(v-N//2)
                if D <= D1:
                     H.real[u,v] = 0.0
     return H

def  CreateMotionFilter(M,N):
    H = np.zeros((M,N), np.complex64)
    a = 0.1
    b = 0.1
    T = 1.0
    phi_prev = 0.0
    for u in range(0,M):
        for v in range(0,N):
            phi = np.pi*((u-M//2)*a + (v-N//2)*b)
            if abs(phi) <1.0e-6:
                phi = phi_prev
            RE = T*np.sin(phi)/phi*np.cos(phi)
            IM = -T*np.sin(phi)/phi*np.sin(phi)
            H.real[u,v] = RE
            H.imag[u,v] = IM
            phi_prev = phi
    return H

def  CreateDemotionFilter(M,N):
    H = np.zeros((M,N), np.complex64)
    a = 0.1
    b = 0.1
    T = 1.0
    phi_prev = 0.0
    for u in range(0,M):
        for v in range(0,N):
            phi = np.pi*((u-M//2)*a + (v-N//2)*b)
            mau_so = np.sin(phi)
            if abs(mau_so) <1.0e-6:
                phi = phi_prev

            RE = phi/(T*np.sin(phi)) * np.cos(phi)
            IM = phi/T
            H.real[u,v] = RE
            H.imag[u,v] = IM
            phi_prev = phi
    return H

def RemoveMoire(imgin):
    M, N = imgin.shape
    H = CreateMoireFiler(M,N)
    imgout = FrequencyFiltering(imgin, H)
    return imgout

def RemoveInterference(imgin):
    M, N = imgin.shape
    H = CreateInterferenceFiler(M,N)
    imgout = FrequencyFiltering(imgin, H)
    return imgout

def CreateMotion(imgin):
    M, N = imgin.shape
    H = CreateMotionFilter(M,N)
    imgout = FrequencyFiltering(imgin, H)
    return imgout

def Demotion(imgin):
    M, N = imgin.shape
    H = CreateDemotionFilter(M,N)
    imgout = FrequencyFiltering(imgin, H)
    return imgout


def DemotionNoise(imgin):
    tmp=cv2.medianBlur(imgin,7)
    return Demotion(tmp)

# === Chương 9 – Morphology & Components ===
def ensure_uint8(img): return cv2.convertScaleAbs(img)
def Erosion(imgin):
    imgin = ensure_uint8(imgin)
    #Bào mòn đối tượng trong ảnh 
    w = cv2.getStructuringElement(cv2.MORPH_RECT,(45,45))
    imgout = cv2.erode(imgin,w)
    return imgout
def Dilation(imgin): 
    imgin = ensure_uint8(imgin)
    w = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    imgout = cv2.dilate(imgin,w)
    return imgout

def Boundary(imgin): 
    imgin = ensure_uint8(imgin)
    w = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    temp = cv2.erode(imgin, w)
    imgout = imgin - temp 
    return imgout

def Contour(imgin):
    imgin = ensure_uint8(imgin) 
    # Luu y : Contour chi dung cho anh nhi phan
    imgout = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(imgin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    n = len(contour)
    for i in range(0, n-1):
        x1 = contour[i,0,0]
        y1 = contour[i,0,1]       
        x2 = contour[i+1,0,0]
        y2 = contour[i+1,0,1]
        cv2.line(imgout, (x1,y1), (x2,y2), (0,0,255), 2)
        
    x1 = contour[n-1,0,0]
    y1 = contour[n-1,0,1]       
    x2 = contour[0,0,0]
    y2 = contour[0,0,1]
    cv2.line(imgout, (x1,y1), (x2,y2), (0,0,255), 2)
    return imgout     

def ConvexHull(imgin):
    # Luu y : Contour chi dung cho anh nhi phan
    # Tinh convex hull phai qua 2 buoc:
    # Buoc 1: Tinh contour 
    # Buoc 2: Tinh convex hull 
    imgin = ensure_uint8(imgin)
    imgout = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(imgin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    hull = cv2.convexHull(contour)
    n = len(hull)
    for i in range(0, n-1):
        x1 = hull[i,0,0]
        y1 = hull[i,0,1]       
        x2 = hull[i+1,0,0]
        y2 = hull[i+1,0,1]
        cv2.line(imgout, (x1,y1), (x2,y2), (0,0,255), 2)      
    x1 = hull[n-1,0,0]
    y1 = hull[n-1,0,1]       
    x2 = hull[0,0,0]
    y2 = hull[0,0,1]
    cv2.line(imgout, (x1,y1), (x2,y2), (0,0,255), 2)
    return imgout

def DefectDetect(imgin):
    # Phai qua 3 buoc 
    # Buoc 1: Tinh Contour 
    imgin = ensure_uint8(imgin)
    imgout = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(imgin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    # Buoc 2: Tinh bao loi cua contour dang point
    p = cv2.convexHull(contour, returnPoints = False)
    n = len(p)
    for i in range(0,n-1):
        vi_tri_1 = p[i,0]
        vi_tri_2 = p[i+1,0]     
        x1 = contour[vi_tri_1, 0, 0]
        y1 = contour[vi_tri_1, 0, 1]       
        x2 = contour[vi_tri_2, 0, 0]
        y2 = contour[vi_tri_2, 0, 1]
        cv2.line(imgout, (x1,y1), (x2,y2), (0,0,255), 2) 
        
    vi_tri_1 = p[n-1,0]
    vi_tri_2 = p[0,0]     
    x1 = contour[vi_tri_1, 0, 0]
    y1 = contour[vi_tri_1, 0, 1]       
    x2 = contour[vi_tri_2, 0, 0]
    y2 = contour[vi_tri_2, 0, 1]
    cv2.line(imgout, (x1,y1), (x2,y2), (0,0,255), 2) 
    
    #Buoc 3: Tinh cho khuyet
    defects = cv2.convexityDefects(contour, p)
    nguong_do_sau = np.max(defects[:,:,3])
    n = len(defects)
    for i in range (0,n):
        do_sau = defects[i,0,3]
        if do_sau > nguong_do_sau:           
            vi_tri_khuyet = defects[i,0,2]
            x = contour[vi_tri_khuyet, 0, 0]
            y = contour[vi_tri_khuyet, 0, 1]
            cv2.circle(imgout, (x,y), 5, (0,255,0), -1)      
    return imgout

def HoleFill(imgin):
    imgin = ensure_uint8(imgin)
    # Anh mau cua opencv là BGR
    imgout = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)
    cv2.floodFill(imgout, None, (104,295), (0,0,255))
    return imgout 

def ConnectedComponents(imgin):
    nguong = 200
    imgin = ensure_uint8(imgin)
    _, temp = cv2.threshold(imgin, nguong, 255, cv2.THRESH_BINARY)  
    imgout  = cv2.medianBlur(temp,7)
    dem, label = cv2.connectedComponents(imgout , None, )
    a = np.zeros(dem, np.int32)
    M,N = label.shape
    for x in range (0,M):
        for y in range(0,N):
            r = label[x,y]
            if r >0:
                a[r] = a[r] + 1
    s = 'co %d thanh phan lỉen thong' % (dem -1)
    cv2.putText(imgout, s, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
    for r in range(1, dem):
        s ='%3d %5d' % (r ,  a[r])
        cv2.putText(imgout, s, (10, (r+1)*15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
    return imgout

def RemoveSmallRice(imgin):
    # lam dam bong hat gao 
    # 81 la kich thuoc lon nhat cua hat gao
    imgin = ensure_uint8(imgin)
    w = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81,81)) 
    temp = cv2.morphologyEx(imgin, cv2.MORPH_TOPHAT, w)
    nguong = 120 
    _,temp = cv2.threshold(temp, nguong, L-1, cv2.THRESH_BINARY |cv2.THRESH_OTSU)
    dem, label = cv2.connectedComponents(temp , None, )
    a = np.zeros(dem, np.int32)
    M,N = label.shape
    for x in range (0,M):
        for y in range(0,N):
            r = label[x,y]
            if r >0:
                a[r] = a[r] + 1
    max_value = np.max(a)
    imgout = np.zeros((M,N), np.uint8)
    for x in range (0,M):
        for y in range(0,N):
            r = label[x,y]
            if r >0:
                if a[r] >0.7*max_value:
                 imgout[x,y] = L-1
    return imgout 
# === Giao diện chính ===
st.sidebar.header("Chọn chương")
chap=st.sidebar.radio("",['Chương 3','Chương 4','Chương 9'])
file=st.sidebar.file_uploader("Upload ảnh",type=["png","jpg","jpeg","bmp","tif"])


if file:
    img=np.array(Image.open(file).convert('L'))
    
    st.markdown(f'<h1 style="text-align:left; color: #CC0000;"> Thị Giác Máy –  {chap}</h1>', unsafe_allow_html=True)
    c1,c2=st.columns(2)
    c1.image(img,caption="Ảnh gốc",use_container_width=True)
    c1.markdown("<div class='img-caption'>Ảnh gốc</div>",unsafe_allow_html=True)

    if chap=='Chương 3':
        func=st.sidebar.selectbox("Chọn hàm Ch3",["Negative","HistEqual","HistStat","Logarit","Power","LocalHist","Sharp","Gradient","SmoothBox","SmoothGauss","MedianFilter"])
    elif chap=='Chương 4':
        func=st.sidebar.selectbox("Chọn hàm Ch4",["Spectrum","RemoveMoire","RemoveInterference","CreateMotion","Demotion","DemotionNoise"])
    else:
        func=st.sidebar.selectbox("Chọn hàm Ch9",["Erosion","Dilation","Boundary","Contour","ConvexHull","DefectDetect","HoleFill","ConnectedComponents","RemoveSmallRice"])

    if st.button("Process"):
        res=globals()[func](img)
        c2.image(res,caption="Ảnh xử lý",use_container_width=True)
        c2.markdown("<div class='img-caption'>Ảnh xử lý</div>",unsafe_allow_html=True)
