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
def Negative(img): return L - 1 - img
def HistEqual(img): return cv2.equalizeHist(img)
def HistStat(img):
    M,N=img.shape; out=img.copy()
    mG,sG=cv2.meanStdDev(img)[0][0],cv2.meanStdDev(img)[1][0]
    C,k1,k3=22.8,0.1,0.1
    for x in range(1,M-1):
        for y in range(1,N-1):
            w=img[x-1:x+2,y-1:y+2]
            m,s=cv2.meanStdDev(w)[0][0],cv2.meanStdDev(w)[1][0]
            if m<=k1*mG and s<=k3*sG: out[x,y]=np.uint8(C*img[x,y])
    return out
def Logarit(img): return np.uint8((L-1)/np.log(L)*np.log(1+img))
def Power(img,gamma=0.5): return np.uint8((L-1)*((img/(L-1))**gamma))
def LocalHist(img):
    M,N=img.shape; out=img.copy()
    for x in range(1,M-1):
        for y in range(1,N-1): out[x,y]=cv2.equalizeHist(img[x-1:x+2,y-1:y+2])[1,1]
    return out
def Sharp(img): return cv2.convertScaleAbs(img - cv2.filter2D(img,cv2.CV_32F,np.array([[1,1,1],[1,-8,1],[1,1,1]],np.float32)))
def Gradient(img): return cv2.convertScaleAbs(cv2.magnitude(cv2.Sobel(img,cv2.CV_32F,1,0,3),cv2.Sobel(img,cv2.CV_32F,0,1,3)))
def SmoothBox(img): return cv2.boxFilter(img,cv2.CV_8UC1,(21,21))
def SmoothGauss(img): return cv2.GaussianBlur(img,(43,43),7.0)
def MedianFilter(img): return cv2.medianBlur(img,9)

# === Chương 4 – Frequency Domain ===
def freq_filter(img,H):
    F=np.fft.fftshift(np.fft.fft2(img.astype(np.float32)))
    g=np.fft.ifft2(np.fft.ifftshift(F*H))
    return cv2.convertScaleAbs(np.real(g))
def Spectrum(img):
    f=np.fft.fftshift(np.fft.fft2(img.astype(np.float32)/(L-1)))
    mag=np.log(1+np.abs(f))
    return cv2.convertScaleAbs(mag/mag.max()*(L-1))
def CreateNotchFilter(shape,centers,D0=10):
    M,N=shape;H=np.ones((M,N),np.float32)
    for u0,v0 in centers:
        for u in range(M):
            for v in range(N):
                if np.hypot(u-u0,v-v0)<=D0: H[u,v]=0
    return H
def RemoveMoire(img):
    M,N=img.shape; pts=[(44,55),(85,55),(41,111),(81,111)]
    centers=pts+[(M-x,y) for x,y in pts]+[(x,N-y) for x,y in pts]
    return freq_filter(img,CreateNotchFilter((M,N),centers))
def CreateInterferenceFilter(M,N,D0=7,D1=7):
    H=np.ones((M,N),np.float32)
    for u in range(M):
        for v in range(N):
            if (u< M//2-D0 or u> M//2+D0) and abs(v-N//2)<=D1: H[u,v]=0
    return H
def RemoveInterference(img): return freq_filter(img,CreateInterferenceFilter(*img.shape))
def CreateMotionFilter(M,N,a=0.1,b=0.1,T=1.0):
    H=np.zeros((M,N),np.complex64); prev=0
    for u in range(M):
        for v in range(N):
            phi=np.pi*((u-M//2)*a+(v-N//2)*b)
            if abs(np.sin(phi))<1e-6: phi=prev
            H.real[u,v]=T*np.sin(phi)/phi*np.cos(phi)
            H.imag[u,v]=-T*np.sin(phi)/phi*np.sin(phi)
            prev=phi
    return H
def CreateDemotionFilter(M,N,a=0.1,b=0.1,T=1.0):
    H=np.zeros((M,N),np.complex64); prev=0
    for u in range(M):
        for v in range(N):
            phi=np.pi*((u-M//2)*a+(v-N//2)*b)
            if abs(np.sin(phi))<1e-6: phi=prev
            H.real[u,v]=phi/(T*np.sin(phi))*np.cos(phi)
            H.imag[u,v]=phi/T
            prev=phi
    return H
def CreateMotion(img): return freq_filter(img,CreateMotionFilter(*img.shape))
def Demotion(img): return freq_filter(img,CreateDemotionFilter(*img.shape))

def DemotionNoise(img):
    tmp=cv2.medianBlur(img,7)
    return Demotion(tmp)

# === Chương 9 – Morphology & Components ===
def ensure_uint8(img): return cv2.convertScaleAbs(img)
def Erosion(img): return cv2.erode(img,cv2.getStructuringElement(cv2.MORPH_RECT,(45,45)))
def Dilation(img): return cv2.dilate(img,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
def Boundary(img): return ensure_uint8(img) - Erosion(img)
def Contour(img):
    img0=ensure_uint8(img); out=cv2.cvtColor(img0,cv2.COLOR_GRAY2BGR)
    cnt,_=cv2.findContours(img0,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out,cnt,-1,(0,0,255),1); return out
def ConvexHull(img):
    img0=ensure_uint8(img); out=cv2.cvtColor(img0,cv2.COLOR_GRAY2BGR)
    cnt,_=cv2.findContours(img0,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if cnt:
        hull=cv2.convexHull(cnt[0]); cv2.drawContours(out,[hull],-1,(0,255,0),1)
    return out
def DefectDetect(img):
    img0=ensure_uint8(img); out=cv2.cvtColor(img0,cv2.COLOR_GRAY2BGR)
    cnt,_=cv2.findContours(img0,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if cnt:
        p=cv2.convexHull(cnt[0],returnPoints=False)
        defects=cv2.convexityDefects(cnt[0],p) if p is not None else None
        if defects is not None:
            thresh=defects[:,:,3].max()
            for d in defects[:,0]:
                if d[3]>thresh:
                    x,y=cnt[0][d[2],0]; cv2.circle(out,(x,y),5,(255,0,0),-1)
    return out
def HoleFill(img):
    img0=ensure_uint8(img); out=cv2.cvtColor(img0,cv2.COLOR_GRAY2BGR)
    h,w=img0.shape; mask=np.zeros((h+2,w+2),np.uint8)
    cv2.floodFill(out,mask,(0,0),(0,0,255)); return out
def ConnectedComponents(img):
    img0=ensure_uint8(img); _,b=cv2.threshold(img0,200,255,cv2.THRESH_BINARY)
    num,labels=cv2.connectedComponents(b); out=cv2.cvtColor(b,cv2.COLOR_GRAY2BGR)
    for i in range(1,num): y,x=np.mean(np.where(labels==i),axis=1).astype(int); cv2.putText(out,str(i),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0))
    return out
def RemoveSmallRice(img):
    img0=ensure_uint8(img); w=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(81,81))
    th=cv2.morphologyEx(img0,cv2.MORPH_TOPHAT,w); _,th2=cv2.threshold(th,120,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    num,labels=cv2.connectedComponents(th2); out=np.zeros_like(img0)
    counts=np.bincount(labels.flatten()); maxc=counts.max()
    for i in range(1,num):
        if counts[i]>0.7*maxc: out[labels==i]=255
    return out

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
