import streamlit as st
import numpy as np
from PIL import Image
import cv2


L = 256
# Cac ham cua chuong 3
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
# Lam giao dien
st.write('# Thị Giác Máy - Chương 3')
col1, col2 = st.columns(2)  # Create two columns for displaying frames
imgin_frame = col1.empty()  # Container for original frame
imgout_frame = col2.empty()  # Container for annotated frame

chuong3_muc = st.sidebar.radio("Các mục chương 3", ("Negative", "Hist Equal","Hist Stat","Logarit","Power","Local Hist","Sharp","Gradient","Smooth Box","Smooth Gauss","Median Filter"))
img_file_buffer = st.sidebar.file_uploader("Upload an image", type=["bmp", "png", "jpg", "jpeg","tif"])

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    imgin_frame.image(image)
    if st.button('Process'):
        imgin = np.array(image)
        if chuong3_muc == "Negative":
             imgout = Negative(imgin)
             imgout_frame.image(imgout)
        elif chuong3_muc == "Hist Equal":
             imgout = cv2.equalizeHist(imgin)
             imgout_frame.image(imgout)
        elif chuong3_muc == "Hist Stat":
             imgout = HistStat(imgin)
             imgout_frame.image(imgout)
        elif chuong3_muc == "Logarit":
             imgout = Logarit(imgin)
             imgout_frame.image(imgout)
        elif chuong3_muc == "Power":
             imgout = Power(imgin)
             imgout_frame.image(imgout)
        elif chuong3_muc == "Local Hist":
             imgout = LocalHist(imgin)
             imgout_frame.image(imgout)
        elif chuong3_muc == "Sharp":
             imgout = Sharp(imgin)
             imgout_frame.image(imgout)
        elif chuong3_muc == "Gradient":
             imgout = Gradient(imgin)
             imgout_frame.image(imgout)
        elif chuong3_muc == "Smooth Box":
             imgout = cv2.boxFilter(imgin, cv2.CV_8UC1, (21,21))
             imgout_frame.image(imgout)
        elif chuong3_muc == "Smooth Gauss":
             imgout = cv2.GaussianBlur(imgin,  (43,43), 7.0)
             imgout_frame.image(imgout)
        elif chuong3_muc == "Median Filter":
             imgout =  cv2.medianBlur(imgin, 9)
             imgout_frame.image(imgout)

# CSS để đổi màu nền
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #e6f7ff; /* xanh nhạt */
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)