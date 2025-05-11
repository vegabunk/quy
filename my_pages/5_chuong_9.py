import streamlit as st
import numpy as np
from PIL import Image
import cv2
L = 256

def ensure_uint8(img):
    if img.dtype == bool:
        return (img.astype(np.uint8)) * 255
    return img.astype(np.uint8)

def Erosion(imgin):
    imgin = ensure_uint8(imgin)
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (45,45))
    return cv2.erode(imgin, w)

def Dilation(imgin):
    imgin = ensure_uint8(imgin)
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    return cv2.dilate(imgin, w)

def Boundary(imgin):
    imgin = ensure_uint8(imgin)
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    temp = cv2.erode(imgin, w)
    return imgin - temp

def Contour(imgin):
    img = ensure_uint8(imgin)
    imgout = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return imgout
    contour = contours[0]
    for i in range(len(contour)):
        x1,y1 = contour[i,0]
        x2,y2 = contour[(i+1)%len(contour),0]
        cv2.line(imgout,(x1,y1),(x2,y2),(0,0,255),2)
    return imgout

def ConvexHull(imgin):
    img = ensure_uint8(imgin)
    imgout = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return imgout
    hull = cv2.convexHull(contours[0])
    for i in range(len(hull)):
        x1,y1 = hull[i,0]
        x2,y2 = hull[(i+1)%len(hull),0]
        cv2.line(imgout,(x1,y1),(x2,y2),(0,0,255),2)
    return imgout

def DefectDetect(imgin):
    imgin = ensure_uint8(imgin)
    imgout = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return imgout
    contour = contours[0]
    p = cv2.convexHull(contour, returnPoints=False)
    if p is None: return imgout
    defects = cv2.convexityDefects(contour, p)
    if defects is None: return imgout
    thresh = np.max(defects[:,:,3])
    for d in defects[:,0]:
        if d[3] > thresh:
            x,y = contour[d[2],0]
            cv2.circle(imgout,(x,y),5,(0,255,0),-1)
    return imgout

def HoleFill(imgin):
    img = ensure_uint8(imgin)
    imgout = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(imgout, mask, (0,0), (0,0,255))
    return imgout

def ConnectedComponents(imgin):
    img = ensure_uint8(imgin)
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    blur = cv2.medianBlur(binary,7)
    num, labels = cv2.connectedComponents(blur)
    out = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
    counts = np.bincount(labels.flatten())
    for i in range(1, num):
        mask = (labels==i)
        coords = np.column_stack(np.where(mask))
        y,x = coords[0]
        cv2.putText(out, f"{i}: {counts[i]}",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0))
    return out

def RemoveSmallRice(imgin):
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

# CSS nền
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{background-color:#e6f7ff;}
</style>
"""

# UI
st.markdown(page_bg_img, unsafe_allow_html=True)
st.write('# Thị Giác Máy - Chương 9')
col1,col2 = st.columns(2)
in_frame,out_frame = col1.empty(), col2.empty()
options = ["All","Erosion","Dilation","Boundary","Contour","ConvexHull","DefectDetect","HoleFill","ConnectedComponents","RemoveSmallRice"]
choice = st.sidebar.selectbox("Chọn chức năng Chương 9", options)
buf = st.sidebar.file_uploader("Ảnh gốc", type=["bmp","png","jpg","jpeg","tif"])

if buf:
    image = Image.open(buf).convert('L')
    arr = np.array(image)
    in_frame.image(image, caption='Input - Ảnh gốc')
    if st.button('Process'):
        if choice == 'All':
            for name in options[1:]:
                res = globals()[name](arr)
                st.subheader(name)
                st.image(res, caption=f'Output - {name}')
        else:
            res = globals()[choice](arr)
            out_frame.image(res, caption='Output - Ảnh xử lý')
