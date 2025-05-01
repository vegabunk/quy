import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
import cv2
import tempfile
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO

# st.set_page_config(layout="wide", page_title="Style Transfer App", page_icon="🎨")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean   = np.array([0.485, 0.456, 0.406])
std    = np.array([0.229, 0.224, 0.225])  

    
class TransformerNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            ConvBlock(3, 32, kernel_size=9, stride=1),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            ConvBlock(64, 128, kernel_size=3, stride=2),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ConvBlock(128, 64, kernel_size=3, upsample=True),
            ConvBlock(64, 32, kernel_size=3, upsample=True),
            ConvBlock(32, 3, kernel_size=9, stride=1, normalize=False, relu=False),
        )

    def forward(self, x):
        return self.model(x)
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=True),
            ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=False),
        )

    def forward(self, x):
        return self.block(x) + x
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, upsample=False, normalize=True, relu=True):
        super(ConvBlock, self).__init__()
        
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2), 
            nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if normalize else None
        self.relu = relu

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        x = self.block(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu:
            x = F.relu(x)
        return x
    

def test_transform(image_size=None):
    resize    = [transforms.Resize(image_size)] if image_size else []
    transform = transforms.Compose(resize + [transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transform
def denormalize(tensors):
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return tensors


def inference(content_image, checkpoint_model):
    # Load model
    transformer = TransformerNet().to(device)
    # transformer.load_state_dict(torch.load(checkpoint_model, weights_only=True))
    transformer.load_state_dict(torch.load(checkpoint_model, map_location=device))
    transformer.eval()
    
    # Inference
    image_tensor = Variable(test_transform()(Image.open(content_image))).to(device)
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        stylized_image = denormalize(transformer(image_tensor)).cpu()
    
    stylized_image_np = stylized_image.squeeze(0).permute(1, 2, 0).numpy()
    stylized_image_np = np.clip(stylized_image_np, 0, 1)
    return stylized_image_np


if __name__ == '__main__':
    st.title("🎨 Style Transfer App")
    st.markdown(
        """
        Welcome to the **Style Transfer App!**  
        Transform your photo or video into stunning artworks inspired by famous artistic styles. Simply upload your image/video, choose a style, and let AI do the rest!
        """
    )
    st.divider()


    sub1, space1, sub2, space2, sub3 = st.columns([10,0.2,10,0.2,10])
    
    with sub1:
        st.subheader('**1. Upload Your Image**')
        st.write('')
        
        uploaded = st.file_uploader(' ', type=['png', 'jpg', 'jpeg', 'tif'], label_visibility='collapsed')
        if uploaded is not None:
            content      = Image.open(uploaded)
            st.image(content, use_container_width=True)
                
    with sub2:
        st.subheader('**2. Choose Your Favourite Style**')
        st.write('')
        
        col1, col2 = st.columns([2,1])
        with col1:
            chose_style = st.selectbox(
                ' ', options=['Post Impressionism', 'Cubism', 'Abstract Expressionism', 'Digital Painting'], 
                label_visibility='collapsed'
            )
        with col2:
            style_btn = st.button('Start Stylizing!', type='secondary', use_container_width=True)
        
        style1 = Image.open('resources/styles/style_1-Post_Impressionism.png')
        style2 = Image.open('resources/styles/style_2-Cubism.jpg')
        style3 = Image.open('resources/styles/style_3-Abstract_Expressionism.png')
        style4 = Image.open('resources/styles/style_4-Digital_Painting.jpg')
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(style1, caption='Post Impressionism', use_container_width=True)
            st.image(style3, caption='Abstract Expressionism', use_container_width=True)
        with col2:
            st.image(style2, caption='Cubism', use_container_width=True)
            st.image(style4, caption='Digital Painting', use_container_width=True)
        
    with sub3:
        st.subheader('**3. Your Styled Image**')
        st.write('')
        
        style_dict = {
            'Post Impressionism'    : 'weights/best_model-Post_Impressionism.pth',
            'Cubism'                : 'weights/best_model-Cubism.pth',
            'Abstract Expressionism': 'weights/best_model-Abstract_Expressionism.pth',
            'Digital Painting'      : 'weights/best_model-Digital_Painting.pth'
        }
            
        if uploaded is not None and style_btn:
            result_img = inference(
                content_image=uploaded,
                checkpoint_model=style_dict[chose_style]
            )
            st.image(result_img, use_container_width=True)
            
            buf = BytesIO()
            Image.fromarray((result_img * 255).astype(np.uint8)).save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="📥 Download Stylized Image",
                data=byte_im,
                file_name='styled_image.png',
                mime="image/png",
                type="primary"
            )
