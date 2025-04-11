import streamlit as st
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO
import io
from ultralytics import YOLO

model = YOLO('best.pt')

def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image

with st.sidebar:
    st.header("Weed Detector")

    option = st.selectbox(
        'Select a weed to detect',
        ('Ragweed', 'Palmer Amaranth'),
        index=None)

if(option):
    st.write('### You have selected', option)
    st.write('#### Please upload an image')
    uploadFile = st.file_uploader("Choose a image file", type=['jpg', 'png'])

    if uploadFile is not None:
        img = load_image(uploadFile)
        results = model(img)
        output_img = results[0].plot(conf=False)

        
        st.write("Original Image")
        st.image(img,width=500)
        
        st.write("Output Image")
        st.image(output_img,width=500)
        
        buf = BytesIO()
        download = Image.fromarray(output_img)

        download.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        st.download_button(label='Download output image',
                        data = byte_im,
                        file_name='output.jpg',
                        mime='image/jpg')

