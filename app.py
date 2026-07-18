# pip install pillow

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow import keras 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model



st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #4B215E;
    color:white;
            
}       

</style>
""", unsafe_allow_html=True)


st.markdown(""" <style>
        .stApp{
        background: linear-gradient(to top, #D9ADED, #FFFFFF);
            }
            
            
  div.stButton > button {
    background-color: #4B215E;
    color: white;
    font-size: 20px;
    font-weight: bold;
    width: 145px;
    height: 45px;
    border-radius: 10px;
    border: 2px double white;
    margin-top : 10px;
    margin-bottom : 10px;
    
    
    
}
            </style>
""",unsafe_allow_html=True)




import os
import gdown
from tensorflow.keras.models import load_model

MODEL_PATH = "mask_final.keras"

if not os.path.exists(MODEL_PATH):
    file_id = "1x6flwcce5FSeFASja6tBd4yjlkN956zw"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

st.title("😷 Face Mask Detection")



st.sidebar.markdown(""" 
            <h1  style="color:#D9ADED;" > <u>About </u></h1>""",unsafe_allow_html=True)

st.sidebar.divider()

st.sidebar.markdown(""" <h5 style="color:#FFFFFF; margin-bottom:10px; padding:0px;">🧠 Model : CNN</h5>""",unsafe_allow_html=True)

st.sidebar.markdown(""" <h5 style="color:#FFFFFF; margin-bottom:10px; padding:0px;">📂 Dataset : Face Mask Dataset</h5>""",unsafe_allow_html=True)
st.sidebar.markdown(""" <h5 style="color:#FFFFFF; margin-bottom:10px; padding:0px;">📌 Classes : <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   • Mask <br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    • No Mask
            </h5>""",unsafe_allow_html=True)

st.sidebar.markdown(""" <h5 style="color:#FFFFFF; margin-bottom:10px; padding:0px;">🖼️ Image Size : 128 x 128</h5>""",unsafe_allow_html=True)


st.sidebar.markdown(""" <h5 style="color:#FFFFFF; margin-bottom:10px; padding:0px;">⚙️ Framework : TensorFlow</h5>""",unsafe_allow_html=True)


st.sidebar.markdown(""" <h5 style="color:#FFFFFF;   padding:0px;">🎯 Accuracy : 91%</h5>""",unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.markdown(""" <h3 style="color:#D9ADED; margin-bottom:5px;">Developed by :</h3>""",unsafe_allow_html=True)
st.sidebar.markdown(""" <h5 style="color:#FFFFFF; padding:0px;">&nbsp;&nbsp;&nbsp;Vishakha Nikam </h5>""",unsafe_allow_html=True)





st.divider()

st.markdown(""" 
            <h5 style="margin-top:20px; color:#6B3882; "> Detect whether a person is wearing a mask.</h5>""",unsafe_allow_html=True)

st.divider()
st.markdown(""" 
            <h3 style="margin-top:20px; color:#000000; text-align:center; "><u> Choose an Input Method</u></h3>""",unsafe_allow_html=True)
st.divider()
st.markdown(""" 
            <h5 style="margin-top:20px; color:#6B3882;  "> 1. Upload an Image</h5>""",unsafe_allow_html=True)



uploaded_file = st.file_uploader("",type = ["jpg","jpeg","png"])

if uploaded_file is not None:
    st.divider()
    
    st.markdown(""" 
            <h5 style="margin-top:20px; color:#6B3882;  "> Image Preview</h5>""",unsafe_allow_html=True)

    st.image(uploaded_file)
    

    img = Image.open(uploaded_file)
    
    img = img.resize((128,128))  # resize image to 128 , 128 as this size we ussed while training

    img_array = image.img_to_array(img) / 255.0 #create array of uploaded image and normalise it
   
    img_array = np.expand_dims(img_array,axis  = 0)  # converts to expected dimension i.e 1,128,128

    prediction = model.predict(img_array)

    confidence = prediction[0][0]

    st.divider()

    st.markdown("""
        <h4 style="color:#6B3882;">🎯 Prediction</h4>
    """, unsafe_allow_html=True)
        
    if confidence > 0.5:
        st.error("❌ Without Mask")
        display_confidence = confidence * 100
    else:
        st.success("✅ With Mask")
        display_confidence = (1 - confidence) * 100

    st.divider()
        
    st.markdown(
        f"""
        
        <h4>📊 Confidence</h4>
        <h4 style="color:#6B3882;">
            {display_confidence:.2f}%
        </h4>
        
        """,
        unsafe_allow_html=True
    )

    st.progress(int(display_confidence))
        
    
st.divider()

if "open_camera" not in st.session_state:
    st.session_state.open_camera = False

st.markdown(""" 
            <h5 style="margin-top:20px; color:#6B3882;  "> 2. 📸 Capture Image</h5>""",unsafe_allow_html=True)
    
if st.button("Click Photo"):
    st.session_state.open_camera = True

    #with col2:
        #if st.button("❌ Close Camera"):
            #st.session_state.open_camera = False

if st.session_state.open_camera:

    camera_image = st.camera_input("")

    if camera_image is not None:
        st.divider()
        
        st.markdown("""
        <h5 style="color:#6B3882;">Image Preview</h5>
        """, unsafe_allow_html=True)

        st.image(camera_image)

        img = Image.open(camera_image)
        img = img.resize((128,128))

        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        confidence = prediction[0][0]

        st.divider()

        st.markdown("""
        <h4 style="color:#6B3882;">🎯 Prediction</h4>
        """, unsafe_allow_html=True)
        
        if confidence > 0.5:
            st.error("❌ Without Mask")
            display_confidence = confidence * 100
        else:
            st.success("✅ With Mask")
            display_confidence = (1 - confidence) * 100

        st.divider()
        
        st.markdown(
        f"""
        
        <h4>📊 Confidence</h4>
        <h4 style="color:#6B3882;">
            {display_confidence:.2f}%
        </h4>
        
        """,
        unsafe_allow_html=True
        )

        st.progress(int(display_confidence))
        # Close camera after prediction
        st.session_state.open_camera = False
        
st.divider()
st.markdown(
    """
    <div style="text-align:center; color:#897094; font-size:17px;">
        Developed using TensorFlow, Keras and Streamlit 
        
    </div>
    """,
    unsafe_allow_html=True
)
