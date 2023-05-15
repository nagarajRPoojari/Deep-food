import streamlit as st

from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from helper_functions import get_model , predict , class_names
from PIL import Image 
    




@st.cache_resource
def load_model():
   
    model_1=get_model()
    model_1.load_weights('model_weights/model_1')
    return model_1

model=load_model()

st.title("Hi! , Welcome to DeepFood")

file=st.file_uploader("upload image ", ['jpg', 'png','jpeg','jfif'])


if st.button("Predict") :
    if file is None :
        st.warning("Please upload image")
    else :
        res,img=predict(model,class_names,path=file)
        st.subheader(res.upper())
        st.image(file , width=500)
    


