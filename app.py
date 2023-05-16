import streamlit as st

from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from helper_functions import get_model , predict , class_names
from PIL import Image 
    

sd=st.sidebar


sd.title("What is DeepFood ?")
sd.write(""" 
DeepFood is an end to end CNN model for food image classification capable of identifying 101 defferent foods.
""")

sd.write("""
**Accuracy :** **`80%`**

**Model :** **`EfficientNetB0`**

**Dataset :** **`Food101`**
""")

@st.cache_resource
def load_model():
   
    model_1=get_model()
    model_1.load_weights('model_weights/model_1')
    return model_1

model=load_model()

st.title("Hi! , Welcome to DeepFood 🍕🍔")
st.write("To know more about this app, visit [**GitHub**](https://github.com/nagarajRPoojari/Deep-food)")

file=st.file_uploader("upload image ", ['jpg', 'png','jpeg','jfif'])



if st.button("Predict") :
    if file is None :
        st.warning("Please upload image")
    else :
        res,img, confidence=predict(model,class_names,path=file)
        st.subheader(res)
        st.progress(confidence , text=str(confidence)+"%")
        st.image(file , width=400)
            
    


