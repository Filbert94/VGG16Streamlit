# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 08:07:51 2021

@author: Filbert Tucio
"""

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

model=0
model=tf.keras.models.load_model('vgg16_1_Moiture_200.h5')

st.write("""
         # Moisture Content Classification
         """
         )

uploaded_file = st.file_uploader("Choose a image file", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, channels="RGB")
    img = tf.image.resize(image, size=(244,244))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    images=np.vstack([img])
    pred=model.predict(images) 

    st.text(pred)
    st.text("The image is classify within: ")
    if pred[0][0] > 0.5:
        st.text("High Moisture")
    elif pred[0][1] > 0.5:
        st.text("Low Moisture")
    elif pred[0][2] > 0.5:
        st.text("MidHigh Moisture")
    elif pred[0][3] > 0.5:
        st.text("MidLow Moisture")
    else:
        st.text("Unidentified")
    st.legacy_caching.caching.clear_cache()
st.legacy_caching.caching.clear_cache()
model=0