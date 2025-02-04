import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
    model= tf.keras.models.load_model(r"D:\project_folder\Cancer-cells_detection\data\images\trained_cancer_model.h5")
    image= tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    predictions= model.predict(input_arr)
    return np.argmax(predictions)
st.sidebar.title("Cancer Cell Detection and Segmentation")
app_mode = st.sidebar.selectbox('select page',['HOME','Cancer Cell Detection'])

from PIL import Image
img= Image.open('D:\project_folder\Cancer-cells_detection\data\images\download.jpg')
st.image(img)

if(app_mode=='HOME'):
    st.markdown("<h1 style='text-align: center;'>Cancer Cell Detection and Segmentation", unsafe_allow_html=True)

elif(app_mode=='Cancer Cell Detection'):
    st.header('Cancer Cell Detection and Segmentation')


test_image= st.file_uploader('Choose an image:')
if(st.button('Show Image')):
    st.image(test_image,width=4,use_column_width=True)

if (st.button('Predict')):
    st.snow()
    st.write('our prediction')
    result_index = model_prediction(test_image)
    class_name=['EOSINOPHIL','LYMPHOCYTE','MONOCYTE','NEUTROPHIL']
    st.success('Model is predicting its a {}'.format(class_name[result_index]))